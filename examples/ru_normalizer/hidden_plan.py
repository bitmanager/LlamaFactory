"""Qwen hidden-prefix adapter. Optimizer, CE, LoRA and decoding stay upstream."""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import DataCollatorForSeq2Seq, PreTrainedModel, PretrainedConfig


PLAN_SYSTEM = (
    "Преобразуй представленную фразу в произносимую русскую форму. "
    "Раскрой числа и сокращения, поставь + перед ударными гласными. "
    "Сохрани слова и смысл. Верни только результат без объяснений."
)
AGENT_SYSTEM = "Ты голосовой ассистент. Отвечай по-русски, кратко и по существу, учитывая историю разговора."


def encode_example(row, tokenizer, max_length):
    """Teacher-force the original answer through Qwen; never expose the plan."""
    written, target = row["written"], row["spoken_stressed"]
    if not written.strip() or not target.strip():
        raise ValueError("Empty written text or spoken plan")
    history = row.get("history") or []
    if any(m["role"] not in ("user", "assistant") for m in history):
        raise ValueError("History must contain user/assistant turns only")
    messages = [{"role": "system", "content": row.get("system") or AGENT_SYSTEM}, *history]
    prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    source_prefix = tokenizer.encode(prefix, add_special_tokens=False)
    answer = tokenizer.encode(written, add_special_tokens=False)
    source_ids = source_prefix + answer
    # Auxiliary decoder has no written-text bypass: only projected hidden + task prompt.
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": PLAN_SYSTEM},
         {"role": "user", "content": "Преобразуй фразу из переданного представления."}],
        tokenize=True, add_generation_prompt=True, return_dict=False,
    )
    target_ids = tokenizer.encode(target, add_special_tokens=False) + [tokenizer.eos_token_id]
    if len(source_ids) > max_length or len(answer) + len(prompt) + len(target_ids) > max_length:
        raise ValueError(f"Example exceeds {max_length} tokens: {row.get('source_id')}")
    return {
        "source_ids": source_ids,
        "answer_mask": [0] * len(source_prefix) + [1] * len(answer),
        "input_ids": prompt + target_ids,
        "attention_mask": [1] * (len(prompt) + len(target_ids)),
        "labels": [-100] * len(prompt) + target_ids,
    }


@dataclass
class PrefixCollator:
    tokenizer: object

    def __call__(self, rows):
        source = self.tokenizer.pad(
            [{"input_ids": r["source_ids"]} for r in rows], padding=True, return_tensors="pt"
        )
        answer_mask = pad_sequence(
            [torch.tensor(r["answer_mask"], dtype=torch.bool) for r in rows], batch_first=True
        )
        decoder = DataCollatorForSeq2Seq(self.tokenizer, padding=True)(
            [{k: r[k] for k in ("input_ids", "attention_mask", "labels")} for r in rows]
        )
        return {**decoder, "source_ids": source.input_ids,
                "source_mask": source.attention_mask, "answer_mask": answer_mask}


class HiddenPlan(PreTrainedModel):
    """A frozen source Qwen and native PEFT decoder joined by one projection."""

    def __init__(self, source, decoder):
        super().__init__(PretrainedConfig(source=source.config.to_dict(),
                                         decoder=decoder.config.to_dict(), tie_word_embeddings=False))
        self.source = source.requires_grad_(False).eval()
        self.decoder = decoder
        self.projector = nn.Sequential(
            nn.LayerNorm(source.config.hidden_size),
            nn.Linear(source.config.hidden_size, decoder.config.hidden_size),
        ).to(dtype=next(decoder.parameters()).dtype)

    def train(self, mode=True):
        super().train(mode)
        self.source.eval()
        return self

    def condition(self, source_ids, source_mask, answer_mask, input_ids, attention_mask):
        if answer_mask.shape != source_ids.shape or not answer_mask.any(dim=1).all():
            raise ValueError("Each example needs at least one original-answer hidden state")
        if (answer_mask & ~source_mask.bool()).any():
            raise ValueError("Padding cannot be selected as answer hidden")
        with torch.no_grad():
            hidden = self.source(input_ids=source_ids, attention_mask=source_mask,
                                 use_cache=False).last_hidden_state
        selected = [h[m] for h, m in zip(hidden, answer_mask)]
        # Left padding keeps each prefix adjacent to the decoder prompt.
        prefix = pad_sequence([h.flip(0) for h in selected], batch_first=True).flip(1)
        mask = torch.arange(prefix.size(1), device=prefix.device)[None] >= (
            prefix.size(1) - answer_mask.sum(1, keepdim=True)
        )
        prefix = self.projector(prefix)
        embeddings = self.decoder.get_input_embeddings()(input_ids)
        return torch.cat((prefix, embeddings), 1), torch.cat((mask, attention_mask), 1)

    def forward(self, source_ids, source_mask, answer_mask, input_ids, attention_mask, labels):
        embeddings, mask = self.condition(source_ids, source_mask, answer_mask, input_ids, attention_mask)
        prefix_labels = labels.new_full((labels.size(0), embeddings.size(1) - labels.size(1)), -100)
        return self.decoder(
            inputs_embeds=embeddings, attention_mask=mask,
            position_ids=(mask.long().cumsum(-1) - 1).clamp_min(0),
            labels=torch.cat((prefix_labels, labels), 1), use_cache=False,
        )

    @torch.no_grad()
    def generate(self, source_ids, source_mask, answer_mask, input_ids, attention_mask, **kwargs):
        embeddings, mask = self.condition(source_ids, source_mask, answer_mask, input_ids, attention_mask)
        return self.decoder.generate(inputs_embeds=embeddings, attention_mask=mask, **kwargs)
