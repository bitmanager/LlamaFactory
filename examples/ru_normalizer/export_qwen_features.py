"""Cache frozen agent features for the unmodified Freeze-Omni decoder runtime."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

AGENT_SYSTEM = "Ты голосовой ассистент. Отвечай по-русски, кратко и по существу, учитывая историю разговора."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=("validation", "train"), default=["validation", "train"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(args.root / "models/agent-tokenizer", padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(args.root / "models/qwen3-4b-instruct-2507",
                                     dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    model.requires_grad_(False)
    metadata = dict(source_revision="cdbee75f17c01a7cc42f958dc650907174af0554",
                    source_dim=model.config.hidden_size, vocab_size=len(tokenizer),
                    hidden_alignment="pre-token final norm, matching Freeze-Omni _generate_one_step",
                    source_sha256={})
    tokenizer.save_pretrained(args.output / "tokenizer")
    for split in args.splits:
        source_path = args.data / f"{split}.jsonl"
        metadata["source_sha256"][split] = hashlib.sha256(source_path.read_bytes()).hexdigest()
        rows = [json.loads(line) for line in source_path.read_text().splitlines()]
        directory = args.output / split
        directory.mkdir()
        for offset in range(0, len(rows), 32):
            group = rows[offset:offset + 32]
            inputs, boundaries = [], []
            for row in group:
                messages = [{"role": "system", "content": row.get("system") or AGENT_SYSTEM},
                            *(row.get("history") or [])]
                prefix = tokenizer.apply_chat_template(messages, tokenize=True,
                             add_generation_prompt=True, return_dict=False)
                answer = tokenizer.encode(row["written"], add_special_tokens=False)
                if not answer or len(prefix) + len(answer) > 768:
                    raise ValueError(f"Invalid source length: {row['source_id']}")
                inputs.append({"input_ids": prefix + answer})
                boundaries.append((len(prefix), len(prefix) + len(answer)))
            batch = tokenizer.pad(inputs, padding=True, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                hidden = model(**batch, use_cache=False).last_hidden_state
                embeddings = model.get_input_embeddings()(batch.input_ids)
            for i, (row, (start, end)) in enumerate(zip(group, boundaries)):
                # Upstream pairs the sampled text token with the hidden that predicts it.
                item = dict(x=embeddings[i, start:end].cpu().clone(),
                            x_prefix=hidden[i, start - 1:end - 1].cpu().clone(),
                            y=torch.tensor(tokenizer.encode(row["spoken_stressed"], add_special_tokens=False)),
                            written=row["written"], reference=row["spoken_stressed"],
                            source_id=row["source_id"])
                torch.save(item, directory / f"{offset + i:06d}.pt")
            if offset % 1024 == 0:
                print(json.dumps(dict(split=split, completed=min(offset + 32, len(rows)), total=len(rows))), flush=True)
        metadata[split] = len(rows)
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
