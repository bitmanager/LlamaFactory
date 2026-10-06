"""Stock HF generation: preserve actual Qwen answers and their conversation prefixes."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, pipeline
from export_qwen_features import AGENT_SYSTEM


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "prompts", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    prompts = [json.loads(line) for line in args.prompts.read_text().splitlines()]
    tokenizer = AutoTokenizer.from_pretrained(args.root / "models/agent-tokenizer", padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    generator = pipeline("text-generation", model=str(args.root / "models/qwen3-4b-instruct-2507"),
                         tokenizer=tokenizer, device=0, dtype=torch.bfloat16,
                         model_kwargs={"attn_implementation": "sdpa"})
    generator.model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    generator.model.requires_grad_(False)
    eos = generator.model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    histories = [[] for _ in prompts]
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "answers.jsonl").open("x") as out, (args.output / "rejected.jsonl").open("x") as rejected:
        for turn in range(2):
            messages = [[{"role": "system", "content": AGENT_SYSTEM}, *history,
                         {"role": "user", "content": row["questions"][turn]}]
                        for row, history in zip(prompts, histories)]
            rendered = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
            results = generator(rendered, batch_size=8, max_new_tokens=96, do_sample=False,
                                temperature=None, top_p=None, top_k=None,
                                add_special_tokens=False, return_tensors=True)
            for index, (source, message, prompt, result) in enumerate(zip(prompts, messages, rendered, results)):
                prefix = tokenizer.encode(prompt, add_special_tokens=False)
                ids = result[0]["generated_token_ids"]
                while ids and ids[0] == tokenizer.pad_token_id:
                    ids = ids[1:]
                if ids[:len(prefix)] != prefix:
                    raise ValueError("Generation prefix alignment mismatch")
                new = ids[len(prefix):]
                ends = [i for i, token in enumerate(new) if token in eos]
                answer = tokenizer.decode(new[:ends[0]] if ends else new, skip_special_tokens=False)
                row = dict(source_id=f"qwen-context:{source['id']}:{turn}", source_group=f"qwen-context:{source['id']}",
                           history=message[1:], system=AGENT_SYSTEM, written=answer, spoken=answer,
                           provenance="Frozen Qwen3-4B-Instruct-2507 greedy answer; real generation history; stress not yet labeled")
                reason = None if ends and answer.strip() else "incomplete_answer"
                (rejected if reason else out).write(json.dumps(dict(row, rejection_reason=reason), ensure_ascii=False) + "\n")
                histories[index] = message[1:] + [{"role": "assistant", "content": answer}]
            out.flush()
            print(json.dumps(dict(turn=turn, conversations=len(prompts))), flush=True)
    (args.output / "manifest.json").write_text(json.dumps(dict(
        prompts_sha256=hashlib.sha256(args.prompts.read_bytes()).hexdigest(),
        model_revision="cdbee75f17c01a7cc42f958dc650907174af0554",
        training_eligible=False, pending="Content-preserving accent annotation and review"), indent=2))


if __name__ == "__main__":
    main()
