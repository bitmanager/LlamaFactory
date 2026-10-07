"""Format-only context augmentation of reviewed normalization pairs; no relabeling."""
import argparse
import hashlib
import json
import re
from pathlib import Path

from transformers import AutoTokenizer
from export_qwen_features import AGENT_SYSTEM


def key(text):
    return " ".join(re.findall(r"\w+", text.lower().replace("+", "").replace("ё", "е")))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "output", "tokenizer", "protected"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--train-size", type=int, default=2048)
    p.add_argument("--validation-size", type=int, default=64)
    args = p.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    protected = [json.loads(line) for line in args.protected.read_text().splitlines()]
    blocked = {key(r[f]) for r in protected for f in ("written", "spoken_stressed")}
    train_keys, groups = set(), {"train": set(), "validation": set()}
    outputs, inputs = {}, {}
    for split, count in (("train", args.train_size), ("validation", args.validation_size)):
        path = args.source / f"{split}.jsonl"
        inputs[split] = hashlib.sha256(path.read_bytes()).hexdigest()
        source = [json.loads(line) for line in path.read_text().splitlines()]
        if split == "train":
            heldout = [json.loads(line) for name in ("validation", "eval_existing")
                       for line in (args.source / f"{name}.jsonl").read_text().splitlines()]
            blocked.update(key(r[f]) for r in heldout for f in ("written", "spoken_stressed"))
        selected = []
        for row in sorted(source, key=lambda r: hashlib.sha256(r["source_id"].encode()).hexdigest()):
            text = row["written"]
            # Isolate literal content preservation, without introducing numeric targets.
            if key(text) != key(row["spoken_stressed"]) or re.search(r"\d|[A-Za-z]|\b[А-ЯЁ]{2,}\b", text):
                continue
            if not 20 <= len(text) <= 350 or row.get("history"):
                continue
            if split == "train" and key(text) in blocked or key(text) in train_keys:
                continue
            context = [{"role": "user", "content": "Повтори дословно: " + text}]
            multi = [{"role": "user", "content": "Сейчас я продиктую фразу. Сохрани все слова."},
                     {"role": "assistant", "content": "Хорошо, продиктуй фразу."}, *context]
            versions = []
            for mode, history in (("plain", []), ("question", context), ("history", multi)):
                prefix = tokenizer.apply_chat_template([{"role": "system", "content": AGENT_SYSTEM}, *history],
                             tokenize=True, add_generation_prompt=True, return_dict=False)
                if len(prefix) + len(tokenizer.encode(text, add_special_tokens=False)) > 768:
                    break
                versions.append(dict(row, source_id=f"context:{mode}:{row['source_id']}",
                    original_source_id=row["source_id"], history=history,
                    context_provenance="Synthetic verbatim-repeat context; original reviewed target unchanged"))
            if len(versions) != 3:
                continue
            selected.extend(versions)
            if row.get("source_group"):
                groups[split].add(row["source_group"])
            if split == "train":
                train_keys.add(key(text))
            if len(selected) == count * 3:
                break
        if len(selected) != count * 3:
            raise ValueError(f"Insufficient eligible {split} pairs: {len(selected) // 3}")
        outputs[split] = selected
    if groups["train"] & groups["validation"]:
        raise ValueError("Source group leakage")
    args.output.mkdir(parents=True, exist_ok=False)
    for split, rows in outputs.items():
        (args.output / f"{split}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    (args.output / "manifest.json").write_text(json.dumps(dict(
        source=str(args.source), source_sha256=inputs,
        protected_sha256=hashlib.sha256(args.protected.read_bytes()).hexdigest(),
        counts={k: len(v) for k, v in outputs.items()},
        purpose="Diagnostic content-preservation pilot; synthetic context, not genuine conversations",
        labels="Unchanged reviewed base targets; original provenance retained"), indent=2))


if __name__ == "__main__":
    main()
