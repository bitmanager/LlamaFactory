"""Apply audited stress-only corrections without recomputing frozen features."""
import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoTokenizer


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def correct(text, wrong, expected):
    if wrong.replace("+", "").lower() != expected.replace("+", "").lower():
        raise ValueError("Correction changes letters")
    pattern = re.compile(r"(?<![\w+])" + re.escape(wrong) + r"(?![\w+])", re.IGNORECASE)

    def replace(match):
        letters = iter(match[0].replace("+", ""))
        return "".join("+" if char == "+" else next(letters) for char in expected)

    result, count = pattern.subn(replace, text)
    if result.replace("+", "") != text.replace("+", ""):
        raise ValueError("Correction changed content/case/punctuation")
    return result, count


def link_tree(source, target):
    target.mkdir()
    for path in source.iterdir():
        if path.is_dir():
            link_tree(path, target / path.name)
        else:
            os.link(path, target / path.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "features", "audit", "output-data", "output-features"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    if args.output_data.exists() or args.output_features.exists():
        raise FileExistsError("Correction outputs must be new directories")
    audit = json.loads(args.audit.read_text())
    metadata = json.loads((args.features / "metadata.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.features / "tokenizer", local_files_only=True)
    expected_counts, rules = Counter(), defaultdict(list)
    for wrong, rule in audit["rules"].items():
        if rule["classification"] != "wrong_stress":
            continue
        for split, details in rule["splits"].items():
            ids = details["source_ids"]
            if len(ids) != details["count"] or len(set(ids)) != len(ids):
                raise ValueError("Audit source IDs/count disagree")
            expected_counts[(split, wrong)] = len(ids)
            for identity in ids:
                rules[(split, identity)].append((wrong, rule["expected"]))
    outputs, changes, observed = {}, {}, Counter()
    for split, info in audit["files"].items():
        path = args.data / f"{split}.jsonl"
        if sha(path) != info["sha256"]:
            raise ValueError(f"Input differs from audit: {split}")
        lines = path.read_text().splitlines(keepends=True)
        if len(lines) != info["rows"]:
            raise ValueError(f"Row count differs from audit: {split}")
        if split in metadata and (metadata[split] != len(lines)
                or metadata["source_sha256"][split] != info["sha256"]):
            raise ValueError(f"Feature metadata differs from data: {split}")
        outputs[split], changes[split] = [], {}
        seen = set()
        for index, line in enumerate(lines):
            row = json.loads(line)
            identity = row["source_id"]
            if identity in seen:
                raise ValueError(f"Duplicate source ID: {identity}")
            seen.add(identity)
            before, applied = row["spoken_stressed"], []
            for wrong, target in rules.get((split, identity), []):
                row["spoken_stressed"], count = correct(row["spoken_stressed"], wrong, target)
                if not count:
                    raise ValueError(f"Audited word missing: {identity}: {wrong}")
                observed[(split, wrong)] += 1
                applied.append(dict(wrong=wrong, corrected=target, occurrences=count))
            if applied:
                changes[split][index] = dict(source_id=identity, before=before,
                    after=row["spoken_stressed"], rules=applied)
                line = json.dumps(row, ensure_ascii=False) + "\n"
            outputs[split].append(line)
    if +observed != +expected_counts:
        raise ValueError("Applied correction counts differ from audit")
    for split in ("train", "validation"):
        names = sorted(p.name for p in (args.features / split).glob("*.pt"))
        if names != [f"{i:06d}.pt" for i in range(metadata[split])]:
            raise ValueError(f"Feature index mismatch: {split}")
    args.output_data.mkdir(exist_ok=False)
    args.output_features.mkdir(exist_ok=False)
    for split, lines in outputs.items():
        (args.output_data / f"{split}.jsonl").write_text("".join(lines))
    for source in args.data.iterdir():
        if source.name not in {f"{s}.jsonl" for s in outputs}:
            os.link(source, args.output_data / source.name)
    link_tree(args.features / "tokenizer", args.output_features / "tokenizer")
    cache_counts = Counter()
    for split in ("train", "validation"):
        output = args.output_features / split
        output.mkdir()
        for index in range(metadata[split]):
            source, target = args.features / split / f"{index:06d}.pt", output / f"{index:06d}.pt"
            if index not in changes[split]:
                os.link(source, target)
                cache_counts["hardlinked"] += 1
                continue
            change = changes[split][index]
            item = torch.load(source, map_location="cpu", weights_only=True)
            if item["source_id"] != change["source_id"] or item["reference"] != change["before"]:
                raise ValueError(f"Feature content mismatch: {source}")
            old_tokens = tokenizer.encode(change["before"], add_special_tokens=False)
            if item["y"].tolist() != old_tokens:
                raise ValueError(f"Original target tokens mismatch: {source}")
            updated = {**item, "reference": change["after"], "y": torch.tensor(
                tokenizer.encode(change["after"], add_special_tokens=False), dtype=item["y"].dtype)}
            torch.save(updated, target)  # New file; never write through an existing hardlink.
            check = torch.load(target, map_location="cpu", weights_only=True)
            for key, value in item.items():
                if key not in ("y", "reference"):
                    equal = torch.equal(value, check[key]) if isinstance(value, torch.Tensor) else value == check[key]
                    if not equal:
                        raise ValueError(f"Frozen feature changed: {source}: {key}")
            if target.stat().st_ino == source.stat().st_ino:
                raise ValueError("Corrected feature unexpectedly aliases original")
            change["feature_sha256_before"], change["feature_sha256_after"] = sha(source), sha(target)
            cache_counts["rewritten_targets"] += 1
    hashes = {s:sha(args.output_data / f"{s}.jsonl") for s in outputs}
    manifest = dict(audit_sha256=sha(args.audit), source_data=str(args.data),
        source_features=str(args.features), rows={s:len(v) for s,v in outputs.items()},
        changed_rows={s:len(v) for s,v in changes.items()}, cache_counts=dict(cache_counts),
        source_sha256={s:audit["files"][s]["sha256"] for s in outputs}, output_sha256=hashes,
        changes={s:[dict(index=i, **v) for i,v in rows.items()] for s,rows in changes.items()},
        ignored_classifications=["context_check"],
        resume_contract="Same source IDs, order, feature x/x_prefix, tokenizer and sample counts; only target labels changed")
    manifest_path = args.output_data / "corrections.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    os.link(manifest_path, args.output_features / "corrections.json")
    metadata["source_sha256"] = {s:hashes[s] for s in metadata["source_sha256"]}
    metadata["label_correction_audit_sha256"] = sha(args.audit)
    metadata["label_correction_manifest_sha256"] = sha(manifest_path)
    (args.output_features / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for split, info in audit["files"].items():
        if sha(args.data / f"{split}.jsonl") != info["sha256"]:
            raise ValueError("Original labels modified")
    print(json.dumps({k:v for k,v in manifest.items() if k != "changes"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
