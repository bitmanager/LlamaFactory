"""Validate and hardlink-concatenate two completed frozen-feature exports."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def link_tree(source, target):
    target.mkdir()
    for path in source.iterdir():
        if path.is_dir():
            link_tree(path, target / path.name)
        else:
            os.link(path, target / path.name)


def check(condition, message):
    if not condition:
        raise ValueError(message)


def validate_datasets(base, extra, quality_review=None):
    """Keep existing rows; reject collisions introduced by the appended data."""
    if quality_review:
        review = load(quality_review)
        check(review["decision"].split()[0] == "GO", "Quality review is not GO")
        for split in ("train", "validation"):
            check(sha(extra / f"{split}.jsonl") == review["sha256"][split], f"Quality review hash mismatch: {split}")
    elif (extra / "manifest.json").exists():
        check(load(extra / "manifest.json").get("training_eligible") is not False,
              "Extra dataset is explicitly marked training_eligible=false")
    groups = {"train": set(), "heldout": set()}
    seen_ids, seen_text, counts = set(), set(), {}
    for kind, directory in (("base", base), ("extra", extra)):
        for split in ("train", "validation", "eval_existing"):
            path = directory / f"{split}.jsonl"
            if not path.exists():
                check(kind == "extra" and split != "train" or split == "eval_existing", f"Missing {path}")
                continue
            raw = path.read_bytes()
            check(raw.endswith(b"\n"), f"JSONL must end with newline: {path}")
            rows = [json.loads(line) for line in raw.splitlines()]
            counts[(kind, split)] = len(rows)
            for row in rows:
                identity = row["source_id"]
                check(identity not in seen_ids, f"Duplicate source_id: {identity}")
                seen_ids.add(identity)
                keys = {" ".join(re.findall(r"\w+", row[f].lower().replace("ё", "е").replace("+", "")))
                        for f in ("written", "spoken_stressed")}
                check("" not in keys, f"Empty text: {identity}")
                if kind == "extra":
                    check(not keys & seen_text, f"Duplicate normalized full text: {identity}")
                seen_text.update(keys)
                if row.get("source_group"):
                    groups["train" if split == "train" else "heldout"].add(row["source_group"])
    check(not groups["train"] & groups["heldout"], "Train/heldout source_group overlap")
    return counts


def merge_data(base, extra, output, quality_review=None):
    counts = validate_datasets(base, extra, quality_review)  # All checks precede output creation.
    check(not (extra / "eval_existing.jsonl").exists(), "Extra frozen eval needs an explicit merge policy")
    output.mkdir()
    inputs, splits = [], {}
    for split in ("train", "validation", "eval_existing"):
        first, second = base / f"{split}.jsonl", extra / f"{split}.jsonl"
        if not first.exists():
            continue
        inputs.append(first)
        if second.exists():
            inputs.append(second)
            (output / first.name).write_bytes(first.read_bytes() + second.read_bytes())
        else:
            os.link(first, output / first.name)
        splits[split] = dict(base_rows=counts[("base", split)], added_rows=counts.get(("extra", split), 0))
        splits[split]["rows"] = sum(splits[split].values())
    provenance = output / "provenance"
    provenance.mkdir()
    link_tree(base, provenance / "base-data")
    link_tree(extra, provenance / "extra-data")
    manifest = dict(base_rows=counts[("base", "train")], added_rows=counts[("extra", "train")],
        train_rows=splits["train"]["rows"], splits=splits,
        source_sha256={str(p):sha(p) for p in inputs},
        output_sha256={p.name:sha(p) for p in sorted(output.glob("*.jsonl"))},
        recipe="Ordered byte concatenation by split; base frozen eval retained; no relabeling or silent dedup")
    if quality_review:
        review = load(quality_review)
        manifest["quality_review"] = dict(path=str(quality_review), sha256=sha(quality_review),
            decision=review["decision"], limits=review.get("limits"),
            reviewed_files_sha256=review["sha256"],
            note="Explicit reviewed approval; original pre-review staging manifest retained unchanged")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(dict(output=str(output), splits=splits)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base-data", "extra-data", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("base", "extra", "data"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--merge-data", action="store_true", help="Only merge JSONL datasets; no feature assembly")
    parser.add_argument("--quality-review", type=Path, help="Explicit GO review bound to extra train/validation hashes")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    for source in (args.base_data, args.extra_data):
        check(not args.output.resolve().is_relative_to(source.resolve()), "Output must be outside input datasets")
    if args.merge_data:
        merge_data(args.base_data, args.extra_data, args.output, args.quality_review)
        return
    if any(value is None for value in (args.base, args.extra, args.data)):
        parser.error("Feature assembly requires --base, --extra and --data")
    counts = validate_datasets(args.base_data, args.extra_data, args.quality_review)
    base, extra = (load(p / "metadata.json") for p in (args.base, args.extra))
    manifest = load(args.data / "manifest.json")
    common = ("source_revision", "source_dim", "vocab_size", "hidden_alignment")
    for field in common:
        check(base[field] == extra[field], f"Feature metadata mismatch: {field}")
    for name, digest in manifest["output_sha256"].items():
        check(sha(args.data / name) == digest, f"Merged dataset hash mismatch: {name}")
    merged_counts = {}
    for split in ("train", "validation"):
        expected = manifest.get("splits", {}).get(split, dict(
            base_rows=manifest["base_rows"] if split == "train" else base["validation"],
            added_rows=manifest["added_rows"] if split == "train" else 0,
            rows=manifest["train_rows"] if split == "train" else base["validation"]))
        raw = b""
        for kind, features, dataset in (("base", base, args.base_data), ("extra", extra, args.extra_data)):
            count = counts.get((kind, split), 0)
            check(features.get(split, 0) == count == expected["base_rows" if kind == "base" else "added_rows"],
                  f"{kind} {split} row count mismatch")
            if count:
                path = dataset / f"{split}.jsonl"
                check(features["source_sha256"][split] == sha(path) == manifest["source_sha256"][str(path)],
                      f"Source hash mismatch: {path}")
                raw += path.read_bytes()
        merged_counts[split] = base[split] + extra.get(split, 0)
        check(merged_counts[split] == expected["rows"], f"Merged {split} count mismatch")
        check((args.data / f"{split}.jsonl").read_bytes() == raw, f"{split} is not ordered base + extra")
    tokenizer_hashes = []
    for source in (args.base, args.extra):
        tokenizer_hashes.append({str(p.relative_to(source / "tokenizer")):sha(p)
                                 for p in sorted((source / "tokenizer").rglob("*")) if p.is_file()})
    check(tokenizer_hashes[0] and tokenizer_hashes[0] == tokenizer_hashes[1], "Tokenizer hash mismatch")
    sequences = [(source, split, meta[split]) for source, meta in ((args.base, base), (args.extra, extra))
                 for split in ("train", "validation") if split in meta]
    for source, split, count in sequences:
        files = sorted((source / split).iterdir())
        check([p.name for p in files] == [f"{i:06d}.pt" for i in range(count)]
              and all(p.is_file() and p.stat().st_size > 0 for p in files),
              f"Incomplete feature sequence: {source}/{split}")

    args.output.mkdir()
    for split in ("train", "validation"):
        (args.output / split).mkdir()
        for source, count, offset in ((args.base, base[split], 0), (args.extra, extra.get(split, 0), base[split])):
            for index in range(count):
                os.link(source / split / f"{index:06d}.pt", args.output / split / f"{index + offset:06d}.pt")
    link_tree(args.base / "tokenizer", args.output / "tokenizer")
    provenance = args.output / "provenance"
    provenance.mkdir()
    for label, dataset in (("base-data", args.base_data), ("extra-data", args.extra_data), ("merged-data", args.data)):
        link_tree(dataset, provenance / label)
    for label, source in (("base", args.base), ("extra", args.extra)):
        os.link(source / "metadata.json", provenance / f"{label}-metadata.json")
        if (source / "corrections.json").exists():
            os.link(source / "corrections.json", provenance / f"{label}-corrections.json")
    metadata = {field:base[field] for field in common}
    metadata.update(**merged_counts,
        source_sha256={s:sha(args.data / f"{s}.jsonl") for s in ("train", "validation")},
        assembled_from={str(p):sha(p / "metadata.json") for p in (args.base, args.extra)},
        tokenizer_sha256=tokenizer_hashes[0], dataset_manifest_sha256=sha(args.data / "manifest.json"))
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    # A missing marker means an interrupted/incomplete assembly, never a usable cache.
    (args.output / "COMPLETE.json").write_text(json.dumps(dict(
        metadata_sha256=sha(args.output / "metadata.json"), train=metadata["train"],
        validation=metadata["validation"]), indent=2) + "\n")
    print(json.dumps(dict(output=str(args.output), train=metadata["train"], validation=metadata["validation"])))


if __name__ == "__main__":
    main()
