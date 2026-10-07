"""Copy pinned Balalaika text labels; whole-source splits, no inferred history."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import pyarrow.parquet as pq

from prepare_local import key, read, record, valid
from prepare_ruslan import signature

REPO = "lab260/openstt_balalaika"
REVISION = "c39c8215542af39855943c7ca5c8845bd39650f0"


def family(part):
    # Inspected podcast_id is the chunk filename, not a recoverable recording ID.
    for prefix in ("asr_public_phone_calls", "public_youtube", "asr_public_stories", "radio"):
        if part.startswith(prefix):
            return prefix
    return part


def convert(raw, part):
    path = raw["filepath"]
    if not isinstance(path, str) or Path(path).parts[0] != part:
        raise ValueError("source_partition_mismatch")
    written, target = raw["punct"], raw["accent"]
    if not isinstance(written, str) or not isinstance(target, str):
        raise ValueError("nontext_label")
    # Same conservative initial/acronym and bare-letter checks as Ruslan/Google adapters.
    if (re.search(r"\b[А-ЯЁ]{2,}\b|\b[А-ЯЁ]\.(?=\s|$)", written)
            or re.search(r"\b[бгджзйлмнпртфхцчшщъыь]\b", written.lower())):
        raise ValueError("raw_abbreviation_or_letter")
    row = record(written, target, "balalaika:" + path, "balalaika:family:" + family(part),
                 "Pinned OpenSTT/Balalaika: RuPunct + RUAccent automatic labels; not human gold")
    reason = valid(row)
    if reason:
        raise ValueError(reason)
    signature(target)
    if key(written) != key(target):
        raise ValueError("source_content_mismatch")
    return dict(row, spoken=target.replace("+", ""), source_revision=REVISION)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-manifest", type=Path, required=True)
    p.add_argument("--existing-train", type=Path, required=True)
    p.add_argument("--heldouts", type=Path, nargs="+", required=True)
    p.add_argument("--validation-family", action="append", default=[])
    p.add_argument("--review-exclusions", type=Path, help="Reviewed source IDs to quarantine, not relabel")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("Output must be a new dataset version")
    manifest = json.loads(args.source_manifest.read_text())
    if manifest["repo"] != REPO or manifest["revision"] != REVISION:
        raise ValueError("Unpinned source")
    inputs = {}; blocked = set(); ids = set(); ownership = {}
    excluded = {}
    if args.review_exclusions:
        inputs[str(args.review_exclusions)] = hashlib.sha256(args.review_exclusions.read_bytes()).hexdigest()
        for row in read(args.review_exclusions):
            if row["source_id"] in excluded:
                raise ValueError("Duplicate reviewed exclusion")
            excluded[row["source_id"]] = row["reason"]
    for split, paths in [("train", [args.existing_train]), ("validation", args.heldouts)]:
        for path in paths:
            inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            for row in read(path):
                ids.add(row["source_id"])
                blocked.update(key(row[f]) for f in ("written", "spoken", "spoken_stressed")
                               if isinstance(row.get(f), str))
                if row["source_id"].startswith("balalaika:"):
                    group = row["source_group"]
                    if group in ownership and ownership[group] != split:
                        raise ValueError("Existing group split overlap")
                    ownership[group] = split
    candidates = []; rejected = []; seen_ids = set()
    families = {family(e["file"].removesuffix("_balalaika.parquet")) for e in manifest["files"]}
    if set(args.validation_family) - families:
        raise ValueError("Unknown validation source family")
    for entry in manifest["files"]:
        path = args.source_manifest.parent / entry["file"]
        if path.name != entry["file"] or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Source file/hash mismatch")
        part = path.name.removesuffix("_balalaika.parquet")
        preferred = "validation" if family(part) in args.validation_family else "train"
        group = "balalaika:family:" + family(part)
        if group in ownership and ownership[group] != preferred:
            raise ValueError("Group split reassignment")
        ownership[group] = preferred
        for raw in pq.read_table(path, columns=["filepath", "punct", "accent"]).to_pylist():
            sid = "balalaika:" + str(raw["filepath"])
            if sid in seen_ids:
                raise ValueError("Duplicate source ID")
            seen_ids.add(sid)
            try:
                if sid in excluded:
                    raise ValueError("reviewed_source:" + excluded[sid])
                row = convert(raw, part)
                if sid in ids or any(key(row[f]) in blocked for f in ("written", "spoken_stressed")):
                    raise ValueError("existing_train_or_heldout")
                candidates.append(dict(row, split=preferred))
            except ValueError as exc:
                rejected.append(dict(source_id=sid, reason=str(exc)))
    counts = Counter(key(row["written"]) for row in candidates)
    if set(excluded) - seen_ids:
        raise ValueError("Unknown reviewed source IDs")
    splits = {"train": [], "validation": []}
    for row in candidates:
        if counts[key(row["written"])] != 1:
            rejected.append(dict(source_id=row["source_id"], reason="duplicate_normalized_input"))
        else:
            splits[row["split"]].append(row)
    if sum(map(len, splits.values())) + len(rejected) != len(seen_ids):
        raise ValueError("Row accounting mismatch")
    args.output.mkdir(parents=True)
    for name, rows in {**splits, "quarantine": rejected}.items():
        (args.output / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    result = dict(counts={s: len(v) for s, v in splits.items()}, rejected=len(rejected),
                  source_manifest=manifest, exclusions_sha256=inputs, split_ownership=ownership,
                  split_policy="Whole source families; podcast_id equals chunk ID in inspected metadata",
                  training_eligible=False, pending="Independent source/target review; no invented history")
    (args.output / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("counts", "rejected", "training_eligible")}))


if __name__ == "__main__":
    main()
