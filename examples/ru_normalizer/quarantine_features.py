"""Remove exactly audited train pairs and hardlink their retained frozen features."""
import argparse
import json
import os
from pathlib import Path

import torch

from assemble_features import check, link_tree, load, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "features", "audit", "output-data", "output-features"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    outputs = (args.output_data, args.output_features)
    for output in outputs:
        check(not output.exists(), f"Output already exists: {output}")
        for source in (args.data, args.features):
            check(not output.resolve().is_relative_to(source.resolve()), "Output is inside source")
    check(not any(a.resolve().is_relative_to(b.resolve()) for a, b in
                  (outputs, outputs[::-1])), "Outputs must be separate directories")
    audit = [json.loads(line) for line in args.audit.read_text().splitlines()]
    rejected = {row["source_id"]: row for row in audit}
    check(audit and len(rejected) == len(audit), "Empty audit or duplicate audited IDs")
    metadata = load(args.features / "metadata.json")
    source_manifest = load(args.data / "manifest.json")
    check(load(args.features / "COMPLETE.json")["metadata_sha256"] == sha(args.features / "metadata.json"),
          "Source cache is incomplete or its metadata changed")
    lines, rows, all_ids = {}, {}, set()
    for split in ("train", "validation", "eval_existing"):
        path = args.data / f"{split}.jsonl"
        raw = path.read_bytes()
        check(sha(path) == source_manifest["output_sha256"][path.name], f"Dataset manifest mismatch: {split}")
        check(raw.endswith(b"\n"), f"Missing terminal newline: {split}")
        lines[split] = raw.splitlines(keepends=True)
        rows[split] = [json.loads(line) for line in lines[split]]
        for row in rows[split]:
            identity = row["source_id"]
            check(identity not in all_ids, f"Duplicate dataset ID: {identity}")
            all_ids.add(identity)
            check(split == "train" or identity not in rejected, f"Audited heldout ID: {identity}")
        if split != "eval_existing":
            check(sha(path) == metadata["source_sha256"][split], f"Source hash mismatch: {split}")
            check(len(rows[split]) == metadata[split], f"Count mismatch: {split}")
            files = sorted((args.features / split).iterdir())
            check([p.name for p in files] == [f"{i:06d}.pt" for i in range(len(rows[split]))]
                  and all(p.is_file() and p.stat().st_size for p in files), f"Incomplete features: {split}")
    check(set(rejected) <= all_ids, "Unknown audited IDs")
    removed, retained = [], []
    for index, row in enumerate(rows["train"]):
        if row["source_id"] not in rejected:
            retained.append(index)
            continue
        item = rejected[row["source_id"]]
        check(row["written"] == item["written"] and row["spoken_stressed"] == item["target"],
              f"Audit text mismatch: {row['source_id']}")
        feature = torch.load(args.features / "train" / f"{index:06d}.pt", map_location="cpu", weights_only=True)
        check(feature["source_id"] == row["source_id"] and feature["written"] == item["written"]
              and feature["reference"] == item["target"], f"Audited feature mismatch: {row['source_id']}")
        removed.append(dict(old_index=index, source_id=row["source_id"]))
    check(len(removed) == len(audit), "Not all audited rows matched")
    tokenizer_hashes = {str(p.relative_to(args.features / "tokenizer")): sha(p)
                        for p in (args.features / "tokenizer").rglob("*") if p.is_file()}
    check(tokenizer_hashes == metadata["tokenizer_sha256"], "Tokenizer hash mismatch")

    # All input checks precede creating either output; source files are never rewritten.
    args.output_data.mkdir()
    (args.output_data / "train.jsonl").write_bytes(b"".join(lines["train"][i] for i in retained))
    for split in ("validation", "eval_existing"):
        os.link(args.data / f"{split}.jsonl", args.output_data / f"{split}.jsonl")
    provenance = args.output_data / "provenance"
    provenance.mkdir()
    link_tree(args.data, provenance / "base-data")
    (provenance / "quarantine.jsonl").write_bytes(args.audit.read_bytes())
    manifest = dict(recipe="Exact audited train exclusions; retained byte order and targets unchanged",
        source_sha256={str(args.data / f"{s}.jsonl"): sha(args.data / f"{s}.jsonl") for s in rows},
        output_sha256={f"{s}.jsonl": sha(args.output_data / f"{s}.jsonl") for s in rows},
        counts=dict(train=len(retained), validation=len(rows["validation"]), eval_existing=len(rows["eval_existing"])),
        audit_sha256=sha(args.audit), removed=removed)
    (args.output_data / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    args.output_features.mkdir()
    (args.output_features / "train").mkdir()
    for index, old in enumerate(retained):
        os.link(args.features / "train" / f"{old:06d}.pt", args.output_features / "train" / f"{index:06d}.pt")
    for directory in ("validation", "tokenizer"):
        link_tree(args.features / directory, args.output_features / directory)
    provenance = args.output_features / "provenance"
    provenance.mkdir()
    link_tree(args.output_data, provenance / "filtered-data")
    os.link(args.features / "metadata.json", provenance / "base-metadata.json")
    result = {k: metadata[k] for k in ("source_revision", "source_dim", "vocab_size", "hidden_alignment")}
    result.update(train=len(retained), validation=metadata["validation"], tokenizer_sha256=tokenizer_hashes,
        source_sha256={s: sha(args.output_data / f"{s}.jsonl") for s in ("train", "validation")},
        filtered_from={str(args.features): sha(args.features / "metadata.json")},
        dataset_manifest_sha256=sha(args.output_data / "manifest.json"), audit_sha256=sha(args.audit))
    (args.output_features / "metadata.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output_features / "COMPLETE.json").write_text(json.dumps(dict(
        metadata_sha256=sha(args.output_features / "metadata.json"), train=result["train"],
        validation=result["validation"]), indent=2) + "\n")
    print(json.dumps(dict(data=str(args.output_data), features=str(args.output_features),
                         counts=manifest["counts"], removed=removed)))


if __name__ == "__main__":
    main()
