"""Convert pinned Ruslan accents; retain only strict agreement with stock RUAccent."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import unicodedata

from prepare_local import digest, key, record, valid

CSV_SHA = "eb9e8a386917567acf20ceeecbde443868e0f9c30571815618752ad4c1565b43"
REVISION = "ba6686afc84ec0518271ddddc7dca2ceca4d176d"
VOWELS = "аеёиоуыэюяАЕЁИОУЫЭЮЯ"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signature(text):
    """Ignore optional monosyllabic stress and explicit +ё, never е/ё content."""
    if re.search(r"\+(?![" + VOWELS + r"])", text) or any(unicodedata.combining(c) for c in text):
        raise ValueError("malformed_stress")
    def word(match):
        token = match.group(); plain = token.replace("+", "")
        marked = {i - token[:i].count("+") for i, c in enumerate(token) if c == "+"}
        marked.update(i for i, c in enumerate(plain) if c in "ёЁ")
        if token.count("+") > 1 or len(marked) > 1:
            raise ValueError("multiple_primary_stresses")
        syllables = sum(c in VOWELS for c in plain)
        if syllables > 1 and not marked:
            raise ValueError("missing_polysyllabic_stress")
        return plain if syllables <= 1 else token.replace("+ё", "ё").replace("+Ё", "Ё")
    return re.sub(r"[а-яА-ЯёЁ+]+", word, text)


def convert(identity, original):
    if not re.fullmatch(r"\d{6}_RUSLAN", identity):
        raise ValueError("invalid_source_id")
    text = unicodedata.normalize("NFC", original)
    if "+" in text:
        raise ValueError("unexpected_plus")
    target = re.sub("([" + VOWELS + "])\u0301", r"+\1", text)
    signature(target)
    written = target.replace("+", "")
    if re.search(r"\b[А-ЯЁ]{2,}\b|\b[А-ЯЁ]\.(?=\s|$)", written):
        raise ValueError("raw_abbreviation_or_initial")
    group = f"ruslan:block:{int(identity[:6]) // 100}"
    row = record(written, target, "ruslan:" + identity, group,
                 "Ruslan derivative: automatic Claude Opus text accents; no independent audio verification or human gold")
    reason = valid(row)
    if reason:
        raise ValueError(reason)
    row.update(spoken=written, original_accented=original, reference_stressed=target,
               split="validation" if int(digest(group)[:8], 16) % 20 == 0 else "train")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--csv", type=Path)
    mode.add_argument("--source", type=Path, help="Prepared source; stock outputs are sibling staging-train/staging-validation")
    parser.add_argument("--exclude", nargs="+", type=Path)
    parser.add_argument("--tokenizer", type=Path, help="Required in consensus mode to validate the retained original target")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    splits = {"train": [], "validation": []}; rejected = []; inputs = {}; ids = set()
    if args.csv:
        if sha(args.csv) != CSV_SHA or not args.exclude:
            raise ValueError("Pinned source hash and explicit train/heldout exclusions required")
        inputs[str(args.csv)] = CSV_SHA; blocked = set(); blocked_ids = set(); candidates = []
        for path in args.exclude:
            inputs[str(path)] = sha(path)
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line); blocked_ids.add(row["source_id"])
                blocked.update(key(row[f]) for f in ("written", "spoken", "spoken_stressed") if f in row)
        with args.csv.open(encoding="utf-8", newline="") as handle:
            for identity, original in csv.reader(handle, delimiter="|"):
                if identity in ids:
                    raise ValueError("Duplicate CSV ID: " + identity)
                ids.add(identity)
                try:
                    row = convert(identity, original)
                    if row["source_id"] in blocked_ids or key(row["written"]) in blocked:
                        raise ValueError("existing_train_or_heldout")
                    candidates.append(row)
                except ValueError as exc:
                    rejected.append(dict(source_id="ruslan:" + identity, original_accented=original, reason=str(exc)))
        counts = Counter(key(r["written"]) for r in candidates)
        for row in candidates:
            if counts[key(row["written"])] > 1:
                rejected.append(dict(row, reason="duplicate_normalized_input"))
            else:
                splits[row["split"]].append(row)
    else:
        if args.tokenizer is None:
            parser.error("Consensus mode requires --tokenizer")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
        inputs.update({str(p): sha(p) for p in sorted(args.tokenizer.rglob("*")) if p.is_file()})
        exclusions = {}
        for path in args.exclude or []:
            inputs[str(path)] = sha(path)
            for line in path.read_text(encoding="utf-8").splitlines():
                item = json.loads(line)
                if item["source_id"] in exclusions:
                    raise ValueError("Duplicate audited exclusion")
                exclusions[item["source_id"]] = item
        manifest_path = args.source / "manifest.json"; source_meta = json.loads(manifest_path.read_text(encoding="utf-8"))
        if source_meta["source_csv_sha256"] != CSV_SHA or source_meta["source_revision"] != REVISION:
            raise ValueError("Unpinned source provenance")
        inputs[str(manifest_path)] = sha(manifest_path)
        quarantine_path = args.source / "quarantine.jsonl"
        if sha(quarantine_path) != source_meta["files_sha256"]["quarantine"]:
            raise ValueError("Source quarantine hash mismatch")
        inputs[str(quarantine_path)] = sha(quarantine_path)
        for split in splits:
            path = args.source / f"{split}.jsonl"; accent = args.source.parent / f"staging-{split}"
            summary_path = accent / "summary.json"; summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if Path(summary["tokenizer"]).resolve() != args.tokenizer.resolve() or summary["max_target_tokens"] != 192:
                raise ValueError("Accent tokenizer/target limit mismatch")
            if sha(path) != source_meta["files_sha256"][split] or sha(path) != summary["source_sha256"] or sha(accent / "train.jsonl") != summary["output_sha256"]:
                raise ValueError("Source/accent hash mismatch")
            inputs.update({str(p): sha(p) for p in [path, summary_path, accent / "train.jsonl", accent / "rejected.jsonl"]})
            source = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
            outputs = {}
            for name in ("train.jsonl", "rejected.jsonl"):
                for line in (accent / name).read_text(encoding="utf-8").splitlines():
                    item = json.loads(line)
                    if item["source_id"] in outputs:
                        raise ValueError("Duplicate accent ID")
                    outputs[item["source_id"]] = (item, name == "rejected.jsonl")
            if len(source) != source_meta["counts"][split] or set(outputs) != {r["source_id"] for r in source}:
                raise ValueError("Accent coverage mismatch")
            for row in source:
                sid = row["source_id"]; item, was_rejected = outputs[sid]
                if (sid in ids or row["split"] != split or row["spoken_stressed"] != row["reference_stressed"]
                        or row["spoken_stressed"].replace("+", "") != row["spoken"] or row["spoken"] != row["written"]
                        or any(item[k] != v for k, v in row.items() if k != "spoken_stressed")):
                    raise ValueError("Source identity/content/split mismatch")
                ids.add(sid)
                if sid in exclusions and exclusions[sid]["original_accented"] != row["original_accented"]:
                    raise ValueError("Audited exclusion text mismatch")
                try:
                    if sid in exclusions:
                        raise ValueError("reviewed_source_quarantine")
                    if was_rejected:
                        raise ValueError("accent_rejected:" + item["rejection_reason"])
                    if signature(row["reference_stressed"]) != signature(item["spoken_stressed"]):
                        raise ValueError("stress_or_content_disagreement")
                    if len(tokenizer.encode(row["spoken_stressed"], add_special_tokens=False)) > 192:
                        raise ValueError("original_target_overlength")
                    splits[split].append(row)  # Original target, never RUAccent replacement.
                except ValueError as exc:
                    rejected.append(dict(row, reason=str(exc), ruaccent_stressed=item["spoken_stressed"]))
        if set(exclusions) - ids:
            raise ValueError("Unknown audited exclusion IDs")
    if {r["source_group"] for r in splits["train"]} & {r["source_group"] for r in splits["validation"]}:
        raise ValueError("Group split overlap")
    if sum(map(len, splits.values())) + len(rejected) != len(ids):
        raise ValueError("Row accounting mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    for name, rows in {**splits, "quarantine": rejected}.items():
        (args.output / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    meta = dict(counts={s: len(rows) for s, rows in splits.items()}, quarantine_count=len(rejected), input_sha256=inputs,
                files_sha256={s: sha(args.output / f"{s}.jsonl") for s in (*splits, "quarantine")},
                source_revision=REVISION, source_csv_sha256=CSV_SHA,
                sources=["https://huggingface.co/datasets/stilletto/ruslan-stressed", "https://ruslan-corpus.github.io/"],
                licenses={"original": "CC-BY-NC-SA-4.0", "derivative_card_claim": "CC-BY-4.0"},
                split_policy="SHA256 contiguous 100-ID block modulo20; not speaker/book-disjoint",
                training_eligible=False, pending="Independent review; stress agreement is not human gold")
    (args.output / "manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
