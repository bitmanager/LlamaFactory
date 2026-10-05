"""Reconstruct abbreviation inputs from existing expansions; never relabel targets."""
import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from export_qwen_features import AGENT_SYSTEM
from prepare_local import TEXT_FIELDS, key, read, valid


# Quarantine whole families, including their otherwise plausible variants.
REJECT = {
    "ОГРНИП": "missing И in recorded expansion",
    "EXW": "variant missing E",
    "EAD": "variant missing E",
    "ESIM": "variant missing E",
    "ИМТ": "variant missing И",
    "TEU": "variant missing T and E",
    "ИПВ": "variant missing И",
    "ИСЖ": "variant missing И",
    "OP": "incomplete extracted span: пи instead of о пи",
    "ИНН": "distorted expansion и нэ нэн",
    "ТЭО": "review required: тэ о may collapse a letter",
    "АУСН": "review required: а у сэн ambiguous letter coverage",
}
TEXT_FIELDS = (*TEXT_FIELDS, "written", "spoken_stressed")


def reconstruct(row):
    """Replace exactly one complete, contiguous token span; keep all other bytes."""
    words = row["words"]
    tokens = list(re.finditer(r"[0-9A-Za-zА-Яа-яЁё]+", words))
    expansion = key(row["acronym_expansion"]).split()
    normalized = [key(token.group()) for token in tokens]
    matches = [i for i in range(len(tokens) - len(expansion) + 1)
               if normalized[i:i + len(expansion)] == expansion]
    if not expansion or len(matches) != 1:
        raise ValueError("nonunique_expansion_span")
    i = matches[0]
    start, end = tokens[i].start(), tokens[i + len(expansion) - 1].end()
    span = words[start:end]
    written = words[:start] + row["acronym"] + words[end:]
    if written[:start] + span + written[start + len(row["acronym"]):] != words:
        raise ValueError("roundtrip_failed")
    return written, start, span


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--existing-data", type=Path, required=True)
    parser.add_argument("--label-audit", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--merged-output", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    campaign = args.root / "grpo/grpo_dpo_finetuning/campaign_v2/data"
    source = campaign / "corpus/abbrev_rows.jsonl"
    audit = json.loads(args.label_audit.read_text())
    stress_rules = {word: re.compile(r"(?<![\w+])" + re.escape(word) + r"(?![\w+])", re.I)
                    for word, rule in audit["rules"].items()
                    if rule["classification"] == "wrong_stress"}
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    prefix = tokenizer.apply_chat_template([{"role": "system", "content": AGENT_SYSTEM}],
                 tokenize=True, add_generation_prompt=True, return_dict=False)
    base = read(args.existing_data / "train.jsonl") if args.merged_output else []
    base_ids = {r["source_id"] for r in base}
    base_keys = {key(r[f]) for r in base for f in ("written", "spoken_stressed")}
    heldout_paths = sorted(set(
        list((args.root / "eval/data/sets").glob("*.jsonl"))
        + list((args.root / "eval/data/raw").glob("*/*.jsonl"))
        + list(campaign.rglob("eval_guard*.jsonl"))
        + list(campaign.rglob("probe.jsonl"))
        + list((campaign / "evals").glob("*.jsonl"))
        + [campaign / "v36/homograph_eval.jsonl",
           args.root / "data/eqv36-dual/holdout.jsonl",
           args.existing_data / "validation.jsonl",
           args.existing_data / "eval_existing.jsonl"]))
    heldout, heldout_ids = set(), set()
    for path in heldout_paths:
        for row in read(path):
            heldout.update(key(row[f]) for f in TEXT_FIELDS
                           if isinstance(row.get(f), str) and row[f].strip())
            for field in ("id", "source_id"):
                if row.get(field) is not None:
                    value = str(row[field])
                    heldout_ids.update((value, value.removeprefix("corpus:")))

    candidates, rejected, lexicon = [], [], defaultdict(Counter)
    rows = read(source)
    for row in rows:
        acronym = row["acronym"]
        lexicon[acronym][row["acronym_expansion"]] += 1
        reason = None
        wrong_stress = [word for word, pattern in stress_rules.items() if pattern.search(row["stressed"])]
        try:
            written, start, span = reconstruct(row)
        except ValueError as error:
            reason = str(error)
        if reason is None:
            item = dict(written=written, spoken_stressed=row["stressed"],
                        source_id="corpus:" + row["id"],
                        provenance=dict(
                            kind="reconstructed_not_original", source=row["source"],
                            source_file=str(source), original_row_id=row["id"],
                            acronym=acronym, expansion=row["acronym_expansion"],
                            replacement_start=start, original_span=span,
                            labels="existing automatic labels, not human gold"))
            reason = valid(item)
            if key(row["words"]) != key(row["stressed"]):
                reason = "source_content_mismatch"
            if acronym.upper() in REJECT:
                reason = "quarantined_acronym"
            if wrong_stress:
                reason = "audited_wrong_stress"
            if len(prefix) + len(tokenizer.encode(written, add_special_tokens=False)) > 768:
                reason = "source_token_limit"
            if len(tokenizer.encode(row["stressed"], add_special_tokens=False)) > 192:
                reason = "target_token_limit"
            if (item["source_id"] in base_ids or any(key(item[f]) in base_keys
                    for f in ("written", "spoken_stressed"))):
                reason = "base_train_duplicate"
            if (row["id"] in heldout_ids or item["source_id"] in heldout_ids
                    or any(key(item[f]) in heldout for f in ("written", "spoken_stressed"))):
                reason = "existing_heldout"
        if reason:
            rejected.append(dict(source_id="corpus:" + row["id"], acronym=acronym,
                                 reason=reason, audited_wrong_stress=wrong_stress))
        else:
            candidates.append(item)

    variants = defaultdict(set)
    for item in candidates:
        variants[key(item["written"])].add(key(item["spoken_stressed"], stress=True))
    train, seen_written, seen_spoken = [], set(), set()
    for item in candidates:
        written, spoken = key(item["written"]), key(item["spoken_stressed"])
        reason = None
        if len(variants[written]) != 1:
            reason = "conflicting_target"
        elif written in seen_written or spoken in seen_spoken:
            reason = "duplicate_written_or_spoken"
        if reason:
            rejected.append(dict(source_id=item["source_id"],
                                 acronym=item["provenance"]["acronym"], reason=reason))
        else:
            train.append(item)
            seen_written.add(written)
            seen_spoken.add(spoken)
    assert len(train) + len(rejected) == len(rows)
    assert train and not (seen_written | seen_spoken) & heldout
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "label-audit.json").write_bytes(args.label_audit.read_bytes())
    for name, values in (("train", train), ("rejected", rejected)):
        (args.output / f"{name}.jsonl").write_text(
            "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values))
    examples, acronyms = [], set()
    for item in train:
        acronym = item["provenance"]["acronym"]
        if acronym not in acronyms and key(acronym) != key(item["provenance"]["expansion"]):
            examples.append(item)
            acronyms.add(acronym)
        if len(examples) == 20:
            break
    summary = dict(source_rows=len(rows), train_rows=len(train),
        rejected=dict(Counter(row["reason"] for row in rejected)),
        source_counts=dict(Counter(row["provenance"]["source"] for row in train)),
        actual_expansion_rows=sum(key(r["provenance"]["acronym"]) !=
                                  key(r["provenance"]["expansion"]) for r in train),
        acronym_families=len({r["provenance"]["acronym"] for r in train}),
        quarantine={a:dict(reason=reason, source_rows=sum(lexicon[a].values()))
                    for a, reason in REJECT.items()},
        heldout_keys=len(heldout), heldout_ids=len(heldout_ids),
        token_limits=dict(source_including_chat_prefix=768, target=192, prefix=len(prefix)),
        tokenizer_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(args.tokenizer.iterdir()) if p.is_file()},
        audited_wrong_stress_rules=list(stress_rules),
        audited_wrong_stress_hits=dict(Counter(w for r in rejected for w in r.get("audited_wrong_stress", []))),
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [source, args.label_audit, *heldout_paths]},
        limitations=["Reconstructed inputs, not recovered original written strings",
                     "Surrounding text may already be normalized; numeric source lost punctuation",
                     "Targets unchanged, automatic labels are not human gold",
                     "Фэ/Эф and Пайтон variants are not silently rewritten",
                     "No fabricated history, audio or independent new validation",
                     "Separate candidate pool; never appended to active training"],
        examples=examples)
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (args.output / "acronym_inventory.json").write_text(
        json.dumps(dict(sorted(lexicon.items())), ensure_ascii=False, indent=2) + "\n")
    if args.merged_output:
        combined = base + train
        assert len({r["source_id"] for r in combined}) == len(combined)
        args.merged_output.mkdir(parents=True, exist_ok=False)
        (args.merged_output / "train.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in combined))
        for split in ("validation", "eval_existing"):
            (args.merged_output / f"{split}.jsonl").hardlink_to(args.existing_data / f"{split}.jsonl")
        inputs = [args.existing_data / f"{s}.jsonl" for s in ("train", "validation", "eval_existing")]
        inputs.append(args.output / "train.jsonl")
        manifest = dict(base_rows=len(base), added_rows=len(train), train_rows=len(combined),
            source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
            output_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted(args.merged_output.glob("*.jsonl"))},
            recipe="base train in original order + filtered abbreviations; validation/eval hardlinks",
            abbreviation_summary=str(args.output / "summary.json"))
        (args.merged_output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k:v for k,v in summary.items() if k not in ("source_sha256", "examples")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
