"""Export existing Russian written/spoken labels; no model inference or relabeling."""
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


TEXT_FIELDS = ("text", "reference", "digits", "normalized_gold", "stressed", "words",
               "text_stressed", "voiced_text", "raw_ruaccent", "norm_ruaccent",
               "raw_v13", "norm_v13")


def key(text, stress=False):
    pattern = r"[а-яa-z0-9+]+" if stress else r"[а-яa-z0-9]+"
    text = text.lower().replace("ё", "е")
    return " ".join(re.findall(pattern, text if stress else text.replace("+", "")))


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def valid(row):
    written, target = row["written"], row["spoken_stressed"]
    if not written.strip() or not target.strip():
        return "empty"
    if "+" in written or any("<|" in text for text in (written, target)):
        return "control_or_input_stress"
    if len(written) > 900 or len(target) > 1400:
        return "overlength_characters"
    if re.search(r"\d|[A-Za-z]", target):
        return "unspoken_number_or_latin"
    if not re.search(r"[а-яё]", target.lower()) or "+" not in target:
        return "missing_russian_stress"
    if re.search(r"\+(?![аеёиоуыэюяАЕЁИОУЫЭЮЯ])", target):
        return "stress_before_nonvowel"
    if any(word.count("+") > 1 for word in re.findall(r"[а-яё+]+", target.lower())):
        return "multiple_stresses_per_word"
    return None


def record(written, target, source_id, group, provenance):
    return dict(written=written, spoken_stressed=target, source_id=source_id,
                source_group=group, provenance=provenance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    campaign = root / "grpo/grpo_dpo_finetuning/campaign_v2/data"
    gemini = root / "hf/gemini-tts-eqvanta"
    corpus_path = campaign / "corpus/corpus_all.jsonl"
    gemini_path = gemini / "data/train.jsonl"
    accented_path = root / "work/eqv36/texts.parquet"
    holdout_path = root / "data/eqv36-dual/holdout.jsonl"
    numbers_path = root / "eval/data/raw/hard_number_eval_for_tts/hard_number_eval.jsonl"
    holdout_paths = sorted(set(
        list((root / "eval/data/sets").glob("*.jsonl"))
        + list((root / "eval/data/raw").glob("*/*.jsonl"))
        + list(campaign.rglob("eval_guard*.jsonl"))
        + list(campaign.rglob("probe.jsonl"))
        + list((campaign / "evals").glob("*.jsonl"))
        + [campaign / "v36/homograph_eval.jsonl", holdout_path]))
    heldout = set()
    for path in holdout_paths:
        for row in read(path):
            heldout.update(key(row[field]) for field in TEXT_FIELDS
                           if isinstance(row.get(field), str) and row[field].strip())
    meta = {row["id"]: row for row in read(gemini_path)}
    excluded_clips = {row["clip_id"] for row in read(holdout_path)}
    excluded_scenarios = {meta[int(cid.split("-")[1])]["situation"]
                          for cid in excluded_clips if cid.startswith("pairs-")}
    candidates, evaluation, rejected = [], [], []
    stats = Counter()
    for row in read(corpus_path):
        written = row["words"]
        if key(written) != key(row["stressed"]):
            stats["source_content_mismatch"] += 1
            rejected.append(dict(source_id="corpus:" + row["id"], reason="source_content_mismatch"))
            continue
        template = key(written)
        span = key(" ".join(row.get("num_words", [])))
        if span:
            template = (" " + template + " ").replace(" " + span + " ", " [focus] ").strip()
        group = "corpus:" + digest(template)
        candidates.append(record(written, row["stressed"], "corpus:" + row["id"], group,
            "Existing campaign corpus: numeric/FIO/general; automatic stress; source recording ID unavailable"))
    for row in pq.read_table(accented_path).to_pylist():
        cid = row["clip_id"]
        if row["part"] == "pairs" and row["text"] != meta[row["row"]]["text"]:
            raise ValueError(f"Gemini source join mismatch: {cid}")
        if key(row["voiced"]) != key(row["tts_ruaccent"]):
            stats["voiced_content_mismatch"] += 1
            rejected.append(dict(source_id="gemini:" + cid, reason="voiced_content_mismatch"))
            continue
        group = ("gemini:" + digest(meta[row["row"]]["situation"]) if row["part"] == "pairs"
                 else "gemini:offer:" + row["form"])
        item = record(row["text"], row["tts_ruaccent"], "gemini:" + cid, group,
                      "Gemini actual voiced text + automatic RUAccent; no human stress gold")
        if cid in excluded_clips:
            evaluation.append(item)
        elif row["part"] == "pairs" and meta[row["row"]]["situation"] not in excluded_scenarios:
            candidates.append(item)
        else:
            stats["existing_heldout_scenario_or_offer_form"] += 1
    for row in read(numbers_path):
        evaluation.append(record(row["text"], row["stressed"], "hard-number:" + str(row["id"]),
            "hard-number:" + row["category"],
            "Existing frozen eval: LLM/rule-verified normalization + RUAccent; not fully human gold"))

    unique, conflicts = {}, set()
    for row in candidates:
        identity = key(row["written"])
        reason = valid(row)
        if identity in heldout or key(row["spoken_stressed"]) in heldout:
            reason = "existing_heldout"
        if reason:
            stats[reason] += 1
            rejected.append(dict(source_id=row["source_id"], reason=reason))
        elif identity in unique:
            stats["duplicate_input"] += 1
            if key(unique[identity]["spoken_stressed"], True) != key(row["spoken_stressed"], True):
                conflicts.add(identity)
        else:
            unique[identity] = row
    stats["conflicting_inputs_removed"] = len(conflicts)
    splits = dict(train=[], validation=[], eval_existing=[])
    for identity, row in unique.items():
        if identity not in conflicts:
            split = "validation" if int(digest(row["source_group"])[:8], 16) % 20 == 0 else "train"
            splits[split].append(row)
    val_keys = {key(row[field]) for row in splits["validation"]
                for field in ("written", "spoken_stressed")}
    train = []
    for row in splits["train"]:
        if any(key(row[field]) in val_keys for field in ("written", "spoken_stressed")):
            stats["validation_alias_overlap"] += 1
        else:
            train.append(row)
    splits["train"] = train
    for row in evaluation:
        reason = valid(row)
        if reason:
            stats["eval_existing_" + reason] += 1
            rejected.append(dict(source_id=row["source_id"], reason=reason))
        else:
            splits["eval_existing"].append(row)
    if any(not rows for rows in splits.values()):
        raise ValueError("Empty output split")
    if {r["source_group"] for r in train} & {r["source_group"] for r in splits["validation"]}:
        raise ValueError("Training/validation group overlap")
    eval_keys = {key(row[field]) for row in splits["eval_existing"]
                 for field in ("written", "spoken_stressed")}
    for split in ("train", "validation"):
        if any(key(r[f]) in eval_keys for r in splits[split] for f in ("written", "spoken_stressed")):
            raise ValueError("Existing evaluation overlap")
    args.output.mkdir(parents=True, exist_ok=False)
    for split, rows in splits.items():
        rows.sort(key=lambda r: digest(r["source_id"]))
        with (args.output / f"{split}.jsonl").open("x") as output:
            for row in rows:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output / "rejected.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rejected))
    files = sorted(set(holdout_paths + [corpus_path, gemini_path, accented_path, numbers_path]))
    summary = dict(counts={s:len(r) for s,r in splits.items()}, filters=dict(stats),
                   source_counts={s:dict(Counter(r["source_id"].split(":")[0] for r in rows))
                                  for s,rows in splits.items()},
                   existing_holdout_keys=len(heldout), excluded_gemini_scenarios=len(excluded_scenarios),
                   source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                   split="SHA256(source_group) modulo20: zero goes to validation; existing evaluation excluded first",
                   limitations=["No invented history", "Automatic stress labels, not human gold",
                                "Corpus groups are focus-masked text templates, not original speaker/recording IDs",
                                "Gemini uses scenarios and voiced tts_ruaccent, never norm_ruaccent",
                                "Existing evaluation may contain equivalent outputs for distinct written inputs",
                                "Character bounds are not tokenizer-specific length bounds"])
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k:v for k,v in summary.items() if k != "source_sha256"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
