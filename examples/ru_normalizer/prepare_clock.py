"""Format a bounded nominative clock dataset using num2words and pymorphy3."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import re

from num2words import num2words
import pymorphy3

SEED = 20261007
TEMPLATES = ["Сейчас {}.", "Время {}."]
EDGE_TIMES = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 11), (2, 2),
              (2, 21), (4, 4), (4, 59), (5, 5), (5, 11), (11, 0), (11, 11),
              (11, 21), (21, 1), (21, 2), (21, 21), (22, 22), (23, 59)]


def text_key(text):
    return " ".join(re.findall(r"[а-яa-z0-9]+", text.lower().replace("ё", "е").replace("+", "")))


def prepare_context(args, versions, nouns):
    """Keep the reviewed clock-value split; let stock libraries inflect units."""
    sha = lambda content: hashlib.sha256(content).hexdigest()
    manifest_bytes = (args.context_from / "manifest.json").read_bytes()
    base_manifest = json.loads(manifest_bytes)
    base, input_hashes, seen = {}, {}, set()
    for split in ["train", "validation"]:
        content = (args.context_from / f"{split}.jsonl").read_bytes()
        input_hashes[split] = sha(content)
        if input_hashes[split] != base_manifest["files_sha256"][split]:
            raise ValueError(f"Base {split} hash mismatch")
        base[split] = [json.loads(line) for line in content.splitlines()]
        if len(base[split]) != base_manifest["counts"][split]:
            raise ValueError(f"Base {split} count mismatch")
        for row in base[split]:
            h, m, clock = row["hour"], row["minute"], row["clock"]
            if (type(h) is not int or type(m) is not int or not 0 <= h < 24 or not 0 <= m < 60
                    or clock != f"{h:02}:{m:02}" or clock in seen
                    or row["source_id"] != f"clock-v1:{clock}"
                    or row["source_group"] != f"clock-value:{clock}" or row["split"] != split):
                raise ValueError(f"Invalid/duplicate base clock identity or split: {row['source_id']}")
            seen.add(clock)
    protected, blocked_text, holdout_hashes = set(), set(), {}
    for path in args.holdout:
        content = path.read_bytes()
        holdout_hashes[str(path)] = sha(content)
        for line in content.splitlines():
            row = json.loads(line)
            blocked_text.update(text_key(row[k]) for k in ["written", "spoken", "spoken_stressed"] if k in row)
            protected.update(f"{int(h):02}:{m}" for h, m in
                             re.findall(r"(?<!\d)([01]?\d|2[0-3]):([0-5]\d)(?!\d)", row.get("written", "")))
    templates = {"g": ["Жду до {}.", "Продолжим после {}."],
                 "d": ["Приходите к {}.", "Подготовьтесь к {}."]}
    rng = random.Random(SEED)
    splits, quarantine = {"train": [], "validation": []}, []
    for split, rows in base.items():
        for row in rows:
            h, m, clock = row["hour"], row["minute"], row["clock"]
            for case, grammeme in [("g", "gent"), ("d", "datv")]:
                identity = dict(source_id=f"clock-oblique-v1:{case}:{clock}",
                                source_group=row["source_group"], split=split, base_source_id=row["source_id"])
                if case == "d" and (h == 0 or m == 0):
                    quarantine.append(dict(**identity, reason="unsupported stock dative zero: hour or minute is zero"))
                    continue
                phrase = " ".join([num2words(h, lang="ru", gender="m", case=case),
                                   nouns[0].inflect({grammeme}).make_agree_with_number(h).word,
                                   num2words(m, lang="ru", gender="f", case=case),
                                   nouns[1].inflect({grammeme}).make_agree_with_number(m).word])
                template = rng.choice(templates[case])
                written, target = template.format(clock), template.format(phrase)
                # Existing clock validation is retained as validation, never moved to train.
                if split == "train" and (clock in protected or text_key(written) in blocked_text
                                         or text_key(target) in blocked_text):
                    quarantine.append(dict(**identity, reason="protected heldout value or full text"))
                    continue
                splits[split].append(dict(**identity, written=written, spoken=target, clock=clock,
                                          hour=h, minute=m, case=case, category="clock_oblique",
                                          provenance="Derived clock-v1 split; num2words 0.5.14 case + pymorphy3 2.0.6 inflect/make_agree_with_number; synthetic, not gold"))
    args.output.mkdir(parents=True, exist_ok=False)
    for split, rows in splits.items():
        (args.output / f"{split}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    (args.output / "quarantine.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in quarantine))
    manifest = dict(seed=SEED, versions=versions, templates=templates,
                    licenses=base_manifest["licenses"], upstream=base_manifest["upstream"],
                    source_directory=str(args.context_from), source_manifest_sha256=sha(manifest_bytes),
                    source_files_sha256=input_hashes, counts={k: len(v) for k, v in splits.items()},
                    source_counts={k: len(v) for k, v in base.items()}, quarantine_count=len(quarantine),
                    quarantine_sha256=sha((args.output / "quarantine.jsonl").read_bytes()),
                    unsupported_dative_clock_values=83, holdout_sha256=holdout_hashes,
                    protected_clock_values=sorted(protected), split_policy="Inherit clock-v1 value groups; heldout exclusions apply to train only",
                    files_sha256={s: sha((args.output / f"{s}.jsonl").read_bytes()) for s in splits},
                    adapter_sha256=sha(Path(__file__).read_bytes()), training_eligible=False,
                    pending="Independent source and new sentence stress QA; no accents or audio yet",
                    limitations="Explicit 24-hour units only; genitive after до/после, dative after к; dative zero unsupported; no general normalization")
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, nargs="+", required=True)
    parser.add_argument("--context-from", type=Path, help="Reviewed clock-v1 source/staging directory; inherit its value splits")
    args = parser.parse_args()
    versions = {p: importlib.metadata.version(p) for p in
                ["num2words", "pymorphy3", "pymorphy3-dicts-ru"]}
    assert versions == {"num2words": "0.5.14", "pymorphy3": "2.0.6",
                        "pymorphy3-dicts-ru": "2.4.417150.4580142"}
    morph = pymorphy3.MorphAnalyzer(lang="ru")
    nouns = [next(p for p in morph.parse(word) if {"NOUN", "sing", "nomn"} <= p.tag.grammemes)
             for word in ["час", "минута"]]
    if args.context_from is not None:
        prepare_context(args, versions, nouns)
        return

    def spoken(hour, minute):
        return " ".join([num2words(hour, lang="ru", gender="m", case="n"),
                         nouns[0].make_agree_with_number(hour).word,
                         num2words(minute, lang="ru", gender="f", case="n"),
                         nouns[1].make_agree_with_number(minute).word])

    protected, blocked_text, hashes = set(), set(), {}
    for path in args.holdout:
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        for line in path.read_text().splitlines():
            row = json.loads(line)
            blocked_text.update(text_key(row[k]) for k in ["written", "spoken", "spoken_stressed"] if k in row)
            protected.update(f"{int(h):02}:{m}" for h, m in
                             re.findall(r"(?<!\d)([01]?\d|2[0-3]):([0-5]\d)(?!\d)", row.get("written", "")))
    rng = random.Random(SEED)
    values = [(h, m) for h in range(24) for m in range(60) if f"{h:02}:{m:02}" not in protected]
    rng.shuffle(values)
    splits = {"train": [], "validation": []}
    rejected = []
    for index, (hour, minute) in enumerate(values):
        clock = f"{hour:02}:{minute:02}"
        split = "validation" if index < 128 else "train"
        template = rng.choice(TEMPLATES)
        written, target = template.format(clock), template.format(spoken(hour, minute))
        if text_key(written) in blocked_text or text_key(target) in blocked_text:
            rejected.append(clock)
            continue
        splits[split].append(dict(written=written, spoken=target, clock=clock, hour=hour, minute=minute,
                                 split=split, source_id=f"clock-v1:{clock}", source_group=f"clock-value:{clock}",
                                 category="clock_nominative", provenance="num2words 0.5.14 + pymorphy3 2.0.6 numeral agreement; fixed nominative template"))
    assert not {r["source_group"] for r in splits["train"]} & {r["source_group"] for r in splits["validation"]}
    args.output.mkdir(parents=True, exist_ok=False)
    for split, rows in splits.items():
        (args.output / f"{split}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    edges = [dict(clock=f"{h:02}:{m:02}", spoken=spoken(h, m), qa_only=True) for h, m in EDGE_TIMES]
    (args.output / "edge-review-20.json").write_text(json.dumps(edges, ensure_ascii=False, indent=2))
    manifest = dict(seed=SEED, versions=versions, licenses={"num2words": "LGPL-2.1-or-later",
                    "pymorphy3_code": "MIT", "pymorphy3_dicts_code": "MIT", "OpenCorpora_dictionary_data": "CC-BY-SA-3.0"},
                    upstream=["https://github.com/savoirfairelinux/num2words/tree/v0.5.14",
                              "https://github.com/no-plagiarism/pymorphy3", "https://opencorpora.org/"],
                    templates=TEMPLATES, counts={k: len(v) for k, v in splits.items()},
                    candidate_clock_values=1440, protected_clock_values=sorted(protected),
                    rejected_fulltext=rejected, holdout_sha256=hashes,
                    files_sha256={s: hashlib.sha256((args.output / f"{s}.jsonl").read_bytes()).hexdigest() for s in splits},
                    adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    training_eligible=False, pending="Independent linguistic QA; no accents or audio yet",
                    limitations="Only 24-hour HH:MM with explicit hour/minute units in nominative; no prepositions, dates, time ranges, colloquial readings or omitted zero minutes")
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
