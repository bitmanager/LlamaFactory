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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    versions = {p: importlib.metadata.version(p) for p in
                ["num2words", "pymorphy3", "pymorphy3-dicts-ru"]}
    assert versions == {"num2words": "0.5.14", "pymorphy3": "2.0.6",
                        "pymorphy3-dicts-ru": "2.4.417150.4580142"}
    morph = pymorphy3.MorphAnalyzer(lang="ru")
    nouns = [next(p for p in morph.parse(word) if {"NOUN", "sing", "nomn"} <= p.tag.grammemes)
             for word in ["час", "минута"]]

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
