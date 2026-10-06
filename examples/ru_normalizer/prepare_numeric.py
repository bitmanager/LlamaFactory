"""Format bounded numeric examples using unmodified num2words; no text normalizer."""
import argparse
from collections import Counter
from decimal import Decimal
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import re

from num2words import num2words

SEED = 20261006
TEMPLATES = ["Число {}.", "Это число {}.", "Записано число {}."]
# Manually identified unambiguous whole values in the protected numeric32 panel.
# Times, telephone components and slash-separated identifiers are not interpreted.
# Values below 10 remain eligible, as agreed: the panel tests contextual usage.
PROTECTED_VALUES = [1111111, 1011, 3333, 661614025, 70707, 9000000, 188825279,
                    3978, 100, 22000, 5555555, 666666, 1234, 100000000, 3000000,
                    51000, 101, 12000000, 1001, 10, 11111111, 1917, 25, 16061192]
EDGE_INPUTS = ["0", "1", "2", "10", "11", "12", "14", "20", "21", "22", "99", "100",
               "101", "110", "111", "999", "1000", "1001", "10000", "61798", "619492",
               "999999", "0.001", "0.010", "0.100", "1.001", "9.17", "16.250", "0.3", "999999.999"]


def text_key(text):
    return " ".join(re.findall(r"[а-яa-z0-9]+", text.lower().replace("ё", "е").replace("+", "")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, nargs="+", required=True)
    parser.add_argument("--millions", action="store_true", help="Integers only: 1,000,000 <= value < 1,000,000,000")
    args = parser.parse_args()
    assert importlib.metadata.version("num2words") == "0.5.14"
    rng = random.Random(SEED)
    blocked_values = set(map(Decimal, PROTECTED_VALUES))
    blocked_text, source_hashes = set(), {}
    for path in args.holdout:
        source_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        for line in path.read_text().splitlines():
            row = json.loads(line, parse_float=Decimal)
            blocked_text.update(text_key(row[k]) for k in ("written", "spoken", "spoken_stressed") if k in row)
            if "numeric_value" in row:
                value = Decimal(row["numeric_value"])
                assert value.is_finite(), f"Nonfinite numeric_value in {path}"
                blocked_values.add(value)
    used_values = set(blocked_values)
    used_text = set(blocked_text)
    lower, upper = (1000000, 1000000000) if args.millions else (0, 1000000)
    source_name = "numeric-millions-v1" if args.millions else "numeric-canonical-v1"
    expected = {"validation": {"integer": 214}, "train": {"integer": 10000}}
    if not args.millions:
        expected["validation"]["decimal"] = 42
        expected["train"]["decimal"] = 2000

    def row_for(raw, split):
        value = Decimal(raw)
        assert value.is_finite() and lower <= value < upper
        kind = "decimal" if "." in raw else "integer"
        assert not args.millions or kind == "integer"
        if kind == "decimal":
            assert 1 <= len(raw.split(".")[1]) <= 3 and value > 0 and value != int(value)
        if value in used_values:
            return None
        spoken = num2words(raw, lang="ru", to="cardinal", case="n", gender="m", animate=False)
        template = rng.choice(TEMPLATES)
        written, spoken = template.format(raw.replace(".", ",")), template.format(spoken)
        if any(text_key(t) in used_text for t in (written, spoken)):
            return None
        used_values.add(value)
        used_text.update(map(text_key, (written, spoken)))
        value_key = format(value.normalize(), "f")
        return dict(written=written, spoken=spoken, number=raw, numeric_value=value_key,
                    decimal_scale=max(0, -value.as_tuple().exponent), category=kind, split=split,
                    source_id=f"{source_name}:{value_key}", source_group=f"numeric-value:{value_key}",
                    provenance="num2words 0.5.14, LGPL-2.1-or-later; bounded nominative cardinal; exact Decimal strings")

    splits = {"validation": [], "train": []}
    for split, counts in expected.items():
        for kind, count in counts.items():
            accepted = 0
            while accepted < count:
                if args.millions:
                    magnitude = rng.randrange(6, 9)
                    whole = rng.randrange(10 ** magnitude, 10 ** (magnitude + 1))
                else:
                    whole = rng.randrange(10 ** rng.randrange(1, 7))
                if split == "validation" and kind == "integer" and whole < 10:
                    continue  # Keep all ten base digits available for training.
                if kind == "decimal":
                    scale = rng.randrange(1, 4)
                    raw = f"{whole}.{rng.randrange(1, 10 ** scale):0{scale}d}"
                else:
                    raw = str(whole)
                row = row_for(raw, split)
                if row:
                    splits[split].append(row)
                    accepted += 1
    train_values = {Decimal(r["number"]) for r in splits["train"]}
    val_values = {Decimal(r["number"]) for r in splits["validation"]}
    assert not train_values & val_values
    assert len(train_values) == sum(expected["train"].values())
    assert len(val_values) == sum(expected["validation"].values())
    assert not (train_values | val_values) & blocked_values
    for rows in splits.values():
        rng.shuffle(rows)
    args.output.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for split, rows in splits.items():
        path = args.output / f"{split}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
    edge_inputs = ([str(n) for n in [1000000, 1000001, 1000010, 1000011, 1000021, 1000100,
                    1001000, 1010000, 1100000, 1111111, 2000000, 2100001, 5000000, 9000000,
                    9999999, 10000000, 10000001, 10101010, 11000000, 11111111, 21000000,
                    99999999, 100000000, 100000001, 101000000, 111111111, 200000000,
                    500000000, 999999998, 999999999]] if args.millions else EDGE_INPUTS)
    edges = [{"number": n, "spoken": num2words(n, lang="ru"), "qa_only": True} for n in edge_inputs]
    (args.output / "edge-review-30.json").write_text(json.dumps(edges, ensure_ascii=False, indent=2))
    manifest = dict(seed=SEED, num2words_version="0.5.14", license="LGPL-2.1-or-later",
                    upstream="https://github.com/savoirfairelinux/num2words/tree/v0.5.14",
                    counts={s: dict(Counter(r["category"] for r in rows)) for s, rows in splits.items()},
                    mode=source_name, expected_counts=expected, range_inclusive_exclusive=[lower, upper],
                    templates=TEMPLATES, protected_numeric32_values=PROTECTED_VALUES,
                    small_values_policy=("Not applicable: millions-only integers" if args.millions else
                                         "Below10 may occur in train; numeric32 evaluates contextual phrases"),
                    numeric_dedup="Decimal equality across splits and scales; no float arithmetic",
                    fulltext_dedup="casefold/yo/stress/punctuation-normalized written AND spoken",
                    validation_value_overlap=0, heldout_keys=len(blocked_text), holdout_sha256=source_hashes,
                    blocked_numeric_values=len(blocked_values),
                    files_sha256=hashes, adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    limits=("1e6 <= integer < 1e9; no decimals, negatives, dates, money, case or ordinal tasks; stress not yet added"
                            if args.millions else "No negatives, millions, dates, money, case or ordinal tasks; stress not yet added"),
                    training_eligible=False, pending="Manual edge review and stock RUAccent stress review")
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
