"""Format bounded RUB pairs using unmodified num2words currency conversion."""
import argparse
from decimal import Decimal
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random

from num2words import num2words
from prepare_numeric import PROTECTED_VALUES, text_key

SEED = 20261008
TEMPLATES = [("К оплате {} ₽.", "К оплате {}."), ("Сумма: {} ₽.", "Сумма: {}.")]
EDGES = ["0", "0.00", "0.01", "0.02", "0.05", "0.11", "0.21", "0.99", "1", "1.00",
         "1.01", "2.21", "4.04", "5.05", "11.11", "21.22", "22.21", "101.01", "1000.01", "999999.99"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, nargs="+", required=True)
    parser.add_argument("--millions", action="store_true", help="RUB: 1,000,000 <= whole amount < 1,000,000,000")
    args = parser.parse_args()
    assert importlib.metadata.version("num2words") == "0.5.14"
    blocked_values = set(map(Decimal, PROTECTED_VALUES))
    blocked_text, source_hashes = set(), {}
    for path in args.holdout:
        source_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        for line in path.read_text().splitlines():
            row = json.loads(line, parse_float=Decimal)
            blocked_text.update(text_key(row[k]) for k in ["written", "spoken", "spoken_stressed"] if k in row)
            if "numeric_value" in row:
                value = Decimal(row["numeric_value"])
                assert value.is_finite(), f"Nonfinite numeric_value in {path}"
                blocked_values.add(value)
    rng = random.Random(SEED)
    used = set(blocked_values)
    used_text = set(blocked_text)
    lower, upper = (1000000, 1000000000) if args.millions else (0, 1000000)
    source_name = "currency-millions-v1" if args.millions else "currency-v1"
    splits = {"validation": [], "train": []}
    for split, count in [("validation", 128), ("train", 4000)]:
        while len(splits[split]) < count:
            if args.millions:
                magnitude = rng.randrange(6, 9)
                whole = rng.randrange(10 ** magnitude, 10 ** (magnitude + 1))
            else:
                whole = rng.randrange(10 ** rng.randrange(1, 7))
            cents = rng.randrange(100)
            raw = f"{whole}.{cents:02}"
            value = Decimal(raw)
            assert lower <= value < upper and value.as_tuple().exponent == -2
            if value in used:
                continue
            # Never pass int (interpreted as cents) or float to the currency API.
            target = num2words(value, lang="ru", to="currency", currency="RUB", separator="")
            written_template, target_template = rng.choice(TEMPLATES)
            written, spoken = written_template.format(raw.replace(".", ",")), target_template.format(target)
            if any(text_key(t) in used_text for t in [written, spoken]):
                continue
            used.add(value)
            used_text.update(text_key(t) for t in [written, spoken])
            canonical = format(value.normalize(), "f")
            splits[split].append(dict(written=written, spoken=spoken, number=raw, numeric_value=canonical,
                                     decimal_scale=2, currency="RUB", category="currency_rub_nominative", split=split,
                                     source_id=f"{source_name}:RUB:{canonical}", source_group=f"numeric-value:{canonical}",
                                     provenance="num2words 0.5.14 currency=RUB, Decimal exact cents; fixed nominative template"))
    assert not {r["source_group"] for r in splits["train"]} & {r["source_group"] for r in splits["validation"]}
    assert {s: len(rows) for s, rows in splits.items()} == {"validation": 128, "train": 4000}
    assert not {Decimal(r["numeric_value"]) for rows in splits.values() for r in rows} & blocked_values
    args.output.mkdir(parents=True, exist_ok=False)
    for split, rows in splits.items():
        (args.output / f"{split}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    edge_inputs = (["1000000.00", "1000000.01", "1000000.02", "1000000.05", "1000000.11",
                    "1000000.21", "1000000.99", "1000001.01", "1000002.21", "1000011.11",
                    "1000021.22", "9999999.99", "10000000.00", "10000001.01", "99999999.99",
                    "100000000.00", "100000001.01", "111111111.11", "999999999.00", "999999999.99"]
                   if args.millions else EDGES)
    edges = [dict(number=n, spoken=num2words(Decimal(n), lang="ru", to="currency", currency="RUB", separator=""), qa_only=True) for n in edge_inputs]
    (args.output / "edge-review-20.json").write_text(json.dumps(edges, ensure_ascii=False, indent=2))
    manifest = dict(seed=SEED, num2words_version="0.5.14", license="LGPL-2.1-or-later",
                    upstream="https://github.com/savoirfairelinux/num2words/tree/v0.5.14", templates=TEMPLATES,
                    counts={s: len(rows) for s, rows in splits.items()},
                    mode=source_name, expected_counts={"validation": 128, "train": 4000},
                    range_inclusive_exclusive=[lower, upper],
                    holdout_sha256=source_hashes, protected_numeric32_values=PROTECTED_VALUES,
                    blocked_numeric_values=len(blocked_values), blocked_fulltext_keys=len(blocked_text),
                    files_sha256={s: hashlib.sha256((args.output / f"{s}.jsonl").read_bytes()).hexdigest() for s in splits},
                    adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    parameters=dict(lang="ru", to="currency", currency="RUB", separator=""),
                    input_type="Decimal from exact fixed-point string; never int/float",
                    limits=f"{lower} <= RUB < {upper}, exactly two decimal places; no negatives, rounding, exchange rates, other currencies or oblique contexts",
                    training_eligible=False, pending="Independent source QA; no accents/audio yet")
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
