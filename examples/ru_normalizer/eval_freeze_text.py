"""Fixed heldout panel, upstream greedy decoding and stock JiWER metrics."""
import argparse
import hashlib
import json
import re
from pathlib import Path

import jiwer
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer
from freeze_text import FreezeText


def sha(path):
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("root", "output", "model-config", "snapshots"):
        p.add_argument("--" + arg, required=True, type=Path)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--numeric", action="store_true", help="Evaluate the separate 32-row protected numeric probe")
    mode.add_argument("--canonical", action="store_true", help="Fixed 32 integer +32 decimal protected canonical rows")
    mode.add_argument("--clock", action="store_true", help="Fixed SHA256 32 clock heldout rows")
    mode.add_argument("--currency", action="store_true", help="Fixed SHA256 32 currency heldout rows")
    p.add_argument("--data", type=Path)
    p.add_argument("--features", type=Path)
    p.add_argument("--training-data", type=Path, help="Training JSONL checked for heldout leakage")
    p.add_argument("--weights", type=Path, nargs="+", help="Explicit immutable snapshots; result names use file stems")
    args = p.parse_args()
    features = args.root / "data/freeze-features-v2"
    data = args.root / "data/hidden-plan-v2"
    if args.numeric:
        features, data = args.root / "data/numeric-probe-features-v1", args.root / "data/numeric-probe-v1"
    if args.canonical:
        features, data = args.root / "data/freeze-canonical-features-v1", args.root / "data/numeric-canonical-v1/staging"
    if args.clock:
        features, data = args.root / "data/freeze-clock-features-v1", args.root / "data/clock-v1/staging"
    if args.currency:
        features, data = args.root / "data/freeze-currency-features-v1", args.root / "data/currency-v1/staging-clean"
    features, data = args.features or features, args.data or data
    rows = [json.loads(x) for x in (data / "validation.jsonl").read_text().splitlines()]
    training_data = args.training_data or args.root / "data/hidden-plan-v2/train.jsonl"
    train = [json.loads(x) for x in training_data.read_text().splitlines()]
    assert not {r["source_id"] for r in rows} & {r["source_id"] for r in train}
    assert not {r["source_group"] for r in rows if r.get("source_group")} & {r["source_group"] for r in train if r.get("source_group")}
    canonical = lambda s: " ".join(re.findall(r"\d+(?:[.,]\d+)*|[^\W\d_]+", s.replace("+", "").casefold().replace("ё", "е")))
    for field in ("written", "spoken_stressed"):
        assert not {canonical(r[field]) for r in rows} & {canonical(r[field]) for r in train}, f"Training overlap: {field}"
    meta = json.loads((features / "metadata.json").read_text())
    assert meta["source_sha256"]["validation"] == sha(data / "validation.jsonl")
    predicates = {"ru_abbreviation": lambda s: re.search(r"\b[А-ЯЁ]{2,}\b", s),
                  "latin": lambda s: re.search("[A-Za-z]", s),
                  "long": lambda s: len(s) >= 180, "short": lambda s: len(s) <= 110}
    if args.numeric:
        predicates = {"numeric": lambda s: re.search(r"\d", s)}
    if args.canonical:
        predicates = {"integer": lambda s: re.search(r"\d", s) and "," not in s,
                      "decimal": lambda s: "," in s}
    if args.clock or args.currency:
        predicates = {"clock" if args.clock else "currency": lambda s: True}
    selected, used = [], set()
    order = sorted(enumerate(rows), key=lambda item: hashlib.sha256(item[1]["source_id"].encode()).hexdigest())
    for stratum, predicate in predicates.items():
        size = 32 if args.numeric or args.canonical or args.clock or args.currency else 16
        choices = [(i, r) for i, r in order if i not in used and predicate(r["written"])][:size]
        assert len(choices) == size, f"Insufficient heldout rows for {stratum}"
        for i, row in choices:
            used.add(i)
            selected.append(dict(index=i, stratum=stratum, **row))
    snapshots = args.snapshots
    weights = {name: snapshots / file for name, file in
               [("v1_final", "v1-final.safetensors"), ("v2_step1800", "v2-step1800.safetensors")]}
    if args.weights:
        weights = {path.stem: path for path in args.weights}
        if len(weights) != len(args.weights):
            raise ValueError("Snapshot file stems must be unique")
    manifest = dict(panel=selected, selection="SHA256 ID; 32 clock or currency rows" if args.clock or args.currency else "SHA256 ID; 32 integers +32 decimals" if args.canonical else "SHA256 ID; 32 protected numeric rows" if args.numeric else "Priority RU-abbr, Latin, long>=180 chars, short<=110; SHA256 ID;16 each",
                    labels="Automatic stress references, not human gold", validation_sha256=sha(data / "validation.jsonl"),
                    training_file=str(training_data), training_sha256=sha(training_data),
                    feature_metadata=meta, weights={k:sha(v) for k, v in weights.items()}, max_tokens=192)
    (args.output / "panel.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    tokenizer = AutoTokenizer.from_pretrained(features / "tokenizer", local_files_only=True)
    norm = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(), jiwer.RemoveMultipleSpaces(), jiwer.Strip()])
    normalize = lambda text: norm(text.replace("+", "").replace("ё", "е").replace("Ё", "Е"))
    summaries = {}
    for name, weight in weights.items():
        model = FreezeText(args.model_config, meta["source_dim"], meta["vocab_size"])
        model.load_state_dict(load_file(str(weight)), strict=True)
        model = model.cuda().eval()
        results = []
        with (args.output / f"{name}.jsonl").open("x") as output, torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for selected_row in selected:
                row = torch.load(features / "validation" / f"{selected_row['index']:06d}.pt", map_location="cpu", weights_only=True)
                assert row["source_id"] == selected_row["source_id"] and row["reference"] == selected_row["spoken_stressed"]
                ids = model.generate(row, max_tokens=192)
                prediction = tokenizer.decode(ids, skip_special_tokens=False)
                result = dict(selected_row, reference=row["reference"], prediction=prediction, hit_token_limit=len(ids)==192,
                    reserved_decoder_ids=sum(i >= meta["vocab_size"] for i in ids),
                    invalid_stress=bool(re.search(r"\+(?![аеёиоуыэюяАЕЁИОУЫЭЮЯ])", prediction) or
                        any(w.count("+") > 1 for w in re.findall(r"[а-яёА-ЯЁ+]+", prediction))),
                    exact_with_stress=prediction==row["reference"],
                    exact_without_stress=prediction.replace("+", "")==row["reference"].replace("+", ""),
                    wer=jiwer.wer(normalize(row["reference"]), normalize(prediction)),
                    cer=jiwer.cer(normalize(row["reference"]), normalize(prediction)))
                results.append(result); output.write(json.dumps(result, ensure_ascii=False) + "\n"); output.flush()
                print(json.dumps(dict(model=name, completed=len(results))), flush=True)
        summaries[name] = {"rows":len(results), "strata":{}}
        for label, group in [("all",results)] + [(s,[r for r in results if r["stratum"]==s]) for s in predicates]:
            refs, hyps = [normalize(r["spoken_stressed"]) for r in group], [normalize(r["prediction"]) for r in group]
            summaries[name]["strata"][label] = dict(wer=jiwer.wer(refs,hyps),cer=jiwer.cer(refs,hyps),
                exact_with_stress=sum(r["exact_with_stress"] for r in group)/len(group),
                exact_without_stress=sum(r["exact_without_stress"] for r in group)/len(group),
                invalid_stress=sum(r["invalid_stress"] for r in group), token_limit=sum(r["hit_token_limit"] for r in group))
        del model
        torch.cuda.empty_cache()
        (args.output / "metrics.json").write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
