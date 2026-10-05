"""Add stock RUAccent pseudo-labels to existing written/spoken JSONL pairs."""
import argparse
import hashlib
import importlib.metadata
import json
import re
import shutil
from collections import Counter
from pathlib import Path


REVISION = "b78ae5ea1e62beaf138bed1865cd8c3b0b5ca855"
# Unambiguous dictionary forms; no contextual homographs or heldout examples.
CUSTOM_DICT = {
    "после": "п+осле", "времени": "вр+емени", "деньги": "д+еньги",
    "какая": "как+ая", "поняла": "понял+а", "готов": "гот+ов",
    "любой": "люб+ой", "кредите": "кред+ите", "кабелей": "к+абелей",
}


def canonical(text):
    return text.lower().replace("+", "").replace("ё", "е")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "output", "accent-models", "tokenizer"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    from ruaccent import RUAccent
    from export_qwen_features import AGENT_SYSTEM
    import onnxruntime as ort

    ort.preload_dlls(directory="")
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
    accent = RUAccent().load(omograph_model_size="turbo3.1", use_dictionary=True,
        custom_dict=CUSTOM_DICT, custom_homographs={w: [v] for w, v in CUSTOM_DICT.items()},
        providers=["CUDAExecutionProvider"], session_options=options,
        workdir=str(args.accent_models), revision=REVISION, local_files_only=True)
    providers = {}
    for name in ("omograph_model", "accent_model", "yo_homograph_model", "stress_usage_predictor"):
        session = getattr(accent, name).session
        providers[name] = session.get_providers()
        if providers[name][0] != "CUDAExecutionProvider":
            raise RuntimeError(f"{name} did not initialize CUDA")
        session.disable_fallback()  # CPU shape operators are not provider fallback.
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    prefix = tokenizer.apply_chat_template([{"role": "system", "content": AGENT_SYSTEM}],
                                          tokenize=True, add_generation_prompt=True)
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.source, args.output / "source.jsonl")
    manifest = {"source": str(args.source), "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
                "ruaccent_version": importlib.metadata.version("ruaccent"), "model_revision": REVISION,
                "custom_dict": CUSTOM_DICT, "custom_homographs": {w: [v] for w, v in CUSTOM_DICT.items()},
                "providers": providers, "tokenizer": str(args.tokenizer),
                "labels": "Google normalization + automatic RUAccent; not human gold",
                "training_eligible": False, "pending_review": "Source normalization QA/quarantine",
                "max_source_tokens": 768, "max_target_tokens": 192}
    stats, ids = Counter(), set()
    with args.source.open() as source, (args.output / "train.jsonl").open("x") as output, \
            (args.output / "rejected.jsonl").open("x") as rejected:
        for line in source:
            row = json.loads(line)
            if row["source_id"] in ids:
                raise ValueError(f"Duplicate source ID: {row['source_id']}")
            ids.add(row["source_id"])
            written, spoken = row["written"], row["spoken"]
            reason = None
            if not written.strip() or not spoken.strip() or "+" in spoken or any("<|" in x for x in (written, spoken)):
                reason = "invalid_input"
            stressed = accent.process_all(spoken) if reason is None else spoken
            if canonical(stressed) != canonical(spoken):
                reason = "content_changed"
            if re.search(r"\+(?![аеёиоуыэюяАЕЁИОУЫЭЮЯ])", stressed):
                reason = "invalid_stress_marker"
            if len(prefix) + len(tokenizer.encode(written, add_special_tokens=False)) > 768 or \
                    len(tokenizer.encode(stressed, add_special_tokens=False)) > 192:
                reason = "overlength"
            record = dict(row, spoken_stressed=stressed, stress_provenance=manifest["labels"])
            if reason:
                record["rejection_reason"] = reason
            (rejected if reason else output).write(json.dumps(record, ensure_ascii=False) + "\n")
            stats[reason or "accepted"] += 1
            if sum(stats.values()) % 100 == 0:
                output.flush()
                print(json.dumps(stats), flush=True)
    if not stats["accepted"]:
        raise ValueError("No accepted pairs")
    manifest.update(counts=dict(stats), output_sha256=hashlib.sha256((args.output / "train.jsonl").read_bytes()).hexdigest())
    (args.output / "summary.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
