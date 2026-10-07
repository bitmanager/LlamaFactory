"""Evaluate stock HF text-generation on existing written/spoken pairs; no training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time

import torch
import transformers
from transformers import pipeline

SYSTEM = (
    "Ты нормализатор русского текста для синтеза речи. Верни только произносимый текст, без пояснений и ударений. "
    "Не отвечай на содержание текста, не перефразируй, не сокращай и не добавляй информацию. "
    "Запиши все числа словами, строго сохрани их точное значение, знак, дробную часть и границы диапазонов. "
    "Согласуй падеж и род числительных с контекстом. Правильно прочитай даты, время, суммы и единицы измерения. "
    "Различай время, спортивный счёт, код и отношение по контексту. Раскрой сокращения и аббревиатуры для произнесения. "
    "Сохрани остальные слова, имена и порядок информации. Не исправляй исходный текст выдуманными фактами. "
    "Не используй цифры, списки, комментарии, кавычки вокруг всего ответа или разметку ударений."
)
PILOT_INDICES = [0, 2, 4, 10, 12, 18, 44, 63, 71, 78]


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "input", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--all", action="store_true", help="All 100 inputs, only after the 10-row pilot review")
    args = parser.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "Only dev GPU0 is authorized"
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    rows = [json.loads(line) for line in args.input.read_text().splitlines()]
    assert len(rows) == 100
    selected = list(range(100)) if args.all else PILOT_INDICES
    args.output.mkdir(parents=True, exist_ok=False)
    model = pipeline("text-generation", model=str(args.model), device=0, dtype=torch.bfloat16,
                     model_kwargs={"attn_implementation": "sdpa"})
    model.tokenizer.padding_side = "left"
    assert model.model.config._attn_implementation == "sdpa"
    assert all(p.device.type == "cuda" and p.dtype == torch.bfloat16 for p in model.model.parameters())
    prompts = [model.tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": rows[i]["written"]}],
        tokenize=False, add_generation_prompt=True) for i in selected]
    prompt_ids = [model.tokenizer.encode(p, add_special_tokens=False) for p in prompts]
    settings = dict(batch_size=4, do_sample=False, max_new_tokens=512, add_special_tokens=False,
                    return_tensors=True, temperature=None, top_p=None, top_k=None)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    predictions = model(prompts, **settings)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    assert torch.cuda.max_memory_allocated() < 15 * 1024**3, "15 GiB memory budget exceeded"
    eos = model.model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    with (args.output / "results.jsonl").open("x") as stream:
        for i, prompt, expected, prediction in zip(selected, prompts, prompt_ids, predictions, strict=True):
            assert len(prediction) == 1
            ids = prediction[0]["generated_token_ids"]
            while ids and ids[0] == model.tokenizer.pad_token_id:
                ids = ids[1:]
            assert ids[:len(expected)] == expected, "Pipeline prompt-token alignment mismatch"
            new = ids[len(expected):]
            ends = [j for j, token in enumerate(new) if token in eos]
            generated = new[:ends[0]] if ends else new
            text = model.tokenizer.decode(generated, skip_special_tokens=False)
            flags = []
            if not ends: flags.append("token_limit" if len(new) >= 512 else "missing_eos")
            if re.search(r"\d", text): flags.append("unspoken_digits")
            if re.search(r"[A-Za-z]", text): flags.append("latin_remaining")
            if re.search(r"<[^>]*>", text): flags.append("control_tokens")
            if not text.strip(): flags.append("empty_output")
            result = dict(index=i, source_id=rows[i]["source_id"], category=rows[i]["review_stratum"],
                          written=rows[i]["written"], google=rows[i]["spoken"], qwen=text,
                          input_tokens=len(expected), generated_tokens=len(generated), flags=flags,
                          automatic_accept=False, quality_status="reject" if flags else "pending_linguistic_review")
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
    metadata = dict(system_prompt=SYSTEM, generation=settings, selected_indices=selected,
                    model_provenance=json.loads((args.model / "download-provenance.json").read_text()),
                    model_config_sha256=sha(args.model / "config.json"), input_sha256=sha(args.input),
                    adapter_sha256=sha(Path(__file__)), results_sha256=sha(args.output / "results.jsonl"),
                    torch=torch.__version__, transformers=transformers.__version__, dtype="bfloat16",
                    attention="sdpa", physical_gpu=0, device=torch.cuda.get_device_name(0),
                    wall_seconds=seconds, amortized_seconds_per_row=seconds / len(selected),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    scope="Teacher pilot only; source weights unchanged; no training or bulk labels")
    (args.output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(json.dumps(dict(output=str(args.output), rows=len(selected), seconds=seconds)), flush=True)


if __name__ == "__main__":
    main()
