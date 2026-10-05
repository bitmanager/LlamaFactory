# Original Freeze-Omni decoder with Russian text output

This replaces the stopped custom embedding-prefix experiment. The frozen source
is the same Qwen3-4B-Instruct-2507 and tokenizer used by the ASR/agent branch.

## Reused code

`third_party/Freeze-Omni` uses the author implementation at
`163a24880e533b2a07038fb8dcfe02dbbb8457e6` plus one isolated padding bugfix
in the Bitmanager fork: target RoPE positions follow the item's true text length,
not the batch's maximum padded length. Masks, cache construction and inference
are unchanged. This makes batched training match individual inference.
`models/decoder/decoder.py:LLM2TTSCodecAR` supplies the noncausal prefix layers,
per-layer DynamicCache K/V, masks, text preprocessing, AR decoder, teacher-forced
forward, summed CE and native `infer()` with EOS termination. Validation uses
its top-k=1 mode. Input/output lengths are independent.

Like upstream inference, this uses both original text embeddings and contextual
hidden states. Two upstream `LinearAdapter`s map Qwen's 2560 dimensions to 896.
The 1024-token speech vocabulary becomes the agent's 151670-token text vocabulary:
only embedding/output weights are newly initialized. Other decoder weights load
from the release with an explicit mismatch allowlist. All small-decoder and
adapter parameters train (439,416,570); source Qwen is frozen. This is the approved
text-output adaptation, **not** the authors' prefix-only speech alignment stage.

The released config is preserved as `freeze_decoder.json`: 4 decoder layers,
4 prefix layers, 2 text-preprocessing layers. The local wrapper normalizes the
upstream summed CE by target tokens plus EOS. HF Trainer handles optimization,
evaluation, checkpoints and resume. A complete author training driver/data
pipeline is not published. The positional-ID fix is the only upstream model change.

## Features and environments

Original Freeze-Omni requires Transformers 4.45.2, while our Qwen environment uses
5.3.0. `export_qwen_features.py` runs frozen Qwen once in the modern environment
and caches BF16 features. Training then uses the original decoder API without
rerunning Qwen or changing the ASR environment.

Cache: `x` = original text embeddings, `x_prefix` = contextual hidden, `y` = target
spoken-text tokens, plus original/reference text and source ID. The source never
receives the normalized target. Hidden states are **pre-token, final-normalized**,
matching `_generate_one_step`: the state that predicts the sampled token. Real
history is preserved when supplied; this pilot has no history and invents none.

Source revision: `Qwen/Qwen3-4B-Instruct-2507` at
`cdbee75f17c01a7cc42f958dc650907174af0554`. Copied base files were SHA256-verified.
Use the exact ASR tokenizer (151670 entries after vocabulary resizing).

Keep the modern environment intact. Install a separate decoder dependency overlay:

```bash
git submodule update --init third_party/Freeze-Omni
python -m pip install --target "$RUN_ROOT/freeze-omni-deps" --no-deps \
  transformers==4.45.2 tokenizers==0.20.3 huggingface-hub==0.36.2 peft==0.13.2
```

Tested base: Python 3.12, Torch 2.9.1+cu130 for Blackwell, NumPy, PyArrow,
Accelerate and TensorBoard. Download `VITA-MLLM/Freeze-Omni/checkpoints/decoder/final.pt`
to `models/freeze-omni/final.pt` under the run root. Verified SHA256:
`2911cc3dc0cd8ad94756b45fda0a8f78f3d6fb92b5f5df8ad6b9961fd4f02228`.

The launcher sets `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` for the old Trainer's
own RNG/optimizer checkpoint restore under Torch 2.9. Resume only trusted local
runs. Feature/pretrained loads explicitly use `weights_only=True` and are unaffected.

```bash
CUDA_VISIBLE_DEVICES=GPU_UUID "$RUN_ROOT/venv/bin/python" \
  examples/ru_normalizer/export_qwen_features.py --root "$RUN_ROOT" \
  --data "$RUN_ROOT/data/hidden-plan-v1" --output "$RUN_ROOT/data/freeze-features-v1"
CUDA_VISIBLE_DEVICES=GPU_UUID bash examples/ru_normalizer/train_freeze_text.sh "$RUN_ROOT"
```

Training requires cache `metadata.json`, written only after successful export.
Defaults: BF16 autocast, batch 16/accumulation 2, one epoch, LR 1e-4, cosine and
3% warmup. Resume the same ordered dataset with `--resume /path/to/checkpoint-N`.
For a new data mixture, use `--warm_start /path/to/checkpoint-N` and a fresh output
directory: this strictly loads model weights but starts a new optimizer/schedule,
without skipping examples based on the old dataset's step count. The two modes
are mutually exclusive. Checkpoints contain the small decoder/adapters, not source
Qwen. Every 100 steps, validation CE and **target-free** generations are saved
(`examples-N.json`), including held-out Russian abbreviations and Latin words.

## Data and checks

`prepare_local.py` exports existing labels, records provenance, rejects conflicts
and malformed stress, and excludes held-out aliases before template/scenario
splitting. Pilot: 30,173 train, 1,493 validation; separate existing evaluation:
2,302. Most training pairs are stress-only, with automatic stress rather than
human gold. Number expansion is underrepresented; template grouping does not
prove original speaker/recording independence. Audio/TTS quality is not tested.

`correct_labels.py` applies a reviewed, source-ID-scoped audit of stress mistakes.
It preserves data order and counts, verifies source hashes, and changes only cached
target tokens/reference text. Frozen Qwen features are reused; v1 remains immutable.
The v2 correction affected 107 train, 8 validation and 11 separate-evaluation rows.
Ambiguous pronunciation variants are not automatically corrected.

The first 943-step pass finished with validation CE 0.949, but native free
generation still repeated and omitted words. This is not a quality pass. Training
continues with corrected positions/labels; inspect generations alongside CE.

`prepare_abbreviations.py` restores a written acronym only when its recorded
expansion has one reversible span in the source. Targets are retained, with
malformed stress, known faulty acronym families, audited stress errors, conflicts,
duplicates and held-out aliases quarantined. Its optional `--merged-output`
combines accepted examples with an existing train split and hardlinks unchanged
validation/evaluation. The v3 mixture contains 38,196 train rows: v2 plus 8,023
abbreviation pairs. These are reconstructed inputs and automatic labels, not gold.

```bash
python examples/ru_normalizer/prepare_abbreviations.py \
  --root "$RU_DATA_ROOT" --existing-data "$RUN_ROOT/data/hidden-plan-v2" \
  --label-audit "$LABEL_AUDIT" --tokenizer "$RUN_ROOT/models/agent-tokenizer" \
  --output "$RUN_ROOT/data/abbrev-reconstructed-v1" \
  --merged-output "$RUN_ROOT/data/hidden-plan-v3"
```

`prepare_google.py` vendors the original NeMo v1.23.0 Google TSV reader and only
exports train-shard numeric pairs; `accent_pairs.py` calls stock GPU RUAccent with
reviewed dictionary/homograph overrides. The Google corpus has **not** been added
to v3: inspection found numeric meaning/case errors and sports scores labeled as
time. Its accented export is explicitly staging (`training_eligible=false`), not
an approved train set. Stress annotation cannot repair incorrect normalization.

```bash
PYTHONPATH="$RUN_ROOT/freeze-omni-deps" python -m pytest -q \
  examples/ru_normalizer/test_freeze_text.py
```

Tests check gradients through original KV-prefix/decoder/adapters and agreement
between first teacher-forced and native inference logits. Evaluate content
preservation and stress using free generation: the stopped custom-prefix pilot
had validation CE 1.211 at step 100 but generated unrelated responses. Compare
text-only and contextual-prefix paths before claiming a benefit from history.

## License

Freeze-Omni restricts its code/weights to academic, research and education and
excludes commercial/production use. This pilot is research; inclusion in this
framework does not relicense it. Its complete license remains in the submodule.
