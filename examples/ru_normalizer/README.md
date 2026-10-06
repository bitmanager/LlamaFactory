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

Export only new training rows with `export_qwen_features.py --splits train`.
`assemble_features.py` verifies revisions, dimensions, tokenizer hashes, source
hashes, exact data concatenation and file indexes before hardlinking old/new
features. It writes `COMPLETE.json` last; training verifies this completion hash.
This avoids rerunning frozen Qwen on unchanged rows.

`prepare_numeric.py` uses pinned `num2words==0.5.14` for a deliberately narrow
domain: nonnegative nominative integers below one million and positive decimal
strings with 1–3 fractional places. Decimal arithmetic preserves values and scale;
numeric values are disjoint across splits. It does not implement a normalizer.
Stock RUAccent adds stress, followed by an independent review. The reviewed
12,000 train / 256 validation pairs extend v3 to **50,196 train / 1,749 validation**
in v4. Dates, money, negative numbers and context-dependent inflection are outside
this added data's scope; performance there must be measured separately.

`assemble_features.py --merge-data` retains all original rows and requires a GO
review bound to exact train/validation hashes before admitting explicitly staged
data. Feature assembly then reuses existing caches and appends only the new
exports. The original staging manifest stays unchanged in provenance.

`prepare_clock.py` prepares a separate, still-staged HH:MM candidate using
`num2words` and `pymorphy3.make_agree_with_number`, with neutral nominative
templates. It contains no handwritten declension tables. It is not part of v4.
Its source data and stress labels require review before training.

The optional `--context-from /path/to/clock-v1/source` (or its `staging` directory)
adds genitive `до/после` and dative `к` contexts using only stock `num2words`
case selection and `pymorphy3.inflect(...).make_agree_with_number(...)`.
It verifies the base manifest hashes, IDs and groups, and preserves each clock
value's train/validation split across both cases. `--holdout` remains required;
protected values and full texts exclude training candidates, while existing
clock validation values remain validation. Dative zero hours or minutes are
quarantined because the stock libraries produce unsupported forms (83 possible
HH:MM values before base exclusions). This mode writes new unstressed source
data only; its sentences need fresh stress QA. The default nominative mode is
unchanged.

`prepare_currency.py` uses the same pinned library's `currency="RUB"` path with
exact `Decimal` arguments and two fractional places. Integer arguments would
mean kopecks in that API, and longer fractions would be rounded; neither is used.
Neutral nominative templates keep case selection explicit. Source values and
texts are excluded against supplied train/holdout files. Its candidate starts as
staging too; after RUAccent, 39 rows with missing polysyllabic stress marks were
quarantined without relabeling (3,962 train / 127 validation retained).

`quarantine_features.py` excludes only explicitly audited training IDs whose
written text and target match the audit and cached feature exactly. Retained
features are hardlinked in the new order; validation and frozen evaluation stay
unchanged. Unknown, duplicate or held-out IDs fail before output creation. A new
manifest and completion marker bind the filtered data and cache to their source
hashes. Full inspection of all 869 abbreviation mappings and 8,023 target spans
found 15 `SEM → сём` and 14 `ТЕР → тёр` errors from automatic ё restoration,
plus one confirmed `смес+и` stress error. These 30 rows are excluded from the
reviewed next mix: **55,341 train / 2,004 validation**, including clock/currency
data above. Targets are never silently rewritten to make checks pass.

A subsequent full-context audit found a task-format conflict inherited from the
original stress-only numeric corpus: 124 targets retain raw `СМС/смс`, four retain
`НДС`, and 33 retain other unambiguous letter abbreviations. These written forms
are not spelling errors, but do not specify the spoken expansion required here.
The same quarantine adapter prepared a separate v6 candidate with **55,180 train /
2,004 validation**, excluding exactly those 161 audited IDs without relabeling.
It preserves the earlier 30 exclusions in provenance and all held-out data.
The completed v5 dataset is unchanged. Ambiguous brand/time shorthand and
word-pronounced acronyms are not automatically excluded. Three old diagnostic
references also retain raw `СМС/НДС`: exact match there is a content-copy check,
not evidence of correct spoken expansion. Keep this limitation when comparing
checkpoints; do not silently rewrite the panel to improve metrics.

Letter-preservation checks do not certify every pronunciation convention. For
example, all 326 current training examples containing written `МФК` use
`эм фэ ка`. [General alphabet rules](https://orfo.ruslang.ru/docs/biblio/russkoe_pravopisanie_s_kommentariiami_1.pdf)
(printed pages 19–21) suggest `эм эф ка`, while established
abbreviation-specific exceptions to letter names exist. The audit did not find
an МФК-specific normative source. This remains an explicit pronunciation-policy
limitation; these source-consistent targets have not been silently rewritten or
declared universally incorrect.

V5 completed three epochs (5,190 steps, 39.4 minutes). Its end checkpoint gave
6.58% content WER on the original 64-example panel, 64/64 exact canonical-number
outputs, 32/32 clock outputs and 32/32 currency outputs ignoring stress marks.
However, the separate mixed-context numeric panel remained **0/32 exact**, with
37.04% WER. Neutral-template success does not establish arbitrary-text number
normalization. The end checkpoint differs from the CE-best checkpoint on only
one of 224 outputs; selecting it is not a statistically significant comparison.

The next reviewed mixture adds 1,806 training and 191 validation oblique-clock
pairs to the v6 data, for **56,986 / 2,195** rows. After library-based inflection,
RUAccent produced `Прих+одите` in 616 invitation-template rows where the intended
imperative requires `Приход+ите`. These ambiguous rows are quarantined unchanged,
not globally relabeled. The earlier 69 dative-zero exclusions also remain in
provenance. Existing clock values retain their original split; the new held-out
32-example diagnostic panel is selected by the existing SHA256 rule (23 genitive,
9 dative). This addition covers bounded clock constructions, not money, dates,
phone numbers or arbitrary grammatical contexts.

V6 continued from V5 on this mixture for one epoch (1,781 steps), batch 32,
LR 1e-5, fresh cosine schedule with 3% warmup. It completed in 703 seconds with
validation CE 0.03811. Native generation on unchanged diagnostic panels gives:

| Panel | V5 | V6 |
|---|---:|---:|
| Original 64, content WER | 6.58% | 5.83% |
| RU-abbreviation stratum, WER | 9.37% | 8.86% |
| Latin stratum, WER | 13.61% | 9.90% |
| Canonical numbers, exact | 64/64 | 63/64 |
| Nominative clock / currency, content exact | 32/32 each | 32/32 each |
| Oblique clock, exact including stress | 0/32 | 32/32 |
| Mixed-context numeric, WER / exact | 37.04% / 0/32 | 35.04% / 1/32 |

V6 is selected for research continuation, not a uniform quality win: old64 exact
content falls from 47 to 44 examples, and `69671,060` is incorrectly spoken as
`69671,006`. Incorrect stress and acronym substitutions remain. Both checkpoints
are retained. V6 final/end1781 SHA256:
`d43f9f0ac522e3a0d095e401c632019a074a465d1ea9b8534e562908f3a69266`.

`prepare_ruslan.py` is a format/consensus adapter for the pinned
`stilletto/ruslan-stressed` CSV, not a new accent model. It converts combining
acute accents to the existing `+` format, preserves the original target, and
excludes existing train/held-out aliases, duplicates, raw abbreviations and
explicitly audited source errors. It then checks stock `accent_pairs.py` output
against the original accents. Agreement ignores optional monosyllabic accents
but preserves the distinction between `е` and `ё`. Disagreements are quarantined;
labels are never replaced with the second model's guesses.

The source export contains 19,846 train and 1,283 validation candidates out of
22,200 rows; 1,071 are quarantined before the second accent check. Splits group
contiguous blocks of 100 source IDs, not independent books or speakers. All 18
previously audited bad IDs are excluded. The first 512-row deterministic pilot
retained 356 matching labels; agreement alone is not human verification. This
candidate remains in staging until independent quality review bound to its exact
file hashes. The original corpus's CC-BY-NC-SA-4.0 terms are preserved alongside
the derivative card's conflicting CC-BY-4.0 claim. It adds literary stress/copy
examples, not new number or abbreviation expansions.

`eval_freeze_text.py` uses original greedy inference and JiWER on a fixed panel:
16 Russian-abbreviation, 16 Latin, 16 long and 16 short held-out examples, chosen
by source-ID hash before generation. `--numeric` selects the separate 32-example
protected numeric probe. `--weights` accepts immutable checkpoint snapshots.
It saves every prediction, content WER/CER, exact output without stress markers,
invalid stress and token-limit flags. These balanced panels are diagnostic, not
population estimates; their automatic references are not human gold.

Using the same corrected runtime, v1-final → v2-step1800 reduced panel WER from
87.05% to 44.14% (CER 65.28% → 32.73%). Numeric-probe WER also fell, but manual
review found **0/32 complete numeric fragments preserved**: improvement was in
surrounding words. This checkpoint does not pass numeric normalization. Both
RUNorm-big and the frozen Qwen teacher pilots also produced number changes, so
their generated labels were not admitted into training.

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
