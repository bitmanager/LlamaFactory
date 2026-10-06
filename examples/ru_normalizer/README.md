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

The Gemini source has another pronunciation-policy risk: all 785 current training
rows containing `Power` use `Павр` (also 40 validation and 72 separate-evaluation
rows). [Cambridge's pronunciation](https://dictionary.cambridge.org/pronunciation/english/power)
contains /aʊ/, and [Microsoft's Russian localization](https://learn.microsoft.com/ru-ru/security-exposure-management/whats-new)
uses `Пауэр` for Power Automate. These support investigating the current target;
they do not measure our TTS audio or establish every product-name convention.
The earlier 869-mapping acronym audit did not include this Gemini family.
Matching `Павр` references is therefore not certified pronunciation quality.
Source provenance and audio need checking before any mass relabeling.

The follow-up source join found `Павр` in the original pre-TTS replacement
dictionary, not an ASR transcript. Three complete joins reproduce the submitted
TTS text from written input and the dictionary, then match the parquet stress
labels and current targets. Their FLAC records exist, but have not been audited
by listening here. In this source, `voiced` means **submitted TTS text**, not an
independently verified waveform transcript. Earlier manifests calling it “actual
voiced text” remain historical artifacts; the exporter now states the narrower
provenance. No training targets or frozen panels were changed by this audit.

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

The reviewed RUSLAN addition contains **13,719 train / 884 validation** rows.
Stock RUAccent checks its own output length; consensus additionally checks the
retained original target against the same tokenizer and 192-token limit. Nine
otherwise agreeing rows exceed this limit and are quarantined. One reviewed
source-spelling ambiguity is also excluded by exact ID and original text, without
rewriting labels. All 22,200 rows are accounted for across accepted data and the
two quarantine stages. A 130-context manual sample found no confirmed ordinary-word
stress errors among retained examples; proper-name pronunciation remains uncertain.
The hash-bound review permits limited research training, not gold-label claims.
Appending this addition retains old data and frozen evaluation, yielding
**70,705 train / 3,079 validation**. `eval_freeze_text.py --uniform --data ...
--features ...` selects 32 new held-out rows by SHA256 without changing native
decoding or existing diagnostic panels.

V7 completed one epoch of this mixture (2,210 steps, batch 32, LR 1e-5) in
917.7 seconds. Validation CE fell from 0.15907 at step 400 to 0.10852 at the end.
All seven native-generation panels retain their original rows and references:

| Panel | V6 | V7 |
|---|---:|---:|
| Original 64, content WER | 5.83% | 4.32% |
| RU-abbreviation / Latin strata, WER | 8.86% / 9.90% | 3.80% / 9.65% |
| New literary 32, content WER | 25.43% | 16.76% |
| New literary 32, exact content / raw stressed text | 7/32 / 2/32 | 11/32 / 8/32 |
| Canonical numbers, exact | 63/64 | 64/64 |
| Nominative clock / currency / oblique clock, content exact | 32/32 each | 32/32 each |
| Mixed-context numeric, WER / exact | 35.04% / 1/32 | 38.18% / 0/32 |

Stratum WER measures whole sentences, not isolated acronym pronunciation.
Raw-reference matches such as unexpanded `НДС`, and the `Павр` convention above,
cannot certify spoken-form correctness. Optional monosyllabic stress marks also
affect raw exact match; inspect word/stress errors separately. V7 improves general
copying but still corrupts unfamiliar words and fails broad numeric normalization.
Its final/end2210 weights SHA256 is
`fc1f3c916e3a6938e2a107ba5ae3291c54ea29015cbbd7a532dfdfd82bbbd4a2`.

V8 completed two additional epochs (4,420 steps, 27.8 minutes) with final
validation CE 0.07717. The expanded literary panel uses the same 256 held-out
rows for V7, V8 step 3200 and V8 final:

| Panel | V7 final | V8 step 3200 | V8 final |
|---|---:|---:|---:|
| Literary 256, content WER | 16.63% | 9.93% | 9.51% |
| Literary 256, exact content | 81 | 123 | 123 |
| Literary 256, raw stressed exact | 28 | 69 | 64 |
| Original 64, content WER | 4.32% | 5.07% | 4.66% |
| Original 64, exact content | 49 | 49 | 48 |
| Mixed-context numeric, exact | 0/32 | 0/32 | 0/32 |

Step 3200 is selected for research continuation because stress and complete-text
preservation are co-primary. Full-context delta review found five real stress-only
regressions in the final snapshot, including `+умные → умн+ые` and
`ст+оящее → сто+ящее` in “написать что-то стоящее”, plus three copy failures among
eight lost raw matches. These are not optional monosyllabic marks. This selection
is a task-priority judgment, not statistical proof; final has lower aggregate WER
and both snapshots remain available. Step 3200 preserves all 160 bounded-number,
clock and currency texts. Its sole currency raw mismatch is optional `две/дв+е`.
Selected weights SHA256:
`51035790933d1453a3353a9c7546abce6b18bcdd72cc9ae04262de50e97594a3`.

A separate, limited Google DATE pilot now admits **55 train / 25 validation**
full contexts after original-target review and stock RUAccent stress review.
Original normalization labels are retained, not regenerated. Train and validation
come from different source shards and exclude existing protected examples. Rejected
contexts and uncertain names remain quarantined; for example, RUAccent's `в+ёсны`
in “не пережило весны 1942 года” was rejected, not silently corrected. These are
reviewed automatic labels, not human gold or general date coverage. On the 25 new
held-out contexts, selected V8 has 48.40% content WER and **0/25 exact**: date values,
inflection and copying still fail.

The existing reviewed-data/cache assembler appends the 80 accepted rows, yielding
**70,760 train / 3,104 validation**, with frozen evaluation unchanged. All new cached
features/targets and all old hardlinks were verified. V9 started from selected V8
for one bounded additional epoch, batch 32, LR 5e-6, with a fresh optimizer/cosine
schedule. This is model-only continuation, not an exact optimizer resume. Its
purpose is to measure the small DATE addition while tracking old content/stress
panels; 55 examples do not establish broad numeric generalization.

V9 completed 2,212 steps in 14.2 minutes, final validation CE 0.08229.
All eight native-generation checks completed. Original-panel WER worsened from
5.07% to 5.55% (content exact 49 to 47 of 64); literary WER improved slightly
from 9.93% to 9.64%, but raw stressed exact fell from 69 to 66 of 256.
Independent full-context review found four genuine stress regressions among
lost raw matches, plus copying errors and optional monosyllabic differences.
DATE WER improved from 48.40% to 43.09%, still **0/25 complete exact outputs**.
All 160 bounded-number/clock/currency outputs now match including stress.
**Selected V8 remains the main research reference; V9 is not promoted.**
V9 final weights SHA256:
`5185dec976d4d680d80b22beae9e32cbef3362ee09a5e2c393f2a407b9b9e96b`.

A separate training-fit diagnostic used only the 55 reviewed DATE train rows,
starting from V9, with the unchanged Trainer: batch 8, LR 3e-5, 20 epochs/140
steps, 52.4 seconds. On the same ten **training** probes, correct date fragments
rose from 0/10 to 10/10; complete stressed text matched in 9/10. On the fixed
held-out DATE panel, only 2/25 complete outputs matched (8/25 correct date
fragments after full-context review), WER 34.57%. Original-panel WER worsened
to 6.85%. This establishes trainability, not generalization; both exposure and
LR changed, so it does not isolate a sampling effect. The diagnostic checkpoint
is retained separately and is not the main model.

The next bounded mixture contains 2,000 unique train rows: 100 reviewed DATE,
400 Latin, 400 Russian-acronym, 400 other numeric, and 700 literary replay.
The additional DATE source has 45 accepted/12 quarantined stressed contexts;
uncertain proper names, missing required marks and one confirmed contextual
`Корпус+а` error remain outside training. Original silver targets are preserved.
Replay is selected by a fixed source-ID hash, independently of generated output;
held-out IDs, normalized full texts and source groups are excluded. The complete
3,104-row validation and existing frozen panels are unchanged. Preparation uses
the existing feature exporter and Trainer, with no new sampler or loss code.
The mixture is an experiment in retention, not a claim of broad date coverage.

V10 completed 20 short epochs / 1,260 steps in 8.3 minutes, batch 32, LR 1e-5,
starting from selected V8 with a fresh optimizer. Final validation CE is 0.08949.
DATE WER improved to 28.19%, but only 2/25 full outputs matched; manual review
found 11/25 correct date fragments amid frequent surrounding-text corruption.
Original-panel content exact fell 49→48/64; literary content exact 123→113 and
raw stressed exact 69→56/256. Independent review of every raw-match transition
found 25 losses: 12 copy errors, 12 stress errors and one optional monosyllabic
mark. Acronym spans remain 50/53, with a duplicate ООО. All 160 bounded numeric
contents are retained. **V10 is not promoted; selected V8 remains unchanged.**
Final weights SHA256:
`98ff5bf2c941764baee91a2db9631638367bedb0ae7080b2b6030d7efa5df3c2`.

The next unique corpus restores all old rows and adds 64 further reviewed DATE
contexts, totaling **70,869 train / 3,104 validation**. Of 180 new source contexts,
78 passed normalization review and 64 passed subsequent stress review; original
targets are retained, with quarantine for missing marks/uncertain proper names.
For example, `зем+ель` was retained after dictionary verification; isolated-word
checks must not reject valid stress transfer in `н+е было`.

For full-corpus replay, optional `--repeat_selection path.json` uses **stock
PyTorch Subset + ConcatDataset**, preserving every base row and repeating only
specified train positions. No new sampler, loss or generation code is introduced.
The JSON contains `metadata_sha256`, ordered `indices`/`source_ids`, and integer
`factor` (total exposures including the base copy). Hash, positions, IDs and
cache length must match; validation is never repeated. Original IDs/targets are
preserved and intentional repetitions are recorded in `repeat_selection.json`
alongside the run. Exact resume requires the original run directory and the same
selection; use model-only `warm_start` for a different run.
With all 164 reviewed DATE rows and factor 20, one epoch has **73,985 positions**:
70,705 other examples once and 164 DATE examples 20 times. At batch 32 this is
2,313 optimizer steps (accumulation 1, no drop-last). This restores full replay
coverage without materializing duplicate examples or inventing new source IDs.
Keep `num_train_epochs=1`; carrying over V10's 20 epochs would multiply the DATE
exposure again. Repetition does not establish unseen-date generalization.
A two-step GPU smoke run verified the stock Trainer with this 73,985-position
view, finite loss/gradients and unchanged 3,104-row evaluation. A cross-run exact
resume was explicitly rejected before creating its output directory. These are
integration checks, not evidence of pronunciation improvement.

V11 completed the full-replay epoch: 2,313 steps in 887 seconds, validation CE
0.07547. Against V8, original-panel content WER improves 5.07%→4.25%, literary
WER 9.93%→8.31%, DATE WER 48.40%→26.33%, and mixed numeric WER
38.46%→33.90%. All 160 bounded numeric outputs match including stress.
DATE still has only 2/25 complete exact outputs and 12/25 correct date fragments;
correct dates frequently coexist with surrounding-text corruption. Literary
raw stressed exact falls 69→65/256, with both real stress regressions and
optional monosyllabic-mark differences. Full-context independent review retains
these failures; lower WER does not establish uniformly better pronunciation.

A broader SHA-selected 512-row panel from the same 3,104-row validation,
evaluated with identical IDs and references for both models, gives V8→V11
WER 4.63%→4.07%, content exact 400→402, raw stressed exact 307→309 and invalid
stress 22→17. This is additional development evidence, not an untouched test.
V11 final is selected only as the next research warm-start; V8 and V11 step1200
remain comparison snapshots. V11 final SHA256:
`57529eef3590d2cf62be7c357c6eb4ffce7d86ab69f6a95c2240e35ee08dc580`.

V12 adds 158 DATE contexts after full source/stress review of 184 candidates;
26 are quarantined for source normalization, stress, coverage or uncertain
names. It retains all old rows: **71,027 unique train / 3,104 validation**.
The 322 DATE positions repeat 10 times, yielding 73,925 epoch positions and
2,311 batch-32 steps. This approximately preserves V11's 4.4% DATE exposure
while increasing diversity. One epoch uses a fresh optimizer at LR 5e-6 from
V11 final; architecture, loss and Trainer are unchanged. Data and LR both change,
so this is continued training, not a single-variable ablation. Independent
integrity review checks all 74,131 feature hardlinks, every new tensor, target
identity, tokenizer hashes, repeat positions and heldout exclusions.

V12 completed 2,311 steps in 886 seconds, validation CE 0.07208. All nine
native-generation panels completed without token-limit failures. Compared with
V11, the identical 512-row validation sample improves content WER 4.07%→3.87%,
content exact 402→410 and raw stressed exact 309→315. Literary WER improves
8.31%→7.83%, raw stressed exact 65→74/256. DATE WER improves 26.33%→18.88%,
with 7/25 complete content matches and 5/25 raw stressed matches. Full-context
review credits 16/25 numeric date fragments, sometimes amid corrupted words.
All 160 bounded numeric outputs still match including stress.

These gains are not uniform: original64 WER worsens 4.25%→4.80%, and mixed
numeric32 worsens 33.90%→38.46%, with no complete matches. Some outputs omit
sentence endings or duplicate digits even when aggregate WER improves. Keep V11
and V8 for comparisons; this is research continuation, not a general-purpose
normalizer release. V12 final SHA256:
`1165d8139413eb0c2b686aab89739ded75e2595009effcfe0eaa79cfe3247da0`.

A full-context source audit checked 80 abbreviation contexts and all 87 targets
containing uppercase abbreviations. It identified 17 incomplete spoken-letter
plans and three adjacent label defects; the existing quarantine adapter excludes
exactly these 20 IDs from a new V13 dataset, without rewriting targets or changing
validation. V12 remains immutable. V13 contains **71,007 train / 3,104 validation**;
the same 322 DATE IDs at factor 10 give 73,905 positions and 2,310 batch-32 steps.
Independent checks cover all 74,111 retained hardlinks, source/target byte order,
the 20 exclusions, unchanged heldouts, tokenizer and fresh repeat-selection hashes.
These are data-integrity checks, not proof that every remaining label is correct.

A separate review of 62 train contexts for observed stress failures found no
confirmed errors on the queried words. Several failing word forms are absent
from training. Model errors therefore cannot simply be repaired by relabeling
these correct source examples; coverage and generalization remain open problems.

The newly located original SOVA RUSLAN stress annotations were verified against
their public archive, but do not justify replacing current targets wholesale.
Most texts overlap existing data or previous quarantine, and full-context
inspection found annotation errors. These sources remain separate candidates;
neither old quarantine nor frozen evaluation is overwritten.

`eval_freeze_text.py` uses original greedy inference and JiWER on a fixed panel:
16 Russian-abbreviation, 16 Latin, 16 long and 16 short held-out examples, chosen
by source-ID hash before generation. `--numeric` selects the separate 32-example
protected numeric probe. `--weights` accepts immutable checkpoint snapshots.
It saves every prediction, content WER/CER, exact output without stress markers,
invalid stress and token-limit flags. These balanced panels are diagnostic, not
population estimates; their automatic references are not human gold.
For a larger heldout check, `--uniform --uniform-size 256` retains the same
source-ID hash ordering and original inference/metrics. The default remains 32;
existing diagnostic panels are unchanged. Use identical selected rows for each
checkpoint comparison, and keep these rows out of training.

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
