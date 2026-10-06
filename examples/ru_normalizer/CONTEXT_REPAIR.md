# Context consistency diagnostic

The live text demo exposed word substitutions, repetition and lost numbers.
All 86,351 V23 training rows lacked dialogue history, whereas live hidden states
include the user's question and prior conversation. The feature alignment itself
remains the existing pre-token, final-normalized convention.

Two data adapters address the distribution mismatch without changing the model,
loss, tokenizer, decoder, feature exporter or HF Trainer:

- `prepare_context.py` selects 2,048 existing reviewed content-preserving training
  pairs and 64 held-out pairs. Each appears with no context, a verbatim-repeat
  question, and a synthetic three-message repeat context. Targets are unchanged.
  These are explicitly synthetic contexts, not genuine conversations. Original
  train/validation groups are retained and the five protected demo cases excluded.
- `generate_context_answers.py` uses stock HF generation with the frozen source
  Qwen and the demo system prompt. `fixtures/context_prompts.jsonl` specifies 48
  two-turn conversations. Actual generated answers and their history are retained.
  The generation limit is 96 new tokens; truncated answers are not admitted.
  Second turns following a truncated first answer are excluded during review.
  This first diagnostic covers ordinary Cyrillic prose, not emoji/numeric repair.

Of 96 candidate turns, 81 finished; 68 contained only the selected plain-text
characters. Full-answer review and incomplete-history exclusions left 42 for
stock GPU RUAccent. All accented outputs were read; one missing-stress example
was excluded. Whole conversation groups split the remaining 41 into 33 train
and 8 validation. These are checked automatic labels, not independent human gold.

Stock `assemble_features.py` concatenates exports: 6,177 training rows and 200
validation rows. Full-cache checks verified IDs, exact source/target text, target
tokens, finite BF16 features, dimensions, held-out exclusion and group separation.
The reviewed real-answer training subset receives factor-10 replay via the
existing Trainer wrapper, giving 6,474 positions per epoch.

V24 is a bounded continuation from the latest V23 final, SHA256
`b53b1a92ce5383bcd09d25e46c22f8220603388c6c031313b7b80f318ea4f4f5`.
Settings: unchanged 439,416,570 trainable parameters, frozen Qwen, BF16, batch 32,
three epochs/609 optimizer steps, LR `3e-6`, cosine schedule, 3% warmup, model-only
warm start. Original training data/caches and previous checkpoints remain intact.

Validation uses stock `eval_freeze_text.py`, upstream greedy generation and JiWER:
200 contextual cases, five protected user-demo cases, the unchanged numeric32
panel, and the old generation64 panel. The user cases in
`fixtures/user_demo_protected.jsonl` are evaluation only. Their references assess
content, not stress; acceptable time/acronym readings need manual review, so a
single-reference WER is not a definitive numeric-semantic score.

V23 baseline: contextual200 WER 8.86%, literal text-exact ignoring stress 44%;
old64 WER 3.56%; numeric32 WER 29.63%; user-demo five WER 47.87%.
On the matched 64 context-control phrases, word-normalized WER is 2.02% without
context, 9.76% with the repeat question and 12.35% with synthetic history. On the
eight actual held-out Qwen answers it is 22.73%. These are diagnostic samples,
not population or audio-ASR metrics. Final-generation results must be compared
before deployment; lower teacher-forced CE alone does not establish success.

Run artifacts and exact launch/integrity records live under
`research/context-repair-v1` in the existing experiment root. The V24 run lives
on dev drive2 as `runs/freeze-text-v24-context-pilot`; dev GPU0 remains reserved
for training, and the demo/evaluation stays on exp GPU3.

## Completed V24 result

All 609 steps completed in 226.6 seconds. Final validation CE on this new
diagnostic set is 0.136669 (not comparable to CE on the old validation mixture).
The immutable final SHA256 is
`d5772fa0eb5b081bac9f8e084df05cceac14f32d36c181d66335af7d20c638e6`.
All four native held-out evaluations completed without runtime errors.

| Panel | V23 WER | V24 WER | Content-exact V23 -> V24 |
|---|---:|---:|---:|
| Contextual200 | 8.86% | 4.72% | 88 -> 148 / 200 |
| Protected user-demo5 | 47.87% | 47.87% | 1 -> 1 / 5 |
| Numeric32 | 29.63% | 32.48% | 1 -> 1 / 32 |
| Original generation64 | 3.56% | 3.50% | 48 -> 49 / 64 |

On matched phrases, word-normalized WER changed from 2.02% to 2.81% without
context, 9.76% to 4.15% with a question, and 12.35% to 4.60% with synthetic
history. On the eight actual held-out Qwen answers, it only changed from 22.73%
to 21.21%, with word-exact improving from one to two answers. Thus most of the
gain is on synthetic repeat contexts; it does not establish reliable live
normalization. Invalid stress outputs in contextual200 increased from three to
five. The numeric regression and unchanged user failures block demo promotion.

An additional inference-only diagnostic recomputed both problematic long answers
with/without emoji and with/without history. Removing emoji did not repair them.
Calling the upstream decoder with `prefix=None` also produced corrupt text.
Neither ablation was adopted as a runtime workaround. Demo remains on V22;
V24 remains the latest training checkpoint for any subsequent continuation.

Conclusion: context coverage matters, but this small repair does not solve
generalization to ordinary free-form Qwen answers. Bigger parameters were not
tested, and this result is not evidence that increasing model size will fix it.
The next data mixture needs broader real-answer coverage and retention of
numeric/abbreviation examples; repeating this narrow pilot alone is unsupported.

## V25 retention and V26 expansion

V25 continued from the immutable V24 final on the previous 86,351 reviewed rows
plus the 33 reviewed real Qwen answers. The existing assembler and trainer were
unchanged. One epoch completed in 1,135 seconds, with 3,049 optimizer steps,
training CE 0.03005 and validation CE 0.07372 on 3,118 rows. Generation WER:
context200 4.61%, protected demo5 45.74%, numeric32 29.34%, original64 4.73%.
Numeric retention improved over V24, but original64 regressed and the long demo
answers remain corrupt. This does not qualify for demo promotion. The demo was
stopped at the user's request; its exp GPU3 still has an unrelated ASR process.

V26 uses the existing generator with the 64 two-turn conversations in
fixtures/context_prompts_v26.jsonl. Of 128 candidates, 115 completed; full-answer
review retained 73 for the existing GPU RUAccent adapter. Reviewing every
accented answer excluded four more (including incorrect узнает -> узнаёт).
The 69 retained rows split by whole conversation into 59 train and 10 validation.
Only decorative music/smile emoji are omitted from spoken targets; original
written input and actual generation history are preserved. Labels are reviewed
automatic annotations, not independent human gold. Exact normalized overlaps
with existing data and protected panels were checked before assembly.

The resulting cache has 86,443 training rows and 3,128 validation rows. All 69
new feature records were checked for IDs, source/reference text, target token
IDs, finite BF16 features and dimensions. No model, trainer, exporter or loss
changes are needed. Continuation uses the latest V25 final, SHA256
7e90bf32c33fdadde39cd24d3b2075e91f78030dee8dd2be58bc5b68dfdadccf.
Research manifests and logs are under research/v26-context-expansion.

The launched V26 run uses one epoch, batch 32, BF16 and LR 1e-6. Existing stock
repeat selection applies factor 32 to all 92 real contextual training rows and
400 numeric rows selected by SHA256 of source ID from the previous reviewed
numeric replay pool. All 86,443 unique base rows remain included; this yields
101,695 sampled positions and 3,178 optimizer steps. The new held-out ten rows
have V25 baseline WER 14.01%; their labels and split were fixed before fitting.
Final generation checks for new10, context200, demo5, numeric32 and original64
are queued after training. The required JiWER dependency is present in the dev
evaluation environment.

V26 completed in 1,185 seconds. Train CE was 0.02856 and validation CE 0.07447.
All five generation panels completed: new10 WER 11.11% (V25 14.01%),
context200 4.79% (4.61%), demo5 46.81% (45.74%), numeric32 30.20% (29.34%),
original64 5.21% (4.73%). This is a narrow improvement on ten new answers,
with regressions elsewhere, not an overall improvement or a deployment result.

At the user's explicit request to continue, V27 starts from the immutable V26
final (SHA256 4778c2c977618dc371363064728cfd36b5b4ced6ee167ed4d30a24b1a02ab4e9).
It is one bounded additional epoch on the unchanged V26 data/replay mixture,
with LR reduced to 5e-7 and a fresh optimizer/schedule. There are no new rows,
architecture or trainer changes. The same five generation panels are queued
after training; research manifests are under research/v27-context-continued.

## V27 result and V28 contextual expansion

V27 completed 3,178 steps in 1,189.9 seconds. Train CE was 0.02448 and validation
CE 0.07452. Generation WER was new10 13.04%, context200 4.57%, protected demo5
43.62%, numeric32 29.34%, and original64 4.87%. The first panel regressed versus
V26 despite small gains elsewhere. Full inspection of the five demo generations
still found corrupt pizza answers, repetition and incorrect MFTI/time readings.
The demo remains stopped; this checkpoint is not promoted as a quality fix.

V28 adds contextual source prompts from `IlyaGusev/ru_turbo_alpaca`, pinned at
revision `460b1f3312aa21ef774e916e532a9576f7938a0d`. Only source rows labeled
`ok` were considered. A deterministic selection of 256 prompts was expanded to
two turns using the existing generator and the same frozen Qwen/system prompt.
Of 512 candidate answers, 394 completed and 118 truncated answers were rejected.
Followups to incomplete first turns are excluded. Unreviewed raw generations
remain outside training.

Full-answer inspection admitted 140 source pairs to stock GPU RUAccent. Every
accented target was read. Missing stress, incorrect yo/stress and normalized
full-text overlaps quarantined 22 rows, leaving 118 checked automatic labels.
Eight whole conversation groups selected by SHA256 were fixed as held out before
baseline evaluation: 107 new train rows and 11 new validation rows. This batch
covers ordinary Russian prose; it does not claim new numeric/acronym labels.
The source answers, actual histories and system prompt are preserved.

The existing exporter produced 118 cached records. All were checked for source
IDs/text, reference/target token IDs, finite BF16 features, dimensions and paired
sequence lengths. The unchanged assembler retained the prior data and produced
86,550 train rows, 3,139 validation rows and the unchanged 2,302-row frozen eval.
Existing factor-32 replay now covers 199 actual contextual train rows plus the
previous 400 numeric rows, giving 105,119 positions and 3,285 optimizer steps.

V28 uses the latest immutable V27 final, SHA256
`441c38bd5939c0f7f81d2bcb6f40e35fcd71d5306c555f369302d1e4656a830f`.
Settings remain BF16, batch 32, one epoch, LR `1e-6`, model-only warm start with a
fresh optimizer/schedule and 439,416,570 trainable parameters. No architecture,
loss, trainer or exporter changes are introduced for this run.

Native host CUDA initialization hung inside `cuInit` before accent processing.
The existing CUDA container successfully initialized the same reserved dev GPU0;
training runs there with the existing venv and pinned Transformers 4.45.2. There
is no CPU fallback, GPU reset or interference with other jobs. Data generation
used the same existing container on exp GPU3; those preparation jobs finished.
Research artifacts, the feature audit, split hashes and exact container launch
are under `research/v28-alpaca-context`. A pre-training V27 baseline on new11 is
recorded: WER 31.42%, one content-exact answer out of eleven, and one invalid
stress output. Manual inspection found repetition, corrupt openings and dropped
words in the longer answers. This is a different panel from new10 and its WER
must not be compared directly with new10. All five prior generation panels plus
new11 are queued after V28.

At step 600, validation CE is 0.07483. A separate native generation run on the
free dev GPU1 completed new11: WER 31.08% versus baseline 31.42%, with one
content-exact answer in both. All eleven outputs were inspected; long answers
still contain corrupt openings, repetitions and dropped words. This early
checkpoint does not establish a useful quality gain. The evaluation container
exited successfully and released that GPU.

While V28 trains, the unchanged stock generator prepares the next 512 source
prompts/two-turn conversations on exp GPU3. Source indices are disjoint from
V28; compressed and decompressed source hashes are recorded separately. This
next raw batch is marked ineligible for training until generation and target
review finish. Its artifacts are under `research/v29-alpaca-context`; no new
trainer or generation algorithm is introduced.

## V28 final and V29 continuation

V28 finished all 3,285 steps in 1,225.1 seconds. Train CE was 0.02632 and
validation CE 0.074497. The immutable final SHA256 is
`f063fde111b500a8650647ba34880a7a7f1cef4b72c907f35601136dedadb332`.
All six final generation panels completed successfully:

| Panel | V28 WER |
|---|---:|
| Contextual200 | 4.82% |
| Protected user-demo5 | 42.55% |
| Numeric32 | 30.48% |
| Original64 | 4.18% |
| Previous new10 | 14.98% |
| New11 | 34.46% |

New11 regressed from the V27 baseline of 31.42%; its step-1200 result was
36.49%. Every final new11, new10 and protected demo5 generation and the ten
final callback examples were read. Long outputs still corrupt beginnings,
drop words and repeat spans. Small CE changes do not establish a quality gain;
these are text-normalization diagnostics, not audio-ASR metrics. No checkpoint
is promoted to the demo, which remains stopped.

The next stock generation batch completed 1,005 of 1,024 candidate answers;
19 truncated answers are excluded. All 432 plain-text first-turn candidates
were read, admitting 291 source pairs to stock GPU RUAccent. All 291 accented
targets were read, quarantining 56 for missing/wrong stress, incorrect yo or
source defects. The remaining 235 checked automatic labels split by whole
conversation into 215 train and 20 validation. They are not independent human
gold. Source questions, actual Qwen answers/history and the system prompt are
preserved; no new numeric/acronym labels are claimed for this batch.

All 235 exported feature records passed source/reference/target-token checks,
finite BF16 and dimension/length checks. Stock assembly produces 86,765 unique
train rows, 3,159 validation rows and the unchanged 2,302-row frozen evaluation.
Existing replay factor 64 covers 414 actual contextual rows and 400 numeric
rows, giving 138,047 sampled positions and 4,314 optimizer steps. This increases
the contextual contribution while retaining all old unique training rows.

V29 is launched on the same dev GPU0 from the immutable latest V28 final,
with BF16, batch 32, one epoch, LR `1e-6`, fresh optimizer/schedule and unchanged
439,416,570 trainable parameters. The existing trainer, sampler, decoder,
loss and exporters are unchanged. New20 was measured before training: WER
18.43%, four content-exact answers out of twenty, three invalid-stress outputs.
All six old panels plus new20 are queued after training. Exact launch and
integrity receipts are under `research/v29-alpaca-context`.

Parallel preparation read all 228 eligible second-turn answer bodies and
quarantined 26 defective sources. Each previous assistant message was verified
against its accepted completed first turn. The remaining 202 are in stock GPU
RUAccent staging on dev GPU1, not yet eligible for training. Admission still
requires full accented-target review, overlap checks and the inherited whole
conversation split; held-out first-turn groups must never enter train through
their followups.

Every one of those 202 accented followups was subsequently read. Sixteen were
quarantined for wrong/missing stress, incorrect yo or normalized full-text
overlap. The remaining 168 train and 18 validation rows inherit the existing
whole-conversation split. Their stock feature export passed all 186 record
checks for IDs/text, target token IDs, finite BF16 features and dimensions.
Stock assembly completed the next cache with 86,933 train and 3,177 validation
rows; the frozen eval remains unchanged. This next continuation is prepared,
not launched, and must warm-start the latest V29 final after its final review.

At V29 step 1200, new20 WER is 18.16% versus 18.43% before training, with six
versus four content-exact answers. On the previous new11 panel it is 26.69%
versus V28 final 34.46%. All 31 generated outputs were read. Some openings and
repeated spans improve, but long responses remain corrupt and repetitive;
this is not evidence of reliable live normalization. Callback examples at
steps 600 and 2400 also retain the long TBC/PayMe repetition failure.

A read-only diagnostic loaded the real step-1200 weights and contextual
features, comparing upstream greedy incremental logits against parallel
teacher-forced logits on the identical generated continuation in BF16. Across
three examples and 149 positions, argmax mismatches were zero. Absolute logits
show small numerical differences (maximum 0.125--0.1875). This narrowly checks
these inference paths; it does not prove all padding/cache cases correct or
exclude other model/data bugs. No decoder/trainer changes were introduced.

The unchanged generator has also completed the next disjoint 512 source
prompts on exp GPU3: 1,006 answers completed, 18 truncated answers excluded.
All 768 preceding source indices were excluded. Raw output stays ineligible
until review. The first 130 of 433 plain-text first-turn candidates were read;
94 fluent sources were admitted to stock GPU accent staging. Every reviewed
source and the 494-row first-turn snapshot were verified unchanged against
the completed generation. Forty-one whole conversation groups were fixed by
SHA256 as held out before fitting; this applies to both turns and any later
admissions. The other 303 plain first-turn sources and the second-turn answers
still require review. Artifacts live under `research/v30-alpaca-next`.

All 94 accented first-turn targets were read; eleven were quarantined for
missing stress, source agreement or full-text overlap. The 83 remaining rows
split into 79 train and four validation using the pre-fixed group split.
Dictionary verification also corrected a manual-review error: `прост+ого` was
valid but had been incorrectly quarantined in the previous followup batch.
The unchanged row is recovered into the next training data; the old quarantine
and immutable cache are retained with a separate correction receipt, citing
[Gramota's adjective inflections](https://gramota.ru/meta/prostoy_4295.h2).

The resulting additional export has 80 train and four validation rows, all
84 feature records checked. The next assembled cache now contains 87,013 train
and 3,181 validation rows: 248 and 22 additions relative to V29. Existing
factor-64 replay covers 662 actual contextual rows plus 400 numeric rows,
yielding 153,919 positions and 4,810 steps. Its prepared plan remains unlaunched;
V29 final/manual review and latest-checkpoint warm start are still required.

At V29 step 2400, new20 WER is 18.43%, matching baseline, and new11 is 28.72%:
better than V28 final, but worse than step 1200. All 31 predictions were read
against the unchanged source/reference panels. Long responses still repeat
and lose content. The generation result remains insufficient for deployment.

## V29 final and V30 continuation

V29 completed 4,314 steps in 1,643.9 seconds. Train CE was 0.03464 and final
validation CE 0.076173. All seven native generation panels completed successfully:

| Panel | V28 WER | V29 WER |
|---|---:|---:|
| Contextual200 | 4.82% | 4.32% |
| Protected user-demo5 | 42.55% | 40.43% |
| Numeric32 | 30.48% | 31.62% |
| Original64 | 4.18% | 4.39% |
| Previous new10 | 14.98% | 9.66% |
| Previous new11 | 34.46% | 26.69% |

On new20, WER fell from the pre-training V28 baseline of 18.43% to 15.99%.
All final user-demo5, new20, new11, numeric32 and ten callback outputs were read.
Long pizza, guitar and encyclopaedia answers still lose words and repeat spans;
numeric outputs still change amounts and omit digits. Improvements on some
contextual panels coexist with numeric/original regressions. These are text
normalization metrics, not audio ASR; this is not a reliable demo checkpoint.

The immutable latest V29 final has SHA256
`09bf65eb55219dc91586a416f00f27c9bb39d83c60bfbf7aac0c7b67d04f76da`.
V30 continues from that final, adding the previously audited 248 train and 22
validation rows: 87,013 unique training rows and 3,181 validation rows. Settings
remain BF16, batch 32, one epoch, LR `1e-6`, fresh optimizer/schedule and
439,416,570 trainable parameters. Existing factor-64 replay includes 662 actual
contextual rows and 400 numeric rows: 153,919 positions and 4,810 steps.
No model, loss, trainer or exporter changes were introduced.

The launch runs held-out baselines on all 18 new followup validation rows and
four new first-turn validation rows before fitting, followed by the seven old
panels and those two new panels after training. Exact launch, immutable snapshot
and split/cache integrity receipts are under `research/v30-context-followup`.
Training stays on the same dev GPU0; the demo remains stopped.

Additional preparation read first-turn source candidates 130--279 from the
completed V30 generation. Of these 150, 38 were excluded for source defects;
the remaining 112 completed stock GPU RUAccent on dev GPU1. These staging rows
are not part of the launched V30 data and still require full target review,
duplicate checks and the fixed whole-conversation held-out split.

All 112 accented targets were subsequently read. Ten were quarantined for
wrong contextual castle stress, incorrect sky -> palate yo replacement,
missing lexical stress or an uncertain movie-title pronunciation; labels were
not rewritten. The fixed split yields 92 train and ten validation rows, with
no normalized full-text overlaps or train/held-out group leakage. Every one of
the 102 exported records passed source/reference/target token checks, finite
BF16 checks and dimension/length checks.

The next immutable cache, `freeze-features-v31-context-next`, is prepared with
87,105 unique training rows and 3,191 validation rows; the frozen 2,302-row eval
is retained. Existing replay includes 754 contextual rows and 400 numeric rows,
giving 159,807 positions and 4,994 steps. Its plan under
`research/v31-context-next` is not launched and must use V30's latest final
after its native generation review. Active V30 inputs remain unchanged.

At V30 step 600, CE is 0.077446. A separate stock generation evaluation on
dev GPU1 completed all 42 control cases: followup18 WER 13.85% -> 12.31%,
first4 22.73% -> 29.09%, and prior new20 15.99% -> 15.45%. All 42 sources and
predictions and all ten callback examples were read. The tiny first-turn panel
regresses, while the other panels show modest gains; corruption and repetition
remain. This is an intermediate mixed result, not a quality fix or deployment.

At V30 step 1200, all 42 generated control outputs were read again. Prior new20
WER is 11.92% (V29 final 15.99%), followup18 is 12.31% (baseline 13.85%), and
first4 is 25.45% (baseline 22.73%). Long responses still corrupt openings,
names and words or repeat spans. This remains a mixed intermediate result.

The final 153 V30 plain first-turn sources were read. Forty-two defective
sources were quarantined and 111 admitted to stock GPU RUAccent. Every accented
target was read; a missing lexical stress and another sky -> palate yo error
were excluded. The retained 103 train/six validation rows preserve actual
Qwen answers, questions/history and system prompt. All 109 feature records
passed the same complete source/target-token, finite BF16 and length/dimension
checks. No exporter, trainer, model or loss changes are needed.

V31's prepared cache now contains 87,208 train and 3,197 validation rows:
195 train and 16 validation additions relative to active V30. Factor-64 replay
includes 857 actual contextual rows and 400 numeric rows, giving 166,399
positions and 5,200 steps. The original prepared cache is retained; this plan
is still unlaunched and must use the latest V30 final after generation review.

All 433 plain first-turn source bodies from the completed V30 batch have now
been reviewed. Their retained checked targets cover 294 conversation groups.
Matching completed second turns were verified against each group's unchanged
first prompt/answer and next user message; 282 literal Cyrillic followups are
eligible for manual review. Raw candidates stay outside training until source
and accented-target QA are complete. Their fixed whole-group split is retained.

The first 75 of these second-turn bodies were read. Eight sources were excluded
and 67 sent through the unchanged GPU accent adapter. Every accented target
was read: uncertain pose-name stress, missing lexical stress and a full-text
duplicate quarantined three rows. The remaining 61 train/three validation rows
inherit the pre-fixed whole-conversation split; every history was verified
against its retained completed first turn. All 64 feature records passed the
same source/target-token, finite BF16 and dimension/length checks.

The next immutable cache is now
`freeze-features-v31-context-next-plus-followup75`: 87,269 train and 3,200
validation rows, adding 256 and 19 respectively to active V30. Replay contains
918 actual contextual rows plus 400 numeric rows; factor 64 gives 170,303
positions and 5,322 optimizer steps. Earlier caches remain intact, and this
plan is still prepared rather than launched. Another 207 followup source
bodies remain unreviewed and outside training. V30's step-2400 callback was
also read in full; the long TBC/PayMe repetition persists.

All 42 native step-2400 control outputs were read. Followup18 WER is 11.54%
(baseline 13.85%) and first4 is 16.36% (22.73%), but prior new20 regresses to
17.34% from the 15.99% baseline and 11.92% at step 1200. The Titanic answer
now stops early; other long answers corrupt beginnings and repeat words.
These fluctuations do not establish reliable normalization. The bounded V30
epoch continues; the latest final and all queued panels must still be reviewed
before any subsequent continuation. No demo promotion is made.

## V30 final and V31 continuation

V30 completed 4,810 optimizer steps in 1,845.0 seconds. Train CE is 0.02956
and final validation CE is 0.077396. All nine native generation panels completed
successfully. The latest final SHA256 is
`81f99feef6ff479b3ff97e6082f5f2f42f1c1d1181d571090f6e3382242c6bc8`.

| Panel | V29 final / pre-training baseline WER | V30 final WER |
|---|---:|---:|
| Contextual200 | 4.32% | 4.82% |
| Protected user-demo5 | 40.43% | 32.98% |
| Numeric32 | 31.62% | 31.91% |
| Original64 | 4.39% | 4.66% |
| Previous new10 | 9.66% | 7.25% |
| Previous new11 | 26.69% | 27.03% |
| Previous new20 | 15.99% | 15.72% |
| New followup18 | 13.85% | 11.92% |
| New first4 | 22.73% | 17.27% |

All 100 source/prediction pairs in demo5, new10, new11, new20, followup18,
first4 and numeric32 were read, along with ten final callback outputs.
Short phrases often preserve content, but long answers still corrupt openings,
omit words and repeat spans. Numeric amounts change or lose digits, and the
protected MFTI/time reading remains wrong. The demo stays stopped; this is a
mixed normalization result, not an audio-ASR metric or a deployment fix.

All remaining 207 V30 literal followup source bodies were read. Thirty-two
defective or out-of-scope sources were quarantined. Stock GPU RUAccent processed
175 sources; every target was read. Eleven rows were excluded for missing,
wrong or uncertain stress, incorrect sky -> palate yo, or normalized full-text
overlap. The retained 150 train and fourteen validation rows preserve actual
Qwen answers, history and system prompt and inherit the fixed whole-group split.
All 164 exported records passed source/reference/target-token, finite BF16 and
dimension/length checks. These remain checked automatic labels, not human gold.

The completed V31 cache, `freeze-features-v31-context-next-all-followup`, contains
87,419 train and 3,214 validation rows: 406 and 33 additions relative to V30.
The unchanged 2,302-row frozen eval is retained. Factor-64 replay covers 1,068
actual contextual training rows and 400 existing numeric rows, giving 179,903
sampled positions and 5,622 optimizer steps. All earlier caches remain intact.

Before fitting, the immutable V30 final was evaluated on all 33 added validation
rows. All sources and predictions were read. The four separate panels have WER
18.04% (first10), 10.28% (first6), 0% (followup3), and 4.52% (followup14).
The three-case result is only a tiny diagnostic; long first turns still corrupt
and repeat despite many content-exact shorter followups.

V31 is launched on the same dev GPU0 from the latest V30 final, with BF16,
batch 32, one epoch, LR `1e-6`, a fresh optimizer/schedule and unchanged
439,416,570 trainable parameters. No model, loss, trainer or exporter changes
were introduced. Thirteen native generation panels are queued after training.
Exact launch, baseline/manual reviews and cache hashes are under
`research/v31-context-next`; V30 final review is under `research/v30-context-followup`.

At V31 step 600, separate native evaluation on dev GPU1 completed eight panels.
All 80 source/prediction pairs and ten callback outputs were read. Prior new20
WER improves from 15.72% to 13.55%, and new followup14 from 4.52% to 3.17%.
Prior followup18 worsens from 11.92% to 12.31%; new first10 from 18.04% to
18.56%, and first6 from 10.28% to 14.95%. Protected demo5 remains 32.98%,
first4 is 18.18% versus 17.27%, and followup3 stays content-exact on all three
cases. Long answers still corrupt openings, drop words and repeat spans.
The callback retains corrupt `поврхив`, though its repeated body span is absent
in this particular sample. This mixed early result does not justify promotion.

The unchanged stock generator is now preparing another 512 two-turn source
prompts on dev GPU1, while V31 trains on GPU0. Source is the same pinned
[ru_turbo_alpaca revision](https://huggingface.co/datasets/IlyaGusev/ru_turbo_alpaca/tree/460b1f3312aa21ef774e916e532a9576f7938a0d).
Both compressed and decompressed source hashes match the preceding batches.
All 1,280 prior source indices are excluded; normalized full questions have
zero overlap with prior prompts and zero internal duplicates. Forty-one whole
conversation groups are fixed as held out before generation or fitting.
The generator uses the same frozen Qwen/system prompt, BF16 CUDA, SDPA,
stock HF generation and greedy 96-token bound. Original dataset answers are
not training targets. Artifacts are under `research/v32-alpaca-next`; all raw
output remains ineligible until full source/target review, incomplete-history
exclusion and overlap checks. Active V31 data and training settings are unchanged.

## V31 step 3000 and the next prepared batch

The separate stock native evaluation of immutable V31 step 3000 completed all
eight panels on dev GPU1. All 80 source/prediction pairs and ten callback
outputs were read. Validation CE is 0.077681.

| Panel | V30 final / pre-fitting baseline WER | V31 step 3000 WER |
|---|---:|---:|
| Previous new20 | 15.72% | 11.92% |
| Previous followup18 | 11.92% | 9.23% |
| Protected user-demo5 | 32.98% | 34.04% |
| First4 | 17.27% | 24.55% |
| First10 | 18.04% | 16.49% |
| First6 | 10.28% | 7.48% |
| Followup3 | 0% | 1.89% |
| Followup14 | 4.52% | 2.71% |

Several contextual panels improve, but this remains a mixed intermediate
result. Long pizza, encyclopaedia, resume and book outputs still corrupt
openings, omit words and repeat spans. The protected MFTI/time reading remains
wrong; the callback repeats an archive fragment. These are text-normalization
metrics, not audio ASR. The demo remains stopped. The step-3000 snapshot is
evaluation-only, with weight SHA256
`5bf12933a980f03f15e6c01c776b9c7c04407834a4f72841317b88d504e9b1b1`.
Receipts are under `research/v31-context-next`.

The V32 source generator exited successfully: 1,002 completed answers and
22 truncated first turns. Followups to those truncated histories are excluded.
The final output agrees with the immutable first-turn review snapshot.
Of 490 completed first turns, 440 are in the literal-Cyrillic annotation scope.
All 440 source bodies were read; 91 malformed or uncertain-quality examples
were quarantined without rewriting their Qwen answers.

Stock GPU RUAccent processed the remaining 349 sources. All 349 targets were
read. Seventeen rows with missing or wrong lexical stress or inappropriate
yo replacement and one normalized full-text duplicate were quarantined.
The retained 306 train and 25 validation rows preserve the actual Qwen history
and system prompt and inherit the whole-group split fixed before generation.
There is no normalized full-text overlap with active train, validation or
frozen eval. These are checked automatic labels, not independent human gold.

All 331 exported feature records passed source/reference/ID, target-token,
finite BF16, paired-shape, hidden-dimension, token-length and source-hash checks.
The unchanged stock assembler produced the separate immutable cache
`freeze-features-v32-first`: 87,725 unique train and 3,239 validation rows,
retaining the 2,302-row frozen eval. Existing factor-64 replay includes 1,374
contextual training rows and 400 numeric rows, giving 199,487 sampled positions
and 6,234 optimizer steps at batch 32. Its plan in `research/v32-alpaca-next`
is prepared, not launched: continuation must use the latest V31 final after
native generation and manual review, with a fresh optimizer/schedule.
Active V31 inputs remain unchanged; no model, loss, trainer or exporter code
was changed.

Another 313 literal followup sources have histories verified against the
retained complete first turns. They remain outside training pending full
source and accented-target review.

## V32 combined preparation and contextual label amendment

Reviewing followup context exposed a missed first-turn annotation error:
`начала лета` had become `начала л+ёта`, although the original question
explicitly refers to summer. This row had only reached the unlaunched prepared
cache; active V31 never used it. The immutable `first-qa2` dataset excludes it,
leaving 305 first-turn train and 25 validation rows. Its review amendment is
recorded under `research/v32-alpaca-next`. The earlier first-turn-only prepared
cache is superseded and must not be used for continuation.

All 313 followup source bodies were read. Thirty-one were quarantined for
malformed wording or unreliable content, including the group of the additionally
excluded first turn. Stock GPU RUAccent processed 282 sources; every target
was read. Eight wrong or missing stress/yo labels and four normalized full-text
duplicates were quarantined without rewriting targets. The retained 249 train
and 21 validation followups preserve actual Qwen history and inherit the fixed
whole-group split. No cross-split group or normalized full-text collisions were
introduced.

The unchanged stock exporter regenerated features for the combined corrected
first turns and followups. All 600 records passed source/reference/ID,
target-token, finite BF16, paired-shape, dimension, token-length and hash checks.
The stock assembler produced `freeze-features-v32-combined-qa2`, containing
87,973 unique train and 3,260 validation rows: 554 and 46 additions relative to
active V31. The frozen 2,302-row eval is unchanged. Factor-64 replay covers
1,622 contextual training rows plus 400 numeric rows, yielding 215,359 sampled
positions and 6,730 steps at batch 32. `research/v32-alpaca-next/next-plan.json`
now points to this completed cache. It remains unlaunched and requires the latest
V31 final after native generation review; earlier caches are retained. No model,
loss, trainer or exporter code changed.

Meanwhile, the unchanged generator has started 512 further two-turn prompts on
dev GPU1 under `research/v33-alpaca-next`. All 1,792 previously selected source
indices and normalized full questions are excluded. Forty-one whole groups
are held out before generation. The source is the same hash-verified pinned
dataset, and original dataset answers are not targets. Outputs remain ineligible
until complete source/target review and history/leakage checks.

## V31 final and V32 continuation

V31 completed 5,622 optimizer steps in 2,165.9 seconds. Train CE is 0.03009
and final validation CE 0.077879. All thirteen native generation panels
completed successfully. The latest final SHA256 is
`bc2303a720c15c9aa0c774b289d0197cf8d7b852060a7a699485ceb14c4d6dfc`.

| Panel | V30 final / pre-fitting baseline WER | V31 final WER |
|---|---:|---:|
| Contextual200 | 4.82% | 4.15% |
| Protected user-demo5 | 32.98% | 31.91% |
| Numeric32 | 31.91% | 31.91% |
| Original64 | 4.66% | 3.91% |
| Previous new10 | 7.25% | 7.25% |
| Previous new11 | 27.03% | 27.03% |
| Previous new20 | 15.72% | 11.92% |
| Followup18 | 11.92% | 10.38% |
| First4 | 17.27% | 19.09% |
| First10 | 18.04% | 11.86% |
| First6 | 10.28% | 8.41% |
| Followup3 | 0% | 0% |
| Followup14 | 4.52% | 2.71% |

All 133 source/prediction pairs outside Contextual200 and Original64 and all
ten final callback outputs were read. The two larger panels' metrics were
checked; this is not a claim that all their texts were manually reviewed.
Short replies often preserve content, but longer answers still corrupt
openings, lose words, stop early and repeat spans. Numeric amounts and digit
counts remain unreliable; MFTI/time is still wrong. These are normalization
metrics, not audio ASR. The demo remains stopped. Review receipts are under
`research/v31-context-next`.

Before V32 fitting, the immutable V31 final was evaluated on all 46 newly
added validation rows. All source/prediction pairs were read. Baseline WER is
10.74%; 22/46 preserve the full normalized content and 10/46 exactly match
the stressed reference. Opening/title corruption, missing words, repetition
and incorrect stress remain. Predictions were not used to rewrite targets.
The baseline and manual-review receipts are under `research/v32-alpaca-next`.

V32 is launched on the same dev GPU0 from the latest V31 final and the
audited corrected `freeze-features-v32-combined-qa2` cache. Settings remain
BF16, batch 32, one epoch, LR `1e-6`, a fresh optimizer/schedule and
439,416,570 trainable parameters. It uses 87,973 unique train and 3,260
validation rows, with 554 and 46 additions; replay gives 215,359 positions
and 6,730 steps. Fourteen native panels are queued after training, including
the new 46-row panel. No model, loss, trainer or exporter changes were made.
The launch receipt is under `research/v32-context-followup`.

The V33 generator also exited successfully: 998 completed and 26 truncated
answers. Its 442 literal first-turn candidates remain outside training pending
full source/target review. Followups to incomplete first turns must be excluded.

## V32 step 600 and V33 first-turn preparation

The immutable V32 step-600 snapshot has SHA256
`445b00c01884d41b328e27792090370c251e5c6f08324233247b2461782e091e`.
All nine requested native panels completed successfully, and all 126
source/prediction pairs plus the ten callback examples were read. New46 WER
changed from 10.74% to 9.74%, and protected user-demo5 from 31.91% to 28.72%.
First10 regressed from 11.86% to 20.62%. Short replies often preserve content,
but word loss, repeated spans, corrupt openings, incorrect MFTI/time and stress
errors remain. Even the short `Дождь увлажняет почву` is corrupted, and
`Никто ещё не пообедал` loses its opening. These small-panel results are mixed,
not evidence of consistent improvement or readiness to deploy. Receipts are
under `research/v32-context-followup/step600`. The ten callback examples at
each of steps 1,200 and 1,800 were also read; archive duplication remains. V32 continues without
rollback, and the demo remains stopped.

All 442 V33 first-turn candidates and their supplied questions were read.
129 malformed, uncertain factual/recipe or task-content examples were
quarantined unchanged. The stock CUDA RUAccent adapter processed the remaining
313; all accented targets were then read. Fourteen further exclusions include
wrong contextual `ё` in sky/commotion words, missing lexical stresses and one
malformed source (`шоколадой`) missed during the initial source review. That
source-review amendment is explicitly recorded before fitting; no labels or
histories were rewritten. Three full-text duplicates were excluded.

The eligible first-turn dataset is `qwen-context-reviewed-v33-first`, with 274
train and 22 validation rows. The forty-one preselected heldout groups are
unchanged; normalized full-text overlaps with the active train/validation,
including frozen evaluation, are excluded. These are manually checked
automatic labels, not independent human gold.

The unchanged BF16/SDPA GPU exporter produced
`freeze-qwen-context-reviewed-v33-first`. Every one of its 296 records passed
source/target identity, target token, length, paired-shape, finite-BF16 and
metadata hash/count checks. Metadata SHA256 is
`3a48a172cebf1bda6e2486edcfe6354f5d0e79a7ae5a7d33aff5a98a26dc1ade`;
the complete audit is `research/v33-alpaca-next/first-feature-audit.json`.
This dataset is prepared for a later continuation, not part of active V32.
Another 280 literal followup candidates have verified actual histories and
await complete source/target review. No model, loss, trainer or exporter code
changed.

The V33 first22 pre-fitting baseline using the immutable V31 final completed;
all 22 source/prediction pairs were read. WER is 19.91%, with 9/22 preserving
normalized content and 5/22 exactly matching automatic stress references.
Long answers still repeat/drop words, and a short quoted grammar answer also
corrupts. This is a pre-V33 reference, not a result from active V32 or a claim
of improvement. It did not drive label rewrites. Its manual receipt is under
`research/v33-alpaca-next/baseline-first22`.

## V32 step 1800 review and prepared V33 continuation

All 126 native source/prediction pairs at step 1,800 were read. The immutable
snapshot SHA256 is
`02f959656a374e8c213ecad2193e1954aa25d1b69e43996d727724477d57aaa6`.
Compared with step 600, New20 WER improved from 13.28% to 8.40% and Followup18
from 10.77% to 8.46%; New46 changed from 9.74% to 9.89%, and user-demo5
regressed from 28.72% to 32.98%. Short book replies improved, but rain/soil
and `Никто ещё` remain corrupted, and another reply loses `футбол`. Long
outputs still repeat or omit words, and MFTI/time errors remain. This is mixed
small-panel evidence, not a deployment result. The receipt is under
`research/v32-context-followup/step1800`.

All ten callback examples at each of steps 2,400, 3,000 and 3,600 were read.
Archive duplication persists; step 3,000 and 3,600 repeat `поврхив` twice.
The wrong stress in `нормализов+ана` remains. At step 3,600, validation CE is
0.078281 and 53.49% of the epoch is complete; V32 continues without rollback.

All 280 V33 followup candidates were read with their actual preceding
questions and answers. Fifty-three malformed, irrelevant, uncertain or
foreign-script examples were quarantined without rewriting. The existing
CUDA accent adapter processed 227 remaining rows; all 227 targets were read.
Ten further exclusions include missing/wrong lexical stress and two source
review amendments, recorded before fitting. Eight normalized full-text
duplicates were removed against active data and retained V33 first turns.
The eligible followups contain 197 train and 12 validation rows, with
unchanged actual histories and the same groups held out before generation.
These are checked automatic labels, not independent human gold.

The unchanged GPU exporter completed, and every one of the 209 followup
records passed identity, token, shape, length, finite-BF16 and metadata checks.
Its metadata SHA256 is
`e748227d8e50c67e66a5a3d0e2efa35abcfa4d48a1957b6428a7c566f6dda774`.
The existing assembler combined these with the 296 audited first-turn records
and then appended them to the V32 dataset/cache using hardlinks. No active
V32 inputs were changed. The prepared V33 cache has 88,444 unique train and
3,294 validation rows, adding 471 and 34 respectively; the frozen 2,302-row
evaluation set is retained. Replay selects 2,093 contextual and 400 numeric
train rows at factor 64, giving 245,503 positions and 7,672 steps at batch 32.

`research/v33-alpaca-next/next-plan.json` is prepared, not launched. Its
warmstart must be the latest completed V32 final after final native review,
with a fresh optimizer/scheduler as before. No model, loss, trainer or
exporter changes were made. The demo remains stopped.

## V32 step 3600 native review and V34 source preparation

The immutable step-3,600 snapshot SHA256 is
`9c50c3a76729dd73015ae4e67257991e46d8494f07696b7db5780325f22b9e42`.
Three native panels completed successfully, and all 85 source/prediction
pairs were read. User-demo5 WER is 27.66% versus V31 final 31.91%, but
MFTI/time and both pizza replies remain corrupted. New46 WER is 10.03%,
with 28/46 normalized-content and 12/46 stress matches; long repetition,
word loss, and the short rain/soil and `Никто ещё` failures persist.

Before V33 fitting, its complete new34 validation panel has WER 15.89%,
17/34 normalized-content and 11/34 stress matches, with six invalid-stress
outputs. A grammar quote loses its sentence, names corrupt, and long
neuron/toothpaste/culture answers repeat or lose content. This intermediate
baseline is not a V33 training result. Targets were not rewritten to match
predictions. Its receipts are under `research/v32-context-followup/step3600`.
The ten step-4,200 callback examples were also read; archive duplication
persists once and `нормализов+ана` is still wrong.

V33's replay-weighted length audit counts 150,907 short (at most 110
characters), 65,117 medium and 29,479 long (at least 180 characters)
positions. These are character lengths, not token lengths; long examples
are present but remain the weakest generation cases.

V34 uses the same pinned source, excluding 2,304 prior source indices and
normalized prior questions. Its remaining 216 eligible literal questions
produced 423 complete and nine truncated two-turn answers using the unchanged
GPU/BF16 generator. Whole groups were held out before generation; original
dataset answers are not targets. All 185 scoped first-turn questions and
answers were read, and 46 malformed, incorrect or uncertain examples were
quarantined unchanged. The existing CUDA accent adapter processed 139 rows;
all 139 targets were read. Eight further exclusions include missing lexical
stresses and two source-review amendments caught before fitting.

`qwen-context-reviewed-v34-first` contains 120 train and 11 validation rows,
with unchanged histories and no normalized full-text overlap with prepared
V33 or frozen evaluation. These are checked automatic labels, not independent
human gold. The unchanged GPU feature exporter completed; all 131 records
passed identity, token, shape, length, finite-BF16 and metadata checks. Metadata
SHA256 is `ad7cc5ec1cbb5f12e338abeec258d4da7699b1821781ba826d7cf25a3b483d21`.
Neither V33 nor V34 has started fitting; active V32 continues. No model, loss,
trainer, generation or exporter code changed.

## V34 followup audit and prepared continuation

All 123 eligible V34 followups were read with their actual previous questions
and answers. Thirty malformed, irrelevant or uncertain sources were quarantined
without rewriting. The existing CUDA accent adapter processed 93 sources; every
target was read. Four additional exclusions catch missing stress, missing
contextual yo and incorrect `весны -> вёсны` / `все -> всё` replacements.
The retained 78 train and 11 validation rows inherit the groups held out before
generation. Actual histories and original written answers remain unchanged;
there are no normalized full-text overlaps with prepared V33, V34 first turns
or frozen evaluation. Labels are checked automatic annotations, not human gold.

The unchanged exporter completed successfully. All 89 records passed source,
reference, target-token, length, finite-BF16, paired-shape and metadata checks.
Followup metadata SHA256 is
`f8f97d88cf81fa37ac2c82228f9aee9ee02b4907e332434ce7e8a3eef2aa0e35`.
The existing assembler combined first turns and followups into 198 train and
22 validation rows, then appended them to the separate prepared V33 cache.
The resulting V34 cache has 88,642 train, 3,316 validation and the unchanged
2,302-row frozen eval. Active V32 inputs remain immutable.

Existing factor-64 replay covers 2,291 contextual and 400 numeric train rows,
giving 258,175 positions and 8,068 steps at batch 32. The plan under
`research/v34-alpaca-next/next-plan.json` is prepared only: it requires the
latest completed V33 final and native review. V33 is the next planned run;
V34 has not started fitting. No model, loss, trainer or exporter code changed.

All ten callback triples at each of V32 steps 4,800, 5,400 and 6,000 were read.
Archive corruption remains, including two repeated fragments at 5,400/6,000,
and stress in `нормализов+ана` remains wrong. Other eight examples preserve
normalized content. These callback cases do not establish broader improvement
or qualify the checkpoint for demo promotion. The demo remains stopped.

## V32 final and launched V33

V32 completed 6,730 optimizer steps in 2,603.1 seconds. Train CE is 0.028481
and final validation CE 0.078322, versus V31 final 0.077879. All fourteen native
generation panels completed without runtime errors; the container exited zero.
Latest-final SHA256 is
`407f746b3bca46c77ffe59778a4f640303d67ea9459824ba271b15d8bd672c63`.

| Panel | V31 final / pre-fitting baseline WER | V32 final WER |
|---|---:|---:|
| Contextual200 | 4.15% | 4.15% |
| Protected user-demo5 | 31.91% | 28.72% |
| Numeric32 | 31.91% | 30.20% |
| Original64 | 3.91% | 3.56% |
| Previous new10 | 7.25% | 5.31% |
| Previous new11 | 27.03% | 23.99% |
| Previous new20 | 11.92% | 7.32% |
| Followup18 | 10.38% | 5.38% |
| First4 | 19.09% | 19.09% |
| First10 | 11.86% | 13.40% |
| First6 | 8.41% | 10.28% |
| Followup3 | 0% | 0% |
| Followup14 | 2.71% | 2.26% |
| Newly added46 | 10.74% | 8.60% |

For the 179 rows outside Contextual200 and Original64, all 61 changed
source/reference/prediction triples were reread in full. The remaining 118
predictions exactly match previously fully read outputs, with unchanged written
text, references and optional history/system. The 46-row comparison uses the
previous step-3,600 review; other panels use V31 final. All ten final callback
triples were read. The two larger panels have metrics checked, without claiming
full manual review of their text. Exact comparison IDs and hashes are in
`research/v32-context-followup/final-manual-review.json`.

The result is mixed: several panels improve, but First10 and First6 regress.
Long replies still corrupt openings, omit words and repeat spans; number scale,
digit-count and MFTI/time errors remain. These are normalization metrics, not
audio ASR, and the checkpoint is not promoted to the stopped demo.

V33 is launched on the same dev GPU0 from the immutable latest V32 final.
Its already audited cache contains 88,444 train and 3,294 validation rows,
including 471 new train and 34 new held-out rows. Existing factor-64 replay
gives 245,503 positions and 7,672 steps. Settings remain BF16, batch 32, one
epoch, LR `1e-6`, 439,416,570 trainable parameters and a fresh optimizer/schedule.
Fifteen native generation panels are queued, including the new 34-row panel.
No model, loss, trainer or exporter changes were introduced. The launch receipt
is under `research/v33-context-followup`; V34 remains prepared for later.

## V33 intermediate review and fresh Saiga data

V33 remains active on dev GPU0. At step 4,200, validation CE is 0.078523;
this is a teacher-forced measure, not evidence that free generation is fixed.
All ten callback triples at steps 1,800, 2,400, 3,000, 3,600 and 4,200 were read.
The telephone number is correct again from step 2,400, but malformed archive
repetitions and stress in `нормализов+ана` persist. Receipts bind each review
to its actual callback file hash. The checkpoint is not promoted to the demo.

The completed step-600 native evaluation covers 85 held-out rows. On the
new 34-row panel, a matched latest-V32-final baseline has WER 15.44%, versus
13.49% at V33 step 600. Content-exact examples remain 17/34. The previous
46-row panel changes from 8.60% to 8.45%, while the protected five user-demo
examples regress from 28.72% to 38.30%. All five demo triples and changed
triples on the other panels were read; unchanged predictions are tied to
previous full reviews. These are text-normalization metrics, not audio ASR.
The result is mixed and does not establish general improvement.

The step-4,200 native rerun also completed all 85 rows, exiting zero. WER is
7.59% on the 46-row panel (30 content-exact rows), 12.59% on the new 34-row
panel (18 content-exact rows), and 34.04% on protected user-demo5 (one
content-exact row). All five demo triples and 31 changed triples on the other
panels were read in full; 49 unchanged predictions match the prior fully
reviewed outputs, with unchanged sources, references, histories and system
prompts. Long replies still repeat and omit spans; MFTI/time errors persist.
`step4200/manual-review.json` records exact IDs, metrics and hashes. The
immutable step-4,200 snapshot is evaluation-only, never a training warmstart.

Fresh questions come from `IlyaGusev/ru_turbo_saiga`, pinned at
`4f92f393386e9e623f39832411aeee3064b0890c` (CC BY 4.0, 37,731 chats).
Only user questions are reused. Targets are newly generated by frozen
Qwen3-4B-Instruct-2507 with the unchanged Russian system prompt and actual
conversation histories; source-dataset bot answers are never training targets.
512 distinct questions, excluding prior normalized questions, were selected
deterministically, with 41 whole conversation groups held out before generation.
The unchanged GPU generator produced 1,004 complete and 20 truncated answers.
Raw generation is not approved training data.

Of 423 scoped first-turn sources, the first 192 questions and answers have
been read; 231 first turns and all followups still await review. The first
128-source cohort retained 62 sources; all 62 accent targets were read and two
bad labels quarantined, leaving 55 train and five validation rows. All 60
feature artifacts passed identity, token, length, paired-shape, finite-BF16
and metadata checks. The next 64-source cohort retained 18 sources; all 18
accent targets were read, with incorrect `д+уша` quarantined. Its remaining
16 train and one validation row also passed all 17 artifact checks.
Labels are manually checked automatic annotations, not independent human gold.
Sources, histories and split assignments remain unchanged.

The existing assembler appended the first cohort to the separate prepared
V34 cache, producing 88,697 train and 3,321 validation rows. Factor-64 replay
gives 261,695 positions and 8,178 steps at batch 32. This V35-first128 plan is
prepared only and requires the latest completed V34 final and native review.
Its source and feature hashes, QA receipts and unlaunched plan are under
`research/v35-saiga-next`; active V33 inputs remain immutable.

The audited second cohort was subsequently appended to another fresh cache,
`freeze-features-v35-first192-qa`: 88,713 train, 3,322 validation and unchanged
2,302 frozen-eval rows. Replay covers 2,362 contextual and 400 numeric rows,
giving 262,719 positions and 8,210 steps. `next-plan-first192.json` is the
newer prepared V35 plan; the first128 plan remains an unlaunched historical
artifact. Both require latest V34 final weights, which are not available yet.

A feature-export warning was checked in context: the tokenizer directory
retains a `lalm` model config, so upstream AutoTokenizer loads a generic
configuration before choosing Qwen2Tokenizer. A separate config check confirms
the actual frozen model directory is Qwen3Config, hidden size 2,560. The
exporter loads its model from that separate Qwen3 directory. The warning is
from tokenizer configuration discovery, not a replacement of Qwen weights;
the check used CPU configuration/tokenizer loading only, not CPU model inference.
No model, trainer, loss or exporter implementation changed. Demo stays stopped;
continuations use latest final weights with a fresh optimizer and schedule.

## V35 first320 data review while V33 continues

The next 128 first-turn questions and actual frozen-Qwen answers were read in
full, bringing source review to 320/423. The remaining 103 first turns and
all followups are still unreviewed and are not admitted to training.
The first 64-source cohort retained 22 sources; all 22 accent targets were
read. Incorrect imperative `рассм+отрите` was quarantined, leaving 21 train
rows and no new eligible heldout rows. Preparation stopped before creating
that standalone dataset; no export was launched. Combining these rows with
the previous reviewed cohorts retained the original six heldout rows, without
moving any conversation between splits.

The next 64-source cohort retained 21 sources. All 21 accent targets were
read; double stress in `ух+одов+ые` and missing lexical stress on `почва`
were quarantined. The remaining 15 train and four validation rows inherit
the split fixed before generation. Written answers and actual histories
remain unchanged, with no normalized full-text overlap with the prepared
dataset or frozen evaluation. Labels are checked automatic annotations,
not independent human gold.

The unchanged GPU exporter completed both fresh combined cohorts. Every one
of the first256 cohort's 98 artifacts and the first320 cohort's 117 artifacts
passed source/reference/token identity, length, finite-BF16, paired shape
and metadata checks. First320 feature metadata SHA256 is
`0b8b29483d63921e5582589acb21d256949603f7faee11f5d13acef7ca4fa12d`.
The existing assembler appended its 107 train and ten validation rows to the
separate prepared V34 cache, producing 88,749 train, 3,326 validation and
the unchanged 2,302 frozen-eval rows.

`research/v35-saiga-next/next-plan-first320.json` is the newer unlaunched V35
plan: factor-64 replay covers 2,398 contextual and 400 numeric train rows,
giving 265,023 positions and 8,282 steps at batch 32. It requires latest V34
final weights and native review; V33 remains active and V34 remains next.
Earlier first128/first192/first256 plans are retained as historical artifacts.
Active V33 data and features were not modified.

A read-only stress-syntax audit of all 88,444 active V33 train rows and all
88,749 prepared first320 train rows found zero invalid plus placements and
zero words with multiple stress markers. Its file-bound receipt explicitly
does not prove correct syllables or pronunciation. All ten callback triples
at V33 steps 4,800, 5,400 and 6,000 were read: the telephone number remains correct,
but malformed archive repetitions and `нормализов+ана` persist. These cases
do not support demo promotion. No model, loss, trainer or exporter code changed.

## V33 final review and V34 continuation

V33 completed one epoch and 7,672 steps in 2,966 seconds. Train CE is
0.024444 and validation CE is 0.078236. Its container then completed all
15 native generation panels and exited zero without OOM. Final callback
triples exactly match the already reviewed steps 6,600/7,200/6,000.
The final review reads all 54 changed source/reference/prediction triples
and all five user-demo triples. Another 154 small-panel predictions match
previously reviewed outputs with unchanged source, reference, history and
system prompt. Context200 and original64 are metrics-only checks.
Receipts and file hashes are under `research/v33-context-followup`.

Results remain mixed: combined46 WER is 8.02% versus V32 final 8.60%, and
combined34 is 14.54% versus the matched V32 final 15.44%. Both are worse
than V33 step 4,200 on those panels. Protected user-demo5 remains 34.04%
versus V32 final 28.72%. Numeric32 remains 30.20%; context200 is 4.32%
and original64 is 3.15%. These are text-normalization comparisons, not
audio ASR. Long replies still repeat or omit spans; time, abbreviation,
numeric-value and grammatical-gender errors persist. No demo promotion.

As requested, continuation uses latest final weights, not the better
intermediate checkpoint. The immutable V33 final snapshot has SHA256
`f604afb6768b095a5dc69c2dfcb98f458e49be55d130ca0e96448d2a5605af61`.
`normalizer-train-v34` is running on reserved GPU0 with those model weights
and a fresh optimizer/schedule. It uses 88,642 unique train rows, 3,316
validation rows, batch32, BF16, LR1e-6 and one epoch: 258,175 replay
positions / 8,068 steps. Its 198 new train and 22 new heldout rows were
already reviewed and audited. All preceding panels plus the new 22-row
panel are queued after training. First optimizer steps were observed.

## Prepared first423 cohort and new numeric-million data

All 423 scoped Saiga first-turn questions and actual frozen-Qwen answers
have now been read. All followups remain unreviewed and are not admitted.
Further source/target quarantines include wrong imperative stress,
unexpanded `ИИ`, incorrect stress in `без вести`, and awkward or unsupported
source answers. The final first-turn cohort retains 139 train and 11
heldout rows, preserving actual histories and the original whole-group
split. Every one of its 150 frozen feature artifacts passed the existing
identity, token, shape, length, finite-BF16 and metadata audit.

A separate stock `num2words==0.5.14` run produced 10,000 new train and 214
heldout numeral pairs, excluding previous numeric values and normalized
full texts. This covers nominative integers from one million to below
one billion; it does not cover times, dates, money or grammatical cases.
The first attempt failed before output creation because the image lacked
num2words. A corrected run used isolated pinned formatter dependencies;
no model, trainer, exporter or normalization implementation changed.

Manual numeric QA covered all 30 source edge pairs, all 47 distinct
stressed lexical forms and ten complete sampled pairs. Every one of the
10,214 rows passed source/content/metadata checks, and every frozen feature
artifact passed the existing audit. This is scoped review of automatic
labels, not independent manual verification of all number expansions.
Artifacts and receipts are under `research/v36-numeric-millions-next`.
These fresh cohorts are separate from active V34 inputs and require their
preceding final checkpoint and native review before training. Demo remains
stopped; all changes stay in the existing PR for user review.

Both future caches were assembled successfully with the unchanged hardlink
assembler. `next-plan-first423.json` supersedes older unlaunched V35 plans:
88,781 train / 3,327 validation rows, 267,071 replay positions and 8,346
steps. It requires latest V34 final. The following V36 numeric plan has
98,781 train / 3,541 validation rows, 277,071 positions and 8,659 steps;
it requires latest V35 final. All 10,000 new numeral examples occur once per
epoch, while the existing reviewed replay selection is preserved. Both
retain the same 2,302 frozen-eval rows. Neither future run has launched.

The first assembly attempt rejected a malformed `GO150` metadata tag before
output creation. A separate approved review file uses the required `GO`
word with unchanged data hashes and QA; the gate was not weakened. Active
V34 reached approximately step285 with finite loss/gradients and no OOM.
GPU0 utilization was 93% in that sample. Disk free space was 82GB on drive1
and 543GB on drive2; feature assembly used hardlinks rather than copying
the base cache. No model/trainer implementation or live training input changed.

## V34 native checks, additional contextual data and batch measurement

V34 is still training. All ten callback triples at steps 1,800, 2,400 and
3,000 were read. Archive repetitions persist; the telephone-number output
regresses at step 3,000. The unchanged greedy evaluator completed four native
panels at steps 600 and 3,000, each covering 107 heldout rows and exiting zero
without OOM. Both snapshots are evaluation-only, never continuation weights.

At step600, combined22 content WER is 11.80% versus the separately checked
latest-V33-final baseline of 16.89%. Combined34 is 10.64% versus 14.54% at
V33 final. Combined46 regresses to 8.88% from 8.02%; protected user-demo5
regresses to 37.23% from 34.04%. All 30 changed triples were read; another
77 predictions match previously reviewed full outputs with unchanged sources,
references, histories and system prompts.

At step3000, content WER is 12.33%, 11.24%, 9.17% and 37.23% on those
same four panels. All 32 changed triples and all five demo triples were read
(35 distinct triples); 72 remaining predictions match the prior full review.
Long outputs still omit words and repeat spans, while MFTI and time expansion
remain wrong. These are text-normalization checks, not audio ASR, and do not
establish general improvement. No demo promotion. Receipts are under
`research/v34-context-followup/step600` and `step3000`.

Independent reference review found incorrect automatic imperative stress
`Из+учите` in the climate task: the intended imperative is `Изуч+ите`.
The [Gramota conjugation table](https://gramota.ru/poisk?mode=all&query=%D0%B8%D0%B7%D1%83%D1%87%D0%B8%D1%82%D0%B5&simple=0)
distinguishes the imperative from the future indicative. This is recorded
out of band in `label-errata.json`; active and protected labels were not
changed. Content WER strips stress; this automatic reference cannot serve
as independent stress gold.

The 150 eligible Saiga followup sources were read in full, including their
actual preceding Qwen answers. Source QA retained 86; all 86 accent targets
were then read, and six questionable labels/sources were quarantined without
rewriting them. The retained 80 followups preserve their original groups,
histories and splits. Together with the 150 already reviewed first turns,
they give 214 new train and 16 validation rows. Other raw followups remain
unreviewed and are not admitted. Every one of the 230 combined feature
artifacts passed the existing identity, token, finite-BF16, shape, length and
metadata audit. Metadata SHA256 is
`fc39e8c9a2ab2d928eb9bf207045864a3e5b2aee3a2fbea5176d00d4b472892a`.

The existing assembler created fresh V35-combined and V36-context-numeric
caches without modifying active inputs. `next-plan-combined.json` supersedes
the unlaunched V35-first423 plan: 88,856 unique train, 3,332 validation and
2,302 frozen-eval rows. Replay includes 2,505 contextual and 400 existing
numeric rows at factor64, giving 271,871 positions. The following
`next-plan-context-numeric.json` appends the already audited 10,000 numeral
train / 214 validation rows: 98,856 train, 3,546 validation and 281,871
positions. The new numeral rows occur once per epoch. Both require the
latest preceding final checkpoint and native review; neither has launched.

An isolated stock-trainer benchmark on GPU1 compared batches32/48/64 over
120 optimizer steps each, using the same active V34 corpus, replay selection,
seed and immutable V33-final starting weights. All three exited zero without
OOM. Measured training throughput was 72.016 / 93.403 / 113.207 samples/s;
batch48 is approximately 30% faster than32 in this short probe. Batch64's
observed physical GPU usage reached 92,329MiB including the unrelated
allocation, so it has limited headroom. Batch48 additionally recorded stock
HF allocator-memory metrics; those do not include reserved-memory peaks.
Variable-length full-epoch peaks and quality are not established by this
benchmark. No benchmark weights are promoted or used for continuation.

Future plans select batch48, giving 5,664 V35 and 5,873 V36 steps. Live V34
remains at batch32. Benchmark results and limitations are recorded under
`research/v34-context-followup/batch-benchmark`. No model, trainer, exporter
or normalization implementation changed; only existing data tools and stock
CLI parameters were used. All continuations retain LR1e-6, one epoch and
latest-final model-only warmstart with a fresh optimizer/schedule. Demo stays
stopped; the existing PR remains open for user review and is not merged.
