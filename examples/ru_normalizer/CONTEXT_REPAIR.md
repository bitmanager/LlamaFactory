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
