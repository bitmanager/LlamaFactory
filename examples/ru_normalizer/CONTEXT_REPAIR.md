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

## V34 step 5,400 review and the next reviewed Saiga cohort

V34 continues on GPU0. Callback triples at steps 4,200/4,800/5,400/6,000/6,600
were checked against their previous full review. The changed archive triple
at 4,200 and the changed archive triple at 6,600 were read in full; intervening
callbacks match exactly. Telephone recovery
persists, but archive repetition and the wrong stress in normalizovana remain.
Live data, learning rate, batch32 and model implementation were not changed.

The step5400 native rerun completed all 107 rows, exiting zero without OOM.
Content WER is 12.06% on combined22, 10.04% on combined34, 7.59% on
combined46 and 35.11% on protected user-demo5. The matched step 3,000 figures
are 12.33% / 11.24% / 9.17% / 37.23%. All 27 changed triples and all five
demo triples were read (30 distinct triples); the remaining 77 predictions
match the previous full review with unchanged source, reference, history and
system prompt. Some quoted outputs stop looping but still lose initial words.
Long neuron, toothpaste, story and art replies continue to omit and repeat
spans; pizza, MFTI and time examples remain unreliable. This is a small
text-normalization check, not audio ASR or proof of general usability.
No demo promotion. Receipts are under `research/v34-context-followup/step5400`.

A fresh 512-question cohort uses the same pinned `IlyaGusev/ru_turbo_saiga`
source. Previously selected normalized questions were excluded; 36 whole
conversation groups were held out before generation. The unchanged CUDA/BF16
generator produced 1,009 complete and 15 truncated answers. All 1,024 turn
records were checked for unique identities, actual preceding answers, original
questions, fixed system prompt and conversation groups. Original source-dataset
bot answers remain unused. Raw outputs are not approved training data.

All 128 first question/actual-Qwen-answer pairs in the scoped cohort were read.
Source review retained 46; all 46 accent targets were read, and one missing
lexical stress was quarantined without rewriting its label. All 45 eligible
followups were also read against their already reviewed actual preceding
answers. Source review retained 30, and every one of those 30 accent targets
was read. The final combined cohort has 71 train and four validation rows,
preserving the original conversation splits and histories. Remaining first
turns and other followups are unreviewed and are not admitted.

Several uncertain stress forms were checked independently against Gramota:
[synergy](https://gramota.ru/poisk?dicts%5B0%5D=24&mode=slovari&query=%D1%81%D0%B8%D0%BD%D0%B5%D1%80%D0%B3%D0%B8%D1%8F&simple=0),
[the imperative of ochistit](https://gramota.ru/meta/ochistit),
[the adjective razvitoy](https://gramota.ru/poisk?dicts%5B0%5D=71&mode=slovari&query=%D1%80%D0%B0%D0%B7%D0%B2%D0%B8%D1%82%D0%BE%D0%B9&simple=0),
and [the participle of perenosit](https://gramota.ru/meta/perenosit).
The direct ochistit page returned 403; its cached search result exposed the
conjugation table. This is bounded independent checking, not independent
human gold for the entire dataset. No labels were derived from head predictions.

Every one of the 75 combined feature artifacts passed the existing identity,
token, paired-shape, length, finite-BF16 and source-hash audit. Metadata SHA256
is `d605c346d27acbce57facac8fbaf1e55db84bfa4830ded2d064d2c832f750e4b`.
The unchanged assembler appended this cohort to the separately prepared V36
cache, giving 98,927 unique train / 3,550 validation and unchanged 2,302
frozen-eval rows. Replay at factor 64 covers 2,576 contextual and 400 existing
numeric rows: 286,415 positions / 5,967 steps at batch 48.
`research/v37-saiga-next/next-plan-combined-first128.json` and its launch configuration for 19 native
panels are prepared only, requiring latest V36 final weights
and native review. V35/V36/V37 remain future continuations; V34 is active.
No model, trainer, exporter or normalization implementation changed.

The live batch 32 process later reached approximately 83 GiB of physical GPU
usage. A separate stock 20-step batch 48 shape stress test therefore
oversampled the union of 64 largest existing feature artifacts and 64 longest
stressed targets by characters (80 distinct examples, factor 50,000).
This disposable test exited zero without OOM. Stock HF metrics record
35,528,516,608 bytes of peak allocated CUDA memory for that short probe;
allocator reservation and CUDA runtime memory are excluded. File size and
character length are shape proxies, and a short run does not prove all
full-epoch peaks. No stress-test weights are promoted or used for continuation.

Batch48 remains the next-run configuration, with stock full-run memory
tracking enabled in each fresh `training-launch-prepared-memory.json` via
`--skip_memory_metrics false`. Active V34 remains unchanged. Evidence is
under `batch-benchmark/shape-stress48-launch.json` and
`shape-stress48-result.json`; no allocator, model or trainer code was changed.

## V34 final review and V35 continuation

V34 completed at step 8,068, epoch one, exiting zero without OOM. Training
took 3,131.45 seconds; final validation CE is 0.0780563. All 16 native panels
completed, covering 499 rows. Final content WER is 11.80% / 9.90% / 7.45%
on combined22 / combined34 / combined46, 4.57% on context200, 3.02% on
original64, 29.91% on numeric32 and 36.17% on protected user-demo5.
These checks measure generated text normalization, not audio recognition.

All 50 selected triples (small-panel changes and all protected demo triples)
were read in full against their source and reference. Exact matches reuse prior full
reviews after source, reference, group and history guards. Context200 and
original64 remain metrics-only reviews. Final callback changes at steps
7,200 / 7,800 / 8,068 were read in full; the telephone number regresses again
at the final step and archive repetition persists. Long outputs still omit
and repeat spans; amounts, percentages, MFTI and pizza examples remain weak.
Final review and limitations are recorded in
`research/v34-context-followup/final-manual-review.json`. No demo promotion.

V35 is launched from the latest V34 final, model-only, with a fresh optimizer
and schedule. Snapshot SHA256 is
`6b5b7a3a75781da840f281feb33910eed584ca023b7cb589ea7a42e337056967`.
It uses 88,856 unique train / 3,332 validation rows, including 214 new train
and 16 new held-out manually reviewed actual-Qwen answers. Existing replay
gives 271,871 positions and 5,664 steps at batch 48; BF16 math, LR 1e-6,
one epoch and stock full-run memory tracking remain enabled. Initial
optimizer steps are running. A separate GPU1 baseline evaluates the new
16-row panel using the exact V34 final snapshot. V36 numeric and V37
contextual caches remain prepared future continuations. No new model or
trainer implementation was introduced; the demo remains stopped.

The new 16-row V35 panel baseline completed without OOM at V34 final:
content WER 11.56%, five exact content outputs and one exact stressed output.
All 16 source/reference/prediction triples were read in full. Long replies
still omit and repeat spans; shorter replies often preserve content but
misplace stress. The baseline is retained for a matched post-training check.

## V35 early native review and expanded V37 cache

V35 remains active. At step 600, validation CE is 0.0786688; at step 1,200
it is 0.0787420. All ten callback triples at step 600 were read in full;
only the archive prediction changes relative to V34 final. Step 1,200
matches the already reviewed step 600 triples exactly. Telephone-number
corruption, archive repetition and the incorrect stress in normalizovana
persist. Active training data, model implementation and CLI settings are
unchanged.

The separate step 600 native check completed 123 rows without OOM. All
42 selected triples (small-panel changes and all five protected demo rows)
were read in full. The remaining 81 predictions exactly match earlier full
reviews after source/reference/history/group guards. Compared with V34
final, content WER changes as follows:

| Panel | V34 final | V35 step 600 |
|---|---:|---:|
| combined22 | 11.80% | 12.33% |
| combined34 | 9.90% | 13.04% |
| combined46 | 7.45% | 8.17% |
| new Saiga16 | 11.56% | 11.06% |
| protected user-demo5 | 36.17% | 25.53% |

The long pizza example retains more content, but time and MFTI errors
persist. Long neuron, toothpaste and story examples still omit and repeat
spans. A short noun changes to the wrong stress. These mixed small-panel
results do not establish broad improvement or justify demo promotion.
Receipts are under `research/v35-context-followup/step600`.

All 126 complete first question/actual-answer pairs from V37 prompt indices
128 through 255 were read; two truncated first answers were excluded.
Source review retained 28. Every accent target was read, and two were
quarantined unchanged: a missing lexical stress and inappropriate yo in a
map-marker sense. The latter was checked against Gramota's
[marker dictionary entry](https://gramota.ru/meta/marker) and
[meaning distinction](https://gramota.ru/biblioteka/spravochniki/slovar-trudnostey/marker-i-marker).
All 26 eligible followups were read against the actual preceding answers;
15 were retained and all 15 stressed targets were read. No targets were
derived from model predictions or manually rewritten.

The new 26 first-turn and 15 followup records extend the already reviewed
75-record cohort to 116 records: 110 train and six validation, with the
original whole-conversation holdouts and histories preserved. Remaining
prompt indices 256 through 511 are unreviewed and excluded. All 116 cached
feature artifacts pass the existing identity, token, paired-shape, length,
finite-BF16 and source-hash checks. Metadata SHA256 is
`577bf3113b66adcca7533984788f6df5c726f46f2b9cb32ac98bfe282025e2f7`.
These remain checked automatic accent labels, not independent human gold.

The unchanged assembler appends the 116-record cohort to the prepared V36
cache. Fresh V37 first256 data/features contain 98,966 unique train / 3,552
validation and unchanged 2,302 frozen-eval rows. Replay covers 2,615
contextual and 400 existing numeric rows at factor 64: 288,911 positions /
6,019 steps at batch 48. A preflight caught an incomplete source-ID list in
the newly prepared replay selection; it was updated to match all ordered
indices before any launch. V35 active, V36 future and the expanded V37
replay identities and metadata hashes now all pass. Active inputs were not
modified.

The expanded future launch is
`research/v37-saiga-next/training-launch-prepared-combined-first256-memory.json`:
19 native panels / 735 rows, latest V36 final model-only warmstart, fresh
optimizer/schedule, stock memory tracking and unchanged LR 1e-6. It is
prepared only. Earlier first128 plans remain historical; use the expanded
first256 plan for the next V37 continuation.

Disk cleanup removed only four completed disposable benchmark model files
(7,030,710,816 bytes). Benchmark metrics, launch records, training data and
all training/final checkpoint weights were retained. Terminal-container
checks and deleted-file hashes are recorded in
`batch-benchmark/disposable-model-cleanup.json`. No model, trainer, exporter
or normalization implementation changed. Demo remains stopped.

## V35 step 3000 review and V37 first320 preparation

V35 continues on GPU0. Callback validation CE at steps 1,800 / 2,400 /
3,000 / 3,600 is 0.0790287 / 0.0789127 / 0.0790268 / 0.0789859.
Every changed callback triple was read in full; unchanged triples reuse
the preceding review after source/reference guards. Only the archive
prediction changes, while telephone corruption and incorrect stress in
normalizovana persist. Archive repetitions still occur.

The separate step 3,000 native check completed all 123 rows without OOM.
All 36 changed triples and all five protected demo triples were read in
full (39 selected triples total). The remaining predictions exactly match
the fully reviewed step 600 outputs after source/reference/history/group
guards. Content WER is:

| Panel | V35 step 600 | V35 step 3000 |
|---|---:|---:|
| combined22 | 12.33% | 14.21% |
| combined34 | 13.04% | 11.99% |
| combined46 | 8.17% | 6.73% |
| new Saiga16 | 11.06% | 13.07% |
| protected user-demo5 | 25.53% | 27.66% |

These are mixed small-panel results. Long generated replies still omit,
repeat and corrupt words; MFTI and time demo outputs remain incorrect.
No broad improvement or demo promotion is claimed. The intermediate
snapshot is evaluation-only and will never be used for continuation.
An initial evaluation attempt failed before generating predictions because
per-panel output directories were missing. Creating those directories and
restarting the same command/snapshot completed successfully; source code
was unchanged. Receipts are under `research/v35-context-followup/step3000`.

All 62 complete first question/actual-Qwen-answer pairs from prompt indices
256 through 319 were read; two incomplete first answers were excluded.
Twenty source answers were retained, and all twenty automatic accent
targets were read. All twenty eligible followups were also read against
their actual histories; fourteen were retained and their full accent
targets read. Six followups with misleading simplifications or awkward
language were quarantined unchanged. No model predictions were used to
rewrite targets.

The fresh first320 cohort extends the previously reviewed 116 rows to
150: 142 train and eight validation. Original whole-conversation holdouts
and actual generated histories remain unchanged. All 150 frozen feature
artifacts pass identity, target-token, shape, length, finite-BF16 and
source-hash checks. Metadata SHA256 is
`e15ae89ef20444ba786a94a15cd3cce60f1ef6ef7a69887b72f05aaf8aeb2b98`.
These are checked automatic accent labels, not independent human gold.
Prompt indices 320 through 511 remain unreviewed and excluded.

The unchanged assembler prepares fresh V37 data/features with 98,998
unique train / 3,554 validation and unchanged 2,302 frozen-eval rows.
Replay covers 2,647 contextual and 400 existing numeric train rows at
factor 64: 290,959 positions / 6,062 steps at batch 48. Ordered replay
source IDs and metadata hashes pass for active V35, prepared V36 and
this new V37 cache. Active inputs and older prepared caches were not
modified.

The newest future V37 launch is
`research/v37-saiga-next/training-launch-prepared-combined-first320-memory.json`:
19 native panels / 737 rows, latest V36 final model-only warmstart and
fresh optimizer/schedule. V36's 10,000 new reviewed-format numeric train
rows remain the immediate next continuation after V35 final review.
Neither future launch has started yet. No model, trainer, exporter or
normalization implementation changed; demo remains stopped.

Disk cleanup removed six older intermediate `optimizer.pt` files from
completed V33/V34 runs, freeing 20,387,778,498 bytes. Terminal states and
reviewed final-weight hashes were checked before removal. All model
weights, native results and each run's complete final optimizer checkpoint
remain. Exact optimizer resume from those six older intermediate checkpoints
is no longer available; latest-final model-only continuation is unaffected.
Evidence is `research/v35-context-followup/retired-optimizer-cleanup.json`.

## V35 step 4200 and expanded first384 cohort

V35 is still running. At step 4,200, validation CE is 0.0788929; all ten
callback source/reference/prediction triples exactly match the fully
reviewed step 3,600 callback. Archive repetitions, telephone-number
corruption and wrong stress persist. The equality and callback hash are
recorded in `step4200-callback-manual-review.json`. This is not evidence
that the remaining full native panels improved.

All 62 complete first question/actual-Qwen-answer pairs from indices 320
through 383 were read; two incomplete first answers were excluded.
Ten first answers and seven of their ten followups were retained. All
seventeen complete automatic accent targets were read. Rejected sources
and followups remain unchanged; no targets are rewritten from predictions.
Original frozen-Qwen history, system prompt and whole-conversation split
are preserved. These remain checked automatic accent labels rather than
independent human gold.

The fresh first384 cohort contains 167 records: 159 train and the same
eight held-out validation rows. All 167 paired feature artifacts pass the
existing source/reference/token/shape/length/finite-BF16/hash audit.
Metadata SHA256 is
`875d4b3270d97a644534d5c6e0203e62bf4cf48479ad0947d589f8803d228deb`.
Prompt indices 384 through 511 remain unreviewed and excluded.

Fresh full V37 data/features now contain 99,015 unique train / 3,554
validation / unchanged 2,302 frozen-eval rows. Replay covers 2,664
contextual and 400 existing numeric train rows, factor 64: 292,047
positions / 6,085 steps at batch 48. Source-ID ordering and metadata-hash
preflight passes for active V35, prepared V36 and this newest V37 cache.
Earlier first320/first256 caches remain immutable historical preparations.

Use the newest future launch
`research/v37-saiga-next/training-launch-prepared-combined-first384-memory.json`.
It remains prepared only: 19 native panels / 737 rows, latest V36 final
model-only warmstart, fresh optimizer/schedule, unchanged LR 1e-6 and
stock memory metrics. V36 numeric-million data remain the immediate next
party after V35 final review. No model, trainer, exporter or normalization
source changed. Demo remains stopped.

## V35 step 4800 and expanded first448 cohort

V35 remains active. Step 4,800 validation CE is 0.0788631; all ten
callback source/reference/prediction triples exactly match the fully
reviewed step 4,200 callback. The same repetitions, number corruption and
stress error persist. Equality and callback hash are recorded in
`step4800-callback-manual-review.json`; no broader native improvement is
inferred from this callback.

At step 5,400, CE is 0.0788434 and all ten callback triples still match
the already reviewed step 4,800 predictions exactly. The receipt is
`step5400-callback-manual-review.json`. The live process passed 96% of
the epoch without OOM; final weights and native review are still pending.
An observation caught the callback before the checkpoint state file was
written; re-polling the same live process confirmed the completed save.
Training was not restarted.

All 63 complete first question/actual-Qwen-answer pairs from indices 384
through 447 were read; the incomplete first answer at index 416 was
excluded. Twenty first answers and fourteen of their twenty followups
passed source review. All 34 automatic accent targets were read. One
followup omitted lexical stress on vlaga and was quarantined unchanged.
The other thirteen followup targets were retained.

Two uncertain stress cases were independently checked. Gramota's indexed
[vypolnimyy entry](https://gramota.ru/meta/vypolnimyy) confirms the suffix
stress used by the automatic target; its direct page returned 403.
Its [reference answer 307633](https://gramota.ru/poisk?mode=spravka&query=%D0%BF%D0%BE%D1%81%D1%82%D0%B5+%D0%B8%D0%BB%D0%B8+%D0%BF%D0%BE%D1%81%D1%82%D1%83+%D0%B2+%D1%81%D0%BE%D1%86%D0%B8%D0%B0%D0%BB%D1%8C%D0%BD%D1%8B%D1%85+%D1%81%D0%B5%D1%82%D1%8F%D1%85&simple=0)
allows both root and ending stress for post in the social-network sense.
The automatic ending-stressed target was therefore retained, rather than
incorrectly relabeled. No predictions were used to rewrite targets.

The newest first448 cohort contains 200 records: 192 train and the same
eight held-out validation rows. All 200 paired feature artifacts pass the
existing identity/token/shape/length/finite-BF16/hash audit. Metadata SHA256:
`8fe77daa4c7b2e630945ac74ea051dadcd6170689d0f1919e44b5ff64b2177ed`.
Histories, system prompts and original whole-conversation splits are
preserved. These are checked automatic accent labels, not independent
human gold. Prompt indices 448 through 511 remain unreviewed and excluded.

Fresh full V37 data/features contain 99,048 unique train / 3,554 validation
and unchanged 2,302 frozen-eval rows. Replay covers 2,697 contextual and
400 existing numeric train rows, factor 64: 294,159 positions / 6,129
steps at batch 48. Source-ID ordering and metadata-hash preflight passes
for active V35, prepared V36 and this newest V37 cache.

Use the newest future launch
`research/v37-saiga-next/training-launch-prepared-combined-first448-memory.json`.
It remains prepared only: 19 native panels / 737 rows, latest V36 final
model-only warmstart and fresh optimizer/schedule. Earlier first384,
first320 and first256 preparations remain immutable. V36's numeric-million
party is still next after V35 final review. No model, trainer, exporter or
normalization implementation changed; demo remains stopped.


## V35 final review and V36 numeric continuation

V35 finished at step 5,664 / epoch 1. Training took 2,496.1709 seconds
(41.60 minutes), with final teacher-forced validation CE 0.0788380.
The final standalone evaluation value comes from `eval_results.json`;
the last checkpoint history contains the earlier step-5,400 CE instead.
The container completed all 17 native panels / 515 rows and exited zero
without OOM.

The native results are mixed. On the fixed context-200 panel, content WER
improved from V34's 4.5747% to 3.6812%, and original-64 from 3.0158% to
2.8787%. The protected five-row demo improved from 36.1702% to 25.5319%,
but still has only one content-exact row. Its protected references lack
complete stress annotations, so stress-exact score is not a valid measure
of accent quality on that panel. Numeric-32 remains poor at 30.1994% versus 29.9145%. Combined-22/34/46
are 12.6005% / 13.1934% / 7.5931%; Saiga-16 is 12.8141%. These panels
show no uniform quality improvement and do not justify demo promotion.

All 62 selected changed/demo source-reference-prediction triples were
read in full, across `final-review-part1` and `final-review-part2`.
Unchanged smaller-panel outputs reuse prior reviews only after exact
prediction and source/reference/history guards. Context-200 and
original-64 are metrics-only here. Long replies still corrupt, omit and
repeat spans; a short smoking followup is now exact. Automatic stress
references remain imperfect, including the already recorded Izuchite
erratum; protected targets were not rewritten from predictions.
Receipts and final model identity are in
`research/v35-context-followup/final-manual-review.json`.

The final root model and trainer state were copied with matching SHA256
to `eval/v35-latest-for-continuation`. Final weights SHA256:
`bc48d22ed7c37a7be7f4881bd68b75dd32f4aa0500317f2027d09f2e329b9fd3`.

V36 was launched on the same GPU 0 as `normalizer-train-v36`, container
`f77918e008b63ae3ee63fdc6f7449cb60266c799b0b51a26367fb36a3e3629ca`.
It uses the exact prepared-memory commands, latest V35 final model-only
warmstart and a fresh optimizer/scheduler. Data contain 98,856 unique
train and 3,546 validation rows; 10,000 new train / 214 new validation
examples expand nominative integers from one million to below one
billion using stock num2words. New rows appear once. Existing contextual
and numeric replay yields 281,871 positions / 5,873 steps, batch 48,
BF16, LR 1e-6, one epoch. The frozen Qwen remains unchanged.

All 18 native-panel output directories were created before launch;
729 generated evaluation rows follow training. The following V37
first448 cohort stays prepared only and must start from V36's reviewed
final weights. No model/trainer/exporter/normalization implementation
changed. Demo remains stopped. Disk free before launch was about
69.8 GiB on drive1 and 150.2 GiB on drive2.


## V36 step 600 and complete V37 source review

V36's step-600 teacher-forced validation CE is 0.0747336. The validation
set now has 3,546 rows, so this number is not directly comparable with
V35's 3,332-row validation. All ten callback identities/references were
guarded against the reviewed V35 final callback. Only the archive/payment
example changed; its complete triple was read and still contains a
corrupted word and repeated span. The other nine predictions match the
previous review exactly. Receipt:
`research/v36-context-followup/step600-callback-manual-review.json`.
A six-panel native review of 155 rows was launched on GPU 1 from a
SHA-verified step-600 copy, strictly for evaluation, never continuation.

All remaining 61 complete first question/actual-Qwen-answer pairs for
prompt indices 448 through 511 were read. First answers at indices
452, 454 and 458 were unavailable/truncated and excluded. Sixteen first
answers passed source review; all sixteen accent targets were read and
retained. Fifteen eligible followups were read with exact generated
history guards. Ten passed source review; all ten accent targets were
read and retained. Dubious claims, malformed wording and nonliteral
normalization outside this cohort were quarantined unchanged. No output
prediction was used to relabel a target.

The entire first512 source selection is now reviewed. The newest cohort
`qwen-context-reviewed-v37-combined-first512` contains 226 records:
217 train / 9 validation. The new held-out first answer inherits its
whole-conversation split assigned before generation. All 226 feature
artifacts pass exact source/reference/token checks, paired finite BF16
shape and source-token lengths, and metadata hashes. Feature audit:
`research/v37-saiga-next/combined-first512-feature-audit.json`.
Automatic accent labels are checked examples, not independent human gold.

Fresh full data/cache contain 99,073 unique train / 3,555 validation and
unchanged 2,302 frozen-eval rows. Contextual/numeric replay covers
2,722 + 400 train rows, factor 64: 295,759 positions / 6,162 steps at
batch 48. Replay ordering and source IDs pass preflight for both live
V36 and the new preparation. Live inputs were not modified.
Newest cache metadata SHA256:
`2fd616fc115648042e086c4a48b4d307b03a9f29ef9081e28a39188fbf87ce59`.

Use the newest future launch
`research/v37-saiga-next/training-launch-prepared-combined-first512-memory.json`.
It remains prepared only: 19 native panels / 738 rows, latest V36 final
model-only warmstart and fresh optimizer/scheduler. Earlier first448 and
smaller preparations stay immutable. The model, trainer, exporter and
normalization source were not changed; demo remains stopped.


The step-600 native evaluation subsequently completed all 155 rows with
exit zero and no OOM. Compared with V35 final, WER is 21.2766% on demo-5,
7.5931% on combined-46, 13.6432% on combined-34, 13.9410% on combined-22,
12.5628% on Saiga-16 and 29.0598% on numeric-32. Results are mixed;
combined-22/34 worsened despite a small numeric-WER improvement.
All 36 selected changed/demo complete source/reference/prediction triples
were read. A style followup is now stress-exact, but quote loss, repeated
spans, corrupt words and wrong stress persist. A phone number is wrongly
expanded as millions, sums still lose magnitudes, and an ordinal date
remains a cardinal number. Protected pizza/MFTI examples are still wrong.
No intermediate checkpoint is promoted or used for continuation.
SHA-bound receipt: `research/v36-context-followup/step600/manual-review.json`.
The live training process passed step 1,110 without OOM. Disk free after
new data preparation/evaluation was about 66.3 GiB on drive1 and 145.4 GiB
on drive2; other workloads remain untouched.


At step 1,200, V36 CE decreased to 0.0744552 on the same expanded
validation set. The one changed archive/payment callback triple was
read in full and still repeats the damaged archive word. All ten outputs
now exactly match the previously reviewed V35 final callback. This is
loss improvement without demonstrated generated-quality improvement in
that callback. Receipt: `step1200-callback-manual-review.json`.


## V36 step 1800 and scored-corpus question preparation

V36's step-1,800 CE is 0.0749663, slightly worse than step 1,200 on the
same expanded validation set. The one changed archive/payment callback
triple was read in full. All ten outputs now match the already reviewed
step-600 callback exactly; the damaged archive word and repeated request
span persist. Receipt: `step1800-callback-manual-review.json`. The same
training container remains live; it was not restarted or rolled back.

Another ready source was downloaded from the pinned
[Saiga scored dataset](https://huggingface.co/datasets/IlyaGusev/saiga_scored/tree/67b92821ebc018ba0e3a0fc0f2b54854c2ab089b).
Its 41,609 records include dialogue scores, language, topic, complexity
and a regex-quality flag. The source Parquet is 114,832,798 bytes, SHA256
`8ba2fecc57f0c85d5d31c5d806541b0eae78b7f45eace089ec885fa0aa7cd982`.
The complete dataset card and schema were read; it is a prompt source
only, with no original assistant responses admitted as targets.

For the future V38 candidate, 512 distinct literal Russian first-user
questions were selected from source dialogues with score at least nine,
no regex-quality flag, and eleven varied topic groups. These metadata
scores describe the original dialogues, not our generated answers.
Prior normalized question overlap is zero. Thirty-six whole-conversation
validation groups were assigned before generation. Prompt source and
selection hashes are in `research/v38-scored-next/source-manifest.json`.

The unchanged CUDA/BF16 frozen-Qwen engine was launched on GPU 1 as
`normalizer-data-v38-gpu1`, container
`bae6f4eafddaf5c554ed3d867b6c55ea0abfe2da20825e261b4af62b17f59f30`.
It generates first and followup answers with actual history and the same
system prompt. Raw generation is not training eligible; full scoped
source/target review and artifact audit are still required. V37's fully
reviewed first512 preparation remains the immediate next party after
V36 final review. No model/trainer/normalization implementation changed.

## V36 step 3000 generated review and V38 first64 preparation

The main V36 container continues without restart or OOM. Callback CE was
0.0744539 at step 2,400 and 0.0742912 at step 3,000. The step-2,400
callback exactly matched the reviewed step-1,800 outputs. At step 3,000,
the one changed archive/payment triple was read completely: a repeated
request span disappeared, but the damaged archive word remained.

The stock step-3,000 native evaluator finished successfully on the same
six fixed panels, 155 rows. All 40 changed predictions plus the unchanged
greeting (41 selected full written/history/reference/baseline/prediction
records) were read against the fully reviewed step-600 baseline. Source,
history, reference and optional-field presence were guarded unchanged.
WER improved on combined22 (13.9410% to 12.8686%), combined34 (13.6432%
to 12.4438%), combined46 (7.5931% to 7.0201%) and Saiga16 (12.5628% to
10.0503%). Numeric32 slightly worsened (29.0598% to 29.3447%); the five
user-demo entries worsened substantially (21.2766% to 30.8511%).
Long-text omissions/repetition, time/MFTI corruption and phone/money/case
errors persist. This is mixed validation, not a successful demo release.
Receipt: `research/v36-context-followup/step3000/manual-review.json`.

A separate unchanged stock evaluator compared V35 final and V36 step
3,000 on 214 fixed new million-range heldouts. Both achieved zero WER and
identical predictions on every row. Twelve representative full triples
were read; the remaining 202 rows are metrics-only. This narrow test has
two short nominative-integer templates. It demonstrates no gain from
the added numeric party and says nothing about dates, phone numbers,
money or grammatical case. Receipt:
`step3000/million-comparison-manual-review.json`.

V38's frozen-Qwen generation completed with 968 complete answers and 56
truncations. All outcomes passed source/actual-history identity checks;
raw answers remain ineligible. The first 64 questions and eligible
first/followup answers were reviewed in full, then all 47 accent targets
were read. Three targets with wrong imperative/surname stress were
quarantined unchanged. The admitted cohort contains 44 rows (40 train,
four whole-conversation heldouts); all 44 cached CUDA/BF16 feature pairs
passed the artifact audit. The remaining questions are not admitted.

An independent existing-label audit read all four matching source/target
pairs. One confirmed wrong writer-surname stress was removed from the
fresh future V38 training cache. Two valid adjective matches were kept.
One heldout-label error is documented separately; protected heldout
bytes were not edited. The live V36 and already prepared V37 inputs
remain immutable.

The future V38 first64 cache has 99,112 unique train rows, 3,559
validation rows and 2,302 fixed evaluation rows. Source-ID replay
preflight passed for live V36, prepared V37 and prepared V38. The stock
train plus 20 native panels (742 rows) is prepared, not launched. V37's
reviewed 226-row first512 cohort remains next after V36 final review;
V38 must then continue from the reviewed V37 final weights with a fresh
optimizer and scheduler. No trainer, model or normalizer source changed.

At step 3,600, CE is 0.0743968. The single changed callback triple was
read fully; its repeated archive/request span returned. Nine other
entries are guarded identical to step 3,000, and all ten now exactly
match the previously reviewed step-600 callback. Receipt:
`step3600-callback-manual-review.json`. Training remains live.

## V38 first128 data review and operational preflight

The next 64 scored-corpus questions produced 53 complete first answers;
all 53 question/actual-Qwen-answer pairs were read in full. Twenty were
retained. Quarantine includes broken grammar, invented film/song facts
and numeric/abbreviation outputs outside this literal-content cohort.
All twenty eligible followup answers were read with their actual first
answer as history; thirteen passed source review. All 33 resulting
accent targets were read, and one unstressed multisyllabic borrowed word
was excluded unchanged. The extra cohort has 32 rows (27 train, five
previously assigned whole-conversation heldouts).

The unchanged CUDA/BF16 exporter completed successfully. All 32 feature
pairs passed source/reference/token-ID, finite-BF16, shape, source-length
and metadata-SHA checks. The unchanged assembler combined these with
the earlier 44 reviewed rows: V38 now has 76 admitted examples, 67 train
and nine validation. No raw or unreviewed answer is admitted. Receipts:
`research/v38-scored-next/first64to128-quality-review.json`,
`first64to128-feature-audit.json` and
`combined-first128-quality-review.json`.

The fresh full future V38 cache contains 99,139 unique train rows,
3,564 validation rows and 2,302 unchanged frozen-evaluation rows. Replay
has 3,189 exact source IDs, factor 64: 300,046 training positions and
6,251 steps at batch 48. The one previously audited bad training label
remains excluded; live V36 and prepared V37 datasets remain unchanged.

An operational review found stale inherited summary counters in older
prepared launch JSONs. Their execution commands already pointed at the
correct caches. Fresh `*-memory-verified.json` launch records reconcile
the summary with authoritative cache/replay counts and stock evaluator
modes: V37 is 226 new rows and 738 native-evaluation rows; V38 is 76 new
rows and 747 native-evaluation rows. No model/trainer code was changed.
`continuation-replay-combined-first128-preflight.json` confirms source-ID
order, metadata/source hashes, counts and evaluation input references.
Both launches are still prepared only; V37 requires reviewed V36 final
weights, and V38 requires reviewed V37 final weights.

V36 callback CE reached 0.0741777 at step 4,200 and 0.0741743 at step
4,800. All ten generated callback entries, including their sources and
references, exactly match the previously reviewed step-3,600 outputs.
The archive/request duplication persists; lower CE is not evidence of
a generated-quality gain. Receipts: `step4200-callback-manual-review.json`
and `step4800-callback-manual-review.json`.

Previously authorized disk housekeeping removed only six optimizer
files from terminal V32/V35 intermediate checkpoints, freeing 18.99 GiB.
Every model file and both final complete optimizers were retained; live
V36 was untouched. Available space on the run disk was 112.37 GiB after
cleanup. Receipt: `old-intermediate-optimizer-cleanup.json`.

## V38 first192 reviewed data; V36 step 5400

All 59 available first question/actual-Qwen-answer pairs from question
indices 128–191 were read completely; 27 passed source review and had
zero normalized full-text collision with preceding data. All 27 eligible
followup pairs were read with their real first-answer history; twenty
passed source review. All 47 automatic accent targets were read. One
wrong accusative-plural stress on “doors” was excluded unchanged after
a [dictionary inflection check](https://gramota.ru/meta/dver); one missing
stress on a multisyllabic content verb was also excluded unchanged.
A suspected “circulate” stress was checked against the dictionary and
retained; suspicion alone was not treated as a confirmed label error.

The retained extra cohort contains 45 rows (39 train, six preassigned
whole-conversation heldouts). The unchanged CUDA/BF16 exporter exited
successfully, and every one of the 45 feature pairs passed the complete
artifact audit. Receipts: `first128to192-quality-review.json` and
`first128to192-feature-audit.json` in `research/v38-scored-next`.

The unchanged assembler produced a fresh combined first192 cohort:
121 admitted source/target pairs, 106 train and 15 validation. The full
future V38 cache now contains 99,178 unique train rows, 3,570 validation
rows and 2,302 unchanged frozen-evaluation rows. Replay has 3,228 exact
source IDs and factor 64, giving 302,542 positions and 6,303 steps at
batch 48. The first128 preparation and all live inputs remain intact.

The newest V38 launch is
`training-launch-prepared-combined-first192-memory-verified.json`:
stock training plus twenty fixed native panels, 753 generated rows.
It remains prepared only and requires reviewed V37 final weights.
V37's immediate next launch remains
`training-launch-prepared-combined-first512-memory-verified.json`,
226 new reviewed examples, requiring reviewed V36 final weights.
`continuation-replay-combined-first192-preflight.json` confirms exact
source-ID order, source/metadata hashes, counts and native input paths
for both launch plans. No model/trainer/normalizer implementation changed.

V36 reached step 5,400 with callback CE 0.07421194. All ten source,
reference and generated-output records exactly match the previously
reviewed step-4,800/3,600 callbacks. The archive/request corruption still
persists. This is a verified unchanged callback, not a quality gain.
Receipt: `step5400-callback-manual-review.json`.

## V36 final validation and V37 continuation launch

V36 finished successfully, no OOM, at step 5,873 and epoch 1.0. Stock
training runtime was 2,565.94 seconds (42.77 minutes), training loss
0.0120230, and standalone final validation CE 0.07419799. The final
ten-example callback exactly matches the previously reviewed step-5,400
callback. The expanded V36 validation set differs from V35's set, so
their raw CE values are not directly compared.

All eighteen stock native panels completed, 729 generated rows, with
source/history/reference identities guarded against V35 final (the new
million214 panel uses its separately recorded V35-final baseline).
Against the fully reviewed V36 step-3,000 critical panels, every changed
prediction plus all five demo entries was read in full: 28 selected
written/history/reference/baseline/prediction records. Other final native
rows are metrics-only; their hashes and metrics are recorded, not
misrepresented as a complete manual read.

Compared with V35 final, WER improved on combined46 (7.5931% to 6.7335%),
combined34 (13.1934% to 12.8936%), combined22 (12.6005% to 12.3324%) and
the five demo entries (25.5319% to 23.4043%). Context200 regressed from
3.6812% to 4.5747%, and original64 from 2.8787% to 3.2214%. Numeric32
remains poor at 29.9145%; the easy million214 panel remains perfect and
unchanged. Time/MFTI corruption, repeated pizza text and severe long-text
omissions persist. This is mixed performance, not a successful demo
release. Receipt: `research/v36-context-followup/final-manual-review.json`.

The immutable model-only continuation snapshot is
`eval/v36-latest-for-continuation`, SHA256
`64c1c3942e863e97ef6926c7961efb2452e7339ee37811d43c5feb877e95592c`.
It matches the final training weights and the independent final-review
snapshot. The final complete optimizer remains saved in the original
run; continuation uses a fresh optimizer and scheduler.

A separate stock GPU-1 evaluation of V36 final on all nine new V37
heldouts completed without OOM. Every full source/history/reference/
prediction triple was read. WER is 2.2472%, seven of nine preserve the
literal text without stress, and two are stress-exact against automatic
references. One response loses a word and repeats a phrase; another
replaces “sell” with “continue”. This is the fixed before-training
baseline, not an already trained V37 result. Receipt:
`research/v37-saiga-next/baseline-v36-final-new9/manual-review.json`.

The verified first512 V37 launch started on the same GPU 0 as
`normalizer-train-v37`, container
`ef7fc11a9a5d6818e3abfc7500a7460012864857e6b609fc6abf59d9b0608709`,
at `2026-10-07T00:59:48.698160843Z`. It continues the reviewed final V36
weights with 226 new reviewed examples, 99,073 unique train rows,
3,555 validation rows and 295,759 replay positions: 6,162 steps, batch
48, BF16, one epoch, LR 1e-6. All nineteen subsequent native panels
(738 rows) are scheduled using unchanged stock commands. Initial finite
loss/gradient entries confirm actual optimizer steps, GPU utilization
99%, and no OOM. The upstream scalar-loss reporting warning remains
unchanged; no model/trainer source was modified. The V38 first192
preparation remains next after reviewed V37 final weights.

## V37 live validation and V38 first256 preparation

V37 continues on GPU 0 from reviewed V36 final, without a restart or
source changes. The step-600 callback CE is 0.07400479. All ten complete
callback records were read; two changed against V36 final: the request
clause repetition disappeared, but the damaged archive word remains,
and the phone-number example still loses a separate five. Step 1,200 CE
is 0.07421462; all ten callback records exactly match the reviewed
step-600 callback. Receipts: `step600-callback-manual-review.json` and
`step1200-callback-manual-review.json` under
`research/v37-context-followup`.

A separate immutable evaluation-only copy of checkpoint 600 was tested
on seven stock native panels, 164 rows, on GPU 1. Source IDs, full source
text, history and references match V36-final baselines. Every changed
output plus all five demo examples was read in full: 49 selected
written/history/reference/previous/prediction records. Other rows are
metrics/hash-only. This intermediate snapshot is never a continuation
source. Snapshot SHA256:
`05b68acefef6021610c204c85d7360499ff32df53a477cc3533d8312b532d3a4`.

Performance remains mixed. Combined34 WER improves from 12.8936% to
10.3448%; new9 from 2.2472% to 1.6854%; demo5 from 23.4043% to 21.2766%.
Saiga16 worsens from 12.5628% to 14.0704%, combined22 WER is unchanged,
and numeric32 remains poor at 29.6296%. The sell/continue substitution,
one career-advice repetition and one film-text repetition were corrected.
The time-only demo restores thirty, but MFTI still becomes MFK and both
pizza replies remain corrupted or repeated. Long neural-network,
toothpaste and artistic passages still lose content. The numeric
100-ruble example becomes ten rubles, and the previously correct
seven-to-ten-day range regresses. No label changes or demo promotion.
Receipt: `research/v37-context-followup/step600/manual-review.json`.

The next 64 V38 source prompts (192..255) were reviewed: all sixty
available first question/answer pairs and all 32 eligible actual-history
followups were read. Stock CUDA RUAccent produced 55 targets; all were
read, with one uncertain proper-name stress target quarantined unchanged.
54 records remain, 48 train and six preassigned whole-conversation
heldouts. All 54 frozen feature artifacts pass the existing full source,
target-token, shape, finite-BF16 and metadata-hash audit. The suspected
`разд+елим` stress was confirmed correct against the indexed official
dictionary conjugation and retained, not incorrectly relabeled.
Receipts: `research/v38-scored-next/first192to256-*.json`.

The combined first256 V38 cohort now contains 175 reviewed examples,
154 train and 21 validation. Fresh stock assemblies preserve all live
inputs and existing heldout references, including the recorded erratum.
Future full cache: `data/freeze-features-v38-combined-first256-qa`,
99,226 unique train rows, 3,576 validation and 2,302 unchanged frozen
evaluation rows. Replay has 3,276 selected rows, factor 64, yielding
305,614 positions and 6,367 steps at batch 48. Its metadata SHA256 is
`aaac7cbf3900ca287c9424421cc9fa8a5d2123eadb9178c48efc849168f5346a`.

The latest prepared launch is
`research/v38-scored-next/training-launch-prepared-combined-first256-memory-verified.json`:
stock training plus twenty native panels, 759 rows. It is prepared only;
reviewed V37 final weights remain required before launch. Replay source
IDs/indices, split disjointness, ordered source concatenation, frozen
evaluation bytes and completed feature metadata pass preflight. No
model/trainer/normalizer implementation was changed. Existing CUDA
accent/export adapters and CPU artifact/assembly operations were reused.

## V37 callbacks through 3000; V38 first320 and contextual numbers

The same V37 training container continues on GPU 0 without a restart.
Step-1800 CE is 0.07402018; two callback outputs changed against 1200:
the fifty-for-sixty substitution and request-clause duplication returned.
All ten source IDs, full written text and references remain identical.
Step-2400 CE is 0.07386696, and step-3000 CE is 0.07396349; both ten-record
callbacks exactly match the reviewed step-1800 output. Lower CE does not
establish better generated text. Callback receipts are saved separately.

The next V38 block contains 53 retained pairs: 47 train and six heldouts
from conversation groups selected before generation. All 52 available
first question/answer pairs and all thirty eligible actual-history
followups were read in full; 55 automatic accent targets were read.
The missing stress on a content noun and an ambiguous imperative stress
were quarantined unchanged. All 53 frozen feature records passed the
existing complete artifact audit. Receipts: `first256to320-*.json`.

All fifty existing Google non-DATE written/spoken pairs at fixed ranks
3501..3550 were read; sixteen passed source review and were accented.
All sixteen targets were read. One double stress marker and one unsupported
genitive-plural stress were excluded unchanged after an indexed
[official dictionary check](https://gramota.ru/meta/sazhen).
The fourteen retained examples cover contextual times, durations,
quantities and inflected numbers. They are train-only because their
source shard already belongs to training. All fourteen feature artifacts
pass the existing audit. These are checked automatic labels, not human
gold or independently verified factual statements. Receipts are under
`research/v38-google-nondate3501to3550`.

The future V38 contextual cohort now has 228 pairs, 201 train and 27
validation; the fourteen numeric-context training pairs bring the new
data total to 242. The unchanged stock assembler produced
`data/freeze-features-v38-first320-google14-qa`: 99,287 unique train rows,
3,582 validation and 2,302 byte-identical frozen-evaluation rows. Live V37
inputs and all quarantined originals remain intact. All 67 newly added
feature records are audited. Replay has 3,337 exact source IDs, factor 64,
309,518 positions and 6,449 steps at batch 48.

The newest prepared launch is
`research/v38-scored-next/training-launch-prepared-first320-google14-memory-verified.json`.
It uses stock training plus twenty fixed native panels, 765 rows, and
requires reviewed V37 final model weights with a fresh optimizer and
scheduler. Preflight verifies ordered concatenation, source-ID replay,
whole-group split disjointness, every feature file, completed metadata,
fixed evaluation bytes and native input paths. Metadata SHA256:
`819bb54c7f892121a1c8e17b9c717e9adf52c5e6d847271af80bee0a339ade82`.
It remains prepared only; no model/trainer/normalizer source changed.

V37 checkpoint 3000 was also copied and hash-verified as an immutable
evaluation-only snapshot. Seven stock panels completed on GPU 1, 164
rows, with full source/history/reference identities guarded against
checkpoint 600. Every changed prediction plus all five demo entries was
read in full: 45 selected complete records; the other 119 are metrics
and hashes only. The first extra-evaluation launch failed before model
loading because I had not prepared its output directories; creating
those directories and rerunning the unchanged evaluator fixed the
operational error. Training was not stopped or restarted.

Against step 600, combined22 WER improves from 12.3324% to 10.1877%,
combined34 from 10.3448% to 9.1454%, numeric32 from 29.6296% to 28.7749%
and demo5 from 21.2766% to 20.2128%. Combined46 worsens from 6.5903% to
6.7335%; new9 and Saiga16 WER are unchanged. The controller followup
recovers its full text, and the second pizza example stops repeating a
clause. However, the time-only example regresses to “eleven whole”,
MFTI remains wrong, the first pizza reply remains corrupted, and one
previously exact film reply gains repetition. One 100-ruble sum is
restored, while 1011 rubles becomes one million eleven. Long-text and
numeric failures persist. No promotion or prediction-driven relabeling.
Receipt: `research/v37-context-followup/step3000/manual-review.json`.

## V38 first384 preparation; V37 continues through 4800

The same V37 container remains live. Callback CE is 0.07385074 at 3600,
0.07395019 at 4200 and 0.07387508 at 4800. All ten complete callback
records at each milestone exactly match the previously reviewed
step-3000/1800 outputs. This verifies unchanged generated behavior,
not a quality gain; the separate selected native review remains at 3000.
Receipts: `step3600/4200/4800-callback-manual-review.json`.

For question indices 320..383, all 59 available first pairs and all 28
eligible actual-history followups were read in full. 51 automatic accent
targets were read; five missing or uncertain contextual stresses were
quarantined unchanged. The 46 retained records, 44 train and two fixed
whole-conversation heldouts, passed every existing feature-artifact audit.
Receipts: `research/v38-scored-next/first320to384-*.json`.

All fifty existing Google non-DATE written/spoken pairs at fixed ranks
3551..3600 were read. Eight passed source review; all eight accent targets
were read, and one unresolved word stress was excluded unchanged without
claiming a confirmed error. Seven train-only pairs remain, including an
inflected six-digit price range, millions, multiplicities and quantities.
The price-range spelling and stress `четырёхсо́т` were checked against the
indexed [official orthographic dictionary](https://gramota.ru/storage/public/normdicts/orfograficheskij_slovar.pdf)
and retained. All seven feature artifacts passed the existing audit.
Automatic normalization/accent labels are not independent gold, and
source factual assertions were not independently verified. Receipts:
`research/v38-google-nondate3551to3600/*.json`.

The latest future V38 cohort has 274 contextual pairs (245 train and 29
validation) plus 21 numeric-context train pairs: 295 new records total.
Fresh stock assembly gives `data/freeze-features-v38-first384-google21-qa`,
99,338 unique train rows, 3,584 validation and 2,302 unchanged frozen eval
rows. Replay has 3,388 exact source IDs, factor 64, 312,782 positions and
6,517 steps at batch 48. All 53 newly appended feature records are audited.
Live V37 inputs and excluded originals remain untouched.

The latest launch is prepared only:
`training-launch-prepared-first384-google21-memory-verified.json` under
`research/v38-scored-next`. It requires reviewed V37 final model weights
and keeps a fresh optimizer/scheduler. Its twenty native panels cover 767
rows. The plan additionally records a stock V37-final baseline command
for all 29 new heldouts, to run before or alongside the V38 start. Preflight
guards exact source-ID replay, ordered concatenation, whole-group split
disjointness, all feature files, COMPLETE metadata, fixed eval bytes and
native/baseline input paths. Metadata SHA256:
`0f945414cdf486f73c0518ab4faf2e215262a964a73b055e696d818390d048ae`.
No model/trainer/normalizer implementation was changed.

## V37 final reviewed; V38 continuation launched

V37 completed step 6162, epoch 1.0, in 2704.4434 seconds. Its container
exited successfully with no OOM after all nineteen native panels, 738
rows. The final model SHA256 is
`42fab89fe5dac08aff6ae4ad3e7e043f341f051180d49e28198878bcd411fc0d`.
The immutable final snapshot and latest continuation snapshot were copied
and hash-verified. Only final model weights are used for continuation.

All 738 native source/history/reference identities were guarded against
the preceding fixed baselines. On seven critical panels, all 21 changed
records against step 3000 plus the five complete demo examples were read
with their histories and references. The other 712 records have metric
and identity checks, not a claim of full manual review. Short hotel text
became exact, while long replies remain damaged and some regressed.
Numeric32 WER is 29.0598%; demo5 WER remains 20.2128%. The time-only demo
still produces “eleven whole”, and MFTI is still wrong. This is a mixed
research checkpoint, not a demo release. Receipts are under
`research/v37-context-followup/final-manual-review.json`. Callbacks 5400
and 6000 exactly match the previously reviewed ten records; their CE is
0.07386626 and 0.07391478 respectively, not evidence of generated gains.

Before V38, all 29 new heldout examples were evaluated on GPU 1 using
the immutable V37 final snapshot and read in full: written text, actual
history, automatic reference and prediction. WER is 7.4695%, literal
exactness 12/29 and automatic stress-target exactness 1/29. Deletions,
repetitions and stress errors persist. Automatic labels are not human
gold. Baseline receipt:
`research/v38-scored-next/baseline-v37-final-new29/manual-review.json`.

V38 is now running as `normalizer-train-v38`, container
`179a774957086948409f6d0efc74494fe20e6f9e4f853ac9eaca7ed98a9c594b`,
on the same reserved GPU 0. It uses the prepared first384/Google21 data:
295 new records, 99,338 unique training rows, 3,584 validation rows,
312,782 replay positions and 6,517 steps. BF16, batch 48, one epoch and
learning rate 1e-6 remain unchanged. Optimizer and scheduler start fresh;
all twenty final native output directories were prepared before launch.
Launch receipt: `research/v38-scored-next/training-launch-first384-google21.json`.
No model/trainer/normalizer implementation changed, and no PR was merged.

The further first384..447 accent staging was read in full, 29 targets.
Three targets with incorrect surname stress or missing multisyllabic
stress were quarantined unchanged. These staged records and pending
followups are not in live V38; they remain future preparation.

## V38 live validation and first448 preparation

The same V38 container remains running. At step 600 callback CE is
0.07374033; all ten complete records were read and identity-guarded.
One phone number restores “sixty”, though a separate five is still lost;
one repeated request clause disappears, but a malformed archive word
remains. At step 1200 CE is 0.07356288 and all ten callback records exactly
match the reviewed step-600 outputs. This is not a generated-quality gain.
Receipts under `research/v38-context-followup`:
`step600-callback-manual-review.json` and `step1200-callback-manual-review.json`.

Checkpoint 600 was copied and hash-verified for evaluation only:
`eval/v38-step600-for-review`, model SHA256
`e930ac3234d7644300fe74b2fb6aae63dcf2fb6bfc504c6bf88efc3f807e24d2`.
Eight stock native panels, 193 rows, completed on GPU 1 with exit 0 and
no OOM. Every changed prediction against V37 final plus all five demo
examples was read in full with written text, history, reference and
previous prediction: 61 records. The other 132 have metrics and identity
checks. All 193 source/history/reference identities remain unchanged.
Receipt: `research/v38-context-followup/step600/manual-review.json`.

Combined22 WER improves 10.9920% to 10.1877%, combined34 9.8951% to
8.6957%, and Saiga16 14.5729% to 12.5628%. Combined46 regresses 6.7335%
to 7.4499%, new9 1.6854% to 5.0562%, and numeric32 29.0598% to 30.4843%.
The new29 panel improves 7.4695% to 7.3171%, literal exactness 12/29 to
14/29; automatic stress-target exactness remains 1/29. The product-designer
answers recover full content; montage repetition disappears. The 7..10
day interval is correct again. A short hotel opening regresses, long
texts remain corrupted/repeated, and time/MFTI/first-pizza demo failures
persist. No demo promotion or label rewriting.

All 29 actual-history followups for first384..447 were also read in full;
eight source pairs were excluded unchanged. All 21 remaining accent
targets were read, and five missing/incorrect stresses excluded. Together
with the already-reviewed first answers, 42 pairs remain: 38 train and
four fixed whole-conversation heldouts. Every feature artifact passed
the existing audit. Receipts: `research/v38-scored-next/first384to448-*.json`.

All fifty Google non-DATE written/spoken pairs at fixed ranks 3601..3650
were read. Fourteen passed source review; all fourteen automatic stress
targets were read, and three missing/unresolved stresses were excluded
unchanged. The eleven remaining train-only records include inflected
quantities and an instrumental three-digit number. Relevant numeral
stresses were checked against indexed normative entries for
[seventy](https://gramota.ru/poisk?mode=slovari&query=Семьдесят) and
[nine hundred](https://ruslang.ru/sites/default/files/doc/normativnyje_slovari/tolkovyj_slovar_chast1_A-N.pdf).
Every feature artifact passed the existing audit. Automatic labels are
not human gold; source factual assertions were not independently checked.
Receipts: `research/v38-google-nondate3601to3650/*.json`.

The unchanged stock assembler produced a fresh future cache:
`data/freeze-features-v38-first448-google32-qa`, 99,387 unique train,
3,588 validation and 2,302 byte-identical frozen-evaluation rows. All
53 newly appended records are audited. Exact source-ID replay has 3,437
rows, factor 64, 315,918 positions and 6,582 batch-48 steps. Metadata SHA256:
`40674ad1b9d15cf8e729e123640652689f5dbc04a3822208e30b0ef95db6ded9`.

The future V39 launch is prepared only:
`research/v38-scored-next/training-launch-prepared-first448-google32-memory-verified.json`.
It requires reviewed V38 final model weights, a fresh optimizer/scheduler,
and the recorded stock baseline on all 33 new heldouts. Twenty final
native panels cover 771 rows. Preflight verifies ordered concatenation,
all feature files, COMPLETE metadata, exact replay IDs, unchanged frozen
evaluation bytes and native input paths. Live V38 inputs remain unchanged;
no model/trainer/normalizer implementation was changed.

## V38 continuation monitoring and further reviewed data

The same training container continues without restart. Callback CE at
steps 1800 and 2400 is 0.07395233 and 0.07379425. All ten complete callback
records exactly match the previously reviewed step-600/1200 outputs;
there is no demonstrated generated gain in this small panel. Receipts:
`research/v38-context-followup/step1800-callback-manual-review.json` and
`step2400-callback-manual-review.json`.

All sixty available first answers in the final preselected source block,
indices 448..511, were read with their actual histories. Twenty-one passed
source review and the existing stock CUDA accent tool. All twenty-one
written/reference pairs were then read. Four uncertain or incorrect stress
targets were quarantined unchanged, leaving seventeen training pairs.
The fixed holdout assignment was preserved; no retained group belongs to
validation and no empty validation file was introduced. Source assertions
are not independently verified factual gold, and automatic accent targets
are not human gold. Receipts: `research/v38-scored-next/first448to512-*.json`.

The stock frozen-Qwen exporter and existing artifact audit completed on
all seventeen pairs: exact source IDs, source text, target token IDs,
finite paired BF16 features, token lengths and metadata hashes. The stock
assembler produced `data/freeze-features-v38-first512-firstonly-google32-qa`:
99,404 unique training rows, 3,588 validation rows and the same 2,302 frozen
evaluation rows. It adds 66 train and four heldout records to live V38.
Live V38 inputs and the earlier prepared caches remain unchanged.

The further V39 plan remains prepared only, requires reviewed V38 final
weights, and starts a fresh optimizer/scheduler. Its source-ID replay
contains 3,454 unique rows, factor 64, 317,006 positions and 6,605 batch-48
steps. All replay indices and source IDs are checked against the new
ordered dataset. The thirty-three fixed new heldouts and final native
panels remain unchanged. Prepared launch:
`research/v38-scored-next/training-launch-prepared-first512-firstonly-google32-memory-verified.json`.
Metadata SHA256:
`2d47d5827a36b36100b940d725e715b63f7af777695e26c73bba80cf46b6aa80`.
No model, trainer or normalizer implementation changed; no PR was merged.

Checkpoint 3000 was copied and double-hash-verified for evaluation only,
SHA256 `25e13219ffca07a37ed2e111e8fca13adc99d406f7db5903ee88218b343b6300`.
All eight stock native panels, 193 rows, completed on GPU 1 with exit 0
and no OOM. All source/history/reference identities were guarded against
step 600. Every changed prediction plus all five demo examples was read
in full, 57 records; the remaining 136 have metric and identity checks.
Receipt: `research/v38-context-followup/step3000/manual-review.json`.

Against step 600, Saiga16 WER improves 12.5628% to 10.8040% and combined46
7.4499% to 7.1633%; combined22 regresses 10.1877% to 11.2601%, combined34
8.6957% to 10.9445%, and new29 7.3171% to 8.3841%. Numeric32 remains
30.4843%, demo5 20.2128%. The 666666-rouble amount becomes exact, but the
7..10-day interval regresses to 7..17 and three million is still wrong.
Some repeated clauses disappear, others return; long replies still lose
words or end early. MFTI, time-only 11:30 and first-pizza demo failures
persist. Callback CE at 3000 is 0.07392206; all ten callback records exactly
match reviewed step 2400. This remains a mixed intermediate checkpoint,
with no demo promotion and no prediction-driven reference changes.

## Further contextual data, monitoring and isolated learning-rate comparison

V38 remains the same active run. All ten callback identities at steps
3600, 4200 and 4800 were guarded; the one changed phone-number prediction
was read in full each time. It alternates between fifty and sixty while
still omitting another five. The other nine records exactly match the
previously reviewed outputs. CE is 0.07398248, 0.07399379 and 0.07391466,
respectively. These small fluctuations do not establish better generated
quality. Receipts: `research/v38-context-followup/step*-callback-manual-review.json`.

All twenty-one actual-history followups in source indices 448..511 were
read; five malformed or uncertain source replies were excluded unchanged.
All sixteen remaining automatic stress targets were read. Three incorrect
or missing stresses were quarantined unchanged, leaving thirteen train
pairs in the fixed pre-generation conversation split. Their artifacts
passed the existing complete feature audit. No heldout was reassigned.

The unchanged stock assembler produced another versioned cache,
`data/freeze-features-v38-first512-google32-qa`: 99,417 unique train,
3,588 validation and the same 2,302 frozen-evaluation rows. This appends
79 train and four heldout pairs to live V38. All 83 appended artifacts
are audited. Replay has 3,467 unique source IDs, factor 64, 317,838
positions and 6,622 batch-48 steps. Metadata SHA256:
`6462a05a7aae9ddd42583bee066cf56204046da11c122d37e1efbcbbd36a95d8`.
Future V39 remains prepared only and requires reviewed V38 final weights.
Prepared launch:
`research/v38-scored-next/training-launch-prepared-first512-google32-memory-verified.json`.
All launch inputs, replay IDs and indices, ordered dataset bytes, feature
files and completion metadata passed preflight. Labels are checked
automatic targets, not independent human/factual gold.

Six intermediate `optimizer.pt` files from terminal V36/V37 runs were
removed, 18.9876 GiB of logical file size, after checking successful
container termination, final epoch/step, final model hashes and absence
of live consumers of those runs. Every model and final optimizer/state
was preserved; active V38 was not touched. Receipt:
`research/v38-scored-next/v36-v37-intermediate-optimizer-cleanup.json`.

Because generated validation remains mixed, a separate stock-trainer
recipe comparison completed on reserved GPU 1:
`normalizer-train-v38-lr3x-probe600`. It uses the same reviewed V37 final,
live V38 dataset/replay, batch 48 and default seed, with LR 3e-6 instead
of 1e-6 for 600 steps. The stock scheduler receives 196 warmup steps and
cosine `num_cycles=0.031956968834045245`, matching the original 6,517-step
cosine fraction over these first 600 steps while changing amplitude only.
Eight fixed native panels, 193 rows, completed after training. Receipt:
`research/v38-lr3x-probe600/launch.json`. This is an isolated comparison,
not continuation or a demo release. Main V38 and prepared V39 recipe
remain unchanged until evidence supports a different setting. No model,
trainer or normalizer implementation changed.

The probe exited successfully without OOM. All 193 source/history/reference
identities were checked. Sixty selected tuples were reviewed: 39 previously
unseen complete written/history/reference/prediction tuples were read, and
21 exactly match tuples already read in prior manual reviews. All five
demo examples were additionally read in full. The other 133 rows have
metric and identity checks. Five panel WERs regress, two remain equal and
one improves versus the main run at step 600. All 120 logged learning-rate
values match the intended 3x ratio. This short, single-seed comparison does
not support changing the main recipe: LR stays 1e-6. Probe weights are
evaluation only. Receipt: `research/v38-lr3x-probe600/manual-review.json`.

## V38 final review and V39 continuation

Another fifty non-DATE Google numeric source pairs, ranked 3651..3700,
were read in full. Thirteen source pairs passed review; all thirteen
automatic accent targets were read. Four targets with missing or unresolved
stresses were quarantined unchanged. Nine retained training pairs passed
the existing feature audit. Receipts:
`research/v38-google-nondate3651to3700/{source-manual-review,target-manual-review,feature-audit}.json`.
These are checked automatic labels, not independent human/factual gold.

The stock assembler produced the latest immutable dataset/cache pair,
`data/{hidden-plan,freeze-features}-v38-first512-google41-qa`: 99,426 unique
train, 3,588 validation and the same 2,302 frozen evaluation rows. It adds
88 train and four heldout records to V38; all 92 appended artifacts were
audited. Replay contains 3,476 unique source IDs, factor 64, giving 318,414
positions and 6,634 batch-48 steps. Ordered source bytes, replay IDs/indices,
metadata and completion hashes passed preflight. Metadata SHA256:
`90d2e601a5deef1f413960585d1dfa2c5067529a80f3d0a7682f4cbaa92100a4`.

V38 completed step 6,517, epoch 1, with train runtime 2,849.7273 seconds,
validation CE 0.07394230, successful container exit and no OOM. All twenty
native panels completed, 767 rows; every source/history/reference identity
was guarded against the prior fixed baseline. In eight critical panels,
193 rows were compared against step 3000. All changed predictions plus
all five demo examples were read in full, 34 records. The other 159
critical records and 574 other-panel records have metric/identity checks.
Receipt: `research/v38-context-followup/final/manual-review.json`.

Final WER is 11.5282% on combined22, 9.4453% on combined34, 7.3066% on
combined46, 30.7692% on numeric32, 4.4944% on the V37-new9 panel,
7.0352% on Saiga16, 7.6220% on new29 and 20.2128% on demo5. Some long
reply clauses are restored, while other replies remain malformed or
repetitive. The 666666-rouble amount regresses from its exact intermediate
output. MFTI, 11:30 and first-pizza failures persist. Final quality is mixed;
there is no demo promotion and no prediction-driven reference rewriting.

Final root and final-checkpoint weights match; both immutable evaluation
and continuation copies were independently hash-verified. SHA256:
`4c9380a1e2ef3ea5e2aedd3776541ea125352272cffa268f96cf889ea382ad37`.
V39 was launched from those reviewed final weights on the same GPU 0,
with fresh optimizer/scheduler, BF16, batch 48, one epoch and LR 1e-6.
Its existing trainer and model implementation are unchanged. All twenty
final native output directories were created before launch. Receipt:
`research/v38-scored-next/training-launch-first512-google41.json`.

The fixed 33-new-heldout baseline completed on GPU 1 from the reviewed
V38 final. All 33 source/history/reference values were checked; 29 output
tuples exactly match the reviewed V38-final new29 panel, including prior
exact manual-review reuse. The four additional complete tuples were read:
both long wood-processing replies corrupt words, while the two gratitude
replies preserve their literal words. Baseline WER is 7.7551%, literal
exact match 17/33 and stressed exact match 2/33. Receipt:
`research/v38-scored-next/baseline-v38-final-new33/manual-review.json`.
No model, trainer or normalizer implementation changed; no PR was merged.

## V39 monitoring and next reviewed data

V39 remains running on its original GPU and immutable inputs. Callback
source/reference identities passed at steps 600, 1200, 1800, 2400, 3000
and 3600. Validation CE at 3000 is 0.07371009 and at 3600 is 0.07372542.
The phone example still drops a separate digit and oscillates between
fifty and sixty. Changed complete tuples were read; unchanged tuples
reuse exact prior reviews. Receipts are under
`research/v39-context-followup/step*-callback-manual-review.json`.

Native validation completed on GPU 1 at steps 600 and 3000, 197 fixed
examples each, with successful container exits and no OOM. Every
source/history/reference identity was checked. All changed predictions
plus all five demo examples were read in full: 50 selected tuples at
600 and 55 at 3000. The other 147 and 142 records respectively have
metric/identity checks. Step-3000 versus step-600 WER improves on two
panels and regresses on six. Numeric WER is 30.4843%, demo WER 20.2128%,
and the fixed new33 WER 7.4830%. Range seven-to-ten and one freelance
reply recover; 666666, million amounts, long clauses and repetitions
remain problematic. Intermediate snapshots are evaluation only; no
demo promotion or prediction-driven target rewriting occurred.
Receipts: `research/v39-context-followup/{step600,step3000}/manual-review.json`.

Another fifty Google non-DATE numeric pairs, ranks 3701..3750, were
read in full. Sixteen source pairs and all sixteen accent targets were
reviewed; eleven training pairs remain after quarantining unresolved
stress targets unchanged. All eleven exported artifacts passed audit.
Receipts: `research/v39-google-nondate3701to3750/`.

From cached SaigaScored, 512 unseen Russian questions across eleven
topics were selected. Thirty-six whole conversation groups were fixed
as heldout before generation. The unchanged frozen-Qwen generator
produced 957 complete answers and rejected 67 incomplete outputs;
all 1024 actual system/history records passed integrity checks.
Saiga assistant answers are not used as targets. Of the first 64
questions, 61 complete first answers were read; 36 source pairs and
all 36 accent targets were reviewed. Two incorrect targets were
quarantined unchanged, leaving 33 train and one fixed heldout pair.
All 34 frozen-feature artifacts passed audit. Automatic accent labels
remain checked automatic labels, not independent human/factual gold.
Unreviewed raw outputs are excluded from training.
Receipts: `research/v39-scored-next/first64-*`.

The stock assembler prepared a fresh dataset/cache,
`data/{hidden-plan,freeze-features}-v39-first64-google52-qa`, with
99,470 unique training pairs, 3,589 validation pairs and the unchanged
2,302 frozen evaluation rows. This adds 44 train and one validation
pair to live V39. Ordered bytes, whole-group separation, source hashes,
feature counts and completion hashes passed. Replay has 3,520 unique
IDs, factor 64, giving 321,230 positions and 6,693 batch-48 steps.
Metadata SHA256:
`0e27aed2fa3daee676a821c4abffea72b56b42ab5128785a290eccdca6bdcab0`.

V40 is prepared only, requiring reviewed V39 final weights with fresh
optimizer/scheduler. Its existing recipe remains BF16, batch 48,
one epoch and LR 1e-6. Twenty-one final native panels contain 772
examples; fixed33 and the new fixed1 require final-weight baselines.
Preflight: `research/v39-scored-next/continuation-replay-first64-google52-preflight.json`.
No live cache, model, trainer or normalizer implementation changed.
No PR was merged.

## V39 continued review and first128 preparation

Live V39 inputs and recipe remain unchanged. Callback identities passed
at steps 4200, 4800 and 5400; all ten complete tuples exactly match their
reviewed predecessor. Validation CE is 0.07386687, 0.07383141 and
0.07383610 respectively. Receipts:
`research/v39-context-followup/step{4200,4800,5400}-callback-manual-review.json`.

Thirty-six complete followup replies from the previously reviewed first64
sources were read with their actual histories. Six malformed, mixed-language
or unresolved-abbreviation answers were excluded unchanged. All thirty
stock accent targets were read; two targets with wrong or missing stresses
were quarantined, leaving 27 train and one fixed heldout reply. All 28
exported feature records passed the existing artifact audit. Receipts:
`research/v39-scored-next/first64-followup-*`.

For question indices 64..127, 58 complete first answers were read; six
incomplete answers were unavailable. Thirty-four sources were retained
and all 34 automatic accent targets were read in full. This contributes
33 train and one preselected whole-group heldout pair. All 34 feature
records passed audit. Sources with malformed grammar, unresolved written
abbreviations, digits lacking spoken targets or unsupported specific
biographies were excluded unchanged. Actual Qwen system/history values
are preserved; source-dataset bot answers remain unused. Receipts:
`research/v39-scored-next/first64to128-*`.

The existing assembler also built the combined fresh context cohort:
`data/{qwen-context-reviewed,freeze-qwen-context-reviewed}-v39-combined-first128`,
93 train and three heldouts. Native evaluation of all three heldouts from
the immutable V39-step3000 snapshot completed with exit 0 and no OOM.
Every source/history/reference identity was checked and all three complete
tuples read. Normalized WER is 7.6923% overall; one AI followup has WER 0
but differs in yo spelling, while the initial AI reply collapses words and
the appetite reply repeats a clause. Literal/stressed exact match remains
0/3. This tiny diagnostic cannot establish generalization and cannot
replace the required reviewed-final baseline. Receipt:
`research/v39-scored-next/baseline-v39-step3000-first128-new3/manual-review.json`.

Fifty additional non-DATE Google numeric pairs, ranks 3751..3800, were
read in full. Fourteen sources and all fourteen automatic accent targets
were reviewed. Six unresolved proper-name, district or specialized-term
stress targets were quarantined unchanged, leaving eight training pairs.
All eight exported records passed audit. Receipts:
`research/v39-google-nondate3751to3800/`.

The latest immutable full dataset/cache is
`data/{hidden-plan,freeze-features}-v39-first128-google60-qa`:
99,538 unique train, 3,591 validation and the unchanged 2,302 frozen
evaluation rows. Relative to live V39, it adds 112 train and three
validation records; all 115 appended artifacts were audited. Ordered
bytes across all five appendices, completion/source hashes and whole-group
separation passed. Replay has 3,588 unique IDs, factor 64, giving 325,582
positions and 6,783 batch-48 steps. Metadata SHA256:
`a02c5a59e950ec22b2185f4f064d28b64d31e10a7666e66aee55b9c59c961276`.

V40 remains prepared only, requiring reviewed V39 final weights. The
unchanged BF16/batch48/LR1e-6/one-epoch recipe uses fresh optimizer and
scheduler. Twenty-one native panels total 774 examples; fixed33 plus
fresh fixed3 require final-weight baselines before continuation.
Latest preflight:
`research/v39-scored-next/continuation-replay-first128-google60-preflight.json`.
Automatic labels remain reviewed automatic labels, not independent
human/factual gold. No model, trainer or normalizer implementation changed;
no live inputs were mutated and no PR was merged.

## V39 final review and additional contextual followups

V39 completed 6,634 steps and one epoch in 2,921.3169 seconds (48.7
minutes). Training throughput was 2.271 steps/s. The step6000 and
step6600 validation CE values were 0.07389706 and 0.07386593. All callback
source/reference identities passed; each changed phone-number prediction
was read in full. The nine unchanged tuples exactly match their prior
reviewed values. The phone example still omits a separate five and
alternates between fifty and sixty. Final native checks are recorded
separately; low teacher-forced CE does not establish correct generation.

Thirty-three complete followup replies from the reviewed questions
64..127 were read with actual three-turn histories. Eight malformed or
unresolved-abbreviation sources were excluded unchanged. All 25 retained
automatic accent targets were read; two lacked a stress mark on a
multisyllabic word and were quarantined unchanged. The remaining 22 train
and one preselected heldout records passed the existing feature audit.
The standalone word `авто́` was checked against the
[Gramota dictionary](https://gramota.ru/meta/avto), preserving that
correct automatic label. Receipts:
`research/v39-scored-next/first64to128-followup-*`.

The stock assembler produced a fresh immutable full dataset/cache:
`data/{hidden-plan,freeze-features}-v39-first128full-google60-qa`,
99,560 unique train, 3,592 validation and 2,302 unchanged frozen rows.
Relative to live V39, all 134 added train and four added validation
records were manually reviewed and feature-audited. Exact ordered
appendix bytes, source/completion hashes and fixed whole-group splits
passed. The combined fresh context cohort has 115 train and four
heldouts. Replay has 3,610 unique IDs, factor64, giving 326,990 training
positions and 6,813 batch48 steps. Metadata SHA256:
`4cd5012f88cddd4c6df8eaa6cf9a683d961a53b80c2947c6dcacaafebca67931`.

The prepared V40 recipe retains BF16, batch48, LR1e-6 and one epoch with
a fresh optimizer/scheduler. It includes 21 final native panels with
775 rows. Final V39 fixed33 and fresh fixed4 baselines are required
before launch. Latest preflight:
`research/v39-scored-next/continuation-replay-first128full-google60-preflight.json`.
No model, trainer or normalizer implementation was changed; reviewed
automatic targets remain distinct from independent human/factual gold.

V39 and its 20 native panels completed with exit0 and no OOM. All 771
source/history/reference identities and panel metrics were checked against
the fixed V38-final baselines. Panel WER improved in nine, worsened in
six and was unchanged in five. Against V39 step3000, the eight critical
panels improved in four, worsened in two and were unchanged in two.
All 29 selected complete tuples were read: every changed critical
prediction and all five demo examples. The other 168 critical and 574
remaining native rows received metric/identity checks, not a claim of
full manual review. Long-text corruption and numeric failures persist;
the demo panel still has 20.2128% WER. Receipt:
`research/v39-context-followup/final/manual-review.json`.

Final-weight baselines also completed on the separate factory GPU. The
fixed33 result exactly matches the corresponding final native tuples;
its two changed predictions were read in the critical review. All four
fresh heldout tuples were read in full after identity checks. Their
normalized WER is 8.4746%, with one AI followup preserving content but
differing in yo spelling; initial AI and appetite examples still collapse
or repeat words. Four samples cannot establish generalization.

Final root and checkpoint6634 weights matched, and both immutable
review/continuation copies were verified by SHA256:
`1bda6e0e78cd295db8be90b17bacf1ac9dc25cbd2859bc810a2de06be4eb1c22`.
The continuation copy contains model weights and provenance; optimizer
and scheduler are fresh. V40 was launched on the same fixed training
GPU as `normalizer-train-v40`, container
`1f99730eb6f1f124c5dd4441a09d0cf95d6c2c4b8550f80f440ec72cd2e024f0`.
Actual `run_inputs.json` confirms the new feature cache, reviewed V39
warm start and null resume. First optimizer steps were observed with no
OOM. Launch receipt:
`research/v39-scored-next/training-launch-first128full-google60.json`.
No demo was promoted and no PR was merged.

## V40 intermediate review and prepared V41 data

V40 remains running on its fixed training GPU, without OOM. Callback
validation CE is 0.07380804 at step600, 0.07385915 at step1200 and
0.07378000 at step1800. All ten callback identities match. Step1200
exactly reuses the previously reviewed step600 tuples. The single changed
step1800 tuple was read in full: the payment sentence loses one repeat,
but archive-word corruption remains. Callback receipts are under
`research/v40-context-followup/step*-callback-manual-review.json`.

The independent step600 native evaluation completed with exit0 and no
OOM: nine panels, 201 rows. All source/history/reference identities passed
against reviewed V39-final baselines. Five panel WER values improved,
two worsened and two tied. All 52 selected tuples were read in full,
including every changed prediction, all five demo examples and all four
fresh heldouts. The other 149 rows received identity/metric checks only.
The rain followup is exact and the devices example restores keyboard and
controller, but the long pizza reply adds a repeat. Demo WER worsened
from 20.2128% to 23.4043%; numeric WER worsened from 30.1994% to 30.4843%.
This intermediate snapshot is evaluation-only. Receipt:
`research/v40-context-followup/step600/manual-review.json`.

The next source slice contains 55 complete actual frozen-Qwen first
answers for questions128..191; all were read with their actual histories.
Twenty-nine source rows were excluded unchanged. All 26 retained accent
targets were read; five double-marked, unmarked or contextually wrong
targets were quarantined unchanged. The remaining 19 train and two
preselected heldout records passed the stock feature audit. Fixed groups
were retained, not reassigned after filtering. Receipts:
`research/v40-scored-next/first128to192-*`.

All 50 Google non-DATE candidates at deterministic ranks3801..3850 were
read. Fourteen source pairs survived; all automatic accent targets were
read and three unresolved or unmarked targets were quarantined unchanged.
The remaining 11 train pairs passed the stock feature audit. No targets
were corrected from model predictions. Receipts:
`research/v40-google-nondate3801to3850/{source-manual-review,target-manual-review,feature-audit}.json`.
Checked automatic labels remain distinct from independent human gold.

The existing assembler produced an immutable next full dataset/cache:
`data/{hidden-plan,freeze-features}-v40-first192-google71-qa`,
99,590 unique train, 3,594 validation and 2,302 unchanged frozen rows.
Relative to live V40, it adds 30 train and two validation records.
Completion/source hashes, ordered appendices and exact replay ID lookup
passed. Metadata SHA256:
`3d48022e4769cf431e2fc596828c427981a357c5a43f9e9b303d0b9e3a8524c8`.
Replay contains 3,640 unique IDs (3,169 contextual and 471 numeric),
factor64: 328,910 positions and 6,853 batch48 steps. The fresh context
cohort now has 134 train and six heldouts.

V41 is prepared only. It retains BF16/batch48/LR1e-6/one epoch, model-only
warm start from reviewed V40 final and fresh optimizer/scheduler.
Twenty-one final native panels total 777 rows. Fixed33 plus all six
fresh heldouts require final-weight baselines before continuation.
Preflight:
`research/v40-scored-next/continuation-replay-first192-google71-preflight.json`.

An operational provenance error was found in V40's scheduled additional
four-row panel: that command points to the preceding V39 warm-start
weights. Its scheduled output must not be reported as a V40-final result.
The other 20 final panels point to the V40 run. After V40 finishes, run
the stock six-row baseline separately on verified V40-final weights and
read all six tuples before continuing. All 21 prepared V41 panels were
checked to point to the future V41 final run, not its warm start. Note:
`research/v40-scored-next/v40-additional-panel-provenance-note.json`.
No model, trainer or normalizer implementation changed, no live job or
inputs were modified, and no demo or PR was promoted.

## Additional V40 contextual followups

Callbacks at steps 2400 and 3000 preserve all ten source/reference identities. The
two changed tuples at each checkpoint were read in full; the other eight
exactly reuse the preceding reviewed tuples. Validation CE is 0.07435153
and 0.07399499 respectively. The phone example still loses a separate
five and alternates between fifty and sixty; the payment sentence still
corrupts the archive word. These are intermediate observations, not proof
of final improvement. Receipts:
`research/v40-context-followup/step{2400,3000}-callback-manual-review.json`.

All 21 followups to the accepted questions 128..191 were read with their
actual three-turn histories and system prompts. Six malformed or
unverified sources were quarantined unchanged. All 15 retained automatic
accent targets were read. Four were excluded unchanged: two omitted
multisyllabic stress, one stressed the plural games incorrectly, and one
changed the warmth noun to an inappropriate yo form. The plural was
checked against the [Gramota dictionary](https://gramota.ru/poisk?mode=slovari&query=игра).
Both rejected preselected heldout followups remain quarantined, without
reassignment. The remaining 11 train-only records passed the existing
feature audit, with exact source/target IDs and finite BF16 paired 2560
features. Receipts:
`research/v40-scored-next/first128to192-followup-*`.

The stock assembler produced a newer immutable next dataset/cache:
`data/{hidden-plan,freeze-features}-v40-first192full-google71-qa`,
99,601 unique train, 3,594 validation and 2,302 unchanged frozen rows.
Relative to live V40, there are 41 additional train and two validation
records. Ordered byte concatenation, completion/source hashes and exact
replay-ID lookup passed. Metadata SHA256:
`7b76ae0f96aca02c8ce2276f496ea8fee69fd9de49778861d55d16baea247de8`.
Replay now has 3,651 unique IDs (3,180 contextual, 471 numeric), factor64:
329,614 positions and 6,867 batch48 steps. The fresh context cohort has
145 train and six heldouts; its validation is unchanged.

This preparation supersedes the preceding prepared V41 plan; both caches
remain immutable. Current pointer:
`research/v40-scored-next/LATEST_PREPARED.json`.
V41 is still prepared only and requires reviewed V40-final weights plus
fixed33 and six-row baselines before launch. All 21 final evaluation
commands point to the future V41 run. No live restart, model/trainer
implementation change, target rewrite, demo promotion or merge occurred.

The step3000 model was independently copied with matching double SHA256
to the evaluation-only snapshot `eval/v40-step3000-for-review`:
`d69a4900691e1240dce6af5f418503546400a8abe813673e5fef6bb227254a77`.
A stock nine-panel, 201-row native evaluation was launched on the factory
GPU, separately from the live trainer:
`research/v40-context-followup/step3000/native-launch.json`.
Its final metric comparison and manual review are recorded separately.

The nine-panel step3000 evaluation completed with exit0 and no OOM. All
201 source/history/reference identities match the reviewed step600
baseline. All 55 selected tuples were read in full: every changed
prediction, all five demo examples and all four fresh heldouts. The other
146 rows received identity/metric checks only. Relative to step600, four
panel WER values improved, three worsened and two tied; relative to
reviewed V39 final, seven improved and two worsened. The short hotel and
education answers now reproduce their references exactly. Initial AI
innovation wording and the neural-network ending were restored, while
long-answer corruption, repeats and severe numeric errors persist. Demo
WER remains 23.4043%; numeric WER is 29.9145%. No token limit was reached.
The intermediate snapshot is neither continuation nor demo eligible.
Receipt: `research/v40-context-followup/step3000/manual-review.json`.

Step3600 validation CE is 0.07407371. All ten complete callback tuples
exactly match the reviewed step3000 values; that full review is explicitly
reused. V40 was verified live beyond 54% of its epoch, with no OOM.
Receipt: `research/v40-context-followup/step3600-callback-manual-review.json`.

## V40 steps4200/4800 and next first-answer slice

Validation CE is 0.07400176 at step4200 and 0.07398513 at step4800.
All ten source/written/reference identities match the preceding reviewed
callbacks. The one changed tuple at4200 and two at4800 were read in full;
unchanged complete tuples reuse the preceding review. The phone still
omits a separate five and alternates sixty/fifty. The payment archive
corruption persists. These intermediate snapshots are evaluation-only.
Receipts: `research/v40-context-followup/step{4200,4800}-callback-manual-review.json`.

For questions192..255, all54 available complete first answers were read
with their actual history and system. Twenty-seven malformed, unexpanded
or factually uncertain sources were quarantined unchanged. All27 retained
automatic accent targets were read; four with missing multisyllabic
stress were excluded unchanged. Limited doubtful forms were checked
against primary references: Internet-post stress permits variation in
[Gramota's answer307633](https://gramota.ru/poisk?mode=spravka&query=пост+цензора&simple=0),
and the stressed vowel of drained matches
[Academos](https://orfo.ruslang.ru/abc/part/de?end=40026&start=39726).
These checks do not make the entire automatic target set human gold.
The original heldout assignment was preserved; all23 surviving records
are train-owned. Their stock export/audit finished exit0 without OOM,
checking every record's IDs, exact target tokens, finite BF16 paired2560
features and source hashes. Receipts: `research/v40-scored-next/first192to256-*`.

The stock assembler finished exit0 without OOM. The new immutable full
dataset/cache is `data/{hidden-plan,freeze-features}-v40-first256-google71-qa`:
99,624 train, 3,594 validation and 2,302 unchanged frozen evaluation rows.
Ordered appendices, validation bytes, frozen rows and completion/source
hashes passed. Metadata SHA256:
`c2befaee844a9bbd5986c039f4f011241174d18c58aeb5fffb048939d28b3ce9`.
The fresh context cohort contains168 train and the same six heldouts.
Replay contains3,674 unique IDs:3,203 contextual and471 numeric; factor64
gives331,086 positions and6,898 batch48 steps. Relative to live V40,
64 train and two validation records are added.

`research/v40-scored-next/LATEST_PREPARED.json` now points to the
first256 V41 preparation. All21 final native commands point to future
V41 weights. Launch remains conditional on reviewed V40-final weights
and the fixed33 plus six-row baselines; no V41 job has been launched.
Earlier caches/plans remain immutable. V40 was verified live beyond76%
of its epoch. No model/trainer implementation, live inputs, demo or PR
approval state was changed.

## V40 late callbacks and additional followups/quantities

Steps 5400 and 6000 have validation CE 0.07396645 and 0.07400539.
All ten source/written/reference identities match. The changed phone
tuple was read in full at each checkpoint; the other nine complete
tuples reuse their prior review. Sixty is restored at5400 and fifty
returns at6000, while the separate five remains missing. Archive-word
corruption is unchanged. These snapshots remain evaluation-only.
Receipts: `research/v40-context-followup/step{5400,6000}-callback-manual-review.json`.

All 23 followups to the accepted first-answer slice were read with their
actual three-turn histories and system prompts. Four questionable or
malformed sources were quarantined unchanged. All 19 automatic targets
were read against the same sources; four omitted multisyllabic stress
and were excluded unchanged. The remaining 15 train-owned records
passed the stock feature audit. No heldout group was reassigned.
Receipts: `research/v40-scored-next/first192to256-followup-*`.

All 50 quantity records at non-DATE ranks3851..3900 were read in full.
The ranking is SHA256 of source ID, verified against all 50 preceding
rank identities. Twelve normalization pairs were retained unchanged;
wrong cases, unresolved abbreviations, ambiguous year readings and
damaged fragments were excluded. All 12 accent targets were read and
passed content/stress guards. Doubtful forms were checked against primary
references: [Gramota](https://gramota.ru/poisk?mode=slovari&page=9&query=шестьсот+шестьдесят+шесть)
confirms the marked vowel in the oblique sixty form, while the
[orthographic dictionary](https://gramota.ru/storage/public/normdicts/orfograficheskij_slovar.pdf)
permits the retained plural genitive of sazhen. No targets were rewritten.
Stock export/audit checked all 12 records and completed exit0 without OOM.
Receipts: `research/v40-google-nondate3851to3900/*`. These automatic
Kestrel/RUAccent labels are not independent human or factual gold.

Two stock assemblies completed exit0 without OOM. The latest immutable
full dataset/cache is `data/{hidden-plan,freeze-features}-v40-first256full-google83-qa`:
99,651 train, 3,594 validation and 2,302 unchanged frozen rows. Metadata:
`80fb2d826586b7434018e25f2992cb56ae94fa23edc8accdf7747cfab410dc3d`.
Ordered appendices, unchanged validation/frozen bytes, completion hashes
and replay-ID mapping passed. Replay has 3,701 IDs: 3,218 contextual and
483 numeric. Factor64 gives 332,814 positions and 6,934 batch48 steps.
The fresh contextual cohort has 183 train and the same six heldouts.
Relative to live V40, 91 train and two validation records are added.
The latest prepared pointer selects this first256full/google83 plan.
V41 remains unlaunched pending reviewed V40-final weights and fixed
baselines. No model/trainer implementation or live inputs changed.

## Reviewed V40 final and V41 continuation

V40 completed all 6,813 training steps and the scheduled native evaluations
with container exit0 and no OOM. Training took3,008.889 seconds,2.264
steps/s; train CE was0.00928299 and final validation CE0.07400079.
The step6600 and final6813 callbacks were read against their preceding
reviews: one changed complete tuple each, with the other nine reused
exactly. The separate phone five and archive corruption persist.

Root final weights, checkpoint6813 and the independent evaluation copy
match SHA256:
`5f2aafe65d3d2624badc2e59e61daf93a8a909e853a5051b133817bc89d89eee`.
All20 final panels point to these weights and preserve all771
source/history/reference identities. Relative to reviewed V39 final,
eight panel WER values improve, four worsen and eight tie. Against the
reviewed step3000 critical panels, four improve, three worsen and one
ties. All31 selected complete tuples were read: every changed critical
prediction plus all five user-demo examples. The other166 critical rows
and574 remaining native rows received identity/metric checks only.

The second pizza reply loses its repeat; the first pizza remains
corrupt. MFTI and11:30 are still wrong. Charger/Pantheon wording improves,
while the neural-network, education and own-day endings regress.
Long-answer corruption, numeric grouping errors and repeats persist.
Demo WER is20.2128%; protected numeric WER is30.1994%. This mixed result
permits the authorized research continuation, with no demo promotion.
The scheduled extra four-row panel uses V39 weights and is explicitly
excluded from V40-final evidence. Receipts:
`research/v40-context-followup/final/{all20-comparison,critical-comparison,manual-review,final-snapshot}.json`.

An independent stock evaluation of verified final weights completed
exit0 without OOM on the factory GPU: fixed33 plus all six fresh context
heldouts. Fixed33 complete records exactly match the reviewed final
native panel. All six fresh tuples were read with full history/system
and reference: the diploma wording is preserved, the short AI reply has
a yo-only difference, and Montessori/appetite/long-AI outputs remain
corrupt. Fresh-six WER is9.1429%; it is not directly compared with the
older four-row cohort. Receipts:
`research/v40-scored-next/final-baseline-first256full-{launch,manual-review}.json`.

Only after these reviews was the separate model-only continuation copy
created and its SHA256 checked again. V41 is running on the same reserved
GPU from that reviewed V40 final, using stock Trainer, BF16 batch48,
LR1e-6 and one epoch with fresh optimizer/scheduler. Startup confirms
439,416,570 trainable parameters,99,651 unique train,3,594 validation,
332,814 replay positions and6,934 expected steps. All21 future final
panels point to V41 weights and cover777 rows. Receipts:
`research/v40-scored-next/training-launch-first256full-google83.json` and
`v41-startup-first256full-google83-verified.json`.

For continued disk headroom, nine inspected intermediate optimizer files
from terminal successful V38/V39/V40 runs were removed, freeing28.48GiB.
All model files and each final optimizer state were verified unchanged;
active V41 was untouched. Drive2 then had60GiB free. Bounded-cleanup
receipt: `research/v40-scored-next/v38-v40-intermediate-optimizer-cleanup.json`.
No model/trainer implementation, targets, demo or merge state changed.

## V41 intermediate review and next immutable data

V41 remains running from reviewed V40-final weights. The step600,1200 and
1800 callbacks have validation CE0.07440100,0.07441442 and0.07414754.
All ten callback identities were checked; the step600 changed tuple and
all ten step1200/1800 written/reference/predictions were read in full.
The complete step1800 tuples equal step1200 exactly. The separate phone five
and archive-word corruption persist. Receipts:
`research/v41-context-followup/step{600,1200,1800}-callback-manual-review.json`.

Independent stock native evaluation of step600 completed exit0 without
OOM on the factory GPU. All nine panel manifests point to snapshot SHA256
`2ac49c26c964085d832578ba9948d617d0e69022580fa4f2fba652e5a5a9d6b9`.
All203 identities match the reviewed V40-final panels/baseline. All53
selected tuples were read in full, including every changed prediction,
all five demo examples and all six fresh contextual heldouts. The other
150 rows received identity/metric checks only. Receipt:
`research/v41-context-followup/step600/{comparison,manual-review}.json`.

Fresh-six WER improves from9.1429% to5.1429%: appetite repetition is
removed and transport wording returns, though the AI compound word
becomes corrupt. Older critical panels regress: demo WER20.2128% to
24.4681%, numeric WER30.1994% to31.3390%. Neural-network and game-device
wording are severely damaged, time and large-number errors persist,
and the second pizza reply now repeats and loses preferences. Some
shorter wording/stress improves. This is mixed research evidence, not
a claim of overall improvement; step600 is neither continuation nor
demo eligible. Automatic targets are not independent human/factual gold.
The same203-row stock native evaluation is now running on the factory
GPU against an independent step1800 snapshot, SHA256
`88868867d2928aed9bcb983ed757e288c1568cf0e31a6264830e082ee0c998ec`.
The first evaluation attempt failed before producing predictions because
its output directories were missing. The nine directories were created
and only the evaluation was relaunched; main training was untouched.
The corrected launch is recorded in
`research/v41-context-followup/step1800/native-launch-directories-fixed.json`;
its generated predictions have not yet been manually reviewed.

All56 available first-answer sources in prompt slice256..319 were read
with actual history/system. Eight missing/incomplete prompt outputs
were excluded. Thirty-one sources were retained unchanged; all31
automatic accent targets were then read in full. Seven targets with
missing or malformed stress were quarantined unchanged. The remaining
24 records comprise20 train and four originally heldout groups. No
heldout group was reassigned and no prediction-driven target was edited.
Source/target receipts and the hash-bound GO review are under
`research/v41-scored-next/first256to320-*`.

Stock feature export/audit completed exit0 without OOM, checking all24
records for exact identity/target IDs, finite BF16 paired2560-dimensional
features, token lengths and metadata/source hashes. Four stock assembly
commands completed exit0: the new full dataset/cache is
`data/{hidden-plan,freeze-features}-v41-first320-google83-qa`, with99,671
train,3,598 validation and2,302 byte-identical frozen rows. The fresh
contextual cohort has203 train and ten heldouts. Ordered source appendices
were checked; live V41 inputs remain unchanged.

V42 is prepared only, pending reviewed V41-final weights and stock fixed33
plus fresh10 baselines. Replay has3,721 IDs, including3,238 contextual and
483 numeric; factor64 yields334,094 positions and6,961 batch48 steps.
All21 future final panels point to V42-final weights and cover781 rows.
The future run and continuation-weight directory were absent at preflight.
Latest prepared pointer: `research/v41-scored-next/LATEST_PREPARED.json`.
No model/trainer implementation, demo or merge state changed.

## Reviewed V41 step1800 and prepared first320 followups

The corrected step1800 native evaluation completed exit0 without OOM.
All nine panels use the independently copied step1800 weights and all203
source/history/system/reference identities match reviewed step600.
All56 selected complete tuples were read in full: every changed output,
all five user-demo examples and all six fresh context heldouts. The other
147 rows received identity/metric checks only. Receipt:
`research/v41-context-followup/step1800/{comparison,manual-review}.json`.

Against step600, six panels improve WER, two worsen and one ties.
Numeric WER falls31.3390% to29.0598%; the666666-ruble sentence is now
exact. The second pizza restores preferences but repeats a phrase.
Charger wording, short rain, the movie answer and the Ruslan joke improve.
MFTI/time/large-number/percentage errors persist. The long story now ends
early; neural-network, art, sport and knitting wording remain corrupt.
Fresh-six WER regresses5.1429% to8%. These mixed intermediate results
are neither continuation nor demo eligible.

Step2400 validation CE is0.07426338. All ten complete callback tuples
were read in full and equal step1800 exactly; the phone omission and
archive corruption persist. Receipt:
`research/v41-context-followup/step2400-callback-manual-review.json`.
Live V41 continues with the original dataset/cache and training recipe.

All31 actual frozen-Qwen followups to the reviewed first-answer sources
were read with their three-turn histories and system prompt. Five
unresolved, malformed or context-inconsistent outputs were quarantined
unchanged. All26 remaining automatic accent targets were then read in
full; three with omitted multisyllabic stress were excluded unchanged.
The retained23 comprise19 train and four originally heldout records.
No heldout group was reassigned and no source/target was rewritten.
Receipts: `research/v41-scored-next/first256to320-followup-*`.

Stock export/audit completed exit0 without OOM and checked all23 records
for identities, target IDs, finite paired BF16 features, source lengths
and hashes. Four stock assembly commands also completed exit0. The latest
immutable full dataset/cache is
`data/{hidden-plan,freeze-features}-v41-first320full-google83-qa`, with
99,690 train,3,602 validation and2,302 byte-identical frozen rows.
Metadata SHA256:
`33cb8a61d42394cf8f763fc1e2249423c4dd3586bc0540a995b2f3677178d18e`.
The fresh contextual cohort has222 train and14 heldouts. Automatic
Qwen/RUAccent labels remain checked pseudo-labels, not independent gold.

This supersedes the prior unlaunched V42 preparation. Replay now contains
3,740 IDs:3,257 contextual and483 numeric; factor64 gives335,310 positions
and6,986 batch48 steps. All21 future native panels point to V42-final
weights and cover785 rows. V42 still waits for reviewed V41-final weights
and stock fixed33 plus fresh14 baselines. Its run and continuation-copy
directories were absent at preflight; output directories must be created
before invoking the stock evaluator. The latest prepared pointer remains
`research/v41-scored-next/LATEST_PREPARED.json`. No model/trainer code,
live input, demo or merge state changed.

## V41 continues; reviewed callbacks and next numeric appendix

V41 remains live without restart. All ten complete callback tuples at
steps3000,3600 and4200 were read; fixed source/reference identities match.
Validation CE is0.07456820,0.07416057 and0.07416787 respectively. The phone
sequence still loses digits and the archive sentence remains corrupt.
This does not demonstrate an overall quality gain. Callback receipts:
`research/v41-context-followup/step{3000,3600,4200}-callback-manual-review.json`.

An independent step4200 evaluation was launched on the factory GPU for
nine critical panels /203 rows. Its model-only snapshot was double-hashed:
`62b9815ca0747e8d8495ece664126784172aa011ae7ff7b54cdef17db44f5b1b`.
All output directories were created before the stock evaluator. This is
evaluation only; no intermediate weights qualify for continuation or demo.
Receipt: `research/v41-context-followup/step4200/native-launch.json`.

All50 Google non-DATE source pairs at ranks3901–3950 were read in full;
16 were retained unchanged before accenting. All16 automatic accent
targets were then read in full. Thirteen were retained unchanged; one
omitted a multisyllabic stress and two unresolved pronunciation checks
were quarantined. The retained `сем+идесяти` agrees with the primary
[Gramota dictionary entry](https://gramota.ru/poisk?mode=slovari&query=%D0%A1%D0%B5%D0%BC%D1%8C%D0%B4%D0%B5%D1%81%D1%8F%D1%82).
No target was rewritten. These are reviewed automatic labels, not human
or factual gold. Receipts: `research/v41-google-nondate3901to3950/`.

Stock export and all13 tensor audits completed exit0 without OOM. Stock
assembly completed exit0; ordered appendices and unchanged heldout bytes
were checked. The next immutable dataset/cache is
`data/{hidden-plan,freeze-features}-v41-first320full-google96-qa`:
99,703 train,3,602 validation and2,302 frozen rows. Metadata SHA256:
`317c9b30c3838c99bc2a052df2b3388f5bc045c3979844efa60c984e656c0736`.
Replay has3,753 IDs (3,257 contextual /496 numeric); factor64 yields
336,142 positions /7,003 batch48 steps. Relative to live V41,52 train
and eight validation records were added. Fresh contextual heldouts remain14.

This supersedes only the prior unlaunched V42 preparation. All21 future
native panels cover785 rows and point to V42-final weights. V42 still
requires reviewed V41-final weights and stock fixed33/fresh14 baselines;
its run and continuation directory were absent at preflight. Pointer:
`research/v41-scored-next/LATEST_PREPARED.json`. No model/trainer code,
live dataset, demo or merge state changed.

## Reviewed step4200 generation and prepared first352 context data

Step4200 independent evaluation completed exit0 without OOM. All203
source/history/system/reference identities match step1800 and every
panel manifest binds the copied step4200 weight hash. All53 selected
complete tuples were read: every changed output plus all five user-demo
and six fresh-context examples. The other150 received identity/metric
checks only. Receipts: `research/v41-context-followup/step4200/`.

Against step1800, three panels improve WER and six worsen. The second
pizza loses its repeated clause; `Никто ещё не пообедал` and the product
designer wording are restored. First-pizza and MFTI/time failures remain.
Numeric WER rises29.0598% to31.0541%:22,000 becomes220,000 and666666
loses its previously correct reading. Fresh-six WER rises8% to9.7143%.
Long neural-network/story/toothpaste/knitting corruption persists.
This intermediate checkpoint is not continuation/demo eligible.

All ten complete step4800 callback tuples were read and are identical
to step4200; validation CE is0.07414708. The main V41 job remains live.
Receipt: `research/v41-context-followup/step4800-callback-manual-review.json`.

For prompt indices320–351,29 complete actual frozen-Qwen first answers
were read with their system/user history; three sources were missing or
incomplete. Ten were retained before accenting. All ten followups to
those sources were also read with actual three-turn histories; seven
were retained before accenting. Unresolved factual assertions, malformed
answers and context problems were excluded unchanged. All17 resulting
accent targets were read; three omitted multisyllabic stress and were
excluded unchanged. The14 retained records comprise13 train and one
originally heldout source. The deliberate falsehood example retains its
original user instruction in the heldout history; it is not factual gold.
Receipts: `research/v41-scored-next/first320to352*`.

Stock CUDA export and all14 paired-tensor audits completed exit0 without
OOM. Four stock assembly commands completed exit0. Ordered appendices
and frozen bytes were checked. The latest full dataset/cache is
`data/{hidden-plan,freeze-features}-v41-first352full-google96-qa`:
99,716 train,3,603 validation and2,302 frozen rows. Metadata SHA256:
`c2089229074478d4fd541fff8b3e2f7bb69d4cdb04d61809f9020526274604db`.
The fresh contextual cohort now has235 train and15 heldouts. Targets
remain reviewed automatic labels, not independent human/factual gold.

The unlaunched V42 preparation now selects3,766 replay IDs (3,270
contextual /496 numeric), factor64,336,974 positions and7,021 batch48
steps. This adds65 train/nine validation records relative to live V41.
All21 future native panels point to V42-final weights and cover786 rows;
the fresh panel and baseline use all15 fresh heldouts. V42 still requires
reviewed V41-final weights and fixed33/fresh15 baselines. Its run and
warm-start directory were absent at preflight. Latest prepared pointer:
`research/v41-scored-next/LATEST_PREPARED.json`. No model/trainer code,
live inputs, demo or merge state changed.

## First384 contextual preparation and step5400 callback

V41 remains live without restart. All ten complete step5400 callback
tuples were read and identity-matched against step4800. Validation CE is
0.07420519. Only the phone output changes: `пятьсот` returns, but the
standalone five is missing and sixty becomes fifty. The other nine,
including the corrupt archive word, are unchanged. Receipt:
`research/v41-context-followup/step5400-callback-manual-review.json`.

For indices352–383,27 complete actual frozen-Qwen first answers were
read in full with system/history; five were missing/incomplete. Eleven
were retained before accenting. All11 followups to those sources were
also read with their actual three-turn histories; seven were retained.
Malformed text, invented prior history, unreliable facts and context
loss were excluded unchanged. All18 accent targets were then read;
three omitted multisyllabic stress and one had unresolved pronunciation.
The14 retained records comprise12 train and two originally heldout
first answers. No source/target was rewritten or heldout reassigned.
Receipts: `research/v41-scored-next/first352to384*`.

Stock CUDA export and all14 tensor audits completed exit0 without OOM.
Four stock assembly commands completed exit0. Ordered appendices and
frozen bytes were checked. Latest immutable full dataset/cache:
`data/{hidden-plan,freeze-features}-v41-first384full-google96-qa`, with
99,728 train,3,605 validation and2,302 frozen rows. Metadata SHA256:
`5d80da1ed0cf600418e6ea1702ac428f9f5e3655a431f4d9c3b5948d7780d94b`.
Fresh contextual cohort:247 train /17 heldouts. These remain reviewed
automatic labels, not independent human/factual gold.

Prepared V42 replay has3,778 IDs (3,282 contextual /496 numeric); factor64
gives337,742 positions and7,037 batch48 steps. Relative to live V41,
77 train and11 validation records were added. All21 future native panels
cover788 rows and point to V42-final weights; the fresh baseline/panel
use all17 fresh heldouts. This supersedes the unlaunched first352 plan.
V42 still requires reviewed V41-final weights and stock fixed33/fresh17
baselines. Its run and continuation directory were absent at preflight.
Latest prepared pointer: `research/v41-scored-next/LATEST_PREPARED.json`.
No model/trainer code, live input, demo or merge state changed.

## First448 preparation and V41 end-of-training callback

The first384–415 sources and followups were read in full. Seventeen
unchanged accented targets survived review, all in existing train groups.
The first416–447 sources and followups were also read in full; 24 targets
survived, comprising23 train and one originally heldout source. Missing
multisyllabic stress, malformed wording and uncertain content were
excluded unchanged. These remain reviewed automatic labels, not human
gold. Receipts: `research/v41-scored-next/first384to416*` and
`research/v41-scored-next/first416to448*`.

Both stock CUDA exports and all41 paired-tensor audits passed; both sets
of four stock assembly commands completed exit0 without OOM. Immutable
full data/features: `data/{hidden-plan,freeze-features}-v41-first448full-google96-qa`:
99,768 train /3,606 validation /2,302 frozen rows. Metadata SHA256:
`cc235da0f80c9023094efd974fb3298c4f81250ed5cbb9228f4b06edce171c50`.
Ordered appendices and frozen bytes were verified. The new contextual
cohort has287 train and18 heldouts. Live V41 inputs are unchanged.

The next V42 preparation selects3,818 replay IDs (3,322 contextual and496
numeric), factor64:340,302 positions and7,090 batch48 steps. Relative to
live V41, this adds117 train and12 validation examples. Its21 native
panels cover789 rows and bind future V42-final weights. Fixed33/fresh18
baseline commands require reviewed V41-final weights; output directories
must exist before stock evaluation. Latest prepared pointer:
`research/v41-scored-next/LATEST_PREPARED.json`.

V41 completed6,934 optimizer steps, one epoch, in3,047.58 seconds
(50.79 minutes;2.275 steps/s). Train CE is0.00788682, final validation CE
0.07412665. All ten complete final callback tuples were manually read and
are identical to step6000: the phone still drops a five and changes sixty
to fifty, and the archive word still repeats. Native final generation
evaluation remains in progress at this preparation point. This is not
evidence of broad quality improvement or approval for demo promotion.
No model/trainer implementation changed.

## Reviewed V41 final and launched V42

V41's Docker job terminated exit0 without OOM after21 native panels
completed777 rows. Root final, checkpoint6934 and model-only review copy
agree on SHA256:
`02caa75eecbacd9409118f9471da31fb1ed00b813596e2567b2175636cca812f`.
All21 panel manifests bind that final weight hash. The20 common panels
identity-match771 V40-final rows: seven improve WER, nine worsen and four
tie. V41 is not a demonstrated overall improvement and is not promoted.

Critical comparison against step4200 covers203 identical tuples. All38
selected complete tuples were read: every changed critical prediction,
all five user demos and all six fresh heldouts. Other native rows received
identity/metric checks only. Five critical panels improve, two worsen
and two tie. The1917 date is restored; long neural/story endings partly
recover, but content corruption persists. Second-pizza repetition returns;
demo WER20.2128% to23.4043%. Numeric WER31.0541% to30.4843% remains high.
Receipts: `research/v41-context-followup/final/`.

After review, a model-only continuation copy was made from final weights.
Stock CUDA fixed33/fresh18 baselines completed exit0 without OOM and bind
the same hash. Fixed33 exactly reuses the final native tuples; it was not
entirely reread. All18 complete fresh tuples were read with their histories
and source/reference/prediction context. Fresh WER is7.4699%, half match
the reference without stress; wrong stress, omissions and long-text
corruption remain. This small automatic-reference panel does not certify
general quality. No target was changed after seeing its prediction.

V42 was launched on the same GPU0 with reviewed V41-final weights, a fresh
optimizer/scheduler and the prepared first448 data/replay configuration.
Docker: `normalizer-train-v42`; run: `freeze-text-v42-context-followup`.
Launch receipt: `research/v41-scored-next/training-launch-first448full-google96.json`.
All21 native output directories were created before launch. The stock
training/evaluation implementation, frozen agent Qwen, and demo remain
unchanged. This is continued research training, not approval to deploy.

## V42 step600 and prepared numeric appendix

V42 is live and was not restarted. All ten complete step600 callback
tuples were manually read and source/reference-matched against V41-final.
Only the phone changes: sixty replaces the wrong fifty, but a standalone
five is still missing. The other nine predictions, including archive
duplication and Power BI stress, are unchanged. Validation CE is0.07451852.
Receipt: `research/v42-context-followup/step600-callback-manual-review.json`.

An independent stock evaluation was launched on GPU1 for nine critical
panels /215 rows, including all18 fresh contextual heldouts. The immutable
step600 model-only copy is evaluation-only, SHA256:
`d2461cdb02e934c55e7bee67b0f8b016d27adb83a51ffe3c89369b59466ad305`.
Docker: `normalizer-eval-v42-step600`; launch receipt:
`research/v42-context-followup/step600/native-launch.json`.
The main training continues on GPU0; the evaluation has not yet been
reviewed and does not qualify this checkpoint for continuation or demo.

All50 complete Google nonDATE written/spoken pairs at ranks3951–4000 were
read. Twelve quantity/model-code pairs were retained unchanged after
train ownership and collision checks; incorrect cases/year readings,
fragments, raw abbreviations and uncertain proper names were excluded.
All12 stock accent targets were read. One omitted stress on multisyllabic
`или` and was excluded unchanged. All11 remaining paired BF16 feature
artifacts passed the stock CUDA export and tensor/hash audit. No label
was revised from a model prediction. These are checked automatic labels,
not independent human/factual gold.

The stock assembled next cache is
`data/{hidden-plan,freeze-features}-v42-first448full-google107-qa`:
99,779 train /3,606 validation /2,302 frozen rows, metadata SHA256:
`61428e4b8e6a5214993fe100af3c1bf6ae4b36cc2337d452ecceabd7a64b04a0`.
Ordered train appendix and unchanged validation/frozen bytes were checked.
Prepared V43 replay has3,829 IDs (3,322 contextual /507 numeric), factor64:
341,006 positions /7,105 batch48 steps. All21 future native panels cover789
rows and bind future V43-final weights. V43 remains unlaunched and requires
reviewed V42-final weights and fixed33/fresh18 baselines. Pointer:
`research/v42-scored-next/LATEST_PREPARED.json`.
No model/trainer code or live input changed.

## V42 validation review and next contextual appendix

The independent step600 evaluation completed exit0 without OOM. All215
tuples identity-match their V41-final baselines and bind the evaluation
snapshot hash recorded above. All64 selected critical tuples were
reviewed:54 complete tuples reread and10 exact unchanged fresh tuples
explicitly reused from their previous full-context review. The other151
rows received identity/metric checks only. Four of nine panels improve
WER and five worsen. The game-device list recovers, but numeric and
long-text corruption persists; the fresh panel has regressions as well
as recoveries. Receipt: `research/v42-context-followup/step600/manual-review.json`.
This intermediate snapshot is not promoted or used for continuation.

All ten complete callback tuples at steps1200,1800 and2400 were read.
Validation CE is0.07437962,0.07468115 and0.07462338 respectively.
The phone alternates between fifty/sixty and drops words; at2400 it also
drops five hundred. Archive duplication and Power BI stress persist.
These small callback panels do not show a general quality improvement.
Receipts: `research/v42-context-followup/step{1200,1800,2400}-callback-manual-review.json`.
The main V42 job remains live and has not been restarted.

The first448–480 source slice had30 complete first answers; prompt451 and
478 were incomplete and excluded. All30 source contexts and16 actual
follow-up contexts were read. Sixteen first answers and12 follow-ups
passed source screening. All28 automatic accent targets were then read;
seven with missing multisyllabic stress were excluded unchanged.
The original fixed whole-conversation split yields20 train and1 heldout.
No source/target was rewritten, and original Saiga bot answers remain
unused. These are checked automatic labels, not independent human gold.
All21 stock CUDA-exported artifacts passed the paired-tensor/hash audit.

Four stock assembly commands completed exit0 without OOM. Next cache:
`data/{hidden-plan,freeze-features}-v42-first480full-google107-qa`:
99,799 train /3,607 validation /2,302 frozen rows. Metadata SHA256:
`7d46e6a034e562834d40fab0df52146d99aa79e90af252202838391123a69526`.
Ordered train/validation appendices and unchanged frozen bytes were
verified. Relative to live V42, this adds31 train and1 validation example,
including the previously prepared11 numeric examples. Replay has3,849
IDs (3,342 contextual /507 numeric), factor64:342,286 positions /7,131
batch48 steps. The fresh contextual cohort has307 train and19 heldouts.
All21 future native panels cover790 rows and bind future V43-final weights.
`research/v42-scored-next/LATEST_PREPARED.json` now points to this preparation.
V43 remains unlaunched pending reviewed V42-final and fixed33/fresh19
baselines. Live inputs and model/trainer implementation remain unchanged.

After verifying V41's terminal exit0 state, only its inspected intermediate
optimizer files at5400/6000/6600 were removed, freeing9.49GiB. All V41
model files and the final6934 optimizer retain their original inode,
size and modification time; live V42 was untouched. Exact optimizer
resume from those three intermediate snapshots is no longer available.
Receipt: `research/v42-scored-next/v41-intermediate-optimizer-cleanup.json`.

## V42 step3000 generation review and remaining source slice

All ten complete callback tuples at3000 and3600 were manually read.
Validation CE is0.07461700 and0.07471382. At3000 the phone restores five
hundred and sixty, but still omits the separate five. At3600 only the
optional monosyllabic stress on `за` changes; archive duplication persists.
Receipts: `research/v42-context-followup/step{3000,3600}-callback-manual-review.json`.

The independent stock GPU1 evaluation of step3000 completed exit0 without
OOM: nine panels /215 rows, all source/history/reference identities
matched against step600 and V41-final. The evaluation-only model copy
matches the checkpoint before/after copying, SHA256:
`efa0ac68c2ca2fd81b46aec7594296f01b3902171e1052ea7de9bab8a6f2af77`.
All nine panel manifests bind this hash. All67 selected tuples were
reviewed:50 complete tuples reread, seven exact V41-final tuples reused
from recorded full reads, and ten unchanged fresh tuples reused from the
recorded step600 review. The remaining148 received identity/metric checks
only. Receipt: `research/v42-context-followup/step3000/manual-review.json`.

Against V41-final, five panels improve WER, one worsens and three tie;
against step600, six improve, two worsen and one ties. Self-analysis
recovers exactly and much of the coffee/neural-network content recovers,
but the game-device list and product-design follow-up regress. User-demo
WER returns to23.4043%; numeric WER30.1994% remains high. Pizza, MFTI/time,
large quantities and long-text repetitions remain unsuitable for demo
promotion. No reference was changed from a prediction.

The remaining first480–512 slice had27 complete first answers; prompt482,
483,489,500 and511 were incomplete. All27 sources and all seven follow-up
contexts of the accepted parents were read. Seven first answers and seven
follow-ups passed source screening and collision checks. All14 accent
targets were read; wrong stress on `Кол+я` and missing multisyllabic stress
were excluded unchanged, leaving eight training-only rows. Original
whole-conversation heldouts were retained, with no accepted heldout in
this slice. All eight stock CUDA-exported paired artifacts passed audit.

Four stock assembly commands completed exit0 without OOM. Latest next
cache: `data/{hidden-plan,freeze-features}-v42-first512full-google107-qa`,
99,807 train /3,607 validation /2,302 frozen rows, metadata SHA256:
`f98eedd9d441b4bcdba3ee5e41f258a54b5331a3250a3e1548f489f1296f20b0`.
Ordered training appendices and unchanged validation/frozen bytes were
verified relative to first480. Compared with live V42 this adds39 train
and1 heldout. Replay has3,857 IDs (3,350 contextual /507 numeric), factor64:
342,798 positions /7,142 batch48 steps. The fresh cohort has315 train and19
heldouts; future21 native panels cover790 rows. The latest prepared pointer
now names first512full. V43 remains unlaunched pending reviewed V42-final
and fixed33/fresh19 baselines. Live training, its inputs, the original
Freeze-Omni implementation and demo remain unchanged.
