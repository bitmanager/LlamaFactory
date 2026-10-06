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
