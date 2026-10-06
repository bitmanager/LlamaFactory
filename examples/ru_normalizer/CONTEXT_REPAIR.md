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
