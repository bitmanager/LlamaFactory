# Russian hidden-to-spoken-text pilot

This experiment predicts a spoken Russian form (normalization and `+` stress
marks) from **answer hidden states**, without synthesizing audio yet. It uses
the same Qwen3-4B-Instruct-2507 base as the existing ASR/agent experiment.

The frozen source Qwen reads the original answer with its system prompt and,
when supplied, real conversation history. Only answer-token hidden states
after the final norm enter a LayerNorm + Linear projector. Its output prefixes
a second Qwen3-4B-Instruct-2507 decoder with PEFT LoRA. Training uses native
causal cross-entropy, `transformers.Trainer`, and native Qwen generation.
Only projector and decoder LoRA parameters update. The source never sees the
normalized target. This is a text-only adaptation inspired by Freeze-Omni's
hidden-conditioned decoder, **not a reproduction of its speech training**.

## Reused implementation

- Qwen models, tokenizer, CE, SDPA, collation, Trainer, optimizer, scheduler,
  checkpoint/resume and decoding: Transformers.
- LoRA: PEFT, rank 16, alpha 32, all linear layers.
- Dataset loading: Hugging Face Datasets.
- Local code: model interface/prefix adapter and existing-label format adapter.

This example lives in the LLaMA-Factory fork but invokes **HF Trainer directly**;
the standard LLaMA-Factory SFT command does not natively implement this
two-model hidden-prefix input. No upstream model or training loop is patched.

Verified environment: Python 3.12, PyTorch 2.9.1+cu130, Transformers 5.3.0,
PEFT 0.18.1, Datasets 4.0.0, PyArrow 25.0.0, Accelerate and TensorBoard.
The example uses a `PreTrainedModel` composition so native safetensors saving
understands Qwen's tied embedding/output weights. Checkpoints currently include
both frozen models, not just the trainable adapter; allow about 16 GB each.

## Source compatibility

Base revision: `Qwen/Qwen3-4B-Instruct-2507` at
`cdbee75f17c01a7cc42f958dc650907174af0554`.
Pass the ASR branch's tokenizer via `--agent_tokenizer`: that branch uses a
151670-entry resized vocabulary, including its audio placeholder. Both models
resize to that exact tokenizer; common text rows are preserved. The source
base is frozen in the ASR experiment too. This pilot does not load its audio
projector or ASR head.

Integration must extract the **same post-token, final-normalized hidden state**.
The state predicting a token, before consuming it, is a different interface.
Compatibility of weights does not establish equality of text-only and
audio-conditioned hidden distributions; test the latter before production.

## Data

JSONL fields: `written`, `spoken_stressed`, `source_id`; optional `history`
(user/assistant chat messages) and `system`. `+` precedes the stressed vowel.
The decoder only receives projected hidden states and a fixed task prompt.
Targets are passed as teacher-forcing labels. Overlength examples raise an
error; there is no silent truncation.

`prepare_local.py --root LOCAL_DATA_ROOT --output NEW_OUTPUT_DIRECTORY` exports
existing campaign/Gemini labels, excludes existing evaluation text aliases,
splits by template/scenario groups, rejects malformed stress and conflicting
targets, and records provenance hashes and rejection reasons. No new text or
stress labels are generated.

Prepared pilot: 30,173 train, 1,493 validation, 2,302 separate existing evaluation
examples. Most training pairs are stress-only; only 1,975 training pairs use
Gemini written-to-voiced text. Labels have automatic stress, not human gold.
History is absent from these sources and is not fabricated. Template grouping
does not prove original recording/speaker independence. The numeric held-out
set tests a skill poorly covered by the initial training mix.

## Run and validate

On an explicitly reserved GPU, run:

```bash
CUDA_VISIBLE_DEVICES=GPU_UUID bash examples/ru_normalizer/train_hidden.sh /path/to/run-root
```

The run root contains `venv`, `models/qwen3-4b-instruct-2507`,
`models/agent-tokenizer`, and `data/hidden-plan-v1/{train,validation}.jsonl`.
Defaults: BF16, SDPA, batch 16, accumulation 2, one epoch, LR 1e-4,
cosine schedule, 3% warmup. Logs and metrics use stock Trainer/TensorBoard.
Resume by adding `--resume_from_checkpoint /path/to/checkpoint-N`.

```bash
python -m pytest -q examples/ru_normalizer/test_hidden_plan.py
```

Tests cover frozen/trainable gradients, padded-vs-individual logits, native
greedy generation, and actual Trainer checkpoint/optimizer resume with tied
weights. For generation, pass **only the decoder task prompt**, never the
target suffix from a training batch. The fixed prompt is the same length for
all rows; variable prompts would require left-padded generation inputs.

Evaluate stress accuracy, normalized output and semantic preservation on
held-out examples; CE alone cannot establish pronunciation quality. Compare
against a text-input decoder and a shuffled/zero-prefix ablation to verify
that the decoder actually uses the source hidden states. Audio/TTS quality is
outside this text-only pilot.
