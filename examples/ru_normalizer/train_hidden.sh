#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1

# The run lives on the GPU host. Select an idle GPU explicitly before launching.
: "${CUDA_VISIBLE_DEVICES:?Set the reserved GPU UUID}"
run_root=${1:?Pass the experiment root containing venv, models and data}
shift
exec "$run_root/venv/bin/python" "$(dirname "$0")/train_hidden.py" \
  --model_path "$run_root/models/qwen3-4b-instruct-2507" \
  --agent_tokenizer "$run_root/models/agent-tokenizer" \
  --data_dir "$run_root/data/hidden-plan-v1" \
  --output_dir "$run_root/runs/hidden-plan-v1" \
  --bf16 true --remove_unused_columns false --prediction_loss_only true \
  --per_device_train_batch_size 8 --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 4 --num_train_epochs 1 \
  --learning_rate 0.0001 --lr_scheduler_type cosine --warmup_ratio 0.03 \
  --eval_strategy steps --eval_steps 100 --save_steps 100 --save_total_limit 2 \
  --logging_steps 5 --report_to tensorboard --dataloader_num_workers 4 \
  --disable_tqdm true "$@"
