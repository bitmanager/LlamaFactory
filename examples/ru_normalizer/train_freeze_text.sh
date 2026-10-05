#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
# Transformers 4.45 restores our own pickled RNG state using the old torch.load default.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
: "${CUDA_VISIBLE_DEVICES:?Set the reserved GPU UUID}"
run_root=${1:?Pass the experiment root}
shift
export PYTHONPATH="$run_root/freeze-omni-deps${PYTHONPATH:+:$PYTHONPATH}"
exec "$run_root/venv/bin/python" "$(dirname "$0")/train_freeze_text.py" \
  --features "$run_root/data/freeze-features-v1" \
  --model_config "$(dirname "$0")/freeze_decoder.json" \
  --pretrained "$run_root/models/freeze-omni/final.pt" \
  --output_dir "$run_root/runs/freeze-text-v1" \
  --bf16 true --remove_unused_columns false --prediction_loss_only true --label_names y \
  --per_device_train_batch_size 16 --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 2 --num_train_epochs 1 \
  --learning_rate 0.0001 --lr_scheduler_type cosine --warmup_ratio 0.03 \
  --eval_strategy steps --eval_steps 100 --save_steps 100 --save_total_limit 2 \
  --logging_steps 5 --report_to tensorboard --dataloader_num_workers 4 \
  --disable_tqdm true "$@"
