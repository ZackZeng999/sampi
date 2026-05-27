cd /root/proj/openpi
export HF_LEROBOT_HOME=/root/autodl-tmp/datasets

python scripts/compute_norm_stats.py --config-name pi05_libero_sam_dim_expert_lora

export HF_LEROBOT_HOME=/root/autodl-tmp/datasets
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 python scripts/train.py pi05_libero_sam_dim_expert_lora \
  --exp-name=sam_dim_expert_lora \
  --resume \
  --no-wandb-enabled \
  --batch-size=32 \
  --num-train-steps=25500 \
  --save-interval=1000 \
  --keep-period=4250 \
  --checkpoint-base-dir=/root/autodl-tmp/openpi_checkpoints