cd /root/proj/openpi
uv run scripts/serve_policy.py --port 8000 policy:checkpoint \
  --policy.config pi05_libero_sam_dim_expert_lora \
  --policy.dir /root/autodl-tmp/openpi_checkpoints/pi05_libero_sam_dim_expert_lora/sam_dim_expert_lora/8499