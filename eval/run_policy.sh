cd /root/proj/openpi
conda activate openpi
source .venv/bin/activate
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY=localhost,127.0.0.1,0.0.0.0
export no_proxy=localhost,127.0.0.1,0.0.0.0
uv run scripts/serve_policy.py --port 8000 policy:checkpoint \
  --policy.config pi05_libero_sam_dim_2_expert_lora \
  --policy.dir /root/autodl-tmp/openpi_checkpoints/pi05_libero_sam_dim_2_expert_lora/sam_dim_2_lora/15370