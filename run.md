# OpenPI + SAM3 + LIBERO Run Guide

This README describes how to run the current integration under `/root/proj`.

## Main Scripts
/root/proj/openpi/scripts/extract_libero_sam_prompts.py
/root/proj/openpi/examples/libero/sam_dim_client.py
/root/proj/sam3/openpi_sam_dim_server.py



The full stack has 3 processes:

## Network Setting
When using codex no nedd to modify.
When not using codex to run benchmark, run this command

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY=localhost,127.0.0.1,0.0.0.0
export no_proxy=localhost,127.0.0.1,0.0.0.0

1. SAM3 server
2. OpenPI policy server
3. LIBERO evaluation client

## Directory Layout

- `/root/proj/openpi`: OpenPI policy and LIBERO eval code
- `/root/proj/sam3`: SAM3 server and agent-style prompt extractor

## Before You Start

Make sure the following are already prepared:

- SAM3 checkpoint exists at `/root/autodl-tmp/sam3_model/sam3.pt`
- OpenPI LIBERO checkpoint exists at `/root/autodl-tmp/pi0.5_libero`
- The SAM3 server file has your LLM API settings filled in:
  - `DEFAULT_LLM_SERVER_URL`
  - `DEFAULT_LLM_MODEL`
  - `DEFAULT_LLM_API_KEY`

For Alibaba Cloud DashScope OpenAI-compatible API, the defaults should look like this:

```python
DEFAULT_LLM_SERVER_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_LLM_MODEL = "qwen3-vl-8b-thinking"
DEFAULT_LLM_API_KEY = "your_api_key"
```

The current implementation uses these environments:

- SAM3 server: `conda activate sam3`
- OpenPI policy server: `uv` inside `/root/proj/openpi`
- LIBERO client: `source /root/proj/openpi/examples/libero/.venv/bin/activate`

## Terminal 1: Start SAM3 Server

```bash
cd /root/proj/sam3
conda activate sam3
python openpi_sam_dim_server.py --checkpoint-path /root/autodl-tmp/sam3_model/sam3.pt
```

What this server provides:

- `POST /segment`: run SAM3 segmentation for one object prompt
- `POST /extract`: use VLM + SAM3 validation to extract prompts from task text and image

Optional health check:

```bash
curl http://127.0.0.1:9001/health
```

A healthy response should look like:

```json
{"ok": true, "extractor_enabled": true}
```

If `extractor_enabled` is `false`, check your API URL / model / key in:

- `/root/proj/sam3/openpi_sam_dim_server.py`

## Terminal 2: Start OpenPI Policy Server

```bash
cd /root/proj/openpi
conda activate openpi
source .venv/bin/activate
uv run scripts/serve_policy.py --env LIBERO
```

This loads the current LIBERO checkpoint and serves policy inference on port `8000`.

## Terminal 3: Run LIBERO Evaluation

```bash
cd /root/proj/openpi
conda activate openpi
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
PYOPENGL_PLATFORM=egl MUJOCO_GL=egl python examples/libero/main.py --args.use-sam
```

This will:

- connect to the OpenPI policy server on `ws://0.0.0.0:8000`
- connect to the SAM3 server on `http://127.0.0.1:9001/segment`
- automatically use `http://127.0.0.1:9001/extract` for prompt extraction

## Useful Variants

Increase SAM timeout if `/extract` is slow:

```bash
PYOPENGL_PLATFORM=egl MUJOCO_GL=egl python examples/libero/main.py   --args.use-sam   --args.sam-timeout-sec 60
```

Run only a specific LIBERO suite:

```bash
PYOPENGL_PLATFORM=egl MUJOCO_GL=egl python examples/libero/main.py   --args.use-sam   --args.task-suite-name libero_spatial
```

Change SAM view:

```bash
PYOPENGL_PLATFORM=egl MUJOCO_GL=egl python examples/libero/main.py   --args.use-sam   --args.sam-view base
```

Common options:

- `--args.sam-view base|wrist|both`
- `--args.sam-timeout-sec 60`
- `--args.replan-steps 5`
- `--args.num-trials-per-task 5`

## Current Behavior

- Prompt extraction happens once per task, before the first action generation that needs replanning.
- SAM dim-background runs only when replanning.
- Saved rollout videos keep the full frame rate.
- Only replan frames are replaced by the SAM-processed image in the saved video.

Videos are saved under:

- `/root/proj/openpi/data/libero/videos`

## Typical Logs

On the SAM3 server, normal logs include:

- `Loading SAM3 image model ...`
- `Serving SAM agent service ...`
- `Extractor round 1 for task ...`
- `Extractor accepted prompt ...`

On the LIBERO client, normal logs include:

- `Cached SAM prompts for task ...`
- rollout success / failure messages

## Troubleshooting

If the SAM3 server fails with `ModuleNotFoundError: No module named 'openai'`:

```bash
cd /root/proj/sam3
conda activate sam3
pip install openai
```

If Mujoco complains about missing `DISPLAY`, use:

```bash
PYOPENGL_PLATFORM=egl MUJOCO_GL=egl
```

If DashScope returns `model_not_found`, double-check the model name. For example:

```python
qwen3-vl-8b-thinking
```

not:

```python
qwen3_vl_8b_thinking
```

If prompt extraction times out, increase:

```bash
--args.sam-timeout-sec 60
```

## Minimal Copy-Paste Run Order

Terminal 1:

```bash
cd /root/proj/sam3
conda activate sam3
python openpi_sam_dim_server.py --checkpoint-path /root/autodl-tmp/sam3_model/sam3.pt
```

Terminal 2:

```bash
cd /root/proj/openpi
uv run scripts/serve_policy.py --env LIBERO
```

Terminal 3:

```bash
cd /root/proj/openpi
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
PYOPENGL_PLATFORM=egl MUJOCO_GL=egl python examples/libero/main.py --args.use-sam
```
