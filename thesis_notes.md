# Thesis Notes: SAM-Guided Visual Preprocessing for OpenPI on LIBERO

> This document is the project memory for thesis writing, experiment tracking, and future debugging. It records the current implementation state under `/root/proj`, the data pipeline, training settings, evaluation plan, and open issues.

## 1. Working Title

Possible Chinese title:

基于 SAM 视觉增强的 VLA 机器人策略动作专家微调方法研究

Possible English title:

SAM-Guided Visual Preprocessing and Action-Expert Adaptation for Vision-Language-Action Robotic Policies

## 2. Research Goal

The project studies whether segmentation-level object perception from SAM3 can improve an OpenPI / pi0.5 VLA policy on LIBERO manipulation tasks.

The current approach does not modify the OpenPI model architecture. Instead, SAM3 is integrated as a visual preprocessing module:

```text
LIBERO RGB observation
-> VLM/SAM-agent extracts important object prompts
-> SAM3 predicts object masks
-> background is dimmed while target/object regions stay bright
-> OpenPI receives the SAM-dimmed image
-> action expert is LoRA fine-tuned on SAM-dimmed LIBERO data
```

The main thesis question:

Can SAM-guided object-level visual enhancement reduce visual distraction and help a VLA action expert generate better robot actions?

## 3. Core Idea

OpenPI already has strong visual-language-action capability, but LIBERO scenes may contain distractors and visually similar objects. SAM3 provides object masks, which can be used to emphasize task-relevant objects.

The method uses `dim_background`:

```text
foreground/object region: keep original brightness
background region: multiply RGB by background_scale
mask boundary: optionally Gaussian blur for smoother transition
```

Current implementation:

- Extract important prompts once per task, before action generation.
- During evaluation, apply SAM dim-background at replan steps.
- During fine-tuning, preprocess the LIBERO dataset offline so the model learns from SAM-dimmed images.
- Only the main `image` view is replaced in the offline dataset; `wrist_image` is preserved.

## 4. Main Repositories and Paths

Project root:

```text
/root/proj
```

OpenPI:

```text
/root/proj/openpi
```

SAM3:

```text
/root/proj/sam3
```

Original OpenPI LIBERO checkpoint:

```text
/root/autodl-tmp/pi0.5_libero
/root/autodl-tmp/pi0.5_libero/params
```

SAM3 checkpoint:

```text
/root/autodl-tmp/sam3_model/sam3.pt
```

Main run guide:

```text
/root/proj/README.md
```

## 5. Code Changes and Important Files

SAM server:

```text
/root/proj/sam3/openpi_sam_dim_server.py
```

Responsibilities:

- Loads SAM3 once and keeps the model resident in memory.
- Provides `/segment` for prompt-based segmentation.
- Provides `/extract` for VLM/SAM-agent prompt extraction.
- Uses VLM calls to propose important prompts and validates them with SAM masks.
- Keeps prompts with SAM score above the configured threshold.

OpenPI LIBERO client:

```text
/root/proj/openpi/examples/libero/main.py
/root/proj/openpi/examples/libero/sam_dim_client.py
```

Responsibilities:

- Connects to OpenPI policy server on port `8000`.
- Optionally connects to SAM server at `http://127.0.0.1:9001`.
- Calls `/extract` to get prompts.
- Calls `/segment` to get masks.
- Applies dim-background preprocessing.
- Logs extract time, SAM segmentation time, OpenPI inference time, and action chunk length.
- Saves rollout videos with full frame rate. Only replan frames use SAM-processed images in the saved video.

Dataset preprocessing scripts:

```text
/root/proj/openpi/scripts/extract_libero_sam_prompts.py
/root/proj/openpi/scripts/build_libero_sam_mask_cache.py
/root/proj/openpi/scripts/render_libero_sam_dim_dataset.py
```

Responsibilities:

- `extract_libero_sam_prompts.py`: reads LIBERO episodes and stores prompt sidecars per episode.
- `build_libero_sam_mask_cache.py`: uses SAM3 to generate per-frame, per-prompt binary masks and stores one `.npz` per episode.
- `render_libero_sam_dim_dataset.py`: reads original images and cached masks, applies the same dim-background rule, and writes a new LeRobot dataset.

Training config:

```text
/root/proj/openpi/src/openpi/training/config.py
```

Added configs:

```text
pi05_libero_sam_dim_expert_lora
pi05_libero_sam_dim_debug_expert_lora
```

The full config points to:

```text
repo_id="physical-intelligence/libero_sam_dim"
```

The debug config points to:

```text
repo_id="physical-intelligence/libero_sam_dim_debug"
```

### 5.1 Implementation Inventory

This subsection records the concrete files that were added or modified for the SAM/OpenPI integration. It is useful for thesis method writing and for future debugging.

OpenPI-side client/evaluation code:

```text
/root/proj/openpi/examples/libero/sam_dim_client.py
/root/proj/openpi/examples/libero/main.py
```

`sam_dim_client.py` is a newly added OpenPI-side HTTP client for the SAM service. It contains:

- `SamDimClient`: wrapper around the SAM server endpoints.
- `extract_prompts_for_image(...)`: sends task description and optional image to `/extract`.
- `dim_background_with_prompts(...)`: loops over accepted prompts, calls SAM segmentation, unions masks, and applies dim-background.
- `segment_object(...)`: sends a single object prompt and image to `/segment`.
- `_dim_background(...)`: keeps masked object regions bright and scales background intensity.
- `_smooth_mask(...)`: applies Gaussian blur to mask edges.

`main.py` was modified for LIBERO evaluation:

- Adds optional SAM arguments such as `use_sam`, `sam_url`, `extract_url`, `sam_view`, `sam_background_scale`, `sam_score_threshold`, `sam_blur_radius`, and `sam_timeout_sec`.
- Creates `SamDimClient` when `--args.use-sam` is enabled.
- Extracts prompts once per task before action generation.
- Runs SAM dim-background only at replanning/action-chunk generation time.
- Uses SAM-processed images for policy input.
- Writes SAM-processed frames into rollout video only at replan frames, while preserving full video frame rate.
- Logs extraction time, segmentation/dim time, OpenPI inference time, and number of generated actions.

Additional LIBERO eval variants currently present:

```text
/root/proj/openpi/examples/libero/main_10.py
/root/proj/openpi/examples/libero/main_90.py
/root/proj/openpi/examples/libero/main_goal.py
/root/proj/openpi/examples/libero/main_o_10.py
/root/proj/openpi/examples/libero/main_o_90.py
/root/proj/openpi/examples/libero/main_o_goal.py
/root/proj/openpi/examples/libero/main_o_object.py
/root/proj/openpi/examples/libero/main_o_spatial.py
```

These are suite-specific or baseline/evaluation variants. They should be checked before final thesis experiments to ensure the same evaluation protocol is used across baselines.

SAM-side server code:

```text
/root/proj/sam3/openpi_sam_dim_server.py
```

`openpi_sam_dim_server.py` is a newly added SAM3 service for OpenPI integration. It contains:

- SAM3 model loading at server startup, so the model remains mounted in memory.
- `/health`: simple health endpoint.
- `/segment`: receives one RGB image and one object prompt, runs SAM3, filters masks by score, unions up to `max_masks`, and returns a binary mask.
- `/extract`: receives task description and image, asks the VLM/SAM agent to propose important object prompts, validates prompts with SAM segmentation, and returns accepted prompts.
- Prompt acceptance based on SAM mask validity and score threshold.
- Conservative and aggressive prompt-rewriting rounds for simulator object naming issues.

Policy serving code:

```text
/root/proj/openpi/scripts/serve_policy.py
```

This file was adjusted so default LIBERO serving can use the local pi0.5-LIBERO checkpoint:

```text
/root/autodl-tmp/pi0.5_libero
```

Training config code:

```text
/root/proj/openpi/src/openpi/training/config.py
```

This file was modified to add:

- `pi05_libero_sam_dim_expert_lora`
- `pi05_libero_sam_dim_debug_expert_lora`

Both configs train only action-expert LoRA parameters.

SAM model builder:

```text
/root/proj/sam3/sam3/model_builder.py
```

This file has local modifications in the working tree. Before final thesis submission, record exactly whether the modification is required for path/assets compatibility or only for local environment setup.

Dataset generation scripts added under OpenPI:

```text
/root/proj/openpi/scripts/extract_libero_sam_prompts.py
/root/proj/openpi/scripts/build_libero_sam_mask_cache.py
/root/proj/openpi/scripts/render_libero_sam_dim_dataset.py
```

Call chain during online evaluation:

```text
examples/libero/main.py
-> SamDimClient.extract_prompts_for_image(...)
-> POST http://127.0.0.1:9001/extract
-> openpi_sam_dim_server.py uses VLM + SAM3 validation
-> prompts returned to main.py
-> SamDimClient.dim_background_with_prompts(...)
-> POST http://127.0.0.1:9001/segment for each prompt
-> union masks
-> dim background
-> send processed observation to OpenPI policy server
```

Call chain during offline dataset construction:

```text
extract_libero_sam_prompts.py
-> SAM server /extract
-> prompt JSON sidecars
-> build_libero_sam_mask_cache.py
-> local SAM3 model inference over all frames/prompts
-> per-episode NPZ masks
-> render_libero_sam_dim_dataset.py
-> LeRobot SAM-dim dataset
-> scripts/train.py pi05_libero_sam_dim_expert_lora
```

## 6. Dataset Pipeline

### 6.1 Original Dataset

Original dataset:

```text
/root/autodl-tmp/datasets/physical-intelligence/libero
```

Size:

```text
33G
```

Format:

- LeRobot dataset.
- Episode-level parquet files.
- Images are stored inside parquet rows as image bytes, not as a plain folder of image files.

### 6.2 Prompt Sidecar Dataset

Prompt sidecars:

```text
/root/autodl-tmp/datasets/physical-intelligence/libero_sam_prompts
```

Size:

```text
8.5M
```

Current count:

```text
1693 JSON files
```

Purpose:

- Store accepted task-relevant prompts for each episode.
- Avoid repeatedly calling VLM/SAM-agent during offline mask generation.

### 6.3 Mask Cache

Mask cache:

```text
/root/autodl-tmp/datasets/physical-intelligence/libero_sam_masks
```

Size:

```text
59M
```

Current count:

```text
1693 NPZ files
```

Format:

```text
one episode -> one .npz
masks: [num_frames, num_prompts, height, width], uint8, values {0, 1}
mask_pixels: [num_frames, num_prompts]
```

Current mask generation settings:

```text
score_threshold = 0.6
max_masks_per_prompt = 3
frame_stride = 1
```

### 6.4 SAM-Dim Dataset

Full SAM-dim dataset:

```text
/root/autodl-tmp/datasets/physical-intelligence/libero_sam_dim
```

Size:

```text
30G
```

Metadata:

```text
total_episodes = 1693
total_frames = 273465
total_tasks = 40
fps = 10
robot_type = panda
```

Debug SAM-dim dataset:

```text
/root/autodl-tmp/datasets/physical-intelligence/libero_sam_dim_debug
```

Metadata:

```text
total_episodes = 156
total_frames = 42455
total_tasks = 10
```

Rendering behavior:

- Replace `image` with SAM-dimmed image.
- Keep `wrist_image` unchanged.
- Preserve `state`, `actions`, task metadata, frame indices, and timestamps.

Current render settings:

```text
background_scale = 0.4
blur_radius = 1.5
```

Note:

`sam_dim_client.py` has an internal dataclass default of `background_scale=0.35`, but `main.py` and `render_libero_sam_dim_dataset.py` currently use `0.4` by default. For strict experiment reporting, use the value from the command/config that generated the specific dataset or rollout.

## 7. SAM Prompt Extraction Logic

Prompt extraction is implemented on the SAM3 server side in:

```text
/root/proj/sam3/openpi_sam_dim_server.py
```

Current high-level logic:

1. Agent receives task description and image.
2. Agent proposes important object prompts in priority order.
3. SAM3 validates each prompt by running segmentation.
4. A prompt is accepted only if SAM produces a valid mask with score above threshold.
5. Maximum returned prompts: 3.
6. If enough prompts are accepted in a conservative round, stop early.
7. If not enough useful prompts are found, retry with more semantic flexibility.

Current extraction acceptance threshold:

```text
extract_accept_score_threshold = 0.5
```

Important design note:

The agent is instructed to prefer original task object names and avoid unnecessary color changes. Conservative replacements are preferred; aggressive semantic substitutions are only used if SAM cannot detect the original object prompt.

Example motivation:

In simulation, a task object such as `chocolate pudding` may be detected better by SAM as `black object` or another visually grounded phrase. The agent may perform this replacement only when needed.

## 8. Fine-Tuning Strategy

The project fine-tunes only the VLA action expert, not the full visual-language backbone.

Current full fine-tuning config:

```text
pi05_libero_sam_dim_expert_lora
```

Model:

```python
pi0_config.Pi0Config(
    pi05=True,
    action_horizon=10,
    discrete_state_input=False,
    action_expert_variant="gemma_300m_lora",
)
```

Frozen/trainable behavior:

```python
freeze_filter=nnx.Not(nnx_utils.PathRegex(".*lora.*"))
```

Meaning:

- Freeze every non-LoRA parameter.
- Since only the action expert uses a LoRA variant, the trainable parameters are action-expert LoRA weights.
- PaliGemma / visual-language backbone remains frozen.

Initial weights:

```text
/root/autodl-tmp/pi0.5_libero/params
```

EMA:

```text
ema_decay = None
```

Reason:

OpenPI's LoRA examples disable EMA. LoRA updates are small, and disabling EMA simplifies checkpoint handling.

## 9. Training Setup

Environment:

```bash
cd /root/proj/openpi
conda activate openpi
source .venv/bin/activate
export HF_LEROBOT_HOME=/root/autodl-tmp/datasets
```

Compute norm stats before training:

```bash
python scripts/compute_norm_stats.py --config-name pi05_libero_sam_dim_expert_lora
```

Expected norm stats path:

```text
/root/proj/openpi/assets/pi05_libero_sam_dim_expert_lora/physical-intelligence/libero_sam_dim/norm_stats.json
```

One epoch calculation:

```text
total_frames = 273465
batch_size = 32
steps_per_epoch = 273465 / 32 ~= 8546
```

Current practical 1 epoch setting:

```text
num_train_steps = 8500
batch_size = 32
```

Reason for batch size:

`batch_size=32` uses about 29GB GPU memory on the current 32GB GPU, so larger batch sizes are risky.

Example 1 epoch training command:

```bash
cd /root/proj/openpi
source .venv/bin/activate
export HF_LEROBOT_HOME=/root/autodl-tmp/datasets

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py pi05_libero_sam_dim_expert_lora \
  --exp-name=sam_dim_expert_lora \
  --overwrite \
  --no-wandb-enabled \
  --batch-size=32 \
  --num-train-steps=8500 \
  --save-interval=1000 \
  --keep-period=2000 \
  --checkpoint-base-dir=/root/autodl-tmp/openpi_checkpoints
```

Note:

The above command keeps multiple checkpoints because `keep_period=2000`. For storage-saving runs, prefer:

```bash
--save-interval=8500 --keep-period=8500
```

## 10. Current Checkpoints

Checkpoint root:

```text
/root/autodl-tmp/openpi_checkpoints
```

Current total size:

```text
31G
```

Debug checkpoint:

```text
/root/autodl-tmp/openpi_checkpoints/pi05_libero_sam_dim_expert_lora/sam_dim_expert_lora_debug/1000
```

Full 1 epoch run checkpoints:

```text
/root/autodl-tmp/openpi_checkpoints/pi05_libero_sam_dim_expert_lora/sam_dim_expert_lora/2000
/root/autodl-tmp/openpi_checkpoints/pi05_libero_sam_dim_expert_lora/sam_dim_expert_lora/4000
/root/autodl-tmp/openpi_checkpoints/pi05_libero_sam_dim_expert_lora/sam_dim_expert_lora/6000
/root/autodl-tmp/openpi_checkpoints/pi05_libero_sam_dim_expert_lora/sam_dim_expert_lora/8000
/root/autodl-tmp/openpi_checkpoints/pi05_libero_sam_dim_expert_lora/sam_dim_expert_lora/8499
```

The final 1 epoch checkpoint is:

```text
/root/autodl-tmp/openpi_checkpoints/pi05_libero_sam_dim_expert_lora/sam_dim_expert_lora/8499
```

Important checkpoint interpretation:

- Training uses LoRA, but OpenPI saves an Orbax checkpoint with full inference parameters.
- The saved `params` directory is directly loadable by OpenPI.
- It is not a standalone HuggingFace PEFT-style adapter.
- No manual merge with `/root/autodl-tmp/pi0.5_libero/params` is needed.

## 11. Policy Server Commands

Original OpenPI LIBERO checkpoint:

```bash
cd /root/proj/openpi
uv run scripts/serve_policy.py --env LIBERO
```

Debug fine-tuned checkpoint:

```bash
cd /root/proj/openpi
uv run scripts/serve_policy.py --port 8000 policy:checkpoint \
  --policy.config pi05_libero_sam_dim_debug_expert_lora \
  --policy.dir /root/autodl-tmp/openpi_checkpoints/pi05_libero_sam_dim_expert_lora/sam_dim_expert_lora_debug/1000
```

Full 1 epoch fine-tuned checkpoint:

```bash
cd /root/proj/openpi
uv run scripts/serve_policy.py --port 8000 policy:checkpoint \
  --policy.config pi05_libero_sam_dim_expert_lora \
  --policy.dir /root/autodl-tmp/openpi_checkpoints/pi05_libero_sam_dim_expert_lora/sam_dim_expert_lora/8499
```

Pass the checkpoint step directory, not the `params` subdirectory.

## 12. SAM Server Command

```bash
cd /root/proj/sam3
conda activate sam3
python openpi_sam_dim_server.py --checkpoint-path /root/autodl-tmp/sam3_model/sam3.pt
```

Expected log:

```text
Serving SAM agent service at http://127.0.0.1:9001 with endpoints /segment and /extract
```

Health check:

```bash
curl --noproxy '*' http://127.0.0.1:9001/health
```

If proxy variables are set, local service calls should bypass proxy:

```bash
export NO_PROXY=localhost,127.0.0.1,0.0.0.0
export no_proxy=localhost,127.0.0.1,0.0.0.0
```

## 13. LIBERO Evaluation Commands

Typical SAM-enabled eval:

```bash
cd /root/proj/openpi
conda activate openpi
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
export NO_PROXY=localhost,127.0.0.1,0.0.0.0
export no_proxy=localhost,127.0.0.1,0.0.0.0

PYOPENGL_PLATFORM=egl MUJOCO_GL=egl python examples/libero/main.py \
  --args.use-sam \
  --args.sam-url http://127.0.0.1:9001/segment
```

Example log destination for 1 epoch fine-tuned model:

```bash
mkdir -p /root/proj/eval/openpi_finetune1epoch

PYOPENGL_PLATFORM=egl MUJOCO_GL=egl python examples/libero/main.py \
  --args.use-sam \
  --args.sam-url http://127.0.0.1:9001/segment \
  > /root/proj/eval/openpi_finetune1epoch/spatial.log 2>&1
```

Current active eval log:

```text
/root/proj/eval/openpi_finetune1epoch/spatial.log
```

## 14. Experiment Design

Recommended core comparison groups:

```text
A. OpenPI pi0.5-libero, no SAM
B. OpenPI pi0.5-libero + SAM-dim at inference, no fine-tuning
C. OpenPI pi0.5-libero + SAM-dim dataset, action expert LoRA, 1 epoch
D. OpenPI pi0.5-libero + SAM-dim dataset, action expert LoRA, 2 epochs
E. Optional: PaliGemma + action expert LoRA, to test whether visual-language backbone adaptation helps
```

Main metric:

```text
LIBERO rollout success rate
```

Secondary metrics:

```text
SAM extract latency
SAM segment/dim latency
OpenPI action inference latency
number of accepted prompts
failure cases by task
```

Interpretation logic:

```text
If B > A: SAM-dim visual preprocessing helps without training.
If C > B: action expert adaptation helps the policy use SAM-dim inputs.
If D < C: longer LoRA training may overfit or over-adapt.
If C <= B: inference-only SAM may be sufficient; fine-tuning may be unnecessary.
```

## 15. Overfitting Notes

Overfitting is possible because the model starts from a LIBERO-trained checkpoint and is trained again on transformed LIBERO images.

Risk is reduced because:

- Only action expert LoRA is trained.
- PaliGemma visual-language backbone is frozen.
- The goal is distribution adaptation from original images to SAM-dim images, not relearning LIBERO from scratch.

Risk increases if:

- Training epochs are too high.
- Evaluation tasks are too similar to demonstrations.
- SAM-dim visual style is too fixed.
- Debug dataset is used for too many steps.

Current first formal setting:

```text
1 epoch ~= 8500 steps at batch size 32
```

This is a conservative starting point.

## 16. Known Issues and Troubleshooting

### 16.1 `Connection refused` for SAM extract

Symptom:

```text
Object extraction failed; using no SAM prompts. Error: <urlopen error [Errno 111] Connection refused>
```

Likely causes:

- SAM server was not running yet.
- Proxy variables routed local `127.0.0.1` traffic through a proxy.
- The eval process cached empty prompts after the first failed extract.

Fix:

```bash
curl --noproxy '*' http://127.0.0.1:9001/health
export NO_PROXY=localhost,127.0.0.1,0.0.0.0
export no_proxy=localhost,127.0.0.1,0.0.0.0
```

Then restart eval. If the first extract failed and returned `prompts=[]`, the current task may not retry because prompts are cached per task.

### 16.2 Norm stats missing

Symptom:

```text
ValueError: Normalization stats not found.
```

Fix:

```bash
cd /root/proj/openpi
export HF_LEROBOT_HOME=/root/autodl-tmp/datasets
python scripts/compute_norm_stats.py --config-name pi05_libero_sam_dim_expert_lora
```

### 16.3 Debug checkpoint asset mismatch

Debug checkpoint uses:

```text
physical-intelligence/libero_sam_dim_debug
```

Full checkpoint uses:

```text
physical-intelligence/libero_sam_dim
```

Use the matching config when serving.

### 16.4 VS Code terminal auto-activation error

Symptom:

```text
source /root/.local/share/uv/python/cpython-3.11.15-linux-x86_64-gnu/bin/activate
bash: .../bin/activate: No such file or directory
```

Cause:

VS Code Python extension treats a uv-managed Python installation as if it were a venv.

Safe fix:

```json
"python.terminal.activateEnvironment": false
```

Do not delete the uv-managed Python directory unless you know it is unused; `uv run` may need it.

## 17. Storage Notes

Major storage users:

```text
33G  /root/autodl-tmp/datasets/physical-intelligence/libero
30G  /root/autodl-tmp/datasets/physical-intelligence/libero_sam_dim
31G  /root/autodl-tmp/openpi_checkpoints
35G  /root/autodl-tmp/hf_cache/datasets/parquet
```

HF datasets cache:

```text
/root/autodl-tmp/hf_cache/datasets/parquet/default-bfbae127a90c2f11
```

This corresponds to the full SAM-dim dataset cache:

```text
num_examples = 273465
size ~= 30G
```

Debug HF cache:

```text
/root/autodl-tmp/hf_cache/datasets/parquet/default-69e155401f4c10cd
num_examples = 42455
size ~= 4.5G
```

Do not delete cache while training is running. After training, the debug cache can be deleted if space is needed.

## 18. Thesis Chapter Draft Structure

Suggested thesis structure:

1. Introduction
   - Background: VLA policies, robotic manipulation, visual clutter.
   - Problem: VLA policies may attend to irrelevant visual regions.
   - Motivation: segmentation-level object priors from SAM.
   - Contributions.

2. Related Work
   - Vision-language-action models.
   - OpenPI / pi0.5.
   - SAM and promptable segmentation.
   - Data-efficient fine-tuning / LoRA.
   - LIBERO benchmark.

3. Method
   - Overall system architecture.
   - SAM-agent prompt extraction.
   - SAM mask generation and dim-background visual transformation.
   - Offline SAM-dim dataset construction.
   - Action expert LoRA fine-tuning.

4. Experiments
   - Dataset: LIBERO, SAM-dim dataset statistics.
   - Baselines and ablations.
   - Training settings.
   - Evaluation protocol.

5. Results and Analysis
   - Success rate comparison.
   - Latency analysis.
   - Prompt extraction quality.
   - Failure cases.
   - Overfitting discussion.

6. Conclusion
   - Summary.
   - Limitations.
   - Future work.

## 19. Current Status

Completed:

- SAM3 server with `/segment` and `/extract`.
- OpenPI LIBERO eval integration with SAM-dim preprocessing.
- Prompt extraction sidecar generation for full LIBERO dataset.
- Per-episode SAM mask cache for full LIBERO dataset.
- Full SAM-dim LeRobot dataset generation.
- OpenPI pi0.5 action expert LoRA training config.
- Debug checkpoint at 1000 steps.
- Full 1 epoch checkpoint at step `8499`.

In progress:

- Evaluation of the 1 epoch fine-tuned checkpoint.

Important current eval log:

```text
/root/proj/eval/openpi_finetune1epoch/spatial.log
```

Next steps:

- Finish eval for 1 epoch checkpoint.
- Compare against original OpenPI and inference-only SAM.
- Decide whether to train 2 epochs.
- Record success rates in this file or a dedicated results table.

## 20. Result Table Placeholder

| Setting | Checkpoint | SAM at Inference | Fine-tuned Data | Train Steps | LIBERO Suite | Success Rate | Notes |
|---|---|---:|---|---:|---|---:|---|
| A. Original OpenPI | `/root/autodl-tmp/pi0.5_libero` | No | Original LIBERO | N/A | spatial | TBD | baseline |
| B. OpenPI + SAM inference | `/root/autodl-tmp/pi0.5_libero` | Yes | Original LIBERO | N/A | spatial | TBD | no fine-tune |
| C. SAM-dim expert LoRA debug | `.../sam_dim_expert_lora_debug/1000` | Yes | SAM-dim debug | 1000 | spatial | TBD | smoke test |
| D. SAM-dim expert LoRA 1 epoch | `.../sam_dim_expert_lora/8499` | Yes | SAM-dim full | 8500 | spatial | running | current |

