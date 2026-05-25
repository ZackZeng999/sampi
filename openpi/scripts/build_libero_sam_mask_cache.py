#!/usr/bin/env python3
"""Build a per-episode SAM mask cache for the LIBERO LeRobot dataset.

This script reads:
1. The original LIBERO LeRobot dataset parquet episodes.
2. Prompt sidecars produced by `extract_libero_sam_prompts.py`.

It then loads SAM3 once, runs segmentation for every frame/prompt pair, and saves
one compressed `.npz` file per episode containing all frame/prompt masks.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
from PIL import Image
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

_THIS_FILE = Path(__file__).resolve()
_PROJ_ROOT = _THIS_FILE.parents[2]
_SAM3_REPO = _PROJ_ROOT / "sam3"
if str(_SAM3_REPO) not in sys.path:
    sys.path.insert(0, str(_SAM3_REPO))

LOGGER = logging.getLogger("build_libero_sam_mask_cache")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("/root/autodl-tmp/datasets/physical-intelligence/libero"), help="Path to the source LIBERO LeRobot dataset.")
    parser.add_argument("--prompts-root", type=Path, default=Path("/root/autodl-tmp/datasets/physical-intelligence/libero_sam_prompts"), help="Path to the prompt sidecar directory produced by extract_libero_sam_prompts.py.")
    parser.add_argument("--output-root", type=Path, default=Path("/root/autodl-tmp/datasets/physical-intelligence/libero_sam_masks"), help="Where to save per-episode mask sidecars.")
    parser.add_argument("--checkpoint-path", default="/root/autodl-tmp/sam3_model/sam3.pt", help="Path to the SAM3 checkpoint.")
    parser.add_argument("--device", default=None, help="Torch device, e.g. cuda or cuda:0. Defaults to cuda if available.")
    parser.add_argument("--confidence-threshold", type=float, default=0.0, help="SAM3 processor confidence threshold before mask extraction.")
    parser.add_argument("--score-threshold", type=float, default=0.6, help="Per-mask score threshold used when selecting predicted masks.")
    parser.add_argument("--max-masks-per-prompt", type=int, default=3, help="Maximum number of masks to union for one prompt on one frame.")
    parser.add_argument("--allow-fallback", action="store_true", help="If no score passes the threshold, keep the top-scoring mask anyway.")
    parser.add_argument("--episode-limit", type=int, default=0, help="If > 0, only process the first N prompt sidecar episodes.")
    parser.add_argument("--start-episode", type=int, default=0, help="Episode index to start from.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Only process every Nth frame. Default 1 means all frames.")
    parser.add_argument("--frame-batch-size", type=int, default=10, help="Number of frames to encode together with SAM3. Default 4.")
    parser.add_argument("--overwrite", action="store_true", help="Rewrite existing episode outputs.")
    parser.add_argument("--num-workers", type=int, default=4, help="Spawn this many parallel worker processes. Default 1.")
    parser.add_argument("--worker-index", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--worker-count", type=int, default=1, help=argparse.SUPPRESS)
    return parser.parse_args()


def _mask_from_batched_output(
    output: dict[str, Any],
    *,
    heights: list[int],
    widths: list[int],
    score_threshold: float,
    max_masks: int,
    allow_fallback: bool,
) -> list[tuple[np.ndarray, list[float]]]:
    masks = output.get("pred_masks")
    logits = output.get("pred_logits")
    presence = output.get("presence_logit_dec")
    if masks is None or logits is None or presence is None:
        return [(np.zeros((h, w), dtype=bool), []) for h, w in zip(heights, widths, strict=True)]

    probs = logits.sigmoid().squeeze(-1)
    presence_score = presence.sigmoid().unsqueeze(1)
    probs = probs * presence_score

    results: list[tuple[np.ndarray, list[float]]] = []
    batch_size = masks.shape[0]
    for batch_idx in range(batch_size):
        img_h = int(heights[batch_idx])
        img_w = int(widths[batch_idx])
        masks_b = masks[batch_idx]
        probs_b = probs[batch_idx].detach().float().cpu().numpy().reshape(-1)
        if masks_b.numel() == 0:
            results.append((np.zeros((img_h, img_w), dtype=bool), []))
            continue
        order = np.argsort(probs_b)[::-1]
        keep = [int(idx) for idx in order if probs_b[idx] >= score_threshold]
        if not keep:
            if allow_fallback and len(order) > 0:
                keep = [int(order[0])]
            else:
                results.append((np.zeros((img_h, img_w), dtype=bool), []))
                continue
        keep = keep[: max(1, max_masks)]
        selected = masks_b[keep].unsqueeze(1)
        upsampled = F.interpolate(selected, size=(img_h, img_w), mode="bilinear", align_corners=False).sigmoid()
        union_mask = torch.any(upsampled > 0.5, dim=0).squeeze(0).detach().cpu().numpy().astype(bool)
        kept_scores = [float(probs_b[idx]) for idx in keep]
        results.append((union_mask, kept_scores))
    return results


class SamMaskBuilder:
    def __init__(self, checkpoint_path: str, *, device: str | None, confidence_threshold: float, score_threshold: float, max_masks_per_prompt: int, allow_fallback: bool):
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model
        except ImportError as exc:
            raise RuntimeError(
                "Failed to import sam3 runtime dependencies. Run this script in an environment that has both OpenPI data dependencies and SAM3 dependencies installed, or install the missing package named in the traceback."
            ) from exc
        LOGGER.info("Loading SAM3 image model from %s on %s", checkpoint_path, self._device)
        model = build_sam3_image_model(checkpoint_path=checkpoint_path, device=self._device)
        self._processor = Sam3Processor(model, confidence_threshold=confidence_threshold, device=self._device)
        self._model = model
        self._use_cuda_autocast = self._device.startswith("cuda")
        self._score_threshold = score_threshold
        self._max_masks_per_prompt = max_masks_per_prompt
        self._allow_fallback = allow_fallback

    def segment_prompts_batch(self, images: list[np.ndarray], prompts: list[str]) -> list[list[dict[str, Any]]]:
        pil_images = [Image.fromarray(image, mode="RGB") for image in images]
        heights = [image.shape[0] for image in images]
        widths = [image.shape[1] for image in images]
        context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if self._use_cuda_autocast else contextlib.nullcontext()
        with torch.inference_mode(), context:
            state = self._processor.set_image_batch(pil_images)
            results_per_image: list[list[dict[str, Any]]] = [[] for _ in images]
            for prompt in prompts:
                text_outputs = self._model.backbone.forward_text([prompt], device=self._device)
                state["backbone_out"].update(text_outputs)
                if "geometric_prompt" not in state:
                    state["geometric_prompt"] = self._model._get_dummy_prompt()
                output = self._model.forward_grounding(
                    backbone_out=state["backbone_out"],
                    find_input=self._processor.find_stage,
                    geometric_prompt=state["geometric_prompt"],
                    find_target=None,
                )
                batched_masks = _mask_from_batched_output(
                    output,
                    heights=heights,
                    widths=widths,
                    score_threshold=self._score_threshold,
                    max_masks=self._max_masks_per_prompt,
                    allow_fallback=self._allow_fallback,
                )
                self._processor.reset_all_prompts(state)
                for image_idx, (mask, scores) in enumerate(batched_masks):
                    results_per_image[image_idx].append(
                        {
                            "prompt": prompt,
                            "mask": mask,
                            "scores": scores,
                            "mask_found": bool(mask.any()),
                            "mask_pixels": int(mask.sum()),
                        }
                    )
        return results_per_image


def _load_episode_prompt_files(prompts_root: Path) -> list[Path]:
    return sorted((prompts_root / "data").glob("chunk-*/episode_*.json"))


def _partition_prompt_files(prompt_files: list[Path], worker_count: int, worker_index: int) -> list[Path]:
    if worker_count <= 1:
        return prompt_files
    return prompt_files[worker_index::worker_count]


def _strip_managed_args(argv: list[str]) -> list[str]:
    managed = {"--num-workers", "--worker-index", "--worker-count"}
    cleaned: list[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        matched = False
        for flag in managed:
            if token == flag:
                i += 2
                matched = True
                break
            if token.startswith(flag + "="):
                i += 1
                matched = True
                break
        if matched:
            continue
        cleaned.append(token)
        i += 1
    return cleaned


def _write_run_info(output_root: Path, payload: dict[str, Any]) -> None:
    _write_json(output_root / "meta" / "info.json", payload)


def _launch_workers(args: argparse.Namespace, total_prompt_files: int) -> None:
    base_argv = _strip_managed_args(sys.argv[1:])
    procs: list[subprocess.Popen[str]] = []
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    for worker_index in range(args.num_workers):
        cmd = [
            sys.executable,
            str(_THIS_FILE),
            *base_argv,
            "--num-workers",
            "1",
            "--worker-count",
            str(args.num_workers),
            "--worker-index",
            str(worker_index),
        ]
        LOGGER.info("Launching worker %s/%s: %s", worker_index + 1, args.num_workers, " ".join(cmd))
        procs.append(subprocess.Popen(cmd, env=env))

    failed = False
    for worker_index, proc in enumerate(procs, start=1):
        return_code = proc.wait()
        if return_code != 0:
            failed = True
            LOGGER.error("Worker %s/%s exited with code %s", worker_index, args.num_workers, return_code)

    if failed:
        raise SystemExit(1)

    episode_files_written = len(list((args.output_root.resolve() / "data").glob("chunk-*/episode_*.npz")))
    _write_run_info(
        args.output_root.resolve(),
        {
            "source_dataset_root": str(args.dataset_root.resolve()),
            "source_prompts_root": str(args.prompts_root.resolve()),
            "checkpoint_path": args.checkpoint_path,
            "device": args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
            "confidence_threshold": args.confidence_threshold,
            "score_threshold": args.score_threshold,
            "max_masks_per_prompt": args.max_masks_per_prompt,
            "allow_fallback": args.allow_fallback,
            "frame_stride": args.frame_stride,
            "frame_batch_size": args.frame_batch_size,
            "num_workers": args.num_workers,
            "episodes_processed": total_prompt_files,
            "episode_files_written": episode_files_written,
            "episode_file_format": {
                "container": "npz",
                "masks": "[num_frames, num_prompts, height, width] uint8 with values {0,1}",
                "scores": "[num_frames, num_prompts] float32 best score per prompt",
                "mask_pixels": "[num_frames, num_prompts] int32 foreground pixel count",
            },
        },
    )


def _episode_output_path(output_root: Path, chunk_index: int, episode_index: int) -> Path:
    return output_root / "data" / f"chunk-{chunk_index:03d}" / f"episode_{episode_index:06d}.npz"


def _string_array(items: list[str]) -> np.ndarray:
    max_len = max((len(item) for item in items), default=1)
    return np.asarray(items, dtype=f"<U{max_len}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, force=True)

    prompt_files = _load_episode_prompt_files(args.prompts_root.resolve())
    if args.start_episode > 0:
        prompt_files = [p for p in prompt_files if int(p.stem.split("_")[1]) >= args.start_episode]
    if args.episode_limit > 0:
        prompt_files = prompt_files[: args.episode_limit]

    total_prompt_files = len(prompt_files)
    if args.num_workers > 1 and args.worker_count == 1:
        _launch_workers(args, total_prompt_files)
        return

    prompt_files = _partition_prompt_files(prompt_files, args.worker_count, args.worker_index)

    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()

    LOGGER.info("Dataset root: %s", dataset_root)
    LOGGER.info("Prompts root: %s", args.prompts_root.resolve())
    LOGGER.info("Output root: %s", output_root)
    LOGGER.info("Episodes to process: %s", len(prompt_files))
    LOGGER.info("Frame stride: %s", args.frame_stride)
    LOGGER.info("Frame batch size: %s", args.frame_batch_size)
    LOGGER.info("Worker shard: %s/%s", args.worker_index + 1, args.worker_count)

    builder = SamMaskBuilder(
        args.checkpoint_path,
        device=args.device,
        confidence_threshold=args.confidence_threshold,
        score_threshold=args.score_threshold,
        max_masks_per_prompt=args.max_masks_per_prompt,
        allow_fallback=args.allow_fallback,
    )

    total_episode_files = 0
    for ordinal, prompt_file in enumerate(prompt_files, start=1):
        prompt_payload = json.loads(prompt_file.read_text(encoding="utf-8"))
        episode_index = int(prompt_payload["episode_index"])
        prompts = [str(prompt).strip() for prompt in prompt_payload.get("prompts", []) if str(prompt).strip()]
        if not prompts:
            LOGGER.info("[%s/%s] Episode %06d has no accepted prompts, skipping.", ordinal, len(prompt_files), episode_index)
            continue

        chunk_index = int(prompt_payload["chunk_index"])
        output_path = _episode_output_path(output_root, chunk_index, episode_index)
        if output_path.exists() and not args.overwrite:
            LOGGER.info("[%s/%s] Skipping episode %06d because %s exists.", ordinal, len(prompt_files), episode_index, output_path)
            continue

        parquet_path = dataset_root / "data" / f"chunk-{chunk_index:03d}" / f"episode_{episode_index:06d}.parquet"
        LOGGER.info("[%s/%s] Processing episode %06d with %s prompts from %s", ordinal, len(prompt_files), episode_index, len(prompts), parquet_path)
        table = pq.read_table(str(parquet_path), columns=["image", "frame_index"])
        rows = table.to_pylist()
        sampled_rows = rows[:: args.frame_stride]
        if not sampled_rows:
            LOGGER.info("[%s/%s] Episode %06d produced no sampled frames, skipping.", ordinal, len(prompt_files), episode_index)
            continue

        sample_image = Image.open(io.BytesIO(sampled_rows[0]["image"]["bytes"])).convert("RGB")
        sample_height, sample_width = np.asarray(sample_image, dtype=np.uint8).shape[:2]
        num_frames = len(sampled_rows)
        num_prompts = len(prompts)

        masks = np.zeros((num_frames, num_prompts, sample_height, sample_width), dtype=np.uint8)
        scores = np.full((num_frames, num_prompts), np.nan, dtype=np.float32)
        mask_pixels = np.zeros((num_frames, num_prompts), dtype=np.int32)
        frame_indices = np.zeros((num_frames,), dtype=np.int64)

        for batch_start in range(0, num_frames, args.frame_batch_size):
            batch_rows = sampled_rows[batch_start : batch_start + args.frame_batch_size]
            batch_images: list[np.ndarray] = []
            for row in batch_rows:
                image = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
                batch_images.append(np.asarray(image, dtype=np.uint8))
            batch_results = builder.segment_prompts_batch(batch_images, prompts)
            for batch_offset, (row, prompt_results) in enumerate(zip(batch_rows, batch_results, strict=True)):
                frame_slot = batch_start + batch_offset
                frame_indices[frame_slot] = int(row["frame_index"])
                for prompt_idx, result in enumerate(prompt_results):
                    masks[frame_slot, prompt_idx] = result["mask"].astype(np.uint8)
                    scores[frame_slot, prompt_idx] = max(result["scores"]) if result["scores"] else np.nan
                    mask_pixels[frame_slot, prompt_idx] = int(result["mask_pixels"])
            if batch_start == 0 or ((batch_start // args.frame_batch_size) + 1) % 10 == 0:
                LOGGER.info("[%s/%s] Episode %06d processed %s/%s frames", ordinal, len(prompt_files), episode_index, min(batch_start + len(batch_rows), num_frames), num_frames)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            episode_index=np.asarray(episode_index, dtype=np.int64),
            chunk_index=np.asarray(chunk_index, dtype=np.int64),
            task_index=np.asarray(int(prompt_payload["task_index"]), dtype=np.int64),
            task=np.asarray(prompt_payload["task"]),
            parquet_path=np.asarray(str(parquet_path)),
            source_prompt_file=np.asarray(str(prompt_file)),
            prompts=_string_array(prompts),
            frame_indices=frame_indices,
            masks=masks,
            scores=scores,
            mask_pixels=mask_pixels,
            frame_stride=np.asarray(args.frame_stride, dtype=np.int64),
            frame_batch_size=np.asarray(args.frame_batch_size, dtype=np.int64),
            score_threshold=np.asarray(args.score_threshold, dtype=np.float32),
            max_masks_per_prompt=np.asarray(args.max_masks_per_prompt, dtype=np.int64),
            allow_fallback=np.asarray(bool(args.allow_fallback)),
        )
        total_episode_files += 1

    if args.worker_count == 1:
        _write_run_info(
            output_root,
            {
                "source_dataset_root": str(dataset_root),
                "source_prompts_root": str(args.prompts_root.resolve()),
                "checkpoint_path": args.checkpoint_path,
                "device": args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
                "confidence_threshold": args.confidence_threshold,
                "score_threshold": args.score_threshold,
                "max_masks_per_prompt": args.max_masks_per_prompt,
                "allow_fallback": args.allow_fallback,
                "frame_stride": args.frame_stride,
                "frame_batch_size": args.frame_batch_size,
                "num_workers": args.num_workers,
                "episodes_processed": len(prompt_files),
                "episode_files_written": total_episode_files,
                "episode_file_format": {
                    "container": "npz",
                    "masks": "[num_frames, num_prompts, height, width] uint8 with values {0,1}",
                    "scores": "[num_frames, num_prompts] float32 best score per prompt",
                    "mask_pixels": "[num_frames, num_prompts] int32 foreground pixel count",
                },
            },
        )
    LOGGER.info("Finished. Saved %s episode mask files in worker %s/%s.", total_episode_files, args.worker_index + 1, args.worker_count)


if __name__ == "__main__":
    main()
