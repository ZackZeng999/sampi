#!/usr/bin/env python3
"""Render a SAM-dimmed LIBERO LeRobot dataset from cached per-episode masks.

This script:
1. Reads the original LIBERO LeRobot parquet episodes.
2. Reads the cached `episode_xxxxxx.npz` mask files.
3. Applies the same dimming rule as `examples/libero/sam_dim_client.py`.
4. Writes a new LeRobot dataset where only `image` is replaced; `wrist_image`
   remains unchanged.
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from PIL import ImageFilter
import pyarrow.parquet as pq

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/root/autodl-tmp/datasets/physical-intelligence/libero"),
        help="Path to the source LIBERO LeRobot dataset.",
    )
    parser.add_argument(
        "--masks-root",
        type=Path,
        default=Path("/root/autodl-tmp/datasets/physical-intelligence/libero_sam_masks"),
        help="Path to the per-episode SAM mask cache.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/root/autodl-tmp/datasets/physical-intelligence/libero_sam_dim"),
        help="Where to write the dimmed LeRobot dataset.",
    )
    parser.add_argument(
        "--repo-id",
        default="physical-intelligence/libero_sam_dim",
        help="Repo id metadata to embed in the output dataset.",
    )
    parser.add_argument(
        "--background-scale",
        type=float,
        default=0.4,
        help="Background intensity scale, matching sam_dim_client defaults.",
    )
    parser.add_argument(
        "--blur-radius",
        type=float,
        default=1.5,
        help="Mask blur radius, matching sam_dim_client defaults.",
    )
    parser.add_argument(
        "--episode-limit",
        type=int,
        default=0,
        help="If > 0, only render the first N episodes.",
    )
    parser.add_argument(
        "--start-episode",
        type=int,
        default=0,
        help="Episode index to start from.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output dataset root first if it already exists.",
    )
    return parser.parse_args()


def _smooth_mask(mask: np.ndarray, radius: float) -> np.ndarray:
    if radius <= 0:
        return mask.astype(np.float32)
    mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.asarray(mask_img, dtype=np.float32) / 255.0


def _dim_background(image: np.ndarray, mask: np.ndarray, *, background_scale: float, blur_radius: float) -> np.ndarray:
    background_scale = float(np.clip(background_scale, 0.0, 1.0))
    image_f = image.astype(np.float32)
    dimmed = image_f * background_scale
    mask_alpha = _smooth_mask(mask, blur_radius)[..., None]
    output = image_f * mask_alpha + dimmed * (1.0 - mask_alpha)
    return np.clip(output, 0, 255).astype(np.uint8)


def _load_source_info(dataset_root: Path) -> dict:
    return json.loads((dataset_root / "meta" / "info.json").read_text(encoding="utf-8"))


def _normalize_features_for_create(features: dict) -> dict:
    normalized = {}
    for key, spec in features.items():
        spec = dict(spec)
        if "shape" in spec and isinstance(spec["shape"], list):
            spec["shape"] = tuple(spec["shape"])
        normalized[key] = spec
    return normalized


def _iter_mask_files(masks_root: Path) -> list[Path]:
    return sorted((masks_root / "data").glob("chunk-*/episode_*.npz"))


def _frame_mask_lookup(mask_npz: np.lib.npyio.NpzFile) -> dict[int, int]:
    frame_indices = mask_npz["frame_indices"]
    return {int(frame_index): idx for idx, frame_index in enumerate(frame_indices.tolist())}


def _decode_image(blob: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(blob)).convert("RGB"), dtype=np.uint8)


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    masks_root = args.masks_root.resolve()
    output_root = args.output_root.resolve()

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_root} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(output_root)

    source_info = _load_source_info(dataset_root)
    mask_files = _iter_mask_files(masks_root)
    if args.start_episode > 0:
        mask_files = [p for p in mask_files if int(p.stem.split("_")[1]) >= args.start_episode]
    if args.episode_limit > 0:
        mask_files = mask_files[: args.episode_limit]

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=output_root,
        fps=int(source_info["fps"]),
        robot_type=source_info["robot_type"],
        features=_normalize_features_for_create(source_info["features"]),
        use_videos=False,
        image_writer_threads=10,
        image_writer_processes=5,
    )

    for ordinal, mask_path in enumerate(mask_files, start=1):
        episode_index = int(mask_path.stem.split("_")[1])
        chunk_index = episode_index // int(source_info["chunks_size"])
        parquet_path = dataset_root / "data" / f"chunk-{chunk_index:03d}" / f"episode_{episode_index:06d}.parquet"

        mask_npz = np.load(mask_path, allow_pickle=False)
        frame_to_slot = _frame_mask_lookup(mask_npz)
        masks = mask_npz["masks"]
        task = str(mask_npz["task"].item() if getattr(mask_npz["task"], "shape", ()) == () else mask_npz["task"])

        table = pq.read_table(str(parquet_path))
        rows = table.to_pylist()

        print(f"[{ordinal}/{len(mask_files)}] Rendering episode {episode_index:06d} from {parquet_path}")
        for row in rows:
            frame_index = int(row["frame_index"])
            image = _decode_image(row["image"]["bytes"])
            if frame_index in frame_to_slot:
                slot = frame_to_slot[frame_index]
                union_mask = np.any(masks[slot].astype(bool), axis=0)
                dimmed_image = _dim_background(
                    image,
                    union_mask,
                    background_scale=args.background_scale,
                    blur_radius=args.blur_radius,
                )
            else:
                dimmed_image = image

            wrist_image = _decode_image(row["wrist_image"]["bytes"])
            dataset.add_frame(
                {
                    "image": dimmed_image,
                    "wrist_image": wrist_image,
                    "state": np.asarray(row["state"], dtype=np.float32),
                    "actions": np.asarray(row["actions"], dtype=np.float32),
                    "task": task,
                }
            )
        dataset.save_episode()

    print(f"Finished rendering {len(mask_files)} episodes to {output_root}")


if __name__ == "__main__":
    main()
