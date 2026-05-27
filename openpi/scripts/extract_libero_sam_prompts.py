#!/usr/bin/env python3
"""Extract SAM prompts for LIBERO episodes and save them as sidecar metadata.

This script reads the LeRobot-format LIBERO dataset, calls the running SAM server's
`/extract` endpoint, and saves prompt extraction results in a layout that mirrors the
source dataset:

    <output_root>/
      meta/
        task_prompts.jsonl
        episode_prompts.jsonl
        info.json
      data/
        chunk-000/
          episode_000000.json
          episode_000001.json
        ...

The default mode caches extraction by `task_index` to avoid repeated agent calls while
still writing one output JSON per episode to match the original dataset structure.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
from pathlib import Path
import urllib.error
import urllib.request

from PIL import Image
import pyarrow.parquet as pq


LOGGER = logging.getLogger("extract_libero_sam_prompts")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/root/autodl-tmp/datasets/physical-intelligence/libero"),
        help="Path to the downloaded LIBERO LeRobot dataset root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/root/autodl-tmp/datasets/physical-intelligence/libero_sam_prompts"),
        help="Where to save mirrored prompt sidecar files.",
    )
    parser.add_argument(
        "--extract-url",
        default="http://127.0.0.1:9001/extract",
        help="Running SAM server /extract endpoint.",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=3,
        help="Maximum number of prompts requested from the server.",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=3,
        help="Maximum number of extraction rounds requested from the server.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=120.0,
        help="HTTP timeout for a single /extract request.",
    )
    parser.add_argument(
        "--episode-limit",
        type=int,
        default=0,
        help="If > 0, only process the first N episodes.",
    )
    parser.add_argument(
        "--start-episode",
        type=int,
        default=0,
        help="Episode index to start from.",
    )
    parser.add_argument(
        "--no-cache-by-task",
        action="store_true",
        help="Disable task-level reuse and call /extract once per episode.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite existing episode sidecar JSON files.",
    )
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _iter_episode_rows(dataset_root: Path) -> list[dict]:
    episodes = _load_jsonl(dataset_root / "meta" / "episodes.jsonl")
    for row in episodes:
        episode_index = int(row["episode_index"])
        chunk_index = episode_index // 1000
        parquet_path = dataset_root / "data" / f"chunk-{chunk_index:03d}" / f"episode_{episode_index:06d}.parquet"
        row["chunk_index"] = chunk_index
        row["parquet_path"] = str(parquet_path)
    return episodes


def _load_task_map(dataset_root: Path) -> dict[int, str]:
    tasks = _load_jsonl(dataset_root / "meta" / "tasks.jsonl")
    return {int(row["task_index"]): str(row["task"]) for row in tasks}


def _read_first_frame(parquet_path: Path) -> tuple[int, int, bytes]:
    table = pq.read_table(str(parquet_path), columns=["task_index", "frame_index", "image"])
    row = table.slice(0, 1).to_pylist()[0]
    image_blob = row["image"]["bytes"]
    return int(row["task_index"]), int(row["frame_index"]), image_blob


def _encode_png_bytes(image_blob: bytes) -> str:
    image = Image.open(io.BytesIO(image_blob)).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _post_extract_request(
    extract_url: str,
    *,
    task_description: str,
    encoded_image: str,
    max_prompts: int,
    max_rounds: int,
    timeout_sec: float,
) -> dict:
    payload = {
        "task_description": task_description,
        "image": encoded_image,
        "max_prompts": max_prompts,
        "max_rounds": max_rounds,
    }
    request = urllib.request.Request(
        extract_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_sec) as response:
        result = json.loads(response.read().decode("utf-8"))
    if "error" in result:
        raise RuntimeError(result["error"])
    return result


def _episode_output_path(output_root: Path, episode_index: int) -> Path:
    chunk_index = episode_index // 1000
    return output_root / "data" / f"chunk-{chunk_index:03d}" / f"episode_{episode_index:06d}.json"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, force=True)

    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    cache_by_task = not args.no_cache_by_task

    task_map = _load_task_map(dataset_root)
    episode_rows = _iter_episode_rows(dataset_root)
    if args.start_episode > 0:
        episode_rows = [row for row in episode_rows if int(row["episode_index"]) >= args.start_episode]
    if args.episode_limit > 0:
        episode_rows = episode_rows[: args.episode_limit]

    LOGGER.info("Dataset root: %s", dataset_root)
    LOGGER.info("Output root: %s", output_root)
    LOGGER.info("Extract URL: %s", args.extract_url)
    LOGGER.info("Episodes to process: %s", len(episode_rows))
    LOGGER.info("Cache by task: %s", cache_by_task)

    task_cache: dict[int, dict] = {}
    task_prompt_rows: dict[int, dict] = {}
    episode_prompt_rows: list[dict] = []

    for ordinal, episode_meta in enumerate(episode_rows, start=1):
        episode_index = int(episode_meta["episode_index"])
        output_path = _episode_output_path(output_root, episode_index)
        if output_path.exists() and not args.overwrite:
            LOGGER.info("Skipping episode %06d because %s already exists.", episode_index, output_path)
            continue

        parquet_path = Path(str(episode_meta["parquet_path"]))
        task_index, frame_index, image_blob = _read_first_frame(parquet_path)
        task_description = task_map[task_index]

        if cache_by_task and task_index in task_cache:
            extract_result = task_cache[task_index]
            LOGGER.info(
                "[%s/%s] Reusing cached prompts for episode %06d task_index=%s",
                ordinal,
                len(episode_rows),
                episode_index,
                task_index,
            )
        else:
            LOGGER.info(
                "[%s/%s] Extracting prompts for episode %06d task_index=%s task=%r",
                ordinal,
                len(episode_rows),
                episode_index,
                task_index,
                task_description,
            )
            try:
                extract_result = _post_extract_request(
                    args.extract_url,
                    task_description=task_description,
                    encoded_image=_encode_png_bytes(image_blob),
                    max_prompts=args.max_prompts,
                    max_rounds=args.max_rounds,
                    timeout_sec=args.timeout_sec,
                )
            except (OSError, urllib.error.URLError, TimeoutError, RuntimeError) as exc:
                raise RuntimeError(f"Failed to extract prompts for episode {episode_index}: {exc}") from exc
            if cache_by_task:
                task_cache[task_index] = extract_result

        episode_payload = {
            "episode_index": episode_index,
            "chunk_index": int(episode_meta["chunk_index"]),
            "frame_index_used": frame_index,
            "parquet_path": str(parquet_path),
            "task_index": task_index,
            "task": task_description,
            "episode_length": int(episode_meta["length"]),
            "prompts": extract_result.get("prompts", []),
            "used_prompts": extract_result.get("used_prompts", []),
            "source_prompts": extract_result.get("source_prompts", []),
            "prompt_trace": extract_result.get("prompt_trace", []),
            "mode_used": extract_result.get("mode_used"),
            "extractor_enabled": bool(extract_result.get("extractor_enabled", True)),
            "evaluation_trace": extract_result.get("evaluation_trace", []),
        }
        _write_json(output_path, episode_payload)
        episode_prompt_rows.append(episode_payload)

        if task_index not in task_prompt_rows:
            task_prompt_rows[task_index] = {
                "task_index": task_index,
                "task": task_description,
                "source_episode_index": episode_index,
                "source_frame_index": frame_index,
                "prompts": extract_result.get("prompts", []),
                "used_prompts": extract_result.get("used_prompts", []),
                "source_prompts": extract_result.get("source_prompts", []),
                "prompt_trace": extract_result.get("prompt_trace", []),
                "mode_used": extract_result.get("mode_used"),
                "extractor_enabled": bool(extract_result.get("extractor_enabled", True)),
                "evaluation_trace": extract_result.get("evaluation_trace", []),
            }

    meta_dir = output_root / "meta"
    task_prompt_list = [task_prompt_rows[idx] for idx in sorted(task_prompt_rows)]
    episode_prompt_list = sorted(episode_prompt_rows, key=lambda row: int(row["episode_index"]))

    _write_jsonl(meta_dir / "task_prompts.jsonl", task_prompt_list)
    _write_jsonl(meta_dir / "episode_prompts.jsonl", episode_prompt_list)
    _write_json(
        meta_dir / "info.json",
        {
            "source_dataset_root": str(dataset_root),
            "extract_url": args.extract_url,
            "max_prompts": args.max_prompts,
            "max_rounds": args.max_rounds,
            "cache_by_task": cache_by_task,
            "total_episode_prompt_files": len(episode_prompt_list),
            "total_unique_tasks": len(task_prompt_list),
        },
    )

    LOGGER.info("Wrote %s episode prompt files and %s unique task prompt records.", len(episode_prompt_list), len(task_prompt_list))


if __name__ == "__main__":
    main()
