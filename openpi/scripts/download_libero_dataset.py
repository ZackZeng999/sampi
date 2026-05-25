#!/usr/bin/env python3
"""Download the OpenPI LIBERO LeRobot dataset to autodl-tmp.

This downloads the Hugging Face dataset repo used by the OpenPI LIBERO
fine-tuning configs:

    physical-intelligence/libero

The default location keeps the large dataset out of /root so it can be reused
by later SAM preprocessing scripts.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys


DEFAULT_REPO_ID = "physical-intelligence/libero"
DEFAULT_CACHE_DIR = Path("/root/autodl-tmp/hf_cache")


def _format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TiB"


def _print_disk_status(path: Path) -> None:
    target = path
    while not target.exists() and target.parent != target:
        target = target.parent
    usage = shutil.disk_usage(target)
    print(f"[download-libero] Disk for {target}:")
    print(f"  total: {_format_bytes(usage.total)}")
    print(f"  used:  {_format_bytes(usage.used)}")
    print(f"  free:  {_format_bytes(usage.free)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face dataset repo id. Default: {DEFAULT_REPO_ID}",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=None,
        help=(
            "Optional plain directory mirror. If omitted, the dataset is only stored in the Hugging Face cache "
            "under --cache-dir, which is the best default for OpenPI/LeRobot."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"Hugging Face cache directory. Default: {DEFAULT_CACHE_DIR}",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional branch, tag, or commit hash to download.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Number of parallel download workers.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print paths and disk status; do not download.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.local_dir is not None:
        args.local_dir.parent.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    # Keep all Hugging Face metadata/cache on the large autodl-tmp volume.
    os.environ.setdefault("HF_HOME", str(args.cache_dir))
    os.environ.setdefault("HF_HUB_CACHE", str(args.cache_dir / "hub"))

    print("[download-libero] Dataset repo:", args.repo_id)
    print("[download-libero] Local dir:   ", args.local_dir or "<cache only>")
    print("[download-libero] HF_HOME:     ", os.environ["HF_HOME"])
    print("[download-libero] HF_HUB_CACHE:", os.environ["HF_HUB_CACHE"])
    _print_disk_status(args.local_dir or args.cache_dir)

    if args.dry_run:
        print("[download-libero] Dry run only; no files downloaded.")
        return

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is not installed in this environment. "
            "Run this from the OpenPI environment, for example: "
            "`uv run scripts/download_libero_dataset.py`."
        ) from exc

    print("[download-libero] Starting download. You can rerun this command to resume if it is interrupted.")
    snapshot_path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=str(args.local_dir) if args.local_dir is not None else None,
        cache_dir=str(args.cache_dir),
        max_workers=args.max_workers,
    )

    print("[download-libero] Finished.")
    print("[download-libero] Snapshot path:", snapshot_path)
    _print_disk_status(args.local_dir or args.cache_dir)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[download-libero] Interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
