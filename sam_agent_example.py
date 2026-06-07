#!/usr/bin/env python3
"""Run the official SAM3 agent on the first frame of a LIBERO rollout."""

from __future__ import annotations

import argparse
import contextlib
from functools import partial
from pathlib import Path
import subprocess
import sys

import torch


DEFAULT_VIDEO = Path(
    "/root/proj/openpi/data/libero/videos/"
    "rollout_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_success.mp4"
)
DEFAULT_PROMPT = "put both the alphabet soup and the tomato sauce in the basket"
DEFAULT_CHECKPOINT = Path("/root/autodl-tmp/sam3_model/sam3.pt")
DEFAULT_API_KEY_FILE = Path("/root/proj/qwen_api_key.txt")
DEFAULT_OUTPUT_DIR = Path("/root/proj/sam_agent_example_output")
DEFAULT_SERVER_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.6-plus"
SAM3_ROOT = Path("/root/proj/sam3")
FRAME_EXTRACTOR_PYTHON = Path("/root/proj/openpi/examples/libero/.venv/bin/python")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--api-key-file", type=Path, default=DEFAULT_API_KEY_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    return parser.parse_args()


def extract_first_frame(video_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    code = (
        "import cv2,sys; "
        "cap=cv2.VideoCapture(sys.argv[1]); ok,frame=cap.read(); cap.release(); "
        "assert ok, f'Could not read first frame from {sys.argv[1]}'; "
        "assert cv2.imwrite(sys.argv[2], frame), f'Could not save {sys.argv[2]}'"
    )
    subprocess.run(
        [str(FRAME_EXTRACTOR_PYTHON), "-c", code, str(video_path), str(output_path)],
        check=True,
    )


def normalize_api_key(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1].strip()
    return value


def main() -> None:
    args = parse_args()
    for path in (args.video, args.checkpoint, args.api_key_file):
        if not path.is_file():
            raise FileNotFoundError(path)

    sys.path.insert(0, str(SAM3_ROOT))
    from sam3.agent.client_llm import send_generate_request as send_generate_request_orig
    from sam3.agent.client_sam3 import call_sam_service as call_sam_service_orig
    from sam3.agent.inference import run_single_image_inference
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame_path = args.output_dir / "input_first_frame.png"
    extract_first_frame(args.video, frame_path)

    api_key = normalize_api_key(args.api_key_file.read_text(encoding="utf-8"))
    if not api_key:
        raise ValueError(f"API key file is empty: {args.api_key_file}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(f"Loading SAM3 checkpoint on {device}: {args.checkpoint}")
    model = build_sam3_image_model(checkpoint_path=str(args.checkpoint), device=device)
    processor = Sam3Processor(model, device=device, confidence_threshold=0.5)

    send_generate_request = partial(
        send_generate_request_orig,
        server_url=args.server_url,
        model=args.model,
        api_key=api_key,
        enable_thinking=False,
    )
    call_sam_service = partial(call_sam_service_orig, sam3_processor=processor)
    llm_config = {
        "provider": "external_api",
        "model": args.model,
        "name": args.model,
        "api_key": api_key,
    }

    print(f"First frame: {frame_path}")
    print(f"Task prompt: {args.prompt}")
    print(f"Output directory: {args.output_dir}")
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device == "cuda"
        else contextlib.nullcontext()
    )
    with torch.inference_mode(), autocast_context:
        run_single_image_inference(
            str(frame_path),
            args.prompt,
            llm_config,
            send_generate_request,
            call_sam_service,
            output_dir=str(args.output_dir),
            debug=True,
        )


if __name__ == "__main__":
    main()
