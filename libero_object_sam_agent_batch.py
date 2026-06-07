#!/usr/bin/env python3
"""Run the official SAM3 agent per task entity across LIBERO datasets."""

from __future__ import annotations

import argparse
import contextlib
from functools import partial
import json
from pathlib import Path
import re
import shutil
import sys
from typing import Any

import cv2
import torch


DEFAULT_BDDL_ROOT = Path(
    "/root/proj/openpi/third_party/libero/libero/libero/bddl_files"
)
DEFAULT_DATASETS = ("libero_10", "libero_spatial", "libero_object", "libero_goal")
DEFAULT_VIDEO_DIR = Path("/root/proj/openpi/data/libero/videos")
DEFAULT_CHECKPOINT = Path("/root/autodl-tmp/sam3_model/sam3.pt")
DEFAULT_API_KEY_FILE = Path("/root/proj/qwen_api_key.txt")
DEFAULT_OUTPUT_DIR = Path("/root/proj/sam_agent_output")
DEFAULT_SERVER_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.6-plus"
SAM3_ROOT = Path("/root/proj/sam3")


ENTITY_SYSTEM_PROMPT = """You identify the physical entities a robot must visually locate to execute a manipulation task.
Use the image as evidence, but every returned entity phrase must be copied exactly from the task description.
Do not add, remove, replace, or paraphrase words. Do not create visual aliases.
Include manipulated objects, destinations/receptacles, and explicitly named parts or controls the robot must locate.
Do not include the robot itself or irrelevant scene objects.
Output strict JSON only."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bddl-root", type=Path, default=DEFAULT_BDDL_ROOT)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DEFAULT_DATASETS,
        default=list(DEFAULT_DATASETS),
        help="Datasets to process; defaults to all four requested LIBERO datasets.",
    )
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--api-key-file", type=Path, default=DEFAULT_API_KEY_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-generations", type=int, default=10)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--task", default="", help="Only run task stems containing this text.")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of tasks across selected datasets; 0 means all.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize_api_key(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1].strip()
    return value


def slugify(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return value or "entity"


def extract_json_blob(text: str) -> str:
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    if fenced:
        return fenced.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    raise ValueError(f"No JSON object found in response: {text!r}")


def read_task_description(bddl_path: Path) -> str:
    text = bddl_path.read_text(encoding="utf-8")
    match = re.search(r"\(:language\s+(.+?)\s*\)", text, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"No :language field found in {bddl_path}")
    return match.group(1).strip()


def find_video(video_dir: Path, task_stem: str, task_description: str) -> Path:
    video_stems = [task_stem, slugify(task_description)]
    for video_stem in video_stems:
        preferred = video_dir / f"rollout_{video_stem}_success.mp4"
        if preferred.is_file():
            return preferred
        candidates = sorted(video_dir.glob(f"rollout_{video_stem}_*.mp4"))
        if candidates:
            return candidates[0]
    raise FileNotFoundError(
        f"No rollout video found for {task_stem} or task description {task_description!r}"
    )


def extract_first_frame(video_path: Path, output_path: Path) -> None:
    capture = cv2.VideoCapture(str(video_path))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Could not read first frame from {video_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), frame):
        raise RuntimeError(f"Could not save first frame to {output_path}")


def original_task_phrase(task: str, proposed: str) -> str | None:
    proposed = proposed.strip()
    if not proposed:
        return None
    match = re.search(re.escape(proposed), task, flags=re.IGNORECASE)
    return match.group(0) if match else None


def extract_entities(
    *,
    image_path: Path,
    task_description: str,
    send_generate_request,
    max_tokens: int,
) -> tuple[list[dict[str, str]], str]:
    user_text = (
        f"Task description: {task_description}\n"
        "Identify every physical entity the robot must visually locate to execute this task. "
        "Each phrase must be an exact contiguous phrase copied from the task description, preserving its words. "
        "Keep object names such as 'orange juice' unchanged; never add container words such as bottle, box, can, or carton. "
        "Return each entity once. Output exactly: "
        '{"entities": [{"phrase": "orange juice", "role": "manipulated_object"}, '
        '{"phrase": "basket", "role": "destination"}]}'
    )
    messages = [
        {"role": "system", "content": ENTITY_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": user_text},
            ],
        },
    ]
    response_text = send_generate_request(messages, max_tokens=max_tokens)
    if not response_text:
        raise RuntimeError("Entity extraction VLM returned no text")
    data = json.loads(extract_json_blob(response_text))
    raw_entities = data.get("entities", [])
    if not isinstance(raw_entities, list):
        raise ValueError("Entity extraction response must contain an entities list")

    entities: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in raw_entities:
        if not isinstance(item, dict):
            continue
        exact_phrase = original_task_phrase(task_description, str(item.get("phrase", "")))
        if not exact_phrase or exact_phrase.lower() in seen:
            continue
        seen.add(exact_phrase.lower())
        entities.append(
            {
                "phrase": exact_phrase,
                "role": str(item.get("role", "other")).strip().lower() or "other",
            }
        )
    if not entities:
        raise ValueError(f"VLM returned no valid original task phrases: {response_text}")
    return entities, response_text


def extract_agent_prompts(history: list[dict[str, Any]]) -> list[str]:
    prompts: list[str] = []
    for message in history:
        if message.get("role") != "assistant":
            continue
        content = message.get("content", [])
        for item in content if isinstance(content, list) else []:
            if not isinstance(item, dict) or item.get("type") != "text":
                continue
            text = str(item.get("text", ""))
            for tool_blob in re.findall(r"<tool>\s*(\{.*?\})\s*</tool>", text, flags=re.DOTALL):
                try:
                    tool_call = json.loads(tool_blob)
                except json.JSONDecodeError:
                    continue
                if tool_call.get("name") == "segment_phrase":
                    prompt = str(tool_call.get("parameters", {}).get("text_prompt", "")).strip()
                    if prompt and prompt not in prompts:
                        prompts.append(prompt)
    return prompts


def normalized_xywh_to_pixel_xyxy(
    box: list[float], *, width: int, height: int
) -> list[int]:
    x, y, w, h = box
    return [
        round(x * width),
        round(y * height),
        round((x + w) * width),
        round((y + h) * height),
    ]


def run_entity_agent(
    *,
    agent_inference,
    image_path: Path,
    entity: dict[str, str],
    entity_dir: Path,
    send_generate_request,
    call_sam_service,
    max_generations: int,
) -> dict[str, Any]:
    entity_dir.mkdir(parents=True, exist_ok=True)
    history_path = entity_dir / "history.json"
    pred_path = entity_dir / "pred.json"
    output_image_path = entity_dir / "pred.png"

    history, final_output, rendered_output = agent_inference(
        str(image_path),
        entity["phrase"],
        send_generate_request=send_generate_request,
        call_sam_service=call_sam_service,
        output_dir=str(entity_dir),
        debug=True,
        max_generations=max_generations,
    )
    final_output = dict(final_output)
    final_output["original_prompt"] = entity["phrase"]
    final_output["role"] = entity["role"]
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    pred_path.write_text(json.dumps(final_output, indent=2), encoding="utf-8")
    rendered_output.save(output_image_path)

    prompts = extract_agent_prompts(history)
    boxes = final_output.get("pred_boxes", [])
    width = int(final_output.get("orig_img_w", 0))
    height = int(final_output.get("orig_img_h", 0))
    return {
        "original_prompt": entity["phrase"],
        "role": entity["role"],
        "agent_prompts": prompts,
        "final_agent_prompt": prompts[-1] if prompts else None,
        "num_masks": len(final_output.get("pred_masks", [])),
        "pred_scores": final_output.get("pred_scores", []),
        "pred_boxes_normalized_xywh": boxes,
        "pred_boxes_pixel_xyxy": [
            normalized_xywh_to_pixel_xyxy(box, width=width, height=height) for box in boxes
        ],
        "pred_masks_rle": final_output.get("pred_masks", []),
        "history_path": str(history_path),
        "pred_path": str(pred_path),
        "output_image_path": str(output_image_path),
    }


def main() -> None:
    args = parse_args()
    for path in (args.bddl_root, args.video_dir, args.checkpoint, args.api_key_file):
        if not path.exists():
            raise FileNotFoundError(path)
    dataset_dirs = [(name, args.bddl_root / name) for name in args.datasets]
    for _, dataset_dir in dataset_dirs:
        if not dataset_dir.is_dir():
            raise FileNotFoundError(dataset_dir)

    sys.path.insert(0, str(SAM3_ROOT))
    from sam3.agent.agent_core import agent_inference
    from sam3.agent.client_llm import send_generate_request as send_generate_request_orig
    from sam3.agent.client_sam3 import call_sam_service as call_sam_service_orig
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    api_key = normalize_api_key(args.api_key_file.read_text(encoding="utf-8"))
    if not api_key:
        raise ValueError(f"API key file is empty: {args.api_key_file}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(f"Loading SAM3 once on {device}: {args.checkpoint}")
    model = build_sam3_image_model(checkpoint_path=str(args.checkpoint), device=device)
    processor = Sam3Processor(
        model,
        device=device,
        confidence_threshold=args.confidence_threshold,
    )
    send_generate_request = partial(
        send_generate_request_orig,
        server_url=args.server_url,
        model=args.model,
        api_key=api_key,
        enable_thinking=False,
    )
    call_sam_service = partial(call_sam_service_orig, sam3_processor=processor)

    tasks = [
        (dataset_name, bddl_path)
        for dataset_name, dataset_dir in dataset_dirs
        for bddl_path in sorted(dataset_dir.glob("*.bddl"))
    ]
    if args.task:
        tasks = [
            (dataset_name, path)
            for dataset_name, path in tasks
            if args.task.lower() in path.stem.lower()
        ]
    if args.limit > 0:
        tasks = tasks[: args.limit]

    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device == "cuda"
        else contextlib.nullcontext()
    )
    with torch.inference_mode(), autocast_context:
        for task_index, (dataset_name, bddl_path) in enumerate(tasks, start=1):
            task_stem = bddl_path.stem
            task_dir = args.output_dir / dataset_name / task_stem
            summary_path = task_dir / "summary.json"
            if summary_path.exists() and not args.overwrite:
                print(
                    f"[{task_index}/{len(tasks)}] Skipping completed task: "
                    f"{dataset_name}/{task_stem}"
                )
                continue
            if task_dir.exists() and args.overwrite:
                shutil.rmtree(task_dir)
            task_dir.mkdir(parents=True, exist_ok=True)

            summary: dict[str, Any] = {
                "dataset": dataset_name,
                "task_stem": task_stem,
                "bddl_path": str(bddl_path),
                "status": "running",
                "entities": [],
            }
            try:
                task_description = read_task_description(bddl_path)
                video_path = find_video(args.video_dir, task_stem, task_description)
                frame_path = task_dir / "input_first_frame.png"
                extract_first_frame(video_path, frame_path)
                summary.update(
                    {
                        "task_description": task_description,
                        "video_path": str(video_path),
                        "first_frame_path": str(frame_path),
                    }
                )
                print(
                    f"[{task_index}/{len(tasks)}] Extracting entities from "
                    f"{dataset_name}: {task_description}"
                )
                entities, entity_response = extract_entities(
                    image_path=frame_path,
                    task_description=task_description,
                    send_generate_request=send_generate_request,
                    max_tokens=args.max_tokens,
                )
                summary["entity_extraction_response"] = entity_response
                summary["extracted_entities"] = entities

                for entity_index, entity in enumerate(entities, start=1):
                    print(
                        f"  [{entity_index}/{len(entities)}] Running official SAM3 agent for "
                        f"{entity['phrase']!r}"
                    )
                    entity_dir = task_dir / "entities" / f"{entity_index:02d}_{slugify(entity['phrase'])}"
                    try:
                        result = run_entity_agent(
                            agent_inference=agent_inference,
                            image_path=frame_path,
                            entity=entity,
                            entity_dir=entity_dir,
                            send_generate_request=send_generate_request,
                            call_sam_service=call_sam_service,
                            max_generations=args.max_generations,
                        )
                        result["status"] = "completed"
                    except Exception as exc:
                        result = {
                            "original_prompt": entity["phrase"],
                            "role": entity["role"],
                            "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}",
                            "entity_output_dir": str(entity_dir),
                        }
                    summary["entities"].append(result)
                    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
                summary["status"] = "completed"
            except Exception as exc:
                summary["status"] = "failed"
                summary["error"] = f"{type(exc).__name__}: {exc}"
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"  Summary: {summary_path}")


if __name__ == "__main__":
    main()
