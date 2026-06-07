"""Persistent SAM3 segmentation and prompt-extraction server for OpenPI LIBERO."""

from __future__ import annotations

import argparse
import base64
import contextlib
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
import io
import json
import logging
import os
import re
import tempfile
import threading
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import torch

from sam3.agent.client_llm import send_generate_request
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


LOGGER = logging.getLogger("openpi_sam_dim_server")

# Default LLM config for prompt extraction. Keep secrets out of source code.
DEFAULT_LLM_SERVER_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_LLM_MODEL = "qwen3.6-plus"
DEFAULT_LLM_API_KEY = ""
DEFAULT_LLM_API_KEY_FILE = "/root/proj/qwen_api_key.txt"


EXTRACT_SYSTEM_PROMPT = """
You are a robot manipulation task-analysis and visual-prompt refinement assistant.
Your goal is to convert a robot task into task-complete, visually grounded, SAM-segmentable prompts.

You operate in one explicitly specified round at a time:
1. Task decomposition: decompose the task and identify the physical entities and parts required to execute every subtask.
2. Visual source grounding: use the task decomposition and image to map functional entities to visible source prompts.
3. Conservative refinement: propose close semantic aliases for source prompts that SAM could not validate.
4. Aggressive refinement: propose broader but still task-grounded visual aliases for source prompts that failed all previous attempts.

Global rules:
- Follow only the instructions for the current round.
- Preserve the identity, task role, parent relationship, and required status of every entity across rounds.
- Every visual prompt and later replacement must remain tied to an entity from task decomposition.
- Use an ordinary semantic visual prompt for an entity that the task requires only once. When the task explicitly requires multiple instances of the same named entity, preserve each required instance and distinguish them using the simplest reliable visible difference; this may be appearance, color, type, or a spatial qualifier when useful.
- Cover every required entity and every task-required instance before considering the task visually complete.
- Do not introduce unrelated visible objects merely because they are easy to segment.
- Use short simple noun phrases for visual prompts, not full referring expressions.
- No articles, no possessives, no numbers, and no verbs in visual prompts.
- Do not casually replace the target object's category or identity. In refinement rounds, clearly image-supported color, material, shape, surface, or appearance attributes may be added, removed, or replaced when necessary to improve SAM grounding, while the prompt must remain tied to the same target object category.
- Output strict JSON only, using exactly the schema requested by the current round.
- Do not output Markdown or explanatory text outside the requested JSON.
""".strip()


def _decode_image(encoded: str) -> np.ndarray:
    raw = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def _encode_mask(mask: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _resize_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    if mask.shape == (height, width):
        return mask.astype(bool)
    mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    mask_img = mask_img.resize((width, height), resample=Image.NEAREST)
    return np.asarray(mask_img) > 0


def _bbox_from_mask(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _bbox_from_sam_boxes(boxes: Any, keep: list[int], *, height: int, width: int) -> list[int] | None:
    selected_boxes = _sam_boxes_as_xyxy(boxes, keep, height=height, width=width)
    return _union_bboxes(selected_boxes)


def _sam_boxes_as_xyxy(boxes: Any, keep: list[int], *, height: int, width: int) -> list[list[int]]:
    if boxes is None or not keep:
        return []
    if hasattr(boxes, "detach"):
        boxes_np = boxes.detach().float().cpu().numpy()
    else:
        boxes_np = np.asarray(boxes, dtype=np.float32)
    if boxes_np.size == 0:
        return []
    boxes_np = boxes_np.reshape((-1, boxes_np.shape[-1]))
    valid_keep = [idx for idx in keep if 0 <= idx < len(boxes_np)]
    if not valid_keep or boxes_np.shape[-1] < 4:
        return []
    selected = boxes_np[valid_keep, :4].astype(np.float32)
    # SAM3 image processor boxes are expected as absolute xyxy. If a future
    # backend returns normalized boxes, scale them to the current image size.
    if float(np.nanmax(np.abs(selected))) <= 1.5:
        selected[:, [0, 2]] *= width
        selected[:, [1, 3]] *= height
    out = []
    for box in selected:
        x_min = max(0, min(width - 1, int(np.floor(box[0]))))
        y_min = max(0, min(height - 1, int(np.floor(box[1]))))
        x_max = max(0, min(width - 1, int(np.ceil(box[2]))))
        y_max = max(0, min(height - 1, int(np.ceil(box[3]))))
        if x_max >= x_min and y_max >= y_min:
            out.append([x_min, y_min, x_max, y_max])
    return out


def _union_bboxes(bboxes: list[list[int] | None]) -> list[int] | None:
    valid = [bbox for bbox in bboxes if bbox]
    if not valid:
        return None
    return [
        int(min(bbox[0] for bbox in valid)),
        int(min(bbox[1] for bbox in valid)),
        int(max(bbox[2] for bbox in valid)),
        int(max(bbox[3] for bbox in valid)),
    ]


def _dedupe(items: list[str]) -> list[str]:
    deduped: list[str] = []
    for item in items:
        normalized = str(item).strip()
        if normalized and normalized not in deduped:
            deduped.append(normalized)
    return deduped


def _as_bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    if value is None:
        return default
    return bool(value)


def _read_api_key_file(path: str) -> str:
    if not path:
        return ""
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8") as file:
        return _normalize_api_key(file.read())


def _normalize_api_key(api_key: str | None) -> str:
    if not api_key:
        return ""
    api_key = api_key.strip()
    if len(api_key) >= 2 and api_key[0] == api_key[-1] and api_key[0] in {"'", '"'}:
        return api_key[1:-1].strip()
    return api_key


def _extract_json_blob(text: str) -> str:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, flags=re.DOTALL)
    if fenced:
        return fenced.group(1)
    start_obj = text.find("{")
    end_obj = text.rfind("}")
    if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
        return text[start_obj : end_obj + 1]
    start_arr = text.find("[")
    end_arr = text.rfind("]")
    if start_arr != -1 and end_arr != -1 and end_arr > start_arr:
        return text[start_arr : end_arr + 1]
    raise ValueError(f"Could not find JSON in extractor response: {text!r}")


class SamAgentService:
    def __init__(
        self,
        checkpoint_path: str,
        *,
        device: str | None = None,
        confidence_threshold: float = 0.0,
        llm_server_url: str = "",
        llm_model: str = "",
        llm_api_key: str | None = None,
        llm_max_tokens: int = 1024,
        extract_max_prompts: int = 3,
        extract_max_rounds: int = 4,
        extract_accept_score_threshold: float = 0.5,
    ):
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        LOGGER.info("Loading SAM3 image model from %s on %s", checkpoint_path, self._device)
        model = build_sam3_image_model(checkpoint_path=checkpoint_path, device=self._device)
        self._processor = Sam3Processor(model, device=self._device, confidence_threshold=confidence_threshold)
        self._use_cuda_autocast = self._device.startswith("cuda")
        self._lock = threading.Lock()
        self._llm_server_url = llm_server_url
        self._llm_model = llm_model
        self._llm_api_key = llm_api_key
        self._llm_max_tokens = llm_max_tokens
        self._extract_max_prompts = extract_max_prompts
        self._extract_max_rounds = extract_max_rounds
        self._extract_accept_score_threshold = extract_accept_score_threshold

    def extractor_enabled(self) -> bool:
        return bool(self._llm_server_url and self._llm_model)

    def segment(
        self,
        image: np.ndarray,
        object_prompt: str,
        *,
        score_threshold: float,
        max_masks: int,
    ) -> dict[str, Any]:
        mask, scores, sam_bbox = self._predict_mask(
            image,
            object_prompt,
            score_threshold=score_threshold,
            max_masks=max_masks,
        )
        return {
            "object": object_prompt,
            "mask": _encode_mask(mask),
            "mask_found": bool(mask.any()),
            "mask_pixels": int(mask.sum()),
            "sam_bbox": sam_bbox,
            "compute_bbox": _bbox_from_mask(mask),
            "scores": [float(score) for score in scores],
        }

    def extract(
        self,
        task_description: str,
        image: np.ndarray | None,
        *,
        max_prompts: int | None = None,
        max_rounds: int | None = None,
    ) -> dict[str, Any]:
        max_prompts = max_prompts or self._extract_max_prompts
        max_rounds = max_rounds or self._extract_max_rounds
        if not task_description.strip():
            return {
                "prompts": [],
                "used_prompts": [],
                "task_decomposition": {"task": "", "subtasks": []},
                "source_prompts": [],
                "prompt_trace": [],
                "extractor_enabled": self.extractor_enabled(),
            }
        if not self.extractor_enabled():
            LOGGER.warning("/extract requested but LLM server/model is not configured.")
            return {
                "prompts": [],
                "used_prompts": [],
                "task_decomposition": {"task": task_description, "subtasks": []},
                "source_prompts": [],
                "prompt_trace": [],
                "extractor_enabled": False,
                "reason": "Set --llm-server-url and --llm-model on the SAM server.",
            }

        image_path = None
        grounding_overlay_path = None
        if image is not None:
            tmp = tempfile.NamedTemporaryFile(prefix="sam3_extract_", suffix=".png", delete=False)
            image_path = tmp.name
            tmp.close()
            Image.fromarray(image, mode="RGB").save(image_path)

        try:
            used: list[str] = []
            evaluation_trace: list[dict[str, Any]] = []
            candidate_budget = max(max_prompts, min(6, max_prompts * 2))

            # Round 1: decompose the language task without relying on visual appearance.
            decomposition_text = self._generate_task_decomposition(task_description=task_description)
            task_decomposition = self._parse_task_decomposition(decomposition_text, task_description=task_description)
            if max_rounds <= 1:
                return self._build_extract_response(
                    [],
                    task_decomposition=task_decomposition,
                    used_prompts=used,
                    evaluation_trace=evaluation_trace,
                    max_prompts=max_prompts,
                    fallback_mode="task_decomposition",
                )

            # Before Round 2, ground each Round 1 entity using its original semantic name.
            initial_grounding, grounding_overlay_path = self._ground_task_entities(
                image=image,
                task_decomposition=task_decomposition,
            )

            # Round 2: inspect the original-entity SAM evidence and map entities to visible SAM prompts.
            response_text = self._generate_source_prompts(
                image_path=grounding_overlay_path or image_path,
                task_description=task_description,
                task_decomposition=task_decomposition,
                initial_grounding=initial_grounding,
                remaining_slots=candidate_budget,
            )
            all_source_candidates = self._apply_task_metadata(
                self._parse_source_prompt_candidates(response_text), task_decomposition
            )
            required_candidates = [source for source in all_source_candidates if source.get("required")]
            optional_candidates = [source for source in all_source_candidates if not source.get("required")]
            source_candidates = required_candidates + optional_candidates[: max(0, candidate_budget - len(required_candidates))]
            source_states: list[dict[str, Any]] = []
            for priority, source in enumerate(source_candidates, start=1):
                state = {
                    "source_key": f"{source['entity']}::{source['visual_prompt']}",
                    "source_prompt": source["entity"],
                    "source_instance": source["visual_prompt"],
                    "source_role": source.get("source_role", "unknown"),
                    "initial_prompt": source["visual_prompt"],
                    "directly_contacted": source.get("directly_contacted", False),
                    "state_change": source.get("state_change"),
                    "inferred": source.get("inferred", False),
                    "parent_entity": source.get("parent_entity"),
                    "required": source.get("required", False),
                    "selected_prompt": None,
                    "selected_round": None,
                    "selected_score": 0.0,
                    "attempts": [],
                }
                source_states.append(state)
                attempt = self._evaluate_source_candidate(
                    image=image,
                    state=state,
                    prompt=state["initial_prompt"],
                    mode="visual_source_grounding",
                    priority=priority,
                    used=used,
                    evaluation_trace=evaluation_trace,
                )
                if attempt["accepted"]:
                    self._select_attempt(state, attempt)

            if max_rounds <= 2:
                return self._build_extract_response(
                    source_states,
                    task_decomposition=task_decomposition,
                    used_prompts=used,
                    evaluation_trace=evaluation_trace,
                    max_prompts=max_prompts,
                    fallback_mode="visual_source_grounding",
                )

            # Round 3: propose close aliases only while required entities still lack a valid prompt.
            if max_rounds >= 3 and not self._required_sources_covered(source_states):
                self._run_refinement_round(
                    image=image,
                    image_path=image_path,
                    task_description=task_description,
                    task_decomposition=task_decomposition,
                    source_states=source_states,
                    used=used,
                    evaluation_trace=evaluation_trace,
                    mode="conservative",
                    max_candidates_per_source=3,
                    allow_successful_replacements=True,
                )
            if max_rounds <= 3:
                return self._build_extract_response(
                    source_states,
                    task_decomposition=task_decomposition,
                    used_prompts=used,
                    evaluation_trace=evaluation_trace,
                    max_prompts=max_prompts,
                    fallback_mode="conservative",
                )

            # Round 4: broaden aliases only while required entities still lack a valid prompt.
            if max_rounds >= 4 and not self._required_sources_covered(source_states):
                self._run_refinement_round(
                    image=image,
                    image_path=image_path,
                    task_description=task_description,
                    task_decomposition=task_decomposition,
                    source_states=source_states,
                    used=used,
                    evaluation_trace=evaluation_trace,
                    mode="aggressive",
                    max_candidates_per_source=5,
                    allow_successful_replacements=False,
                )
            return self._build_extract_response(
                source_states,
                task_decomposition=task_decomposition,
                used_prompts=used,
                evaluation_trace=evaluation_trace,
                max_prompts=max_prompts,
                fallback_mode="aggressive",
            )
        finally:
            for temporary_path in (grounding_overlay_path, image_path):
                if temporary_path and os.path.exists(temporary_path):
                    os.remove(temporary_path)

    def _ground_task_entities(
        self,
        *,
        image: np.ndarray | None,
        task_decomposition: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], str | None]:
        entities: list[dict[str, Any]] = []
        by_entity: dict[str, dict[str, Any]] = {}
        for subtask in task_decomposition.get("subtasks", []):
            for argument in subtask.get("arguments", []):
                entity = str(argument.get("entity", "")).strip().lower()
                if not entity:
                    continue
                if entity not in by_entity:
                    summary = {
                        "entity": entity,
                        "role": argument.get("role", "other"),
                        "inferred": bool(argument.get("inferred", False)),
                        "required": bool(argument.get("required", False)),
                    }
                    by_entity[entity] = summary
                    entities.append(summary)
                else:
                    by_entity[entity]["required"] = bool(by_entity[entity]["required"] or argument.get("required"))

        grounding: list[dict[str, Any]] = []
        all_regions: list[tuple[str, dict[str, Any]]] = []
        for entity_summary in entities:
            regions = self._predict_regions(
                image,
                entity_summary["entity"],
                score_threshold=0.0,
                max_regions=3,
            ) if image is not None else []
            validated = any(region.get("score", 0.0) >= self._extract_accept_score_threshold for region in regions)
            grounding_item = {
                **entity_summary,
                "sam_prompt": entity_summary["entity"],
                "found": bool(regions),
                "validated": validated,
                "accept_score_threshold": self._extract_accept_score_threshold,
                "regions": regions,
            }
            grounding.append(grounding_item)
            all_regions.extend((entity_summary["entity"], region) for region in regions)
            LOGGER.info(
                "Extractor initial entity grounding for %r found=%s regions=%s",
                entity_summary["entity"],
                bool(regions),
                regions,
            )

        if image is None:
            return grounding, None
        return grounding, self._render_grounding_overlay(image, all_regions)

    def _render_grounding_overlay(
        self,
        image: np.ndarray,
        labeled_regions: list[tuple[str, dict[str, Any]]],
    ) -> str:
        canvas = Image.fromarray(image, mode="RGB").convert("RGBA")
        colors = [(255, 59, 48), (0, 122, 255), (52, 199, 89), (255, 149, 0), (175, 82, 222), (255, 45, 146)]
        draw = ImageDraw.Draw(canvas)
        for index, (entity, region) in enumerate(labeled_regions, start=1):
            bbox = region.get("sam_bbox") or region.get("compute_bbox")
            if not bbox:
                continue
            color = colors[(index - 1) % len(colors)]
            draw.rectangle(bbox, outline=(*color, 255), width=3)
            label = f"{entity} {region.get('region_id', f'R{index}')}"
            label_y = max(0, int(bbox[1]) - 12)
            text_bbox = draw.textbbox((int(bbox[0]), label_y), label)
            draw.rectangle(text_bbox, fill=(0, 0, 0, 220))
            draw.text((int(bbox[0]), label_y), label, fill=(*color, 255))
        tmp = tempfile.NamedTemporaryFile(prefix="sam3_grounding_", suffix=".png", delete=False)
        overlay_path = tmp.name
        tmp.close()
        canvas.convert("RGB").save(overlay_path)
        return overlay_path

    def _evaluate_source_candidate(
        self,
        *,
        image: np.ndarray | None,
        state: dict[str, Any],
        prompt: str,
        mode: str,
        priority: int,
        used: list[str],
        evaluation_trace: list[dict[str, Any]],
    ) -> dict[str, Any]:
        prompt = str(prompt).strip().lower()
        used.append(prompt)
        mask, scores, sam_bbox = self._predict_mask(
            image,
            prompt,
            score_threshold=self._extract_accept_score_threshold,
            max_masks=3,
            allow_fallback=False,
        )
        best_score = max(scores) if scores else 0.0
        attempt = {
            "mode": mode,
            "priority": priority,
            "source_key": state["source_key"],
            "source_prompt": state["source_prompt"],
            "source_instance": state.get("source_instance"),
            "source_role": state.get("source_role", "unknown"),
            "prompt": prompt,
            "scores": [float(score) for score in scores],
            "best_score": float(best_score),
            "accepted": bool(mask.any()),
            "sam_bbox": sam_bbox,
            "compute_bbox": _bbox_from_mask(mask),
        }
        state["attempts"].append(attempt)
        evaluation_trace.append(attempt.copy())
        if attempt["accepted"]:
            LOGGER.info(
                "Extractor accepted prompt %r for source %r in %s mode at priority %s with best_score=%.3f and scores=%s",
                prompt,
                state["source_prompt"],
                mode,
                priority,
                best_score,
                scores,
            )
        else:
            LOGGER.info(
                "Extractor rejected prompt %r for source %r in %s mode at priority %s with best_score=%.3f and scores=%s (need best_score >= %.2f)",
                prompt,
                state["source_prompt"],
                mode,
                priority,
                best_score,
                scores,
                self._extract_accept_score_threshold,
            )
        return attempt

    def _select_attempt(self, state: dict[str, Any], attempt: dict[str, Any]) -> None:
        state["selected_prompt"] = attempt["prompt"]
        state["selected_round"] = attempt["mode"]
        state["selected_score"] = float(attempt["best_score"])
        state["selected_sam_bbox"] = attempt.get("sam_bbox")
        state["selected_compute_bbox"] = attempt.get("compute_bbox")

    def _selected_prompt_count(self, source_states: list[dict[str, Any]]) -> int:
        return len(_dedupe([state["selected_prompt"] for state in source_states if state.get("selected_prompt")]))

    def _required_sources_covered(self, source_states: list[dict[str, Any]]) -> bool:
        required_states = [state for state in source_states if state.get("required")]
        states_to_cover = required_states or source_states
        return bool(states_to_cover) and all(state.get("selected_prompt") for state in states_to_cover)

    def _run_refinement_round(
        self,
        *,
        image: np.ndarray | None,
        image_path: str | None,
        task_description: str,
        task_decomposition: dict[str, Any],
        source_states: list[dict[str, Any]],
        used: list[str],
        evaluation_trace: list[dict[str, Any]],
        mode: str,
        max_candidates_per_source: int,
        allow_successful_replacements: bool,
    ) -> None:
        failed_states = [state for state in source_states if not state.get("selected_prompt")]
        if not failed_states and not allow_successful_replacements:
            return
        response_text = self._generate_replacement_prompts(
            image_path=image_path,
            task_description=task_description,
            task_decomposition=task_decomposition,
            source_states=source_states,
            mode=mode,
            max_candidates_per_source=max_candidates_per_source,
            allow_successful_replacements=allow_successful_replacements,
        )
        valid_sources = [state["source_key"] for state in source_states]
        if mode == "aggressive":
            valid_sources = [state["source_key"] for state in failed_states]
        replacement_map = self._parse_replacement_candidates(response_text, valid_sources=valid_sources)
        state_by_source = {state["source_key"]: state for state in source_states}
        for source_key, replacement_info in replacement_map.items():
            state = state_by_source.get(source_key)
            if state is None:
                continue
            already_selected = bool(state.get("selected_prompt"))
            if mode == "aggressive" and already_selected:
                continue
            if already_selected and not (
                allow_successful_replacements and replacement_info.get("replace_successful", False)
            ):
                continue

            accepted_attempts: list[dict[str, Any]] = []
            tried_for_source = {attempt["prompt"] for attempt in state.get("attempts", [])}
            candidates = [
                candidate
                for candidate in replacement_info.get("candidates", [])
                if candidate and candidate not in tried_for_source and candidate != state["source_prompt"]
            ]
            for priority, candidate in enumerate(candidates[:max_candidates_per_source], start=1):
                attempt = self._evaluate_source_candidate(
                    image=image,
                    state=state,
                    prompt=candidate,
                    mode=mode,
                    priority=priority,
                    used=used,
                    evaluation_trace=evaluation_trace,
                )
                if attempt["accepted"]:
                    accepted_attempts.append(attempt)

            if not accepted_attempts:
                continue
            best_attempt = max(
                enumerate(accepted_attempts),
                key=lambda item: (item[1]["best_score"], -item[0]),
            )[1]
            if already_selected and best_attempt["best_score"] < float(state.get("selected_score", 0.0)):
                LOGGER.info(
                    "Keeping original successful prompt %r for source %r because conservative replacement %r scored lower (%.3f < %.3f).",
                    state.get("selected_prompt"),
                    state["source_prompt"],
                    best_attempt["prompt"],
                    best_attempt["best_score"],
                    state.get("selected_score", 0.0),
                )
                continue
            self._select_attempt(state, best_attempt)

    def _build_extract_response(
        self,
        source_states: list[dict[str, Any]],
        *,
        task_decomposition: dict[str, Any],
        used_prompts: list[str],
        evaluation_trace: list[dict[str, Any]],
        max_prompts: int,
        fallback_mode: str,
    ) -> dict[str, Any]:
        selected_states = [state for state in source_states if state.get("selected_prompt")]
        prompts = _dedupe([state["selected_prompt"] for state in selected_states])[:max_prompts]
        selected_modes = [state["selected_round"] for state in selected_states if state.get("selected_round")]
        round_rank = {"task_decomposition": 1, "visual_source_grounding": 2, "conservative": 3, "aggressive": 4}
        mode_used = max(selected_modes, key=lambda mode: round_rank.get(mode, 0)) if selected_modes else fallback_mode
        prompt_trace = []
        for state in source_states:
            prompt_trace.append(
                {
                    "source_key": state.get("source_key"),
                    "source_prompt": state["source_prompt"],
                    "source_instance": state.get("source_instance"),
                    "source_role": state.get("source_role", "unknown"),
                    "initial_prompt": state.get("initial_prompt"),
                    "directly_contacted": state.get("directly_contacted", False),
                    "state_change": state.get("state_change"),
                    "inferred": state.get("inferred", False),
                    "parent_entity": state.get("parent_entity"),
                    "required": state.get("required", False),
                    "selected_prompt": state.get("selected_prompt"),
                    "selected_round": state.get("selected_round"),
                    "selected_score": float(state.get("selected_score", 0.0)),
                    "selected_sam_bbox": state.get("selected_sam_bbox"),
                    "selected_compute_bbox": state.get("selected_compute_bbox"),
                    "status": "selected" if state.get("selected_prompt") else "failed",
                    "attempts": state.get("attempts", []),
                }
            )
        missing_required_states = [
            state
            for state in source_states
            if state.get("required") and state.get("selected_prompt") not in prompts
        ]
        missing_required_sources = [state["source_prompt"] for state in missing_required_states]
        missing_required_instances = [state.get("source_instance") or state["source_prompt"] for state in missing_required_states]
        return {
            "prompts": prompts,
            "used_prompts": _dedupe(used_prompts),
            "task_decomposition": task_decomposition,
            "source_prompts": [
                {
                    "source_key": state.get("source_key"),
                    "source_prompt": state["source_prompt"],
                    "source_instance": state.get("source_instance"),
                    "source_role": state.get("source_role", "unknown"),
                    "visual_prompt": state.get("initial_prompt"),
                    "directly_contacted": state.get("directly_contacted", False),
                    "state_change": state.get("state_change"),
                    "inferred": state.get("inferred", False),
                    "parent_entity": state.get("parent_entity"),
                    "required": state.get("required", False),
                }
                for state in source_states
            ],
            "prompt_trace": prompt_trace,
            "extractor_enabled": True,
            "mode_used": mode_used,
            "required_coverage_complete": not missing_required_sources,
            "missing_required_sources": missing_required_sources,
            "missing_required_instances": missing_required_instances,
            "evaluation_trace": evaluation_trace,
        }

    def _predict_mask(
        self,
        image: np.ndarray | None,
        object_prompt: str,
        *,
        score_threshold: float,
        max_masks: int,
        allow_fallback: bool = True,
    ) -> tuple[np.ndarray, list[float], list[int] | None]:
        if image is None:
            return np.zeros((1, 1), dtype=bool), [], None
        height, width = image.shape[:2]
        pil_image = Image.fromarray(image, mode="RGB")
        context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self._use_cuda_autocast
            else contextlib.nullcontext()
        )
        with self._lock, torch.inference_mode(), context:
            state = self._processor.set_image(pil_image)
            return self._segment_from_state(
                state,
                object_prompt,
                height=height,
                width=width,
                score_threshold=score_threshold,
                max_masks=max_masks,
                allow_fallback=allow_fallback,
            )

    def _predict_regions(
        self,
        image: np.ndarray,
        object_prompt: str,
        *,
        score_threshold: float,
        max_regions: int,
    ) -> list[dict[str, Any]]:
        height, width = image.shape[:2]
        pil_image = Image.fromarray(image, mode="RGB")
        context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self._use_cuda_autocast
            else contextlib.nullcontext()
        )
        with self._lock, torch.inference_mode(), context:
            state = self._processor.set_image(pil_image)
            output = self._processor.set_text_prompt(state=state, prompt=object_prompt.strip())

        masks = output.get("masks")
        if masks is None or len(masks) == 0:
            return []
        masks_np = masks.detach().cpu().numpy() if hasattr(masks, "detach") else np.asarray(masks)
        if masks_np.ndim == 4 and masks_np.shape[1] == 1:
            masks_np = masks_np[:, 0]
        scores = output.get("scores")
        if scores is None:
            scores_np = np.ones((masks_np.shape[0],), dtype=np.float32)
        elif hasattr(scores, "detach"):
            scores_np = scores.detach().float().cpu().numpy().reshape(-1)
        else:
            scores_np = np.asarray(scores, dtype=np.float32).reshape(-1)
        keep = [int(index) for index in np.argsort(scores_np)[::-1] if scores_np[index] >= score_threshold]
        keep = keep[:max(1, max_regions)]
        sam_boxes = _sam_boxes_as_xyxy(output.get("boxes"), keep, height=height, width=width)
        regions = []
        for region_index, mask_index in enumerate(keep, start=1):
            mask = _resize_mask(masks_np[mask_index], height, width)
            regions.append({
                "region_id": f"R{region_index}",
                "score": float(scores_np[mask_index]),
                "sam_bbox": sam_boxes[region_index - 1] if region_index <= len(sam_boxes) else None,
                "compute_bbox": _bbox_from_mask(mask),
                "mask_pixels": int(mask.sum()),
            })
        return regions

    def _segment_from_state(
        self,
        state: dict[str, Any],
        object_prompt: str,
        *,
        height: int,
        width: int,
        score_threshold: float,
        max_masks: int,
        allow_fallback: bool = True,
    ) -> tuple[np.ndarray, list[float], list[int] | None]:
        object_prompt = object_prompt.strip()
        if not object_prompt:
            return np.zeros((height, width), dtype=bool), [], None
        output = self._processor.set_text_prompt(state=state, prompt=object_prompt)
        return self._mask_from_output(
            output,
            height=height,
            width=width,
            score_threshold=score_threshold,
            max_masks=max_masks,
            allow_fallback=allow_fallback,
        )

    def _mask_from_output(
        self,
        output: dict[str, Any],
        *,
        height: int,
        width: int,
        score_threshold: float,
        max_masks: int,
        allow_fallback: bool = True,
    ) -> tuple[np.ndarray, list[float], list[int] | None]:
        masks = output.get("masks")
        if masks is None or len(masks) == 0:
            return np.zeros((height, width), dtype=bool), [], None

        masks_np = masks.detach().cpu().numpy()
        if masks_np.ndim == 4 and masks_np.shape[1] == 1:
            masks_np = masks_np[:, 0]

        scores = output.get("scores")
        if scores is None:
            scores_np = np.ones((masks_np.shape[0],), dtype=np.float32)
        else:
            scores_np = scores.detach().float().cpu().numpy().reshape(-1)

        order = np.argsort(scores_np)[::-1]
        keep = [int(idx) for idx in order if scores_np[idx] >= score_threshold]
        if not keep:
            if allow_fallback and len(order) > 0:
                keep = [int(order[0])]
            else:
                return np.zeros((height, width), dtype=bool), [], None
        keep = keep[: max(1, max_masks)]

        union_mask = np.zeros((height, width), dtype=bool)
        kept_scores = []
        for idx in keep:
            resized = _resize_mask(masks_np[idx], height, width)
            union_mask |= resized
            kept_scores.append(float(scores_np[idx]))
        sam_bbox = _bbox_from_sam_boxes(output.get("boxes"), keep, height=height, width=width)
        return union_mask, kept_scores, sam_bbox

    def _generate_task_decomposition(self, *, task_description: str) -> str:
        user_text = (
            "Current round: task_decomposition. "
            f"Task description: {task_description}. "
            "Decompose the task into ordered atomic subtasks. This is a language-only planning round: do not use visual "
            "appearance and do not generate SAM prompts, visual aliases, masks, or bounding boxes. For every subtask, "
            "identify the physical objects, regions, controls, or movable parts needed to execute it. Explicitly consider "
            "what the robot must physically operate. Resolve pronouns. Preserve task wording whenever possible. Infer a "
            "part only when it is necessary or meaningfully useful for execution. For state-changing actions such as turn "
            "on, open, close, or press, include the physical control or movable part the robot must operate; use a functional "
            "name such as stove control when its exact visual form is unknown. Use role values manipulated_object, "
            "interaction_target, state_change_target, destination, placement_region, context, or other. Set "
            "directly_contacted=true only when the robot intentionally operates that entity. Set inferred=true only for "
            "entities not explicitly named in the task. Set parent_entity to the complete containing object for a part or "
            "region. Set required=true only when visually locating the entity is necessary to execute the subtask. "
            "Output strict JSON only in exactly this form: "
            '{"task": "pick up the moka pot", "subtasks": [{"subtask_id": "subtask_1", "action": "pick_up", '
            '"description": "Pick up the moka pot.", "arguments": [{"entity": "moka pot", '
            '"role": "manipulated_object", "directly_contacted": true, '
            '"state_change": "supported_to_grasped", "inferred": false, "parent_entity": null, "required": true}]}]}'
        )
        messages = self._build_llm_messages(image_path=None, user_text=user_text)
        LOGGER.info("Extractor task decomposition round for task %r", task_description)
        response_text = send_generate_request(
            messages,
            server_url=self._llm_server_url,
            model=self._llm_model,
            api_key=self._llm_api_key,
            max_tokens=self._llm_max_tokens,
            enable_thinking=False,
        )
        if not response_text:
            raise RuntimeError("LLM task decomposition returned no text")
        return response_text

    def _generate_source_prompts(
        self,
        *,
        image_path: str | None,
        task_description: str,
        task_decomposition: dict[str, Any],
        initial_grounding: list[dict[str, Any]],
        remaining_slots: int,
    ) -> str:
        user_text = (
            "Current round: visual_source_grounding. "
            f"Task description: {task_description}. "
            f"Task decomposition: {json.dumps(task_decomposition, ensure_ascii=True)}. "
            f"Initial SAM grounding from each Round 1 entity name: {json.dumps(initial_grounding, ensure_ascii=True)}. "
            f"Return at most {remaining_slots} visually grounded object instances in ranked order, with required instances first. "
            "The provided image is annotated with the independently returned regions from running SAM on the original Round 1 "
            "entity names. Use both the annotated image and the structured initial grounding results as evidence. Region scores "
            "express SAM confidence, and validated indicates whether any region reached the normal acceptance threshold; low-score "
            "regions are uncertain evidence rather than confirmed targets. If an original entity prompt already identifies a plausible "
            "task target, preserve that original semantic prompt rather than replacing "
            "it unnecessarily. Use the evidence to map each useful functional entity from task decomposition to one or more short "
            "visible noun phrases suitable for SAM. R1, R2, and R3 are region markers for reference only and must not appear in visual_prompt. "
            "For an entity required only once, use its ordinary semantic visual name without "
            "adding an instance distinction merely because of its position in the image. Only split an entity into multiple "
            "source_prompts items when the task explicitly requires multiple physical instances of that same named entity, "
            "such as two moka pots or all bowls. The phrase both A and B refers to two different named entities and does not "
            "make either A or B a multi-instance entity. For true same-entity multi-instance cases, repeat the same entity value "
            "and distinguish each visual_prompt using the simplest reliable visible difference. Prefer stable appearance, color, "
            "or type differences when available; spatial qualifiers such as left, right, front, back, or middle are also allowed "
            "when they are the clearest distinction. Do not collapse multiple task-required instances into one item. Preserve "
            "each entity's role, directly_contacted, state_change, inferred, parent_entity, "
            "and required values exactly. For a functional inferred entity whose exact visual form is unknown, inspect the "
            "image and select its most likely visible form; for example, stove control may become stove knob, switch, or "
            "button. Be especially cautious when adding, removing, or replacing a container-type word such as bottle, box, "
            "carton, can, jar, or package for any noun explicitly present in the task description. Do so only when the annotated "
            "image provides clear visual evidence that the task entity has that container type. The existence of other bottles, "
            "boxes, or cans in the scene is not evidence about the target entity. When uncertain, preserve the task noun unchanged. "
            "Do not remove required entities, invent new functional entities, or introduce unrelated visible objects. If the exact "
            "visible form of a required entity remains uncertain, keep the functional entity itself as visual_prompt instead of omitting it. "
            "Output strict JSON only in exactly this form: "
            '{"source_prompts": [{"entity": "cream cheese box", "source_role": "manipulated_object", '
            '"visual_prompt": "cream cheese box", "directly_contacted": true, "state_change": "supported_to_grasped", '
            '"inferred": false, "parent_entity": null, "required": true}, {"entity": "basket", '
            '"source_role": "placement_region", "visual_prompt": "basket", "directly_contacted": false, '
            '"state_change": null, "inferred": false, "parent_entity": null, "required": true}]}'
        )
        messages = self._build_llm_messages(image_path=image_path, user_text=user_text)
        LOGGER.info("Extractor visual source grounding round for task %r", task_description)
        response_text = send_generate_request(
            messages,
            server_url=self._llm_server_url,
            model=self._llm_model,
            api_key=self._llm_api_key,
            max_tokens=self._llm_max_tokens,
            enable_thinking=False,
        )
        if not response_text:
            raise RuntimeError("LLM visual source grounding returned no text")
        return response_text

    def _generate_replacement_prompts(
        self,
        *,
        image_path: str | None,
        task_description: str,
        task_decomposition: dict[str, Any],
        source_states: list[dict[str, Any]],
        mode: str,
        max_candidates_per_source: int,
        allow_successful_replacements: bool,
    ) -> str:
        state_summary = [
            {
                "source_key": state["source_key"],
                "entity": state["source_prompt"],
                "source_instance": state.get("source_instance"),
                "source_role": state.get("source_role", "unknown"),
                "initial_prompt": state.get("initial_prompt"),
                "directly_contacted": state.get("directly_contacted", False),
                "state_change": state.get("state_change"),
                "inferred": state.get("inferred", False),
                "parent_entity": state.get("parent_entity"),
                "required": state.get("required", False),
                "selected_prompt": state.get("selected_prompt"),
                "selected_round": state.get("selected_round"),
                "attempts": [
                    {
                        "round": attempt["mode"],
                        "prompt": attempt["prompt"],
                        "accepted": attempt["accepted"],
                        "best_score": attempt["best_score"],
                        "scores": attempt["scores"],
                    }
                    for attempt in state.get("attempts", [])
                ],
            }
            for state in source_states
        ]
        if mode == "conservative":
            mode_text = (
                "Current round: conservative_refinement. Primarily refine entities whose visual grounding prompt failed. "
                "Keep successful prompts unchanged by default. If a successful prompt is clearly too broad, too narrow, or "
                "semantically risky, you may cautiously propose replacements, but set replace_successful=true and explain "
                "why. If those replacements fail, the service keeps the previous successful prompt. For each failed entity, "
                f"propose 1 to {max_candidates_per_source} close semantic or visually precise aliases. Do not casually change "
                "the target object's category or identity. When the image provides evidence and it is necessary for SAM grounding, "
                "you may add, remove, or replace color, material, shape, surface, or appearance attributes, including for a "
                "single-instance entity. Good examples: black bowl -> metallic bowl, reflective bowl, dark bowl; stove control -> "
                "control knob, black knob, round black control; chocolate pudding -> chocolate dessert, pudding cup, dessert; "
                "alphabet soup -> soup can, can; salad dressing -> dressing bottle, bottle."
            )
        else:
            mode_text = (
                "Current round: aggressive_refinement. Only refine entities that failed all previous visual prompts. Never "
                "replace or modify a successful prompt. Read all previous attempts, then propose 1 to "
                f"{max_candidates_per_source} broader but still task-grounded visual aliases. Good examples: cream cheese "
                "-> box, carton; alphabet soup -> can; bbq sauce -> bottle."
            )
        user_text = (
            f"Task description: {task_description}. "
            f"Task decomposition: {json.dumps(task_decomposition, ensure_ascii=True)}. "
            f"Current entity states and SAM validation results: {json.dumps(state_summary, ensure_ascii=True)}. "
            f"{mode_text} Preserve source_key, entity identity, instance distinction, role, parent relationship, and required "
            "status. Refine each source_key independently. For a single-instance entity, use ordinary semantic aliases and do "
            "not introduce an instance distinction merely based on image position. Do not casually change the target object's "
            "category or identity. Color, material, shape, surface, and appearance attributes are visual descriptions rather than "
            "object identity; they may be added, removed, or replaced when clearly supported by the image and useful for SAM. "
            "For multiple task-required instances of the same named entity, stable visual attributes or spatial qualifiers may "
            "also distinguish instances when needed. "
            "For an entity noun explicitly present in the task description, be especially cautious about adding, removing, or "
            "replacing a container-type word such as bottle, box, carton, can, jar, or package. Such a change requires clear "
            "visual evidence that the target entity itself has that container type; unrelated containers elsewhere in the image "
            "are not evidence. Preserve the entity's core semantic name in candidates whenever possible, and do not reduce it to "
            "a generic container word alone. Every candidate must remain visually and semantically tied to that exact instance. "
            "Do not propose unrelated visible objects. Output strict JSON only in exactly this form: "
            '{"replacements": [{"source_key": "cream cheese box::cream cheese box", "entity": "cream cheese box", '
            '"candidates": ["cream cheese carton", "rectangular box"], '
            '"replace_successful": false, "reason": "the initial visual prompt failed SAM validation"}]}'
        )
        messages = self._build_llm_messages(image_path=image_path, user_text=user_text)
        LOGGER.info("Extractor %s refinement round for task %r", mode, task_description)
        response_text = send_generate_request(
            messages,
            server_url=self._llm_server_url,
            model=self._llm_model,
            api_key=self._llm_api_key,
            max_tokens=self._llm_max_tokens,
            enable_thinking=False,
        )
        if not response_text:
            raise RuntimeError("LLM extractor returned no text")
        return response_text

    def _build_llm_messages(self, *, image_path: str | None, user_text: str) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        if image_path:
            content.append({"type": "image", "image": image_path})
        content.append({"type": "text", "text": user_text})
        return [
            {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

    def _parse_task_decomposition(self, response_text: str, *, task_description: str) -> dict[str, Any]:
        blob = _extract_json_blob(response_text)
        data = json.loads(blob)
        if not isinstance(data, dict):
            raise ValueError("Task decomposition must be a JSON object")
        raw_subtasks = data.get("subtasks", [])
        if not isinstance(raw_subtasks, list):
            raise ValueError("Task decomposition subtasks must be a list")

        valid_roles = {
            "manipulated_object",
            "interaction_target",
            "state_change_target",
            "destination",
            "placement_region",
            "context",
            "other",
        }
        subtasks = []
        for index, raw_subtask in enumerate(raw_subtasks, start=1):
            if not isinstance(raw_subtask, dict):
                continue
            arguments = []
            raw_arguments = raw_subtask.get("arguments", [])
            if isinstance(raw_arguments, dict):
                raw_arguments = [raw_arguments]
            for raw_argument in raw_arguments if isinstance(raw_arguments, list) else []:
                if not isinstance(raw_argument, dict):
                    continue
                entity = str(raw_argument.get("entity", "")).strip().lower()
                if not entity:
                    continue
                role = str(raw_argument.get("role", "other")).strip().lower()
                if role not in valid_roles:
                    role = "other"
                parent = raw_argument.get("parent_entity")
                parent = str(parent).strip().lower() if parent is not None and str(parent).strip() else None
                state_change = raw_argument.get("state_change")
                state_change = str(state_change).strip().lower() if state_change is not None and str(state_change).strip() else None
                arguments.append(
                    {
                        "entity": entity,
                        "role": role,
                        "directly_contacted": _as_bool(raw_argument.get("directly_contacted")),
                        "state_change": state_change,
                        "inferred": _as_bool(raw_argument.get("inferred")),
                        "parent_entity": parent,
                        "required": _as_bool(raw_argument.get("required")),
                    }
                )
            subtasks.append(
                {
                    "subtask_id": str(raw_subtask.get("subtask_id") or f"subtask_{index}").strip(),
                    "action": str(raw_subtask.get("action", "other")).strip().lower(),
                    "description": str(raw_subtask.get("description", "")).strip(),
                    "arguments": arguments,
                }
            )
        return {"task": str(data.get("task") or task_description).strip(), "subtasks": subtasks}

    def _apply_task_metadata(
        self, sources: list[dict[str, Any]], task_decomposition: dict[str, Any]
    ) -> list[dict[str, Any]]:
        metadata: dict[str, dict[str, Any]] = {}
        entity_order: list[str] = []
        for subtask in task_decomposition.get("subtasks", []):
            for argument in subtask.get("arguments", []):
                entity = argument.get("entity")
                if not entity:
                    continue
                if entity not in metadata:
                    metadata[entity] = dict(argument)
                    entity_order.append(entity)
                current = metadata[entity]
                current["required"] = bool(current.get("required") or argument.get("required"))
                current["directly_contacted"] = bool(
                    current.get("directly_contacted") or argument.get("directly_contacted")
                )
                current["inferred"] = bool(current.get("inferred") and argument.get("inferred"))
                if not current.get("parent_entity") and argument.get("parent_entity"):
                    current["parent_entity"] = argument["parent_entity"]
                if not current.get("state_change") and argument.get("state_change"):
                    current["state_change"] = argument["state_change"]

        grounded = []
        grounded_entities: set[str] = set()
        for source in sources:
            task_metadata = metadata.get(source["entity"])
            if task_metadata is None:
                LOGGER.warning("Ignoring visual source %r because it is absent from task decomposition", source["entity"])
                continue
            source = dict(source)
            source["source_role"] = task_metadata.get("role", source.get("source_role", "unknown"))
            for key in ("directly_contacted", "state_change", "inferred", "parent_entity", "required"):
                source[key] = task_metadata.get(key)
            grounded.append(source)
            grounded_entities.add(source["entity"])

        for entity in entity_order:
            task_metadata = metadata[entity]
            if not task_metadata.get("required") or entity in grounded_entities:
                continue
            LOGGER.warning(
                "Visual grounding omitted required entity %r; using its functional name as the fallback prompt", entity
            )
            grounded.append(
                {
                    "entity": entity,
                    "source_role": task_metadata.get("role", "other"),
                    "visual_prompt": entity,
                    "directly_contacted": task_metadata.get("directly_contacted", False),
                    "state_change": task_metadata.get("state_change"),
                    "inferred": task_metadata.get("inferred", False),
                    "parent_entity": task_metadata.get("parent_entity"),
                    "required": True,
                }
            )
        return sorted(grounded, key=lambda source: not source["required"])

    def _parse_source_prompt_candidates(self, response_text: str) -> list[dict[str, Any]]:
        blob = _extract_json_blob(response_text)
        data = json.loads(blob)
        if isinstance(data, dict):
            raw_sources = data.get("source_prompts", data.get("sources", data.get("prompts", data.get("objects", []))))
        elif isinstance(data, list):
            raw_sources = data
        else:
            raw_sources = []
        if isinstance(raw_sources, (str, dict)):
            raw_sources = [raw_sources]

        sources: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for item in raw_sources:
            if not isinstance(item, dict):
                continue
            entity = str(item.get("entity", item.get("source_prompt", ""))).strip().lower()
            visual_prompt = str(
                item.get("visual_prompt", item.get("prompt", item.get("text", item.get("name", entity))))
            ).strip().lower()
            instance_key = (entity, visual_prompt)
            if not entity or not visual_prompt or instance_key in seen:
                continue
            parent = item.get("parent_entity")
            parent = str(parent).strip().lower() if parent is not None and str(parent).strip() else None
            state_change = item.get("state_change")
            state_change = str(state_change).strip().lower() if state_change is not None and str(state_change).strip() else None
            sources.append(
                {
                    "entity": entity,
                    "source_role": str(item.get("source_role", item.get("role", "unknown"))).strip().lower() or "unknown",
                    "visual_prompt": visual_prompt,
                    "directly_contacted": _as_bool(item.get("directly_contacted")),
                    "state_change": state_change,
                    "inferred": _as_bool(item.get("inferred")),
                    "parent_entity": parent,
                    "required": _as_bool(item.get("required")),
                }
            )
            seen.add(instance_key)
        return sorted(sources, key=lambda source: not source["required"])

    def _parse_replacement_candidates(
        self,
        response_text: str,
        *,
        valid_sources: list[str],
    ) -> dict[str, dict[str, Any]]:
        valid_lookup = {str(source).strip().lower(): str(source).strip().lower() for source in valid_sources}
        blob = _extract_json_blob(response_text)
        data = json.loads(blob)
        if isinstance(data, dict) and ("source_key" in data or "entity" in data or "source_prompt" in data):
            raw_replacements = [data]
        elif isinstance(data, dict):
            raw_replacements = data.get("replacements", data.get("refinements", data.get("sources", [])))
            if isinstance(raw_replacements, dict):
                raw_replacements = [
                    {"source_key": source, "candidates": candidates}
                    for source, candidates in raw_replacements.items()
                ]
        elif isinstance(data, list):
            raw_replacements = data
        else:
            raw_replacements = []
        if isinstance(raw_replacements, dict):
            raw_replacements = [raw_replacements]

        replacements: dict[str, dict[str, Any]] = {}
        for item in raw_replacements:
            if not isinstance(item, dict):
                continue
            source = str(
                item.get("source_key", item.get("entity", item.get("source_prompt", item.get("source", item.get("original_prompt", item.get("prompt", ""))))))
            ).strip().lower()
            if source not in valid_lookup:
                continue
            raw_candidates = item.get(
                "candidates",
                item.get("replacement_prompts", item.get("replacements", item.get("aliases", item.get("candidate", [])))),
            )
            if isinstance(raw_candidates, str):
                raw_candidates = [raw_candidates]
            candidates = _dedupe([str(candidate).strip().lower() for candidate in raw_candidates if str(candidate).strip()])
            replace_successful = item.get("replace_successful", False)
            if isinstance(replace_successful, str):
                replace_successful = replace_successful.strip().lower() in {"true", "yes", "1"}
            replacements[source] = {
                "candidates": candidates,
                "replace_successful": bool(replace_successful),
                "reason": str(item.get("reason", "")),
            }
        return replacements



class SamRequestHandler(BaseHTTPRequestHandler):
    service: SamAgentService

    def do_GET(self) -> None:
        if self.path != "/health":
            self._write_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
            return
        self._write_json({"ok": True, "extractor_enabled": self.service.extractor_enabled()})

    def do_POST(self) -> None:
        try:
            payload = self._read_json()
            if self.path == "/segment":
                image = _decode_image(payload["image"])
                object_prompt = payload.get("object") or payload.get("prompt") or ""
                result = self.service.segment(
                    image,
                    str(object_prompt),
                    score_threshold=float(payload.get("score_threshold", 0.35)),
                    max_masks=int(payload.get("max_masks", 3)),
                )
                self._write_json(result)
                return
            if self.path == "/extract":
                encoded_image = payload.get("image")
                image = _decode_image(encoded_image) if encoded_image else None
                result = self.service.extract(
                    str(payload.get("task_description", "")),
                    image,
                    max_prompts=int(payload.get("max_prompts", 0) or 0) or None,
                    max_rounds=int(payload.get("max_rounds", 0) or 0) or None,
                )
                self._write_json(result)
                return
            self._write_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
        except Exception as exc:
            LOGGER.exception("Failed to process request for %s", self.path)
            self._write_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, fmt: str, *args: Any) -> None:
        LOGGER.info("%s - %s", self.address_string(), fmt % args)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def _write_json(self, payload: dict[str, Any], *, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Persistent SAM3 segmentation and extraction server for OpenPI LIBERO.")
    parser.add_argument("--checkpoint-path", default="/root/autodl-tmp/sam3_model/sam3.pt")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9001)
    parser.add_argument("--device", default=None)
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--llm-server-url", default=os.environ.get("SAM3_AGENT_LLM_SERVER_URL", DEFAULT_LLM_SERVER_URL))
    parser.add_argument("--llm-model", default=os.environ.get("SAM3_AGENT_MODEL", DEFAULT_LLM_MODEL))
    parser.add_argument("--llm-api-key", default=os.environ.get("SAM3_AGENT_API_KEY", DEFAULT_LLM_API_KEY))
    parser.add_argument("--llm-api-key-file", default=os.environ.get("SAM3_AGENT_API_KEY_FILE", DEFAULT_LLM_API_KEY_FILE))
    parser.add_argument("--llm-max-tokens", type=int, default=int(os.environ.get("SAM3_AGENT_MAX_TOKENS", "4096")))
    parser.add_argument("--extract-max-prompts", type=int, default=int(os.environ.get("SAM3_EXTRACT_MAX_PROMPTS", "5")))
    parser.add_argument("--extract-max-rounds", type=int, default=int(os.environ.get("SAM3_EXTRACT_MAX_ROUNDS", "4")))
    parser.add_argument("--extract-accept-score-threshold", type=float, default=float(os.environ.get("SAM3_EXTRACT_ACCEPT_SCORE_THRESHOLD", "0.5")))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, force=True)
    llm_api_key = _normalize_api_key(args.llm_api_key) or _read_api_key_file(args.llm_api_key_file)
    if args.llm_api_key:
        LOGGER.info("Using LLM API key from --llm-api-key or SAM3_AGENT_API_KEY.")
    elif llm_api_key:
        LOGGER.info("Using LLM API key from %s.", args.llm_api_key_file)
    else:
        LOGGER.warning("No LLM API key configured. /extract requests may fail.")

    SamRequestHandler.service = SamAgentService(
        args.checkpoint_path,
        device=args.device,
        confidence_threshold=args.confidence_threshold,
        llm_server_url=args.llm_server_url,
        llm_model=args.llm_model,
        llm_api_key=llm_api_key,
        llm_max_tokens=args.llm_max_tokens,
        extract_max_prompts=args.extract_max_prompts,
        extract_max_rounds=args.extract_max_rounds,
        extract_accept_score_threshold=args.extract_accept_score_threshold,
    )
    server = ThreadingHTTPServer((args.host, args.port), SamRequestHandler)
    LOGGER.info(
        "Serving SAM agent service at http://%s:%s with endpoints /segment and /extract (extractor_enabled=%s)",
        args.host,
        args.port,
        SamRequestHandler.service.extractor_enabled(),
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
