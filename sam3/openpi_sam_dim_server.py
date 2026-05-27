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
from PIL import Image
import torch

from sam3.agent.client_llm import send_generate_request
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


LOGGER = logging.getLogger("openpi_sam_dim_server")

# Default LLM config for prompt extraction. Keep secrets out of source code.
DEFAULT_LLM_SERVER_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_LLM_MODEL = "qwen3-vl-8b-instruct"
DEFAULT_LLM_API_KEY = ""
DEFAULT_LLM_API_KEY_FILE = "/root/proj/qwen_api_key.txt"


EXTRACT_SYSTEM_PROMPT = """
You are a robot manipulation prompt refinement assistant. Your job is to map task-relevant physical objects to short SAM-segmentable noun phrases while preserving source-object identity.
Core rules:
- Keep every later prompt tied to a source_prompt derived from the original robot task.
- Use short simple noun phrases, not full referring expressions.
- No articles, no possessives, no numbers, no verbs.
- Prefer whole physical objects over object parts, unless the task truly requires an articulated part such as microwave door.
- Focus on manipulated objects, target receptacles/supports, and necessary articulated parts.
- Do not introduce unrelated visible objects just because they are easy to segment.
- Do not casually change colors. Add color only when it is clearly visible and needed for disambiguation.
- Conservative replacements should stay very close to the source prompt.
- Aggressive replacements may use broader visual aliases, but only for source prompts that failed in previous rounds.
- Output strict JSON only, in the exact schema requested by the current round.
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


def _dedupe(items: list[str]) -> list[str]:
    deduped: list[str] = []
    for item in items:
        normalized = str(item).strip()
        if normalized and normalized not in deduped:
            deduped.append(normalized)
    return deduped


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
        llm_max_tokens: int = 512,
        extract_max_prompts: int = 3,
        extract_max_rounds: int = 3,
        extract_accept_score_threshold: float = 0.5,
    ):
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        LOGGER.info("Loading SAM3 image model from %s on %s", checkpoint_path, self._device)
        model = build_sam3_image_model(checkpoint_path=checkpoint_path, device=self._device)
        self._processor = Sam3Processor(model, confidence_threshold=confidence_threshold)
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
        mask, scores = self._predict_mask(
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
        max_prompts = min(max_prompts or self._extract_max_prompts, 3)
        max_rounds = max_rounds or self._extract_max_rounds
        if not task_description.strip():
            return {
                "prompts": [],
                "used_prompts": [],
                "source_prompts": [],
                "prompt_trace": [],
                "extractor_enabled": self.extractor_enabled(),
            }
        if not self.extractor_enabled():
            LOGGER.warning("/extract requested but LLM server/model is not configured.")
            return {
                "prompts": [],
                "used_prompts": [],
                "source_prompts": [],
                "prompt_trace": [],
                "extractor_enabled": False,
                "reason": "Set --llm-server-url and --llm-model on the SAM server.",
            }

        image_path = None
        if image is not None:
            tmp = tempfile.NamedTemporaryFile(prefix="sam3_extract_", suffix=".png", delete=False)
            image_path = tmp.name
            tmp.close()
            Image.fromarray(image, mode="RGB").save(image_path)

        try:
            used: list[str] = []
            evaluation_trace: list[dict[str, Any]] = []
            candidate_budget = max(max_prompts, min(6, max_prompts * 2))

            response_text = self._generate_source_prompts(
                image_path=image_path,
                task_description=task_description,
                remaining_slots=candidate_budget,
            )
            source_candidates = self._parse_source_prompt_candidates(response_text)[:candidate_budget]
            source_states: list[dict[str, Any]] = []
            for priority, source in enumerate(source_candidates, start=1):
                state = {
                    "source_prompt": source["source_prompt"],
                    "source_role": source.get("source_role", "unknown"),
                    "selected_prompt": None,
                    "selected_round": None,
                    "selected_score": 0.0,
                    "attempts": [],
                }
                source_states.append(state)
                attempt = self._evaluate_source_candidate(
                    image=image,
                    state=state,
                    prompt=state["source_prompt"],
                    mode="importance_only",
                    priority=priority,
                    used=used,
                    evaluation_trace=evaluation_trace,
                )
                if attempt["accepted"]:
                    self._select_attempt(state, attempt)

            if self._selected_prompt_count(source_states) >= max_prompts or max_rounds <= 1:
                return self._build_extract_response(
                    source_states,
                    used_prompts=used,
                    evaluation_trace=evaluation_trace,
                    max_prompts=max_prompts,
                    fallback_mode="importance_only",
                )

            if max_rounds >= 2:
                self._run_refinement_round(
                    image=image,
                    image_path=image_path,
                    task_description=task_description,
                    source_states=source_states,
                    used=used,
                    evaluation_trace=evaluation_trace,
                    mode="conservative",
                    max_candidates_per_source=3,
                    allow_successful_replacements=True,
                )
            if self._selected_prompt_count(source_states) >= max_prompts or max_rounds <= 2:
                return self._build_extract_response(
                    source_states,
                    used_prompts=used,
                    evaluation_trace=evaluation_trace,
                    max_prompts=max_prompts,
                    fallback_mode="conservative",
                )

            if max_rounds >= 3:
                self._run_refinement_round(
                    image=image,
                    image_path=image_path,
                    task_description=task_description,
                    source_states=source_states,
                    used=used,
                    evaluation_trace=evaluation_trace,
                    mode="aggressive",
                    max_candidates_per_source=5,
                    allow_successful_replacements=False,
                )
            return self._build_extract_response(
                source_states,
                used_prompts=used,
                evaluation_trace=evaluation_trace,
                max_prompts=max_prompts,
                fallback_mode="aggressive",
            )
        finally:
            if image_path and os.path.exists(image_path):
                os.remove(image_path)

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
        mask, scores = self._predict_mask(
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
            "source_prompt": state["source_prompt"],
            "source_role": state.get("source_role", "unknown"),
            "prompt": prompt,
            "scores": [float(score) for score in scores],
            "best_score": float(best_score),
            "accepted": bool(mask.any()),
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

    def _selected_prompt_count(self, source_states: list[dict[str, Any]]) -> int:
        return len(_dedupe([state["selected_prompt"] for state in source_states if state.get("selected_prompt")]))

    def _run_refinement_round(
        self,
        *,
        image: np.ndarray | None,
        image_path: str | None,
        task_description: str,
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
            source_states=source_states,
            mode=mode,
            max_candidates_per_source=max_candidates_per_source,
            allow_successful_replacements=allow_successful_replacements,
        )
        valid_sources = [state["source_prompt"] for state in source_states]
        if mode == "aggressive":
            valid_sources = [state["source_prompt"] for state in failed_states]
        replacement_map = self._parse_replacement_candidates(response_text, valid_sources=valid_sources)
        state_by_source = {state["source_prompt"]: state for state in source_states}
        for source_prompt, replacement_info in replacement_map.items():
            state = state_by_source.get(source_prompt)
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
                if candidate and candidate not in tried_for_source and candidate != source_prompt
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
                    source_prompt,
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
        used_prompts: list[str],
        evaluation_trace: list[dict[str, Any]],
        max_prompts: int,
        fallback_mode: str,
    ) -> dict[str, Any]:
        selected_states = [state for state in source_states if state.get("selected_prompt")]
        prompts = _dedupe([state["selected_prompt"] for state in selected_states])[:max_prompts]
        selected_modes = [state["selected_round"] for state in selected_states if state.get("selected_round")]
        mode_used = selected_modes[-1] if selected_modes else fallback_mode
        prompt_trace = []
        for state in source_states:
            prompt_trace.append(
                {
                    "source_prompt": state["source_prompt"],
                    "source_role": state.get("source_role", "unknown"),
                    "selected_prompt": state.get("selected_prompt"),
                    "selected_round": state.get("selected_round"),
                    "selected_score": float(state.get("selected_score", 0.0)),
                    "status": "selected" if state.get("selected_prompt") else "failed",
                    "attempts": state.get("attempts", []),
                }
            )
        return {
            "prompts": prompts,
            "used_prompts": _dedupe(used_prompts),
            "source_prompts": [
                {"source_prompt": state["source_prompt"], "source_role": state.get("source_role", "unknown")}
                for state in source_states
            ],
            "prompt_trace": prompt_trace,
            "extractor_enabled": True,
            "mode_used": mode_used,
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
    ) -> tuple[np.ndarray, list[float]]:
        if image is None:
            return np.zeros((1, 1), dtype=bool), []
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
    ) -> tuple[np.ndarray, list[float]]:
        object_prompt = object_prompt.strip()
        if not object_prompt:
            return np.zeros((height, width), dtype=bool), []
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
    ) -> tuple[np.ndarray, list[float]]:
        masks = output.get("masks")
        if masks is None or len(masks) == 0:
            return np.zeros((height, width), dtype=bool), []

        masks_np = masks.detach().cpu().numpy()
        if masks_np.ndim == 4 and masks_np.shape[1] == 1:
            masks_np = masks_np[:, 0]

        scores = output.get("scores")
        if scores is None:
            scores_np = np.ones((masks_np.shape[0],), dtype=np.float32)
        else:
            scores_np = scores.detach().float().cpu().numpy().reshape(-1)

        order = np.argsort(scores_np)[::-1]
        keep = [idx for idx in order if scores_np[idx] >= score_threshold]
        if not keep:
            if allow_fallback and len(order) > 0:
                keep = [int(order[0])]
            else:
                return np.zeros((height, width), dtype=bool), []
        keep = keep[: max(1, max_masks)]

        union_mask = np.zeros((height, width), dtype=bool)
        kept_scores = []
        for idx in keep:
            union_mask |= _resize_mask(masks_np[idx], height, width)
            kept_scores.append(float(scores_np[idx]))
        return union_mask, kept_scores

    def _generate_source_prompts(
        self,
        *,
        image_path: str | None,
        task_description: str,
        remaining_slots: int,
    ) -> str:
        user_text = (
            f"Task description: {task_description}. "
            f"Return at most {remaining_slots} source prompts in ranked order of task importance. "
            "This is the importance_only round. Use exact object wording from the task whenever possible. "
            "Do not rewrite, simplify, paraphrase, generalize, or replace object phrases in this round. "
            "Return primary manipulated objects, target receptacles/supports, and necessary articulated parts only. "
            "If the task uses an implicit object part that is necessary for manipulation, such as close it for microwave, you may include a conservative articulated part like microwave door. "
            "Assign source_role as one of manipulated_object, target_receptacle, support, articulated_part, or other. "
            "Output strict JSON only in this form: "
            "{\"source_prompts\": [{\"source_prompt\": \"alphabet soup\", \"source_role\": \"manipulated_object\"}, "
            "{\"source_prompt\": \"basket\", \"source_role\": \"target_receptacle\"}]}"
        )
        messages = self._build_llm_messages(image_path=image_path, user_text=user_text)
        LOGGER.info("Extractor source round for task %r", task_description)
        response_text = send_generate_request(
            messages,
            server_url=self._llm_server_url,
            model=self._llm_model,
            api_key=self._llm_api_key,
            max_tokens=self._llm_max_tokens,
        )
        if not response_text:
            raise RuntimeError("LLM extractor returned no text")
        return response_text

    def _generate_replacement_prompts(
        self,
        *,
        image_path: str | None,
        task_description: str,
        source_states: list[dict[str, Any]],
        mode: str,
        max_candidates_per_source: int,
        allow_successful_replacements: bool,
    ) -> str:
        state_summary = [
            {
                "source_prompt": state["source_prompt"],
                "source_role": state.get("source_role", "unknown"),
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
                "This is the conservative refinement round. Only refine source_prompts whose first-round attempt failed. "
                "Source prompts that already succeeded should be kept unchanged by default. If a successful source prompt is clearly too broad, too narrow, or semantically risky, you may cautiously propose replacements for it, but set replace_successful to true and explain why. "
                "If those cautious replacements fail, the service will fall back to the previously successful prompt. "
                "For each failed source_prompt, propose 1 to "
                f"{max_candidates_per_source} close semantic aliases derived from that source_prompt and the task. "
                "Good conservative examples: chocolate pudding -> chocolate dessert, pudding cup, dessert; "
                "alphabet soup -> soup can, can; salad dressing -> dressing bottle, bottle. "
                "Do not propose unrelated visible objects."
            )
        else:
            mode_text = (
                "This is the aggressive refinement round. Only refine source_prompts that failed in all previous rounds. "
                "Never replace or modify any source_prompt that already has selected_prompt. "
                "For each still-failed source_prompt, read the previous attempts and failures, then propose 1 to "
                f"{max_candidates_per_source} broader but still task-derived visual aliases. "
                "Good aggressive examples: cream cheese -> box, carton; alphabet soup -> can; bbq sauce -> bottle. "
                "Even in aggressive mode, every candidate must remain semantically tied to its source_prompt and task role. "
                "Do not propose unrelated objects such as robot arm, floor, phone, or toilet unless the source_prompt itself really refers to that object."
            )
        user_text = (
            f"Task description: {task_description}. "
            f"Current source prompt states and SAM validation results: {json.dumps(state_summary, ensure_ascii=True)}. "
            f"{mode_text} "
            "Output strict JSON only in this form: "
            "{\"replacements\": [{\"source_prompt\": \"alphabet soup\", \"candidates\": [\"soup can\", \"can\"], "
            "\"replace_successful\": false, \"reason\": \"original source prompt failed in SAM\"}]}"
        )
        messages = self._build_llm_messages(image_path=image_path, user_text=user_text)
        LOGGER.info("Extractor %s refinement round for task %r", mode, task_description)
        response_text = send_generate_request(
            messages,
            server_url=self._llm_server_url,
            model=self._llm_model,
            api_key=self._llm_api_key,
            max_tokens=self._llm_max_tokens,
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

    def _parse_source_prompt_candidates(self, response_text: str) -> list[dict[str, str]]:
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

        sources: list[dict[str, str]] = []
        seen: set[str] = set()
        for item in raw_sources:
            if isinstance(item, dict):
                prompt = item.get("source_prompt", item.get("prompt", item.get("text", item.get("name", ""))))
                role = item.get("source_role", item.get("role", "unknown"))
            else:
                prompt = item
                role = "unknown"
            prompt = str(prompt).strip().lower()
            role = str(role).strip().lower() or "unknown"
            if prompt and prompt not in seen:
                sources.append({"source_prompt": prompt, "source_role": role})
                seen.add(prompt)
        return sources

    def _parse_replacement_candidates(
        self,
        response_text: str,
        *,
        valid_sources: list[str],
    ) -> dict[str, dict[str, Any]]:
        valid_lookup = {str(source).strip().lower(): str(source).strip().lower() for source in valid_sources}
        blob = _extract_json_blob(response_text)
        data = json.loads(blob)
        if isinstance(data, dict) and "source_prompt" in data:
            raw_replacements = [data]
        elif isinstance(data, dict):
            raw_replacements = data.get("replacements", data.get("refinements", data.get("sources", [])))
            if isinstance(raw_replacements, dict):
                raw_replacements = [
                    {"source_prompt": source, "candidates": candidates}
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
                item.get("source_prompt", item.get("source", item.get("original_prompt", item.get("prompt", ""))))
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
    parser.add_argument("--llm-max-tokens", type=int, default=int(os.environ.get("SAM3_AGENT_MAX_TOKENS", "512")))
    parser.add_argument("--extract-max-prompts", type=int, default=int(os.environ.get("SAM3_EXTRACT_MAX_PROMPTS", "3")))
    parser.add_argument("--extract-max-rounds", type=int, default=int(os.environ.get("SAM3_EXTRACT_MAX_ROUNDS", "3")))
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
