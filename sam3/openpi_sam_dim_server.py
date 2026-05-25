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

# Default LLM config for prompt extraction. You can edit these directly in code.
DEFAULT_LLM_SERVER_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_LLM_MODEL = "qwen3-vl-8b-instruct"
DEFAULT_LLM_API_KEY = "sk-731781a388b64e22b90341f192834152"


EXTRACT_SYSTEM_PROMPT = """
You help convert a robot manipulation task into simple noun phrases that are suitable for SAM-style segmentation.
Follow these rules inspired by the SAM3 agent:
- Return only the primary physical objects that matter for the manipulation task.
- Use short simple noun phrases, not full referring expressions.
- No articles, no possessives, no numbers, no verbs.
- Prefer whole objects, not object parts.
- If the task mentions relationships, containers, or supports, keep the actual object categories.
- If a phrase is too specific for segmentation, generalize it slightly while keeping visual usefulness.
- Use the image to resolve colors or object categories when the task text is ambiguous.
- Be conservative when rewriting task wording. Prefer the original object category whenever it is reasonably compatible with the scene.
- Only replace a prompt when the original wording is clearly mismatched with the scene, too niche for SAM, or very unlikely to segment successfully.
- Do not casually change colors. By default, return the object category without a color modifier.
- Only add or correct a color when color is clearly visible, materially disambiguates the object, or the original color description is obviously wrong.
- If the simulator name is odd or niche and likely to fail, replace it with a nearby visual alias or broader category, but keep the replacement semantically close.
- Good replacements are conservative object aliases such as bowl, plate, cup, mug, drawer handle, dessert, candy bar. Use color+object only when needed.
- If a task object may be recognized under multiple reasonable visual names, prefer the simplest and most segmentable object phrase first.
- Output strict JSON of the form {"prompts": ["prompt one", "prompt two"]} and nothing else.
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
        stop_after_successful_prompts = min(2, max_prompts)
        if not task_description.strip():
            return {"prompts": [], "used_prompts": [], "extractor_enabled": self.extractor_enabled()}
        if not self.extractor_enabled():
            LOGGER.warning("/extract requested but LLM server/model is not configured.")
            return {
                "prompts": [],
                "used_prompts": [],
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
            modes = ["importance_only", "conservative", "aggressive"][:max_rounds]

            last_mode = modes[-1]
            for round_idx, mode in enumerate(modes):
                response_text = self._generate_candidate_prompts(
                    image_path=image_path,
                    task_description=task_description,
                    used_prompts=used,
                    failed_prompts=[entry["prompt"] for entry in evaluation_trace if not entry["accepted"]],
                    remaining_slots=candidate_budget,
                    round_idx=round_idx,
                    mode=mode,
                )
                candidates = self._parse_prompt_candidates(response_text)
                candidates = [p for p in candidates if p not in used]
                if not candidates:
                    LOGGER.info("Extractor %s round produced no new prompts.", mode)
                    continue

                accepted_in_order: list[str] = []
                for priority, prompt in enumerate(candidates, start=1):
                    used.append(prompt)
                    mask, scores = self._predict_mask(
                        image,
                        prompt,
                        score_threshold=self._extract_accept_score_threshold,
                        max_masks=3,
                        allow_fallback=False,
                    )
                    best_score = max(scores) if scores else 0.0
                    accepted = bool(mask.any())
                    evaluation_trace.append(
                        {
                            "mode": mode,
                            "priority": priority,
                            "prompt": prompt,
                            "scores": [float(score) for score in scores],
                            "best_score": float(best_score),
                            "accepted": accepted,
                        }
                    )
                    if accepted:
                        accepted_in_order.append(prompt)
                        LOGGER.info(
                            "Extractor accepted prompt %r in %s mode at priority %s with best_score=%.3f and scores=%s",
                            prompt,
                            mode,
                            priority,
                            best_score,
                            scores,
                        )
                    else:
                        LOGGER.info(
                            "Extractor rejected prompt %r in %s mode at priority %s with best_score=%.3f and scores=%s (need best_score >= %.2f)",
                            prompt,
                            mode,
                            priority,
                            best_score,
                            scores,
                            self._extract_accept_score_threshold,
                        )
                    if len(accepted_in_order) >= max_prompts:
                        break

                if len(accepted_in_order) >= stop_after_successful_prompts:
                    return {
                        "prompts": accepted_in_order[:max_prompts],
                        "used_prompts": _dedupe(used),
                        "extractor_enabled": True,
                        "mode_used": mode,
                        "evaluation_trace": evaluation_trace,
                    }

            final_prompts = [entry["prompt"] for entry in evaluation_trace if entry["accepted"]][:max_prompts]
            return {
                "prompts": final_prompts,
                "used_prompts": _dedupe(used),
                "extractor_enabled": True,
                "mode_used": last_mode,
                "evaluation_trace": evaluation_trace,
            }
        finally:
            if image_path and os.path.exists(image_path):
                os.remove(image_path)

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

    def _generate_candidate_prompts(
        self,
        *,
        image_path: str | None,
        task_description: str,
        used_prompts: list[str],
        failed_prompts: list[str],
        remaining_slots: int,
        round_idx: int,
        mode: str,
    ) -> str:
        retry_text = ""
        if used_prompts:
            retry_text += f" Previously tried prompts: {used_prompts}."
        if failed_prompts:
            retry_text += f" The following prompts failed to produce useful SAM masks and should not be repeated as-is: {failed_prompts}."
        if mode == "importance_only":
            mode_text = (
                " Rank the most important task-relevant object prompts from highest importance to lowest importance."
                " Use the exact object wording from the task whenever possible. Do not rewrite, simplify, paraphrase, generalize, or replace the object phrases in this stage."
                " Your job in this stage is only to decide which prompts are most important, not to improve them."
                " Return only object prompts that are explicitly supported by the task description."
            )
        elif mode == "conservative":
            mode_text = (
                " The original important prompts did not work well enough. Rank the most important task-relevant prompts from highest importance to lowest importance."
                " You may now use only conservative wording changes. Keep the original object meaning whenever possible."
                " Only do mild semantic replacement when the task wording is clearly too niche or slightly mismatched for SAM."
                " Do not casually change colors. Usually return only the object category without color."
                " Add or correct color only when it is clearly visible and necessary for disambiguation."
            )
        else:
            mode_text = (
                " The conservative prompts did not work well enough. Rank the most important task-relevant prompts from highest importance to lowest importance."
                " You may now use more aggressive but still task-relevant visual paraphrases to help SAM segment the object."
                " You may replace a niche simulator object name with a broader or more segmentable visual description such as dessert, candy bar, black object, bowl, or cup when appropriate."
                " Keep replacements semantically close to the intended target role in the task."
            )
        user_text = (
            f"Task description: {task_description}."
            f" Return at most {remaining_slots} prompts in ranked order of importance."
            " Focus on the manipulated object(s), target receptacle/support, or other primary physical objects needed for the task."
            " Do not return relations like left/right/on top of/in front of as standalone prompts."
            " Return prompts ordered from most important to least important."
            " Preserve importance ordering in your returned prompt list."
            " Prefer prompts that SAM can segment reliably over exact wording from the task description, except in importance_only mode where wording should stay unchanged."
            " Output strict JSON only in the form {\"prompts\": [\"most important\", \"second\", \"third\"]}."
            f"{mode_text}"
            f"{retry_text}"
        )
        content: list[dict[str, Any]] = []
        if image_path:
            content.append({"type": "image", "image": image_path})
        content.append({"type": "text", "text": user_text})
        messages = [
            {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        LOGGER.info("Extractor round %s for task %r", round_idx + 1, task_description)
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

    def _parse_prompt_candidates(self, response_text: str) -> list[str]:
        blob = _extract_json_blob(response_text)
        data = json.loads(blob)
        if isinstance(data, dict):
            prompts = data.get("prompts", data.get("objects", []))
        elif isinstance(data, list):
            prompts = data
        else:
            prompts = []
        if isinstance(prompts, str):
            prompts = [prompts]
        return _dedupe([str(prompt).strip().lower() for prompt in prompts if str(prompt).strip()])


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
    parser.add_argument("--llm-max-tokens", type=int, default=int(os.environ.get("SAM3_AGENT_MAX_TOKENS", "512")))
    parser.add_argument("--extract-max-prompts", type=int, default=int(os.environ.get("SAM3_EXTRACT_MAX_PROMPTS", "3")))
    parser.add_argument("--extract-max-rounds", type=int, default=int(os.environ.get("SAM3_EXTRACT_MAX_ROUNDS", "3")))
    parser.add_argument("--extract-accept-score-threshold", type=float, default=float(os.environ.get("SAM3_EXTRACT_ACCEPT_SCORE_THRESHOLD", "0.5")))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, force=True)
    SamRequestHandler.service = SamAgentService(
        args.checkpoint_path,
        device=args.device,
        confidence_threshold=args.confidence_threshold,
        llm_server_url=args.llm_server_url,
        llm_model=args.llm_model,
        llm_api_key=args.llm_api_key,
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
