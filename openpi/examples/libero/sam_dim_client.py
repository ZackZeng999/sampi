"""Client for SAM-based dim-background pre-processing in LIBERO eval."""

from __future__ import annotations

import base64
import dataclasses
import io
import json
import logging
import urllib.error
import urllib.request

import numpy as np
from PIL import Image
from PIL import ImageFilter


LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class SamDimClient:
    sam_url: str
    extract_url: str = ""
    prompt: str | None = None
    background_scale: float = 0.35
    score_threshold: float = 0.35
    max_masks_per_prompt: int = 3
    blur_radius: float = 1.5
    timeout_sec: float = 20.0
    fail_open: bool = True

    def extract_prompts(self, task_description: str) -> list[str]:
        """Return single-object prompts for SAM without an image."""
        return self.extract_prompts_for_image(task_description, image=None)

    def extract_prompts_for_image(self, task_description: str, image: np.ndarray | None) -> list[str]:
        """Return single-object prompts for SAM using an optional reference image."""

        if self.prompt:
            return _dedupe([part.strip() for part in self.prompt.split(",") if part.strip()])
        extract_url = self.extract_url or _derive_extract_url(self.sam_url)
        if extract_url:
            return self._extract_prompts_from_url(extract_url, task_description, image)
        _ = task_description
        return []

    def dim_background_with_prompts(self, image: np.ndarray, prompts: list[str]) -> np.ndarray:
        prompts = _dedupe([str(prompt).strip() for prompt in prompts if str(prompt).strip()])
        if not prompts:
            return image

        union_mask = np.zeros(image.shape[:2], dtype=bool)
        for prompt in prompts:
            union_mask |= self.segment_object(image, prompt)

        if not union_mask.any():
            LOGGER.warning("SAM did not find any masks. Prompts used: %s", prompts)
            return image
        return _dim_background(image, union_mask, background_scale=self.background_scale, blur_radius=self.blur_radius)

    def segment_object(self, image: np.ndarray, object_prompt: str) -> np.ndarray:
        payload = {
            "image": _encode_image(image),
            "object": object_prompt,
            "score_threshold": self.score_threshold,
            "max_masks": self.max_masks_per_prompt,
        }
        try:
            result = self._post_json(self.sam_url, payload)
            if "error" in result:
                raise RuntimeError(result["error"])
            if not result.get("mask_found", False):
                LOGGER.warning("SAM did not find a mask for object prompt: %s", object_prompt)
            return _decode_mask(result["mask"], image.shape[:2])
        except (OSError, RuntimeError, urllib.error.URLError, TimeoutError) as exc:
            if not self.fail_open:
                raise
            LOGGER.warning("SAM segment request failed for %r; ignoring this object. Error: %s", object_prompt, exc)
            return np.zeros(image.shape[:2], dtype=bool)

    def _extract_prompts_from_url(
        self,
        extract_url: str,
        task_description: str,
        image: np.ndarray | None,
    ) -> list[str]:
        try:
            payload = {"task_description": task_description}
            if image is not None:
                payload["image"] = _encode_image(image)
            result = self._post_json(extract_url, payload)
            prompts = result.get("prompts", result.get("objects", result))
            if isinstance(prompts, str):
                prompts = [prompts]
            return _dedupe([str(prompt).strip() for prompt in prompts if str(prompt).strip()])
        except (OSError, RuntimeError, urllib.error.URLError, TimeoutError, TypeError) as exc:
            if not self.fail_open:
                raise
            LOGGER.warning("Object extraction failed; using no SAM prompts. Error: %s", exc)
            return []

    def _post_json(self, url: str, payload: dict) -> dict:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
            return json.loads(response.read().decode("utf-8"))


def _dedupe(items: list[str]) -> list[str]:
    deduped = []
    for item in items:
        if item and item not in deduped:
            deduped.append(item)
    return deduped


def _derive_extract_url(sam_url: str) -> str:
    if sam_url.endswith("/segment"):
        return sam_url[: -len("/segment")] + "/extract"
    return ""


def _encode_image(image: np.ndarray) -> str:
    image = np.asarray(image)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(image, mode="RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _decode_mask(encoded: str, shape: tuple[int, int]) -> np.ndarray:
    mask = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("L")
    if mask.size != (shape[1], shape[0]):
        mask = mask.resize((shape[1], shape[0]), resample=Image.NEAREST)
    return np.asarray(mask) > 0


def _dim_background(image: np.ndarray, mask: np.ndarray, *, background_scale: float, blur_radius: float) -> np.ndarray:
    background_scale = float(np.clip(background_scale, 0.0, 1.0))
    image_f = image.astype(np.float32)
    dimmed = image_f * background_scale
    mask_alpha = _smooth_mask(mask, blur_radius)[..., None]
    output = image_f * mask_alpha + dimmed * (1.0 - mask_alpha)
    return np.clip(output, 0, 255).astype(np.uint8)


def _smooth_mask(mask: np.ndarray, radius: float) -> np.ndarray:
    if radius <= 0:
        return mask.astype(np.float32)
    mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.asarray(mask_img, dtype=np.float32) / 255.0
