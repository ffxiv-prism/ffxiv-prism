from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

from ffxiv_prism.paths import CACHE_DIR
from ffxiv_prism.torch_device import resolve_torch_device

DEFAULT_HUMAN_PARSING_MODEL = "mattmdjaga/segformer_b2_clothes"
DEFAULT_HUGGINGFACE_CACHE = CACHE_DIR / "huggingface" / "hub"

LABEL_NAMES = {
    0: "background",
    1: "hat",
    2: "hair",
    3: "sunglasses",
    4: "upper_clothes",
    5: "skirt",
    6: "pants",
    7: "dress",
    8: "belt",
    9: "left_shoe",
    10: "right_shoe",
    11: "face",
    12: "left_leg",
    13: "right_leg",
    14: "left_arm",
    15: "right_arm",
    16: "bag",
    17: "scarf",
}

SLOT_LABEL_IDS = {
    "HEAD": frozenset((1, 2, 3, 11, 17)),
    "BODY": frozenset((4, 7, 8, 14, 15, 17)),
    "HANDS": frozenset((14, 15)),
    "LEGS": frozenset((5, 6, 7, 12, 13)),
    "FEET": frozenset((9, 10, 12, 13)),
}


@dataclass(frozen=True)
class MaskResult:
    image: Image.Image
    mask: Image.Image
    person_fraction: float
    selected_fraction: float
    used_fallback: bool


@dataclass(frozen=True)
class GlobalForegroundResult:
    mask: Image.Image
    person_fraction: float
    foreground_fraction: float
    repaired_hole_fraction: float
    keypoint_coverage: float
    used_pose_envelope: bool
    quality_flags: tuple[str, ...]


class HumanParser:
    def __init__(
        self,
        model_id: str = DEFAULT_HUMAN_PARSING_MODEL,
        device: str = "auto",
        cache_dir: str | None = None,
    ) -> None:
        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

        self.model_id = model_id
        self.device = resolve_torch_device(device)
        resolved_cache = cache_dir or str(DEFAULT_HUGGINGFACE_CACHE)
        self.processor = AutoImageProcessor.from_pretrained(
            model_id,
            cache_dir=resolved_cache,
            local_files_only=True,
        )
        self.model = AutoModelForSemanticSegmentation.from_pretrained(
            model_id,
            cache_dir=resolved_cache,
            local_files_only=True,
        ).to(self.device).eval()

    @torch.inference_mode()
    def parse(self, images: Sequence[Image.Image]) -> list[np.ndarray]:
        inputs = self.processor(images=list(images), return_tensors="pt")
        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
            if isinstance(value, torch.Tensor)
        }
        outputs = self.model(**inputs)
        target_sizes = [(image.height, image.width) for image in images]
        segmentations = self.processor.post_process_semantic_segmentation(
            outputs,
            target_sizes=target_sizes,
        )
        return [
            segmentation.cpu().numpy().astype(np.uint8, copy=False)
            for segmentation in segmentations
        ]


def apply_human_mask(
    image: Image.Image,
    segmentation: np.ndarray,
    *,
    mode: str,
    slot: str,
    dilation: int = 9,
    feather: float = 2.0,
    min_person_fraction: float = 0.02,
    min_slot_share: float = 0.08,
) -> MaskResult:
    rgb = image.convert("RGB")
    if segmentation.shape != (rgb.height, rgb.width):
        raise ValueError("Segmentation shape must match the image")
    person = segmentation != 0
    person_fraction = float(person.mean())
    if mode == "person":
        selected = person
        used_fallback = False
    elif mode == "slot":
        if slot not in SLOT_LABEL_IDS:
            raise ValueError(f"Unsupported slot: {slot}")
        selected = np.isin(segmentation, tuple(SLOT_LABEL_IDS[slot]))
        selected_share = float(selected.sum() / max(person.sum(), 1))
        used_fallback = (
            person_fraction < min_person_fraction
            or selected_share < min_slot_share
        )
        if used_fallback:
            selected = person
    else:
        raise ValueError(f"Unsupported mask mode: {mode}")

    selected_fraction = float(selected.mean())
    mask = Image.fromarray((selected.astype(np.uint8) * 255), mode="L")
    if dilation > 1:
        dilation = dilation if dilation % 2 == 1 else dilation + 1
        mask = mask.filter(ImageFilter.MaxFilter(dilation))
    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(feather))
    background = Image.new("RGB", rgb.size, (0, 0, 0))
    masked = Image.composite(rgb, background, mask)
    return MaskResult(
        image=masked,
        mask=mask,
        person_fraction=person_fraction,
        selected_fraction=selected_fraction,
        used_fallback=used_fallback,
    )


def build_global_foreground_mask(
    image: Image.Image,
    segmentation: np.ndarray,
    *,
    keypoints: Sequence[Any] = (),
    dilation_fraction: float = 0.012,
    feather: float = 2.0,
    min_keypoint_score: float = 0.2,
) -> GlobalForegroundResult:
    """Build one high-recall person mask in original-image coordinates.

    Semantic parsing supplies detailed boundaries. A conservative pose envelope
    is unioned with it so fantasy armor pieces missed by the clothing parser are
    less likely to create holes through the character. The resulting mask is
    intended to be cropped later with the exact same boxes as the source image.
    """

    rgb = image.convert("RGB")
    if segmentation.shape != (rgb.height, rgb.width):
        raise ValueError("Segmentation shape must match the image")

    semantic = segmentation != 0
    person_fraction = float(semantic.mean())
    mask = Image.fromarray(semantic.astype(np.uint8) * 255, mode="L")
    dilation_radius = max(
        1,
        min(15, int(round(min(rgb.size) * max(0.0, dilation_fraction)))),
    )
    dilation_size = dilation_radius * 2 + 1
    mask = mask.filter(ImageFilter.MaxFilter(dilation_size))

    indexed_keypoints = {
        index: point
        for index, value in enumerate(keypoints)
        if (point := _normalize_keypoint(value)) is not None
        and point[2] >= min_keypoint_score
    }
    normalized_keypoints = list(indexed_keypoints.values())
    used_pose_envelope = len(indexed_keypoints) >= 6
    if used_pose_envelope:
        pose_mask = _build_pose_envelope(
            rgb.size,
            indexed_keypoints,
            dilation_radius,
        )
        mask = Image.fromarray(
            np.maximum(
                np.asarray(mask, dtype=np.uint8),
                np.asarray(pose_mask, dtype=np.uint8),
            ),
            mode="L",
        )
        pose_mask.close()

    mask, repaired_hole_fraction = _repair_mask_gaps_and_holes(mask)
    keypoint_coverage = _keypoint_coverage(
        mask,
        normalized_keypoints,
    )
    foreground_fraction = float(
        (np.asarray(mask, dtype=np.uint8) >= 128).mean()
    )
    quality_flags = []
    if person_fraction < 0.02:
        quality_flags.append("low_semantic_foreground")
    if foreground_fraction < 0.03:
        quality_flags.append("low_final_foreground")
    if foreground_fraction > 0.78:
        quality_flags.append("excessive_foreground")
    if normalized_keypoints and keypoint_coverage < 0.9:
        quality_flags.append("low_keypoint_coverage")
    if person_fraction > 0 and foreground_fraction / person_fraction > 3.0:
        quality_flags.append("excessive_mask_expansion")
    if repaired_hole_fraction > 0.05:
        quality_flags.append("large_hole_repair")
    if not used_pose_envelope:
        quality_flags.append("pose_envelope_unavailable")

    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(float(feather)))
    return GlobalForegroundResult(
        mask=mask,
        person_fraction=person_fraction,
        foreground_fraction=foreground_fraction,
        repaired_hole_fraction=repaired_hole_fraction,
        keypoint_coverage=keypoint_coverage,
        used_pose_envelope=used_pose_envelope,
        quality_flags=tuple(quality_flags),
    )


def composite_foreground(
    image: Image.Image,
    mask: Image.Image,
    background: tuple[int, int, int] = (127, 127, 127),
) -> Image.Image:
    """Composite a source image after segmentation without changing geometry."""

    rgb = image.convert("RGB")
    if mask.size != rgb.size:
        raise ValueError("Mask size must match the image")
    canvas = Image.new("RGB", rgb.size, background)
    return Image.composite(rgb, canvas, mask.convert("L"))


def transparent_foreground(image: Image.Image, mask: Image.Image) -> Image.Image:
    """Return a lossless RGBA foreground for storage or presentation."""

    rgb = image.convert("RGB")
    if mask.size != rgb.size:
        raise ValueError("Mask size must match the image")
    rgba = rgb.convert("RGBA")
    rgba.putalpha(mask.convert("L"))
    return rgba


def _normalize_keypoint(value: Any) -> tuple[float, float, float] | None:
    if isinstance(value, dict):
        try:
            return float(value["x"]), float(value["y"]), float(value.get("score", 1.0))
        except (KeyError, TypeError, ValueError):
            return None
    if hasattr(value, "x") and hasattr(value, "y"):
        try:
            return (
                float(value.x),
                float(value.y),
                float(getattr(value, "score", 1.0)),
            )
        except (TypeError, ValueError):
            return None
    if isinstance(value, Sequence) and len(value) >= 2:
        try:
            score = float(value[2]) if len(value) >= 3 else 1.0
            return float(value[0]), float(value[1]), score
        except (TypeError, ValueError):
            return None
    return None


def _build_pose_envelope(
    image_size: tuple[int, int],
    keypoints: dict[int, tuple[float, float, float]],
    dilation_radius: int,
) -> Image.Image:
    width, height = image_size
    coordinates = [(point[0], point[1]) for point in keypoints.values()]
    xs = [point[0] for point in keypoints.values()]
    span_x = max(xs) - min(xs) if len(xs) > 1 else width * 0.1
    body_width = max(width * 0.035, min(width * 0.18, span_x * 0.22))
    line_width = max(5, int(round(body_width + dilation_radius * 1.5)))
    joint_radius = max(3, line_width // 2)

    envelope = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(envelope)
    skeleton = (
        (0, 1), (0, 2), (1, 3), (2, 4),
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16),
    )
    for start, end in skeleton:
        if start in keypoints and end in keypoints:
            draw.line(
                (
                    keypoints[start][0],
                    keypoints[start][1],
                    keypoints[end][0],
                    keypoints[end][1],
                ),
                fill=255,
                width=line_width,
            )
    for x, y in coordinates:
        draw.ellipse(
            (
                x - joint_radius,
                y - joint_radius,
                x + joint_radius,
                y + joint_radius,
            ),
            fill=255,
        )
    if all(index in keypoints for index in (5, 6, 11, 12)):
        torso = [
            (keypoints[5][0], keypoints[5][1]),
            (keypoints[6][0], keypoints[6][1]),
            (keypoints[12][0], keypoints[12][1]),
            (keypoints[11][0], keypoints[11][1]),
        ]
        draw.polygon(torso, fill=255)
    if all(index in keypoints for index in (11, 12, 15, 16)):
        lower_padding = line_width * 1.15
        lower_body = [
            (
                min(keypoints[11][0], keypoints[12][0]) - lower_padding,
                min(keypoints[11][1], keypoints[12][1]),
            ),
            (
                max(keypoints[11][0], keypoints[12][0]) + lower_padding,
                min(keypoints[11][1], keypoints[12][1]),
            ),
            (
                max(keypoints[15][0], keypoints[16][0]) + lower_padding * 0.45,
                max(keypoints[15][1], keypoints[16][1]),
            ),
            (
                min(keypoints[15][0], keypoints[16][0]) - lower_padding * 0.45,
                max(keypoints[15][1], keypoints[16][1]),
            ),
        ]
        draw.polygon(lower_body, fill=255)
    return envelope


def _keypoint_coverage(
    mask: Image.Image,
    keypoints: Sequence[tuple[float, float, float]],
) -> float:
    if not keypoints:
        return 0.0
    array = np.asarray(mask, dtype=np.uint8)
    height, width = array.shape
    covered = 0
    for x, y, _ in keypoints:
        column = max(0, min(width - 1, int(round(x))))
        row = max(0, min(height - 1, int(round(y))))
        covered += array[row, column] >= 128
    return float(covered / len(keypoints))


def _repair_mask_gaps_and_holes(mask: Image.Image) -> tuple[Image.Image, float]:
    """Bridge narrow parser gaps, then fill background islands."""

    original = np.asarray(mask.convert("L"), dtype=np.uint8) >= 128
    repaired = original.copy()
    max_horizontal_gap = max(2, int(round(mask.width * 0.12)))
    max_vertical_gap = max(2, int(round(mask.height * 0.035)))
    for row in range(repaired.shape[0]):
        _fill_short_false_runs(repaired[row], max_horizontal_gap)
    for column in range(repaired.shape[1]):
        _fill_short_false_runs(repaired[:, column], max_vertical_gap)

    binary = Image.fromarray(repaired.astype(np.uint8) * 255, mode="L")
    padded = Image.new("L", (binary.width + 2, binary.height + 2), 0)
    padded.paste(binary, (1, 1))
    ImageDraw.floodfill(padded, (0, 0), 128)
    flooded = np.asarray(padded, dtype=np.uint8)[1:-1, 1:-1]
    holes = flooded == 0
    outside = flooded == 128
    filled = np.where(outside, 0, 255).astype(np.uint8)
    repaired_fraction = float(((filled >= 128) & ~original).mean())
    binary.close()
    padded.close()
    return Image.fromarray(filled, mode="L"), repaired_fraction


def _fill_short_false_runs(line: np.ndarray, max_gap: int) -> None:
    foreground = np.flatnonzero(line)
    if len(foreground) < 2:
        return
    for start, end in zip(foreground[:-1], foreground[1:], strict=True):
        gap = int(end - start - 1)
        if 0 < gap <= max_gap:
            line[start + 1 : end] = True
