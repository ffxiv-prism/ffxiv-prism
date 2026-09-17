from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# Initialize Conda's BLAS/OpenMP runtime before CUDA PyTorch loads its bundled
# copy. If Torch is already loaded, do not introduce the second runtime merely
# by importing this geometry module.
if "torch" not in sys.modules:
    np.linalg.inv(np.eye(2, dtype=np.float32))

import torch
from PIL import Image, ImageDraw, ImageFont

from ffxiv_prism.model_assets import load_pretrained
from ffxiv_prism.torch_device import resolve_torch_device

DEFAULT_POSE_MODEL = "usyd-community/vitpose-base-simple"
KEYPOINT_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
SKELETON = (
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13),
    (13, 15), (12, 14), (14, 16),
)
SLOT_COLORS = {
    "HEAD": "#ff5a5f", "BODY": "#27ae60", "HANDS": "#3498db",
    "LEGS": "#f39c12", "FEET": "#9b59b6",
}


@dataclass(frozen=True)
class Keypoint:
    x: float
    y: float
    score: float


@dataclass(frozen=True)
class CropRegion:
    slot: str
    name: str
    box: tuple[int, int, int, int]
    confidence: float
    visibility: str
    role: str = "primary"
    tags: tuple[str, ...] = ()


def _install_vitpose_pil_fallbacks() -> None:
    """Patch two SciPy-only helpers with NumPy/Pillow equivalents."""
    import transformers.utils as transformers_utils

    original_scipy_check = transformers_utils.is_scipy_available
    transformers_utils.is_scipy_available = lambda: False
    try:
        import transformers.models.vitpose.image_processing_pil_vitpose as vitpose_module
    finally:
        transformers_utils.is_scipy_available = original_scipy_check

    if getattr(vitpose_module, "_ffxiv_prism_patched", False):
        return

    def pil_warp_affine(src: np.ndarray, M: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        a, b, c = (float(value) for value in M[0])
        d, e, f = (float(value) for value in M[1])
        determinant = a * e - b * d
        if abs(determinant) < 1e-12:
            raise ValueError("Affine transform is singular")
        coefficients = tuple(
            float(value)
            for value in (
                e / determinant,
                -b / determinant,
                (b * f - e * c) / determinant,
                -d / determinant,
                a / determinant,
                (d * c - a * f) / determinant,
            )
        )
        array = np.asarray(src)
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        transformed = Image.fromarray(array).transform(
            (int(size[1]), int(size[0])),
            Image.Transform.AFFINE,
            coefficients,
            resample=Image.Resampling.BILINEAR,
        )
        return np.asarray(transformed)

    def numpy_gaussian_filter(
        array: np.ndarray,
        sigma: float = 0.8,
        radius: int | tuple[int, int] | None = None,
        axes: tuple[int, int] | None = None,
    ) -> np.ndarray:
        del axes
        if isinstance(radius, tuple):
            kernel_radius = int(radius[0])
        elif radius is None:
            kernel_radius = max(1, round(4 * sigma))
        else:
            kernel_radius = int(radius)
        coordinates = np.arange(-kernel_radius, kernel_radius + 1, dtype=np.float32)
        kernel = np.exp(-(coordinates**2) / (2 * sigma**2))
        kernel /= kernel.sum()
        output = np.asarray(array, dtype=np.float32)
        for axis in (0, 1):
            padding = [(0, 0)] * output.ndim
            padding[axis] = (kernel_radius, kernel_radius)
            padded = np.pad(output, padding, mode="edge")
            output = np.apply_along_axis(
                lambda values: np.convolve(values, kernel, mode="valid"), axis, padded
            )
        return output

    vitpose_module.scipy_warp_affine = pil_warp_affine
    vitpose_module.gaussian_filter = numpy_gaussian_filter
    vitpose_module._ffxiv_prism_patched = True


class SinglePersonPoseEstimator:
    def __init__(self, model_id: str = DEFAULT_POSE_MODEL, device: str = "auto") -> None:
        _install_vitpose_pil_fallbacks()
        from transformers import VitPoseForPoseEstimation, VitPoseImageProcessorPil

        self.model_id = model_id
        self.device = resolve_torch_device(device)
        self.processor = load_pretrained(VitPoseImageProcessorPil, model_id)
        self.model = load_pretrained(VitPoseForPoseEstimation, model_id)
        self.model = self.model.to(self.device).eval()

    def estimate(self, image: Image.Image, refine: bool = True) -> tuple[list[Keypoint], tuple[int, int, int, int]]:
        width, height = image.size
        full_box = (0, 0, width, height)
        first = self._estimate_in_box(image, full_box)
        selected, selected_box = first, full_box
        if pose_quality(first, full_box) < 0.55:
            proposals = single_person_search_boxes(image.size)
            poses = self._estimate_in_boxes(image, proposals)
            selected_box, selected = max(
                zip(proposals, poses),
                key=lambda pair: pose_selection_score(pair[1], pair[0], image.size),
            )
        if not refine:
            return selected, selected_box
        refined_box = estimate_person_box(selected, image.size)
        if box_iou(selected_box, refined_box) > 0.9:
            return selected, selected_box
        second = self._estimate_in_box(image, refined_box)
        if pose_quality(second, refined_box) + 0.05 < pose_quality(selected, selected_box):
            return selected, selected_box
        return second, refined_box

    def _estimate_in_box(self, image: Image.Image, box: tuple[int, int, int, int]) -> list[Keypoint]:
        return self._estimate_in_boxes(image, [box])[0]

    def _estimate_in_boxes(
        self, image: Image.Image, boxes: list[tuple[int, int, int, int]]
    ) -> list[list[Keypoint]]:
        coco_boxes = [[
            [float(left), float(top), float(right - left), float(bottom - top)]
            for left, top, right, bottom in boxes
        ]]
        inputs = self.processor(images=image, boxes=coco_boxes, return_tensors="pt")
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        with torch.inference_mode():
            outputs = self.model(**inputs)
        results = self.processor.post_process_pose_estimation(outputs, coco_boxes)[0]
        poses = []
        for result in results:
            coordinates = result["keypoints"].cpu().tolist()
            scores = result["scores"].cpu().tolist()
            poses.append([
                Keypoint(float(x), float(y), float(score))
                for (x, y), score in zip(coordinates, scores)
            ])
        return poses



def pose_quality(keypoints: list[Keypoint], box: tuple[int, int, int, int]) -> float:
    scores = np.asarray([point.score for point in keypoints], dtype=np.float32)
    core_indices = (0, 5, 6, 11, 12, 13, 14, 15, 16)
    core_mean = float(np.mean([scores[index] for index in core_indices]))
    reliable_fraction = float((scores >= 0.35).mean())
    left, top, right, bottom = box
    inside_fraction = float(np.mean([
        left <= point.x <= right and top <= point.y <= bottom for point in keypoints
    ]))
    visible = [point for point in keypoints if point.score >= 0.25]
    if visible:
        span_x, span_y = point_span(visible)
        extent_score = min(1.0, span_x / max(1.0, (right - left) * 0.12))
        extent_score *= min(1.0, span_y / max(1.0, (bottom - top) * 0.30))
    else:
        extent_score = 0.0
    return 0.45 * float(scores.mean()) + 0.30 * core_mean + 0.15 * reliable_fraction + 0.05 * inside_fraction + 0.05 * extent_score



def pose_selection_score(
    keypoints: list[Keypoint],
    box: tuple[int, int, int, int],
    image_size: tuple[int, int],
) -> float:
    image_width, _ = image_size
    box_center = (box[0] + box[2]) / 2
    center_prior = max(0.0, 1.0 - abs(box_center - image_width / 2) / max(1.0, image_width / 2))
    return pose_quality(keypoints, box) + 0.18 * center_prior

def single_person_search_boxes(image_size: tuple[int, int]) -> list[tuple[int, int, int, int]]:
    width, height = image_size
    boxes: list[tuple[int, int, int, int]] = [(0, 0, width, height)]
    aspect = width / max(1, height)
    width_fractions = (0.38, 0.52, 0.68) if aspect >= 1.15 else (0.62, 0.82)
    centers = (0.18, 0.34, 0.50, 0.66, 0.82) if aspect >= 1.15 else (0.35, 0.50, 0.65)
    for fraction in width_fractions:
        crop_width = width * fraction
        for center_fraction in centers:
            center = width * center_fraction
            left = center - crop_width / 2
            right = center + crop_width / 2
            boxes.append(clamp_box((left, 0, right, height), image_size))
    unique = []
    seen = set()
    for box in boxes:
        if box not in seen:
            seen.add(box)
            unique.append(box)
    return unique

def estimate_person_box(
    keypoints: list[Keypoint], image_size: tuple[int, int], threshold: float = 0.25
) -> tuple[int, int, int, int]:
    width, height = image_size
    visible = [point for point in keypoints if point.score >= threshold]
    if len(visible) < 6:
        return (0, 0, width, height)
    xs = [point.x for point in visible]
    ys = [point.y for point in visible]
    span_x = max(max(xs) - min(xs), width * 0.18)
    span_y = max(max(ys) - min(ys), height * 0.35)
    left = min(xs) - span_x * 0.28
    top = min(ys) - span_y * 0.18
    right = max(xs) + span_x * 0.28
    bottom = max(ys) + span_y * 0.10

    # COCO pose landmarks stop at the ankles and contain no shoe-tip or sole
    # points.  The refined pose crop must nevertheless retain the full shoe;
    # otherwise a second pose pass can be run on an image that has already
    # clipped thick soles or forward-projecting toes.  Reuse the same
    # lower-leg scale as the downstream full-shoe diagnostic regions.
    shoulder_width = max(distance(keypoints[5], keypoints[6]), width * 0.12)
    torso_length = max(
        distance(midpoint(keypoints[5], keypoints[6]), midpoint(keypoints[11], keypoints[12])),
        height * 0.18,
    )
    for knee_index, ankle_index in ((13, 15), (14, 16)):
        knee = keypoints[knee_index]
        ankle = keypoints[ankle_index]
        if min(knee.score, ankle.score) < threshold:
            continue
        shin = max(distance(knee, ankle), torso_length * 0.45)
        left = min(left, ankle.x - shin * 0.58)
        right = max(right, ankle.x + shin * 0.58)
        bottom = max(bottom, ankle.y + shin * 0.55)

    # Facial landmarks likewise describe the face, not broad brims or tall
    # crowns.  Keep the full-headwear QA envelope inside the refined crop when
    # the face/shoulder landmarks are reliable.  This does not make the large
    # crop a retrieval input; it only prevents pose refinement and debug
    # visualization from contradicting the saved full-headwear crop.
    face = keypoints[:5]
    if sum(point.score >= threshold for point in face) >= 4:
        face_span = max(point_span(face)[0], shoulder_width * 0.42)
        face_center = average_point(face)
        left = min(left, face_center.x - face_span * 2.70)
        top = min(top, face_center.y - face_span * 2.15)
        right = max(right, face_center.x + face_span * 2.70)

    return clamp_box(
        (left, top, right, bottom),
        image_size,
    )


def build_outfit_regions(
    keypoints: list[Keypoint], image_size: tuple[int, int], threshold: float = 0.35
) -> list[CropRegion]:
    width, height = image_size
    shoulder_width = max(distance(keypoints[5], keypoints[6]), width * 0.12)
    torso_length = max(distance(midpoint(keypoints[5], keypoints[6]), midpoint(keypoints[11], keypoints[12])), height * 0.18)
    regions: list[CropRegion] = []

    def add(
        slot: str,
        name: str,
        indices: Iterable[int],
        pad_x: float,
        pad_y: float,
        role: str = "primary",
        tags: tuple[str, ...] = (),
    ) -> None:
        points = [keypoints[index] for index in indices]
        confidence, visibility = assess_visibility(points, image_size, threshold)
        raw = bounds(points, pad_x, pad_y)
        regions.append(CropRegion(slot, name, clamp_box(raw, image_size), confidence, visibility, role, tags))

    face = keypoints[:5]
    face_span = max(point_span(face)[0], shoulder_width * 0.42)
    face_center = average_point(face)
    head_tight = (
        face_center.x - face_span * 0.82,
        face_center.y - face_span * 1.05,
        face_center.x + face_span * 0.82,
        face_center.y + face_span * 1.25,
    )
    head_conf, head_vis = assess_visibility(face + keypoints[5:7], image_size, threshold)
    regions.append(CropRegion(
        "HEAD",
        "head_tight",
        clamp_box(head_tight, image_size),
        head_conf,
        head_vis,
        "primary",
        ("may_include_face_hair",),
    ))
    hat_box = (
        face_center.x - face_span * 0.95,
        face_center.y - face_span * 1.35,
        face_center.x + face_span * 0.95,
        face_center.y + face_span * 0.45,
    )
    # Facial keypoints cannot reveal the outline of a broad brim or a tall
    # crown.  Preserve the promoted model's compact upper-head crop, while a
    # zero-weight companion captures unusually large headwear for visual QA.
    full_headwear_box = (
        # A 2.4-face-span margin still clipped the outer left brim of the
        # saved wide-straw-hat regression image (user sample 13).  This crop
        # is diagnostic-only, so prefer complete silhouette coverage over a
        # tighter background ratio without perturbing promoted retrieval.
        face_center.x - face_span * 2.70,
        face_center.y - face_span * 2.15,
        face_center.x + face_span * 2.70,
        face_center.y + face_span * 1.25,
    )
    headwear_edge_risk = box_exceeds_source(
        full_headwear_box,
        image_size,
        tolerance=face_span * 0.45,
    )
    hat_tags = (
        "hat_or_hair_accessory_focus",
        "may_include_hair",
        "full_headwear_companion_available",
    )
    if headwear_edge_risk:
        hat_tags += ("source_edge_truncation_risk",)
    regions.append(CropRegion(
        "HEAD",
        "head_upper_accessory",
        clamp_box(hat_box, image_size),
        head_conf,
        head_vis,
        "auxiliary",
        hat_tags,
    ))
    full_headwear_visibility = (
        downgrade_visibility(head_vis, "partially_visible")
        if headwear_edge_risk
        else head_vis
    )
    full_headwear_tags = (
        "wide_or_tall_headwear_context",
        "expanded_for_full_headwear",
    )
    if headwear_edge_risk:
        full_headwear_tags += ("source_edge_truncation_risk",)
    regions.append(CropRegion(
        "HEAD",
        "head_full_accessory",
        clamp_box(full_headwear_box, image_size),
        head_conf,
        full_headwear_visibility,
        "diagnostic",
        full_headwear_tags,
    ))
    eye_box = (
        face_center.x - face_span * 0.95,
        face_center.y - face_span * 0.40,
        face_center.x + face_span * 0.95,
        face_center.y + face_span * 0.55,
    )
    regions.append(CropRegion(
        "HEAD",
        "head_face_accessory",
        clamp_box(eye_box, image_size),
        head_conf,
        head_vis,
        "auxiliary",
        ("glasses_mask_or_facewear_focus", "may_include_face"),
    ))
    add(
        "HEAD",
        "head_context",
        (0, 1, 2, 3, 4, 5, 6),
        shoulder_width * 0.16,
        torso_length * 0.08,
        "context",
        ("may_include_face_hair_shoulders",),
    )

    add("BODY", "body_torso", (5, 6, 11, 12), shoulder_width * 0.28, torso_length * 0.16)
    upper_core_points = [keypoints[index] for index in (5, 6, 11, 12)]
    upper_core_conf, upper_core_vis = assess_visibility(
        upper_core_points,
        image_size,
        threshold,
    )
    upper_core_box = (
        min(point.x for point in upper_core_points) - shoulder_width * 0.28,
        min(keypoints[5].y, keypoints[6].y) - torso_length * 0.12,
        max(point.x for point in upper_core_points) + shoulder_width * 0.28,
        midpoint(keypoints[11], keypoints[12]).y - torso_length * 0.08,
    )
    regions.append(CropRegion(
        "BODY",
        "body_upper_core",
        clamp_box(upper_core_box, image_size),
        upper_core_conf,
        upper_core_vis,
        "diagnostic",
        (
            "upper_garment_style_focus",
            "shoulder_and_waist_coverage",
            "reduced_legs_overlap",
        ),
    ))
    add(
        "BODY",
        "body_upper_arm_overlap",
        (5, 6, 7, 8, 11, 12),
        shoulder_width * 0.26,
        torso_length * 0.14,
        "auxiliary",
        ("body_hands_overlap", "long_sleeve_or_arm_gear_ambiguous"),
    )
    add(
        "BODY",
        "body_context",
        (5, 6, 7, 8, 11, 12, 13, 14),
        shoulder_width * 0.22,
        torso_length * 0.12,
        "context",
        ("may_include_sleeves_hands_waist",),
    )

    limb_pad = max(shoulder_width * 0.24, 12.0)
    add("HANDS", "hands_left", (5, 7, 9), limb_pad, limb_pad)
    add("HANDS", "hands_right", (6, 8, 10), limb_pad, limb_pad)
    add(
        "HANDS",
        "hands_forearm_left",
        (7, 9),
        limb_pad * 0.70,
        limb_pad * 0.70,
        "auxiliary",
        ("long_glove_or_sleeve_overlap",),
    )
    add(
        "HANDS",
        "hands_forearm_right",
        (8, 10),
        limb_pad * 0.70,
        limb_pad * 0.70,
        "auxiliary",
        ("long_glove_or_sleeve_overlap",),
    )
    add(
        "HANDS",
        "hands_arm_cover_context",
        (5, 6, 7, 8, 9, 10),
        limb_pad * 0.75,
        limb_pad * 0.65,
        "context",
        ("body_hands_overlap", "may_include_sleeves"),
    )

    # COCO pose ends at the wrist. Extend along the forearm to include the
    # fingers, without shrinking the trained arm crops or feeding a new crop
    # distribution to their query adapter. These companions are used only by
    # the confidence-gated hand-shape check.
    for side, elbow_index, wrist_index in (("left", 7, 9), ("right", 8, 10)):
        elbow, wrist = keypoints[elbow_index], keypoints[wrist_index]
        dx, dy = wrist.x - elbow.x, wrist.y - elbow.y
        forearm_length = distance(elbow, wrist)
        radius = max(forearm_length * 0.60, shoulder_width * 0.20, 12.0)
        center_x, center_y = wrist.x + dx * 0.40, wrist.y + dy * 0.40
        raw = (center_x - radius, center_y - radius,
               center_x + radius, center_y + radius)
        confidence, visibility = assess_visibility([elbow, wrist], image_size, threshold)
        tags = ("full_hand_shape", "extended_beyond_wrist")
        if min(elbow.score, wrist.score) < threshold or forearm_length < 1.0:
            visibility = "pose_uncertain"
        elif box_exceeds_source(raw, image_size):
            visibility = downgrade_visibility(visibility, "partially_visible")
            tags += ("source_edge_truncation_risk",)
        regions.append(CropRegion(
            "HANDS", f"hands_palm_{side}", clamp_box(raw, image_size),
            confidence, visibility, "diagnostic", tags,
        ))

    leg_pad = max(shoulder_width * 0.20, 12.0)
    add("LEGS", "legs_left", (11, 13, 15), leg_pad, torso_length * 0.08)
    add("LEGS", "legs_right", (12, 14, 16), leg_pad, torso_length * 0.08)
    add(
        "LEGS",
        "legs_thighs",
        (11, 12, 13, 14),
        leg_pad,
        torso_length * 0.10,
        "auxiliary",
        ("reduced_boot_overlap",),
    )
    thigh_midpoints = (
        midpoint(keypoints[11], keypoints[13]),
        midpoint(keypoints[12], keypoints[14]),
    )
    waist_thigh_points = [
        keypoints[11],
        keypoints[12],
        *thigh_midpoints,
    ]
    waist_thigh_conf, waist_thigh_vis = assess_visibility(
        [keypoints[index] for index in (11, 12, 13, 14)],
        image_size,
        threshold,
    )
    style_leg_pad = max(leg_pad, shoulder_width * 0.32)
    waist_thigh_box = (
        min(point.x for point in waist_thigh_points) - style_leg_pad,
        min(keypoints[11].y, keypoints[12].y) - torso_length * 0.14,
        max(point.x for point in waist_thigh_points) + style_leg_pad,
        max(point.y for point in thigh_midpoints) + torso_length * 0.10,
    )
    regions.append(CropRegion(
        "LEGS",
        "legs_waist_thighs",
        clamp_box(waist_thigh_box, image_size),
        waist_thigh_conf,
        waist_thigh_vis,
        "diagnostic",
        (
            "lower_garment_style_focus",
            "waistband_and_hem",
            "reduced_lower_leg_overlap",
        ),
    ))
    add(
        "LEGS",
        "legs_boot_overlap",
        (13, 14, 15, 16),
        leg_pad,
        torso_length * 0.08,
        "auxiliary",
        ("legs_feet_overlap", "long_boot_ambiguous"),
    )
    add(
        "LEGS",
        "legs_context",
        (11, 12, 13, 14, 15, 16),
        leg_pad,
        torso_length * 0.10,
        "context",
        ("may_include_boots",),
    )

    full_feet_boxes: list[tuple[float, float, float, float]] = []
    footwear_height_tops: list[float] = []
    feet_context_points: list[Keypoint] = []
    feet_context_edge_risk = False
    for side, knee_index, ankle_index in (("left", 13, 15), ("right", 14, 16)):
        knee = keypoints[knee_index]
        ankle = keypoints[ankle_index]
        shin = max(distance(knee, ankle), torso_length * 0.45)
        # Keep the compact ankle crop that the existing FEET query adapters were
        # trained on.  The COCO ankle keypoint is not the centre of the shoe, so
        # the companion boot crop below deliberately carries the full-shoe view.
        foot_box = (
            ankle.x - shin * 0.34,
            ankle.y - shin * 0.28,
            ankle.x + shin * 0.34,
            ankle.y + shin * 0.28,
        )
        confidence, visibility = assess_visibility([knee, ankle], image_size, threshold)
        full_box = (
            ankle.x - shin * 0.55,
            ankle.y - shin * 0.92,
            ankle.x + shin * 0.55,
            ankle.y + shin * 0.52,
        )
        edge_risk = box_exceeds_source(
            full_box,
            image_size,
            tolerance=shin * 0.04,
        )
        foot_tags = ("ankle_focus", "full_shoe_companion_available")
        if edge_risk:
            foot_tags += ("source_edge_truncation_risk",)
        regions.append(CropRegion(
            "FEET",
            f"feet_{side}",
            clamp_box(foot_box, image_size),
            confidence,
            visibility,
            "primary",
            foot_tags,
        ))
        # Preserve the legacy long-boot query crop as well.  Existing query
        # adapters depend on this crop distribution; the diagnostic full crop
        # is saved for QA but deliberately receives zero retrieval weight.
        boot_box = (
            ankle.x - shin * 0.40,
            ankle.y - shin * 0.72,
            ankle.x + shin * 0.40,
            ankle.y + shin * 0.30,
        )
        boot_visibility = (
            visibility
        )
        boot_tags = (
            "legs_feet_overlap",
            "long_boot_focus",
            "full_shoe_companion_available",
        )
        if edge_risk:
            boot_tags += ("source_edge_truncation_risk",)
        regions.append(CropRegion(
            "FEET",
            f"feet_boot_{side}",
            clamp_box(boot_box, image_size),
            confidence,
            boot_visibility,
            "auxiliary",
            boot_tags,
        ))
        full_visibility = (
            downgrade_visibility(visibility, "partially_visible")
            if edge_risk
            else visibility
        )
        full_tags = (
            "full_shoe_or_boot_focus",
            "expanded_for_full_shoe",
        )
        if edge_risk:
            full_tags += ("source_edge_truncation_risk",)
        regions.append(CropRegion(
            "FEET",
            f"feet_full_{side}",
            clamp_box(full_box, image_size),
            confidence,
            full_visibility,
            "diagnostic",
            full_tags,
        ))
        full_feet_boxes.append(full_box)
        # A knee/ankle-only crop cannot distinguish a calf boot from a boot
        # whose opening sits above the knee.  Keep a separate morphology crop
        # that reaches into the lower thigh while retaining the complete shoe.
        footwear_height_tops.append(knee.y - shin * 0.72)
        feet_context_points.extend((knee, ankle))
        feet_context_edge_risk = feet_context_edge_risk or edge_risk

    # Keep the original retrieval context pixel-compatible with existing
    # checkpoints, then add a separate full-shoe context for visual QA.
    legacy_context_box = bounds(
        feet_context_points,
        leg_pad,
        torso_length * 0.08,
    )
    context_confidence, context_visibility = assess_visibility(
        feet_context_points,
        image_size,
        threshold,
    )
    context_tags = ("may_include_lower_legs", "full_shoe_companion_available")
    if feet_context_edge_risk:
        context_tags += ("source_edge_truncation_risk",)
    regions.append(CropRegion(
        "FEET",
        "feet_context",
        clamp_box(legacy_context_box, image_size),
        context_confidence,
        context_visibility,
        "context",
        context_tags,
    ))
    full_context_box = (
        min(box[0] for box in full_feet_boxes),
        min(box[1] for box in full_feet_boxes),
        max(box[2] for box in full_feet_boxes),
        max(box[3] for box in full_feet_boxes),
    )
    full_context_visibility = context_visibility
    full_context_tags = ("full_shoe_or_boot_context", "expanded_for_full_shoe")
    if feet_context_edge_risk:
        full_context_visibility = downgrade_visibility(
            full_context_visibility,
            "partially_visible",
        )
        full_context_tags += ("source_edge_truncation_risk",)
    regions.append(CropRegion(
        "FEET",
        "feet_full_context",
        clamp_box(full_context_box, image_size),
        context_confidence,
        full_context_visibility,
        "diagnostic",
        full_context_tags,
    ))
    height_context_box = (
        min(box[0] for box in full_feet_boxes),
        min(footwear_height_tops),
        max(box[2] for box in full_feet_boxes),
        max(box[3] for box in full_feet_boxes),
    )
    height_context_tags = (
        "footwear_height_style_focus",
        "includes_lower_thigh_and_full_shoe",
    )
    if feet_context_edge_risk:
        height_context_tags += ("source_edge_truncation_risk",)
    regions.append(CropRegion(
        "FEET",
        "feet_height_context",
        clamp_box(height_context_box, image_size),
        context_confidence,
        full_context_visibility,
        "diagnostic",
        height_context_tags,
    ))
    return regions

def save_region_crops(image: Image.Image, regions: list[CropRegion], output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for region in regions:
        path = output_dir / f"{stem}-{region.name}.jpg"
        image.crop(region.box).convert("RGB").save(path, quality=92)
        paths.append(path)
    return paths


def render_debug_overlay(
    image: Image.Image,
    keypoints: list[Keypoint],
    person_box: tuple[int, int, int, int],
    regions: list[CropRegion],
) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(person_box, outline="#ffffff", width=3)
    for start, end in SKELETON:
        a, b = keypoints[start], keypoints[end]
        if min(a.score, b.score) >= 0.2:
            draw.line((a.x, a.y, b.x, b.y), fill="#00e5ff", width=3)
    for index, point in enumerate(keypoints):
        color = "#00ff88" if point.score >= 0.35 else "#ffcc00"
        radius = 3
        draw.ellipse((point.x - radius, point.y - radius, point.x + radius, point.y + radius), fill=color)
        if point.score < 0.35:
            draw.text((point.x + 4, point.y), f"{index}:{point.score:.2f}", fill=color)
    primary_names = {
        # Show the actual completeness envelopes.  The former compact
        # head/feet context boxes remain retrieval inputs, but displaying them
        # as the only slot boxes made a fully saved hat or shoe look clipped.
        "head_full_accessory",
        "body_context",
        "hands_arm_cover_context",
        "legs_context",
        "feet_full_context",
        "body_upper_arm_overlap",
        "legs_boot_overlap",
    }
    for region in regions:
        if region.name not in primary_names:
            continue
        color = SLOT_COLORS[region.slot]
        draw.rectangle(region.box, outline=color, width=3)
        completeness = " full" if region.name in {
            "head_full_accessory",
            "feet_full_context",
        } else ""
        draw.text(
            (region.box[0] + 3, region.box[1] + 3),
            f"{region.slot}{completeness} {region.visibility} {region.confidence:.2f}",
            fill=color,
        )
    return canvas


def render_crop_contact_sheet(image: Image.Image, regions: list[CropRegion], thumb_size: tuple[int, int] = (220, 220)) -> Image.Image:
    columns = 3
    rows = (len(regions) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_size[0], rows * (thumb_size[1] + 28)), "#181818")
    draw = ImageDraw.Draw(sheet)
    for index, region in enumerate(regions):
        crop = image.crop(region.box).convert("RGB")
        crop.thumbnail(thumb_size, Image.Resampling.LANCZOS)
        x = (index % columns) * thumb_size[0] + (thumb_size[0] - crop.width) // 2
        y0 = (index // columns) * (thumb_size[1] + 28)
        y = y0 + (thumb_size[1] - crop.height) // 2
        sheet.paste(crop, (x, y))
        draw.text((index % columns * thumb_size[0] + 4, y0 + thumb_size[1] + 4), f"{region.name} {region.visibility} {region.confidence:.2f}", fill=SLOT_COLORS[region.slot])
    return sheet


def region_manifest(keypoints: list[Keypoint], person_box: tuple[int, int, int, int], regions: list[CropRegion]) -> dict[str, Any]:
    return {
        "person_box": list(person_box),
        "keypoints": [
            {"name": name, "x": point.x, "y": point.y, "score": point.score}
            for name, point in zip(KEYPOINT_NAMES, keypoints)
        ],
        "regions": [
            {
                "slot": region.slot,
                "name": region.name,
                "box": list(region.box),
                "confidence": region.confidence,
                "visibility": region.visibility,
                "role": region.role,
                "tags": list(region.tags),
            }
            for region in regions
        ],
    }


def assess_visibility(points: list[Keypoint], image_size: tuple[int, int], threshold: float) -> tuple[float, str]:
    if not points:
        return 0.0, "pose_uncertain"
    scores = np.asarray([point.score for point in points], dtype=np.float32)
    confidence = float(scores.mean())
    reliable = float((scores >= threshold).mean())
    width, height = image_size
    margin_x = max(6.0, width * 0.035)
    margin_y = max(6.0, height * 0.035)
    near_edge = any(
        point.x < margin_x
        or point.y < margin_y
        or point.x > width - margin_x
        or point.y > height - margin_y
        for point in points
    )
    if reliable >= 0.75 and confidence >= 0.55 and not near_edge:
        return confidence, "visible"
    if reliable >= 0.45 and confidence >= 0.35:
        return confidence, "partially_visible" if near_edge else "occluded"
    if confidence < 0.2:
        return confidence, "out_of_frame" if near_edge else "pose_uncertain"
    return confidence, "pose_uncertain"


def bounds(points: list[Keypoint], pad_x: float, pad_y: float) -> tuple[float, float, float, float]:
    xs = [point.x for point in points]
    ys = [point.y for point in points]
    return min(xs) - pad_x, min(ys) - pad_y, max(xs) + pad_x, max(ys) + pad_y


def clamp_box(box: tuple[float, float, float, float], image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = image_size
    left = max(0, min(width - 1, int(round(box[0]))))
    top = max(0, min(height - 1, int(round(box[1]))))
    right = max(left + 1, min(width, int(round(box[2]))))
    bottom = max(top + 1, min(height, int(round(box[3]))))
    return left, top, right, bottom


def box_exceeds_source(
    box: tuple[float, float, float, float],
    image_size: tuple[int, int],
    *,
    tolerance: float = 0.0,
) -> bool:
    """Return whether a desired crop extends meaningfully beyond the source image."""
    width, height = image_size
    return (
        box[0] < -tolerance
        or box[1] < -tolerance
        or box[2] > width + tolerance
        or box[3] > height + tolerance
    )


def downgrade_visibility(current: str, target: str) -> str:
    """Only lower visibility confidence; never upgrade an uncertain pose."""
    order = {
        "visible": 4,
        "partially_visible": 3,
        "occluded": 2,
        "pose_uncertain": 1,
        "out_of_frame": 0,
    }
    return target if order.get(target, 0) < order.get(current, 0) else current


def midpoint(a: Keypoint, b: Keypoint) -> Keypoint:
    return Keypoint((a.x + b.x) / 2, (a.y + b.y) / 2, (a.score + b.score) / 2)


def average_point(points: list[Keypoint]) -> Keypoint:
    weights = np.asarray([max(point.score, 0.05) for point in points], dtype=np.float32)
    return Keypoint(
        float(np.average([point.x for point in points], weights=weights)),
        float(np.average([point.y for point in points], weights=weights)),
        float(np.mean([point.score for point in points])),
    )


def point_span(points: list[Keypoint]) -> tuple[float, float]:
    return max(point.x for point in points) - min(point.x for point in points), max(point.y for point in points) - min(point.y for point in points)


def distance(a: Keypoint, b: Keypoint) -> float:
    return float(np.hypot(a.x - b.x, a.y - b.y))


def box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return intersection / float(area_a + area_b - intersection)
