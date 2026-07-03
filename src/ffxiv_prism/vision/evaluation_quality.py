from __future__ import annotations

from dataclasses import asdict, dataclass
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ffxiv_prism.vision.retrieval import VALID_SLOTS


QUALITY_POLICY_VERSION = 1
REQUIRED_KEYPOINTS = (
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


@dataclass(frozen=True)
class FrontFullBodyThresholds:
    min_keypoint_score: float = 0.45
    min_required_mean_score: float = 0.62
    min_person_height_px: float = 340.0
    min_person_height_fraction: float = 0.62
    min_head_to_ankle_fraction: float = 0.54
    min_shoulder_width_px: float = 42.0
    min_shoulder_to_torso_ratio: float = 0.45
    max_shoulder_slope: float = 0.28
    max_hip_slope: float = 0.34
    max_torso_axis_offset: float = 0.34
    max_ankle_y_fraction: float = 0.985
    min_nose_y_fraction: float = 0.015


@dataclass(frozen=True)
class FrontFullBodyAssessment:
    passed: bool
    quality_score: float
    rejection_reasons: tuple[str, ...]
    metrics: Mapping[str, float]
    policy_version: int = QUALITY_POLICY_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def assess_front_full_body(
    regions: Mapping[str, Any],
    image_size: tuple[int, int],
    thresholds: FrontFullBodyThresholds = FrontFullBodyThresholds(),
) -> FrontFullBodyAssessment:
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    keypoints = {
        str(row["name"]): (
            float(row["x"]),
            float(row["y"]),
            float(row.get("score", 0.0)),
        )
        for row in regions.get("keypoints", [])
        if row.get("name") is not None
    }
    reasons = []
    missing = [name for name in REQUIRED_KEYPOINTS if name not in keypoints]
    if missing:
        reasons.append("missing_required_keypoints")
        return FrontFullBodyAssessment(
            passed=False,
            quality_score=0.0,
            rejection_reasons=tuple(reasons),
            metrics={"missing_keypoint_count": float(len(missing))},
        )

    required_scores = np.asarray(
        [keypoints[name][2] for name in REQUIRED_KEYPOINTS],
        dtype=np.float32,
    )
    low_count = int((required_scores < thresholds.min_keypoint_score).sum())
    mean_score = float(required_scores.mean())
    if low_count:
        reasons.append("low_required_keypoint_confidence")
    if mean_score < thresholds.min_required_mean_score:
        reasons.append("low_mean_pose_confidence")

    person_box = regions.get("person_box", ())
    if len(person_box) != 4:
        reasons.append("missing_person_box")
        person_height = 0.0
    else:
        person_height = max(0.0, float(person_box[3]) - float(person_box[1]))
    person_height_fraction = person_height / height
    if person_height < thresholds.min_person_height_px:
        reasons.append("person_too_few_pixels")
    if person_height_fraction < thresholds.min_person_height_fraction:
        reasons.append("person_too_small_in_frame")

    nose = keypoints["nose"]
    left_shoulder = keypoints["left_shoulder"]
    right_shoulder = keypoints["right_shoulder"]
    left_hip = keypoints["left_hip"]
    right_hip = keypoints["right_hip"]
    left_ankle = keypoints["left_ankle"]
    right_ankle = keypoints["right_ankle"]

    shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
    hip_width = abs(left_hip[0] - right_hip[0])
    shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2.0
    hip_y = (left_hip[1] + right_hip[1]) / 2.0
    torso_height = max(1.0, abs(hip_y - shoulder_y))
    shoulder_to_torso_ratio = shoulder_width / torso_height
    shoulder_slope = abs(left_shoulder[1] - right_shoulder[1]) / max(
        shoulder_width,
        1.0,
    )
    hip_slope = abs(left_hip[1] - right_hip[1]) / max(hip_width, 1.0)
    shoulder_mid_x = (left_shoulder[0] + right_shoulder[0]) / 2.0
    hip_mid_x = (left_hip[0] + right_hip[0]) / 2.0
    torso_axis_offset = abs(shoulder_mid_x - hip_mid_x) / max(
        shoulder_width,
        1.0,
    )
    if shoulder_width < thresholds.min_shoulder_width_px:
        reasons.append("shoulders_too_few_pixels")
    if shoulder_to_torso_ratio < thresholds.min_shoulder_to_torso_ratio:
        reasons.append("likely_side_view")
    if shoulder_slope > thresholds.max_shoulder_slope:
        reasons.append("excessive_shoulder_tilt")
    if hip_slope > thresholds.max_hip_slope:
        reasons.append("excessive_hip_tilt")
    if torso_axis_offset > thresholds.max_torso_axis_offset:
        reasons.append("excessive_torso_turn")

    max_ankle_y = max(left_ankle[1], right_ankle[1])
    head_to_ankle_fraction = (max_ankle_y - nose[1]) / height
    ankle_y_fraction = max_ankle_y / height
    nose_y_fraction = nose[1] / height
    if head_to_ankle_fraction < thresholds.min_head_to_ankle_fraction:
        reasons.append("not_full_body")
    if ankle_y_fraction > thresholds.max_ankle_y_fraction:
        reasons.append("feet_likely_cropped")
    if nose_y_fraction < thresholds.min_nose_y_fraction:
        reasons.append("head_likely_cropped")

    metrics = {
        "required_keypoint_mean_score": mean_score,
        "low_required_keypoint_count": float(low_count),
        "person_height_px": person_height,
        "person_height_fraction": person_height_fraction,
        "head_to_ankle_fraction": head_to_ankle_fraction,
        "shoulder_width_px": shoulder_width,
        "shoulder_to_torso_ratio": shoulder_to_torso_ratio,
        "shoulder_slope": shoulder_slope,
        "hip_slope": hip_slope,
        "torso_axis_offset": torso_axis_offset,
        "ankle_y_fraction": ankle_y_fraction,
        "nose_y_fraction": nose_y_fraction,
    }
    quality_score = _quality_score(metrics, thresholds)
    return FrontFullBodyAssessment(
        passed=not reasons,
        quality_score=quality_score,
        rejection_reasons=tuple(reasons),
        metrics=metrics,
    )


def quality_allowlist(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    return {
        str(row["example_id"])
        for row in rows
        if bool(row.get("passed"))
    }


def filter_examples_by_quality(
    examples: Sequence[Mapping[str, Any]],
    quality_rows: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    allowed = quality_allowlist(quality_rows)
    return [
        example
        for example in examples
        if str(example.get("example_id")) in allowed
    ]


def summarize_quality_split_coverage(
    examples: Sequence[Mapping[str, Any]],
    quality_rows: Sequence[Mapping[str, Any]],
    split_assignments: Mapping[str, str],
    *,
    target_groups: int = 50,
    target_cases: int = 250,
    slots: Iterable[str] = VALID_SLOTS,
    require_piece_ids: bool = True,
) -> dict[str, Any]:
    """Summarize reviewed high-quality coverage by split and slot.

    The summary is prediction-independent: it only uses source labels, split
    assignments, and the frozen quality manifest. It is intended to decide
    whether a split is large enough for model selection or sealed evaluation,
    not to score a model.
    """

    selected_slots = tuple(str(slot) for slot in slots)
    allowed = quality_allowlist(quality_rows)
    quality_by_example_id = {
        str(row["example_id"]): row
        for row in quality_rows
    }
    accepted = []
    rejected_reasons = Counter()
    for example in examples:
        example_id = str(example.get("example_id"))
        quality = quality_by_example_id.get(example_id)
        if quality is None:
            continue
        if example_id in allowed:
            accepted.append(example)
        else:
            rejected_reasons.update(
                str(reason)
                for reason in quality.get("rejection_reasons", [])
            )
    split_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for example in accepted:
        glamour_id = str(example.get("glamour_id"))
        role = split_assignments.get(glamour_id, "unassigned")
        split_rows[role].append(example)

    split_summary = {
        role: _summarize_quality_examples(
            rows,
            selected_slots,
            require_piece_ids=require_piece_ids,
        )
        for role, rows in sorted(split_rows.items())
    }
    for role in (
        "train",
        "validation",
        "development_test",
        "blind_test",
        "confirmation_test",
        "unassigned",
    ):
        split_summary.setdefault(
            role,
            _summarize_quality_examples(
                [],
                selected_slots,
                require_piece_ids=require_piece_ids,
            ),
        )
    sealed = split_summary["development_test"]
    return {
        "accepted_image_count": len(accepted),
        "case_policy": {
            "require_piece_ids": require_piece_ids,
        },
        "quality_manifest_rows": len(quality_rows),
        "rejected_reason_counts": dict(sorted(rejected_reasons.items())),
        "slots": list(selected_slots),
        "splits": split_summary,
        "sealed_target": {
            "split": "development_test",
            "target_groups": target_groups,
            "target_cases": target_cases,
            "current_groups": sealed["glamour_group_count"],
            "current_cases": sealed["slot_case_count"],
            "group_gap": max(
                0,
                target_groups - int(sealed["glamour_group_count"]),
            ),
            "case_gap": max(
                0,
                target_cases - int(sealed["slot_case_count"]),
            ),
            "ready": (
                int(sealed["glamour_group_count"]) >= target_groups
                and int(sealed["slot_case_count"]) >= target_cases
            ),
        },
    }


def _summarize_quality_examples(
    examples: Sequence[Mapping[str, Any]],
    slots: Sequence[str],
    *,
    require_piece_ids: bool,
) -> dict[str, Any]:
    by_slot = {slot: 0 for slot in slots}
    image_ids = set()
    glamour_ids = set()
    for example in examples:
        image_ids.add(str(example.get("example_id")))
        glamour_ids.add(str(example.get("glamour_id")))
        labels = example.get("core_slot_labels", {})
        if not isinstance(labels, Mapping):
            labels = {}
        for slot in slots:
            label = labels.get(slot)
            has_item = isinstance(label, Mapping) and label.get("item_name")
            has_piece_ids = has_item and bool(label.get("piece_ids"))
            if has_item and (has_piece_ids or not require_piece_ids):
                by_slot[slot] += 1
    return {
        "image_count": len(image_ids),
        "glamour_group_count": len(glamour_ids),
        "slot_case_count": sum(by_slot.values()),
        "by_slot": by_slot,
    }


def _quality_score(
    metrics: Mapping[str, float],
    thresholds: FrontFullBodyThresholds,
) -> float:
    positive = (
        min(1.0, metrics["required_keypoint_mean_score"]),
        min(1.0, metrics["person_height_px"] / thresholds.min_person_height_px),
        min(
            1.0,
            metrics["person_height_fraction"]
            / thresholds.min_person_height_fraction,
        ),
        min(
            1.0,
            metrics["head_to_ankle_fraction"]
            / thresholds.min_head_to_ankle_fraction,
        ),
        min(
            1.0,
            metrics["shoulder_width_px"] / thresholds.min_shoulder_width_px,
        ),
        min(
            1.0,
            metrics["shoulder_to_torso_ratio"]
            / thresholds.min_shoulder_to_torso_ratio,
        ),
    )
    penalties = (
        max(0.0, metrics["shoulder_slope"] - thresholds.max_shoulder_slope),
        max(0.0, metrics["hip_slope"] - thresholds.max_hip_slope),
        max(
            0.0,
            metrics["torso_axis_offset"] - thresholds.max_torso_axis_offset,
        ),
    )
    return float(np.clip(np.mean(positive) - sum(penalties) / 3.0, 0.0, 1.0))
