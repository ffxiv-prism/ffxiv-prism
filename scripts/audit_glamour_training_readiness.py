from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ffxiv_prism.jsonl import read_jsonl, write_json  # noqa: E402
from ffxiv_prism.paths import portable_path, resolve_project_path  # noqa: E402


SLOTS = ("HEAD", "BODY", "HANDS", "LEGS", "FEET")
DEFAULT_EXAMPLES = Path(
    "data/processed/glamour_examples_huiji_models_closed.jsonl"
)
DEFAULT_QUALITY = Path("data/processed/front_fullbody_quality_final_v46.jsonl")
DEFAULT_DB = Path("data/processed/eorzea.sqlite")
DEFAULT_METRICS = Path("data/processed/current_recognition_metrics.json")
DEFAULT_VISIBILITY_REVIEWS = Path(
    "data/outputs/hands-visibility-audit-sample-20260730/manual-reviews.jsonl"
)
DEFAULT_POSE_VISIBILITY = Path(
    "data/outputs/training-readiness-audit-20260730/"
    "pose-slot-visibility.jsonl"
)
DEFAULT_OUTPUT_DIR = Path("data/outputs/training-readiness-audit-20260730")
INDEX_DIRS = {
    "HEAD": Path(
        "data/indexes/facebook-dinov2-base-head-multiscale-v4-20260730"
    ),
    "BODY": Path(
        "data/indexes/"
        "facebook-dinov2-base-slot-aligned-v3-body-multiscale-20260730"
    ),
    "HANDS": Path(
        "data/indexes/facebook-dinov2-base-hands-multiscale-v5-20260730"
    ),
    "LEGS": Path(
        "data/indexes/facebook-dinov2-base-legs-multiscale-v6-20260730"
    ),
    "FEET": Path(
        "data/indexes/"
        "facebook-dinov2-base-feet-multiscale-v7-clean-20260730"
    ),
}
ENSEMBLE_INDEX_DIRS = {
    "HEAD": (
        Path(
            "data/indexes/"
            "google-siglip2-base-patch16-224-head-multiscale-v4-20260730"
        ),
        Path(
            "data/indexes/"
            "facebook-dinov3-vitb16-head-multiscale-v4-20260730"
        ),
    ),
    "BODY": (
        Path(
            "data/indexes/"
            "facebook-dinov3-vitb16-body-multiscale-v3-20260730"
        ),
    ),
    "HANDS": (),
    "LEGS": (
        Path(
            "data/indexes/"
            "facebook-dinov3-vitb16-legs-multiscale-v6-20260730"
        ),
    ),
    "FEET": (
        Path(
            "data/indexes/"
            "google-siglip2-base-patch16-224-feet-multiscale-v7-clean-20260730"
        ),
    ),
}
FOLD_DIRS = {
    "BODY": Path("data/interim/appearance-disjoint-cv-v52-huiji-closed"),
    "HEAD": Path("data/interim/appearance-disjoint-cv-v54-head-huiji-closed"),
    "HANDS": Path(
        "data/interim/appearance-disjoint-cv-v54-hands-huiji-closed"
    ),
    "LEGS": Path("data/interim/appearance-disjoint-cv-v54-legs-huiji-closed"),
    "FEET": Path("data/interim/appearance-disjoint-cv-v54-feet-huiji-closed"),
}
SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}


@dataclass(frozen=True)
class Thresholds:
    min_labelled_images: int = 1000
    min_train_images_per_fold: int = 600
    min_validation_images_per_fold: int = 250
    min_gender_images: int = 100
    min_race_images: int = 30
    min_multi_glamour_identity_rate: float = 0.5
    min_independent_visibility_sample: int = 100
    min_independent_sample_usable_rate: float = 0.95
    min_independent_sample_wilson_low: float = 0.90
    max_ci_half_width: float = 0.03
    near_duplicate_hamming: int = 4


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[int, int] = {}

    def find(self, value: int) -> int:
        self.parent.setdefault(value, value)
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union_all(self, values: Iterable[int]) -> None:
        values = list(dict.fromkeys(values))
        if not values:
            return
        root = self.find(values[0])
        for value in values[1:]:
            other = self.find(value)
            if root != other:
                self.parent[other] = root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit player-glamour training readiness before any model training."
        )
    )
    parser.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES)
    parser.add_argument("--quality", type=Path, default=DEFAULT_QUALITY)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument(
        "--visibility-reviews",
        type=Path,
        action="append",
        help=(
            "JSONL semantic visibility reviews. Repeat to merge independent "
            "review batches; defaults to the HANDS pilot."
        ),
    )
    parser.add_argument(
        "--pose-visibility",
        type=Path,
        default=DEFAULT_POSE_VISIBILITY,
    )
    parser.add_argument(
        "--visibility-curation-report",
        type=Path,
        action="append",
        default=[],
        help=(
            "Validated slot-specific curation report. Repeat for multiple "
            "slots; a ready report can supersede a failed pre-remediation "
            "independent sample."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-labelled-images", type=int, default=1000)
    parser.add_argument("--min-train-images-per-fold", type=int, default=600)
    parser.add_argument("--min-validation-images-per-fold", type=int, default=250)
    parser.add_argument("--min-gender-images", type=int, default=100)
    parser.add_argument("--min-race-images", type=int, default=30)
    parser.add_argument(
        "--min-multi-glamour-identity-rate",
        type=float,
        default=0.5,
        help=(
            "Minimum share of appearance identities represented by at least "
            "two independent glamours."
        ),
    )
    parser.add_argument(
        "--min-independent-visibility-sample",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--min-independent-sample-usable-rate",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--min-independent-sample-wilson-low",
        type=float,
        default=0.90,
    )
    parser.add_argument("--max-ci-half-width", type=float, default=0.03)
    parser.add_argument("--near-duplicate-hamming", type=int, default=4)
    return parser.parse_args()


def resolve_input(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def label_piece_ids(example: Mapping[str, Any], slot: str) -> tuple[int, ...]:
    label = (example.get("core_slot_labels") or {}).get(slot) or {}
    output = []
    for value in label.get("piece_ids") or []:
        try:
            piece_id = int(value)
        except (TypeError, ValueError):
            continue
        if piece_id > 0:
            output.append(piece_id)
    return tuple(sorted(set(output)))


def build_identity_roots(
    examples: Sequence[Mapping[str, Any]],
    slot: str,
) -> dict[int, int]:
    groups = UnionFind()
    for example in examples:
        groups.union_all(label_piece_ids(example, slot))
    return {piece_id: groups.find(piece_id) for piece_id in groups.parent}


def identity_for(
    example: Mapping[str, Any],
    slot: str,
    roots: Mapping[int, int],
) -> int | None:
    piece_ids = label_piece_ids(example, slot)
    if not piece_ids:
        return None
    return min(roots.get(piece_id, piece_id) for piece_id in piece_ids)


def wilson_half_width(
    n: int,
    proportion: float = 0.8,
    z: float = 1.959963984540054,
) -> float | None:
    if n <= 0:
        return None
    denominator = 1.0 + z * z / n
    variance = proportion * (1.0 - proportion) / n + z * z / (4.0 * n * n)
    return z * math.sqrt(variance) / denominator


def wilson_interval(
    successes: int,
    total: int,
    z: float = 1.959963984540054,
) -> tuple[float, float] | None:
    if total <= 0 or successes < 0 or successes > total:
        return None
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (
        proportion + z * z / (2.0 * total)
    ) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return center - half_width, center + half_width


def dhash(image: Image.Image, size: int = 8) -> int:
    grayscale = image.convert("L").resize((size + 1, size))
    pixels = list(grayscale.getdata())
    value = 0
    for row in range(size):
        offset = row * (size + 1)
        for column in range(size):
            value <<= 1
            value |= pixels[offset + column] > pixels[offset + column + 1]
    return value


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def inspect_passed_images(
    examples: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    unreadable: list[dict[str, str]] = []
    dimension_mismatch: list[dict[str, Any]] = []
    sha_to_ids: defaultdict[str, list[str]] = defaultdict(list)

    for example in examples:
        example_id = str(example.get("example_id"))
        path = resolve_project_path(example.get("local_path"))
        if path is None or not path.is_file():
            missing.append(example_id)
            continue
        try:
            sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            with Image.open(path) as image:
                image.load()
                width, height = image.size
                perceptual_hash = dhash(image)
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            unreadable.append({"example_id": example_id, "error": str(exc)})
            continue
        expected = (example.get("width"), example.get("height"))
        if all(value is not None for value in expected) and (
            width,
            height,
        ) != tuple(int(value) for value in expected):
            dimension_mismatch.append(
                {
                    "example_id": example_id,
                    "expected": list(expected),
                    "actual": [width, height],
                }
            )
        rows[example_id] = {
            "sha256": sha256,
            "dhash": perceptual_hash,
            "width": width,
            "height": height,
        }
        sha_to_ids[sha256].append(example_id)

    exact_groups = [ids for ids in sha_to_ids.values() if len(ids) > 1]
    return rows, {
        "checked": len(examples),
        "readable": len(rows),
        "missing_count": len(missing),
        "missing_examples": missing[:25],
        "unreadable_count": len(unreadable),
        "unreadable_examples": unreadable[:25],
        "dimension_mismatch_count": len(dimension_mismatch),
        "dimension_mismatch_examples": dimension_mismatch[:25],
        "exact_duplicate_group_count": len(exact_groups),
        "exact_duplicate_row_count": sum(len(ids) for ids in exact_groups),
        "exact_duplicate_examples": exact_groups[:25],
    }


def find_near_duplicate_pairs(
    examples: Sequence[Mapping[str, Any]],
    image_rows: Mapping[str, Mapping[str, Any]],
    max_hamming: int,
) -> list[dict[str, Any]]:
    comparable = []
    for example in examples:
        example_id = str(example.get("example_id"))
        image_row = image_rows.get(example_id)
        if image_row is None:
            continue
        comparable.append(
            (
                example_id,
                str(example.get("glamour_id")),
                int(image_row["dhash"]),
                str(image_row["sha256"]),
            )
        )
    pairs: list[dict[str, Any]] = []
    for index, left in enumerate(comparable):
        for right in comparable[index + 1 :]:
            if left[1] == right[1] or left[3] == right[3]:
                continue
            distance = hamming_distance(left[2], right[2])
            if distance <= max_hamming:
                pairs.append(
                    {
                        "left_example_id": left[0],
                        "left_glamour_id": left[1],
                        "right_example_id": right[0],
                        "right_glamour_id": right[1],
                        "hamming": distance,
                    }
                )
    return pairs


def load_index_piece_ids(index_dir: Path, slot: str) -> set[int]:
    rows = read_jsonl(resolve_input(index_dir) / "metadata.jsonl")
    output: set[int] = set()
    for row in rows:
        if str(row.get("slot") or "").upper() != slot:
            continue
        for piece in row.get("pieces") or []:
            try:
                piece_id = int(piece.get("piece_id"))
            except (AttributeError, TypeError, ValueError):
                continue
            if piece_id > 0:
                output.add(piece_id)
    return output


def load_sqlite_piece_ids(path: Path) -> dict[str, set[int]]:
    output = {slot: set() for slot in SLOTS}
    connection = sqlite3.connect(resolve_input(path))
    try:
        for piece_id, slot, occupied_json in connection.execute(
            "SELECT piece_id, slot, occupied_slots_json FROM gear_pieces"
        ):
            slots = {str(slot).upper()}
            try:
                slots.update(str(value).upper() for value in json.loads(occupied_json))
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
            for target in slots.intersection(SLOTS):
                output[target].add(int(piece_id))
    finally:
        connection.close()
    return output


def percentile(values: Sequence[int], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def profile_slot(
    slot: str,
    passed_examples: Sequence[Mapping[str, Any]],
    sqlite_piece_ids: set[int],
    primary_piece_ids: set[int],
    ensemble_piece_sets: Sequence[set[int]],
    roots: Mapping[int, int],
    thresholds: Thresholds,
) -> tuple[dict[str, Any], list[Mapping[str, Any]]]:
    labelled = [
        example for example in passed_examples if label_piece_ids(example, slot)
    ]
    covered = [
        example
        for example in labelled
        if set(label_piece_ids(example, slot)).intersection(primary_piece_ids)
    ]
    sqlite_covered = [
        example
        for example in labelled
        if set(label_piece_ids(example, slot)).intersection(sqlite_piece_ids)
    ]
    all_branch_covered = [
        example
        for example in labelled
        if all(
            set(label_piece_ids(example, slot)).intersection(piece_ids)
            for piece_ids in (primary_piece_ids, *ensemble_piece_sets)
        )
    ]

    def uncovered_examples(piece_ids: set[int]) -> list[dict[str, Any]]:
        rows = []
        for example in labelled:
            label_ids = label_piece_ids(example, slot)
            if set(label_ids).intersection(piece_ids):
                continue
            label = (example.get("core_slot_labels") or {}).get(slot) or {}
            rows.append(
                {
                    "example_id": str(example.get("example_id")),
                    "glamour_id": str(example.get("glamour_id")),
                    "item_name": str(label.get("item_name") or ""),
                    "piece_ids": list(label_ids),
                }
            )
        return rows

    primary_uncovered = uncovered_examples(primary_piece_ids)
    sqlite_uncovered = uncovered_examples(sqlite_piece_ids)
    all_branch_uncovered = [
        row
        for row in (
            {
                "example_id": str(example.get("example_id")),
                "glamour_id": str(example.get("glamour_id")),
                "item_name": str(
                    (
                        (example.get("core_slot_labels") or {}).get(slot) or {}
                    ).get("item_name")
                    or ""
                ),
                "piece_ids": list(label_piece_ids(example, slot)),
            }
            for example in labelled
        )
        if not all(
            set(row["piece_ids"]).intersection(piece_ids)
            for piece_ids in (primary_piece_ids, *ensemble_piece_sets)
        )
    ]
    identity_counts = Counter(
        identity_for(example, slot, roots) for example in labelled
    )
    identity_counts.pop(None, None)
    identity_sizes = list(identity_counts.values())
    glamours_by_identity: defaultdict[int, set[str]] = defaultdict(set)
    for example in labelled:
        identity = identity_for(example, slot, roots)
        if identity is not None:
            glamours_by_identity[identity].add(str(example.get("glamour_id")))
    identity_glamour_sizes = [
        len(glamour_ids) for glamour_ids in glamours_by_identity.values()
    ]
    multi_glamour_identity_count = sum(
        count >= 2 for count in identity_glamour_sizes
    )
    multi_glamour_identity_rate = (
        multi_glamour_identity_count / len(identity_glamour_sizes)
        if identity_glamour_sizes
        else 0.0
    )
    gender_counts = Counter(
        str(example.get("model_gender") or "unknown") for example in covered
    )
    race_counts = Counter(
        str(example.get("model_race") or "unknown") for example in covered
    )
    label_type_counts = Counter(
        str(
            ((example.get("core_slot_labels") or {}).get(slot) or {}).get(
                "label_type"
            )
            or "unknown"
        )
        for example in labelled
    )
    invalid_piece_rows = [
        str(example.get("example_id"))
        for example in passed_examples
        if (example.get("core_slot_labels") or {}).get(slot)
        and not label_piece_ids(example, slot)
    ]
    low_gender = {
        key: count
        for key, count in gender_counts.items()
        if key != "unknown" and count < thresholds.min_gender_images
    }
    low_race = {
        key: count
        for key, count in race_counts.items()
        if key != "unknown" and count < thresholds.min_race_images
    }
    return {
        "quality_passed_images": len(passed_examples),
        "labelled_images": len(labelled),
        "labelled_glamours": len(
            {str(example.get("glamour_id")) for example in labelled}
        ),
        "unlabelled_images": len(passed_examples) - len(labelled),
        "invalid_piece_label_count": len(invalid_piece_rows),
        "invalid_piece_label_examples": invalid_piece_rows[:25],
        "label_type_counts": dict(sorted(label_type_counts.items())),
        "appearance_identity_count": len(identity_counts),
        "singleton_identity_count": sum(
            count == 1 for count in identity_counts.values()
        ),
        "identity_images_p50": percentile(identity_sizes, 0.5),
        "identity_images_p90": percentile(identity_sizes, 0.9),
        "single_glamour_identity_count": sum(
            count == 1 for count in identity_glamour_sizes
        ),
        "multi_glamour_identity_count": multi_glamour_identity_count,
        "multi_glamour_identity_rate": multi_glamour_identity_rate,
        "identity_glamours_p50": percentile(identity_glamour_sizes, 0.5),
        "identity_glamours_p90": percentile(identity_glamour_sizes, 0.9),
        "sqlite_covered_images": len(sqlite_covered),
        "sqlite_coverage_rate": (
            len(sqlite_covered) / len(labelled) if labelled else 0.0
        ),
        "sqlite_uncovered_examples": sqlite_uncovered[:25],
        "primary_index_piece_count": len(primary_piece_ids),
        "primary_covered_images": len(covered),
        "primary_coverage_rate": len(covered) / len(labelled) if labelled else 0.0,
        "primary_uncovered_examples": primary_uncovered[:25],
        "all_active_branches_covered_images": len(all_branch_covered),
        "all_active_branches_coverage_rate": (
            len(all_branch_covered) / len(labelled) if labelled else 0.0
        ),
        "all_active_branches_uncovered_examples": all_branch_uncovered[:25],
        "gender_counts": dict(sorted(gender_counts.items())),
        "race_counts": dict(sorted(race_counts.items())),
        "under_threshold_gender_counts": low_gender,
        "under_threshold_race_counts": low_race,
        "wilson_half_width_at_80pct": wilson_half_width(len(covered)),
    }, covered


def role_pairs_are_leakage(left: str, right: str) -> bool:
    roles = {left, right}
    return "train" in roles and bool(roles.intersection({"validation", "blind_test"}))


def audit_folds(
    slot: str,
    examples: Sequence[Mapping[str, Any]],
    roots: Mapping[int, int],
    image_rows: Mapping[str, Mapping[str, Any]],
    near_pairs: Sequence[Mapping[str, Any]],
    thresholds: Thresholds,
    registered_eligible: int | None,
) -> dict[str, Any]:
    fold_rows = []
    pooled_validation_examples: set[str] = set()
    identity_leaks = []
    exact_content_leaks = []
    near_content_leaks = []
    assignments_by_fold = []

    examples_by_id = {
        str(example.get("example_id")): example for example in examples
    }
    for fold_path in sorted(resolve_input(FOLD_DIRS[slot]).glob("fold-*.json")):
        payload = json.loads(fold_path.read_text(encoding="utf-8"))
        assignments = {
            str(key): str(value)
            for key, value in (payload.get("assignments") or {}).items()
        }
        assignments_by_fold.append(assignments)
        role_counts = Counter()
        identities_by_role: defaultdict[str, set[int]] = defaultdict(set)
        hashes_by_role: defaultdict[str, set[str]] = defaultdict(set)
        missing_assignments = []
        for example in examples:
            example_id = str(example.get("example_id"))
            glamour_id = str(example.get("glamour_id"))
            role = assignments.get(glamour_id)
            if role is None:
                missing_assignments.append(example_id)
                continue
            role_counts[role] += 1
            identity = identity_for(example, slot, roots)
            if identity is not None:
                identities_by_role[role].add(identity)
            image_row = image_rows.get(example_id)
            if image_row is not None:
                hashes_by_role[role].add(str(image_row["sha256"]))
            if role == "validation":
                pooled_validation_examples.add(example_id)
        leaked_identities = sorted(
            identities_by_role["train"].intersection(
                identities_by_role["validation"]
            )
        )
        leaked_hashes = sorted(
            hashes_by_role["train"].intersection(hashes_by_role["validation"])
        )
        for identity in leaked_identities:
            identity_leaks.append(
                {"fold": fold_path.name, "identity": identity}
            )
        for sha256 in leaked_hashes:
            exact_content_leaks.append(
                {"fold": fold_path.name, "sha256": sha256}
            )
        for pair in near_pairs:
            left = examples_by_id.get(str(pair["left_example_id"]))
            right = examples_by_id.get(str(pair["right_example_id"]))
            if left is None or right is None:
                continue
            left_role = assignments.get(str(left.get("glamour_id")))
            right_role = assignments.get(str(right.get("glamour_id")))
            if left_role and right_role and role_pairs_are_leakage(
                left_role, right_role
            ):
                near_content_leaks.append(
                    {
                        "fold": fold_path.name,
                        **pair,
                        "left_role": left_role,
                        "right_role": right_role,
                    }
                )
        fold_rows.append(
            {
                "fold": fold_path.name,
                "role_image_counts": dict(sorted(role_counts.items())),
                "missing_assignment_count": len(missing_assignments),
                "identity_leak_count": len(leaked_identities),
                "exact_content_leak_count": len(leaked_hashes),
                "train_quantity_ok": (
                    role_counts["train"]
                    >= thresholds.min_train_images_per_fold
                ),
                "validation_quantity_ok": (
                    role_counts["validation"]
                    >= thresholds.min_validation_images_per_fold
                ),
            }
        )

    assignment_rotation_issues = []
    glamour_ids = {str(example.get("glamour_id")) for example in examples}
    for glamour_id in sorted(glamour_ids):
        roles = [
            assignments.get(glamour_id) for assignments in assignments_by_fold
        ]
        if all(role == "blind_test" for role in roles):
            continue
        if roles.count("validation") != 1 or roles.count("train") != 2:
            assignment_rotation_issues.append(
                {"glamour_id": glamour_id, "roles": roles}
            )

    pooled = len(pooled_validation_examples)
    return {
        "folds": fold_rows,
        "fold_count": len(fold_rows),
        "pooled_validation_images": pooled,
        "registered_eligible": registered_eligible,
        "registered_denominator_matches": (
            registered_eligible is None or pooled == registered_eligible
        ),
        "identity_leak_count": len(identity_leaks),
        "identity_leak_examples": identity_leaks[:25],
        "exact_content_leak_count": len(exact_content_leaks),
        "exact_content_leak_examples": exact_content_leaks[:25],
        "near_content_leak_count": len(near_content_leaks),
        "near_content_leak_examples": near_content_leaks[:25],
        "assignment_rotation_issue_count": len(assignment_rotation_issues),
        "assignment_rotation_issue_examples": assignment_rotation_issues[:25],
    }


def finding(
    severity: str,
    code: str,
    title: str,
    evidence: str,
    impact: str,
    remediation: str,
    confidence: str = "high",
) -> dict[str, str]:
    return {
        "severity": severity,
        "code": code,
        "title": title,
        "evidence": evidence,
        "impact": impact,
        "remediation": remediation,
        "confidence": confidence,
    }


def build_findings(
    manifest: Mapping[str, Any],
    image_audit: Mapping[str, Any],
    slot_profiles: Mapping[str, Mapping[str, Any]],
    fold_audits: Mapping[str, Mapping[str, Any]],
    visibility_audit: Mapping[str, Mapping[str, Any]],
    pose_visibility_audit: Mapping[str, Mapping[str, Any]],
    visibility_curation: Mapping[str, Mapping[str, Any]],
    thresholds: Thresholds,
) -> list[dict[str, str]]:
    findings = []
    if (
        manifest["duplicate_example_id_count"]
        or manifest["quality_duplicate_example_id_count"]
        or manifest["quality_missing_example_count"]
    ):
        findings.append(
            finding(
                "critical",
                "manifest_key_integrity",
                "标签与质量清单的图片键不完整或不唯一",
                (
                    f"标签重复 {manifest['duplicate_example_id_count']}，"
                    f"质量重复 {manifest['quality_duplicate_example_id_count']}，"
                    f"质量缺失 {manifest['quality_missing_example_count']}。"
                ),
                "训练行可能重复、漏筛或错误联结，所有下游指标均不可信。",
                "修复 example_id 主键并重新生成最终质量清单。",
            )
        )
    if any(
        image_audit[key]
        for key in (
            "missing_count",
            "unreadable_count",
            "dimension_mismatch_count",
        )
    ):
        findings.append(
            finding(
                "critical",
                "image_integrity",
                "质量通过图片并非全部可读且尺寸一致",
                (
                    f"缺失 {image_audit['missing_count']}，不可读 "
                    f"{image_audit['unreadable_count']}，尺寸不符 "
                    f"{image_audit['dimension_mismatch_count']}。"
                ),
                "训练可能中断或把错误图片映射到标签。",
                "剔除或重新下载问题图片，再冻结新的质量清单。",
            )
        )
    (
        evidence_ready_slots,
        curation_ready_slots,
        visibility_sample_failures,
    ) = resolve_visibility_evidence(
        visibility_audit,
        visibility_curation,
        thresholds,
    )
    incomplete_visibility = {
        slot: row
        for slot, row in visibility_audit.items()
        if row["reviewed_images"] < row["labelled_images"]
    }
    unresolved_incomplete_visibility = {
        slot: row
        for slot, row in incomplete_visibility.items()
        if slot not in evidence_ready_slots
    }
    if unresolved_incomplete_visibility:
        evidence = "；".join(
            (
                f"{slot} {row['reviewed_images']}/{row['labelled_images']}"
                f"（clear {row['visibility_counts'].get('clear', 0)}，"
                f"partial {row['visibility_counts'].get('partial', 0)}，"
                f"unusable {row['visibility_counts'].get('unusable', 0)}）"
            )
            for slot, row in unresolved_incomplete_visibility.items()
        )
        findings.append(
            finding(
                "critical",
                "slot_visibility_incomplete",
                "逐槽可见性复核尚未覆盖全部训练样本",
                evidence,
                "HANDS、HEAD 等小部位可能被袖子、头发、武器或画面边缘遮挡，形成不可学习的标签噪声。",
                "继续独立人工复核并写入 clear/partial/unusable 与遮挡原因；HANDS 优先，训练只消费 usable 行。",
            )
        )
    elif incomplete_visibility:
        findings.append(
            finding(
                "medium",
                "slot_visibility_statistically_validated",
                "未做全量人工普查，但逐槽可见性已有独立统计验证",
                "；".join(
                    f"{slot}={('curation' if slot in curation_ready_slots else 'independent_sample')}"
                    for slot in sorted(incomplete_visibility)
                ),
                "残余标签噪声由独立样本或完整审计的修复策略约束，已知不可用行会被训练清单剔除。",
                "后续训练必须冻结并消费本次人工决策与剔除清单。",
            )
        )
    for slot, failure in visibility_sample_failures.items():
        if slot in curation_ready_slots:
            continue
        findings.append(
            finding(
                "high",
                f"{slot.lower()}_independent_visibility_quality",
                f"{slot} 独立语义可见性样本未达到训练质量门槛",
                failure["evidence"],
                (
                    "玩家图中的真值可能实际不可辨，导致模型被迫把同套装、"
                    "角色或场景上下文学成装备身份。"
                ),
                (
                    "按遮挡原因剔除不可用行、补充独立 glamour，并在另一批"
                    "固定随机样本上重新验证；自动规则只可用于人工队列排序。"
                ),
            )
        )
    duplicate_visibility_reviews = sum(
        row["duplicate_review_count"] for row in visibility_audit.values()
    )
    if duplicate_visibility_reviews:
        findings.append(
            finding(
                "critical",
                "slot_visibility_duplicate_keys",
                "逐槽可见性复核键不唯一",
                f"重复 example_id/slot 复核行 {duplicate_visibility_reviews}。",
                "后写覆盖会令训练过滤结果不可复现。",
                "去重并保留单一、可追溯的最终人工决策。",
            )
        )
    structural_failures = {
        slot: row
        for slot, row in pose_visibility_audit.items()
        if row["missing_images"]
        or row["error_images"]
        or row["decision_counts"].get("unusable", 0)
    }
    if structural_failures:
        findings.append(
            finding(
                "critical",
                "pose_slot_visibility",
                "结构可见性预筛存在缺失、错误或不可用部位",
                "；".join(
                    (
                        f"{slot} missing {row['missing_images']}，"
                        f"errors {row['error_images']}，"
                        f"unusable {row['decision_counts'].get('unusable', 0)}"
                    )
                    for slot, row in structural_failures.items()
                ),
                "训练会包含不在画面内、关键点不可信或有效裁剪过小的目标。",
                "剔除结构不可用图片并重新冻结分组折；错误行必须修复后重跑。",
            )
        )
    for slot, profile in slot_profiles.items():
        if profile["primary_coverage_rate"] < 1.0:
            findings.append(
                finding(
                    "critical",
                    f"{slot.lower()}_candidate_coverage",
                    f"{slot} 标签未被活动候选库完全覆盖",
                    (
                        f"{profile['primary_covered_images']}/"
                        f"{profile['labelled_images']}，覆盖率 "
                        f"{profile['primary_coverage_rate']:.2%}。"
                    ),
                    "真值不在库时 Top-k 必然失败，训练负样本与评估分母也会失真。",
                    "补齐装备渲染/索引，或在冻结前明确排除并记录缺库原因。",
                )
            )
        if profile["labelled_images"] < thresholds.min_labelled_images:
            findings.append(
                finding(
                    "high",
                    f"{slot.lower()}_image_volume",
                    f"{slot} 高质量有标签图片低于预登记门槛",
                    (
                        f"{profile['labelled_images']} < "
                        f"{thresholds.min_labelled_images}；"
                        f"覆盖后仅 {profile['primary_covered_images']}。"
                    ),
                    "难以覆盖姿态、遮挡、染色、种族和装备形态长尾。",
                    "按失败形态定向扩充独立 glamour，而不是复制同一搭配多图。",
                )
            )
        if (
            profile["multi_glamour_identity_rate"]
            < thresholds.min_multi_glamour_identity_rate
        ):
            findings.append(
                finding(
                    "high",
                    f"{slot.lower()}_independent_glamour_coverage",
                    f"{slot} 独立玩家搭配覆盖不足",
                    (
                        f"{profile['multi_glamour_identity_count']}/"
                        f"{profile['appearance_identity_count']} 外观族至少有 "
                        "2 个独立 glamour，比例 "
                        f"{profile['multi_glamour_identity_rate']:.2%} < "
                        f"{thresholds.min_multi_glamour_identity_rate:.2%}。"
                    ),
                    "单一搭配的多张近似视图无法证明模型学到装备本身，容易把角色、整套服装或场景上下文当成捷径。",
                    "优先为单 glamour 外观补充不同玩家、姿态、染色和场景的独立搭配图，并保持 glamour 分组划分。",
                )
            )
        if profile["under_threshold_gender_counts"] or profile[
            "under_threshold_race_counts"
        ]:
            findings.append(
                finding(
                    "high",
                    f"{slot.lower()}_domain_volume",
                    f"{slot} 存在低样本性别/种族切片",
                    (
                        f"性别 {profile['under_threshold_gender_counts']}；"
                        f"种族 {profile['under_threshold_race_counts']}。"
                    ),
                    "总体提升可能掩盖小域退化，冻结门控缺少稳定验证分母。",
                    "优先补采低覆盖切片，或预登记合并切片及不确定性口径。",
                )
            )
    for slot, audit in fold_audits.items():
        quantity_failures = [
            {
                "fold": row["fold"],
                "train": row["role_image_counts"].get("train", 0),
                "validation": row["role_image_counts"].get("validation", 0),
            }
            for row in audit["folds"]
            if not row["train_quantity_ok"]
            or not row["validation_quantity_ok"]
        ]
        if quantity_failures:
            findings.append(
                finding(
                    "high",
                    f"{slot.lower()}_fold_volume",
                    f"{slot} 至少一折低于预登记训练/验证数量门槛",
                    (
                        f"问题折 {quantity_failures}；门槛 train >= "
                        f"{thresholds.min_train_images_per_fold}，validation >= "
                        f"{thresholds.min_validation_images_per_fold}。"
                    ),
                    "学习曲线和折间方差可能受单折数据不足影响，无法确认数量充足。",
                    "补充独立 glamour，或在训练前基于学习曲线重新登记数量门槛。",
                )
            )
        if any(
            row["missing_assignment_count"] for row in audit["folds"]
        ) or audit["assignment_rotation_issue_count"]:
            findings.append(
                finding(
                    "critical",
                    f"{slot.lower()}_split_completeness",
                    f"{slot} 折分配不完整或轮换不符合预登记规则",
                    (
                        "各折缺失分配 "
                        f"{[row['missing_assignment_count'] for row in audit['folds']]}；"
                        "轮换异常 "
                        f"{audit['assignment_rotation_issue_count']}。"
                    ),
                    "样本可能被静默遗漏或重复进入不正确角色，验证口径失真。",
                    "重新生成分组折并通过完整性、轮换和泄漏检查。",
                )
            )
        if audit["identity_leak_count"] or audit["exact_content_leak_count"]:
            findings.append(
                finding(
                    "critical",
                    f"{slot.lower()}_split_leakage",
                    f"{slot} 严格折存在身份或图片内容泄漏",
                    (
                        f"身份泄漏 {audit['identity_leak_count']}，"
                        f"精确内容泄漏 {audit['exact_content_leak_count']}。"
                    ),
                    "验证指标会高估未知外观泛化。",
                    "按官方外观连通分量和内容哈希重新分组划分。",
                )
            )
        if audit["near_content_leak_count"]:
            findings.append(
                finding(
                    "high",
                    f"{slot.lower()}_near_duplicate_leakage",
                    f"{slot} 折间存在疑似近重复图片",
                    f"dHash 阈值内跨训练/验证配对 {audit['near_content_leak_count']}。",
                    "同一截图的裁切/压缩版本可能造成隐性泄漏。",
                    "人工复核配对；确认同源后按内容组重新划分。",
                    confidence="medium",
                )
            )
        if not audit["registered_denominator_matches"]:
            findings.append(
                finding(
                    "critical",
                    f"{slot.lower()}_denominator_drift",
                    f"{slot} 当前数据无法复现注册评估分母",
                    (
                        f"审计得到 {audit['pooled_validation_images']}，"
                        f"注册值 {audit['registered_eligible']}。"
                    ),
                    "现有指标与当前数据版本不可直接比较。",
                    "冻结数据版本并重建评估报告，禁止沿用旧分母。",
                )
            )
        profile = slot_profiles[slot]
        if profile["wilson_half_width_at_80pct"] > thresholds.max_ci_half_width:
            findings.append(
                finding(
                    "high",
                    f"{slot.lower()}_evaluation_precision",
                    f"{slot} 验证分母不足以稳定判断 80% 门槛",
                    (
                        "80% 命中率下 Wilson 95% 区间半宽 "
                        f"{profile['wilson_half_width_at_80pct']:.2%} > "
                        f"{thresholds.max_ci_half_width:.2%}。"
                    ),
                    "接近 80% 时，随机误差足以改变是否达标的结论。",
                    "增加独立验证 glamour，直到预登记置信区间精度达标。",
                )
            )
    if image_audit["exact_duplicate_group_count"]:
        findings.append(
            finding(
                "medium",
                "exact_duplicate_images",
                "高质量集合含精确重复图片",
                (
                    f"{image_audit['exact_duplicate_group_count']} 组，"
                    f"{image_audit['exact_duplicate_row_count']} 行。"
                ),
                "重复图片会放大个别搭配权重；若跨 glamour 还可能泄漏。",
                "按内容哈希归组，只保留一个训练权重或做 group-aware 采样。",
            )
        )
    return sorted(
        findings,
        key=lambda row: (SEVERITY_ORDER[row["severity"]], row["code"]),
    )


def audit_slot_visibility(
    passed_examples: Sequence[Mapping[str, Any]],
    review_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    review_counts = Counter(
        (str(row.get("example_id")), str(row.get("slot")))
        for row in review_rows
    )
    review_by_key = {
        (str(row.get("example_id")), str(row.get("slot"))): row
        for row in review_rows
    }
    output = {}
    glamour_by_id = {
        str(example.get("example_id")): str(example.get("glamour_id"))
        for example in passed_examples
    }
    for slot in SLOTS:
        labelled_ids = {
            str(example.get("example_id"))
            for example in passed_examples
            if label_piece_ids(example, slot)
        }
        relevant = [
            review_by_key[(example_id, slot)]
            for example_id in sorted(labelled_ids)
            if (example_id, slot) in review_by_key
        ]
        visibility_counts = Counter(
            str(row.get("visibility") or "unreviewed") for row in relevant
        )
        usability_counts = Counter(
            str(row.get("usability") or "unreviewed") for row in relevant
        )
        issue_counts = Counter(str(row.get("issue") or "none") for row in relevant)
        independent_rows = [
            row
            for row in relevant
            if "independent_glamour" in str(row.get("review_source") or "")
        ]
        independent_usable = sum(
            str(row.get("usability")) == "usable"
            for row in independent_rows
        )
        independent_interval = wilson_interval(
            independent_usable,
            len(independent_rows),
        )
        output[slot] = {
            "labelled_images": len(labelled_ids),
            "reviewed_images": len(relevant),
            "reviewed_glamours": len(
                {
                    glamour_by_id[str(row.get("example_id"))]
                    for row in relevant
                    if str(row.get("example_id")) in glamour_by_id
                }
            ),
            "review_coverage_rate": (
                len(relevant) / len(labelled_ids) if labelled_ids else 0.0
            ),
            "visibility_counts": dict(sorted(visibility_counts.items())),
            "usability_counts": dict(sorted(usability_counts.items())),
            "issue_counts": dict(sorted(issue_counts.items())),
            "independent_sample_images": len(independent_rows),
            "independent_sample_usable": independent_usable,
            "independent_sample_usable_rate": (
                independent_usable / len(independent_rows)
                if independent_rows
                else None
            ),
            "independent_sample_usable_wilson_95": (
                list(independent_interval)
                if independent_interval is not None
                else None
            ),
            "duplicate_review_count": sum(
                max(0, review_counts[(example_id, slot)] - 1)
                for example_id in labelled_ids
            ),
        }
    return output


def independent_visibility_sample_failures(
    visibility_audit: Mapping[str, Mapping[str, Any]],
    thresholds: Thresholds,
) -> dict[str, dict[str, Any]]:
    failures = {}
    for slot, audit in visibility_audit.items():
        sample_size = int(audit.get("independent_sample_images") or 0)
        usable = int(audit.get("independent_sample_usable") or 0)
        usable_rate = audit.get("independent_sample_usable_rate")
        interval = audit.get("independent_sample_usable_wilson_95")
        wilson_low = interval[0] if interval else None
        reasons = []
        if sample_size < thresholds.min_independent_visibility_sample:
            reasons.append(
                f"独立样本 {sample_size} < "
                f"{thresholds.min_independent_visibility_sample}"
            )
        if (
            usable_rate is None
            or usable_rate < thresholds.min_independent_sample_usable_rate
        ):
            usable_text = (
                f"{usable_rate:.2%}"
                if usable_rate is not None
                else "missing"
            )
            reasons.append(
                "usable 比例 "
                f"{usable_text} < "
                f"{thresholds.min_independent_sample_usable_rate:.2%}"
            )
        if (
            wilson_low is None
            or wilson_low < thresholds.min_independent_sample_wilson_low
        ):
            wilson_text = (
                f"{wilson_low:.2%}"
                if wilson_low is not None
                else "missing"
            )
            reasons.append(
                "Wilson 95% 下界 "
                f"{wilson_text} < "
                f"{thresholds.min_independent_sample_wilson_low:.2%}"
            )
        if reasons:
            failures[slot] = {
                "sample_size": sample_size,
                "usable": usable,
                "usable_rate": usable_rate,
                "wilson_low": wilson_low,
                "reasons": reasons,
                "evidence": (
                    f"独立样本 {usable}/{sample_size} usable；"
                    + "；".join(reasons)
                    + "。"
                ),
            }
    return failures


def resolve_visibility_evidence(
    visibility_audit: Mapping[str, Mapping[str, Any]],
    visibility_curation: Mapping[str, Mapping[str, Any]],
    thresholds: Thresholds,
) -> tuple[set[str], set[str], dict[str, dict[str, Any]]]:
    sample_failures = independent_visibility_sample_failures(
        visibility_audit,
        thresholds,
    )
    curation_ready_slots = {
        slot
        for slot, report in visibility_curation.items()
        if report.get("status") == "ready"
        and bool(report.get("checks"))
        and all((report.get("checks") or {}).values())
    }
    ready_slots = (
        set(visibility_audit) - set(sample_failures)
    ) | curation_ready_slots
    return ready_slots, curation_ready_slots, sample_failures


def audit_pose_slot_visibility(
    passed_examples: Sequence[Mapping[str, Any]],
    pose_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_id = {
        str(row.get("example_id")): row
        for row in pose_rows
    }
    output = {}
    for slot in SLOTS:
        labelled_ids = {
            str(example.get("example_id"))
            for example in passed_examples
            if label_piece_ids(example, slot)
        }
        present_rows = [
            by_id[example_id]
            for example_id in labelled_ids
            if example_id in by_id
        ]
        decisions = Counter(
            str(
                ((row.get("slot_visibility") or {}).get(slot) or {}).get(
                    "decision"
                )
                or "missing"
            )
            for row in present_rows
            if not row.get("error")
        )
        output[slot] = {
            "labelled_images": len(labelled_ids),
            "reviewed_images": len(present_rows),
            "missing_images": len(labelled_ids) - len(present_rows),
            "error_images": sum(bool(row.get("error")) for row in present_rows),
            "decision_counts": dict(sorted(decisions.items())),
        }
    return output


def markdown_report(payload: Mapping[str, Any]) -> str:
    lines = [
        "# 玩家搭配训练准入审计",
        "",
        f"生成时间：{payload['generated_at']}",
        "",
        "## 技术结论",
        "",
    ]
    if payload["training_ready"]:
        lines.append("**通过：当前规范数据满足已登记的训练准入检查。**")
    else:
        lines.append(
            "**不通过：当前数据不能启动新训练。** "
            "必须先解决所有 critical 问题，并重新运行本审计。"
        )
    lines.extend(
        [
            "",
            f"- 质量通过图片：{payload['manifest']['passed_quality_count']:,}",
            f"- Critical / High 问题：{payload['severity_counts'].get('critical', 0)} / "
            f"{payload['severity_counts'].get('high', 0)}",
            f"- 精确重复组：{payload['images']['exact_duplicate_group_count']}",
            f"- 疑似近重复跨 glamour 配对：{payload['near_duplicates']['pair_count']}",
            "",
            "## 五部位数据与装备库覆盖",
            "",
            "| 部位 | 有标签图 | 外观族 | 单例族 | 姿态 clear / 人审 | 主索引覆盖 | 全分支覆盖 | 注册分母复现 |",
            "| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |",
        ]
    )
    for slot in SLOTS:
        profile = payload["slots"][slot]
        folds = payload["folds"][slot]
        lines.append(
            f"| {slot} | {profile['labelled_images']} | "
            f"{profile['appearance_identity_count']} | "
            f"{profile['singleton_identity_count']} | "
            f"{payload['pose_visibility'][slot]['decision_counts'].get('clear', 0)}"
            f" / {payload['visibility'][slot]['reviewed_images']} | "
            f"{profile['primary_coverage_rate']:.2%} | "
            f"{profile['all_active_branches_coverage_rate']:.2%} | "
            f"{'是' if folds['registered_denominator_matches'] else '否'} |"
        )
    lines.extend(
        [
            "",
            "图表省略说明：本报告的核心证据是逐槽准入状态和精确计数，审计表比比例图更能暴露分母与覆盖缺口。",
            "",
            "## 主要问题",
            "",
        ]
    )
    for row in payload["findings"]:
        lines.extend(
            [
                f"### [{row['severity'].upper()}] {row['title']}",
                "",
                f"- 证据：{row['evidence']}",
                f"- 影响：{row['impact']}",
                f"- 修复：{row['remediation']}",
                f"- 置信度：{row['confidence']}",
                "",
            ]
        )
    if not payload["findings"]:
        lines.extend(["没有发现违反准入门槛的问题。", ""])
    lines.extend(
        [
            "## 范围与定义",
            "",
            "- 粒度：一行是一张玩家搭配图片；同一 glamour 的多张图片必须作为同组。",
            "- 高质量：最终质量清单 `passed=true`，不是仅通过自动姿态模型。",
            "- 装备库覆盖：至少一个闭包真值 piece ID 出现在目标槽位活动索引。",
            "- 外观族：对闭包标签 piece ID 建连通分量，不把同模型 ID 当独立身份。",
            "- 近重复：64-bit dHash 汉明距离不超过登记阈值；仅作人工复核线索。",
            "",
            "## 方法与稳健性",
            "",
            "- 核对标签/质量主键、文件存在性、PIL 解码和声明尺寸。",
            "- 计算 SHA-256 精确重复与 dHash 近重复。",
            "- 读取实际 `current-best` 主索引/融合索引 metadata，而非只读 SQLite 目录。",
            "- 在每折检查 train/validation 外观连通分量和图片内容泄漏。",
            "- 全量姿态预筛只判断入镜、关键点和裁剪像素；语义遮挡仍由独立人工复核。",
            "- 数量门槛同时检查总图数和每个外观族的独立 glamour 数，避免把同一搭配多视图当作独立覆盖。",
            "- 用 80% 命中率下 Wilson 95% 区间半宽衡量验证分母统计精度。",
            "",
            "## 限制与不确定性",
            "",
            "- dHash 可能把构图相似但内容不同的图片标成近重复，必须人工复核。",
            "- 全身质量清单不能证明每个装备部位可见；这是当前最关键的未观测变量。",
            "- 数量门槛是训练准入下限，不证明数据足以达到 80%；达到目标仍需学习曲线验证。",
            "",
            "## 建议的下一步",
            "",
            "1. 先建立 HANDS 逐槽可见性复核表，并把不可见/被占用/袖子遮挡分开。",
            "2. 补齐所有活动候选库缺失真值，再冻结新分母。",
            "3. 人工复核跨折近重复配对并按同源组重划分。",
            "4. 数据门槛通过后先做 HANDS 学习曲线和失败漏斗，不直接长轮次训练。",
            "",
            "## 待确认问题",
            "",
            "- 多槽主件占用 HANDS/LEGS/FEET 时，目标是识别主件、返回“无独立装备”，还是两者并列？",
            "- 80% 目标是否要求每个性别/种族切片也达到 80%，还是只要求总体且切片不退化？",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    thresholds = Thresholds(
        min_labelled_images=args.min_labelled_images,
        min_train_images_per_fold=args.min_train_images_per_fold,
        min_validation_images_per_fold=args.min_validation_images_per_fold,
        min_gender_images=args.min_gender_images,
        min_race_images=args.min_race_images,
        min_multi_glamour_identity_rate=(
            args.min_multi_glamour_identity_rate
        ),
        min_independent_visibility_sample=(
            args.min_independent_visibility_sample
        ),
        min_independent_sample_usable_rate=(
            args.min_independent_sample_usable_rate
        ),
        min_independent_sample_wilson_low=(
            args.min_independent_sample_wilson_low
        ),
        max_ci_half_width=args.max_ci_half_width,
        near_duplicate_hamming=args.near_duplicate_hamming,
    )
    examples = read_jsonl(resolve_input(args.examples))
    quality_rows = read_jsonl(resolve_input(args.quality))
    example_counts = Counter(str(row.get("example_id")) for row in examples)
    quality_counts = Counter(str(row.get("example_id")) for row in quality_rows)
    quality_by_id = {
        str(row.get("example_id")): row for row in quality_rows
    }
    passed_examples = [
        row
        for row in examples
        if bool((quality_by_id.get(str(row.get("example_id"))) or {}).get("passed"))
    ]
    passed_ids = {str(row.get("example_id")) for row in passed_examples}
    manifest = {
        "example_count": len(examples),
        "quality_count": len(quality_rows),
        "duplicate_example_id_count": sum(
            count - 1 for count in example_counts.values() if count > 1
        ),
        "quality_duplicate_example_id_count": sum(
            count - 1 for count in quality_counts.values() if count > 1
        ),
        "quality_missing_example_count": sum(
            example_id not in quality_counts for example_id in example_counts
        ),
        "quality_orphan_count": sum(
            example_id not in example_counts for example_id in quality_counts
        ),
        "passed_quality_count": len(passed_examples),
        "passed_quality_rate": len(passed_examples) / len(examples) if examples else 0,
        "passed_unique_glamours": len(
            {str(row.get("glamour_id")) for row in passed_examples}
        ),
    }

    image_rows, image_audit = inspect_passed_images(passed_examples)
    near_pairs = find_near_duplicate_pairs(
        passed_examples,
        image_rows,
        thresholds.near_duplicate_hamming,
    )
    sqlite_piece_sets = load_sqlite_piece_ids(args.db)
    metrics_payload = json.loads(
        resolve_input(args.metrics).read_text(encoding="utf-8")
    )
    registered = {
        str(row["slot"]): int(row["eligible"])
        for row in metrics_payload.get("slots") or []
    }

    slot_profiles: dict[str, Any] = {}
    fold_audits: dict[str, Any] = {}
    for slot in SLOTS:
        roots = build_identity_roots(passed_examples, slot)
        primary = load_index_piece_ids(INDEX_DIRS[slot], slot)
        ensemble = [
            load_index_piece_ids(path, slot)
            for path in ENSEMBLE_INDEX_DIRS[slot]
        ]
        profile, covered = profile_slot(
            slot,
            passed_examples,
            sqlite_piece_sets[slot],
            primary,
            ensemble,
            roots,
            thresholds,
        )
        slot_profiles[slot] = profile
        fold_audits[slot] = audit_folds(
            slot,
            covered,
            roots,
            image_rows,
            near_pairs,
            thresholds,
            registered.get(slot),
        )

    visibility_paths = [
        resolve_input(path)
        for path in (
            args.visibility_reviews
            or [DEFAULT_VISIBILITY_REVIEWS]
        )
    ]
    visibility_rows = [
        row
        for path in visibility_paths
        if path.is_file()
        for row in read_jsonl(path)
    ]
    visibility_audit = audit_slot_visibility(passed_examples, visibility_rows)
    pose_visibility_path = resolve_input(args.pose_visibility)
    pose_visibility_rows = (
        read_jsonl(pose_visibility_path)
        if pose_visibility_path.is_file()
        else []
    )
    pose_visibility_audit = audit_pose_slot_visibility(
        passed_examples,
        pose_visibility_rows,
    )
    visibility_curation_paths = [
        resolve_input(path) for path in args.visibility_curation_report
    ]
    visibility_curation = {}
    for path in visibility_curation_paths:
        if not path.is_file():
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        slot = str(report.get("slot") or "").upper()
        if slot not in SLOTS:
            raise ValueError(f"Invalid visibility curation slot in {path}: {slot}")
        if slot in visibility_curation:
            raise ValueError(f"Duplicate visibility curation report for {slot}")
        visibility_curation[slot] = report
    findings = build_findings(
        manifest,
        image_audit,
        slot_profiles,
        fold_audits,
        visibility_audit,
        pose_visibility_audit,
        visibility_curation,
        thresholds,
    )
    severity_counts = Counter(row["severity"] for row in findings)
    training_ready = not any(
        row["severity"] in {"critical", "high"} for row in findings
    )
    payload = {
        "version": 1,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "training_ready": training_ready,
        "decision": "train" if training_ready else "do_not_train",
        "thresholds": thresholds.__dict__,
        "sources": {
            "examples": portable_path(resolve_input(args.examples)),
            "quality": portable_path(resolve_input(args.quality)),
            "sqlite": portable_path(resolve_input(args.db)),
            "metrics": portable_path(resolve_input(args.metrics)),
            "visibility_reviews": [
                portable_path(path)
                for path in visibility_paths
                if path.is_file()
            ],
            "pose_visibility": (
                portable_path(pose_visibility_path)
                if pose_visibility_path.is_file()
                else None
            ),
            "visibility_curation_reports": [
                portable_path(path)
                for path in visibility_curation_paths
                if path.is_file()
            ],
            "primary_indexes": {
                slot: portable_path(resolve_input(path))
                for slot, path in INDEX_DIRS.items()
            },
            "fold_dirs": {
                slot: portable_path(resolve_input(path))
                for slot, path in FOLD_DIRS.items()
            },
        },
        "manifest": manifest,
        "images": image_audit,
        "near_duplicates": {
            "threshold": thresholds.near_duplicate_hamming,
            "pair_count": len(near_pairs),
            "examples": near_pairs[:100],
        },
        "visibility": visibility_audit,
        "pose_visibility": pose_visibility_audit,
        "visibility_curation": visibility_curation,
        "slots": slot_profiles,
        "folds": fold_audits,
        "severity_counts": dict(sorted(severity_counts.items())),
        "findings": findings,
        "passed_example_ids_hashed": hashlib.sha256(
            "\n".join(sorted(passed_ids)).encode("utf-8")
        ).hexdigest(),
    }
    output_dir = resolve_input(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "audit.json", payload)
    (output_dir / "report.md").write_text(
        markdown_report(payload),
        encoding="utf-8",
    )
    print(json.dumps(
        {
            "training_ready": training_ready,
            "severity_counts": payload["severity_counts"],
            "output": portable_path(output_dir),
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0 if training_ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
