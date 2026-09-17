from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


BODY_STYLE_CLASSES = (
    "crop_top",
    "sleeveless",
    "regular_top",
    "long_top",
)
LEGS_STYLE_CLASSES = (
    "hot_shorts",
    "long_shorts",
    "short_skirt",
    "asymmetric_thigh_cut",
    "long_bottom",
)
FEET_STYLE_CLASSES = (
    "short_footwear",
    "tall_footwear",
)
STYLE_CLASSES_BY_SLOT = {
    "BODY": BODY_STYLE_CLASSES,
    "LEGS": LEGS_STYLE_CLASSES,
    "FEET": FEET_STYLE_CLASSES,
}

# The descriptions are deliberately about silhouette and skin coverage instead
# of colour or material.  They can be passed directly to a zero-shot
# image/text model, but this module intentionally does not own model inference.
BODY_STYLE_PROMPTS = (
    "a cropped jacket or top clearly exposing the waist or midriff",
    "a sleeveless top that exposes the shoulders",
    "a regular waist-length shirt or jacket covering the waist",
    "a long coat, tunic, robe, coverall, or one-piece outfit",
)
LEGS_STYLE_PROMPTS = (
    (
        "close-fitting very short shorts or hot pants with a clearly divided "
        "crotch and two separate leg openings"
    ),
    (
        "knee-length shorts with a clearly divided crotch and two separate "
        "trouser legs"
    ),
    (
        "a short skirt or loincloth forming continuous hanging fabric between "
        "the legs without a divided crotch"
    ),
    (
        "asymmetric bottoms or thigh-high stockings with a diagonal hem "
        "exposing one thigh"
    ),
    "long pants, tights, or leggings covering both legs",
)
FEET_STYLE_PROMPTS = (
    (
        "short footwear ending below the knee, including shoes, ankle boots, "
        "and calf boots"
    ),
    (
        "tall footwear reaching the knee or extending above the knee, "
        "including knee-high and thigh-high boots"
    ),
)
STYLE_PROMPTS_BY_SLOT = {
    "BODY": BODY_STYLE_PROMPTS,
    "LEGS": LEGS_STYLE_PROMPTS,
    "FEET": FEET_STYLE_PROMPTS,
}

# Convenience aliases for callers that prefer shorter names.
STYLE_CLASSES = STYLE_CLASSES_BY_SLOT
STYLE_PROMPTS = STYLE_PROMPTS_BY_SLOT

TARGET_STYLE_CLASSES = {
    "BODY": frozenset(("crop_top", "sleeveless", "long_top")),
    "LEGS": frozenset(
        ("hot_shorts", "long_shorts", "short_skirt", "asymmetric_thigh_cut")
    ),
    "FEET": frozenset(("tall_footwear",)),
}

BODY_STYLE_CROP_MULTIPLIERS = {
    "body_torso": 1.0,
    "body_upper_core": 1.0,
    "body_upper_arm_overlap": 0.55,
    "body_context": 0.25,
}
BODY_STYLE_CANDIDATE_CROP_ROUTES = {
    "body_torso": ("torso_core",),
    "body_upper_core": ("torso_core",),
    "body_upper_arm_overlap": ("upper_body",),
    "body_context": ("upper_body",),
}

LONG_BODY_STYLE_CROP_MULTIPLIERS = {
    "body_torso": 0.75,
    "body_upper_core": 0.35,
    "body_upper_arm_overlap": 0.45,
    "body_context": 1.35,
}
LONG_BODY_STYLE_CANDIDATE_CROP_ROUTES = {
    "body_torso": ("upper_body",),
    "body_upper_core": ("upper_body",),
    "body_upper_arm_overlap": ("upper_body",),
    "body_context": ("upper_body",),
}

LEGS_STYLE_CROP_MULTIPLIERS = {
    "legs_waist_thighs": 1.25,
    "legs_thighs": 1.25,
    "legs_left": 0.60,
    "legs_right": 0.60,
    "legs_context": 0.40,
    "legs_boot_overlap": 0.20,
}
LEGS_STYLE_CANDIDATE_CROP_ROUTES = {
    "legs_waist_thighs": ("legs_thighs",),
    "legs_thighs": ("legs_thighs",),
    "legs_left": ("legs_left",),
    "legs_right": ("legs_right",),
    "legs_context": ("full",),
    "legs_boot_overlap": ("legs_boot_overlap",),
}

LONG_BODY_LEGWEAR_QUERY_CROPS = (
    "legs_waist_thighs",
    "legs_thighs",
    "legs_context",
)
LONG_BODY_LEGWEAR_VISIBILITY_CLASSES = (
    "visible",
    "hidden",
)
LONG_BODY_LEGWEAR_VISIBILITY_PROMPTS = (
    (
        "the separate lower-body clothing is clearly visible, such as pants, "
        "shorts, or a skirt, even if framed by an open long coat"
    ),
    (
        "the separate lower-body clothing cannot be seen because a closed long "
        "coat, robe, or dress covers the hips and legs"
    ),
)
LONG_BODY_LEGWEAR_COVERAGE_CLASSES = (
    "exposed_legs",
    "covered_legs",
)
LONG_BODY_LEGWEAR_COVERAGE_PROMPTS = (
    (
        "the center of the image shows exposed natural skin on the thighs or "
        "legs below shorts or a short skirt"
    ),
    (
        "the center of the image shows opaque fabric covering the thighs and "
        "legs as full-length pants, trousers, tights, or leggings"
    ),
)
LONG_BODY_COVERED_LEGWEAR_CLASSES = (
    "plain_trousers",
    "wide_trousers",
    "complex_legwear",
)
LONG_BODY_COVERED_LEGWEAR_QUERY_PROMPTS = (
    (
        "plain simple straight-leg full-length tailored trousers or suit pants "
        "visible between the open panels of a long coat"
    ),
    "wide or baggy trousers, culottes, bloomers, hakama, or heavily draped pants",
    "ornate armored layered or asymmetric complex legwear",
)
LONG_BODY_COVERED_LEGWEAR_CANDIDATE_PROMPTS = (
    (
        "plain simple straight-leg tailored trousers or suit pants with a clean "
        "narrow silhouette"
    ),
    "wide or baggy trousers, culottes, bloomers, hakama, or heavily draped pants",
    "a skirt, layered panels, armored, or highly ornate complex legwear",
)
LONG_BODY_LEGWEAR_MIN_LONG_TOP_PROBABILITY = 0.75
LONG_BODY_LEGWEAR_MIN_VISIBILITY = 0.55
LONG_BODY_LEGWEAR_MIN_EXPOSED_PROBABILITY = 0.55
LONG_BODY_LEGWEAR_MIN_STYLE_CONFIDENCE = 0.45
LONG_BODY_COVERED_LEGWEAR_RERANK_BLEND = 0.55

# Kept here as an optional dependency-free source for future index builders.
STYLE_CANDIDATE_CROP_BOXES_NORMALIZED = {
    "BODY": {
        "upper_body": (0.0, 0.0, 1.0, 0.72),
        "torso_core": (0.06, 0.14, 0.94, 0.62),
    },
    "LEGS": {
        "full": (0.0, 0.0, 1.0, 1.0),
        "legs_left": (0.0, 0.0, 0.55, 1.0),
        "legs_right": (0.45, 0.0, 1.0, 1.0),
        "legs_thighs": (0.04, 0.0, 0.96, 0.58),
        "legs_boot_overlap": (0.04, 0.42, 0.96, 1.0),
    },
}


@dataclass(frozen=True)
class StylePrediction:
    """Validated, model-independent garment-style probabilities."""

    slot: str
    classes: tuple[str, ...]
    probabilities: tuple[float, ...]
    predicted_class: str
    confidence: float

    def __post_init__(self) -> None:
        slot = str(self.slot).upper()
        classes = tuple(str(value) for value in self.classes)
        probabilities = _validated_probability_vector(self.probabilities)
        predicted_class = str(self.predicted_class)
        confidence = float(self.confidence)

        expected_classes = STYLE_CLASSES_BY_SLOT.get(slot)
        if expected_classes is None:
            raise ValueError(f"unsupported garment-style slot: {self.slot}")
        if len(classes) != len(set(classes)):
            raise ValueError("style classes must be unique")
        if set(classes) != set(expected_classes):
            raise ValueError(
                f"style classes for {slot} must be {expected_classes}"
            )
        if len(probabilities) != len(classes):
            raise ValueError("style classes and probabilities must align")

        predicted_index = max(
            range(len(probabilities)),
            key=probabilities.__getitem__,
        )
        expected_prediction = classes[predicted_index]
        expected_confidence = probabilities[predicted_index]
        if predicted_class != expected_prediction:
            raise ValueError(
                "predicted_class must identify the maximum probability"
            )
        if not math.isfinite(confidence) or not math.isclose(
            confidence,
            expected_confidence,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise ValueError("confidence must equal the maximum probability")

        object.__setattr__(self, "slot", slot)
        object.__setattr__(self, "classes", classes)
        object.__setattr__(self, "probabilities", probabilities)
        object.__setattr__(self, "predicted_class", predicted_class)
        object.__setattr__(self, "confidence", expected_confidence)

    @classmethod
    def from_probabilities(
        cls,
        slot: str,
        probabilities: Sequence[float] | Mapping[str, float],
        *,
        classes: Sequence[str] | None = None,
    ) -> StylePrediction:
        """Build a prediction from an aligned sequence or class-value mapping."""

        normalized_slot = str(slot).upper()
        expected_classes = STYLE_CLASSES_BY_SLOT.get(normalized_slot)
        if expected_classes is None:
            raise ValueError(f"unsupported garment-style slot: {slot}")
        class_order = (
            tuple(str(value) for value in classes)
            if classes is not None
            else expected_classes
        )
        if isinstance(probabilities, Mapping):
            if set(probabilities) != set(class_order):
                raise ValueError(
                    "probability mapping must contain every style class exactly once"
                )
            values = tuple(float(probabilities[name]) for name in class_order)
        else:
            values = tuple(float(value) for value in probabilities)
        values = _validated_probability_vector(values)
        if len(values) != len(class_order):
            raise ValueError("style classes and probabilities must align")
        predicted_index = max(range(len(values)), key=values.__getitem__)
        return cls(
            slot=normalized_slot,
            classes=class_order,
            probabilities=values,
            predicted_class=class_order[predicted_index],
            confidence=values[predicted_index],
        )


@dataclass(frozen=True)
class StylePolicy:
    """Routing and crop-weight changes activated by a confident style."""

    active: bool
    crop_multipliers: Mapping[str, float]
    candidate_crop_routes: Mapping[str, tuple[str, ...]]
    retrieval_role_overrides: Mapping[str, str]
    reason: str


@dataclass(frozen=True)
class LongBodyLegwearDecision:
    """Visibility-aware lower-garment routing under a long BODY garment."""

    observable: bool
    visible_probability: float
    mode: str | None
    mode_probability: float
    classes: tuple[str, ...]
    probabilities: tuple[float, ...]
    predicted_class: str | None
    confidence: float
    crop_index: int | None
    reason: str


def style_policy(
    prediction: StylePrediction,
    *,
    minimum_confidence: float = 0.5,
) -> StylePolicy:
    """Return a conservative crop policy for exposed/short garment styles."""

    if not 0.0 <= minimum_confidence <= 1.0:
        raise ValueError("minimum_confidence must be between zero and one")
    # BODY's short and long silhouettes require opposite crop policies. Their
    # combined probability is not evidence for either one (nor is 1-P(regular)
    # a confidence estimate). Unlike overlapping LEGS attributes, require one
    # target class to win confidently before changing retrieval or reranking.
    if prediction.slot == "BODY" and (
        prediction.confidence < minimum_confidence
        or prediction.predicted_class not in TARGET_STYLE_CLASSES["BODY"]
    ):
        return StylePolicy(
            active=False,
            crop_multipliers={},
            candidate_crop_routes={},
            retrieval_role_overrides={},
            reason="body_style_ambiguous_or_regular",
        )
    target_probability = sum(
        probability
        for class_name, probability in zip(
            prediction.classes,
            prediction.probabilities,
            strict=True,
        )
        if class_name in TARGET_STYLE_CLASSES[prediction.slot]
    )
    if target_probability < minimum_confidence:
        return StylePolicy(
            active=False,
            crop_multipliers={},
            candidate_crop_routes={},
            retrieval_role_overrides={},
            reason="target_probability_below_threshold",
        )
    if (
        prediction.slot == "BODY"
        and prediction.probabilities[
            prediction.classes.index("long_top")
        ] >= minimum_confidence
    ):
        multipliers = LONG_BODY_STYLE_CROP_MULTIPLIERS
        routes = LONG_BODY_STYLE_CANDIDATE_CROP_ROUTES
        reason = "high_confidence_long_top"
    elif prediction.slot == "BODY":
        multipliers = BODY_STYLE_CROP_MULTIPLIERS
        routes = BODY_STYLE_CANDIDATE_CROP_ROUTES
        reason = "high_confidence_target_style"
    elif prediction.slot == "FEET":
        # Footwear height is used only as a conservative reranking signal.
        # The promoted identity retriever keeps its validated crop distribution
        # and candidate pool unchanged.
        multipliers = {}
        routes = {}
        reason = "high_confidence_tall_footwear"
    else:
        multipliers = LEGS_STYLE_CROP_MULTIPLIERS
        routes = LEGS_STYLE_CANDIDATE_CROP_ROUTES
        # This waist-to-mid-thigh crop deliberately excludes most boots and
        # lower-body occluders.  It is retrieval-active only after the style
        # gate confirms that a short/exposed lower garment is observable.
        role_overrides = {"legs_waist_thighs": "auxiliary"}
        reason = "high_confidence_target_style"
    if prediction.slot != "LEGS":
        role_overrides = {}
    return StylePolicy(
        active=True,
        crop_multipliers=dict(multipliers),
        candidate_crop_routes=dict(routes),
        retrieval_role_overrides=role_overrides,
        reason=reason,
    )


def apply_style_crop_multipliers(
    records: Sequence[Mapping[str, Any]],
    policy_or_multipliers: StylePolicy | Mapping[str, float],
) -> list[dict[str, Any]]:
    """Copy records and attach the multiplier consumed by ``crop_weight``."""

    if isinstance(policy_or_multipliers, StylePolicy):
        if not policy_or_multipliers.active:
            return [dict(record) for record in records]
        multipliers = policy_or_multipliers.crop_multipliers
        role_overrides = policy_or_multipliers.retrieval_role_overrides
    else:
        multipliers = policy_or_multipliers
        role_overrides = {}

    parsed_multipliers: dict[str, float] = {}
    for name, raw_multiplier in multipliers.items():
        multiplier = float(raw_multiplier)
        if not math.isfinite(multiplier) or multiplier < 0.0:
            raise ValueError("style crop multipliers must be finite and non-negative")
        parsed_multipliers[str(name)] = multiplier

    output: list[dict[str, Any]] = []
    for record in records:
        copied = dict(record)
        copied["style_weight_multiplier"] = parsed_multipliers.get(
            str(record.get("name")),
            1.0,
        )
        role_override = role_overrides.get(str(record.get("name")))
        if role_override is not None:
            copied["role"] = role_override
        output.append(copied)
    return output


def rerank_candidates_with_style(
    candidates: Sequence[Mapping[str, Any]],
    prediction: StylePrediction,
    candidate_probabilities_by_piece: Mapping[int, Sequence[float]],
    *,
    rerank_k: int,
    blend: float,
    minimum_confidence: float = 0.5,
) -> list[dict[str, Any]]:
    """Blend retrieval percentile and normalized style agreement in a safe pool.

    The target-style score is a query-conditioned dot product. Query
    probabilities are squared and normalized over only the exposed/short
    classes, while candidate probabilities retain their absolute mass. Keeping
    the complete target mixture is important because crop length, sleeve
    coverage, and asymmetric thigh cuts can coexist in the same outfit.

    Gating and profile completeness are checked before any diagnostics are
    attached, so every inactive path is a true no-op apart from copying rows.
    """

    if rerank_k < 1:
        raise ValueError("rerank_k must be positive")
    if not 0.0 <= blend <= 1.0:
        raise ValueError("blend must be between zero and one")
    policy = style_policy(
        prediction,
        minimum_confidence=minimum_confidence,
    )
    output = [dict(row) for row in candidates]
    if not policy.active:
        return output

    pool_size = min(rerank_k, len(output))
    if pool_size < 2:
        return output
    leading = output[:pool_size]

    # Missing profiles disable the entire reranker rather than partially
    # penalizing a candidate whose profile has not been generated yet.
    profiles: list[tuple[float, ...]] = []
    for row in leading:
        piece_id = int(row["piece_id"])
        raw_profile = candidate_probabilities_by_piece.get(piece_id)
        if raw_profile is None:
            return output
        profile = _validated_probability_vector(raw_profile)
        if len(profile) != len(prediction.classes):
            raise ValueError(
                f"candidate style profile for piece {piece_id} has "
                f"{len(profile)} classes; expected {len(prediction.classes)}"
            )
        profiles.append(profile)

    target_indices = tuple(
        index
        for index, class_name in enumerate(prediction.classes)
        if class_name in TARGET_STYLE_CLASSES[prediction.slot]
    )
    target_query_mass = sum(
        prediction.probabilities[index] ** 2 for index in target_indices
    )
    if target_query_mass <= 0.0:
        return output
    normalized_query = tuple(
        prediction.probabilities[index] ** 2 / target_query_mass
        for index in target_indices
    )
    style_scores = [
        sum(
            query_probability * profile[index]
            for query_probability, index in zip(
                normalized_query,
                target_indices,
                strict=True,
            )
        )
        for profile in profiles
    ]
    maximum_style_score = max(style_scores)
    if maximum_style_score <= 1e-8:
        return output
    normalized_style_scores = [
        score / maximum_style_score for score in style_scores
    ]
    retrieval_scores = [float(row["score"]) for row in leading]
    if any(not math.isfinite(score) for score in retrieval_scores):
        raise ValueError("candidate retrieval scores must be finite")
    retrieval_percentiles = _percentiles(retrieval_scores)
    style_percentiles = _percentiles(style_scores)

    original_positions = {id(row): index for index, row in enumerate(leading)}
    for (
        row,
        retrieval_score,
        style_score,
        normalized_style_score,
        retrieval_percentile,
        style_percentile,
    ) in zip(
        leading,
        retrieval_scores,
        style_scores,
        normalized_style_scores,
        retrieval_percentiles,
        style_percentiles,
        strict=True,
    ):
        row["retrieval_score"] = retrieval_score
        row["style_score"] = style_score
        row["style_normalized_score"] = normalized_style_score
        row["retrieval_percentile"] = retrieval_percentile
        row["style_percentile"] = style_percentile
        row["score"] = (
            (1.0 - blend) * retrieval_percentile
            + blend * normalized_style_score
        )

    leading.sort(
        key=lambda row: (
            -float(row["score"]),
            -float(row["retrieval_score"]),
            original_positions[id(row)],
        )
    )
    output[:pool_size] = leading
    for rank, row in enumerate(output, start=1):
        row["rank"] = rank
    return output


def long_top_probability(
    style_diagnostic: Mapping[str, Any] | None,
) -> float:
    if not style_diagnostic:
        return 0.0
    probabilities = style_diagnostic.get("probabilities")
    if not isinstance(probabilities, Mapping):
        return 0.0
    try:
        probability = float(probabilities.get("long_top", 0.0))
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(probability):
        return 0.0
    return min(1.0, max(0.0, probability))


def long_body_legwear_decision(
    visibility_probabilities: Sequence[Sequence[float]],
    coverage_probabilities: Sequence[Sequence[float]],
    covered_style_probabilities: Sequence[Sequence[float]],
    *,
    minimum_visibility: float = LONG_BODY_LEGWEAR_MIN_VISIBILITY,
    minimum_exposed_probability: float = (
        LONG_BODY_LEGWEAR_MIN_EXPOSED_PROBABILITY
    ),
) -> LongBodyLegwearDecision:
    """Choose an observable lower-garment profile without assuming trousers.

    Several leg crops are evaluated because long coat panels can hide one crop
    while leaving another useful.  A hidden result carries no style profile;
    callers must keep retrieval unchanged rather than inventing an unseen item.
    When the lower garment is visible, exposed-leg styles and covered-leg
    styles use separate prompt spaces so shorts are not compared against a
    fixed trouser prior.
    """

    if not 0.0 <= minimum_visibility <= 1.0:
        raise ValueError("minimum_visibility must be between zero and one")
    if not 0.0 <= minimum_exposed_probability <= 1.0:
        raise ValueError(
            "minimum_exposed_probability must be between zero and one"
        )
    profile_groups = (
        visibility_probabilities,
        coverage_probabilities,
        covered_style_probabilities,
    )
    profile_count = len(visibility_probabilities)
    if profile_count < 1 or any(
        len(group) != profile_count for group in profile_groups
    ):
        raise ValueError(
            "long-body legwear profile groups must align and be non-empty"
        )

    visibility_profiles = tuple(
        _validated_probability_vector(profile)
        for profile in visibility_probabilities
    )
    coverage_profiles = tuple(
        _validated_probability_vector(profile)
        for profile in coverage_probabilities
    )
    covered_profiles = tuple(
        _validated_probability_vector(profile)
        for profile in covered_style_probabilities
    )
    if any(
        len(profile) != len(LONG_BODY_LEGWEAR_VISIBILITY_CLASSES)
        for profile in visibility_profiles
    ):
        raise ValueError("visibility profiles have the wrong class count")
    if any(
        len(profile) != len(LONG_BODY_LEGWEAR_COVERAGE_CLASSES)
        for profile in coverage_profiles
    ):
        raise ValueError("coverage profiles have the wrong class count")
    if any(
        len(profile) != len(LONG_BODY_COVERED_LEGWEAR_CLASSES)
        for profile in covered_profiles
    ):
        raise ValueError("covered-leg style profiles have the wrong class count")

    visible_index = LONG_BODY_LEGWEAR_VISIBILITY_CLASSES.index("visible")
    visible_scores = tuple(
        profile[visible_index] for profile in visibility_profiles
    )
    visible_probability = max(visible_scores)
    if visible_probability < minimum_visibility:
        return LongBodyLegwearDecision(
            observable=False,
            visible_probability=visible_probability,
            mode=None,
            mode_probability=0.0,
            classes=(),
            probabilities=(),
            predicted_class=None,
            confidence=0.0,
            crop_index=None,
            reason="lower_garment_not_observable",
        )

    exposed_index = LONG_BODY_LEGWEAR_COVERAGE_CLASSES.index("exposed_legs")
    exposed_scores = tuple(
        profile[exposed_index] for profile in coverage_profiles
    )
    maximum_exposed_probability = max(exposed_scores)
    if maximum_exposed_probability >= minimum_exposed_probability:
        mode = "exposed"
        crop_index = max(
            range(profile_count),
            key=exposed_scores.__getitem__,
        )
        mode_probability = maximum_exposed_probability
        classes = LONG_BODY_LEGWEAR_COVERAGE_CLASSES
        probabilities = coverage_profiles[crop_index]
        reason = "visible_exposed_legwear"
    else:
        covered_index = LONG_BODY_LEGWEAR_COVERAGE_CLASSES.index("covered_legs")
        covered_scores = tuple(
            profile[covered_index] for profile in coverage_profiles
        )
        mode = "covered"
        crop_index = max(
            range(profile_count),
            key=covered_scores.__getitem__,
        )
        mode_probability = covered_scores[crop_index]
        classes = LONG_BODY_COVERED_LEGWEAR_CLASSES
        probabilities = covered_profiles[crop_index]
        reason = "visible_covered_legwear"

    predicted_index = max(
        range(len(probabilities)),
        key=probabilities.__getitem__,
    )
    return LongBodyLegwearDecision(
        observable=True,
        visible_probability=visible_probability,
        mode=mode,
        mode_probability=mode_probability,
        classes=classes,
        probabilities=probabilities,
        predicted_class=classes[predicted_index],
        confidence=probabilities[predicted_index],
        crop_index=crop_index,
        reason=reason,
    )


def rerank_candidates_with_legwear_profile(
    candidates: Sequence[Mapping[str, Any]],
    query_probabilities: Sequence[float],
    candidate_probabilities_by_piece: Mapping[int, Sequence[float]],
    *,
    rerank_k: int,
    blend: float,
    minimum_confidence: float = LONG_BODY_LEGWEAR_MIN_STYLE_CONFIDENCE,
) -> list[dict[str, Any]]:
    """Rerank visible legwear by query/candidate style agreement."""

    if rerank_k < 1:
        raise ValueError("rerank_k must be positive")
    if not 0.0 <= blend <= 1.0:
        raise ValueError("blend must be between zero and one")
    if not 0.0 <= minimum_confidence <= 1.0:
        raise ValueError("minimum_confidence must be between zero and one")

    output = [dict(row) for row in candidates]
    query_profile = _validated_probability_vector(query_probabilities)
    if max(query_profile) < minimum_confidence:
        return output
    pool_size = min(rerank_k, len(output))
    if pool_size < 2:
        return output
    leading = output[:pool_size]
    profiles: list[tuple[float, ...]] = []
    for row in leading:
        piece_id = int(row["piece_id"])
        raw_profile = candidate_probabilities_by_piece.get(piece_id)
        if raw_profile is None:
            return output
        profile = _validated_probability_vector(raw_profile)
        if len(profile) != len(query_profile):
            raise ValueError(
                f"candidate legwear style profile for piece {piece_id} has "
                f"{len(profile)} classes; expected {len(query_profile)}"
            )
        profiles.append(profile)

    query_mass = sum(value**2 for value in query_profile)
    normalized_query = tuple(value**2 / query_mass for value in query_profile)
    style_scores = [
        sum(
            query_value * candidate_value
            for query_value, candidate_value in zip(
                normalized_query,
                profile,
                strict=True,
            )
        )
        for profile in profiles
    ]
    maximum_style_score = max(style_scores)
    if maximum_style_score <= 1e-8:
        return output
    normalized_style_scores = [
        score / maximum_style_score for score in style_scores
    ]
    retrieval_scores = [float(row["score"]) for row in leading]
    if any(not math.isfinite(score) for score in retrieval_scores):
        raise ValueError("candidate retrieval scores must be finite")
    retrieval_percentiles = _percentiles(retrieval_scores)

    original_positions = {id(row): index for index, row in enumerate(leading)}
    for (
        row,
        retrieval_score,
        retrieval_percentile,
        style_score,
        normalized_style_score,
    ) in zip(
        leading,
        retrieval_scores,
        retrieval_percentiles,
        style_scores,
        normalized_style_scores,
        strict=True,
    ):
        row["retrieval_score"] = retrieval_score
        row["retrieval_percentile"] = retrieval_percentile
        row["legwear_style_score"] = style_score
        row["legwear_style_normalized_score"] = normalized_style_score
        row["score"] = (
            (1.0 - blend) * retrieval_percentile
            + blend * normalized_style_score
        )

    leading.sort(
        key=lambda row: (
            -float(row["score"]),
            -float(row["retrieval_score"]),
            original_positions[id(row)],
        )
    )
    output[:pool_size] = leading
    for rank, row in enumerate(output, start=1):
        row["rank"] = rank
    return output


def _validated_probability_vector(
    probabilities: Sequence[float],
) -> tuple[float, ...]:
    values = tuple(float(value) for value in probabilities)
    if not values:
        raise ValueError("style probabilities cannot be empty")
    if any(
        not math.isfinite(value) or value < 0.0 or value > 1.0
        for value in values
    ):
        raise ValueError("style probabilities must be finite and between zero and one")
    total = sum(values)
    if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-5):
        raise ValueError("style probabilities must sum to one")
    # Eliminate harmless float32/serialized rounding drift while preserving the
    # probability semantics required by callers.
    return tuple(value / total for value in values)


def _percentiles(values: Sequence[float]) -> list[float]:
    if len(values) == 1:
        return [1.0]
    ordered = sorted(range(len(values)), key=values.__getitem__)
    output = [0.0] * len(values)
    denominator = len(values) - 1
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while (
            end < len(ordered)
            and values[ordered[end]] == values[ordered[cursor]]
        ):
            end += 1
        average_rank = (cursor + end - 1) / 2.0
        percentile = average_rank / denominator
        for ordered_index in ordered[cursor:end]:
            output[ordered_index] = percentile
        cursor = end
    return output
