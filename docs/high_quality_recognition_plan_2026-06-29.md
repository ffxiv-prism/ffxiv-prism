# High-Quality Recognition Plan — 2026-06-29

## Update — 2026-06-30

- Dataset: 656 glamour groups and 2,673 images, with no missing local images.
- Strict front/full-body manifest v24: 428 accepted images.
- Narrow Hyur-female BODY scope: 109 images (83 train, 17 validation, plus
  7 images consumed in the latest one-time blind evaluation).
- A fresh blind pool is being rebuilt and currently has only 1 accepted group
  / 2 images; its predictions have not been inspected.
- Canonical appearance-equivalence labels now use direct DINOv2 similarity:
  at least 0.94 within one canonical family and 0.975 across families.
- Best development stack: slot-aligned query encoder v42 plus pair reranker
  v15. Validation appearance Top-1 was 12/15 (80%).
- One-time sealed BODY result was 3/7 (42.9%) and Top-10 was 4/7 (57.1%).
  The 80% product target is therefore not met.
- The consumed blind groups have moved to `confirmation_test`. New collection
  must create a fresh blind pool before any further confirmation attempt.

## Goal

The product target is appearance Top-1 near 80% for screenshots that satisfy a
documented input-quality contract. The target is not considered met until it is
confirmed on newly collected, sealed glamour groups whose quality decisions
were frozen before any model predictions were inspected.

The primary metric is appearance Top-1 across eligible HEAD, BODY, HANDS, LEGS,
and FEET cases. Every report must also include Recall@10, MRR, per-slot metrics,
eligible case counts, and rejected-image counts. Exact item ID and
same-appearance retrieval must remain separately labeled.

Initial acceptance target:

- at least 50 sealed high-quality glamour groups;
- at least 250 eligible slot cases;
- overall appearance Top-1 at or above 75%, with 80% as the target;
- no core slot below 60% Top-1;
- quality review completed before retrieval output is generated.

## Input-quality contract

Accepted screenshots must be single-person, front or near-front, full-body
images with a large character subject and no collage or major occlusion. HEAD,
BODY, HANDS, LEGS, and FEET visibility should additionally be reviewed per
slot. An image-level pass does not imply that every small slot is identifiable.

Exact equipment labels come only from source page structure. Vision models may
review framing and slot visibility but must not invent exact item labels.

## Current development set

`front_fullbody_quality_final_v1` contains:

| Split | Images | Glamour groups | Slot cases |
|---|---:|---:|---:|
| train | 229 | 170 | 1,145 |
| validation | 39 | 30 | 195 |
| development_test | 5 | 4 | 25 |

This frozen whitelist covers the first 1,882 source images. It is a development
asset, not a new blind set. The remaining 287 images require a separately
versioned full visual review before they may be included.

## Experiment order

1. Train the v17-compatible v10 pair reranker on the full development data.
2. Fine-tune a v17-derived query encoder using only the frozen high-quality
   train images, selecting epochs only on high-quality validation.
3. Train a matching high-quality pair reranker from that query encoder.
4. Report raw-query, reranked, and per-slot results on identical validation
   cases.
5. Add slot-level visibility filtering, prioritizing HANDS and FEET.
6. Evaluate SAM2 only as an auxiliary HANDS branch and a HEAD fusion branch.
7. Freeze the selected architecture and routing policy.
8. Collect and quality-review a new sealed blind set, then evaluate it once.

## Reproducibility

Both training entry points support:

```text
--quality-manifest data/processed/front_fullbody_quality_final_v1.jsonl
```

Only records with `passed=true` are loaded. The existing
`glamour_split_v2.json` continues to enforce group-level isolation. Generated
weights, embeddings, masks, and detailed reports remain local artifacts.

## Guardrails

- Do not reuse consumed blind v5 for model selection.
- Do not choose different checkpoints on the blind set for individual slots.
- Do not claim improvement when compared methods cover different cases.
- Do not replace raw crops globally with strict parser or SAM2 masks.
- Do not treat a high validation score as the final 80% result.

## Progress update - 2026-06-29 evening

`front_fullbody_quality_final_v2` extends the original reviewed development
manifest without trusting pose-only automation wholesale:

| Manifest | Source images | Accepted images | Notes |
|---|---:|---:|---|
| `front_fullbody_quality_final_v1` | 1,882 | 273 | Original frozen development whitelist. |
| `front_fullbody_quality_pose_v2_2169` | 2,169 | 708 | Pose-only candidate pool; not safe to use directly. |
| `front_fullbody_quality_final_v2` | 2,169 | 325 | v1 plus 52 manually accepted new images. |
| `front_fullbody_quality_final_v3` | 2,189 | 331 | v2 plus 6 manually accepted images from 5 newly collected glamour groups. |

The current CLIP prompt-based view gate is not reproducible enough for
expansion: on the 2,169-image run it rejected all candidates, including known
good v1 examples, as non-front/cropped. Until that gate is fixed, v2 uses pose
filters plus visual review instead.

Latest development experiments:

| Run | Scope | Quality manifest | Validation Top-1 | Test Top-1 | Conclusion |
|---|---|---|---:|---:|---|
| v10 pair reranker | all slots | v1 high-quality report | 31.2% | n/a | Did not beat the v17 query baseline. |
| v18 query encoder | all slots | v1 | 31.2% | n/a | High-quality fine-tuning alone did not improve. |
| v21 query encoder | all slots | v2 | 30.9% | 12.0% | New images did not create a general breakthrough. |
| v22 query encoder | BODY only | v2 | 72.7% | 0.0% | BODY remains the most promising slot, but the test slice is too small and shifted. |
| v23 query encoder | BODY only | v3 | 72.7% | 0.0% | The first 5-page expansion did not change the tiny BODY validation/test slice. |
| confidence router v1 | BODY only | v1 validation | 79.5% | 0.0% | Useful validation signal, not a sealed-set result. |

The work has therefore moved from "keep tuning the same data" to "expand and
seal the high-quality evaluation target." The next useful milestone is a new
quality-reviewed sealed set of at least 50 glamour groups / 250 eligible slot
cases, with BODY allowed as the first narrow production target if all-slot
accuracy remains blocked by HANDS/HEAD/FEET visibility.

Coverage reports are now generated with:

```powershell
python scripts\summarize_quality_split_coverage.py `
  --quality-manifest data\processed\front_fullbody_quality_final_v3.jsonl
```

Current v3 development-test coverage is still only 4 glamour groups / 25
locally evaluable slot cases. BODY-only coverage is 4 groups / 5 cases. The
latest expansion queue is `data/interim/glamour_expansion_queue_20260630.txt`;
the first 5 candidates collected successfully and produced 20 new images, but
only 6 passed the strict front/full-body visual review.
