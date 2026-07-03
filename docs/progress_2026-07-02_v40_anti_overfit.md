# v40 anti-overfitting checkpoint — 2026-07-02

## Frozen data state

- 1,055 Eorzea Collection glamour groups and 4,290 local images.
- `front_fullbody_quality_final_v40.jsonl`: 696 accepted images.
- High-quality train BODY: 579 cases after the corrected visual review.
- Male train BODY coverage increased from 134 to 190 without adding new
  official-validation or blind groups.
- Blind predictions remain uninspected.

## Public validation (image-disjoint, not item-disjoint)

- Reranker v32, fixed seven epochs: Top-1 53/81 (65.4%), Top-10 73/81
  (90.1%).
- Exemplar v5: Top-1 47/81 (58.0%), Top-10 53/81 (65.4%).
- The previously selected margin router at scale 0.3 reaches 62/81 (76.5%),
  but this gain does not reproduce in train-only cross-validation and must not
  be treated as an unbiased generalization estimate.
- A train-only-selected high-confidence exemplar rule
  (`neighbor_vote_fraction_5 >= 0.8`) is stable but only ties canonical v32 at
  53/81 on public validation.

## Train-only grouped cross-validation

Three folds contain only official-train glamour groups. Official validation,
development, confirmation, and blind groups are excluded.

- Standard sampling, seven epochs: pooled Top-1 64.5%.
- Race/gender-balanced sampling: pooled Top-1 64.5%, lower Top-10; rejected.
- Five / seven / ten epochs: pooled Top-1 64.4% / 64.5% / 64.7%.
  Ten epochs loses about two Top-10 points, so seven remains the fixed choice.
- A three-feature router improves the pooled fitting score but fails
  leave-one-fold-out evaluation; rejected as overfit.
- A single vote threshold is stable in all folds and gives 65.2%, a small
  cross-fitted gain over 64.5%.

## Train-only BODY-item-disjoint cross-validation

All glamour groups sharing a BODY piece identity remain in one fold.

- Reranker qv51 Top-1 by fold: 60.3%, 53.6%, 55.7%.
- Pooled Top-1: 328/578 (56.7%).
- Mean Top-10: 77.1%.
- This is the primary estimate for unseen-equipment generalization.

Raw retrieval diagnostics show that the reranker has positive Top-1 value, but
candidate recall is the main bottleneck. Existing qv52 beats qv51 raw retrieval
in all three item-disjoint folds:

| Fold | qv51 Top-1 / Top-10 | qv52 Top-1 / Top-10 |
|---|---:|---:|
| 1 | 49.8% / 77.2% | 51.1% / 78.1% |
| 2 | 34.4% / 73.4% | 37.5% / 74.5% |
| 3 | 44.9% / 74.3% | 47.3% / 77.2% |

Continuing qv51 for five epochs on v40 failed the Fold-1 sentinel and was
stopped without public-validation evaluation.

## qv52 reranker progression

The qv52-compatible reranker improved item-disjoint Top-1 in all three folds:

- qv51 / hard-256: 328/578 (56.7%), mean Top-10 77.1%.
- qv52 / hard-256: 343/578 (59.3%), mean Top-10 78.9%.
- qv52 / hard-512: 347/578 (60.0%), mean Top-10 80.9%.
- qv52 / hard-1000: 350/578 (60.6%), mean Top-10 81.9%.

The hard-negative sweep stopped at 1,000 because that is the retrieval
candidate limit. Dropout 0.2 lost one pooled Top-1 case and was rejected.
Canonical qv51/qv52 confidence routing did not beat qv52 alone in cross-fitted
evaluation.

The promoted formal checkpoint is
`data/models/glamour_pair_reranker_v35_hq_v40_fixed7_h1000_body_qv52.pt`.
After freezing the crop profile from train-only CV, it reaches 55/81 (67.9%)
Top-1 and 72/81 (88.9%) Top-10 on public validation. The earlier 56/81 figure
included one case of optimism from selecting the crop profile on that same
public split. The fixed high-confidence exemplar route does not improve it.

## Multiscale clean-render candidates

The v3 index adds a tight `torso_core` crop alongside the historical
`upper_body` crop for each clean BODY render. It contains 31,081 aligned
candidates, compared with 27,057 in v2.

Using frozen v35 weights rather than retraining on v3 is best:

- Train-only BODY-item-disjoint cross-fitted Top-1: 361/578 (62.5%).
- Top-10: 495/578 (85.6%).
- MRR: 70.0%.
- All three leave-one-fold-out selections independently choose
  `balanced / no tag penalty / max`.

Retraining against v3 reaches only 61.1% cross-fitted Top-1. Candidate-level
margin routing, torso score gating, and top-two view averaging all fail to beat
the direct max-aggregation v3 pipeline.

On public validation, v3 reaches 53/81 (65.4%) Top-1 and 72/81 (88.9%) Top-10,
so v2 remains the mixed/seen-equipment default while v3 is the unseen-equipment
mode.

## Next action

Freeze v35 as the canonical weights. Improve unseen-item Top-1 without reusing
public validation for selection, focusing on candidate/query representation
rather than more routing. Keep item-disjoint three-fold CV and the fixed
`balanced / none / max` crop profile as the selection protocol. Do not inspect
blind predictions.
