# v42 strict item-disjoint checkpoint — 2026-07-03

## Decision summary

The project now reports two separate tracks instead of blending their numbers:

- Mixed/seen-equipment deployment: keep qv52 plus reranker v35 on the v2
  candidate index. Its frozen public-validation score is 55/81 (67.9%) Top-1
  and 72/81 (88.9%) Top-10.
- Unknown-item research: use end-to-end item-disjoint cross-validation with
  a fresh query encoder and reranker inside every fold. The v42 result on the
  v3 multiscale index is 307/821 (37.4%) Top-1 and 574/821 (69.9%) Top-10.

The 80% Top-1 product target is not met. Blind predictions remain uninspected.

## Why the previous 62.5% is not the primary estimate

The earlier v35/v3 item-disjoint result was 361/578 (62.5%) Top-1 and 495/578
(85.6%) Top-10. Its reranker folds were item-disjoint, but qv52 had already
been trained on the complete official-train domain, including the BODY
identities later held out from individual folds.

That result is still useful for measuring the reranker's behavior with a
strong globally trained encoder. It is not an unbiased estimate of the entire
pipeline on an unseen equipment identity. The strict protocol now initializes
the query encoder from base DINOv2 separately in every fold before training
the fold-specific reranker.

## Strict encoder calibration

Fold 1 was used as a train-only sentinel to choose a fixed encoder duration:

| Encoder | Top-1 | Top-10 | Decision |
|---|---:|---:|---|
| Base DINOv2, 0 epochs | 19.2% | 40.4% | Lower bound |
| Last-block tuning, 5 epochs | 31.5% | 60.6% | Improving |
| Last-block tuning, 10 epochs | 35.2% | 67.1% | Freeze duration |
| Last-block tuning, 30 epochs | 30.5% | 64.3% | Rejected: overfit |
| Rank-16 residual adapter, 10 epochs | 23.0% | 52.6% | Rejected |

All formal strict folds therefore use ten encoder epochs and seven reranker
epochs. The public validation split did not choose these settings.

## v42 data expansion

The broad v6 collection queue contained 400 train-only source pages. Four
collection shards produced 287 pages with exact equipment mappings. The
resulting corpus contains:

- 1,423 glamour groups and 5,734 local images, with no missing files;
- 939 visually reviewed front/full-body images;
- 822 official-train BODY cases, up from 635 in v41;
- unchanged validation, development, confirmation, and blind groups.

The 187 newly accepted BODY images cover 142 glamour groups and 118 BODY
identities. Their race distribution is Au Ra 51, Elezen 8, Hrothgar 11, Hyur
25, Highlander 4, Lalafell 12, Miqo'te 58, Roegadyn 2, and Viera 16.

Gender coverage is still a material weakness: 182 of the 187 new images are
female and only 5 are male. The next collection round must explicitly select
male source pages rather than relying on a broad queue to balance gender.

## Paired end-to-end strict result

v41 and v42 were trained and evaluated on exactly the same v42 item-disjoint
folds and evaluation cases:

| Fold | v41 Top-1 / Top-10 | v42 Top-1 / Top-10 |
|---|---:|---:|
| 1, 272 cases | 95 (34.9%) / 193 (71.0%) | 110 (40.4%) / 197 (72.4%) |
| 2, 274 cases | 103 (37.6%) / 180 (65.7%) | 102 (37.2%) / 185 (67.5%) |
| 3, 275 cases | 89 (32.4%) / 190 (69.1%) | 95 (34.5%) / 192 (69.8%) |
| Pooled, 821 cases | 287 (35.0%) / 563 (68.6%) | 307 (37.4%) / 574 (69.9%) |

The exact paired gain is 20 Top-1 cases (+2.4 percentage points) and 11 Top-10
cases (+1.3 points). Improvement also appears on the 634 pre-v42 cases:
33.1% to 36.4% Top-1, so the gain is not explained only by evaluating the
newly added images.

Male Top-1 moves from 78/214 (36.4%) to 80/214 (37.4%); female Top-1 moves
from 209/607 (34.4%) to 227/607 (37.4%). The current evidence supports further
high-quality data expansion, but the small male gain and female-heavy batch
justify male-targeted collection next.

## Implementation corrections

`group_training_samples` previously iterated only over grouped values while
using a stale `group_id` from an earlier loop. Consequently every group
received the weight associated with that final stale ID, so prior
domain-balancing and replay-weight experiments did not apply their intended
per-group weights. The loop now carries both `group_id` and rows. Tests cover
group weighting and the restored reranker initialization behavior.

The item-disjoint splitter also accepts a quality manifest and balances folds
by eligible reviewed train images while keeping each BODY identity wholly
inside one fold.

## Full-data candidate

The strict-protocol full-data candidate is:

- query encoder:
  `data/models/glamour_query_encoder_v58_hq_v42_strict_e10_body.pt`;
- reranker:
  `data/models/glamour_pair_reranker_v36_hq_v42_fixed7_h1000_body_qv58.pt`.

Its one-shot frozen public-validation results are:

| Candidate index | Top-1 | Top-10 |
|---|---:|---:|
| v2 aligned | 41/81 (50.6%) | 65/81 (80.2%) |
| v3 multiscale | 44/81 (54.3%) | 65/81 (80.2%) |

Because these are below v35/v2, v36 is not promoted as the deployment default.
It is retained as the full-data counterpart of the honest unseen-item research
protocol.

## Next iteration

1. Collect a train-only, explicitly male source-page batch and apply the same
   pose plus manual visual-quality filters.
2. Repeat paired strict folds before considering any public-validation run.
3. Prefer data and candidate-recall improvements over extra router or
   hyperparameter sweeps; the 30-epoch sentinel already shows the current
   encoder can overfit.
4. Keep validation configuration fixed at `balanced / no tag penalty / max`.
5. Do not inspect blind predictions until a predeclared candidate is ready for
   final confirmation.
