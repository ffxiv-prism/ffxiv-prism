# v44 male item-page expansion — 2026-07-03

## Outcome

v44 opens a new Eorzea Collection discovery path after male related-link
expansion became nearly exhausted:

| Data | Groups | Images | Reviewed images | Train BODY cases |
|---|---:|---:|---:|---:|
| v43 | 1,463 | 5,875 | 970 | 853 |
| v44 | 1,524 | 6,097 | 1,024 | 907 |

The new data improves paired end-to-end strict CV, but the full-data candidate
does not beat the existing mixed/seen deployment model. qv52 plus reranker v35
therefore remains the deployment default. Blind predictions remain uninspected.

## Item-page discovery

`build_glamour_item_queue.py` now supports:

- one or more `--exclude-file` arguments, preventing retries of previously
  attempted and rejected glamour IDs;
- `--shard-size`, producing bounded numbered collection batches.

The implementation is covered by queue exclusion and shard-writing tests. The
full suite passes 113 tests.

Twenty-four BODY item pages were selected from reviewed male examples, with
emphasis on male-specific or male-heavy clothing. One bounded listing-page
pass discovered 859 glamour IDs. After excluding existing records, previous
attempts, and non-train stable assignments, 538 candidates remained. A
round-robin per-item cap selected 240 train-only pages in twelve 20-page
shards.

Target pages were still required to parse as male and resolve every non-empty
core slot to an exact local item. This produced 61 exact male glamour groups
and 222 images. Pose filtering yielded 73 candidates; independent visual review
rejected 19 back, side, action, distant, effect-obscured, or weapon-obscured
views. The final increment is 54 reviewed BODY images.

New accepted images by race:

| Race | Images |
|---|---:|
| Miqo'te | 16 |
| Viera | 12 |
| Lalafell | 11 |
| Hyur | 7 |
| Au Ra | 4 |
| Elezen | 2 |
| Highlander | 1 |
| Roegadyn | 1 |

All 54 are male. Validation, blind, development, and confirmation assignments
remain unchanged. The normalized dataset has no missing images.

## Paired strict evaluation

The fixed protocol remains unchanged: fresh base-DINOv2 query encoder for ten
epochs, seven-epoch reranker, 1,000 hard negatives, item-disjoint train-only
folds, v3 multiscale candidates, and `balanced / none / max`.

v43 and v44 were retrained on the same v44 folds and evaluated on the same 906
cases:

| Fold | v43 Top-1 / Top-10 | v44 Top-1 / Top-10 |
|---|---:|---:|
| 1, 302 cases | 122 (40.4%) / 229 (75.8%) | 127 (42.1%) / 234 (77.5%) |
| 2, 302 cases | 127 (42.1%) / 222 (73.5%) | 128 (42.4%) / 226 (74.8%) |
| 3, 302 cases | 107 (35.4%) / 213 (70.5%) | 110 (36.4%) / 216 (71.5%) |
| Pooled, 906 cases | 356 (39.3%) / 664 (73.3%) | 365 (40.3%) / 676 (74.6%) |

All folds improve. The pooled gain is nine Top-1 cases (+1.0 percentage point)
and twelve Top-10 cases (+1.3 points). Top-1 paired discordance is 38 v44-only
wins versus 29 v43-only wins.

The 852 pre-v44 cases improve by six Top-1 and eleven Top-10 cases. The 54 new
cases improve by three Top-1 and one Top-10 case, so the result is not caused
only by adding easy cases.

Male cases improve by two Top-1 and five Top-10; female cases improve by seven
Top-1 and seven Top-10. This suggests the male expansion adds useful
cross-domain variation rather than merely fitting male examples. Race-level
effects remain mixed, with Hyur and Miqo'te Top-1 slightly lower.

## Full-data candidate

The formal full-data counterpart is:

- query encoder:
  `data/models/glamour_query_encoder_v61_hq_v44_strict_e10_body.pt`;
- reranker:
  `data/models/glamour_pair_reranker_v37_hq_v44_fixed7_h1000_body_qv61.pt`.

Its one-shot fixed public-validation results are:

| Candidate index | Top-1 | Top-10 |
|---|---:|---:|
| v2 aligned | 41/81 (50.6%) | 67/81 (82.7%) |
| v3 multiscale | 43/81 (53.1%) | 69/81 (85.2%) |

These remain below v35/v2 at 55/81 (67.9%) Top-1 and 72/81 (88.9%) Top-10.
v37 is therefore retained as a strict-protocol research checkpoint and is not
promoted for deployment.

## Next action

Continue item-page discovery with a second non-overlapping set of male-heavy
BODY items, using the new exclusion mechanism. Keep model selection entirely
inside paired strict folds. The larger strategic gap is now clear: clean data
expansion steadily raises honest unknown-item performance, but a model trained
from base inside strict folds has not recovered the globally trained qv52
deployment performance. Representation and candidate-recall work remains
necessary before the 80% target is plausible.
