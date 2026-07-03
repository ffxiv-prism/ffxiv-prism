# v43 male-targeted expansion — 2026-07-03

## Outcome

v43 adds a small, clean, male-only training increment and produces a small
paired strict-CV improvement:

| Data | Groups | Images | Reviewed images | Train BODY cases |
|---|---:|---:|---:|---:|
| v42 | 1,423 | 5,734 | 939 | 822 |
| v43 | 1,463 | 5,875 | 970 | 853 |

The deployment default remains qv52 plus reranker v35 on the v2 index. No
v43 public-validation or blind evaluation was run.

## Collection and quality gates

The queue followed related links only from known male source glamours,
restricted all stable split assignments to `train`, and excluded the broad v5
and v6 queues plus all earlier male queues. After those exclusions, only 86
unattempted links remained, indicating that this particular expansion source
is close to exhausted.

Every target page was required to:

- parse as a male model on the target page itself;
- resolve every non-empty core equipment slot to an exact local item;
- pass the existing pose geometry filter;
- pass a prediction-independent manual front/full-body review.

The 86 attempted pages yielded 40 exact male glamour groups and 141 local
images. Pose filtering produced 42 new candidates; manual review rejected 11
back, side, crouched, distant, effect-obscured, or weapon-obscured views. The
final v43 increment is 31 reviewed BODY images.

New accepted images by race:

| Race | Images |
|---|---:|
| Hyur | 9 |
| Au Ra | 7 |
| Miqo'te | 7 |
| Highlander | 2 |
| Roegadyn | 2 |
| Viera | 2 |
| Elezen | 1 |
| Lalafell | 1 |

All 31 are male. Validation, blind, development, and confirmation assignments
remain unchanged; the normalized dataset has no missing image files.

## Fixed strict protocol

The v43 comparison keeps the anti-overfitting settings frozen:

- three train-only folds, with every BODY identity wholly inside one fold;
- fold balancing by reviewed eligible image count;
- query encoder trained from base DINOv2 for exactly 10 epochs;
- reranker trained for exactly 7 epochs with 1,000 hard negatives;
- v3 multiscale candidate index;
- `balanced / no tag penalty / max`;
- no early stopping, crop-profile search, public-validation selection, or
  blind inspection.

To isolate the data change, the v42 and v43 pipelines were retrained on the
same v43 fold manifests and both evaluated on the same 852 v43 cases.

## Paired strict result

| Fold | v42 Top-1 / Top-10 | v43 Top-1 / Top-10 |
|---|---:|---:|
| 1, 283 cases | 109 (38.5%) / 195 (68.9%) | 109 (38.5%) / 198 (70.0%) |
| 2, 285 cases | 111 (38.9%) / 206 (72.3%) | 123 (43.2%) / 204 (71.6%) |
| 3, 284 cases | 117 (41.2%) / 209 (73.6%) | 111 (39.1%) / 214 (75.4%) |
| Pooled, 852 cases | 337 (39.6%) / 610 (71.6%) | 343 (40.3%) / 616 (72.3%) |

The net gain is six Top-1 and six Top-10 cases, or +0.7 percentage points for
each metric. Top-1 paired discordance is 22 v43-only wins versus 16 v42-only
wins. The direction is positive but the effect is too small and fold-variable
to justify a promotion claim.

The 31 new cases themselves tie at 9 Top-1 and 21 Top-10 for both pipelines.
The 821 pre-v43 cases improve from 328 to 334 Top-1 and from 589 to 595 Top-10.
This rules out an apparent gain caused only by adding easy evaluation images.

Gender breakdown:

| Domain | Cases | v42 Top-1 / Top-10 | v43 Top-1 / Top-10 |
|---|---:|---:|---:|
| Male | 245 | 90 / 167 | 93 / 173 |
| Female | 607 | 247 / 443 | 250 / 443 |

Male Top-10 receives the clearest benefit, while female Top-1 does not regress.
Race results are mixed: Miqo'te and Viera improve most at Top-1, while Au Ra
and Elezen regress. More balanced coverage is still required.

## Decision and next action

Do not run another public-validation evaluation for this small strict-CV gain.
Keep v35 as the mixed/seen deployment checkpoint and v43 as the current honest
unknown-item research dataset.

Related-link expansion from known male pages is nearly exhausted. The next data
round should use a new source-discovery path that still ends at Eorzea
Collection pages with exact equipment labels, such as bounded search/listing
page discovery or item-page-to-glamour discovery. It must retain the same
train-only assignment, target-page gender verification, exact equipment gate,
pose filter, and independent visual review.
