# v45 balance guard — 2026-07-03

## Outcome

A second non-overlapping male item-page batch expands the reviewed corpus but
does not produce a stable enough model gain:

| Data | Groups | Images | Reviewed images | Train BODY cases |
|---|---:|---:|---:|---:|
| v44 | 1,524 | 6,097 | 1,024 | 907 |
| v45 | 1,578 | 6,299 | 1,072 | 955 |

v45 data is retained because every accepted image passes the same source,
label, pose, and visual-quality gates. The v45 qv62 fold checkpoints are not
promoted, no full-data v45 candidate is trained, and public validation and
blind predictions remain untouched.

## Collection

Twenty-four new BODY equipment pages discovered 824 glamour IDs. After
excluding existing records, all earlier attempted queues, and non-train stable
assignments, 420 candidates remained; per-item balancing selected 214.

Collection produced 54 exact male glamour groups and 202 images. Pose filtering
yielded 76 candidates. Manual review rejected 28 back, side, composite,
distant, action, effect-obscured, or weapon-obscured views. The final increment
contains 48 male BODY images:

| Race | Images |
|---|---:|
| Viera | 15 |
| Lalafell | 10 |
| Miqo'te | 7 |
| Hrothgar | 5 |
| Au Ra | 4 |
| Hyur | 3 |
| Roegadyn | 3 |
| Highlander | 1 |

This materially improves Hrothgar, Roegadyn, Lalafell, and Viera male coverage.

## Paired strict result

The fixed v44 and v45 pipelines were retrained on the same v45 item-disjoint
folds:

| Fold | v44 Top-1 / Top-10 | v45 Top-1 / Top-10 |
|---|---:|---:|
| 1, 318 cases | 122 (38.4%) / 237 (74.5%) | 121 (38.1%) / 237 (74.5%) |
| 2, 318 cases | 115 (36.2%) / 225 (70.8%) | 121 (38.1%) / 227 (71.4%) |
| 3, 318 cases | 137 (43.1%) / 244 (76.7%) | 136 (42.8%) / 245 (77.0%) |
| Pooled, 954 cases | 374 (39.2%) / 706 (74.0%) | 378 (39.6%) / 709 (74.3%) |

The pooled gain is only four Top-1 cases (+0.4 points) and three Top-10 cases
(+0.3 points). Top-1 paired discordance is 33 v45-only wins versus 29
v44-only wins. Two of the three folds lose one Top-1 case, so the effect is not
fold-stable.

The 906 pre-v45 cases improve by two Top-1 and six Top-10 cases. The 48 new
cases improve by two Top-1 but lose three Top-10 cases.

Most importantly, gender behavior moves in opposite directions:

| Domain | Cases | v44 Top-1 / Top-10 | v45 Top-1 / Top-10 |
|---|---:|---:|---:|
| Male | 347 | 127 / 256 | 136 / 256 |
| Female | 607 | 247 / 450 | 242 / 453 |

Male Top-1 gains nine while female Top-1 loses five. This is not catastrophic,
but it is exactly the type of specialization shift that the anti-overfitting
protocol is intended to catch. The aggregate score alone would hide it.

## Decision

- Keep v45 as a larger, source-clean dataset.
- Keep v44 qv60 strict folds as the selected unknown-item research baseline.
- Keep qv52 plus reranker v35 as the mixed/seen deployment default.
- Do not train or publicly evaluate a full-data v45 checkpoint.
- Stop consecutive male-only collection rounds.

The next data round should target rare races across both genders, or alternate
female and male source strata, and should require non-regression in both gender
domains before promotion. Model-side work should focus on representation and
candidate recall rather than tuning public-validation routing.
