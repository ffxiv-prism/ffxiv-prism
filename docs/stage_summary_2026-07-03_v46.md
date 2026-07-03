# v46 stage summary ? 2026-07-03

## Goal and current answer

The product goal is roughly 80% Top-1 BODY-equipment recognition for arbitrary
character-outfit images, without assuming a particular race, gender, visual
style, or source. Only FFXIV glamour pages with real equipment labels are used
for supervised training and accuracy validation. Unlabelled game, CG, or
illustration images remain valid inference targets but cannot establish
accuracy.

The goal is not yet met. The current mixed/seen-equipment deployment default is
still qv52 plus reranker v35:

- public fixed validation: 55/81 (67.9%) appearance Top-1;
- public fixed validation: 72/81 (88.9%) appearance Top-10.

The selected unknown-equipment research baseline remains v44:

- paired three-fold item-disjoint evaluation: 365/906 (40.3%) Top-1;
- paired three-fold item-disjoint evaluation: 676/906 (74.6%) Top-10;
- all three folds improve over the v43 baseline.

The v44 full-data model reaches only 41/81 (50.6%) Top-1 with the v2 index and
43/81 (53.1%) with the v3 index on public validation. It therefore does not
replace the deployment default. Blind data has not been inspected.

## What this stage completed

The data pipeline now contains 1,617 glamour groups and 6,447 images. The v46
quality manifest accepts 1,105 front/full-body images, of which 988 are
official-train BODY cases. Relative to v45, v46 adds 33 accepted female images
without removing earlier accepted images:

| Race | Added images |
|---|---:|
| Miqo'te | 17 |
| Au Ra | 9 |
| Viera | 3 |
| Hyur | 2 |
| Elezen | 2 |

Collection queues can now exclude all previously attempted IDs and write
bounded shards. This makes later balanced collection rounds reproducible and
prevents repeated requests for already reviewed pages.

Evaluation was corrected and hardened during this stage:

- query encoder and reranker are both trained from the base checkpoint inside
  each item-disjoint fold;
- train-only quality and item constraints are enforced;
- group weighting no longer accidentally multiplies image-rich groups;
- the fixed comparison uses 10 query epochs, 7 reranker epochs, and 1,000 hard
  negatives;
- public validation is not used to choose the v45 or v46 data variants.

The earlier 62.5% item-disjoint result reused a globally trained query encoder.
It remains a useful diagnostic but is not an unbiased unseen-item estimate.

## v46 paired strict result

The v45 and v46 pipelines were retrained on exactly the same v46 folds:

| Fold | Cases | v45 Top-1 / Top-10 | v46 Top-1 / Top-10 |
|---|---:|---:|---:|
| 1 | 327 | 140 / 229 | 140 / 231 |
| 2 | 330 | 129 / 250 | 132 / 248 |
| 3 | 330 | 117 / 253 | 116 / 249 |
| Pooled | 987 | 386 (39.1%) / 732 (74.2%) | 388 (39.3%) / 728 (73.8%) |

The 33 new cases improve from 8 to 10 Top-1 and remain 18 Top-10. On the 954
pre-v46 cases, Top-1 is unchanged at 378 while Top-10 falls from 714 to 710.

The gender guard shows another specialization shift:

| Domain | Cases | v45 Top-1 / Top-10 | v46 Top-1 / Top-10 |
|---|---:|---:|---:|
| Female | 640 | 254 / 470 | 258 / 467 |
| Male | 347 | 132 / 262 | 130 / 261 |

Female Top-1 recovers four cases, but male Top-1 loses two and aggregate
Top-10 loses four. Fold Top-1 deltas are 0, +3, and -1. This is too small and
too domain-dependent to justify promotion.

## Stage decision

- Keep v46 as the current, larger source-clean dataset.
- Keep v44 as the selected strict unknown-item research baseline.
- Keep qv52 plus reranker v35 as the mixed/seen deployment default.
- Do not train or publicly evaluate a full-data v46 checkpoint.
- Do not inspect the blind pool.
- Stop alternating large single-gender rounds as the main optimization method.

## Next stage

1. Build a deliberately stratified rare-domain increment. Select equipment
   pages first, then balance race and gender within each source stratum. Avoid
   another broad one-gender batch.
2. Measure the retrieval ceiling inside strict folds: truth-in-candidate
   coverage, base-encoder candidate recall, query-encoder recall, and reranker
   conversion. This decides whether the next bottleneck is indexing,
   representation, or ranking.
3. Run model-side experiments only under the frozen item-disjoint protocol.
   The first candidates are gender-balanced sampling and a domain-invariant
   query objective. Each experiment gets one predeclared configuration before
   all three folds are evaluated.
4. Promote a strict candidate only when pooled Top-1 and Top-10 both improve,
   at least two of three folds do not regress, and neither male nor female
   domains materially regress.
5. Use public validation once only after a candidate passes those guards.
   Preserve blind evaluation for the final confirmation near the product
   target.

The practical near-term milestone is not to claim 80% prematurely. It is to
raise strict unseen-item Top-1 while keeping candidate recall and race/gender
behavior stable, then close the gap between that robust model and the current
67.9% mixed/seen deployment result.
