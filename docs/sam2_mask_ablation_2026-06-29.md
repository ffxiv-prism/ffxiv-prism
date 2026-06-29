# SAM2 Mask Ablation — 2026-06-29

## Question

Can promptable SAM2 masks improve screenshot-to-canonical-render retrieval over
the existing raw pose-relative crops?

## Pilot configuration

- Evaluation subset: all 37 locally available examples accepted by
  `front_fullbody_quality_final_v1`.
- Eligible slot cases: 184.
- Query crops: one highest-priority crop per slot and example.
- Segmenter: `facebook/sam2.1-hiera-small`.
- Segment prompt: an inset box over each pose-relative slot crop.
- Mask selection: SAM predicted IoU with slot-specific area sanity bounds.
- Mask composition: dilated and feathered mask on a fixed gray background.
- Retrieval encoder: `facebook/dinov2-small`.
- Candidate index: the existing 19,721-image DINOv2-small index.
- Fusion: weighted reciprocal-rank fusion, raw weight 0.6.
- Device: CPU.

The experiment is implemented in
`scripts/evaluate_sam2_mask_ablation.py`. Generated masks are written below
`data/debug/`, and detailed reports are ignored local artifacts.

## Results

| Method | Appearance Top-1 | Recall@10 | MRR |
|---|---:|---:|---:|
| Raw crop | 3.80% | 15.76% | 0.0678 |
| SAM2 masked crop | 3.26% | 10.87% | 0.0544 |
| Raw + SAM2 weighted RRF | 4.35% | 16.85% | 0.0773 |
| Exploratory slot router | 4.89% | 17.93% | 0.0841 |

The raw/SAM2 result union had an oracle Recall@10 of 20.11%, showing that SAM2
adds some candidates that the raw branch misses even though it is weaker by
itself.

### Per-slot raw versus SAM2

| Slot | Raw Top-1 | SAM2 Top-1 | Raw R@10 | SAM2 R@10 |
|---|---:|---:|---:|---:|
| HEAD | 2.70% | 5.41% | 8.11% | 8.11% |
| BODY | 10.81% | 5.41% | 37.84% | 18.92% |
| HANDS | 2.70% | 5.41% | 8.11% | 16.22% |
| LEGS | 0.00% | 0.00% | 13.89% | 11.11% |
| FEET | 2.70% | 0.00% | 10.81% | 0.00% |

The exploratory router uses SAM2 for HANDS, raw/SAM2 RRF for HEAD, and raw
crops for BODY, LEGS, and FEET. It was selected on this same pilot and is not an
unbiased validation result.

## Conclusion

SAM2 should not replace raw crops globally. It materially helped HANDS and
added complementary HEAD candidates, but it damaged BODY and FEET retrieval.
The supported direction is a slot-aware auxiliary branch:

1. Keep raw crops as the default recall representation.
2. Add SAM2 masks for HANDS.
3. Test raw/SAM2 fusion for HEAD.
4. Keep SAM2 disabled for BODY, LEGS, and FEET unless a better prompting policy
   passes a separate validation.

## Limitations and next test

This is a development ablation, not a blind result. It uses the old local
DINOv2-small assets rather than the current DINOv2-base v17 query encoder, and
the router was chosen after inspecting this evaluation.

The next GPU experiment should:

1. Rebuild SAM2 masks for the full current validation split.
2. Limit the first pass to HANDS and HEAD.
3. Encode both raw and SAM2 variants with the v17 DINOv2-base query encoder.
4. Train or evaluate the corresponding pair reranker without consuming the
   sealed blind set.
5. Freeze the routing policy before running a newly collected blind set.

## Reproduction

```powershell
python scripts\evaluate_sam2_mask_ablation.py `
  --limit-examples 37 `
  --max-crops-per-slot 1 `
  --sam-batch-size 2 `
  --device auto `
  --output-dir data\debug\sam2-mask-ablation-hq37 `
  --report data\processed\sam2_mask_ablation_hq37.json
```
