# FFXIV Prism

Current visual-recognition status, reproducible commands, rejected experiments,
and next steps are recorded in
[`docs/handoff_2026-06-29.md`](docs/handoff_2026-06-29.md).

FFXIV Prism is a local Python project for building a Final Fantasy XIV glamour
appearance dataset. The current milestone is a source-backed gearset and gear
piece dataset with English and Simplified Chinese item names.

## Scope

- Local Python tooling only; no frontend yet.
- Eorzea Collection is the primary visual source.
- HuijiWiki/CafeMaker data is used to enrich Chinese item names.
- Downloaded HTML pages, browser profiles, and image bodies are local cache
  artifacts. JSON and JSONL metadata are the source-controlled dataset.

## Main Commands

Use the dedicated Anaconda `ffxiv-prism` environment with Python 3.12. Create
it from the checked-in environment file when setting up a new machine:

```powershell
conda env create -f environment.yml
conda activate ffxiv-prism
```

Run the standard refresh pipeline:

```powershell
python scripts\run_data_pipeline.py
```

Run the full discovery plus refresh pipeline:

```powershell
python scripts\run_data_pipeline.py --all
```

Run only selected stages:

```powershell
python scripts\run_data_pipeline.py --stage collect-details --stage build-image-manifest
python scripts\run_data_pipeline.py --stage cafemaker-zh
python scripts\run_data_pipeline.py --stage huiji-missing-zh
python scripts\run_data_pipeline.py --stage build-index
python scripts\run_data_pipeline.py --stage validate-images
python scripts\run_data_pipeline.py --stage build-sqlite
```

For HuijiWiki enrichment, Selenium opens a dedicated Chrome profile. If
Cloudflare asks for verification, complete it in that automation window. Proxy
usage is optional and is disabled by default; pass it only when needed:

```powershell
python scripts\run_data_pipeline.py --stage huiji-missing-zh --huiji-proxy http://127.0.0.1:10808
```

Single-item Huiji lookup:

```powershell
python scripts\search_huiji_item_zh_name.py "Augmented Courtly Lover's Surcoat of Maiming" --json
```

## Pipeline Stages

`scripts/run_data_pipeline.py` is a thin runner over the individual scripts:

1. `discover`: discover Eorzea Collection gearset URLs.
2. `collect-details`: fetch and parse gearset detail pages.
3. `build-image-manifest`: rebuild text image URL manifests from cached details.
4. `cafemaker-zh`: refresh CafeMaker name cache and exact English-to-Chinese matches.
5. `huiji-missing-zh`: use Selenium/HuijiWiki to fill remaining Chinese names.
6. `build-index`: build processed image index and coverage summary.
7. `validate-images`: verify local image files and write image QA outputs.
8. `build-sqlite`: build the normalized SQLite dataset.

The default refresh pipeline runs stages 2 through 8. Use `--all` when the
gearset discovery list itself should be refreshed.

## Dataset Files

Important text dataset files:

- `data/raw/eorzea_collection/metadata/gearset_discovery_raw.jsonl`
- `data/raw/eorzea_collection/metadata/gearset_discovery_by_model_raw.jsonl`
- `data/interim/gearset_detail_queue.jsonl`
- `data/raw/eorzea_collection/metadata/gearset_details_raw.jsonl`
- `data/raw/eorzea_collection/metadata/gear_pieces_raw.jsonl`
- `data/raw/eorzea_collection/metadata/image_url_manifest_raw.jsonl`
- `data/raw/eorzea_collection/metadata/image_url_checks_raw.jsonl`
- `data/raw/game_data/cafemaker_item_names_raw.jsonl`
- `data/raw/game_data/cafemaker_item_names_for_gear_pieces_raw.jsonl`
- `data/processed/eorzea_images_index.jsonl`
- `data/processed/eorzea_images_qa_index.jsonl`
- `data/processed/eorzea_image_qa_summary.json`
- `data/processed/eorzea.sqlite`
- `data/processed/eorzea_sqlite_summary.json`
- `data/processed/eorzea_dataset_summary.json`

Downloaded images and raw HTML caches are intentionally not versioned.

## Notes

Project collection scripts skip robots.txt checks by default after local user
approval. Pass `--respect-robots` on a command when you want to re-enable
robots.txt checks for that run.

Canonical images come from `gearsets.eorzeacollection.com` static gearset files.
Player-uploaded glamour pages are excluded from the canonical single-piece image
dataset. A separate, bounded `glamour_examples` dataset may retain their
source-backed screenshots and equipment labels for whole-outfit understanding
and evaluation.




## Labeled Player Glamour Examples

Player-uploaded glamour pages are collected separately from canonical gearset
images. Each page supplies several in-game screenshots plus an equipment list,
dyes, character model, level, tags, and provenance. The default collector uses a
browser-like HTTP session, warming the site home page before loading explicit
glamour URLs. A normal Chrome session remains available with
`--fetch-mode browser` when needed.

Collect a small explicit allowlist, with a hard per-run page bound:

```powershell
python scripts\collect_eorzea_glamours.py 302518 --refresh --max-pages 1
```

For training and evaluation collections, require every non-empty core slot to
resolve to the exact local item record before downloading screenshots:

```powershell
python scripts\collect_eorzea_glamours.py --url-file candidates.txt `
  --max-pages 50 --require-all-core-matched
```

Equipment resolution distinguishes three outcomes:

- `exact_item`: the uploaded glamour item matches a local item name exactly.
- `appearance_equivalent`: a known variant name points to the same appearance,
  such as augmented, replica, Anemos, or costume variants. Retrieval may return
  these candidates, but presentation must label them as same appearance with a
  different item ID.
- `unmatched`: no trustworthy exact or same-appearance local candidate exists.

The image filter accepts only the current glamour's static gallery directory,
requires an image content type, verifies the body with Pillow, and rejects images
smaller than 200x200. Navigation icons, slot icons, item icons, advertisements,
and related-glamour thumbnails are therefore excluded.
Cached pages are parsed without revisiting the site on later runs. Build the
image-level manifest, including labels inherited from multi-slot equipment:

```powershell
python scripts\build_glamour_examples_dataset.py
```

Important outputs:

- `data/raw/eorzea_collection/glamours/metadata/glamour_examples_raw.jsonl`
- `data/processed/glamour_examples.jsonl`
- `data/processed/glamour_examples_summary.json`

Raw HTML and player screenshots remain local cache artifacts. The JSONL records
retain source URLs and must not be treated as canonical item renders.

## Appearance Matching Plan

The first matching milestone is a local visual retrieval loop over the normalized
SQLite dataset:

1. Build image embeddings for ready slot images from `v_piece_images`.
2. Embed a reference image with the same model.
3. Retrieve candidates by slot, then rerank with structural constraints and data
   quality signals.
4. Combine per-slot candidates into a glamour recommendation with source URLs,
   local preview paths, and confidence notes.

Dye handling is deliberately conservative:

- A gear piece is known dyeable when any same-piece or same-gearset slot image has
  a `dyed` variant.
- Missing `dyed` variants do not prove that a piece is undyeable; they mean the
  dataset lacks positive dye evidence.
- `dyed` images are evidence for dyeability only. Their current colors must not be
  used as visual matching features.
- When a candidate is dyeable, color mismatch should not penalize it strongly,
  because the player can recolor it in game.
- When a candidate has no dyeability evidence, color can remain part of reranking,
  but it should be treated as a weaker signal than shape, silhouette, slot, and
  style.

A practical first scoring shape is:

```text
score =
  visual_embedding_similarity
+ silhouette_or_style_bonus
+ slot_and_constraint_fit
+ data_quality_bonus
- non_dyeable_color_mismatch_penalty
```

Color features should therefore be stored separately from embeddings and applied
only during reranking with dyeability awareness.

## Visual Retrieval MVP

The dedicated Conda environment keeps Hugging Face, Torch, and pip caches under
the project `var/cache/` directory on the D drive. Activate it before running
GPU experiments:

```powershell
conda activate ffxiv-prism
```

Verify the CUDA runtime before a long indexing run:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Build a CPU-compatible CLIP index for normal slot images:

```powershell
python scripts\build_visual_index.py --batch-size 16
```

For a quick smoke test, build a bounded index first:

```powershell
python scripts\build_visual_index.py --limit 100 --output-dir data\indexes\clip-smoke
```

Search one armor slot:

```powershell
python scripts\search_visual_index.py reference.png --slot BODY --top-k 10

# Restrict candidates to the target character model when known.
python scripts\search_visual_index.py reference.png --slot BODY --gender female --race hyur
```

Run the checked-in 10-case transformed-query evaluation set. It measures whether
mirroring, cropping, and mild color changes still retrieve the correct piece:

```powershell
python scripts\evaluate_visual_retrieval.py --top-k 10
```

Run the separate cross-model stress set. This intentionally asks female and male
views of the same item to retrieve each other, even when their designs differ:

```powershell
python scripts\evaluate_visual_retrieval.py `
  --cases data\samples\retrieval_eval_cross_view.jsonl `
  --output data\processed\retrieval_eval_cross_view_results.json `
  --top-k 10
```

Regenerate deterministic evaluation cases when the dataset changes:

```powershell
python scripts\build_retrieval_eval_set.py --mode transformed `
  --output data\samples\retrieval_eval.jsonl
python scripts\build_retrieval_eval_set.py --mode cross-view `
  --output data\samples\retrieval_eval_cross_view.jsonl
```

Generate larger on-disk noise suites. Each profile samples six pieces per slot
and writes real JPEG query images with flips, non-uniform stretching, rotation,
shear, blur, compression, and deterministic pixel noise:

```powershell
python scripts\build_augmented_retrieval_eval_set.py `
  --profile mild --per-slot 6 `
  --output data\samples\retrieval_eval_mild.jsonl

python scripts\build_augmented_retrieval_eval_set.py `
  --profile stress --per-slot 6 `
  --output data\samples\retrieval_eval_stress.jsonl
```

Evaluate both generated suites:

```powershell
python scripts\evaluate_visual_retrieval.py `
  --cases data\samples\retrieval_eval_mild.jsonl `
  --output data\processed\retrieval_eval_mild_results.json

python scripts\evaluate_visual_retrieval.py `
  --cases data\samples\retrieval_eval_stress.jsonl `
  --output data\processed\retrieval_eval_stress_results.json
```

The generated JPEG bodies stay under `data/samples/retrieval_eval_images/` and
are ignored by Git; their JSONL manifests remain source-controlled and
reproducible from the local source image cache.

The default index excludes `dyed` images so current dye colors do not become the
primary visual signal. Generated indexes and evaluation outputs remain local.

### Screenshot-to-Render Domain Adapter

Train and evaluate the slot-conditioned low-rank adapter on deterministic
`glamour_id` groups:

```powershell
python scripts\train_glamour_adapter.py --device cuda
```

The command uses 30/10/10 glamour groups for training, validation, and held-out
testing. It also evaluates DINOv2/CLIP score fusion and a cascade that takes
Top-1 from the precision fusion model while filling the rest of Top-10 from the
higher-recall adapted DINOv2 model. The checkpoint is written under
`data/models/`; generated checkpoints and reports stay local.

### Human Parsing and Background Masks

Build both person-foreground and strict slot-mask DINOv2 query caches:

```powershell
python scripts\build_masked_glamour_crop_embeddings.py --device cuda
```

Train the person-mask adapter without changing the clean-render candidate
index:

```powershell
python scripts\train_glamour_adapter.py `
  --crop-cache data\processed\glamour_crop_embeddings_facebook-dinov2-base-color-mask-person.npz `
  --checkpoint data\models\glamour_adapter_dinov2_base_mask_person.pt `
  --output data\processed\glamour_adapter_eval_dinov2_base_mask_person.json `
  --device cuda
```

Combine person-mask Top-1 precision with the raw-image cascade's retrieval tail:

```powershell
python scripts\evaluate_mask_ensemble.py
```

Human parsing uses SegFormer clothing labels, dilated and feathered before
compositing onto the same black background used by transparent clean renders.
Strict slot masks fall back to the full person mask when the requested semantic
region is too small.

## Next Milestone: Whole-Outfit Screenshot Understanding

The next milestone is an end-to-end baseline that accepts a real player glamour
screenshot, extracts approximate regions for the five core armor slots, searches
the existing single-piece CLIP index, and compares the results with the equipment
labels collected from the source glamour page.

Implementation plan:

1. Expand the reviewed player-glamour allowlist to 50-100 pages, covering both
   model genders, several races, varied styles, and complete and multi-slot
   outfits.
2. Measure item-to-SQLite resolution coverage and investigate unmatched core
   armor items before treating the labels as evaluation truth.
3. Split examples by `glamour_id`, never by image, so multiple screenshots of the
   same outfit cannot leak across development and validation sets.
4. Build a deterministic person/outfit crop and approximate HEAD, BODY, HANDS,
   LEGS, and FEET regions from each full screenshot.
5. Search each visible region against its corresponding slot index, with optional
   model gender and race filters.
6. Evaluate Recall@1, Recall@5, Recall@10, and MRR per slot and overall. Report
   empty slots, multi-slot items, unmatched labels, and invisible or occluded
   regions separately.
7. Save annotated crop-debug images and retrieval previews for human review
   before adding more complex segmentation or reranking.

### Pose-Driven Outfit Region Debugging

Player screenshots are treated as single-person images. The baseline first runs
ViTPose over the full image. When that pose is weak, it searches overlapping,
center-biased windows for the best single pose; it does not count or classify
multiple people. A second pass tightens the box around the selected skeleton.

Each slot receives multiple pose-relative crops. These are intentionally
slightly redundant because glamour model coverage does not always follow body
anatomy: long sleeves can be part of body gear or hand gear, long boots can cover
shin areas, and head items can be hats, hair ornaments, eyewear, masks, or other
small accessories.

- HEAD: full head, upper-accessory, face-accessory, and head-with-shoulders context crops. These are tagged when they may include face or hair features.
- BODY: torso, torso-plus-upper-arm overlap, and wider body context crops. The upper-arm overlap crop is tagged as BODY/HANDS-ambiguous.
- HANDS: left/right arm crops, forearm-focused crops for long gloves, and combined arm-cover context.
- LEGS: left/right leg crops, thigh-focused crops that reduce boot leakage, lower-leg/boot-overlap crops, and combined lower-body context.
- FEET: left/right foot crops, boot-height crops for long boots, and combined lower-leg context.

The JSON manifest includes each crop's `role` (`primary`, `auxiliary`, or
`context`) and `tags` such as `body_hands_overlap`, `legs_feet_overlap`,
`may_include_face_hair`, and `long_boot_focus`. Retrieval should treat primary
crops as precision evidence, auxiliary overlap crops as recall evidence, and
context crops as tie-breakers or visual-debug evidence. If adjacent slots
strongly disagree, the overlap-tagged crops should be down-weighted rather than
forcing a hard boundary between equipment slots.

Run a small visual QA batch:

```powershell
python scripts\debug_outfit_regions.py reference.png `
  --output-dir data\debug\outfit-regions\sample
```

Run all labeled screenshots:

```powershell
python scripts\debug_outfit_regions.py `
  --dataset data\processed\glamour_examples.jsonl `
  --output-dir data\debug\outfit-regions\full --quiet
```

The command writes skeleton/region overlays, crop contact sheets, individual
crops, JSON manifests, and a visibility summary. Generated debug assets remain
local. Slots marked `pose_uncertain` or `out_of_frame` should be excluded from
retrieval evaluation; `partially_visible` and `occluded` crops should be retained
with reduced weight.
Initial acceptance criteria:

- At least 50 labeled glamour pages and 150 usable screenshots.
- At least 95% of non-empty core-slot labels resolved to local item records, or
  explicitly categorized as missing canonical data.
- A reproducible split manifest grouped by `glamour_id`.
- One CLI command that produces five-slot candidates and an evaluation report
  from the labeled screenshot set.
- Crop-debug output that makes region failures inspectable without rerunning the
  collector.
