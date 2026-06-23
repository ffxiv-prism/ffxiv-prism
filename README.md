# FFXIV Prism

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

Use the local Anaconda `dataprocess` environment. Keep the environment name the
same on each machine, and let `python` come from the activated environment.

```powershell
conda activate dataprocess
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
Player-uploaded glamour pages are intentionally excluded from the dataset.




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
