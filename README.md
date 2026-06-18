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

Use the local Anaconda `dataprocess` environment:

```powershell
$py = 'C:\Softwares\Anaconda3\envs\dataprocess\python.exe'
```

Run the standard refresh pipeline:

```powershell
& $py scripts\run_data_pipeline.py
```

Run the full discovery plus refresh pipeline:

```powershell
& $py scripts\run_data_pipeline.py --all
```

Run only selected stages:

```powershell
& $py scripts\run_data_pipeline.py --stage collect-details --stage build-image-manifest
& $py scripts\run_data_pipeline.py --stage cafemaker-zh
& $py scripts\run_data_pipeline.py --stage huiji-missing-zh
& $py scripts\run_data_pipeline.py --stage build-index
```

For HuijiWiki enrichment, Selenium opens a dedicated Chrome profile. If
Cloudflare asks for verification, complete it in that automation window. Proxy
usage is optional and is disabled by default; pass it only when needed:

```powershell
& $py scripts\run_data_pipeline.py --stage huiji-missing-zh --huiji-proxy http://127.0.0.1:10808
```

Single-item Huiji lookup:

```powershell
& $py scripts\search_huiji_item_zh_name.py "Augmented Courtly Lover's Surcoat of Maiming" --json
```

## Pipeline Stages

`scripts/run_data_pipeline.py` is a thin runner over the individual scripts:

1. `discover`: discover Eorzea Collection gearset URLs.
2. `collect-details`: fetch and parse gearset detail pages.
3. `build-image-manifest`: rebuild text image URL manifests from cached details.
4. `cafemaker-zh`: refresh CafeMaker name cache and exact English-to-Chinese matches.
5. `huiji-missing-zh`: use Selenium/HuijiWiki to fill remaining Chinese names.
6. `build-index`: build processed image index and coverage summary.

The default refresh pipeline runs stages 2 through 6. Use `--all` when the
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
- `data/processed/eorzea_dataset_summary.json`

Downloaded images and raw HTML caches are intentionally not versioned.

## Notes

Project collection scripts skip robots.txt checks by default after local user
approval. Pass `--respect-robots` on a command when you want to re-enable
robots.txt checks for that run.

Canonical images come from `gearsets.eorzeacollection.com` static gearset files.
Player-uploaded glamour pages are intentionally excluded from the dataset.
