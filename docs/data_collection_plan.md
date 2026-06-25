# Data Collection Plan

Date: 2026-06-16

## Source Priority

Primary visual source:

- Eorzea Collection gearsets and gearset detail pages.

Structured metadata supplement:

- XIVAPI v2 Item sheet and related game-data sheets.

Optional fallback sources for later phases:

- MIRAPRI SNAP for extra player glamour examples.
- FFXIV Armoury Collection for set-level visual cross-checking.
- Gamer Escape for wiki-style set composition checks.

## Why Eorzea Collection First

Eorzea Collection is the best initial source because its gearset pages expose the data shape this project needs:

- Gearset list pages can be filtered by slot, role, content type, patch, gender/model, dyeability, and item level.
- Gearset detail pages include set preview images.
- Gearset detail pages list equipment by slot: HEAD, BODY, HANDS, LEGS, FEET.
- Each listed piece has a "Glamours using this piece" link, which can later expand the dataset from full-set previews into player-created mixed outfits.

This does not mean Eorzea Collection provides a public API. Treat it as page parsing, not API integration.

## Collection Principles

Use a conservative crawler:

- Respect the site's public access boundaries and review robots/terms before enabling broad collection.
- Start with a small allowlisted pilot, not a full-site crawl.
- Use low concurrency, ideally one request at a time during development.
- Add delays and exponential backoff.
- Cache every HTML page and image response locally.
- Use conditional requests or skip already downloaded files.
- Store source URLs and timestamps for provenance.
- Do not redistribute downloaded images; keep them for local personal research/use.

The implementation respects robots.txt by default. On 2026-06-16, a pilot request against an Eorzea Collection gearset page was blocked by robots.txt, so the project should not silently bypass that boundary. The `--ignore-robots` switch exists only for a user-reviewed policy decision or for importing pages from an explicitly permitted mirror/source.

## Gender and Race Variant Strategy

Each gear piece may look slightly different across character model, gender, and race. The dataset should therefore treat visual examples as variants, not as one universal image.

Variant sources:

- Gearset static images from `https://gearsets.eorzeacollection.com/...`.
- Gearset detail pages only as an index that exposes the static image URLs and equipment names.
- Do not use player-uploaded glamour pages as canonical single-item renders.
  Keep a separate, bounded `glamour_examples` collection for whole-outfit
  understanding and evaluation, with page URLs, creator attribution, equipment
  labels, and image provenance.
- When a detail page exposes multiple role-specific gearsets in the same series, collect each role-specific gearset because `fending`, `casting`, `healing`, and other role variants may have different appearances.
- Keep `front` and `back` images as full-set views. They are not single slots, but they are useful for whole-outfit matching and visual QA.

Variant metadata to preserve:

- `model_gender`: Female, Male, or unknown.
- `race`: Hyur Midlander, Highlander, Elezen, Miqo'te, Lalafell, Roegadyn, Au Ra, Viera, Hrothgar, or unknown.
- `slot`: HEAD, BODY, HANDS, LEGS, FEET, or unknown.
- `image_kind`: gearset preview, slot thumbnail, or glamour variant.
- `source_type`, `source_path`, `collection_slug`, `gearset_slug`, `image_part`, and `dye_state` parsed from the static image URL.
- `role_slug`: parsed from the gearset slug suffix, such as `fending`, `maiming`, `striking`, `scouting`, `aiming`, `casting`, or `healing`.

Coverage should be measured after every batch with `scripts/build_eorzea_dataset_index.py`. Missing slot/gender/race combinations should stay visible in `data/processed/eorzea_dataset_summary.json` so later collection runs can target gaps instead of downloading randomly.

Static image URL example:

```text
https://gearsets.eorzeacollection.com/crafted/leveling/courtly-lovers/courtly-lovers-healing/hyur-male-legs.png
```

Parsed fields:

- `source_type`: `crafted`
- `source_path`: `crafted/leveling/courtly-lovers`
- `collection_slug`: `courtly-lovers`
- `gearset_slug`: `courtly-lovers-healing`
- `race`: `hyur`
- `model_gender`: `male`
- `image_part`: `legs`
- `slot`: `LEGS`

The collector downloads only static gearset images by default. Race and gender values come from fixed enums:

- races: `hyur`, `elezen`, `miqote`, `lalafell`, `roegadyn`, `aura`, `viera`, `hrothgar`
- genders: `male`, `female`

By default, the pilot uses the fixed race/gender values exposed on the source page. For broader sweeps, pass explicit fixed ranges with `--races` and `--genders`.

## Phase A: Pilot Crawl

Scope:

- 5 to 10 gearset detail pages across different roles and content types.
- Include dyeable and undyeable sets.
- Include male/female model examples where available.

Steps:

1. Fetch the gearset list page.
2. Parse gearset cards: title, URL slug, content type, item level, minimum level, dyeability, patch marker, visible role/model filters.
3. Fetch each selected gearset page.
4. Parse equipment rows for HEAD, BODY, HANDS, LEGS, FEET.
5. Parse all preview image URLs and slot thumbnail URLs visible on the page.
6. Download images into `data/raw/eorzea_collection/images`.
7. Save raw HTML into `data/raw/eorzea_collection/html`.
8. Save one JSON metadata file per gearset into `data/raw/eorzea_collection/metadata`.

Acceptance criteria:

- The crawler can resume without redownloading unchanged files.
- The parser extracts all five core armor slots when a page has them.
- Every downloaded image has a source page URL and a local metadata record.

## Phase B: Gearset Index Crawl

Scope:

- All paginated Eorzea Collection gearset list pages.

Steps:

1. Crawl only list pages first.
2. Build `gearsets_raw.jsonl` with one row per gearset card.
3. Deduplicate by gearset URL slug.
4. Flag records that look like outfit-only, facewear-only, or incomplete gearsets.
5. Pick a bounded batch of detail pages for the next run.

Recommended output:

- `data/raw/eorzea_collection/metadata/gearsets_raw.jsonl`
- `data/interim/gearset_detail_queue.jsonl`

## Phase C: Gearset Detail Crawl

Scope:

- Detail pages queued from Phase B.

Steps:

1. Fetch detail pages with rate limiting.
2. Parse equipment slots and Lodestone item links.
3. Extract set preview images, model/gender variants, dye state where distinguishable, and slot thumbnails.
4. Save raw page data before normalization.
5. Record parse errors separately instead of dropping pages silently.

Recommended output:

- `data/raw/eorzea_collection/metadata/gearset_details_raw.jsonl`
- `data/raw/eorzea_collection/metadata/image_manifest_raw.jsonl`
- `var/logs/eorzea_collection_parse_errors.jsonl`

## Phase D: XIVAPI Enrichment

Scope:

- Gear pieces discovered from Eorzea Collection.

Steps:

1. Resolve item names and Lodestone links to candidate XIVAPI item rows.
2. Fetch item fields useful for filtering and matching.
3. Store the raw XIVAPI response separately from normalized records.
4. Keep unmatched Eorzea-only records; do not block the visual dataset on perfect item matching.

Useful fields:

- Item row ID.
- Name.
- Equip slot/category.
- Class/job restrictions.
- Level equip.
- Item level.
- Rarity.
- Dyeability if available.
- Icon path.
- Patch/version fields if available.

Recommended output:

- `data/raw/xivapi/items_raw.jsonl`
- `data/interim/item_resolution_candidates.jsonl`

## Phase E: Normalization

Create local tables:

- `gearsets.parquet` or `gearsets.sqlite`
- `gear_pieces.parquet` or `gear_pieces.sqlite`
- `images.parquet` or `images.sqlite`
- `gear_piece_images.parquet` or `gear_piece_images.sqlite`
- `source_pages.parquet` or `source_pages.sqlite`

Minimum normalized gear piece fields:

- `piece_id`
- `source`
- `source_item_url`
- `xivapi_item_id`
- `name_en`
- `slot`
- `gearset_id`
- `role_group`
- `job_codes`
- `equip_level`
- `item_level`
- `dyeable`
- `gender_fit`
- `content_type`

Minimum normalized image fields:

- `image_id`
- `source`
- `source_page_url`
- `source_image_url`
- `local_path`
- `sha256`
- `width`
- `height`
- `image_kind`
- `slot`
- `gearset_id`
- `piece_id`
- `model_gender`
- `dye_state`
- `downloaded_at`

## Phase F: Visual QA and Labeling

Automated checks:

- File exists and opens.
- Minimum resolution threshold.
- Duplicate detection by hash and perceptual hash.
- Broken placeholder/ad images removed.
- Extremely cropped or tiny images flagged.

Human review sample:

- Review 50 random gearsets.
- Review 20 random items per slot.
- Mark parser failures and ambiguous images.

Initial tags:

- Color: dominant palette and approximate named colors.
- Shape: robe, coat, jacket, armor plate, skirt, trousers, boots, gloves, hat, hood, mask.
- Style: fantasy, battle, formal, casual, cute, elegant, modern, eastern, tribal, gothic.
- Material hints: metal, leather, cloth, fur, lace, plate, chain.

## Recommended Python Libraries

Collection:

- `httpx` for HTTP requests.
- `selectolax` or `beautifulsoup4` for HTML parsing.
- `tenacity` for retries.
- `pydantic` for typed records.

Storage:

- `sqlite-utils` or built-in `sqlite3` for local metadata.
- `pyarrow` if choosing Parquet.
- Plain JSONL for raw crawl manifests.

Images:

- `Pillow` for image validation and dimensions.
- `imagehash` for perceptual dedupe.

Vision and matching, later:

- `open_clip_torch` or `sentence-transformers` CLIP variants for embeddings.
- `faiss-cpu` or `hnswlib` for vector search.

## First Implementation Target

Build `scripts/pilot_collect_eorzea.py` or an IDE-runnable module that:

1. Accepts a short list of Eorzea Collection gearset URLs.
2. Downloads and caches HTML.
3. Parses title, equipment slots, item names, Lodestone URLs, and image URLs.
4. Downloads images.
5. Writes JSONL manifests.
6. Prints a summary: pages fetched, pieces parsed, images downloaded, skipped files, errors.

This target is intentionally small. Once it is reliable, expand from a curated URL list to paginated gearset discovery.

Current local scripts:

- `scripts/pilot_collect_eorzea.py`: collect one or more gearset detail pages and optionally follow single-piece glamour examples.
- `scripts/pilot_collect_eorzea.py`: collect one or more gearset detail pages and the static gearset image URLs exposed on those pages.
- `scripts/discover_eorzea_gearsets.py`: discover gearset detail URLs from paginated list pages.
- `scripts/build_eorzea_dataset_index.py`: dedupe downloaded images and summarize slot/gender/race coverage.
## Player Glamour Example Pilot

Player-uploaded glamour pages can provide weakly supervised whole-outfit data:
several in-game screenshots are paired with the five core armor slots
(HEAD/BODY/HANDS/LEGS/FEET), item names,
dyes, character model, intended jobs, and level. These records must remain
separate from canonical gearset images because poses, lighting, reshade presets,
backgrounds, occlusion, and multi-slot equipment make them a different data
modality.

Collection rules:

- Use explicit glamour URLs or a reviewed allowlist; do not crawl the full site.
- Keep a hard page limit, one browser session, sequential requests, and delays.
- Use the browser-like HTTP session first and keep normal browser access as an
  explicit fallback. Cache HTML locally. Do not automate Cloudflare
  verification or attempt to bypass access controls.
- Download screenshot bodies only from the current glamour ID's public static
  image directory.
- Reject navigation icons, item icons, related-card thumbnails, non-image responses,
  undecodable files, and images smaller than 200x200.
- Preserve creator attribution and source page/image URLs.
- Keep screenshot bodies local; source-control only metadata and processed labels.
- Resolve item names against the normalized local SQLite dataset when possible.
- Expand multi-slot items across their occupied slots in the processed labels.

Pilot outputs:

- `data/raw/eorzea_collection/glamours/metadata/glamour_examples_raw.jsonl`
- `data/processed/glamour_examples.jsonl`
- `data/processed/glamour_examples_summary.json`
