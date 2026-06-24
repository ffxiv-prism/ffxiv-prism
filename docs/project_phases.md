# Project Phases

## Phase 0: Local Project Foundation

Goal: make the project easy to run, inspect, and extend locally.

Deliverables:

- Python package layout under `src/ffxiv_prism`.
- Data folders for raw, interim, processed, and indexed assets.
- Logging and cache folders for repeatable collection runs.
- Basic CLI or IDE-run entry points.

Relevant directories:

- `src/ffxiv_prism/cli`
- `var/logs`
- `var/cache`
- `tests`

## Phase 1: Gear Metadata Inventory

Goal: establish a canonical list of gear pieces and gearsets before downloading large images.

Deliverables:

- Gearset list from Eorzea Collection.
- Gear piece records by slot: head, body, hands, legs, feet.
- Supplemental item metadata from XIVAPI where available.
- Stable local IDs for gearsets, gear pieces, source URLs, and images.

Relevant directories:

- `src/ffxiv_prism/collectors/eorzea_collection`
- `src/ffxiv_prism/sources/xivapi`
- `src/ffxiv_prism/parsers`
- `data/raw/eorzea_collection/metadata`
- `data/raw/xivapi`

## Phase 2: Visual Asset Collection

Goal: download and preserve source-backed appearance images for each gearset and gear piece.

Deliverables:

- Cached HTML for every visited source page.
- Original downloaded images.
- Image metadata: source URL, page URL, slot, gearset, gender/model option, dye state, and crawl timestamp.
- A retryable manifest so interrupted runs can resume safely.

Relevant directories:

- `data/raw/eorzea_collection/html`
- `data/raw/eorzea_collection/images`
- `data/raw/eorzea_collection/metadata`
- `var/cache`
- `var/logs`

## Phase 3: Dataset Normalization

Goal: convert raw source records into a queryable local dataset.

Deliverables:

- One normalized table for gear pieces.
- One normalized table for gearsets.
- One normalized table for images.
- Slot-level links between images and pieces.
- Deduped images and quality flags.

Relevant directories:

- `src/ffxiv_prism/dataset`
- `data/interim`
- `data/processed`

## Phase 4: Tagging and Visual Feature Extraction

Goal: describe each piece visually enough to support outfit matching.

Deliverables:

- Rule-based tags from source fields: slot, role, job, source type, dyeable, gender fit, item level, patch.
- Vision tags from images: dominant colors, silhouette, clothing category, material hints, exposed skin/coverage, accessories.
- Embeddings for image similarity search.

Relevant directories:

- `src/ffxiv_prism/tagging`
- `src/ffxiv_prism/vision`
- `data/processed`
- `data/indexes`

## Phase 5: Outfit Image Understanding

Goal: analyze an arbitrary uploaded reference image into comparable outfit components.

Deliverables:

- Person/outfit crop.
- Slot segmentation or approximate region extraction.
- Reference tags and image embeddings by region.
- Confidence scores and missing-slot handling.
- A separate source-backed player glamour example set pairing whole screenshots with explicit equipment labels.

Relevant directories:

- `src/ffxiv_prism/vision`
- `data/samples`

## Phase 6: Glamour Matching

Goal: recommend a mixed FF14 glamour set that visually resembles the reference image.

Deliverables:

- Candidate retrieval per slot.
- Cross-slot compatibility scoring.
- Optional constraints: playable job, gender/model, dyeability, obtainable source, level.
- Final recommendation with gear names, slots, source URLs, local preview images, and confidence notes.

Relevant directories:

- `src/ffxiv_prism/matching`
- `data/indexes`

## Phase 7: Evaluation and Iteration

Goal: make recommendations measurably better over time.

Deliverables:

- Hand-picked test references.
- Expected or preferred glamour outputs.
- Evaluation scripts for retrieval quality.
- Error categories: wrong color, wrong silhouette, wrong role restriction, poor image quality, missing item.

Relevant directories:

- `tests`
- `notebooks`
- `data/samples`

