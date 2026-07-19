# FFXIV Prism

The latest production BODY promotion, including the frozen DINOv3-adapted
Top-1-only gate, strict three-fold domain audit, real-image acceptance test,
and final Demo usability judgment, is recorded in
[`docs/body_dinov3_adapted_gate_v99_stage_summary_2026-07-20_zh.md`](docs/body_dinov3_adapted_gate_v99_stage_summary_2026-07-20_zh.md).
The promoted BODY DINOv2 + DINOv3 ensemble, strict three-fold results, domain
audit, and production integration are recorded in
[`docs/body_dinov3_v96_stage_summary_2026-07-19_zh.md`](docs/body_dinov3_v96_stage_summary_2026-07-19_zh.md).
The current HANDS production promotion, including the frozen occupied-slot
Top-1 gate and cross-slot BODY consistency safeguard, is recorded in
[`docs/hands_occupied_slot_gate_v93_stage_summary_2026-07-19_zh.md`](docs/hands_occupied_slot_gate_v93_stage_summary_2026-07-19_zh.md).
The earlier five-slot baseline is recorded in
[`docs/non_body_optimization_stage_summary_2026-07-10_zh.md`](docs/non_body_optimization_stage_summary_2026-07-10_zh.md).

Current visual-recognition status, reproducible commands, rejected experiments,
and next steps are recorded in
[`docs/handoff_2026-06-29.md`](docs/handoff_2026-06-29.md). The latest SAM2
slot-mask ablation is recorded in
[`docs/sam2_mask_ablation_2026-06-29.md`](docs/sam2_mask_ablation_2026-06-29.md).
The high-quality recognition target and evaluation guardrails are recorded in
[`docs/high_quality_recognition_plan_2026-06-29.md`](docs/high_quality_recognition_plan_2026-06-29.md).

FFXIV Prism is a local Python project for building a Final Fantasy XIV glamour
appearance dataset. The current milestone is a source-backed gearset and gear
piece dataset with English and Simplified Chinese item names.

## 中文用户手册（Demo）

本项目当前的初期目标流程是：输入一张人物穿搭图片，自动裁剪 `HEAD`、
`BODY`、`HANDS`、`LEGS`、`FEET` 五个部位，并为每个部位输出 FF14 中
视觉最相似的 10 件装备。结果以本地 HTML 报告呈现，每个候选包含装备名、
装备图片、相对排序分数和主要裁剪证据。

当前最适合单人物、全身可见、正面或接近正面的 FF14 截图。插画、真人照片、
多人图和严重遮挡图片也可以尝试，但 Demo 阶段尚不保证识别准确率；Top-10
应当作为供人工挑选的相似候选，而不是确定的装备鉴定结果。

### 1. 激活环境并检查本地资产

在项目根目录打开 PowerShell，激活已经安装好的 CPU 或 GPU 环境，然后先运行：

```powershell
python scripts\recognize_glamour.py --check
```

输出中的 `ok` 应为 `true`。如果为 `false`，请根据 `missing` 或 `invalid`
列表补齐本地模型、索引或检查点。模型推理默认使用 `--device auto`：有可用
CUDA 时使用 GPU，否则回退到 CPU。

### 2. 输入图片并生成五部位 Top-10 报告

```powershell
python scripts\recognize_glamour.py "D:\images\my-glamour.png" `
  --top-k 10 `
  --profile current-best `
  --output-dir data\outputs\glamour-match\my-glamour
```

如果已知角色模型性别，可以增加 `--gender female` 或 `--gender male` 来减少
不适用候选。不传 `--output-dir` 时，默认输出到
`data/outputs/glamour-match/<输入文件名>/`。

### 3. 查看结果

运行成功后，直接用浏览器打开输出目录中的 `report.html`。报告按五个部位分节，
每个部位保留 Top-10，并显示本地装备预览图片。目录还包含：

- `results.json`：完整机器可读结果；
- `overlay.jpg`：人物姿态和部位区域覆盖图；
- `crops.jpg`：检索裁剪与质检裁剪的联系表；
- `crops/`：逐部位裁剪图片，包含检索裁剪和零权重质检裁剪；
- `report.html`：推荐给普通用户查看的最终报告。

FEET 会额外生成 `feet_full_left`、`feet_full_right` 和
`feet_full_context` 三张完整鞋/长靴质检裁剪。它们用于检查鞋尖、鞋底是否被框住，
不会送入当前已验证的检索模型，也不会改变候选排名。如果原图边缘本身截断了鞋，
`results.json` 会把该部位标为低可信并提示 `source_edge_truncation_risk`；算法无法从
原图之外恢复缺失的鞋子。

HEAD 会额外生成 `head_full_accessory`，用于覆盖宽帽檐、高帽冠和超出脸部范围的
大型头饰。它同样是 `role=diagnostic` 的零权重质检图；原有
`head_upper_accessory` 保持不变，因此不会改变当前已验证的 HEAD 排名。

`overlay.jpg` 中 HEAD 与 FEET 的红/紫色主框显示的是完整性检查范围，并分别标记为
`HEAD full`、`FEET full`；它们不是模型实际使用的紧凑检索框。白色人体细化框也会按
头饰跨度和小腿长度预留空间，避免第二次姿态估计在输入阶段切掉宽檐帽、鞋尖或厚底。
如果完整框已经贴到原图边缘且仍看不到装备外沿，应优先换用没有裁掉人物的原图。

只识别部分部位时，可使用例如
`--slots HEAD BODY LEGS`。`--candidate-k` 默认是 1000，通常无需修改；它是
内部召回池大小，不是 HTML 中显示的候选数。

## Scope

- Local Python tooling only; no frontend yet.
- Eorzea Collection is the primary visual source.
- HuijiWiki/CafeMaker data is used to enrich Chinese item names.
- Downloaded HTML pages, browser profiles, and image bodies are local cache
  artifacts. JSON and JSONL metadata are the source-controlled dataset.

## Main Commands

Use Python 3.12 and select the environment that matches the machine:

```powershell
# CPU-only development machine
conda env create -f environment-cpu.yml
conda activate ffxiv-prism-cpu

# NVIDIA GPU development machine (CUDA 13.0)
conda env create -f environment-gpu.yml
conda activate ffxiv-prism-gpu
```

`environment.yml` remains a hardware-neutral fallback. Model commands default
to `--device auto`: CUDA is selected when the installed PyTorch build can use
it, otherwise execution falls back to CPU. Pass `--device cpu` for a forced CPU
run or `--device cuda` when a missing CUDA runtime should be treated as an
error. CPU training is supported but is substantially slower, so start with a
small `--batch-size` and bounded inputs for smoke tests.

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

The project keeps Hugging Face model files under `var/cache/`. Activate the CPU
or GPU environment described above before running model workloads.

Verify the selected PyTorch runtime before a long indexing or training run:

```powershell
python -c "import torch; print(torch.__version__); print('cuda' if torch.cuda.is_available() else 'cpu'); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU mode')"
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

### High-Quality Recognition Track

Query-encoder and pair-reranker training can be restricted to a reviewed
quality manifest. Only rows with `passed=true` participate in either training
or evaluation, while the split remains grouped by `glamour_id`:

```powershell
python scripts\train_glamour_query_encoder.py `
  --index-dir data\indexes\facebook-dinov2-base-slot-aligned-v2 `
  --quality-manifest data\processed\front_fullbody_quality_final_v33.jsonl `
  --slots BODY `
  --group-by-glamour `
  --checkpoint data\models\glamour_query_encoder_hq.pt `
  --output data\processed\glamour_query_encoder_eval_hq.json `
  --device auto

python scripts\train_glamour_pair_reranker.py `
  --index-dir data\indexes\facebook-dinov2-base-slot-aligned-v2 `
  --quality-manifest data\processed\front_fullbody_quality_final_v33.jsonl `
  --slots BODY `
  --group-by-glamour `
  --checkpoint data\models\glamour_pair_reranker_hq.pt `
  --output data\processed\glamour_pair_reranker_eval_hq.json `
  --device auto
```

The quality manifest can also be narrowed to a single reliable slot, for
example `--slots BODY`. `--checkpoint` is the model path; `--output` is the JSON
evaluation report.

Before claiming a split is large enough for model selection or confirmation,
summarize prediction-independent coverage:

```powershell
python scripts\summarize_quality_split_coverage.py `
  --quality-manifest data\processed\front_fullbody_quality_final_v33.jsonl `
  --slots BODY

python scripts\summarize_evaluation_domains.py `
  --report data\processed\glamour_pair_reranker_eval.json `
  --split validation `
  --output data\processed\glamour_pair_reranker_eval.domain_coverage.json
```

Quality decisions must be frozen before inspecting predictions. A repeatedly
inspected validation set is for development only; the usable-model target must
be confirmed once on newly collected sealed high-quality glamour groups.

Hyur-female BODY recognition is retained only as a narrow regression track.
It reached 80% on 15 development images, but only 42.9% on the subsequent
one-time 7-image sealed BODY test. It does not establish cross-race,
cross-gender, or cross-source usability.

The current all-domain BODY dataset is v46: 1,617 glamour groups, 6,447 images,
and 1,105 reviewed front/full-body images, including 988 official-train BODY
cases. The mixed/seen-equipment deployment default remains the v35 reranker
with qv52, seven fixed epochs, and 1,000 mined hard negatives. With the crop
configuration frozen from train-only cross-validation, it reaches 55/81
(67.9%) single-image appearance Top-1 and 72/81 (88.9%) Top-10 on public
validation.

Unseen-equipment research now uses fully end-to-end BODY-item-disjoint
three-fold evaluation: both the query encoder and reranker are trained from
scratch inside each fold. The selected research baseline remains v44: on the
same v44 folds and 906 evaluation images,
v44 reaches 365/906 (40.3%) Top-1 and 676/906 (74.6%) Top-10 with the
multiscale v3 index, versus 356/906 (39.3%) and 664/906 (73.3%) for the paired
v43 data baseline. All three folds improve. v45 is retained as data but not
promoted as a model-selection result: on its paired folds it adds only +0.4
Top-1 points, regresses in two of three folds, and shifts Top-1 from female to
male cases. The female-balanced v46 increment is also retained as data but not
promoted: against v45 on the same 987 cases it changes Top-1 from 386 to 388
(+0.2 points) and Top-10 from 732 to 728 (-0.4 points). Female Top-1 gains four
cases while male Top-1 loses two, so the result still fails the cross-domain
non-regression guard. The earlier 361/578 (62.5%)
result reused a globally trained
qv52 encoder; it remains a useful reranker diagnostic but is optimistic and is
not an unbiased end-to-end unseen-item estimate. The v44 full-data v37
checkpoint reaches only 41/81 (50.6%) Top-1 on v2 and 43/81 (53.1%) on v3
public validation, so it does not replace v35 for mixed/seen deployment. The
blind pool remains uninspected, and the 80% Top-1 product target is not yet
met. See the
[v42 strict-CV note](docs/progress_2026-07-03_v42_strict_cv.md) and
[v43 male-expansion note](docs/progress_2026-07-03_v43_male_expansion.md), and
[v44 item-page expansion note](docs/progress_2026-07-03_v44_item_page_expansion.md).
The v45 non-promotion is recorded in
[the v45 balance note](docs/progress_2026-07-03_v45_balance_guard.md). The
current stage outcome and next experiments are summarized in
[the v46 stage note](docs/stage_summary_2026-07-03_v46.md).

Training and accuracy validation use only source pages with real FFXIV
equipment labels. Arbitrary non-FFXIV images remain an inference aspiration,
not a supervised training or validation source. See
`docs/cross_domain_recognition_plan_2026-07-01.md`.

### Screenshot-to-Render Domain Adapter

Train and evaluate the slot-conditioned low-rank adapter on deterministic
`glamour_id` groups:

```powershell
python scripts\train_glamour_adapter.py --device auto
```

The command uses 30/10/10 glamour groups for training, validation, and held-out
testing. It also evaluates DINOv2/CLIP score fusion and a cascade that takes
Top-1 from the precision fusion model while filling the rest of Top-10 from the
higher-recall adapted DINOv2 model. The checkpoint is written under
`data/models/`; generated checkpoints and reports stay local.

### Human Parsing and Background Masks

Build both person-foreground and strict slot-mask DINOv2 query caches:

```powershell
python scripts\build_masked_glamour_crop_embeddings.py --device auto
```

Train the person-mask adapter without changing the clean-render candidate
index:

```powershell
python scripts\train_glamour_adapter.py `
  --crop-cache data\processed\glamour_crop_embeddings_facebook-dinov2-base-color-mask-person.npz `
  --checkpoint data\models\glamour_adapter_dinov2_base_mask_person.pt `
  --output data\processed\glamour_adapter_eval_dinov2_base_mask_person.json `
  --device auto
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
