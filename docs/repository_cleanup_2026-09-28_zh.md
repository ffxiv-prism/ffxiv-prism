# 2026-09-28 仓库清理审计

审计基线：本地 `dev` 与远程 `origin/dev` 为 `159dcad`，`origin/main` 为 `3b354f3`。两分支的数据与模型树相同；`dev` 额外包含报告 PNG 导出与打开本地目录功能。本次清理提交至 `dev`，不合并到 `main`。远程仅保留这两个分支。

## 范围与判断方法

核对 Git 跟踪文件、远程两个分支的文件树、本地 `data/`、`var/models/` 和 Hugging Face 缓存；检查 Python AST、全库调用与文档引用，并沿以下流程核对依赖：

- Web / CLI → `current_best_configs()` → `data/current_best_selection.json` / `data/current_best_assets.json`。
- `run_lifecycle_pipeline.py` → 采集、下载、质量审核、标签关联、SQLite、候选索引更新及元数据补充。
- `audit_slot_quality_folds.py` → `run_weight_refresh_training.py` → 评估、登记、发布。

日期旧或没有被主入口直接调用，不足以判定文件无用。保留训练、回归、手动审查和故障恢复脚本；`run_data_pipeline.py` 仍是内部阶段执行器和兼容入口。`base` 推断也继续使用当前候选索引。

## 已删除与合并

- 删除 4 个从未实现功能、全库无引用的占位包：`parsers`、`sources`、`sources/xivapi`、`tagging`，及其 4 个本地 `.pyc` 缓存。
- 删除 `scripts/.gitkeep`、`tests/.gitkeep` 和空白 `notebooks/.gitkeep`。
- 删除 15 个本地 `.orig` 备份和 1 个 `.rej`。拒绝补丁中的 HEAD crop 条件顺序修复已存在于正式源码。
- 删除 7 个全库无调用、无文档入口的函数：`infer_gender`、`infer_race`、`is_eorzea_url`、`is_probable_image_url`、`best_spatial_score_by_piece`、`device_summary`、`choose_device`。设备选择保留共用的 `resolve_torch_device`。
- 三个采集/下载脚本重复的 `format_duration` 合并至 `src/ffxiv_prism/progress.py`，原脚本通过导入继续提供该名称。
- 删除 13 个未使用导入，以及人体分割中计算后未使用的 `holes` 数组；实际填洞逻辑不变。

实际移除文件共 27 个，其中 7 个原先在 Git 中、20 个仅在本地。完整路径：

```text
src/ffxiv_prism/parsers/__init__.py
src/ffxiv_prism/sources/xivapi/__init__.py
src/ffxiv_prism/sources/__init__.py
src/ffxiv_prism/tagging/__init__.py
scripts/.gitkeep
tests/.gitkeep
notebooks/.gitkeep
scripts/build_item_disjoint_cv_split_manifests.py.orig
scripts/build_recognition_optimization_artifact_v68.py.orig
scripts/build_recognition_optimization_report.py.orig
scripts/build_slot_aligned_candidate_index.py.orig
scripts/build_slot_aligned_candidate_index.py.rej
scripts/evaluate_head_coverage_reranker.py.orig
scripts/evaluate_rank_ensemble_retrieval.py.orig
scripts/evaluate_retrieval_funnel.py.orig
scripts/train_glamour_pair_reranker.py.orig
scripts/train_glamour_query_encoder.py.orig
src/ffxiv_prism/cli/match.py.orig
src/ffxiv_prism/matching/coverage_profiles.py.orig
src/ffxiv_prism/matching/inference.py.orig
src/ffxiv_prism/matching/spatial_profiles.py.orig
tests/test_glamour_inference.py.orig
tests/test_item_disjoint_cv_split_manifests.py.orig
src/ffxiv_prism/parsers/__pycache__/__init__.cpython-312.pyc
src/ffxiv_prism/sources/__pycache__/__init__.cpython-312.pyc
src/ffxiv_prism/sources/xivapi/__pycache__/__init__.cpython-312.pyc
src/ffxiv_prism/tagging/__pycache__/__init__.cpython-312.pyc
```

## 数据与模型：全部保留

以下是需要人工复核的资产，不是可立即删除的清单。容量为本地文件逻辑大小，不代表可释放的 Git LFS 配额，也未对重复内容去重。没有改动任何数据、样本、索引或模型文件。逐文件清单保存在本地 `var/tmp/cleanup-audit-20260928/`，不提交可能含玩家图片路径的详细清单。

- `protected-files.csv`：全部受保护文件，包含路径、字节数、Git 跟踪状态、用途分类和保留原因。
- `review-candidates.csv`：历史模型/索引、旧样本划分/嵌入和数据备份的逐文件复核清单。
- `protected-groups.json`：按目录汇总。

### 历史候选索引

当前激活的是后缀 `20260926-134639` 的 10 组。下列 23 组（73 个文件，约 1.50 GiB）没有被当前推断配置选中；其中 20 组在远程仓库中、3 组仅本地。仍可能用于历史评估或回滚。

| 路径 | 文件数 | MiB | Git 跟踪文件数 |
| --- | ---: | ---: | ---: |
| `data/indexes/facebook-dinov2-base-feet-multiscale-v7-clean-20260730` | 3 | 69.72 | 3 |
| `data/indexes/facebook-dinov2-base-feet-multiscale-v7-clean-20260829-024239` | 3 | 69.94 | 3 |
| `data/indexes/facebook-dinov2-base-hands-multiscale-v5-20260730` | 3 | 79.77 | 3 |
| `data/indexes/facebook-dinov2-base-hands-multiscale-v5-20260829-024239` | 3 | 79.94 | 3 |
| `data/indexes/facebook-dinov2-base-head-multiscale-v4-20260730` | 3 | 46.63 | 3 |
| `data/indexes/facebook-dinov2-base-head-multiscale-v4-20260829-024239` | 3 | 46.65 | 3 |
| `data/indexes/facebook-dinov2-base-legs-multiscale-v6-20260730` | 3 | 69.82 | 3 |
| `data/indexes/facebook-dinov2-base-legs-multiscale-v6-20260828-210541` | 3 | 69.93 | 0 |
| `data/indexes/facebook-dinov2-base-legs-multiscale-v6-20260829-024239` | 3 | 70.11 | 3 |
| `data/indexes/facebook-dinov2-base-slot-aligned-v3-body-multiscale-20260730` | 3 | 112.70 | 3 |
| `data/indexes/facebook-dinov2-base-slot-aligned-v3-body-multiscale-20260828-210541` | 3 | 112.72 | 0 |
| `data/indexes/facebook-dinov2-base-slot-aligned-v3-body-multiscale-20260829-024239` | 3 | 113.09 | 3 |
| `data/indexes/facebook-dinov3-vitb16-body-multiscale-v3-20260730` | 3 | 29.07 | 3 |
| `data/indexes/facebook-dinov3-vitb16-body-multiscale-v3-20260829-024239` | 3 | 29.27 | 3 |
| `data/indexes/facebook-dinov3-vitb16-head-multiscale-v4-20260730` | 4 | 46.63 | 4 |
| `data/indexes/facebook-dinov3-vitb16-head-multiscale-v4-20260829-024239` | 4 | 46.65 | 4 |
| `data/indexes/facebook-dinov3-vitb16-legs-multiscale-v6-20260730` | 3 | 69.82 | 3 |
| `data/indexes/facebook-dinov3-vitb16-legs-multiscale-v6-20260828-210541` | 3 | 69.93 | 0 |
| `data/indexes/facebook-dinov3-vitb16-legs-multiscale-v6-20260829-024239` | 3 | 70.11 | 3 |
| `data/indexes/google-siglip2-base-patch16-224-feet-multiscale-v7-clean-20260730` | 3 | 69.72 | 3 |
| `data/indexes/google-siglip2-base-patch16-224-feet-multiscale-v7-clean-20260829-024239` | 3 | 69.93 | 3 |
| `data/indexes/google-siglip2-base-patch16-224-head-multiscale-v4-20260730` | 4 | 46.63 | 4 |
| `data/indexes/google-siglip2-base-patch16-224-head-multiscale-v4-20260829-024239` | 4 | 46.65 | 4 |

### 非当前默认模型

37 个文件，约 3.05 GiB，其中 7 个在远程 `data/models/current-best/` 中。`refresh_20260926_*` 是 18 个尚未晋升的训练结果，需要保留作对照；`refresh_v110_body_*_dinov3_*` 是未启用的 BODY 分支。名称相同的根目录训练权重和 `current-best` 推断权重可能保存不同字段，不能仅按名称去重。

| 路径 | MiB | Git 跟踪 |
| --- | ---: | --- |
| `data/models/appearance_disjoint_cv_v52_fold01_qv64_closed_e10.pt` | 38.33 | 否 |
| `data/models/appearance_disjoint_cv_v54_head_fold01_qv65_e10.pt` | 96.96 | 否 |
| `data/models/appearance_disjoint_cv_v59_hands_fold01_qv71_v5_e10.pt` | 96.96 | 否 |
| `data/models/appearance_disjoint_cv_v60_legs_fold01_qv72_e10.pt` | 96.96 | 否 |
| `data/models/appearance_disjoint_cv_v61_feet_fold01_qv73_e10.pt` | 96.96 | 否 |
| `data/models/appearance_disjoint_cv_v71_head_fold01_dinov3_e3.pt` | 96.95 | 否 |
| `data/models/appearance_disjoint_cv_v97_body_fold01_dinov3_e10.pt` | 96.95 | 否 |
| `data/models/dev_hands_v101_fold01_slot_quality_baseline_e10.pt` | 96.96 | 否 |
| `data/models/dev_hands_v106_fold01_pose_tube_mixed_e10.pt` | 96.96 | 否 |
| `data/models/refresh_20260926_body_fold01_dinov2_e13.pt` | 96.96 | 否 |
| `data/models/refresh_20260926_body_fold02_dinov2_e13.pt` | 96.96 | 否 |
| `data/models/refresh_20260926_body_fold03_dinov2_e13.pt` | 96.96 | 否 |
| `data/models/refresh_20260926_feet_fold01_dinov2_e13.pt` | 96.96 | 否 |
| `data/models/refresh_20260926_feet_fold02_dinov2_e13.pt` | 96.96 | 否 |
| `data/models/refresh_20260926_feet_fold03_dinov2_e13.pt` | 96.96 | 否 |
| `data/models/refresh_20260926_hands_fold01_dinov2_e13.pt` | 96.96 | 否 |
| `data/models/refresh_20260926_hands_fold02_dinov2_e13.pt` | 96.96 | 否 |
| `data/models/refresh_20260926_hands_fold03_dinov2_e13.pt` | 96.96 | 否 |
| `data/models/refresh_20260926_head_fold01_dinov2_e13.pt` | 96.96 | 否 |
| `data/models/refresh_20260926_head_fold01_dinov3_e4.pt` | 96.94 | 否 |
| `data/models/refresh_20260926_head_fold02_dinov2_e13.pt` | 96.96 | 否 |
| `data/models/refresh_20260926_head_fold02_dinov3_e4.pt` | 96.94 | 否 |
| `data/models/refresh_20260926_head_fold03_dinov2_e13.pt` | 96.96 | 否 |
| `data/models/refresh_20260926_head_fold03_dinov3_e4.pt` | 96.94 | 否 |
| `data/models/refresh_20260926_legs_fold01_dinov2_e13.pt` | 96.96 | 否 |
| `data/models/refresh_20260926_legs_fold02_dinov2_e13.pt` | 96.96 | 否 |
| `data/models/refresh_20260926_legs_fold03_dinov2_e13.pt` | 96.96 | 否 |
| `data/models/refresh_v110_body_fold01_dinov3_e10.pt` | 96.94 | 否 |
| `data/models/refresh_v110_body_fold02_dinov3_e10.pt` | 96.94 | 否 |
| `data/models/refresh_v110_body_fold03_dinov3_e10.pt` | 96.94 | 否 |
| `data/models/current-best/appearance_disjoint_cv_v52_fold01_qv64_closed_e10.pt` | 38.33 | 是 |
| `data/models/current-best/appearance_disjoint_cv_v54_head_fold01_qv65_e10.pt` | 38.33 | 是 |
| `data/models/current-best/appearance_disjoint_cv_v59_hands_fold01_qv71_v5_e10.pt` | 38.33 | 是 |
| `data/models/current-best/appearance_disjoint_cv_v60_legs_fold01_qv72_e10.pt` | 38.33 | 是 |
| `data/models/current-best/appearance_disjoint_cv_v61_feet_fold01_qv73_e10.pt` | 38.33 | 是 |
| `data/models/current-best/appearance_disjoint_cv_v71_head_fold01_dinov3_e3.pt` | 38.32 | 是 |
| `data/models/current-best/appearance_disjoint_cv_v97_body_fold01_dinov3_e10.pt` | 38.32 | 是 |

### 旧样本划分、裁剪与嵌入

共 5,803 个本地文件，约 36.00 MiB，未进入当前默认训练输入；历史实验可能仍引用，全部保留。详细图片文件名只在本地 CSV 中列出。

| 路径 | 文件数 | MiB |
| --- | ---: | ---: |
| `data/debug/outfit-regions/hands-pose-tube-w038-fold01-train-validation` | 5755 | 17.76 |
| `data/debug/outfit-regions/slot-quality-smoke` | 30 | 0.16 |
| `data/interim/appearance-disjoint-cv-v52-huiji-closed` | 3 | 0.12 |
| `data/interim/appearance-disjoint-cv-v54-feet-huiji-closed` | 3 | 0.12 |
| `data/interim/appearance-disjoint-cv-v54-hands-huiji-closed` | 3 | 0.12 |
| `data/interim/appearance-disjoint-cv-v54-head-huiji-closed` | 3 | 0.12 |
| `data/interim/appearance-disjoint-cv-v54-legs-huiji-closed` | 3 | 0.12 |
| `data/processed/glamour_query_embeddings_hands_v101_fold01_train_role.npz` | 1 | 8.75 |
| `data/processed/glamour_query_embeddings_hands_v101_fold01_validation_role.npz` | 1 | 4.36 |
| `data/processed/glamour_query_embeddings_hands_v102_fold01_pose_normalized.npz` | 1 | 4.36 |

### 数据断点和备份

下列 5 个文件约 35.02 MiB。`.tmp` 是可供 `--resume` 使用的下载断点，保持未提交状态；其他数据备份也未删除。

| 路径 | MiB |
| --- | ---: |
| `data/raw/eorzea_collection/metadata/gear_pieces_raw.jsonl.before_huiji_zh_names` | 11.41 |
| `data/raw/eorzea_collection/metadata/gear_pieces_raw.jsonl.before_lodestone_equip_notes` | 11.41 |
| `data/raw/eorzea_collection/metadata/gear_pieces_raw.jsonl.before_zh_names` | 11.39 |
| `data/raw/game_data/cafemaker_item_models_raw.jsonl.tmp` | 0.13 |
| `data/raw/game_data/cafemaker_item_names_raw.jsonl.tmp` | 0.68 |

### 已确认仍需要的旧名称资产

- 当前发布清单中的 45 个资产、10 组候选索引、6 个 query encoder 权重和 HEAD 形态模型。
- 当前三折续训所需的 18 个 `data/models/refresh_v110_*` 父权重；保留优化器状态以继续训练。
- `data/models/head_morphology_v73_fold01_qv65_upper_binary_ridge.npz`：当前形态模型的发布源文件。
- `data/processed/glamour_examples_huiji_models_closed.jsonl`、`slot_quality_eligibility_20260801.jsonl`、`training_visibility_reviews_20260801.jsonl`：当前训练数据。
- `data/interim/appearance-disjoint-cv-v55-slot-recovery-{slot}/` 和 `data/debug/outfit-regions/slot-quality-20260801/`：当前划分及裁剪。
- `front_fullbody_quality_final_v46.jsonl`、`slot_quality_recovery_reviews_20260801.jsonl`：质量恢复/训练准入输入。
- `data/interim/gearset_detail_queue.jsonl`：采集流程仍使用的队列。
- `data/samples/retrieval_eval*.jsonl` 四个清单：基础检索和增强评估仍使用。
- SQLite、装备原图、玩家原始截图、原始标签及基础模型缓存：属于当前流程或可追溯来源，未判定无用。

## 验证

- `python -m unittest discover -s tests -v`：538 项通过。
- `python scripts/recognize_glamour.py --check`：`ok: true`，五槽资产检查通过。
- `python scripts/export_current_best_assets.py --check`：发布文件大小及 SHA-256 校验通过。
- `ruff check src scripts --select F401,F811,F821,F841`：通过。
- `node --check web/app.js` 和 `git diff --check`：通过。

这次清理不修改推断排序、训练配置或发布资产，也不重写 Git 历史。远程旧资产按要求保留；今后如果决定缩减模型分发，需要单独处理 LFS 存储和发布策略。
