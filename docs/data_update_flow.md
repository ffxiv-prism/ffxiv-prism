# 图片优先的数据更新流程

## 唯一面向用户的入口

网页“更新装备图片库”和 `python scripts/run_lifecycle_pipeline.py` 完全相同：
`sample-library + refresh + extend + device auto`。普通更新不训练或发布模型。
网页“修复本地图片库与索引”使用 `--source-mode local`，不采集远程图片或资料。
旧网页 API 的 full 模式仍兼容，但等价于 standard，不再显示第三个选项。

`run_data_pipeline.py` 的默认和 `--all` 调用转交统一入口；显式 `--stage` 保留为
底层执行器。过去网页更新不下载图片、不更新候选索引的行为已取消。

## 图片主线与完成语义

1. 按增量预算刷新装备来源页，从中发现图片种子；仅首次初始化自动全扫。
2. 构造并验证新图片 URL，保留历史成功记录；历史失败 URL 按冷却期和预算重试。
3. 下载新增/缺失图片，完整解码检查缓存并重下损坏文件，输出本次下载统计。
4. 从装备详情页关联装备与部位标签。缺标签图片保留待处理，不进入正式候选。
5. 按来源去重，进行完整像素解码与尺寸校验，原子构建 SQLite。
6. 生成逐图片样本清单：eligible / duplicate / pending_labels / quality_rejected /
   unsupported_view。相同字节与相同装备标签的图片不重复新增；不同装备标签保留。
7. 只为合格且尚未在当前活动索引中的图片提取特征。统计图片数时跨模型/裁剪去重，
   不把多条向量误报为多张图片。
8. 所有分支结构校验通过后原子更新活动索引映射；任一分支失败则保留旧配置。
9. 后置补充中文名、染色、官方槽位/限制和同模资料。阶段失败只记录警告，
   不撤销已完成的图片检索更新。

中文名、染色能力不是图片准入前提。当前可用图片需要能够解码、有支持的部位，
并能关联装备标签；正常未染色视图沿用现有检索策略。

## 增量扫描与请求预算

默认 `--discovery-mode auto`。有历史发现清单、详情快照或 SQLite 装备记录即可视为
已有库，不要求本机有 HTML 缓存。另一台机器缺少扫描状态时也不会因此自动全扫。
完全没有装备记录时才进行首次全扫；中断的首次扫描继续初始化，而不是把半成品当完整库。

每条 male/female 列表流独立执行：

1. 从第 1 页开始，至少检查 3 页；连续 2 页的装备链接与图片种子均已知或页面指纹未变，
   就停止首页段，最多检查 8 页。相同装备出现新图片也算变化。
2. 再从保存的历史游标复查 3 页，与本次已访问页去重。下次接着往后查。
   连续空页或重复末页达到阈值后回到第 4 页；最大页数为 500。
3. 每个有效页面之后保存装备清单、详情队列和扫描状态。局部扫描只合并历史，
   不删除本次未看到的装备。HTTP 失败、验证页或无法识别的空响应不会提交该页进度。
4. 首次全扫也按页保存恢复点；达到 500 页仍未找到末尾则返回未完成，
   必须通过底层扫描器提高 `--max-pages` 继续，不能误报初始化成功。

默认每次通常共 **12 页**（每类 3 页首页 + 3 页历史），最多 **22 页**；
这是列表页面预算，不是全流程 HTTP 请求总数。新图片验证、下载和缺失详情另计。
未验证远端排序稳定性，因此仍保留历史轮查。较旧页面新增图片可能要等轮查到该页
才能发现；首页新增超过 8 页时也会分次覆盖。增量更新不代表远端全站已同步完成。

扫描状态为 `var/cache/eorzea_discovery/state.json`（本机缓存，不提交 Git）。
同目录 `last_scan_summary.json` 记录扫描模式、实际页码、远端/缓存页数、下一历史页、
新增装备数及初始化是否完成。删除该状态会重置轮查进度，但已有资料仍避免自动全扫。

图片验证合并局部列表产生的新 URL 和全部历史检查 URL，避免旧图片族退出重试范围。
成功记录直接复用；新 URL 立即验证；历史失败默认每次最多重试 **200 个**，
HTTP 404/410 冷却 **7 天**，其他失败冷却 **6 小时**，到期者按最早检查时间优先。
浏览器验证同样遵守历史重试预算。未到期或超预算的记录保留以后处理，
统计中包含 `historical_retries`、`deferred_historical_retries`、`new_url_count`。
详情采集继续使用 `--resume`，只请求缺失详情；下载继续复用有效本地图片。

日常更新无需增加参数。调整扫描预算示例：

```powershell
python scripts/run_lifecycle_pipeline.py --discovery-head-pages 3 --discovery-max-head-pages 8 --discovery-history-pages 5
```

确有需要时手动核对全部列表（不是另一个更新流程）：

```powershell
python scripts/run_lifecycle_pipeline.py --discovery-mode full
```

`run_data_pipeline.py` 兼容入口支持相同参数。`--source-mode refresh` 表示刷新选中的页面，
**不再代表遍历全部列表**。底层图片验证器提供 `--retry-limit`、`--missing-retry-days`、
`--error-retry-hours` 维护参数，`--retry-limit 0` 禁止历史重试而仍允许新 URL 验证。
这些变化减少来源发现的远端请求；本地图片质检、索引整理及后置物品资料同步仍有自己的耗时。

## 检索索引与模型

索引来源从 `data/current_best_selection.json` 的 `candidate_indexes` 映射读取；
没有映射的分支使用既有 `candidate_suffix`。更新前保存来源清单快照，避免续跑
从已激活的新索引错误地开始。新索引使用版本目录，无新增的分支复用原目录。
推理读取同一映射，所以完成激活后新增图片可以参与检索。

激活仅改变候选索引，保留模型权重与模型策略；旧评估指标置为 refresh_required，
不能当作新候选库的评估结果继续发布。训练候选来源写入
`data/processed/equipment_image_samples.jsonl`；训练状态为 pending_review_and_split。
未执行独立训练准入时，`training_admitted_images` 为 null，不能推断为已训练或准入为零。

## 结果与恢复

JSON 报告 version 3 保留阶段状态和 image_update：

| 字段 | 含义 |
| --- | --- |
| new_discovered_images | 相对更新前清单新发现的可访问正常视图 URL |
| images_downloaded / download_failures | 本次真实下载成功/失败数（未执行为 null） |
| new_local_images | 相对基线新增的本地图片来源数 |
| missing_download_images | 已发现但本地仍缺少文件的图片数 |
| new_qa_ready_images / new_eligible_images | 新通过质检/满足检索准入的图片数 |
| new_indexed_images | 本次激活后新增可检索的唯一图片数，跨分支去重 |
| pending_label_images | 有正常图片但缺装备标签的图片数 |
| quality_rejected_images / duplicate_images | 质量不合格/重复图片数 |
| training_candidate_images | 可供后续训练审核的合格图片数 |
| training_admitted_images | 独立训练准入未执行，因此为 null |
| outcome / message | 新样本已检索、待索引、无新增有效样本、失败或仅计划 |

下载不完整或资料补全失败显示待处理提示；流程可让其他合格图片先参与检索。
核心图片/索引阶段失败返回失败状态。无新增有效图片时不能称为样本库扩充。

每个阶段完成后原子保存报告。恢复时复用通过且命令相同的阶段，保留原始图片
基线、来源快照和输出后缀；阶段输入变化或指定 --restart-stage 后，后续阶段重跑。
完整性校验会拒绝使用未完成的索引目录。旧 v2 报告不会跳过新版图片链路。

`local` 离线校验现有图片并补齐索引；`cached` 复用目录与物品缓存；`refresh`
发现远端新图片并更新补充资料。check/skip 为维护参数，不承诺新增图片已可检索。

## 本次新增数据字段

采集源：Item 表的 `DyeCount`。API 字段选择及分页遵循
[XIVAPI 官方 Reading Sheets 文档](https://v2.xivapi.com/docs/guides/sheets/)。

主要输出（均为现有的 Git 跟踪文件）：

- `data/raw/game_data/cafemaker_item_names_raw.jsonl`：完整物品资料。
- `data/raw/game_data/cafemaker_item_names_for_gear_pieces_raw.jsonl`：装备名称精简资料。
- `data/raw/game_data/cafemaker_item_names_summary.json`：采集来源、版本及覆盖统计。

每条物品记录现在包含：

| 字段 | 来源或含义 | 本次新增 |
| --- | --- | --- |
| `game_item_id` | API `row_id`，物品 ID | 否 |
| `name_en` | `Name@lang(en)`，英文名 | 否 |
| `name_zh` | `Name@lang(chs)`，中文名 | 否 |
| `equip_slot_category_id` | `EquipSlotCategory@as(raw)`，槽位类别编号 | 否 |
| `item_ui_category_en` | `ItemUICategory@lang(en).Name`，物品分类英文名 | 否 |
| `source` | 采集接口地址 | 否 |
| `dye_count` | `DyeCount`，非负整数；0 不可染色、正数为通道数、null 未知 | **是** |

采集摘要新增三个字段（可装备记录按槽位类别编号 > 0 统计）：

| 字段 | 含义 |
| --- | --- |
| `wearable_rows_with_dye_count` | 有有效染色通道数的可装备记录数 |
| `wearable_rows_missing_dye_count` | 缺少有效染色通道数的可装备记录数 |
| `dyeable_wearable_rows` | 染色通道数 > 0 的可装备记录数 |

已有的 `fields` 请求字段列表追加 `DyeCount`；`schema`、`game_version`、
`finished_at`、`source` 等元数据继续记录上游版本与采集时间。
不存储冗余 `dyeable` 布尔值，也不采集玩家实际使用的染料颜色。


## GitHub 与本机核对记录（修改前，2026-09-23）

本机与 GitHub 开发分支 codex/recognition-handoff-20260629 均为 866eb85；
GitHub 默认分支 main 为 c52686b，仍使用旧 CafeMaker 接口且缺少生命周期编排。
以上为修改前的实时核验结果。统一流程、染色采集与增量扫描随本次开发分支提交发布；
另一台机器需拉取包含这些修改的分支。开发分支的更新不会自动合并到默认分支 main。
