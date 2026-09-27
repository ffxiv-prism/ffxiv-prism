# FFXIV Prism

FFXIV Prism 是一个本地优先的《最终幻想 XIV》装备外观候选检索工具。输入一张单人角色截图，系统按 `HEAD`、`BODY`、`HANDS`、`LEGS`、`FEET` 五个部位裁剪并返回装备候选。

当前版本适合作为“Top-10 候选 + 人工确认”的本地 Demo，不应描述为自动装备鉴定器。2026-09-26 数据更新后已完成 18 个分折续训任务和新旧权重对比，但新权重均未通过分组不退步门槛，现有最佳权重保持不变。只有 BODY 达到 Top-10 80%；HANDS 仍是首要瓶颈。详见[本轮训练与评测结论](docs/weight_refresh_2026-09-26_zh.md)。

## 快速开始

仓库通过 Git LFS 提供 `current-best` 的候选向量索引、推理检查点和本地
DINOv3 权重。首次 clone 前需安装 Git LFS；若已经 clone，再执行 `git lfs pull`：

```powershell
git lfs install
git clone https://github.com/ffxiv-prism/ffxiv-prism.git
cd ffxiv-prism
git lfs pull
```

已有 Python 3.12 环境时，安装项目并准备可从官方模型仓下载的公共权重：

```powershell
python -m pip install -e .
python scripts\prepare_current_best.py
```

公共 DINOv2、SigLIP2 和姿态模型仅在首次准备时下载到项目内的
`var/cache/huggingface`。仓库内的推理资产会按 SHA-256 校验；若提示 Git LFS
pointer，请先运行 `git lfs pull`。

下载 `current-best` 报告预览所需的装备图片：

```powershell
python scripts\download_eorzea_images_from_url_manifest.py `
  --current-best-only `
  --image-workers 8 `
  --delay 0.2
python scripts\download_eorzea_images_from_url_manifest.py --check
```

CPU-only 环境可使用 `environment-cpu.yml` 创建；完整五部位 CPU 推理会明显更慢。

识别一张图片：

```powershell
python scripts\recognize_glamour.py reference.png `
  --profile current-best `
  --device cpu `
  --top-k 10 `
  --output-dir data\outputs\glamour-match\reference
```

只识别指定部位：

```powershell
python scripts\recognize_glamour.py reference.png `
  --slots HEAD BODY FEET `
  --top-k 10
```

输出目录包含：

- `report.html`：带候选预览图的结果页；
- `report.png`：自动将原始 HTML 报告以 1280 像素桌面宽度导出为完整长图，保留相同样式；
- `results.json`：完整结构化结果；
- `overlay.jpg`：姿态点与部位包络；
- `crops.jpg` 和 `crops/`：实际查询裁剪。

PNG 和 HTML 默认保存在同一输出目录。导出使用本机 Chrome、Edge 或 Chromium 的
无界面浏览器，等待字体和图片加载后截图，不需要另外安装浏览器驱动。
浏览器不可用或导出失败时仍保留 HTML/JSON，工作台和日志会显示失败原因。
已有报告也可单独补导出，无需重新推断：

```powershell
python scripts/export_report_png.py data/outputs/glamour-match/reference/report.html
```

工作台识别完成后，“打开原始报告”旁的“打开本地目录”会打开该报告所在文件夹，
可直接找到 HTML、PNG 和 JSON。该按钮需在运行工作台的本机浏览器中使用。

候选装备存在同模装备时，HTML 和 JSON 报告会列出其名称。同模关系快照位于
`data/processed/piece_appearance_equivalence_huiji_exact_models.json`，属于随仓库
发布并校验的运行资产，无需克隆用户另行生成。

## 本地网页

项目自带无需前端构建工具的本地工作台：

```powershell
python scripts\web_app.py
```

默认地址为 `http://127.0.0.1:8765`。网页支持点击选择、拖拽或按 `Ctrl+V` 从剪贴板粘贴图片，并提供五部位选择、Top-k 设置、CPU/GPU 运行环境选择、结果查看和本地数据更新。服务只监听回环地址；如不希望自动打开浏览器，添加 `--no-open`。

WebUI 会检测启动它的 Python/PyTorch 环境：GPU 环境可在每次任务前手动选择 GPU 或 CPU；CPU-only 环境会禁用 GPU 选项并显示原因。选择 CPU 时，推断子进程会显式使用 `--device cpu` 并隐藏 CUDA；选择 GPU 时，后端会先确认 CUDA 可用，再使用 `--device cuda`，不会静默回退到 CPU。

## 最近一次注册严格指标

口径为玩家搭配图上的严格外观不相交三折 pooled Appearance Recall，候选按官方同外观组去重。
本次在相同更新后候选库上对比新旧权重，使用既有验证集选择模型，不是独立盲测；
不含风格重排、无装备判断和 HEAD 局部召回。与旧版未去重的注册值不可直接比较。

| 部位 | 样本数 | Top-1 | Top-10 | 状态 |
| --- | ---: | ---: | ---: | --- |
| BODY | 1,013 | 59.72% | 85.00% | Top-10 达标；保留原权重 |
| HEAD | 927 | 28.69% | 52.64% | 未达标；保留原权重 |
| HANDS | 959 | 3.02% | 10.01% | 最差、优先处理；保留原权重 |
| LEGS | 970 | 12.89% | 33.51% | 未达标；保留原权重 |
| FEET | 930 | 20.32% | 45.48% | 未达标；保留原权重 |

机器可读快照位于 `data/processed/current_recognition_metrics.json`，已登记为
`release_ready`，对应候选快照 `20260926-134639`。目标仍是所有部位 Top-10 ≥ 80%；
本次经明确的 `--allow-below-target` 发布现有最佳模型，不用 Top-1000 候选覆盖代替
Top-10 成绩。

## 训练数据准入

新的训练不得直接读取任意历史中间文件。规范本地输入固定为：

- 标签：`data/processed/glamour_examples_huiji_models_closed.jsonl`
- 逐槽准入清单：`data/processed/slot_quality_eligibility_20260801.jsonl`
- 可见性复核：`data/processed/training_visibility_reviews_20260801.jsonl`
- 裁剪：`data/debug/outfit-regions/slot-quality-20260801`
- 五槽严格三折：`data/interim/appearance-disjoint-cv-v55-slot-recovery-{slot}`

这些玩家派生资产只保留在获授权的本机环境，不随代码仓库上传。

开始训练前必须由审计确认：

1. 被选图片文件存在、可解码，且最终质量清单为通过；
2. 每个目标槽位有非空闭包标签；
3. 至少一个真值 piece ID 存在于该槽位当前活动候选索引；
4. glamour、图片内容和外观身份在训练/验证之间无泄漏；
5. 样本量、外观多样性和关键可见性切片达到预先登记的门槛。

任何一项未通过都应先修数据或采集，不得用继续扫参掩盖数据问题。

运行当前训练准入审计：

```powershell
python scripts\audit_slot_quality_folds.py
```

该命令在未通过时返回非零退出码，并把本地证据写到
`data/outputs/slot-quality-fold-audit-20260801/summary.json`。当前审计状态为
`ready_for_training`：五槽三折无身份或内容泄漏，裁剪与候选覆盖完整。

## 数据库与候选索引

规范装备库为 `data/processed/eorzea.sqlite`。2026-09-26 生命周期快照包含
2,460 个套装、11,815 个装备部件和 40,076 张图片，其中 40,074 张通过 QA；
发布预览所需的 19,826 张图片全部可用。玩家逐槽准入集的五部位标签均已达到
100% 活动候选覆盖。

## 更新装备图片库

网页和命令行使用同一套图片优先流程。默认执行远端图片发现与采集、质量校验、
必要标签关联和检索索引扩展，随后补充名称、染色等展示资料，不自动训练或发布模型：

```powershell
python scripts/run_lifecycle_pipeline.py
```

网页只提供“更新装备图片库”和“修复本地图片库与索引”。命令行离线修复：

```powershell
python scripts/run_lifecycle_pipeline.py --source-mode local
```

流程依次执行：来源发现 → 图片 URL 发现与验证 → 图片下载 → 装备/部位标签关联
→ 图片去重与完整解码检查 → SQLite → 样本准入清单 → 特征提取与检索索引激活
→ 中文名、染色、装备限制和同模关系补充。现有装备新增的图片同样会被发现；
历史不可用 URL 按冷却期和每次最多 200 个的预算重试，损坏的本地缓存图片会重新下载。
报告补充资料失败只记录警告，不阻断合格图片的检索更新。

来源扫描默认 `--discovery-mode auto`：仅无历史装备记录时自动全扫；已有库即使缺少
本机网页缓存也采用增量。每类列表检查 3–8 页首页，再轮查 3 页历史，男女合计通常
12 页、最多 22 页。游标按页持久化，较旧页面的新增图片可能延后至轮查时发现。
需要人工全量核对时使用 `--discovery-mode full`。扫描预算及图片重试规则见
[增量扫描说明](docs/data_update_flow.md#增量扫描与请求预算)。

候选扩展以当前活动索引为基础，不再固定从旧日期索引开始。所有分支完成并验证后，
通过 `data/current_best_selection.json` 的 `candidate_indexes` 映射一次切换；
无新增图片的分支继续使用原索引。模型权重保持不变，旧评估指标标记为需要刷新。
任一分支失败时不切换，已有索引仍可用；SQLite 也在新库构建成功后才原子替换。

`data/processed/equipment_image_samples.jsonl` 记录每张图片是否合格、重复、缺标签、
质量失败或不属于当前支持的视图。缺少中文名或染色字段不会影响图片准入。
合格图片同时成为训练候选，但仍需质量审核、防泄漏划分和训练流程；
**图片入库不代表模型已经训练，也不代表通过了训练准入。**

结束报告列出新发现、下载成功、下载失败、质检通过、新增合格样本、实际新增检索图片，
以及待标签和重复图片数量。无新增可用图片会明确显示“无新增有效检索样本”，
不会把单纯元数据更新称为样本扩充。JSON 报告及各阶段证据位于
`data/outputs/lifecycle-reports/`，每完成一个阶段都会保存恢复点。

中断后恢复（沿用来源快照、更新前图片基线和候选索引后缀）：

```powershell
python scripts/run_lifecycle_pipeline.py --resume-report data/outputs/lifecycle-reports/<上次运行时间>.json
```

恢复时应使用与原任务一致的来源和候选模式。`--restart-stage <阶段名>` 可从指定阶段
重做。旧版本流程报告会重新执行新流程，避免跳过新增图片准入与索引激活步骤。

高级参数：`--source-mode cached` 复用目录缓存并下载缺失图片；
`--candidate-mode check` 只检查新增候选，`skip` 不更新索引；两者都不保证新增图片
可检索。`--device auto` 为默认，可指定 `cpu` / `cuda`。
`--dry-run` 只显示计划。`run_data_pipeline.py` 的无参数和 `--all` 旧调用转交统一入口，
显式 `--stage` 仅用于内部执行或单阶段维护。

### 染色资料

refresh 更新在图片流程之后请求 Item 的 `DyeCount`，写入受 Git 管理的
`data/raw/game_data/cafemaker_item_names_raw.jsonl`，装备精简表也保留 `dye_count`。
0 表示不可染色，正数为通道数，null 表示未知；旧 CSV 仅作为旧数据兼容回退。
仅补充报告资料时可单独运行：

```powershell
python scripts/download_cafemaker_item_names.py
python scripts/filter_cafemaker_item_names_for_gear_pieces.py
```

### 模型训练与发布

训练与数据更新独立：更新候选图片和索引不会自动更新查询模型权重。
本地保存了上一轮各分折完整训练检查点、玩家标签、可见性审核和裁剪时，可执行续训：

```powershell
python scripts/audit_slot_quality_folds.py --output data/outputs/model-refresh-20260926/fold-audit.json
python scripts/run_weight_refresh_training.py --output-dir data/outputs/model-refresh-20260926 --fold-audit data/outputs/model-refresh-20260926/fold-audit.json --prefix refresh_20260926 --execute
python scripts/evaluate_weight_refresh.py --run-dir data/outputs/model-refresh-20260926
python scripts/register_weight_refresh.py --run-dir data/outputs/model-refresh-20260926 --release-id 20260926-weight-refresh --candidate-suffix 20260926-134639 --evidence-output data/processed/weight_refresh_20260926.json
```

每次新实验使用新的输出目录和检查点前缀；登记时的候选后缀必须对应实际激活的索引。
训练固定为各分折 DINOv2 续训 3 轮、HEAD DINOv3 续训 1 轮，沿用该分折优化器；
不跨折复用权重，不根据验证结果搜索训练轮数。输入哈希或已完成检查点变化时停止恢复。
新旧权重在同一候选快照上重新评测，按官方同外观组去重后计算身份召回。
只有 pooled Top-1 或 Top-10 提升，且每个分折、种族、性别分组的 Top-1/Top-10
均不下降，才替换相应分支；其余分支保留原权重。这里使用既有验证集选择模型，
不是独立盲测，也不包含风格重排、无装备判断和 HEAD 局部召回等额外策略。
评测明细和训练产物留在本地，仓库仅保存汇总证据与选中的推理资产。

重新评估并登记有效指标后，才可执行模型发布门禁：

```powershell
python scripts/run_lifecycle_pipeline.py --workflow model-release --release-mode check
```

显式 `--release-mode publish` 才导出资产。`--workflow all` 为高级串联模式；
如果刚扩展图片库而未重新评估，发布门禁会因指标过期阻止发布。
公开模型与图片缓存仍由各主机准备；已发布运行资产由 Git LFS 提供。

具体字段、恢复语义和分支核对记录见[数据更新流程说明](docs/data_update_flow.md)。

## 项目结构

```text
src/ffxiv_prism/        核心数据、视觉与推理代码
scripts/                采集、构建、训练、评估和本地应用入口
tests/                  标准库 unittest 测试
web/                    无构建步骤的本地网页
docs/                   当前合并状态文档
data/processed/         规范小型清单与装备库快照
data/interim/           本地采集队列、严格三折清单和历史划分（忽略）
data/indexes/           当前及保留的历史发布索引随 Git LFS 管理；其他索引忽略
data/models/            current-best 当前及历史推理检查点随 Git LFS 管理；训练检查点忽略
data/debug/             训练裁剪、人工审核和调试输出（忽略；清理前核对依赖）
data/outputs/           运行报告、训练计划及评估证据（忽略；清理前核对依赖）
var/                    模型运行时与缓存（忽略）
```

已删除的旧代码与保留资产清单见[仓库清理审计](docs/repository_cleanup_2026-09-28_zh.md)。

## 验证

```powershell
python -m unittest discover -s tests -v
python scripts\recognize_glamour.py --check
```

测试必须在提交前通过；资产检查必须在本机运行资产存在时通过。

## 约束

- 项目只处理单主角、清晰度足够的 FFXIV 风格图片；
- 染色、遮挡、动作、宽檐头饰、小型饰品和多槽占用会显著影响排序；
- 排名分数是相对检索信号，不是校准概率；
- 装备图片、训练数据、训练检查点、调试输出和公共模型缓存不进入 Git；`current-best` 推理资产通过 Git LFS 提供。仓库还保留历史发布资产，当前实际使用范围以 `data/current_best_assets.json` 为准。
当 Eorzea Collection 出现 Cloudflare 验证时，可启用可见浏览器。流程会先用
普通 Chrome 打开验证页；手工验证成功后回到命令行按 Enter，程序才会接管同一
会话。后续的页面缓存、图片下载、QA、SQLite、候选索引和发布门禁会自动继续：

```powershell
python scripts/run_lifecycle_pipeline.py --workflow all --source-mode refresh --browser-verification --candidate-mode extend --release-mode publish --device cuda
```
