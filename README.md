# FFXIV Prism

FFXIV Prism 是一个本地优先的《最终幻想 XIV》装备外观候选检索工具。输入一张单人角色截图，系统按 `HEAD`、`BODY`、`HANDS`、`LEGS`、`FEET` 五个部位裁剪并返回装备候选。

当前版本适合作为“Top-10 候选 + 人工确认”的本地 Demo，不应描述为自动装备鉴定器。最近一次已注册的严格物品不相交三折评估中，只有 BODY 达到 Top-10 80%；其余部位仍有明显差距，HANDS 是首要瓶颈。2026-07-30 修复候选覆盖后评估分母已变化，下表只作为历史基准，必须重跑后才能登记为当前成绩。完整现状、训练准入审计和根因证据见 [当前状态](docs/current_status_2026-07-30_zh.md)。

## 快速开始

仓库通过 Git LFS 提供 `current-best` 的候选向量索引、推理检查点和本地
DINOv3 权重。首次 clone 前需安装 Git LFS；若已经 clone，再执行 `git lfs pull`：

```powershell
git lfs install
git clone https://github.com/cool2341999/ffxiv-prism.git
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
- `results.json`：完整结构化结果；
- `overlay.jpg`：姿态点与部位包络；
- `crops.jpg` 和 `crops/`：实际查询裁剪。

## 本地网页

项目自带无需前端构建工具的本地工作台：

```powershell
python scripts\web_app.py
```

默认地址为 `http://127.0.0.1:8765`。网页支持点击选择、拖拽或按 `Ctrl+V` 从剪贴板粘贴图片，并提供五部位选择、Top-k 设置、CPU/GPU 运行环境选择、结果查看和本地数据更新。服务只监听回环地址；如不希望自动打开浏览器，添加 `--no-open`。

WebUI 会检测启动它的 Python/PyTorch 环境：GPU 环境可在每次任务前手动选择 GPU 或 CPU；CPU-only 环境会禁用 GPU 选项并显示原因。选择 CPU 时，推断子进程会显式使用 `--device cpu` 并隐藏 CUDA；选择 GPU 时，后端会先确认 CUDA 可用，再使用 `--device cuda`，不会静默回退到 CPU。

## 最近一次注册严格指标

口径为玩家搭配图上的严格物品不相交三折 pooled Appearance Recall：

| 部位 | 样本数 | Top-1 | Top-10 | 状态 |
| --- | ---: | ---: | ---: | --- |
| BODY | 987 | 63.53% | 82.88% | Top-10 达标 |
| HEAD | 899 | 27.36% | 47.50% | 未达标 |
| HANDS | 945 | 3.07% | 8.78% | 最差、优先处理 |
| LEGS | 955 | 13.93% | 31.94% | 未达标 |
| FEET | 967 | 19.34% | 41.16% | 未达标 |

机器可读快照位于 `data/processed/current_recognition_metrics.json`，并已标记为 `refresh_required_after_candidate_coverage_repair`。目标是所有部位 Top-10 ≥ 80%，不是用 Top-1000 候选覆盖代替 Top-10 成绩。

## 训练数据准入

新的训练不得直接读取任意历史中间文件。规范本地输入固定为：

- 标签：`data/processed/glamour_examples_huiji_models_closed.jsonl`
- 最终质量清单：`data/processed/front_fullbody_quality_final_v46.jsonl`
- BODY 严格三折：`data/interim/appearance-disjoint-cv-v52-huiji-closed`
- 其他部位严格三折：`data/interim/appearance-disjoint-cv-v54-{slot}-huiji-closed`

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
python scripts\audit_glamour_training_readiness.py
```

该命令在未通过时返回非零退出码，并把本地证据写到 `data/outputs/training-readiness-audit-20260730`。当前仍因逐槽可见性复核不完整和候选修复后的评估分母尚未重跑而阻塞训练。

## 数据库与候选索引

规范装备库为 `data/processed/eorzea.sqlite`。2026-07-30 快照包含 2,443 个套装、11,739 个装备部件和 19,810 张已关联槽位图；活动索引引用缺图为 0。玩家最终质量集的五部位标签均已达到 100% 活动候选覆盖。

更新本地数据：

```powershell
python scripts\run_data_pipeline.py
```

强制刷新远端目录发现：

```powershell
python scripts\run_data_pipeline.py --refresh-source
```

装备图片、远端页面、公共模型缓存、训练缓存、调试图与运行输出均属于本地可再
生成资产，不进入 Git。已发布 `current-best` 所需的候选向量索引、推理检查点和
DINOv3 权重属于版本化运行资产，通过 Git LFS 随仓库提供；文件清单和哈希位于
`data/current_best_assets.json`。

## 项目结构

```text
src/ffxiv_prism/        核心数据、视觉与推理代码
scripts/                采集、构建、训练、评估和本地应用入口
tests/                  标准库 unittest 测试
web/                    无构建步骤的本地网页
docs/                   当前合并状态文档
data/processed/         规范小型清单与装备库快照
data/interim/           仅保留登记的严格三折清单
data/indexes/           current-best 索引随 Git LFS 发布；其他索引忽略
data/models/            current-best 推理检查点随 Git LFS 发布；训练检查点忽略
data/debug/             可删除调试输出（忽略）
data/outputs/           可删除运行输出（忽略）
var/                    模型运行时与缓存（忽略）
```

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
- 装备图片、训练数据、训练检查点、调试输出和公共模型缓存不进入 Git；仅发布清单中登记的 `current-best` 推理资产通过 Git LFS 提供。
