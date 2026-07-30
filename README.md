# FFXIV Prism

FFXIV Prism 是一个本地优先的《最终幻想 XIV》装备外观候选检索工具。输入一张单人角色截图，系统按 `HEAD`、`BODY`、`HANDS`、`LEGS`、`FEET` 五个部位裁剪并返回装备候选。

当前版本适合作为“Top-10 候选 + 人工确认”的本地 Demo，不应描述为自动装备鉴定器。最近一次已注册的严格物品不相交三折评估中，只有 BODY 达到 Top-10 80%；其余部位仍有明显差距，HANDS 是首要瓶颈。2026-07-30 修复候选覆盖后评估分母已变化，下表只作为历史基准，必须重跑后才能登记为当前成绩。完整现状、训练准入审计和根因证据见 [当前状态](docs/current_status_2026-07-30_zh.md)。

## 快速开始

推荐使用项目 Conda 环境：

```powershell
conda env create -f environment-gpu.yml
conda activate ffxiv-prism
python scripts\recognize_glamour.py --check
```

CPU 环境可改用 `environment-cpu.yml`，但完整五部位推理会明显更慢。

识别一张图片：

```powershell
python scripts\recognize_glamour.py reference.png `
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

默认地址为 `http://127.0.0.1:8765`。网页支持拖拽上传、五部位选择、Top-k 设置、结果查看和本地数据更新。服务只监听回环地址；如不希望自动打开浏览器，添加 `--no-open`。

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

远端页面和图片、模型、向量索引、训练缓存、调试图与运行输出均属于本地可再生成资产，不进入 Git。

## 项目结构

```text
src/ffxiv_prism/        核心数据、视觉与推理代码
scripts/                采集、构建、训练、评估和本地应用入口
tests/                  标准库 unittest 测试
web/                    无构建步骤的本地网页
docs/                   当前合并状态文档
data/processed/         规范小型清单与装备库快照
data/interim/           仅保留登记的严格三折清单
data/indexes/           本地活动向量索引（忽略）
data/models/            本地训练检查点（忽略）
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
- `data/raw`、`data/indexes`、`data/models`、`data/debug`、`data/outputs` 与 `var` 中的大型资产不适合上传代码仓库。
