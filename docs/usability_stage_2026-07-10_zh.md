# FFXIV Prism 可用性阶段总结（2026-07-10）

## 总体目标

项目的长期目标是：输入一张参考人物/穿搭图，理解五个核心装备槽位，并从本地、可追溯的 FFXIV 装备数据中推荐视觉相似的幻化候选，附带装备名、预览、来源和可靠性说明。

当前可诚实交付的范围是“FFXIV 单人物截图的本地候选检索工具”，不是对任意插画或现实服装都能高准确率识别的成品服务。

## 审计前进展

| 阶段 | 状态 | 结论 |
|---|---|---|
| Phase 0 本地工程基础 | 完成 | Python 3.12 CPU/GPU 环境、路径、缓存和测试已建立 |
| Phase 1 装备元数据 | 完成 | Eorzea Collection 主数据和中英文名称补全管线已建立 |
| Phase 2 视觉资产 | 完成 | 标准装备渲染图、玩家截图和可恢复采集清单已建立 |
| Phase 3 数据归一化 | 完成 | JSONL、QA 索引和 SQLite 数据集已建立 |
| Phase 4 视觉特征 | 部分完成 | 五槽 DINOv2 索引和多尺度候选已建立，属性标签仍不完整 |
| Phase 5 截图理解 | 部分完成 | 单人物 ViTPose 与五槽多裁剪已建立，不支持稳定多人选择 |
| Phase 6 幻化匹配 | 研究能力已有、用户入口缺失 | 有大量评估脚本，但审计前没有单图五槽统一命令 |
| Phase 7 评估迭代 | 持续中 | 严格物品不相交评估已建立，非 BODY 准确率仍低 |

审计基线为 133 个 `unittest` 全部通过。本地已有约 1.47 GB 候选索引、9.78 GB 模型和完整姿态裁剪产物。

## 本轮阶段目标与完成情况

### 阶段 1：目标与可用性基线

验收条件：明确研究能力、产品缺口和不应夸大的指标。

结果：完成。主要缺口确定为缺少“单张截图 → 五槽裁剪 → 各槽模型/索引 → 候选融合 → 可读报告”的统一入口。

### 阶段 2：统一推理入口

验收条件：一个命令处理单张截图，允许选择槽位、性别、设备、候选数和基础模型回退。

结果：完成。入口为：

```powershell
python scripts\recognize_glamour.py reference.png --device auto
```

新增能力：

- `--check` 在耗时推理前检查索引、向量行数和模型文件；
- `--slots` 选择 HEAD/BODY/HANDS/LEGS/FEET；
- `--profile current-best|base` 选择训练模型或基础 DINOv2 回退；
- `--gender` 为已知角色模型提供候选性别路由；
- 自动输出 JSON、HTML、姿态覆盖图、裁剪联系表和区域清单；
- 对同一装备的多个候选视图和多个查询裁剪去重、加权并融合；
- HANDS 使用阶段冻结的裁剪路由、性别路由和 RRF(k=1)；
- BODY 使用当前部署版 qv52 + v35 + v2 索引。

### 阶段 3：可靠性与诚实呈现

验收条件：坏输入和缺资产尽早失败，结果不把相对分数伪装成概率。

结果：完成。

- 拒绝不存在、不可读或小于 200×200 的输入；
- 检查 metadata 行数与 embedding 第一维严格一致；
- 没有可用槽位裁剪时返回结构化状态；
- 输出每个候选的主要裁剪证据和来源；
- 报告明确写出分数未经校准；
- 每槽附上评估范围，且明确评估指标不等于本图置信度；
- HTML 对装备名和链接做转义，避免本地数据污染页面结构。

### 阶段 4：真实端到端验收

验收条件：不能只依赖 mock；需要用本地真实玩家截图、真实姿态模型、真实候选索引和真实训练检查点跑通。

结果：完成。使用 `terra-branford.jpeg`（`example_id=6220:0`）在 RTX 5070 Ti 上验收：

- 五槽完整推理约 16 秒完成；
- 姿态质量 0.973；
- 五个槽位均生成 5 个候选，共 25 张候选卡片；
- BODY 真实标签为 `Hempen Camise`，Top-1 返回 `Hempen Camise`；
- HANDS 真实标签为 `Star Velvet Long Gloves of Casting`，Top-1 返回同系列的 `Star Velvet Long Gloves of Healing`；
- JSON、HTML、overlay、contact sheet 和逐裁剪文件均生成成功；
- 报告 DOM 含 HEAD/BODY/HANDS/LEGS/FEET 五节，没有横向页面溢出。

## 当前基准与解释

| 槽位 | 默认方法 | Top-1 | Top-10 | 使用解释 |
|---|---|---:|---:|---|
| BODY | qv52 + v53 reranker + v3 | 45.9% | 74.5% | 外观不相交严格三折 |
| HEAD | DINOv2 qv65 + SigLIP2 rank ensemble | 17.6% | 41.2% | 外观不相交严格三折 |
| HANDS | qv71 + v5 + routing + RRF | 2.1% | 8.8% | 严格三折；路由曾在 fold-1 选参 |
| LEGS | qv72 + multiscale v6 + query-crop RRF | 9.1% | 26.4% | 外观不相交严格三折 |
| FEET | DINOv2 qv73 + SigLIP2 rank ensemble + RRF | 18.4% | 39.4% | 外观不相交严格三折 |

因此，“项目可用”在本阶段意味着：本地命令稳定产出可检查、可追溯的候选短名单；不意味着五槽 Top-1 已达到产品级精度。BODY 可作为主要识别能力，其他槽位更适合候选浏览和人工辅助。

## 已知限制与下一阶段

1. 只假设一个主要人物，宽图会搜索中心偏置窗口，但不会可靠计数或选择多人。
2. 最适合 FFXIV 正面/近正面全身图；任意插画、真人照片和严重遮挡不在已验证分布内。
3. 非 BODY 槽位像素少、遮挡和相邻装备泄漏严重，当前精度不足以自动确认装备。
4. 模型和索引是可重建的本地产物，没有随 Git 仓库分发；换机后需先构建或复制资产。
5. HTML 报告引用本地原图和候选预览，应在当前项目机器上直接打开。

后续优先级：先扩充各非 BODY 槽位的独立训练数据与候选外观覆盖，再冻结配置做三折/密封确认；不要继续只在单一诊断折上扫参。

## 验证命令

```powershell
D:\Anaconda\envs\ffxiv-prism\python.exe -m unittest discover -s tests -v
D:\Anaconda\envs\ffxiv-prism\python.exe scripts\recognize_glamour.py --check
D:\Anaconda\envs\ffxiv-prism\python.exe scripts\recognize_glamour.py `
  data\raw\eorzea_collection\glamours\images\6220\terra-branford.jpeg `
  --gender female --top-k 5 --candidate-k 1000 --device cuda
```
