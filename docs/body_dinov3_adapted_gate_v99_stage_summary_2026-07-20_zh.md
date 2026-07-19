# BODY DINOv3 适配门控 v99 阶段总结（2026-07-20）

## 阶段结论

BODY v99 已通过物品不相交、冻结三折评估并接入 `current-best` 实际推理。严格三折合计
Appearance Top-1 从 v96 的 600/987（60.79%）提升到 627/987（63.53%）；
Top-10、Top-100、Top-1000 分别保持 818/987、958/987、987/987，完整保留 v96
的候选覆盖能力。性别与九个种族切片在 Top-1/10/100/1000 共 44 个审计单元中均无退化。

该版本已达到可部署的阶段门槛，但 63.53% 仍低于项目长期设定的 80% BODY Top-1
目标。项目当前应视为“可运行、可输出五部位 Top-10 HTML 候选的 Demo”，而不是自动装备鉴定器。

## 冻结方案

每个 BODY 查询使用三个分支：

1. DINOv2 qv64 查询适配器；
2. DINOv3 零样本查询；
3. DINOv3 fold-01、e10 查询适配器。

基线排名使用分支权重 `[1.0, 1.0, 0.0]`，候选排名使用 `[1.0, 0.0, 1.25]`。
每个分支先按现有原始余弦分数融合多裁剪，再用 RRF(k=1) 融合分支。最终候选严格限制在
DINOv2 Top-1000 内。

只有同时满足以下无标签一致性条件时，候选排名的第一名才会替换基线第一名：

- 候选第一名原本位于基线 Top-10；
- 它在适配 DINOv3 分支中排名第 1；
- 基线第一名仍位于适配 DINOv3 Top-10；
- 适配分支中旧第一名的名次不优于新第一名。

通过门控时只移动 Top-1，其余顺序完全沿用基线；未通过时整条基线排名不变。因此 v99
不会用新模型的深层排序替换已经验证的 v96 尾部。

## 严格三折结果

| 折 | v96 Top-1/10/100/1000 | v99 Top-1/10/100/1000 | Top-1 变化 |
| --- | --- | --- | ---: |
| fold-01（选参） | 199 / 276 / 319 / 328 | 208 / 276 / 319 / 328 | +9 |
| fold-02（冻结） | 196 / 271 / 319 / 329 | 206 / 271 / 319 / 329 | +10 |
| fold-03（冻结） | 205 / 271 / 320 / 330 | 213 / 271 / 320 / 330 | +8 |
| 三折合计 | 600 / 818 / 958 / 987 | 627 / 818 / 958 / 987 | +27 |

| 指标 | v96 | v99 | 绝对变化 |
| --- | ---: | ---: | ---: |
| Top-1 | 60.79% | 63.53% | +2.74 个百分点 |
| Top-10 | 82.88% | 82.88% | 0 |
| Top-100 | 97.06% | 97.06% | 0 |
| Top-1000 | 100.00% | 100.00% | 0 |

## 性别与种族域审计

相对 v96 的 Top-1 命中变化如下；所有列出的域在 Top-10/100/1000 均为 0 变化。

| 域 | v99 - v96 Top-1 |
| --- | ---: |
| female | +17 |
| male | +10 |
| Au Ra | +6 |
| Elezen | 0 |
| Highlander | +2 |
| Hrothgar | +1 |
| Hyur | 0 |
| Lalafell | 0 |
| Miqo'te | +11 |
| Roegadyn | 0 |
| Viera | +7 |

## 未采用实验

- 直接用权重 1.25 的适配 DINOv3 替换零样本 DINOv3，三折 Top-1 可达到
  639/987，但 Hrothgar 出现 1 个 Top-1 回退，因此未部署。
- 等权替换达到 629/987，但 Roegadyn Top-1 回退 3 个，同样拒绝。
- DINOv2、零样本 DINOv3、适配 DINOv3 三分支直接集成在 fold-01 仅约 205 个
  Top-1，弱于替换方案。
- 采用整条候选排名的门控在 fold-03 造成 Top-100 与 female/Elezen Top-100
  各减少 1 个命中，因此改为只移动 Top-1。

## 生产接入与真实图片验收

生产配置位于 `src/ffxiv_prism/matching/inference.py`，命令行执行链路位于
`src/ffxiv_prism/cli/match.py`。通用实现新增加权分支 RRF、crop-then-branch 加权融合和
保守 Top-1 门控；HTML/JSON 方法快照会记录权重和门控阈值。

使用 `C:\Users\Administrator\Desktop\20.png` 完成真实 BODY 端到端验收：

```powershell
D:\Anaconda\envs\ffxiv-prism\python.exe scripts\recognize_glamour.py `
  "C:\Users\Administrator\Desktop\20.png" --slots BODY `
  --top-k 10 --candidate-k 1000 `
  --output-dir data\outputs\glamour-match-v99-body-20
```

推理在 CUDA 上约 36 秒完成，生成 `results.json`、`report.html`、`overlay.jpg`、
`crops.jpg` 和 10 个带本地预览图的 BODY 候选。该样本未满足替换条件，Top-1 与 v96
一致，符合保守门控预期。

头饰与鞋部完整包络的最终验收另见
`docs/pose_full_equipment_envelope_v95_stage_summary_2026-07-19_zh.md`。真实草帽样本 13
确认旧紧凑框漏帽檐；当前 `head_full_accessory=[0, 0, 1080, 670]` 已覆盖源图内所有
可见帽饰，并标记原图自身的边缘截断风险。该诊断裁剪权重为 0，不改变 HEAD 排名。

## 可复核产物

- fold-01：`data/processed/retrieval_gate_v99_body_fold01_d3e10_top1_frozen.json`
- fold-02：`data/processed/retrieval_gate_v99_body_fold02_d3e10_top1_frozen.json`
- fold-03：`data/processed/retrieval_gate_v99_body_fold03_d3e10_top1_frozen.json`
- DINOv3 查询检查点：
  `data/models/appearance_disjoint_cv_v97_body_fold01_dinov3_e10.pt`
- 真实图片 HTML：`data/outputs/glamour-match-v99-body-20/report.html`
- 草帽真实调试图：
  `data/debug/outfit-regions-v95-full-envelopes-strawhat/13/overlay.jpg`

## 当前可用性判断

项目的本地资产检查、五槽位推理入口、每槽位 Top-10、带图片 HTML 报告、JSON 结果、
人体/裁剪覆盖图和边缘截断提示均已形成闭环。完成本轮完整自动化测试后，可将当前版本
作为本机 Demo 使用。低准确率槽位和任意风格域外图片仍需人工从 Top-10 中确认，后续研究
应从新的冻结版本继续，不应把当前输出描述为确定答案。
