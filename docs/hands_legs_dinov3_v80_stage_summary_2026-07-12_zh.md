# HANDS / LEGS DINOv3 v80 阶段总结（2026-07-12）

## 结论

- HANDS：DINOv3 零样本、e3/e10 查询适配、名称形态分类和像素质量拒识均未形成可推广提升，线上保持 qv71 + v5 路由/RRF 配置。
- LEGS：DINOv3 与 DINOv2 明显互补。冻结 DINOv2 最终 Top-1000 piece 集合后，DINOv3 只在池内重排，严格三折 pooled Top-1 从 10.58% 提升到 13.93%，Top-10 从 26.39% 提升到 31.94%，Top-1000 完全保持 93.61%。该配置已接入推理。

## 索引构建

候选索引构建器新增 `--reuse-metadata`，可复用现有多尺度 metadata 和 `candidate_crop_box_normalized`，避免把同一原图的多个候选裁剪错误编码成相同全图向量。

- HANDS DINOv3：22,008 行，3,668 张源渲染 × 6 个裁剪。
- LEGS DINOv3：19,325 行，固定复用 v6 metadata。
- 模型：`var/models/dinov3-vitb16-pretrain-lvd1689m`，CLS pooling，768 维。

## HANDS 实验

### DINOv3 零样本与查询适配

第 1 折、315 例：

| 配置 | Top-1 | Top-10 | Top-100 | Top-1000 |
|---|---:|---:|---:|---:|
| 当前 DINOv2 qv71 | 11 | 32 | 101 | 283 |
| DINOv3 zero-shot | 5 | 22 | 65 | 201 |
| DINOv3 e3 | 5 | 22 | 65 | 203 |
| DINOv3 e10 | 5 | 22 | 68 | 210 |

DINOv2 固定候选池 + DINOv3 融合的 Top-1 降为 6，且深层覆盖下降，因此未训练另外两折。

### 形态与质量诊断

- 名称类别：`minimal`（nails/claws/wrist 等）、`long_armor`（gauntlet/armguard/armlet 等）、`glove`。
- 拼接左右手、左右前臂和双臂上下文 5 个 qv71 embedding 后，第 1 折分类准确率 52.31%、宏召回 47.60%；训练内部不存在满足 95% 精度的可用置信区间。
- 945 个严格样本中，主手裁剪宽度、面积和姿态置信度与 Top-1 没有稳定单调关系；简单像素阈值拒识不能提高可靠性。

因此 HANDS 的主要瓶颈仍是原始截图手部信息量和跨域排序，而非可由简单路由或阈值修复的问题。

## LEGS 实验

### DINOv3 单支

第 1 折 DINOv3 zero-shot 达到 Top-1 46/314、Top-10 106；当前 DINOv2 基础为 33/82，已有空间重排为 40/82。DINOv3 前排更强，但 Top-1000 只有 270，低于 DINOv2 的 289。

### 最终装备级候选池

旧候选池只在每个裁剪内固定，跨裁剪 RRF 后仍可能丢失 DINOv2 最终 Top-1000。v80 增加最终装备级候选池：

1. 用 DINOv2 单支计算每个样本的最终 Top-1000 piece 集合；
2. DINOv2+DINOv3 在每个裁剪内做 RRF(k=1)；
3. 跨裁剪做 RRF(k=1)；
4. 只保留步骤 1 的 piece，补回 ensemble 排序中缺失的基础候选。

| 折 | 当前 Top-1 / Top-10 / Top-100 / Top-1000 | v80 Top-1 / Top-10 / Top-100 / Top-1000 |
|---|---|---|
| 1 | 40 / 82 / 172 / 289 | 52 / 106 / 175 / 289 |
| 2 | 30 / 77 / 160 / 294 | 29 / 94 / 179 / 294 |
| 3 | 31 / 93 / 189 / 311 | 52 / 105 / 199 / 311 |
| pooled | 101 / 252 / 521 / 894 | 133 / 305 / 553 / 894 |

Pooled 指标：Top-1 10.58%→13.93%，Top-10 26.39%→31.94%，Top-100 54.55%→57.91%，Top-1000 保持 93.61%。第 2 折 Top-1 少 1 个，但 Top-10 增加 17、Top-100 增加 19；主要性别域和大多数种族域均提升，Roegadyn 小样本 Top-1 3→1 是已知风险。

### 空间重排组合

- 直接把冻结空间重排叠加在 ensemble 上：第 1 折 Top-1 52→47，拒绝。
- 当前 DINOv2+空间排序与 DINOv2+DINOv3 排序做最终 late RRF：第 1 折 Top-1 52→39，拒绝。

因此线上 LEGS 使用纯 DINOv2+DINOv3 最终池，不再串联空间重排。

## 端到端回归

Terra Branford：

- 标准答案 `Faire Kohakama` 仍为 Top-1；
- `Orison Skirt` 为 Top-3；
- 报告正确显示标准答案、等价 piece IDs 和 Top-1 命中；
- 单槽推理约 10.5 秒。

报告：`data/debug/glamour-match-e2e-v80-legs-dinov3-finalpool/report.html`。

## 验证

- `python -m unittest discover -s tests`：159 项通过。
- `python scripts/recognize_glamour.py --check`：五槽位全部通过，无缺失或无效资产。
