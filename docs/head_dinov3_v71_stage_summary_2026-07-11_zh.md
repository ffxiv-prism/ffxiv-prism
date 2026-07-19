# HEAD DINOv3 v71 与真值报告阶段总结（2026-07-11）

## 结论

项目当前可作为“上传截图后返回各槽位候选并人工确认”的本地工具使用，尚不能宣称为高准确率自动识别器。冻结、外观不相交三折评估中，HEAD 的 promoted 配置由 v69 的 Top-1 27.14% / Top-10 47.39% 提升到 v71 的 Top-1 27.36% / Top-10 47.50%；Top-1000 保持 94.10%。目标 Top-1 80% 尚未达到。

## 已并入的改动

- 将本地 DINOv3 ViT-B/16 模型迁入 `var/models/dinov3-vitb16-pretrain-lvd1689m`，建立 12,900 行 HEAD 多尺度索引和 4,300 行 full-render 训练索引。
- 查询编码器支持 DINOv3 的 `model.layer` 布局、最终 `norm` 和 4 个 register token；slot attention 不再把 register token 当作图像 patch。
- 以与 qv65 相同的配置训练三折 DINOv3 HEAD 查询适配器：最后 1 个 block、学习率 5e-7、固定 3 轮、每槽最多 3 个裁剪、按 glamour 分组。
- 推理 ensemble 的每个分支可配置独立查询检查点；HEAD 的 DINOv3 分支使用 fold-01 e3 检查点，SigLIP2 仍为零样本分支。
- 训练入口会校验候选索引 `model_id` 与查询骨干，检查点加载也会校验模型身份，防止同为 768 维的 DINOv2/DINOv3 被静默混用。
- 对冻结标注集内的已知本地图片，端到端 JSON/HTML 报告会显示标准答案、等价 piece IDs、真实候选名次、Top-1 命中状态，并高亮已显示的真值候选。

## 严格三折结果

| 配置 | Top-1 | Top-10 | Top-100 | Top-1000 |
|---|---:|---:|---:|---:|
| v69 DINOv2 + SigLIP2 + DINOv3 zero-shot | 244/899 (27.14%) | 426/899 (47.39%) | 640/899 (71.19%) | 846/899 (94.10%) |
| v71 DINOv2 + SigLIP2 + DINOv3 e3 query adapter | 246/899 (27.36%) | 427/899 (47.50%) | 641/899 (71.30%) | 846/899 (94.10%) |

融合保持 RRF k=10，并固定用前两个分支的 RRF k=1 Top-1000 作为候选池，因此 DINOv3 不会破坏原有候选覆盖率。

## Terra Branford 视觉诊断

- HEAD 标准答案是 `Thavnairian Headdress`（piece 10220）。v69 排名 120，v71 排名 112；Top-1 仍是外形明显不相似的 `Virtu Orison Hood`。该案例仍未解决。
- 查询图中真正头饰是头发两侧的小型红色装饰，而通用视觉 embedding 主要关注绿色头发、脸和红色服装；大面积兜帽因此被错误提前。
- LEGS 标准答案实际是 `Faire Kohakama`，并包含同模型 Shisui Kohakama 等价项。当前 Top-1 正是 `Faire Kohakama`；`Orison Skirt` 是 Top-2 且袜长、裙色都更不相似。新版报告会直接显示这一事实。

可复核报告：`data/debug/glamour-match-e2e-v72-known-truth-head-legs/report.html`。

## 未并入的实验

- `head_tight` 全局路由与多查询路由：第 1 折及独立第 2/3 折不稳定，拒绝。
- 基于单个 hair/face exposed fraction 的阈值路由：第 1 折微增但第 2/3 折回退，拒绝。
- 16×16 三通道人像解析布局重排：Terra 小样本中真值相似度高于 Top-1 兜帽，但第 1 折即使只给 2% 权重，Top-1 仍从 77/305 降到 57/305，Top-10 从 137 降到 135，拒绝接入线上。

## 当前五槽位严格基线与阶段目标

| 槽位 | 当前严格 Top-1 | 当前严格 Top-10 | 下一阶段重点 |
|---|---:|---:|---|
| BODY | 45.90% | 74.47% | 统一线上与严格指标口径；继续外观不相交训练 |
| HEAD | 27.36% | 47.50% | 小型头饰/遮脸/整帽分型监督；避免头发主导 |
| HANDS | 2.12% | 8.78% | 手套可见性筛选、左右手局部高分辨率监督 |
| LEGS | 10.58% | 26.39% | 保留袜长/裙长空间特征，扩展跨骨干局部监督 |
| FEET | 18.41% | 39.40% | 鞋型局部裁剪、多骨干适配训练 |

阶段目标按风险从低到高推进：

1. 保持候选覆盖和端到端可用性，所有改动必须通过完整测试与资产检查。
2. 为每个槽位建立“整体式、局部小件、遮挡/不可见”形态分层，分别训练或路由，先提升 Top-10 与人工确认效率。
3. 在每折训练集内学习路由/重排，不再从测试折手调阈值；只推广跨三折无退化方案。
4. 达到稳定的严格 Top-1/Top-10 后，再评估是否接近 80% 自动识别目标；当前数据和通用 embedding 尚不足以支持该承诺。

## 验证

- `python -m unittest discover -s tests`：151 项通过。
- `python scripts/recognize_glamour.py --check`：五槽位全部 `ok`，无缺失或无效资产。
- HEAD+LEGS 真实端到端：成功生成 JSON、HTML、overlay 和裁剪图。

## v73 训练内形态路由阶段

下一阶段按物品名称把 HEAD 真值分为 `enclosing`（hood/helm/cowl 等整顶包覆）、`face`（mask/visor/glasses 等面部型）和 `open`（headdress/ribbon/horns/hat 等开放型）。分类器只使用每折 `train` 角色；训练集内部再按物品名哈希划出开发子集，选择 ridge 正则和满足 95% 开发精度时覆盖率最大的置信门槛。验证角色不参与训练或门槛选择。

qv65 查询 embedding 的 `enclosing` 对其余类别二分类结果：

| 折 | 准确率 | 宏召回 | 非包覆召回 | 包覆召回 |
|---|---:|---:|---:|---:|
| 1 | 93.61% | 88.41% | 95.00% | 81.82% |
| 2 | 91.37% | 88.12% | 93.88% | 82.35% |
| 3 | 93.59% | 86.08% | 95.68% | 76.47% |

将高置信分类结果用于 Top-10 硬分区重排时，第 1 折 Top-1 77→78，但第 2 折 92→85，因此排序方案被拒绝。主要损失来自少量错误预测在同一物品的多张图片上重复发生，以及 mask/visor 的视觉覆盖程度与名称类别边界不一致。

分类器最终仅作为零排序风险诊断接入 HEAD：当高置信预测与 Top-1 名称形态冲突时，报告标记低可信并建议复核。Terra Branford 被正确诊断为“查询更像开放式/小型头饰，但 Top-1 是包覆式兜帽”；候选名次保持不变。

诊断报告：`data/debug/glamour-match-e2e-v75-head-morphology-warning/report.html`。
