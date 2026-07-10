# v47 预注册实验：严格折内查询困难负样本（2026-07-04）

## 假设

v44 漏斗显示折内查询编码器的 Recall@1000 已达 99.8%，但 Recall@10 只有
63.7%。正确装备通常已经进入大候选池，却没有稳定超过外观接近的错误装备。

本实验只改变查询编码器损失：在原有多正样本对比损失之外，加入显式困难负样本
softplus margin 损失。它要求正确装备的聚合分数超过当前最相似的一组错误装备。

## 冻结配置

- 数据：v44 高质量清单；
- 划分：现有三折 BODY-item-disjoint manifests；
- 查询训练索引：v2 slot-aligned，与 v44 基线相同；
- 查询骨干：每折从基础 `facebook/dinov2-base` 独立初始化；
- 可训练结构：最后一个 Transformer block + slot-attention pooler；
- 查询训练：固定 10 epoch，学习率 `5e-7`；
- 新增项：`hard_negative_weight=0.2`、`K=20`、`margin=0.5`；
- 重排器：与基线相同，固定 7 epoch、每例 1000 个困难候选；
- 正式评估索引：v3 BODY multiscale；
- 推理配置：固定 `balanced / none / max`；
- 不使用公开 validation 选择配置，不查看 blind。

## 基线与晋级条件

基线为 v44：

- pooled Top-1：365/906（40.3%）；
- pooled Top-10：676/906（74.6%）。

候选只有在以下条件同时满足时才晋级：

1. pooled Top-1 和 Top-10 都提高；
2. 至少两个折的 Top-1 不退化；
3. male 和 female Top-1 均不出现实质性退化；
4. 不用公开 validation 为该候选补选 epoch、权重或路由。

本文件在三个折的候选结果生成前冻结。
