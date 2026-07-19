# FEET 候选清洗 v84 阶段总结（2026-07-12）

## 阶段结论

FEET 索引中发现明确的候选元数据污染：部分部位渲染图同时错误关联同套装的
耳环、项链、手镯、戒指和武器，导致这些非足部物品以相同图像分数占据独立名次。
Terra 端到端报告中曾直接出现“FEET Top-7 为耳环”。

本阶段新增 `primary-family-v1` 清洗规则：保留每张部位图的首件视觉装备，以及
与它同名族的 Augmented、Replica、Weathered 等外观变体；剔除后续无关套装件。
FEET 独立候选由 3418 件降为 2005 件，19320 行嵌入及其顺序保持不变。

## 严格三折结果

完整三折清洗双模（DINOv2 + SigLIP2，967 张）：

| 指标 | 清洗前 | 清洗后 |
|---|---:|---:|
| Top-1 | 178 / 967（18.41%） | 187 / 967（19.34%） |
| Top-10 | 381 / 967（39.40%） | 398 / 967（41.16%） |
| Top-100 | 612 / 967（63.29%） | 678 / 967（70.11%） |
| Top-1000 | 892 / 967（92.24%） | 935 / 967（96.69%） |

清洗三模（再加入 DINOv3）已完成折 1/2：在 649 张图上，清洗双模为
Top-1 120、Top-10 257、Top-100 450、Top-1000 624；清洗三模为
Top-1 149、Top-10 301、Top-100 473、Top-1000 624。DINOv3 的前排收益仍然
成立，但按阶段性收尾要求未继续运行折 3，因此暂不作为完整三折基准发布。

## 晋升决策

默认 FEET 推理切换到完整三折验证的清洗双模索引。这样保证 Demo 报告中的每个
FEET 名次确实对应足部装备，同时改善所有统计层级。清洗三模保留为下一阶段候选，
待补完折 3 后再决定是否晋升。

相关产物：

- `data/indexes/facebook-dinov2-base-feet-multiscale-v7-clean`
- `data/indexes/google-siglip2-base-patch16-224-feet-multiscale-v7-clean`
- `data/indexes/facebook-dinov3-vitb16-feet-multiscale-v7-clean-complete`
- `scripts/filter_slot_candidate_index.py`
