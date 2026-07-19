# FEET DINOv3 一致性门控 v88 阶段总结（2026-07-14）

## 目标与约束

在 v84 的干净候选索引、DINOv2 + SigLIP2 双模型及冻结 Top-1000 piece
集合上，评估本地 DINOv3 是否能安全改善 FEET 排名。提升必须同时满足：

- item-disjoint 三折 pooled Top-1/Top-10 提升；
- Top-1000 候选覆盖率不下降；
- 性别域和种族域在 Top-1、Top-10、Top-100、Top-1000 的原始命中数均不
  下降。

## 精确评估基础设施

原始检索每个实验约需数分钟。v87/v88 新增了逐裁剪 piece 排名紧凑缓存与稳定
RRF 评估器，在保持 Python 稳定排序及逐查询矩阵乘法结果完全一致的前提下，将
单个融合实验降到约 6–11 秒。没有采用更快的批量矩阵乘法版本，因为它会造成
少量浮点名次漂移。

折 1 上 0.00–1.00、步长 0.05 的 DINOv3 权重精扫显示：所有非零权重至少
造成一个性别或种族域命中数下降。最接近的权重 0.80 将折 1 Top-1 从
67 提高到 76、Top-10 从 138 提高到 159、Top-1000 保持 313，但 Elezen
Top-1 从 2 降到 1，因此不能直接接入。

## 标签无关一致性门控

门控不读取性别、种族或标准答案，只使用线上可取得的排名证据：

1. DINOv3 权重固定为 0.80；
2. 新 Top-1 必须已经位于双模型基线 Top-20；
3. 新 Top-1 必须位于 DINOv3 Top-50；
4. DINOv3 对新 Top-1 的排名不得差于旧 Top-1。

阈值仅在折 1 筛选并冻结。折 2、折 3 只应用同一阈值，未继续调参。

| 折 | 基线 Top-1 / Top-10 / Top-100 / Top-1000 | 门控 Top-1 / Top-10 / Top-100 / Top-1000 |
|---|---:|---:|
| 1，326 例 | 67 / 138 / 226 / 313 | 80 / 143 / 230 / 313 |
| 2，323 例 | 53 / 119 / 224 / 311 | 71 / 128 / 229 / 311 |
| 3，318 例 | 67 / 141 / 228 / 311 | 72 / 145 / 230 / 311 |
| pooled，967 例 | 187 / 398 / 678 / 935 | 223 / 416 / 689 / 935 |

Pooled Top-1 为 19.34%→23.06%，Top-10 为 41.16%→43.02%，Top-100 为
70.11%→71.25%，Top-1000 保持 96.69%。女性 Top-1 135→160、男性
Top-1 52→63；但是 Hrothgar Top-1 从 9 降到 6，违反域不退化约束。

## 决策

不提升该门控，也不改动默认推理。FEET 继续使用 v84 干净 DINOv2 + SigLIP2
双模型配置。DINOv3 的总体互补性已得到确认，但需要新的独立开发/验证设计或更
稳健的、与身份域无关的置信度特征后才能重新评估，不能继续用折 2/3 反向调整
本轮门控。

## 可复核产物

- `data/processed/retrieval_compact_v87_feet_fold01_d3_weight_grid_000_100_step005.json`：
  折 1 精细权重扫描；
- `data/processed/retrieval_gate_v88_feet_fold01_d3w080_screen.json`：折 1 门控
  筛选；
- `data/processed/retrieval_gate_v88_feet_fold01_d3w080_base20_d3top50_ratio1_frozen.json`：
  冻结折 1；
- `data/processed/retrieval_gate_v88_feet_fold02_d3w080_base20_d3top50_ratio1_frozen.json`：
  冻结折 2；
- `data/processed/retrieval_gate_v88_feet_fold03_d3w080_base20_d3top50_ratio1_frozen.json`：
  冻结折 3；
- `data/processed/retrieval_gate_v88_feet_threefold_pooled_d3w080_base20_d3top50_ratio1_frozen.json`：
  pooled 域守门报告。
