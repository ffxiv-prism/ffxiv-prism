# 非 BODY 部位优化阶段总结（v63，2026-07-10）

## 结论

此前成熟、可作为目标线的模型确实主要针对 `BODY` 特调。本阶段已把物品不相交拆分、候选索引、多裁剪查询训练与评估扩展到 `HEAD`、`HANDS`、`LEGS`、`FEET`，并完成了各部位的独立优化和对照实验。

截至 v63，所有非 BODY 部位仍未达到 BODY 的严格三折水平。本阶段目标没有实现；根据“只再进行一轮优化”的要求，本轮完成后停止继续实验，不把阶段停止误记为目标达成。

| 部位 | 评估范围 | 当前阶段最佳方法 | Top-1 | Top-10 | 与 BODY 的可比性 |
|---|---|---|---:|---:|---|
| BODY | 严格三折 pooled，987 例 | qv52 + v53 reranker + v3 index | 453/987（45.90%） | 735/987（74.47%） | 目标线 |
| HEAD | 三折 pooled，899 例 | qv65 + head multiscale v4 + face candidate routing | 145/899（16.13%） | 310/899（34.48%） | 可作三折比较 |
| HANDS | fold‑1 探索，315 例 | qv71 + hands multiscale v5 + paired crop/gender routing + weighted RRF(k=1) | 11/315（3.49%） | 32/315（10.16%） | 同折选参，不能视为无偏三折结果 |
| LEGS | 三折 pooled，955 例 | qv72 + legs multiscale v6 | 67/955（7.02%） | 201/955（21.05%） | 可作三折比较 |
| FEET | 三折 pooled，967 例 | qv73 + feet multiscale v6 | 92/967（9.51%） | 209/967（21.61%） | 可作三折比较 |

机器可读汇总见 [`non_body_optimization_current_best_v63_summary.json`](../data/processed/non_body_optimization_current_best_v63_summary.json)。

## 本阶段完成的工程改造

1. 将原本只按 BODY 身份拆分的严格交叉验证生成器泛化到五个核心槽位，保证同一目标部位物品不会同时进入训练与验证。
2. 将候选索引构建器扩展为按槽位构建，并为 HEAD、HANDS、LEGS、FEET 增加专用的多尺度候选裁剪。
3. 将查询编码器、成对重排器和检索漏斗泛化到任意槽位，支持断点恢复、跳过重复评估和指定折号。
4. 新增冻结候选索引上的 query-only 评估入口，统一输出 Top‑1/10/100/1000 与逐样本 rank。
5. 为 HEAD 增加语义裁剪到候选裁剪的路由；本轮进一步加入候选性别路由及跨查询裁剪的加权 RRF 融合。
6. 为上述拆分、索引裁剪和融合逻辑补充单元测试。

## 最后一轮优化：HANDS v63

### 动机

v59 的裁剪矩阵显示，`hands_forearm_left -> left_arm_tight` 单路已经达到 12/315（3.81%）Top‑1，但把所有查询和候选裁剪直接按最大余弦分数合并后只有 7/315（2.22%）。这说明不同侧、不同尺度的高分伪匹配会互相污染，继续简单增加裁剪并不能解决问题。

### 固定配置

最后一轮采用三项结构性约束：

- 左侧查询只检索 `left_arm_tight` 与 `left_forearm_context`，右侧查询使用对应的右侧候选；全臂上下文只检索 `both_hands` 与 `hands_upper_context`。
- 已知玩家模型性别时，只在相同性别的标准渲染候选中检索；当前 HANDS 候选库只有 Hyur 渲染，因此没有启用种族过滤。
- 不再跨裁剪取最大原始余弦分数，而是按可见性、裁剪角色和姿态置信度加权，使用 reciprocal-rank fusion，`k=1`。

正式结果文件为 [`retrieval_query_only_v63_hands_fold01_qv71_v5_consensus.json`](../data/processed/retrieval_query_only_v63_hands_fold01_qv71_v5_consensus.json)。

| 指标 | v59 基线 | v63 最后一轮 | 变化 |
|---|---:|---:|---:|
| Top‑1 | 7/315（2.22%） | 11/315（3.49%） | +4 命中，+1.27 个百分点 |
| Top‑10 | 21/315（6.67%） | 32/315（10.16%） | +11 命中，+3.49 个百分点 |
| Top‑100 | 83/315（26.35%） | 101/315（32.06%） | +18 命中，+5.71 个百分点 |
| Top‑1000 | 266/315（84.44%） | 283/315（89.84%） | +17 命中，+5.40 个百分点 |
| MRR | 0.04120 | 0.06334 | +0.02214 |

这是有方向性的正收益，但配置是在同一 fold‑1 诊断矩阵上选择的，结果属于探索性开发成绩。要形成无偏结论，需要冻结配置后训练并评估 HANDS fold‑2/3，或另设未参与选择的确认集；本阶段按要求不再追加该实验。

## 各部位实验结论

### HEAD

上半头部查询与 face-only 候选最有效；三折 pooled Top‑1 为 16.13%，是非 BODY 中最高的 Top‑1，但仍比 BODY 低 29.77 个百分点。RRF 能略增 Top‑10，却会轻微损伤 Top‑1，因此当前记录保留 face-candidate-only 的 Top‑1 方案。

### HANDS

手部像素少、左右侧和袖子/护臂重叠严重。专用多尺度索引、qv71、直接 reranker 和更长 reranker 训练均没有解决 Top‑1；v63 证明“对齐并融合一致证据”优于“所有证据取最大值”，但绝对精度仍很低。

### LEGS

v6 多尺度索引将三折 pooled 提升到 Top‑1 7.02%、Top‑10 21.05%。在 v6 上重新训练 qv74 的 fold‑1 Top‑1 为 8.92%，没有超过 qv72 在同一 v6 索引上的 9.24%；直接 reranker 与 RRF 也会降低 Top‑1，因此没有晋级。

### FEET

v6 多尺度索引的三折 pooled 为 Top‑1 9.51%、Top‑10 21.61%。将 qv73 延长到 20 epoch 和直接 reranker 都退化；保留 e10 query-only。保守 RRF 只改善深层召回，不改善 Top‑1。

## 验证与复现

核心代码验证命令：

```powershell
D:\Anaconda\envs\ffxiv-prism\python.exe -m py_compile `
  scripts\build_slot_aligned_candidate_index.py `
  scripts\evaluate_glamour_embedding_model.py `
  scripts\evaluate_query_only_retrieval.py `
  scripts\evaluate_retrieval_funnel.py `
  scripts\train_glamour_query_encoder.py `
  scripts\train_glamour_pair_reranker.py

D:\Anaconda\envs\ffxiv-prism\python.exe -m unittest `
  tests.test_slot_aligned_candidate_index `
  tests.test_item_disjoint_cv_split_manifests `
  tests.test_retrieval_funnel `
  tests.test_query_encoder `
  tests.test_pair_reranker -v
```

复现 v63 HANDS 最后一轮：

```powershell
D:\Anaconda\envs\ffxiv-prism\python.exe scripts\evaluate_query_only_retrieval.py `
  --examples data\processed\glamour_examples_huiji_models_closed.jsonl `
  --quality-manifest data\processed\front_fullbody_quality_final_v46.jsonl `
  --slot HANDS `
  --regions-dir data\debug\outfit-regions-v6-1882 `
  --index-dir data\indexes\facebook-dinov2-base-hands-multiscale-v5 `
  --fold 1 `
  --fold-manifest data\interim\appearance-disjoint-cv-v54-hands-huiji-closed\fold-01.json `
  --query-checkpoint data\models\appearance_disjoint_cv_v59_hands_fold01_qv71_v5_e10.pt `
  --output data\processed\retrieval_query_only_v63_hands_fold01_qv71_v5_consensus.json `
  --device auto --candidate-k 1000 `
  --candidate-crop-routing --candidate-gender-routing `
  --fusion rrf --rrf-k 1
```

候选索引、模型检查点、查询 embedding cache 和人体裁剪图属于可重建的大型本地产物，不提交到 Git；代码、测试、阶段文档、机器可读汇总和 v63 逐样本评估报告提交到仓库。

## 阶段决策

- 保留 BODY 的 qv52 + v53 + v3 作为严格目标线。
- 登记 HEAD、LEGS、FEET 的三折 pooled 最佳值。
- 将 HANDS v63 登记为本阶段最佳探索配置，不宣称为无偏三折成绩。
- 明确记录“非 BODY 达到 BODY 同等或更高水平”的目标尚未完成。
- 根据用户要求，本阶段在 v63 后结束，不继续训练、扫参或开启 blind pool。