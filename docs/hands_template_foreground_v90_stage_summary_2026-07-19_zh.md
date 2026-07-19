# HANDS 模板差分与查询前景 v90 阶段总结（2026-07-19）

## 结论

本阶段验证了两类“减少域差异”的低成本方案：标准素材侧模板差分、玩家图侧全局人物前景抑制。两者均能贡献少量新的正确候选，但在冻结折 1 上都无法同时提升 Appearance Top-1、Top-10，并保持性别/种族域及 Top-1000 不退化，因此均不接入生产推理，也不继续运行折 2/3。

当前 HANDS 生产基线仍为 DINOv2 qv71 v5、多裁剪候选路由、性别路由、裁剪 RRF(k=1)。折 1 严格结果为：

| 方法 | Eligible | Top-1 | Top-10 | Top-100 | Top-1000 |
|---|---:|---:|---:|---:|---:|
| 当前基线 | 315 | 11 | 32 | 101 | 283 |
| 候选模板差分单分支 | 315 | 6 | 26 | 98 | 281 |
| 基线 + 模板差分等权融合 | 315 | 12 | 35 | 103 | 284 |
| 模板差分最佳全域安全门控 | 315 | 11 | 32 | 103 | 283 |
| 查询全局前景单分支 | 315 | 10 | 37 | 105 | 281 |
| 基线 + 全局前景等权融合 | 315 | 12 | 33 | 108 | 285 |
| 全局前景最佳全域安全门控 | 315 | 11 | 32 | 103 | 283 |

所有数字均来自冻结、物品不相交折 1；没有用测试标签训练新参数。

## 1. 标准素材侧模板差分

### 构建

- 输入：3,668 张 HANDS 标准素材。
- 裁剪：`both_hands`、左右紧裁、左右前臂上下文、上半身手部上下文，共 6 种。
- 输出：22,008 条 DINOv2 候选向量。
- 模板：按槽位、种族、性别、染色状态生成中位图，再以阈值 48 做差分软掩码。
- 空掩码回退：0/22,008。

QA 表明手套与长护臂主体通常保留，但部分皮肤、躯干边界也会被保留。单分支结果全面弱于基线。它在折 1 新增 3 个 Top-1，但同时丢失 8 个基线 Top-1；等权融合虽达到 12 个 Top-1，却使 Hyur 与 Roegadyn 各减少 1 个 Top-1。

对 320 组无标签排名一致性门控做域检查后，共有 33 组不造成任一性别/种族、任一 cutoff 退化；最佳安全策略只能保持 Top-1/Top-10，不能推广。

主要产物：

- `data/indexes/facebook-dinov2-base-hands-template-v89-t48`
- `data/indexes/facebook-dinov2-base-hands-template-v89-t48-aligned-v5`
- `data/processed/retrieval_query_only_v89_hands_fold01_template_t48_rrf_k1.json`
- `data/processed/retrieval_query_only_v89_hands_fold01_template_t48_weight_grid.json`
- `data/processed/retrieval_gate_v89_hands_fold01_template_t48_screen.json`

## 2. 玩家图侧全局前景抑制

### 构建与视觉检查

全局人物前景在原始玩家图上解析一次，再将同一掩码应用到已有装备框。语义人体掩码与姿态包络取并集，以降低奇幻盔甲、长手套和武器附近出现孔洞的风险。

- 折 1 validation 且质量通过：325 张玩家图。
- 仅输出 HANDS 五种裁剪：1,625 张 JPEG。
- 使用灰色背景；不输出全分辨率掩码或透明 PNG。
- 总磁盘增量：约 16 MB。
- 自动质量告警：0。
- 缺失 manifest、区域或裁剪：0。
- 均匀抽样 QA：48 张；另检查了低、中、高前景占比样本，未发现手套或护臂被截断。

单分支将 Top-10 从 32 提高到 37，但 Top-1 从 11 降到 10，Top-1000 从 283 降到 281。它新增 3 个 Top-1，且这些正确物品在基线中分别位于第 3、2、2 名，说明背景确实影响少量近邻排序。

基线与前景等权融合为 12/33/108/285，但 Roegadyn Top-10 从 3 降到 2。对前景权重 0.25、0.5、0.75、1.0、1.25、1.5、2.0 分别执行无标签一致性门控；所有“性别、种族、Top-1/10/100/1000 全部不退化”的安全策略都停在 Top-1=11、Top-10=32。

五条单裁剪路由的 Top-1 分别为：

| 查询裁剪 | Top-1 | Top-10 | Top-100 | Top-1000 |
|---|---:|---:|---:|---:|
| `hands_left` | 2 | 21 | 79 | 241 |
| `hands_forearm_left` | 5 | 14 | 67 | 253 |
| `hands_right` | 7 | 25 | 81 | 245 |
| `hands_forearm_right` | 7 | 19 | 69 | 247 |
| `hands_arm_cover_context` | 6 | 20 | 86 | 254 |

没有单路达到生产多裁剪基线，因此不存在可单独推广的前景裁剪路由。

主要产物：

- `data/debug/global-foreground-v89-hands-fold01`
- `data/processed/appearance_disjoint_cv_v90_hands_fold01_qv71_globalfg_embeddings.npz`
- `data/processed/appearance_disjoint_cv_v90_hands_fold01_qv71_globalfg_relabel_rawpaths_embeddings.npz`
- `data/processed/retrieval_query_only_v90_hands_fold01_qv71_globalfg_rrf_k1.json`
- `data/processed/retrieval_query_only_v90_hands_fold01_raw_globalfg_weight_grid.json`
- `data/processed/retrieval_gate_v90_hands_fold01_raw_globalfg_screen.json`

## 3. 工程改动

- `build_equipment_only_candidate_index.py`：增加槽位过滤与 HANDS 多尺度策略，避免为无关槽位重复构建。
- `build_global_foreground_crops.py`：增加折、质量、槽位过滤与 crop-only 模式，完整折 1 产物仅约 16 MB。
- `reorder_candidate_index.py`：按 `local_path + candidate_crop` 做候选索引严格重排，拒绝重复、缺失和额外候选。
- `relabel_query_embedding_cache.py`：按稳定路径后缀校验并重标查询缓存，避免不同前景根目录导致语义裁剪错配。
- 为上述过滤、重排、重标逻辑增加单元测试。

## 4. 阶段判断与下一方向

本阶段进一步证明 HANDS 的主要瓶颈不是单纯背景或标准人台像素，而是玩家任意姿态与标准交叉手臂姿态之间的局部对应关系。继续叠加全局 CLS 向量、统一权重或阈值启发式，理论上限和实测收益都很低。

下一阶段应优先验证表示层方案：基于 DINOv3/DINOv2 patch token 的局部匹配、左右臂姿态归一化或从标准素材派生多姿态增强。任何方案仍须先通过折 1 的 Top-1/Top-10/Top-1000 与性别/种族门槛，再进入折 2/3。

项目总目标（五槽 Appearance Top-1 至少 80%）仍未达成；本报告不把候选覆盖或个别样例改善等同于自动识别可用。
