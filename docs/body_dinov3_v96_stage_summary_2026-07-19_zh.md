# BODY DINOv3 v96 阶段总结（2026-07-19）

## 阶段结论

BODY v96 已通过冻结、官方同外观族不相交三折评估，并接入 `current-best`
推理。严格 Appearance Top-1 从 v53 的 453/987（45.90%）提升到
600/987（60.79%），相对命中数提升 32.45%；Top-10、Top-100 同时提升，
Top-1000 保持 100%。性别和种族切片在合并三折的 Top-1/10/100/1000 上均无
退化。

该结果显著提高了项目当前最成熟的 BODY 能力，但仍未达到 80% Top-1 总目标，
也不能外推为 HEAD、HANDS、LEGS、FEET 的准确率。

## 冻结方案

1. 主分支使用 DINOv2 Base、每折 qv64 查询适配器和 v3 BODY 多尺度候选。
2. 新分支使用本地 DINOv3 ViT-B/16 零样本特征和同一批 8,048 张 BODY 候选图。
3. 每个分支先用原始余弦分数对同一角色的 BODY 多裁剪做加权 `max`，保留各模型
   自己的分数尺度和既有裁剪协议。
4. 两个分支完成裁剪融合后，再以等权 RRF(k=1) 合并名次。
5. 最终候选严格限制在 DINOv2 的 Top-1000 中；DINOv3 只能重排，不能损坏已有
   深层覆盖率。

权重仅在 fold-01 上选择。等权方案冻结后，fold-02、fold-03 未再调参。生产配置
使用 fold-01 qv64 检查点，这与项目其他槽位使用一个折模型作为部署快照的方式一致；
三折指标来自每折各自训练且物品不相交的查询检查点。

## 严格三折结果

| 折 | v53 reranker Top-1/10/100/1000 | v96 等权集成 Top-1/10/100/1000 |
| --- | --- | --- |
| fold-01（选参） | 143 / 240 / 310 / 328 | 199 / 276 / 319 / 328 |
| fold-02（冻结） | 163 / 250 / 313 / 329 | 196 / 271 / 319 / 329 |
| fold-03（冻结） | 147 / 245 / 312 / 330 | 205 / 271 / 320 / 330 |
| 三折合计 | 453 / 735 / 935 / 987 | 600 / 818 / 958 / 987 |

对应合并召回率：

| 指标 | v53 | v96 | 绝对变化 |
| --- | ---: | ---: | ---: |
| Top-1 | 45.90% | 60.79% | +14.89 个百分点 |
| Top-10 | 74.47% | 82.88% | +8.41 个百分点 |
| Top-100 | 94.73% | 97.06% | +2.33 个百分点 |
| Top-1000 | 100.00% | 100.00% | 0 |

fold-01 的等权方案曾在 Elezen Top-10 上比 v53 少 1 个命中（16/21 对
17/21），因此没有把单折结果直接视为通过。冻结后合并三折，Elezen 变为
54/66，对 v53 的 51/66 增加 3 个命中。

## 性别与种族域审计

合并三折逐域对 v53 的命中变化如下，顺序为 Top-1 / Top-10 / Top-100 /
Top-1000：

| 域 | 样本数 | v96 - v53 |
| --- | ---: | --- |
| female | 640 | +113 / +61 / +12 / 0 |
| male | 347 | +34 / +22 / +11 / 0 |
| Au Ra | 167 | +31 / +16 / +4 / 0 |
| Elezen | 66 | +18 / +3 / +2 / 0 |
| Highlander | 59 | +9 / +8 / +5 / 0 |
| Hrothgar | 41 | +8 / 0 / +1 / 0 |
| Hyur | 184 | +37 / +26 / +4 / 0 |
| Lalafell | 48 | +5 / +3 / +4 / 0 |
| Miqo'te | 246 | +26 / +14 / +2 / 0 |
| Roegadyn | 31 | +3 / +2 / 0 / 0 |
| Viera | 145 | +10 / +11 / +1 / 0 |

所有 40 个性别/种族 × 阈值切片均未退化。

## 工程修复与生产接入

- `evaluate_glamour_embedding_model.py` 现在拒绝用不匹配的 metadata 静默覆盖
  已存在索引，并校验向量行数，修复了本阶段发现的危险缓存覆盖路径。
- 分支搜索缓存新增原始余弦 `scores`；旧调用仍可按原有两返回值接口读取名次。
- `evaluate_rank_ensemble_retrieval.py --cache-only` 可只生成紧凑缓存，避免耗时的
  大型 Python 对象合并。
- 紧凑评估器新增 `crop-then-branch`，权重 0 能精确复现 qv64：fold-01 为
  129/242/307/328。错误的“先按裁剪融合分支名次”实验不能通过这个自检，未被采用。
- 通用推理新增保持 metadata/embedding 行对齐的按槽位筛选。DINOv2 v3 物理索引有
  31,081 行、DINOv3 BODY 索引有 8,048 行；运行时会先把前者正确筛到同一批
  8,048 个 BODY 视图，再核对路径。
- `current-best` BODY 已改为 DINOv2 qv64 + DINOv3 零样本等权集成，使用
  `crop_then_branch` 并冻结 DINOv2 Top-1000 池，不再调用旧 v35 reranker。

## 真实图片验收

使用 `C:\Users\Administrator\Desktop\20.png` 完成 BODY 端到端验收：

```powershell
python scripts\recognize_glamour.py "C:\Users\Administrator\Desktop\20.png" `
  --slots BODY --top-k 10 --candidate-k 1000 --device cuda `
  --output-dir data\outputs\glamour-match-v96-body-20
```

结果成功生成 `results.json`、`report.html`、`overlay.jpg`、`crops.jpg` 和 10 个
带本地预览图的 BODY 候选。输入为宽松黑色图案短袖，Top-1 为同类宽松图案短袖；
人工视觉冒烟检查确认轮廓、袖长和衣摆长度相符，没有部位或候选图片路径错位。

可直接打开：
`data/outputs/glamour-match-v96-body-20/report.html`。

## 可复核产物

- fold-01 网格：
  `data/processed/retrieval_compact_rank_ensemble_v96_body_fold01_dinov3_weight_grid_crop_then_branch.json`
- fold-02 冻结结果：
  `data/processed/retrieval_compact_rank_ensemble_v96_body_fold02_dinov3_equal_crop_then_branch.json`
- fold-03 冻结结果：
  `data/processed/retrieval_compact_rank_ensemble_v96_body_fold03_dinov3_equal_crop_then_branch.json`
- 旧严格基线：
  `data/processed/retrieval_funnel_v53_huiji_closed_v3_reranker_strict.json`
- BODY DINOv3 索引：
  `data/indexes/facebook-dinov3-vitb16-body-multiscale-v3`

## 下一阶段

BODY 的下一主要瓶颈已经从 Top-10 召回转为 Top-10 内的细粒度款式排序。后续应优先
分析 v96 中“正确答案已在 Top-10 但未到 Top-1”的案例，尝试局部图案、领型、袖长、
衣摆长度和层叠结构特征；任何新重排器仍必须锁定 v96 候选池并通过相同三折与域门槛。
五槽总体目标仍需继续优先提升当前明显偏低的 HANDS、LEGS、FEET 和 HEAD。
