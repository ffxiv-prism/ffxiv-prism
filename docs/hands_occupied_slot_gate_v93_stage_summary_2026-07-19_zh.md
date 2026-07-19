# HANDS 多槽占位门控 v93 阶段总结（2026-07-19）

## 结论

v93 已接入默认 HANDS 推理。它不增加模型或显存开销，而是识别候选元数据中的
`link_type=occupied_slot`：这类候选是因 BODY 多槽装备占用手部而进入 HANDS
索引的整套上衣/盔甲，并不一定是独立手部装备。

冻结的 Top-1 门控在三折 945 个物品不相交样本上，把 HANDS Appearance
Top-1 从 `20/945 = 2.116%` 提升到 `29/945 = 3.069%`，相对提升 45%；
Top-10、Top-100、Top-1000 分别保持 `83/945`、`337/945`、`863/945`
不变，性别与种族分组均无命中回退。

## 视觉诊断

直接读取了 fold1 中 12 个“标准答案在 Top-10、但 Top-1 错”的近失误样本，
把玩家原图、左右前臂裁剪、错误 Top-1 和正确候选并排检查。主要发现：

1. `Brand-new Alphinaud's Attire`、`Herklaedi`、`Fat Cat Loungewear`
   等整套服装频繁借助大面积袖子或上衣纹理压过真正手甲；
2. 染色、袖子遮挡和 340×570 低分辨率会让真值手套只剩很少可见像素；
3. False Nails 等极小装备在部分图中几乎不可见，此时“实际装备真值”不等于
   “可见外观最像真值”；
4. 个别职业变体候选肉眼完全相同但未进入外观等价闭包，仍存在标签噪声。

视觉复核产物位于
`data/debug/hands-retrieval-failure-review-v93-fold01/`。

## 冻结策略

策略只在 fold1 选择，fold2/3 不再调参：

- 原始 HANDS 多裁剪 RRF 排名保持为主干；
- 在每个查询裁剪内，把 `occupied_slot` 的秩分数乘以 `0.25`；
- 重新做跨裁剪 RRF，只把结果当作结构性辅助证据；
- 仅当原 Top-1 为 `occupied_slot`、辅助 Top-1 为 `exact_slot`，且该
  exact HANDS 候选已在原始 Top-3 时，允许把它提升为 Top-1；
- 除 Top-1 相邻前移外，最终候选顺序和候选集合沿用原始主干。

## 三折结果

| fold | 样本 | 基线命中（1/10/100/1000） | v93 命中（1/10/100/1000） | 接受改写 | 分域回退 |
|---|---:|---:|---:|---:|---:|
| 1 | 315 | 11 / 32 / 101 / 283 | 14 / 32 / 101 / 283 | 66 | 0 |
| 2 | 311 | 4 / 23 / 98 / 279 | 7 / 23 / 98 / 279 | 61 | 0 |
| 3 | 319 | 5 / 28 / 138 / 301 | 8 / 28 / 138 / 301 | 59 | 0 |
| pooled | 945 | 20 / 83 / 337 / 863 | 29 / 83 / 337 / 863 | 186 | 0 |

逐样本 Top-1 审计为 10 个新增命中、1 个旧命中损失，净增 9。女性为
14→22、男性为 6→7；Au Ra、Hyur、Miqo'te、Roegadyn 均有新增命中，
其余种族不变。

## 跨槽一致性保护

唯一的旧命中损失 `294033:0` 经视觉检查确认是真实风险：玩家确实穿着黑色
染色的 Fat Cat 多槽家居服，独立 HANDS 门控错误地把小腕带提到 Top-1。

默认五槽流程因此增加了 BODY→HANDS 一致性保护：如果 HANDS 将一个多槽
服装降级，但 BODY Top-3 同时出现同名装备，则恢复该多槽服装为 HANDS
Top-1。该保护只撤销门控，不会凭空引入候选。

两个真实端到端回归均通过：

- `294033:0`：BODY Top-3 支持 Fat Cat Loungewear，HANDS 恢复它为 Top-1；
- `255834:0`：BODY 返回 Troian Lorica，不支持 Alphinaud 套装，HANDS
  保持 Paglth'an Armguards 从原第 3 提升到 Top-1。

当用户只请求 `--slots HANDS` 时，没有 BODY 证据，系统仍使用单槽冻结门控；
多槽候选始终保留在 Top-10，用户仍可人工确认。

## 被拒绝的相邻实验

- SAM2 v92：pooled Top-1 仅增加 1/945，运行成本高，未接入生产；
- 49 组 primary/auxiliary/context 裁剪权重扫描：只有原始权重满足全截断、
  全分域零回退，未修改生产权重；
- 全局 `occupied_slot` 抑制虽能提高 Top-1/Top-10，但 fold1 的 Roegadyn
  Top-1 和 Top-10 出现回退，因此改为只动 Top-1 的保守门控。

## 可复核产物

- pooled 审计：
  `data/processed/retrieval_occupied_penalty_gate_v93_hands_pooled.json`；
- fold1 选参与 fold2/3 冻结报告：
  `data/processed/retrieval_occupied_penalty_gate_v93_hands_fold*.json`；
- 视觉近失误清单：
  `data/debug/hands-retrieval-failure-review-v93-fold01/manifest.json`；
- 正向端到端：
  `data/debug/glamour-match-e2e-v93-crossslot-255834-gain/report.html`；
- 跨槽反例回归：
  `data/debug/glamour-match-e2e-v93-crossslot-294033-restored/report.html`；
- 评估脚本：`scripts/evaluate_hands_occupied_slot_penalty.py`；
- pooled 汇总：`scripts/summarize_hands_occupied_slot_gate.py`；
- 生产实现：`src/ffxiv_prism/matching/inference.py` 和
  `src/ffxiv_prism/cli/match.py`。

## 当前边界与下一步

v93 解决的是候选元数据结构污染，不是 HANDS 视觉表征本身。绝对 Top-1
仍只有 3.07%，离产品目标很远。下一阶段应优先修复同模型职业变体等价闭包，
并建立“手部装备可见性/可辨识性”评估层，把完全被袖子遮住或像素不足的样本
与真正可识别样本分开报告。
