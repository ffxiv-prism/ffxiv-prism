# 上衣跨主机差异与手部漏检修复（2026-09-17）

## 对比对象与结论

- 附件：`900b3964f161.zip`，报告目录 `900b3964f161`，CPU 推理。
- 本机修改前最新报告：`data/outputs/web/9469bdec7861`，CUDA 推理。
- 两张输入 PNG 的 SHA-256 均为
  `8f737110ac8ec532c7e5655d40db85c144692b94e2b1a7f8879676eede8926b3`。
- 两份报告逐槽位 `method` 配置一致，包括索引、查询检查点、融合与样式参数。
- 图片右侧标注的上衣是旭革围裙（Hiderise Apron，piece 2204），手部是
  冬贝利手套（Tonberry Hands，piece 3636）。文字只用于人工核对，没有输入排序逻辑。

| 部位 | 附件 | 本机修改前 | 修改后 CUDA 实测 |
| --- | --- | --- | --- |
| 旭革围裙 | 第 1 名 | 二次排序前第 1，最终前 20 未显示 | 第 1 名 |
| 冬贝利手套 | 前 20 未显示 | 前 20 未显示；重放完整去重候选池为第 438 名 | 第 2 名 |

## 上衣：错误样式判断覆盖正确检索

本机 `style_diagnostic` 直接记录了
`previous_top1_piece_id=2204`、`current_top1_piece_id=1540`。错误发生在
SigLIP 样式二次排序，不是初始外观检索找不到旭革围裙，也不是训练的 pair reranker
（两份报告的 `reranker_checkpoint` 均为空）。

两台主机都把样式误判成 `crop_top`：附件约 0.500，本机约 0.499。
旧门控把短上衣、无袖、长外套三个目标概率相加，得到约 0.756，超过 0.45
即启动相互矛盾的样式路由。随后以 0.70 的权重混合样式得分与检索百分位，
导致正确候选被压低。GPU/CPU 的微小数值差异并不足以解释这次翻转。

附件报告记录 `rerank_reason=candidate_preview_unavailable`，候选预览不齐全，
因此跳过了二次排序，意外保住了正确第一名。本机预览齐全，实际执行了这一步。
报告能证明跳过的原因，但不能仅凭 ZIP 确认另一台主机具体缺少哪一个预览文件。

修复后 BODY 必须由一个目标样式以至少 0.75 的置信度胜出，才能改变裁剪权重、
候选路由和二次排序。常规上衣获胜或样式含糊时保留原始检索；LEGS 的可重叠
属性组合规则不受影响。此图的 `style_diagnostic.active=false`，上衣恢复第 1 名。

## 手部：手指截断、衣袖污染与图库姿势差异

原 `hands_left/right` 使用肩、肘、腕的包围盒；`hands_forearm_*` 仍以腕点为终点。
COCO 姿态没有指尖点。此图右腕约为 `(405, 1307)`，旧裁剪下界约为 `1383`，
实际手套指尖达到 `1540` 左右，大量连指手套轮廓被截掉。候选中的半指手套、
腕饰和袖口因此得到较高排名。

图库中冬贝利手套是交叉举手展示，输入则双手下垂；单纯把手掌新裁剪直接送入
原有手臂查询适配器，实验中正确候选没有改善。因此没有用这种方式替换已有输入。

修复新增沿肘腕方向延伸的 `hands_palm_left/right`，包含袖口与指尖，保留为
diagnostic 裁剪，不参与原始手臂 RRF。使用已有本地 SigLIP 模型做独立形状校验：

1. 两张完整手掌均可见，且“连指手套”得分均至少 0.70。
2. 原第一名的连指手套得分不超过 0.20，才允许修正。
3. 同模去重后，对现有候选池预览分类，得分至少 0.80 的 exact-slot 连指手套
   最多提升五种外观；不新增候选、不使用装备名称或标准答案作先验。
4. 单手不确定、截断、缺图、无匹配候选或原首位并未明显矛盾时，保留原排序。

`hand_shape_diagnostic` 保存触发原因、各手概率、候选数量和提升的 piece IDs；
候选的 `hand_shape_gate` 保存原排名。形状门控不改写原检索分数，报告不再把
提升后两名的原始分差当作可信 margin。已有严格身份 benchmark 不包含此新规则。

冬贝利手套目前进入第 2 名，第一名仍是另一款连指手套，尚未实现可靠 Top-1。
这是找回正确候选的优化，不是五部位全部正确或总体准确率提升的证明。

## 验证与重现

已执行完整五槽位 CUDA 推理、BODY/HANDS CPU 对照，以及 455 项单元回归，全部通过。
CPU 与 CUDA 均得到旭革围裙第 1 名、冬贝利手套第 2 名。
本机 17 份历史报告的手掌形状门控检查中，仅本次样本触发，其他 16 份未触发；
其中包含重复图片，不能当作独立的标注评测集。测试覆盖含糊 BODY 样式、
指尖范围、横向手臂、低可信关键点、边缘截断、双手不一致、缺图回退、
候选去重分类、提升上限与占位装备限制。

```powershell
python -m unittest discover -s tests
python scripts/recognize_glamour.py data/outputs/web/9469bdec7861/input/source.png `
  --device cuda --top-k 20 --output-dir data/outputs/recognition-fix-20260917
python scripts/recognize_glamour.py data/outputs/web/9469bdec7861/input/source.png `
  --slots BODY HANDS --device cpu --top-k 20 `
  --output-dir data/outputs/recognition-fix-20260917-cpu
```

结果分别保存在上述目录的 `report.html`、`results.json`、`crops.jpg` 和 `regions.json`。
原附件及本机历史报告保留不变。新的手部校验仅在双手形状满足条件时批量读取
候选预览，CPU 的这种条件分支会增加推理时间。缺少预览时应补齐 current-best
图片资产，诊断会明确记录回退原因。
