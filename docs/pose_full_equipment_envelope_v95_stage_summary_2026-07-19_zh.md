# 人体、完整头饰与完整鞋包络框 v95 阶段总结（2026-07-19）

## 结论

本阶段确认并修复了一个真实的“生成正确、展示错误”问题，以及一个姿态细化输入问题：

- 系统已经保存 `head_full_accessory` 与 `feet_full_context`，但 `overlay.jpg` 仍只画旧的紧凑 `head_context`、`feet_context`，因此宽檐草帽和厚底鞋看起来仍被框截断；
- `estimate_person_box()` 只在最低脚踝点下方增加人体纵向跨度的 10%，用户样本 `20.png` 的白色人体框止于 `y=697`，而完整鞋框延伸到 `y=724`，第二次姿态估计可能在输入阶段切掉鞋底；
- 草帽样本 13 的完整头饰框已覆盖所有画面内可见帽檐，但左侧帽檐本身被原图边缘截断，不能通过继续放大裁剪恢复。

## 修复

1. 人体细化框在膝、踝点可靠时，按小腿长度向两侧和下方扩展，给鞋尖、鞋跟及厚底预留完整空间。
2. 面部关键点可靠时，人体细化框并入与 `head_full_accessory` 相同的宽檐/高帽冠上半身包络范围。
3. `render_debug_overlay()` 改为显示 `HEAD full` 和 `FEET full`，不再把紧凑检索框误画成部位完整性框。
4. 大框继续保持 `role=diagnostic`、检索权重为 0；本轮不改变已冻结的 HEAD/FEET 查询分布和排名协议。

## 真实图回归

### 厚底鞋样本 `20.png`

- 修复前人体框：`(136, 79, 489, 697)`；
- 修复后人体框：`(125, 41, 489, 723)`；
- `feet_full_context` 覆盖两只长靴、鞋尖与厚底，紫框在叠加图中直接显示为 `FEET full`；
- 修复后各部位姿态状态均为 `visible`，没有因扩大细化框而退化。

产物：

- `data/debug/outfit-regions-v95-full-envelopes-20/Desktop-20/Desktop-20-overlay.jpg`
- `data/debug/outfit-regions-v95-full-envelopes-20/Desktop-20/Desktop-20-crops.jpg`

### 草帽样本 13

- `head_full_accessory` 为 `[0, 0, 1080, 670]`，覆盖帽冠与画面内全部可见帽檐；
- 新叠加图直接绘制该完整红框并标记 `HEAD full`；
- 框已触及原图左、上、右边缘，`source_edge_truncation_risk` 与 `partially_visible` 标记正确，说明剩余缺口来自源图截断。

产物：

- `data/debug/outfit-regions-v95-full-envelopes-strawhat/13/overlay.jpg`
- `data/debug/outfit-regions-v95-full-envelopes-strawhat/13/crops/overlay-head_full_accessory.jpg`

## 回归保护

`tests/test_outfit_regions.py` 新增：

- 用户 `20.png` 的真实关键点几何回归，要求人体细化框完整包含头饰与鞋包络；
- 像素级叠加图回归，要求绘制完整头饰/完整鞋框，而不是旧紧凑框；
- 既有草帽样本 13、宽檐/高帽冠、完整鞋下扩展和源图边缘截断测试继续保留。

## v94 合成域适配结论

同期完成的 HANDS 合成域随机化查询头没有晋级：4,000 张合成训练图上的装备级验证 Top-1/Top-10 为 78.2%/98.7%，但严格物品不相交 fold1 实图只有 `3/28/71/260`，低于现有基线 `11/32/101/283`。低权重融合最多得到 `11/36/99/283`，Top-1 不增且 Top-100 下降；权重 1.0 时 Top-1 降至 9。因此未继续 fold2/3，未修改生产配置。
