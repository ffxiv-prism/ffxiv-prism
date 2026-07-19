# HEAD 完整头饰裁剪 v88 阶段总结（2026-07-14）

## 结论

重新直接读取第 13 张用户测试图的裁剪产物后，确认 v86 的
`head_full_accessory` 仍然遗漏了草帽最左侧的外缘：旧框为
`[58, 0, 1080, 670]`。因此 v86 只能算部分修复，不能称为完整框选。

v88 将完整头饰诊断框的横向范围从脸宽的 `±2.40` 倍扩大到 `±2.70`
倍，并将帽冠上方范围从 `1.90` 倍扩大到 `2.15` 倍。第 13 张的新框为
`[0, 0, 1080, 670]`，覆盖源图中可见的左右帽檐、帽冠及顶部枝状装饰。
草叶在源图左边缘仍然贴边，因此该样本正确标记
`source_edge_truncation_risk`：项目不能恢复源图以外的内容，但不会再把这种
情况误报为确定完整。

## 额外回归

18 张用户样本全部重建后，发现 08 号高礼帽的帽顶虽然没有越出旧框，但正好
贴在裁剪上边缘。v88 的顶部余量把裁剪起点从 `y=186` 移到 `y=174`，保留了
约 12 像素安全边界。17 号宽檐礼帽也保持完整。

四个现有 HEAD 检索裁剪 `head_tight`、`head_upper_accessory`、
`head_face_accessory`、`head_context` 在 18/18 张图上逐像素不变。扩大后的
`head_full_accessory` 仍为 `role=diagnostic`，推理时权重为 0，因此不会未经
严格评估就改变已经发布的 HEAD 候选排名。

## 自动化保护

- 新增第 13 张草帽真实姿态坐标回归，要求完整框覆盖源图全宽并写入边缘风险；
- 新增第 08 张高礼帽真实姿态坐标回归，要求帽顶上方保留安全余量；
- 保留宽帽檐/高帽冠通用几何测试；
- HEAD 专项测试共 13 项通过。

## 可复核产物

- `data/debug/head-full-headwear-diagnostic-v88-user18-final/`：18 张 HEAD
  视觉回归；
- `data/debug/head-full-headwear-diagnostic-v88-user18-final/13/crops/overlay-head_full_accessory.jpg`：
  草帽完整裁剪；
- `data/debug/head-full-headwear-diagnostic-v88-user18-final/08/crops/overlay-head_full_accessory.jpg`：
  高礼帽顶部余量回归；
- `tests/test_outfit_regions.py`：第 08、13 张真实姿态坐标的回归断言。
