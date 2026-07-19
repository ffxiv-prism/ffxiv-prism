# HEAD 完整头饰裁剪 v86 阶段总结（2026-07-13）

## 发现的问题

复查最近 18 张用户测试图后，确认 13 号图的大草帽没有被完整框选。帽檐横向接近
整幅图宽，而旧 `head_upper_accessory` 只根据眼、耳等脸部关键点生成，实际只覆盖
帽子中央。17 号宽檐猫耳礼帽存在较轻的同类风险；08 号高礼帽还要求向帽冠上方
保留更多空间。这些图的头饰内容存在于原图中，属于裁剪框不足，不是源图截断。

## 实现方案

- 保留 `head_tight`、`head_upper_accessory`、`head_face_accessory`、
  `head_context` 四张既有检索裁剪的像素框；
- 新增 `head_full_accessory`，横向覆盖约 4.8 个脸宽，向上覆盖约 1.9 个脸宽，
  同时保留帽檐下方上下文；
- 新框标记为 `role=diagnostic` 和 `expanded_for_full_headwear`，检索权重为 0；
- 如果期望的完整头饰区域明显超出源图，添加 `source_edge_truncation_risk`，并让
  HEAD 结果显示对应低可信提示；
- CLI 在模型编码前过滤诊断框，因此不会增加 HEAD 模型的显存和推理开销。

## 回归结果

最终视觉复查确认：

- 13 号草帽的左右帽檐、帽冠和顶部装饰均进入 `head_full_accessory`；
- 17 号宽檐礼帽完整进入新框；
- 08 号高礼帽的帽冠顶部不再被截断；
- 16 号人物头部靠近原图边缘，被标记为潜在源图截断风险。

18 张样本的旧 4 张 HEAD 框全部保持不变。新框相对
`head_upper_accessory` 平均增加 151.2% 宽度和 68.6% 高度。

Terra Branford 端到端回归仍只编码 1 张已验证的 HEAD 查询裁剪，标准答案
Thavnairian Headdress 保持第 112 名，Top-10 顺序与 v84 完全相同。

验证产物：

- `data/debug/head-full-headwear-diagnostic-v86-user18-final/`：18 张 HEAD 视觉回归；
- `data/debug/glamour-match-e2e-v86-head-full-headwear-diagnostic/report.html`：
  Terra HEAD 端到端报告；
- `data/debug/head-full-headwear-diagnostic-v86-user18-final/13/crops.jpg`：
  草帽专项回归图。

## 边界

当前修复保证大型头饰可被完整查看和诊断，但不会把新大框直接送入既有检索模型。
宽帽檐裁剪包含更多背景，未经重新训练就参与排名可能改变已验证分布。若后续要利用
完整帽型提升候选排名，应新增宽檐/高冠路由或重新训练 HEAD 查询适配器，并通过严格
三折评估后再提升为默认检索输入。
