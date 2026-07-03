# 跨种族、性别与来源的搭配检索计划（2026-07-01）

## 最终目标

最终推理场景可以包含任意性别、种族和来源的人物搭配图，系统返回视觉上最相似
的 FFXIV 游戏内装备或搭配。Hyur 女性 BODY 只保留为受限回归基准。

当前训练和准确率验证只使用 Eorzea Collection 中带真实 FFXIV 装备标签的
页面。其他游戏、CG、真人或插画没有可核验的真实装备标签，不进入训练集、
验证集或 80% 准确率计算。其跨来源效果只能作为未来无监督推理观察，不能替代
有标签的 FFXIV 验收。

## 当前证据

- 数据集 v35：796 套、3,263 张图，522 张通过严格正面/全身复核。
- 当前全域 BODY validation：80 个有效图像样本、57 个 glamour 组。
- v24 reranker：单图 Top-1 65.0%，Top-10 86.3%。
- v24 固定 reciprocal-rank 多视角融合为组级 Top-1 64.9%，未超过单图。
- v24 按性别：女性 43/59（72.9%），男性 9/21（42.9%）。
- v24 race 宏平均 Top-1 64.3%，race × gender 宏平均 55.8%。
- 新一轮 blind_test 为 14 组 / 17 BODY case，预测尚未查看。

因此当前模型仍是“FFXIV 高质量截图 BODY 检索原型”，不是任意人物搭配模型。

## 数据质量风险

严重级别为 **High**，结论置信度高：

- Hyur 女性占高质量数据的最大单一分组，男性及 Lalafell、Hrothgar 等长尾
  组合样本极少。
- 当前 query 训练图全部来自 Eorzea Collection；跨来源不在有标签评估范围内。
- 标准候选渲染主要是 Hyur 模型；即便装备相同，体型、比例和性别外观变化也
  可能成为域偏移。
- 仅看总体平均会掩盖男性及稀有种族的失败。

逆频率加权已实现并试验，但没有优于非加权 reranker；这说明缺失视觉变化不能
靠重复采样凭空补出。

## 评估规则

每次 FFXIV 精确识别报告至少包含：

- overall micro Top-1 / Top-10 / MRR；
- gender、race、race × gender 的 case 数、组数和 Top-1；
- race 与 race × gender 宏平均；
- 最弱分组 Top-1；
- 未达到最小样本量的分组清单；
- 单图与固定多视角规则分别报告。

正式 FFXIV BODY 验收建议：

- 冻结独立测试集至少 100 个 glamour 组、200 个 BODY case；
- 每个性别至少 50 组；
- 每个主要 race × gender 组合至少 10 组，无法达到者明确标成未覆盖；
- overall Top-1 目标 80%，race × gender 宏平均不低于 70%，任何样本充分的
  分组不低于 60%；
- 质量决策必须在查看预测前冻结。

## 推进顺序

1. 继续从 Eorzea Collection 补男性和最弱 race × gender 组合。
2. 为同一装备收集或生成跨种族、性别、体型的标准视图，减少候选侧 Hyur 偏置。
3. 保持 BODY 作为首个可用槽位，同时逐步恢复 HEAD、HANDS、LEGS、FEET。
4. 在数据补齐前，以 v24 非加权 reranker 作为全域开发基线；均衡版本保留为
   消融，不替换当前基线。
5. 不使用无真实装备标签的其他游戏或 CG 图做监督训练、模型选择或准确率验证。
6. 只有独立 FFXIV 有标签测试达到门槛，才宣称达到可验证的装备识别准确率；
   “任意来源”只能作为尚未量化的泛化目标。

## 可复现产物

- 全域 query：
  `data/models/glamour_query_encoder_v51_hq_v35_all_domains_body.pt`
- 当前全域 reranker：
  `data/models/glamour_pair_reranker_v24_hq_v35_all_domains_body_qv51.pt`
- 当前单图报告：
  `data/processed/glamour_pair_reranker_eval_v24_hq_v35_all_domains_body_qv51.json`
- 分层报告：
  `data/processed/glamour_pair_reranker_eval_v24_hq_v35_all_domains_body_qv51.domain_coverage.json`
- 多视角报告：
  `data/processed/glamour_pair_reranker_eval_v24_hq_v35_all_domains_body_qv51_multiview.json`
