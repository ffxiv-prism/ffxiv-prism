# 眼部装备局部检索改进（2026-09-30）

本轮已将通用眼部检索分支接入 `current-best` 的 HEAD 推理。此前未进入 Top20 的血盟双剑师头巾（Scion Rogue’s Bandana，8327），在原始截图的完整五部位识别报告中排到 **第 14**。没有按装备 ID、名称或指定输出位置添加规则，也没有替换现有训练权重。

## 改进方法

问题之一是查询与候选的局部视野不一致：查询按眼部关键点裁剪，候选原来使用固定比例框。现在候选也使用与查询相同的 `build_eye_regions` 几何规则；关键点可见性或置信度不足时，两只眼一起退回原固定框。此次 4,309 张候选渲染中，3,593 张使用关键点对齐，716 张回退。

对齐后，用现有 DINOv2 和 SigLIP2 分别检索左右眼。额外按图像内容建立四类检索分支：单眼遮挡、双眼遮挡、透明眼镜、厚镜片护目镜；头盔／面具和裸眼作为负向类别。所有类别提示语均描述通用外观，没有装备名称。SigLIP2 的六类相对 softmax 分数不是经过校准的概率。

候选外观的类别分数取其渲染图分数的平均值，重复渲染链接去重。达到统一阈值的候选，在对应类别内按局部图像相似性重新排名。两个查询视图（紧头部、面部配饰）必须共同支持该类别，取二者类别分数的较小值作为连续权重。两种模型的倒数排名分数取平均，再取左右眼较强的一侧；保留单眼配饰的证据，也降低单一模型的影响。

新增分数为：

```text
原 HEAD 眼部 gate × weight × Σ类别[
  min(两个查询视图的类别分数)
  × max左右眼(mean两模型(1 / (rrf_k + 类内排名)))
]
```

冻结参数是 `candidate_threshold=0.9`、`weight=1.0`、`rrf_k=10`。新增分数与原全局／局部融合分数相加，再正常排序；没有保送位置、Top20 插槽、装备白名单或特定 ID 加分。原裁剪可靠性、官方部位、同模装备和跨部位规则继续生效。仅 `current-best` HEAD 启用新分支。

几何、类别分数和局部向量均缓存；缓存绑定图片内容、模型资产信息及裁剪／预处理契约。模型资产信息使用文件路径、大小和修改时间。首次冷启动需要估计候选关键点并编码，耗时会增加；后续复用缓存。没有新增需要下载的模型。

## 原图完整推理结果

- 输入：`data/outputs/web/c842d68e9c68/input/source.png`。
- 输出：[HTML 报告](../data/outputs/recognition-eye-experts-20260930/report.html)、[结构化结果](../data/outputs/recognition-eye-experts-20260930/results.json)。
- HEAD 中目标装备排第 14。此前主检索约第 320、旧局部分支约第 74 是诊断阶段的排名，不能直接当作所有后处理阶段的同口径对比。
- 与上一版完整报告 `recognition-local-fusion-20260929/results.json` 对比，BODY、HANDS、LEGS、FEET 的 Top20 装备 ID 顺序完全相同。这只证明这张原图的四部位结果未变，不代表重新完成了四部位全量评测。
- 本机无界面浏览器启动失败，PNG 未导出；HTML、JSON 和实际裁剪均已保留，识别结果有效。

## 共享生产代码的评测

以下数字来自 `scripts/evaluate_head_eye_experts.py`，直接调用生产裁剪、缓存、类别聚合和计分函数。样本为 HEAD 查询样本，不能理解为同数量的独立装备。基线是冻结的全局身份检索与旧局部分支；此批量回放不包含生产中的形态重排、旧头部细节召回、无装备判断和跨部位后处理。完整原图推理另行验证如上。

| 数据集 | 样本数 | Top1 命中（前→后） | Top10 命中 | Top20 命中 | MRR |
|---|---:|---:|---:|---:|---:|
| 验证集全部 | 927 | 264→265 | 487→488 | 563→563 | 0.36966→0.37077 |
| 验证集眼部代理分组 | 63 | 11→12 | 20→22 | **27→26** | 0.23425→0.24860 |
| 既有留出集全部 | 107 | 37→38 | 67→69 | 75→77 | 0.43343→0.44755 |
| 既有留出集眼部代理分组 | 10 | 2→3 | 5→6 | 5→7 | 0.26368→0.39975 |

这不是所有样本或子集都不退步：验证集第 3 折 Top1 从 84 降到 82、Top10 从 151 降到 150；眼部代理分组 Top20 少 1 个。留出集全部 Top10 有 3 个新命中，也有 1 个原命中退出。因此本轮收益是解决原图召回问题，并改善汇总与留出集指标，不能宣称全面无回退。

眼部代理分组沿用名称正则 `glasses|spectacles|goggles|eyepatch|blindfold|visor`，仅用于评测分组，**不参与推理计分**。它会漏掉名称含 Bandana 等装备，不是真正完备的眼部语义标注。留出分组只有 10 个查询，结论仍需更多独立截图检验。

开发中反复使用过验证集。最终在双模型方案中，按“全部 MRR + 0.5 × 眼部代理分组 MRR”选择，并要求汇总 Top1／Top10／Top20 不回退；这不是逐折或逐子集不回退约束。本轮冻结策略后才打开既有留出集，没有再据其结果调参。但该集合在此前任务中使用过，不能称为新采集、从未见过的独立盲测。用户原图用于问题诊断和回归检查，没有作为训练样本或参数网格评分项；它仍是开发案例，不是独立泛化证据。

权威回放结果保存在：

- `data/outputs/head-eye-experts-20260930/production-evaluation/policy.json`
- `data/outputs/head-eye-experts-20260930/production-evaluation/validation.json`
- `data/outputs/head-eye-experts-20260930/production-evaluation/blind_test.json`

文件包含策略哈希、基线哈希、逐例排名和分组统计。`locked-policy.json` 是开发冻结记录；本节采用最终共享代码回放数字，而不是早期实验脚本的近似值。

## 复现与检查

在已准备好仓库数据及 `current-best` 资产的环境中运行：

```powershell
python scripts/evaluate_head_eye_experts.py --role validation --device cuda
python scripts/evaluate_head_eye_experts.py --role blind_test --device cuda
python scripts/recognize_glamour.py data/outputs/web/c842d68e9c68/input/source.png --profile current-best --device cuda --top-k 20 --output-dir data/outputs/recognition-eye-experts-20260930
```

可用 `--baseline-dir` 指定基线目录、`--output-dir` 指定新的评测目录。基线缺失时脚本调用现有流程生成；输入图片、索引或模型变化后，应使用新评测目录。脚本会检查策略和特征来源，不会默默混用旧缓存。先运行验证角色再运行留出角色；脚本只复现固定策略，不自动搜索参数。

全量单元测试 **568 项通过**，新增 10 项覆盖类别约束、双视图共识、禁用 gate、批量／单例一致性、渲染聚合、候选置换、异常输入、裁剪回退、缓存失效与启用范围。修改文件通过 Ruff，`git diff --check` 通过。

核心实现：`matching/aligned_eyes.py`、`matching/eye_experts.py`；接入点：`cli/match.py` 的 `_fuse_local_evidence`。算法、评测脚本和新增测试中没有目标装备的 ID 或名称。
