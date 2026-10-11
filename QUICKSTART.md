# FFXIV Prism 源码快速开始

普通用户请先阅读 [Windows 正式版安装与升级](docs/portable_release_zh.md)；本页仅面向源码开发环境。

当前项目已经提供单张截图到五个核心装备槽位候选的本地推理闭环。它适合清晰、单人物、正面或近正面的 FFXIV 全身截图，输出候选排名供人工确认。

## 1. 准备 clone 后的运行资产

```powershell
git lfs install
git lfs pull
python -m pip install -e .
python scripts\prepare_current_best.py
```

`prepare_current_best.py` 会校验随仓库提供的五槽索引、推理检查点和 DINOv3
权重，并从官方模型仓下载公共 DINOv2、SigLIP2 与姿态模型。输出
`"ok": true` 表示这些资产可以用于推理。

## 2. 下载装备图片

报告中的候选预览需要与当前索引精确对应的装备图片。只下载 `current-best`
实际引用的图片，并在完成后检查覆盖率：

```powershell
python scripts\download_eorzea_images_from_url_manifest.py `
  --current-best-only `
  --image-workers 8 `
  --delay 0.2
python scripts\download_eorzea_images_from_url_manifest.py --check
```

第二个命令返回 `"ok": true` 时，所有报告候选预览图均已就绪。下载可重复执行，
已有文件会作为本地缓存复用。

## 3. 在 CPU 上识别一张截图

```powershell
python scripts\recognize_glamour.py reference.png `
  --gender female `
  --profile current-best `
  --device cpu
```

`--gender` 可省略；已知角色模型性别时填写 `female` 或 `male`，目前主要用于 HANDS 候选路由。

只识别部分槽位：

```powershell
python scripts\recognize_glamour.py reference.png `
  --slots BODY HEAD `
  --device cpu `
  --top-k 5
```

如果只有候选索引而没有训练检查点，可使用基础模型回退：

```powershell
python scripts\recognize_glamour.py reference.png `
  --device cpu `
  --profile base
```

## 4. 查看结果

默认输出目录是 `data/outputs/glamour-match/<图片名>/`：

- `report.html`：可直接打开的候选预览报告；
- `results.json`：适合后续程序处理的完整结果；
- `overlay.jpg`：人体姿态和槽位区域质检；
- `crops.jpg`：检索裁剪与质检裁剪的联系表；
- `regions.json`：关键点、可见性、置信度和裁剪路径；
- `crops/`：检索裁剪以及不参与排名的质检裁剪。

多槽位装备只有在所有占用部位的外观均进入各自筛选前的 Top-5 候选时，才会保留在报告中；同模装备也遵循此规则。未识别或没有可用候选的部位不算通过。系统先保留完整检索候选池，在同模去重和多槽位筛选后，按原排名用后续合格候选补足 `--top-k`；补位不会放宽多槽位校验。只有候选池中的合格外观确实不足，或部位没有可用识别结果时，才会少于设定数量。`--candidate-k` 控制检索池大小，默认 1000。当 `--top-k` 小于 5 时，多槽位支持范围也相应缩小，确保支撑外观出现在最终报告中。

FEET 的 `feet_full_left`、`feet_full_right`、`feet_full_context` 专门用于确认
完整鞋尖和鞋底是否被覆盖，`role` 为 `diagnostic`，检索权重为 0。若报告出现
`source_edge_truncation_risk`，说明原图边缘很可能已经把鞋截断，此时应更换完整截图。

HEAD 的 `head_full_accessory` 用于完整显示宽檐帽、高帽和大型头饰，也属于
`diagnostic`，不会参与当前模型排名。
新增 `head_lower_face`、`head_ear_left/right` 覆盖下半脸、下颌及两侧耳部的小饰品，
裁剪尺度取自面部关键点，边界随双眼连线倾斜调整。它们目前也是零权重质检裁剪；
关键点不可信、极端侧脸或原图截断时会标记风险。
实验效果及限制见 [头部局部裁剪验证](docs/head_crop_optimization_2026-09-24_zh.md)。

`current-best` 还会从上述三个局部区域生成 `*_detail` 细节图，使用独立的
SigLIP2 零样本检索补充小型面部附件候选。两个视图的首位必须为同一官方同模外观，
且第三个视图也在前十支持它，才允许将该外观补到第六名；原前五名保持不变。
下半脸还须以至少 0.75 的相对分类分数支持一种局部附件类型，并获得一个耳侧
视图至少 0.5 分的同类支持；裸脸、帽子、头盔及头顶发饰不会触发此补充路径。
裁剪不完整、姿态不可信或视图意见不一致时保持原结果。报告 JSON 中的
`head_detail_diagnostic` 保存裁剪、支持排名和是否触发；这仍是候选召回，
不是物品身份确认。验证记录见 [小型头饰候选召回](docs/head_detail_recall_2026-09-26_zh.md)。

`current-best` 的 HEAD、LEGS 还会将全局检索与局部视觉证据连续融合：
眼部检索覆盖完整头部装备库，腿部比较腰至大腿和大腿两个视图；按官方同模外观合并证据。
融合可改变第一名并引入原候选池外的外观，没有固定插入名次或前十保护。
参数在三折验证集选择后冻结，再评测项目独立测试集。报告 JSON 的
`local_fusion_diagnostic` 记录参数与视觉分数；首次运行会建立眼部候选特征缓存。
结果、复现命令与评测范围见 [局部视觉融合优化](docs/recognition_local_recall_2026-09-29_zh.md)。

候选 `score` 只用于相对排序，不是准确率或概率。请先检查 `overlay.jpg` 和 `crops.jpg`，再人工确认候选，尤其是 HEAD、HANDS、LEGS、FEET。

BODY 的样式路由和二次排序仅在单个目标样式置信度达到 0.75 时启用，
且 `body_context` 与 `body_upper_core` 必须都以至少 0.75 的置信度支持同一样式；
躯干裁剪缺失、不可见或判断不确定时保留原始视觉检索。不会把短上衣、无袖和长外套的概率相加作为确定判断。
验证结果和未解决问题见 [2026-09-24 识别诊断](docs/recognition_regression_2026-09-24_zh.md)。HANDS 新增
`hands_palm_left/right` 完整手掌裁剪，用于双手连指手套形状校验，
不会直接输入原有手臂查询适配器。只有双手形状判断均通过、原第一名
明显不符且候选预览齐全时，才从现有候选池提升最多五种匹配外观；
诊断记录在 `hand_shape_diagnostic`。这些规则不代表身份已确认，仍需检查候选。

`current-best` 的 HEAD 使用 DINOv2、SigLIP2 与 DINOv3，FEET 使用 DINOv2 与
SigLIP2，BODY 使用 DINOv2、零样本 DINOv3 与查询适配 DINOv3 的保守 Top-1
门控。`--check` 会验证候选索引、查询检查点和本地权重；公共模型缓存尚未下载时
会列入 `downloadable_models`，首次推理也会自动下载。推荐提前运行
`prepare_current_best.py`，以便在正式 CPU 推理前完成下载和完整性检查。

当前五部位严格指标、历史实验合并结论、数据准入规则和已知限制见
[当前状态](docs/current_status_2026-07-30_zh.md)。
