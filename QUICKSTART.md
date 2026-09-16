# FFXIV Prism 可用版快速开始

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

候选 `score` 只用于相对排序，不是准确率或概率。请先检查 `overlay.jpg` 和 `crops.jpg`，再人工确认候选，尤其是 HEAD、HANDS、LEGS、FEET。

`current-best` 的 HEAD 使用 DINOv2、SigLIP2 与 DINOv3，FEET 使用 DINOv2 与
SigLIP2，BODY 使用 DINOv2、零样本 DINOv3 与查询适配 DINOv3 的保守 Top-1
门控。`--check` 会验证候选索引、查询检查点和本地权重；公共模型缓存尚未下载时
会列入 `downloadable_models`，首次推理也会自动下载。推荐提前运行
`prepare_current_best.py`，以便在正式 CPU 推理前完成下载和完整性检查。

当前五部位严格指标、历史实验合并结论、数据准入规则和已知限制见
[当前状态](docs/current_status_2026-07-30_zh.md)。
