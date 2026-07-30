# FFXIV Prism 可用版快速开始

当前项目已经提供单张截图到五个核心装备槽位候选的本地推理闭环。它适合清晰、单人物、正面或近正面的 FFXIV 全身截图，输出候选排名供人工确认。

## 1. 检查本地资产

```powershell
D:\Anaconda\envs\ffxiv-prism\python.exe scripts\recognize_glamour.py --check
```

`"ok": true` 表示五槽索引、查询模型、集成分支和检查点的文件及向量形状均可用。

## 2. 识别一张截图

```powershell
D:\Anaconda\envs\ffxiv-prism\python.exe scripts\recognize_glamour.py reference.png `
  --gender female `
  --device auto
```

`--gender` 可省略；已知角色模型性别时填写 `female` 或 `male`，目前主要用于 HANDS 候选路由。

只识别部分槽位：

```powershell
D:\Anaconda\envs\ffxiv-prism\python.exe scripts\recognize_glamour.py reference.png `
  --slots BODY HEAD `
  --top-k 5
```

如果只有候选索引而没有训练检查点，可使用基础模型回退：

```powershell
D:\Anaconda\envs\ffxiv-prism\python.exe scripts\recognize_glamour.py reference.png `
  --profile base
```

## 3. 查看结果

默认输出目录是 `data/outputs/glamour-match/<图片名>/`：

- `report.html`：可直接打开的候选预览报告；
- `results.json`：适合后续程序处理的完整结果；
- `overlay.jpg`：人体姿态和槽位区域质检；
- `crops.jpg`：检索裁剪与质检裁剪的联系表；
- `regions.json`：关键点、可见性、置信度和裁剪路径；
- `crops/`：检索裁剪以及不参与排名的质检裁剪。

FEET 的 `feet_full_left`、`feet_full_right`、`feet_full_context` 专门用于确认
完整鞋尖和鞋底是否被覆盖，`role` 为 `diagnostic`，检索权重为 0。若报告出现
`source_edge_truncation_risk`，说明原图边缘很可能已经把鞋截断，此时应更换完整截图。

HEAD 的 `head_full_accessory` 用于完整显示宽檐帽、高帽和大型头饰，也属于
`diagnostic`，不会参与当前模型排名。

候选 `score` 只用于相对排序，不是准确率或概率。请先检查 `overlay.jpg` 和 `crops.jpg`，再人工确认候选，尤其是 HEAD、HANDS、LEGS、FEET。

`current-best` 的 HEAD 使用 DINOv2、SigLIP2 与 DINOv3，FEET 使用 DINOv2 与
SigLIP2，BODY 使用 DINOv2、零样本 DINOv3 与查询适配 DINOv3 的保守 Top-1
门控。`--check` 会验证这些候选索引、查询检查点和本地权重缓存。缺失任一资产时先
复制或重建资产，再运行识别。

当前五部位严格指标、历史实验合并结论、数据准入规则和已知限制见
[当前状态](docs/current_status_2026-07-30_zh.md)。
