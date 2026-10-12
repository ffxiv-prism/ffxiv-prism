# FFXIV Prism 源码快速开始

普通用户请使用 [Windows 安装包](docs/portable_release_zh.md)。以下命令在仓库根目录运行。

## 准备环境与资产

使用 Python 3.12。CPU 环境可通过仓库中的 Conda 配置建立：

```powershell
conda env create -f environment-cpu.yml
conda activate ffxiv-prism-cpu
git lfs install
git lfs pull
python scripts\prepare_current_best.py
```

需要 NVIDIA GPU 时可使用 `environment-gpu.yml`，环境名为 `ffxiv-prism-gpu`，需配套驱动。推断依赖版本由环境配置和 `pyproject.toml` 约束。

下载当前候选索引引用的装备预览图片：

```powershell
python scripts\download_eorzea_images_from_url_manifest.py --current-best-only --image-workers 8 --delay 0.2
python scripts\download_eorzea_images_from_url_manifest.py --check
```

准备与检查命令返回 `"ok": true` 后再开始识别。

## 运行

```powershell
python scripts\web_app.py
```

也可直接识别截图：

```powershell
python scripts\recognize_glamour.py reference.png --profile current-best --device cpu
```

使用 `--device cuda` 选择 GPU；使用 `--slots HEAD BODY` 限制部位。命令输出中包含报告路径，候选结果需要人工确认。

## 测试

```powershell
python -m unittest discover -s tests
```
