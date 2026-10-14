# FFXIV Prism

本项目是一个基于图像识别的本地 FF14 装备外观检索工具，自带 web 界面，主要用途：识别幻化调料包，角色 COSPLAY，相似风格外观检索。完整项目需要 **6-10G**，请确保磁盘空间充足。

需求输入：任意风格清晰单人、正面或近正面全身图片。遮挡、特效和拍摄角度会影响结果。
期望输出：按装备部位返回评分排序后的指定数量检索结果，同模装备去重。单次推理在本地生成网页形式报告x1，png长图报告x1。

原理简介：本项目主要构成为开源 cv 模型本体+训练后的权重以及索引文件+游戏装备样品图片，检索范围包含当前大部分游戏装备，后续新装备若想纳入可检索范围，需要更新权重以及索引，届时会发布增量更新 release（也可能攒一波更新

## 下载

请下载 release 版本压缩包 **FFXIV-Prism-v0.2.1-windows.zip** 与图片资源压缩包 **FFXIV-Prism-gallery.zip**，适用于 **Windows x64**，无需预先安装环境：

| 文件 | 下载地址 | 用途 |
| --- | --- | --- |
| `FFXIV-Prism-v0.2.1-windows.zip` | [下载安装包]（Github release）(https://github.com/ffxiv-prism/ffxiv-prism/releases/download/v0.2.1/FFXIV-Prism-v0.2.1-windows.zip) | Release 版本程序与安装入口 |
| `FFXIV-Prism-gallery-20261009.zip` | [下载装备图库（Google Drive）](https://drive.google.com/drive/folders/1G5N7_xYPplvjrjVKcpQRvAbjZO-2SzcQ?usp=sharing) | 识别报告所需的装备图片与检索数据 |
| | [下载装备图库（Baidu 网盘）](https://pan.baidu.com/s/1qCPLl6pJhXBDCZj4bi6rOA?pwd=jnu4) | 同上，择一即可，提取码jnu4 |

图库zip的sha为：433eb67ff0a00b60b3690598d038f449e704eb92257f8af74ba15ba844f6e33c

## 开始使用

1. 将 Windows 压缩包**解压**到英文路径下。
2. 把图库 ZIP **放入项目目录中的 `imports` 文件夹**，**无需**解压文件。
3. 双击 **安装环境.bat**，安装依赖并下载cv模型。默认仅安装 CPU 环境即可使用全部识别功能；有兼容 NVIDIA 显卡的用户可以选择额外安装 GPU 环境（没快多少，多占空间）。首次安装需要联网下载，请等待安装和自检完成。安装成功后会清理本次导入的原始图库 ZIP 和临时下载文件，识别报告与已安装模型会保留。
4. 双击 **启动.bat**，自检后显示本地127地址，ctrl+左键或复制到浏览器即可打开web页面。
5. 上传截图并开始识别。运行期间请保留命令行窗口。
6. 项目全部依赖均位于项目根目录下，如需卸载，删除文件夹即可。

安装完成后可离线识别。报告保存在项目根目录 `user/reports`，可查看 HTML/JSON/PNG 报告。

## 常见问题

- **提示没有图库 ZIP**：确认图库zip文件（文件名带有gallery字样）放在 `imports` 中，再运行安装环境。已经成功导入过图库时，可以直接复用，不必再次下载。
- **环境下载中断**：重新运行安装环境即可重试，已有下载会尽量续传。请留意窗口中的失败原因和剩余磁盘空间。
- **GPU 安装失败**：CPU 环境仍可使用。关闭程序，重新运行安装环境并选择 2，可继续准备 GPU。额外安装 GPU 依赖请预留至少 12 GiB 临时空间。
- **提示程序或环境异常**：关闭程序后运行 **修复.bat**。如果提示图库文件损坏，请重新下载配套图库 ZIP，放入 `imports`，再运行安装环境。

反馈问题时，请提供错误提示、复现步骤，以及相关日志：

- 安装自检：`work/logs/install-check.log`
- 网页服务：`work/logs/server.log`

## 开发者

### 依赖与开源说明

本项目使用以下公开项目和网站：

- **模型与框架**：[DINOv2](https://github.com/facebookresearch/dinov2)、[DINOv3](https://github.com/facebookresearch/dinov3) 、[SigLIP 2](https://huggingface.co/google/siglip2-base-patch16-224) 、[ViTPose](https://huggingface.co/usyd-community/vitpose-base-simple)；通过 [PyTorch](https://pytorch.org/) 与 [Transformers](https://github.com/huggingface/transformers) 训练和推理。
- **图片与游戏资料**：[Eorzea Collection](https://ffxiv.eorzeacollection.com/) 提供装备外观样本，[FF14 灰机维基](https://ff14.huijiwiki.com/)、[XIVAPI](https://v2.xivapi.com/)  提供物品资料。

本项目原创代码采用 [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)，允许使用、修改及商业分发，需保留许可与相关声明，并标明修改。第三方依赖、模型及其衍生权重遵循各自许可。

游戏素材、网站图片与玩家幻化投稿不包含在本项目代码许可范围内，相关权利归原权利人所有。

### 源码运行

安装 Git、Git LFS 和 Conda，在 Windows PowerShell 中执行：

```powershell
git lfs install
git clone https://github.com/ffxiv-prism/ffxiv-prism.git
cd ffxiv-prism
git lfs pull
conda env create -f environment-cpu.yml
conda activate ffxiv-prism-cpu
python scripts/prepare_current_best.py
python scripts/download_eorzea_images_from_url_manifest.py --current-best-only --image-workers 8 --delay 0.2
python scripts/download_eorzea_images_from_url_manifest.py --check
python scripts/web_app.py
```

检查通过后打开 `http://127.0.0.1:8765`，按 Ctrl+C 停止。环境使用 Python 3.12；GPU 改用 `environment-gpu.yml` 和 `ffxiv-prism-gpu` 环境。Git LFS 获取权重与索引，准备脚本下载其余模型。

源码与 Release 统一使用 `data/images/` 图片目录及 `data/indexes/<部位>-<模型>/` 当前索引目录。已有开发副本可运行 `python scripts/migrate_gallery_layout.py --apply` 迁移，原始图片和历史索引保留，元数据备份位于 `work/gallery-layout-backup/`。

命令行识别：`python scripts/recognize_glamour.py reference.png --device cpu`。源码报告在 `data/outputs/` 下，PNG 导出需 Chromium 内核浏览器。

### 数据维护、训练与打包

核心代码在 `src/ffxiv_prism/`，网页在 `web/`。以下脚本均位于 `scripts/`，用 `python scripts/<脚本名> --help` 查看参数。

- **更新图库**：使用 `run_lifecycle_pipeline.py --workflow sample-library --source-mode refresh --candidate-mode extend`，先加 `--dry-run` 预览，确认后去掉该参数执行。采集时遵守网站规则并控制频率，报告在 `data/outputs/lifecycle-reports/`。
- **训练评估**：依次使用 `audit_slot_quality_folds.py`、`run_weight_refresh_training.py`、`evaluate_weight_refresh.py`、`register_weight_refresh.py` 完成审核、续训、评估和登记。训练需另备有权使用的图片、标注、数据划分及各折检查点，仓库不含完整训练数据。登记后使用生命周期脚本的 `model-release` 流程检查并导出资产。
- **测试打包**：测试命令为 `python -m unittest discover -s tests`。使用 `build_portable_release.py --app-only` 构建程序包；图库需完整数据，先运行 `prepare_portable_features.py` 再以 `--gallery-only` 构建。还需准备 `work/python-3.12.10-embed-amd64.zip` 及匹配公钥的私钥，私钥不随仓库分发。
