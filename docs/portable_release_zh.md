# Windows 便携测试版

本版为预发布测试版，适合 Windows x64。无需安装 Python、Conda 或 Git。首次准备需要下载所选运行依赖和基础模型，完成后可以断网使用。CPU 识别耗时较长，请保留命令行窗口。

## 第一次使用

1. 下载 `FFXIV-Prism-v0.2.0-beta.6-windows.zip` 并解压到较短、可写的英文目录，例如 `D:\FFXIV-Prism`。整个存放路径不能包含中文，避免过深的目录。不要直接在压缩包里运行，也不要选择 `Source code`。
2. 从维护者提供的网盘下载 `FFXIV-Prism-gallery-20261009-visor-feet-v3.zip`，原样放入解压目录的 `imports` 文件夹。网盘链接由维护者上传后提供；测试草稿暂未填写链接。
3. 双击 `导入更新.bat`。它会校验并安装图库、下载固定版本的运行依赖和模型。网络中断后可以重新运行；不要删除 `.part` 文件。
4. 完成后双击 `启动.bat`，复制命令行中的 `http://127.0.0.1:8765` 到浏览器。不会自动打开浏览器。
5. 上传单人截图，选择部位，点击“开始识别”。页面显示当前版本和可检索图库日期。报告保存在 `user/reports`。

按命令行中的 `Ctrl+C` 停止服务。关闭后可移动整个文件夹；删除整个文件夹即可移除项目管理的文件。

## 准备状态与下载来源

五个 BAT 一打开就会显示操作名称。后续会显示扫描 ZIP、验签、解压、依赖安装、
模型检查、自检和网页启动等当前阶段。下载显示文件序号、源网站、完整下载地址、
实际重定向网站、进度条、百分比、已下载/总大小、速度和预计剩余秒数。
连接等待、重试、换源、断点续传以及 SHA-256 校验也有提示。连接或读取超时为 20 秒。
文件已缓存且校验通过时显示复用状态；没有网络传输时不会显示虚假的下载进度。

运行依赖来自公开网站，使用固定文件及 SHA-256，不会自动升级版本：

- 普通 Python 包：PyPI 文件站 `files.pythonhosted.org`，失败后依次尝试
  华为云 `mirrors.huaweicloud.com`、清华 TUNA `mirrors.tuna.tsinghua.edu.cn`。
- PyTorch CPU 及清单中来自其仓库的包：`download.pytorch.org`；
  不以普通 PyPI 的不同构建替换 CPU wheel。这些文件暂无已验证的国内备用源。
- 模型：`huggingface.co`，有已锁定备用文件时尝试 `modelscope.cn`。
  CDN 重定向后会显示实际下载网站。模型的来源列表见 `manifests/models.json`。
- Python 解释器已包含在完整 ZIP 内，用户运行 BAT 时无需下载 Python。

镜像可能尚未同步某个固定版本，失败会继续尝试下一个来源；所有来源都失败时，
窗口保留各来源错误与手动放置路径。镜像与官方源使用相同的文件大小及 SHA-256。

## 从 beta.1 至 beta.5 升级到 beta.6

1. 关闭旧版服务。下载 `FFXIV-Prism-v0.2.0-beta.6-update.zip`，将解压后的 `FFXIV-Prism-Update` 文件夹放入原项目根目录，例如 `D:\FFXIV-Prism\FFXIV-Prism-Update`。
2. 双击其中的 `更新.bat`，脚本自动识别父目录中的原安装，无需输入路径。
3. 脚本验证签名和文件 SHA-256，自动备份并替换 `bootstrap`、五个 BAT 和使用说明，然后安装完整程序包并自检。
4. 完成后在原安装目录双击 `启动.bat`。无需逐版安装，支持 beta.1 至 beta.5 直接更新，以及 beta.6 重试。

脚本自动移走旧启动器（含已废弃文件），删除过期的待更新状态文件，并把 `imports` 中已签名的程序 ZIP 归档到 `work/upgrade-backups`。图库 ZIP 不移动。模型、依赖缓存、图库与用户报告保留；失败恢复原启动器、活动版本和 Python 路径。服务仍在运行时拒绝更新。

`回退.bat` 可恢复升级前程序版本；启动器备份留在 `work/upgrade-backups`。要启用本次多外观头饰与鞋部优化，升级程序后还需导入新的 `gallery-20261009-visor-feet-v3` 图库。旧图库仍可使用，但不包含新功能所需样本和索引。识别模型保持原版本；CPU 依赖新增锁定的 torchvision，与开发 CUDA 环境使用相同预处理后端。详见 [运行环境一致性](runtime_parity_zh.md)。

图库日期固定为 **2026-10-09**，来自新增装备样本的 `downloaded_at`，不是安装或索引构建时间。新增 879 张可调节头饰状态图；来源、样本数与索引范围见 `manifests/galleries/gallery-20261009-visor-feet-v3.json`。

## 模型下载失败

程序只尝试清单中固定且有 SHA-256 校验的模型文件。能访问官方源时使用官方源，否则尝试已核实的魔搭源；不会搜索或替换成其他模型。姿态模型尚无已验证的魔搭备用源。

按照报错中的目标目录手动放置文件，然后重新运行。目录位于便携包内：

| 模型 | 目标目录 |
| --- | --- |
| DINOv2 | `models/facebook--dinov2-base/` |
| SigLIP2 | `models/google--siglip2-base-patch16-224/` |
| VitPose | `models/usyd-community--vitpose-base-simple/` |
| DINOv3 | `models/facebook--dinov3-vitb16-pretrain-lvd1689m/` |

准确文件名、下载链接、大小和 SHA-256 见 `app/v0.2.0-beta.6/manifests/models.json`。必须包含该模型列出的权重、配置和处理器/词表文件。不要只放一个权重文件。DINOv3 等第三方模型仍适用各自许可。

运行依赖下载失败时，将提示的同名 `.whl` 文件放入 `work/downloads` 后重试。准确链接和校验值见同目录的 `manifests/runtime.json`。不需要手动运行 pip。

## 以后更新

程序不会检查新版本或连接 GitHub 获取更新。程序更新优先使用对应 Release 的 `update.zip` 与内附脚本；不要把更新脚本包放进 `imports`。

图库更新或仅程序内容更新：关闭服务，把签名的资源 ZIP 或 `app.zip` 放入 `imports`，双击 `导入更新.bat`。程序与图库都要升级时一起导入配套包；移走旧同类 ZIP。Web 页面不再提供导入入口或安装接口。

完整校验和自检通过后才切换版本；`回退.bat` 可恢复上一组合。配置、上传图片和历史报告保持在 `user`。旧版本保留用于回退，升级需要额外磁盘空间。

## 修复与报告

- 双击 `检查修复.bat`：完整校验资源与模型，重新准备运行依赖。图库损坏时需重新导入原 ZIP；请按报错提示操作。
- HTML、JSON 是基础报告。PNG 使用现有 Edge/Chrome/Chromium 导出；没有兼容浏览器或导出失败时只记录 warning，识别结果仍有效，不会下载浏览器。
- 程序日志在命令行及 `work/logs`。反馈问题时附上页面版本、图库日期和错误日志。

本版支持 CPU 环境及可选的独立 GPU 环境；不要直接复制系统 Python 或 Conda 环境覆盖 `runtime`。

## 维护者构建

运行 `scripts/prepare_portable_features.py` 准备 HEAD 候选侧资源，然后运行 `scripts/build_portable_release.py`。
程序 ZIP 位于 `dist`；图库 ZIP 和 SHA-256 位于项目根目录。图库只包含活动候选索引引用的装备图片、索引、自训练权重、装备元数据和候选特征，不包含玩家截图或基础模型。

发布私钥在 `work/signing/portable-private.xml`，必须另行安全备份。私钥不能上传仓库、Release 或公开网盘。后续更新使用该私钥签名；公钥随启动器分发。运行依赖清单和模型下载清单修改后应重新验证首次准备及离线推理。

仅重建程序包并复用旧图库：`python scripts/build_portable_release.py --app-only`。
私钥缺失时构建器会停止，不会静默换钥。确实丢失私钥且需要重新发布时，可显式运行
`scripts/sign_portable.ps1` 的 `-InitializeKey -RotateKey`；它会保留旧公钥并生成新私钥。
此操作需要指定用于初始签名的 `-Manifest`、新私钥 `-KeyPath` 和已有公钥 `-PublicPath`。
新启动器随 update.zip 和 windows.zip 分发；已有用户使用更新脚本自动替换。

beta.6 以 Pre-release 发布，复用 beta.5 的图库。已有该图库的用户只需升级程序。

## beta.6 安装环境与自动升级

双击 `安装环境.bat`（或在 `导入更新.bat` 中选择）：

1. **安装cpu环境（推荐）**：只准备 CPU 环境，无需 NVIDIA 显卡。
2. **额外安装gpu环境（cpu环境+gpu环境）**：先保证 CPU 可用，再追加 GPU 环境；模型和图库共用。

已有 CPU 环境时，关闭服务后重跑 `安装环境.bat` 并选择 2 即可追加 GPU，无需重装、重新导入图库或重复下载模型。已完成且校验通过的依赖会复用，中断下载可续传。选择 1 不删除已有 GPU 环境。

GPU 安装自动检测第一张 NVIDIA 显卡及驱动：Turing 及以后架构、Windows 驱动至少 580.88 时选 CUDA 13.0；兼容的旧架构或旧驱动选择 CUDA 12.6（驱动至少 560.76，不支持 Blackwell）。不支持的硬件保留 CPU 并显示原因。安装后验证实际 CUDA 矩阵和卷积运算；具体显卡仍须通过模型推理验收。AMD/Intel GPU 暂不支持。

CPU 与 GPU 的 torch/torchvision 核心版本保持 2.11.0/0.26.0，分别安装在 `runtime/envs`。网页和 CPU 任务使用 CPU 环境，GPU 任务使用独立解释器；GPU 加载失败不影响 CPU，任务失败后可手动选择 CPU 重新识别，不会静默自动重试。GPU 环境额外占用数 GiB，含下载缓存建议预留至少 12 GiB；安装器还会按清单检查可用空间。

Python、运行库、模型、下载包、临时文件和缓存均位于项目根目录。系统必须已安装兼容 NVIDIA 驱动；本项目不安装驱动或系统级 CUDA Toolkit。

升级示例：`D:\FFXIV-Prism\FFXIV-Prism-Update\更新.bat`。请先关闭服务，将升级包解压后的文件夹放入原项目根目录，再执行该 BAT。不要将更新文件夹内容覆盖到根目录，也不要放进 `imports`。**更新成功并确认可以启动后，可删除本次解压的 `FFXIV-Prism-Update` 文件夹**；用于恢复的备份保留在项目根目录的 `work/upgrade-backups`。更新器不会安装系统组件。
