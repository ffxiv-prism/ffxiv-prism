# Windows 便携测试版

本版为预发布测试版，适合 Windows x64。无需安装 Python、Conda 或 Git。首次准备需要下载 CPU 运行依赖和基础模型，完成后可以断网使用。CPU 识别耗时较长，请保留命令行窗口。

## 第一次使用

1. 下载 `FFXIV-Prism-v0.2.0-beta.2-windows.zip` 并解压到较短、可写的英文目录，例如 `D:\FFXIV-Prism`。整个存放路径不能包含中文，避免过深的目录。不要直接在压缩包里运行，也不要选择 `Source code`。
2. 从维护者提供的网盘下载 `FFXIV-Prism-gallery-20260926-head-v2.zip`，原样放入解压目录的 `imports` 文件夹。网盘链接由维护者上传后提供；测试草稿暂未填写链接。
3. 双击 `导入更新.bat`。它会校验并安装图库、下载固定版本的运行依赖和模型。网络中断后可以重新运行；不要删除 `.part` 文件。
4. 完成后双击 `启动.bat`，复制命令行中的 `http://127.0.0.1:8765` 到浏览器。不会自动打开浏览器。
5. 上传单人截图，选择部位，点击“开始识别”。页面显示当前版本和可检索图库日期。报告保存在 `user/reports`。

按命令行中的 `Ctrl+C` 停止服务。关闭后可移动整个文件夹；删除整个文件夹即可移除项目管理的文件。

## 准备状态与下载来源

四个 BAT 一打开就会显示操作名称。后续会显示扫描 ZIP、验签、解压、依赖安装、
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

## 从 beta.1 升级到 beta.2

本次修改了根目录 BAT 与 `bootstrap`；**仅导入 app.zip 不会更新这些启动器文件**。

1. 关闭旧版服务，将新版 `windows.zip` 解压到一个临时目录。
2. 将其中的 `bootstrap` 文件夹、四个 BAT 和 `使用说明.md` 复制覆盖到旧安装根目录。
   保留旧目录的 `manifests`、`runtime`、`resources`、`models`、`user` 和 `work`，
   不要用完整 ZIP 直接覆盖旧根目录的 `manifests/active.json`。
3. 将新版 `FFXIV-Prism-v0.2.0-beta.2-app.zip` 放入旧目录的 `imports`，
   移走旧的程序更新 ZIP，然后运行新版 `导入更新.bat`。
4. 完成后运行 `启动.bat`。现有图库、模型、wheel 下载缓存与历史报告可以继续使用。

本版沿用 `gallery-20260926-head-v2` 图库，不需要重新下载图库。
由于原构建机私钥不可用，beta.2 使用新发布密钥。新版 `bootstrap/public-key.json`
同时保留旧公钥以验证旧图库和回退版本；旧版启动器无法验证 beta.2，必须先完成步骤 2。

## 模型下载失败

程序只尝试清单中固定且有 SHA-256 校验的模型文件。能访问官方源时使用官方源，否则尝试已核实的魔搭源；不会搜索或替换成其他模型。姿态模型尚无已验证的魔搭备用源。

按照报错中的目标目录手动放置文件，然后重新运行。目录位于便携包内：

| 模型 | 目标目录 |
| --- | --- |
| DINOv2 | `models/facebook--dinov2-base/` |
| SigLIP2 | `models/google--siglip2-base-patch16-224/` |
| VitPose | `models/usyd-community--vitpose-base-simple/` |
| DINOv3 | `models/facebook--dinov3-vitb16-pretrain-lvd1689m/` |

准确文件名、下载链接、大小和 SHA-256 见 `app/v0.2.0-beta.2/manifests/models.json`。必须包含该模型列出的权重、配置和处理器/词表文件。不要只放一个权重文件。DINOv3 等第三方模型仍适用各自许可。

运行依赖下载失败时，将提示的同名 `.whl` 文件放入 `work/downloads` 后重试。准确链接和校验值见同目录的 `manifests/runtime.json`。不需要手动运行 pip。

## 以后更新

程序不会检查新版本或连接 GitHub 获取更新。自行获取维护者提供的签名更新包后，可以：

- 在 Web 页面点击“导入更新包”，选择 ZIP，核对版本后确认安装。停止并重新启动后刷新网页。
- 或关闭服务，把 ZIP 放入 `imports`，双击 `导入更新.bat`。程序与图库都要升级时，将两个配套包一起导入；移走此前的旧更新 ZIP，避免同时选择多个同类包。

完整校验通过后才切换版本，失败保留原版本；`回退.bat` 可恢复上一组合。配置、上传图片和历史报告保持在 `user`。旧版本暂不自动清理，升级需要额外磁盘空间。

## 修复与报告

- 双击 `检查修复.bat`：完整校验资源与模型，重新准备运行依赖。图库损坏时需重新导入原 ZIP；请按报错提示操作。
- HTML、JSON 是基础报告。PNG 使用现有 Edge/Chrome/Chromium 导出；没有兼容浏览器或导出失败时只记录 warning，识别结果仍有效，不会下载浏览器。
- 程序日志在命令行及 `work/logs`。反馈问题时附上页面版本、图库日期和错误日志。

此版本仅提供 CPU 运行依赖。GPU 组件尚未作为本次测试附件分发；不应直接复制系统 Python 或 Conda 环境覆盖 `runtime`。

## 维护者构建

运行 `scripts/prepare_portable_features.py` 准备 HEAD 候选侧资源，然后运行 `scripts/build_portable_release.py`。
程序 ZIP 位于 `dist`；图库 ZIP 和 SHA-256 位于项目根目录。图库只包含活动候选索引引用的装备图片、索引、自训练权重、装备元数据和候选特征，不包含玩家截图或基础模型。

发布私钥在 `work/signing/portable-private.xml`，必须另行安全备份。私钥不能上传仓库、Release 或公开网盘。后续更新使用该私钥签名；公钥随启动器分发。运行依赖清单和模型下载清单修改后应重新验证首次准备及离线推理。

仅重建程序包并复用旧图库：`python scripts/build_portable_release.py --app-only`。
私钥缺失时构建器会停止，不会静默换钥。确实丢失私钥且需要重新发布时，可显式运行
`scripts/sign_portable.ps1` 的 `-InitializeKey -RotateKey`；它会保留旧公钥并生成新私钥。
此操作需要指定用于初始签名的 `-Manifest`、新私钥 `-KeyPath` 和已有公钥 `-PublicPath`。
新启动器必须作为完整 windows ZIP 分发，让用户先替换启动器再导入新 app 包。

beta.2 以 Pre-release 发布。图库单独分发，已有 beta.1 图库 ZIP 可直接复用。
