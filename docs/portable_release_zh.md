# Windows 便携正式版 v0.2.0

适用于 Windows x64，无需安装 Python、Conda 或 Git。所有项目管理的解释器、运行库、模型、图库、下载、临时文件和缓存位于安装根目录。系统显卡驱动由用户已有系统提供，项目不安装驱动或系统级 CUDA Toolkit。

## 首次安装

1. 下载 `FFXIV-Prism-v0.2.0-windows.zip`，解压到短英文路径，例如 `D:\FFXIV-Prism`。根目录绝对路径最多 **31 个字符**（含盘符和分隔符），不要直接在 ZIP 内运行。
2. 从维护者网盘下载 `FFXIV-Prism-gallery-20261009.zip`，原样放入 `imports`。
3. 双击 `导入更新.bat`，选择 **安装cpu环境（推荐）** 或 **额外安装gpu环境（cpu环境+gpu环境）**。等待图库验签、依赖和模型准备、自检完成。
4. 双击 `启动.bat`，将窗口中的 `http://127.0.0.1:8765` 复制到浏览器。按 Ctrl+C 停止服务，报告在 `user/reports`。

程序和模型准备完成后可以离线使用。GPU 环境独立安装，CPU 保留，模型和图库共用。已有 CPU 的用户关闭服务后重跑 `安装环境.bat` 并选择 GPU，即可追加安装。选择 CPU 不删除已安装 GPU。

GPU 自动按第一张 NVIDIA 显卡和驱动选择 cu130 或 cu126：支持的 Turing 及以后架构且驱动至少 580.88 使用 cu130；旧架构或兼容旧驱动使用 cu126（驱动至少 560.76，不能用于 Blackwell）。AMD/Intel GPU 暂不支持。安装后做 CUDA 矩阵与卷积自检；GPU 失败时 CPU 仍可用，可手动选择 CPU 重试，不会静默回退。

核心版本固定为 torch 2.11.0、torchvision 0.26.0、Transformers 5.12.1、NumPy 2.4.6、Pillow 11.1.0。额外 GPU 安装请预留至少 12 GiB 临时空间；完成后会清理 wheel 下载包。运行契约见 [运行环境一致性](runtime_parity_zh.md)。

## 从 beta.1 至 beta.6 升级

1. 关闭旧版服务，下载 `FFXIV-Prism-v0.2.0-update.zip`。
2. 将解压出的 `FFXIV-Prism-Update` 文件夹直接放在原安装根目录下，例如 `D:\FFXIV-Prism\FFXIV-Prism-Update`。
3. 执行该文件夹中的 `更新.bat`，自动定位父目录，无需输入路径。它验证签名、备份启动器、安装程序并自检，失败恢复原启动器和活动版本。
4. 在原根目录运行 `启动.bat`。**确认更新成功并能启动后，可删除本次解压的 `FFXIV-Prism-Update` 文件夹。**

不要用 windows ZIP 覆盖原安装，不要把更新文件夹放进 imports，也不要覆盖根目录。上一版本和启动器备份在 `work/upgrade-backups` 等项目内位置保留，用于回退；`回退.bat` 恢复上一程序/图库组合。正式版复用 beta.5/beta.6 图库，无需重新下载。

## 资源包改名与自动清理

`FFXIV-Prism-gallery-20261009.zip` 与旧名 `FFXIV-Prism-gallery-20261009-visor-feet-v3.zip` 内容、SHA-256 相同，外部文件名不参与签名验证。已经上传网盘的旧包**只需改名，不必重新上传**。不要修改包内文件或重新压缩。已安装旧包也不必重新导入。

程序和图库导入完成并自检成功后，自动删除 imports 内对应 ZIP；导入或自检失败则保留 ZIP 供重试。可选 GPU 安装失败不影响已成功完成的 CPU 安装和资源导入。已安装的依赖 wheel 自动清理，未完成的 .part 下载保留。imports 之外的原始下载不删除。模型、已解压图库、推理缓存、用户报告和上一版回退文件保留。

资料库更新时间为 **2026-10-09**，表示当前可检索图片的采集日期，不是软件发布、索引构建或安装日期。

## Windows 路径限制

当前图库最长相对路径为 217 个字符。按项目保守上限 249 计算：249 − 1 个分隔符 − 217 = **31 个字符**的根目录预算。安装器同时检查最终路径、暂存路径和 wheel 解压路径；超限会在写入前报出需缩短的长度。此预算针对当前安装内容，不能保证任意第三方浏览器产生的缓存路径。

## 下载、修复和反馈

下载显示来源、进度、速度、续传和 SHA-256 校验。普通依赖来自 PyPI，固定文件的已验证备用源为华为云、清华；PyTorch 使用官方仓库。模型优先使用 Hugging Face，存在已锁定备用文件时使用 ModelScope。下载清单在 `app/v0.2.0/manifests` 的 runtime*.json、models.json；下载失败按窗口给出的目录手动放置原文件，勿换成其他版本。

`检查修复.bat` 完整校验资源并重装依赖；wheel 已清理时需要重新下载，损坏图库需要重新下载并导入资源 ZIP。模型位于 models 或清单指定的项目内目录。日志在窗口和 `work/logs`。PNG 报告使用系统已有 Edge/Chrome，无可用浏览器时保留 HTML/JSON 报告。

## 维护者

`python scripts/build_portable_release.py --app-only` 构建正式版程序、完整安装包和升级包，输出在 dist；网盘图库 ZIP 与 SHA-256 位于项目根目录。图库未改变时复用原始 ZIP，仅改中性外部名称。

签名私钥只存于本地 `work/signing`，必须备份，不能上传仓库、Release 或网盘；公钥随启动器发布。构建前提交全部发布代码，完成全量测试和包校验。源码使用说明见 [README](../README.md)、[QUICKSTART](../QUICKSTART.md)。
