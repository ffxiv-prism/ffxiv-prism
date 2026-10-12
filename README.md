# FFXIV Prism

在本地从 FFXIV 角色截图检索头、身体、手、腿、脚的装备外观候选。适合清晰、单人、正面或近正面的截图；候选排名需要人工确认。

## Windows 安装

当前正式版为 **v0.2.1**。无需预先安装 Python、Conda 或 Git。

1. 在 [Releases](https://github.com/ffxiv-prism/ffxiv-prism/releases/latest) 下载 `FFXIV-Prism-v0.2.1-windows.zip`，完整解压到较短的英文路径，例如 `D:\FFXIV-Prism`。根目录绝对路径最多 64 个字符。
2. 从维护者网盘获取 `FFXIV-Prism-gallery-20261009.zip`，原样放入 `imports`，不要自行解压。网盘链接由维护者提供。
3. 双击 **安装环境.bat**，选择 **1：CPU** 或 **2：CPU + GPU**。安装器先检查并导入图库，再安装依赖和模型；自检成功后清理安装下载文件。
4. 双击 **启动.bat**，在浏览器打开窗口显示的地址，上传截图。停止服务时在命令行窗口按 Ctrl+C。

首次安装需要联网，完成后可离线识别。CPU 可运行全部识别功能。GPU 为额外的独立环境，按 NVIDIA 显卡与驱动自动选择适配版本；不安装系统驱动或系统级 CUDA Toolkit。

已经安装 CPU 环境后，关闭服务，再次运行 **安装环境.bat** 并选择 **2** 即可追加 GPU。已有图库、CPU 环境和模型会复用，无需重新放入图库 ZIP。

## 修复与输出

- **修复.bat**：重新安装运行依赖，完整校验模型、程序与图库。损坏图库请重新放入配套 ZIP，再运行安装环境。
- 识别报告位于 `user/reports`；自检日志位于 `work/logs/install-check.log`。
- 安装失败时保留下载文件，重新运行可继续；成功后只清理安装器管理的 ZIP、wheel 和临时状态。
- 三个入口分别为 **安装环境.bat、启动.bat、修复.bat**。

详细步骤见 [使用说明](docs/portable_release_zh.md)。图库采集日期为 **2026-10-09**，与软件版本日期分别记录。

本版本按首次正式发布提供全新安装包，不提供升级包或旧版迁移。源码运行方式见 [源码快速开始](QUICKSTART.md)。
