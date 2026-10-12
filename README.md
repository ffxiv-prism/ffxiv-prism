# FFXIV Prism

看到喜欢的 FFXIV 装备外观，却不知道它叫什么？上传一张角色截图，FFXIV Prism 会在本地检索头、身体、手、腿、脚的装备候选，帮助你寻找相近外观。

适合清晰的单人、正面或近正面截图。遮挡、特效和拍摄角度会影响结果，请结合候选图片人工确认。

## 下载

适用于 **Windows x64**，无需预先安装 Python、Conda 或 Git。请下载以下两个文件：

| 文件 | 下载地址 | 用途 |
| --- | --- | --- |
| `FFXIV-Prism-v0.2.1-windows.zip` | [下载安装包](https://github.com/ffxiv-prism/ffxiv-prism/releases/latest) | 程序与安装入口 |
| `FFXIV-Prism-gallery-20261009.zip` | [下载装备图库（Google Drive）](https://drive.google.com/drive/folders/1G5N7_xYPplvjrjVKcpQRvAbjZO-2SzcQ?usp=sharing) | 识别所需的装备图片与检索数据 |

GitHub 自动提供的 **Source code** 是源码，请选择上表中的 Windows 安装包。

## 开始使用

1. 将 Windows 安装包**完整解压**到短英文路径，例如 `D:\FFXIV-Prism`。不要在压缩包内运行；安装路径最多 64 个字符。
2. 把图库 ZIP **原样放入解压目录中的 `imports` 文件夹**，不要解压图库，该文件夹只放一个图库 ZIP。
3. 双击 **安装环境.bat**。选择 **1：CPU** 即可使用全部识别功能；有兼容 NVIDIA 显卡的用户可以选择 **2：CPU + GPU**。首次安装需要联网，请等待安装和自检完成。依赖和模型的下载、安装提示为英文。
4. 双击 **启动.bat**，在浏览器打开窗口显示的本地地址，上传截图并开始识别。运行期间保留命令行窗口；结束时在该窗口按 **Ctrl+C**。

安装完成后可离线识别。报告保存在 `user/reports`，可查看 HTML/JSON 报告；生成 PNG 报告需要电脑上已有 Edge 或 Chrome。

## 安装遇到问题

- **提示没有图库 ZIP**：确认图库放在 `imports` 中，再运行安装环境。已经成功导入过图库时，可以直接复用，不必再次下载。
- **下载中断**：重新运行安装环境即可重试，已有下载会尽量续传。请留意窗口中的失败原因和剩余磁盘空间。
- **GPU 安装失败**：CPU 环境仍可使用。稍后关闭程序，再运行安装环境并选择 2，可继续准备 GPU。额外安装 GPU 请预留至少 12 GiB 临时空间。
- **提示程序或环境异常**：关闭程序后运行 **修复.bat**。如果提示图库文件损坏，请重新下载配套图库 ZIP，放入 `imports`，再运行安装环境。

安装成功后会清理本次导入的图库 ZIP 和安装下载文件，识别报告与已安装模型会保留。

更多步骤、下载校验和日志位置见 [使用说明](docs/portable_release_zh.md)。如需反馈问题，请在 [Issues](https://github.com/ffxiv-prism/ffxiv-prism/issues) 中附上错误提示和复现步骤。
