# 首次发布与分支管理

GitHub 的公开/私有设置作用于整个仓库，不能只公开 main、隐藏 dev。
公开仓库中的分支、标签和可访问提交历史都属于公开内容，删除分支不等于删除历史。
来源：[GitHub 仓库可见性](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/managing-repository-settings/setting-repository-visibility)。

当前两个临时分支 `codex/portable-preview`、`codex/portable-download-progress` 均指向
`dbab7e0`，已完整包含在 main/dev 的 `7eb689a` 中，没有继续保留的必要。
功能分支合并后删除；main 保持可发布，dev 汇总下一版改动，正式版本使用标签。
若维护成本更重要，也可采用 main 加短期功能分支，省去长期 dev。

若训练代码和实验历史不打算公开，推荐保留私有开发仓库，另建公开推断仓库或发布仓库：

1. 从审核后的推断文件白名单生成新的工作树和独立初始提交，不推送开发历史、分支或标签。
2. 公开仓库仅保留推断源码（如选择开源）、用户说明和正式版本标签；Release 附件仅放 Windows 安装包、SHA-256 和网盘图库校验值。
3. 在私有仓库继续训练开发，用构建脚本导出公开内容，不通过整库镜像推送发布。
4. GitHub 自动生成的 Source code ZIP/TAR 来自标签对应源码，不能用精简二进制附件取代它；因此直接公开当前仓库仍会公开训练源码和历史。

若接受全部源码和历史公开，可直接公开当前仓库并删除已合并临时分支。
本次整理不改变仓库可见性，也不删除其他历史 Release。

## 构建

运行 `python scripts/build_portable_release.py` 构建图库和 Windows 安装包；图库不变时可使用 `--app-only`。
`dist` 只分发 Windows ZIP 及其校验值，签名的内部 app ZIP 位于 `work/build`，不作为首次安装附件。
网盘 ZIP 及校验值在项目根目录。本次目录结构变更后必须替换网盘内容，不能只改名。
签名私钥仅保存在本地 `work/signing`，不进入任何发布包。

推断源码由入口的本地导入依赖闭包选择；图库按显式必需文件清单构建。
目录压缩只发生在构建副本中，源码数据、实验记录和训练资产保留。
新布局采用独立兼容标识 `prism-inference-v1`，不接受旧测试图库。
