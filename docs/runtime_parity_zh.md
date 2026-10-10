# 本地与便携环境的一致性

beta.5 统一使用 PyTorch 2.11.0、torchvision 0.26.0、Transformers 5.12.1、NumPy 2.4.6、Pillow 11.1.0。CPU 的 `+cpu` 和 CUDA 的 `+cu130` 是同一版本的设备构建，并非算法版本差异。Windows Pillow 统一使用官方 wheel 的 libjpeg-turbo 3.1.0，避免同版本 Conda 构建仍有不同 JPEG 像素。

这组选用的是两套现有环境的共同版本；没有把版本号相同误判为其中一套更旧。便携环境增加与已有 CUDA 环境匹配的 torchvision，消除 AutoProcessor 自动降级到 PIL 后端的问题。准确 CPU wheel URL、大小、SHA-256 与运行环境 ID 位于 `manifests/runtime.json`。

## 源码环境

使用 Python 3.12。CUDA 环境先从官方对应索引安装设备构建，再安装项目依赖：

```powershell
python -m pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.11.0 torchvision==0.26.0
python -m pip install -e .
python -m pip install --force-reinstall --no-deps --only-binary=:all: --index-url https://pypi.org/simple pillow==11.1.0
python scripts/check_runtime_environment.py
```

CPU 源码环境把第一条的 `cu130` 改为 `cpu`。不要在 CPU 便携版目录中直接执行这些 pip 命令；便携版通过正式升级包安装清单锁定的依赖。原 Conda 环境中相同版本的 Pillow 可能来自不同构建，所以这里明确重新安装官方 wheel。

## 推断约定

- 所有设备均由 CPU 估计姿态，避免微小热图误差跨过整数裁剪边界。
- 供识别的裁剪保存为无损 PNG；预览图仍可使用 JPEG。
- SigLIP 图像处理明确指定 torchvision 后端，CPU/GPU 均在同一路径完成预处理。
- 排名相关的查询编码与样式分类使用 IEEE FP32，不再仅在 CUDA 样式分类中使用 FP16；明确关闭 CUDA 矩阵与 cuDNN 卷积的 TF32，避免张量标为 FP32 但内部卷积仍使用较低精度。
- 余弦分数按六位小数进行稳定并列排序，保留图库行顺序，避免后端排序差异经 RRF 放大。报告仍保留原始分数。
- 运行前检查数值相关依赖及 Windows JPEG 构建，报告写入实际环境、预处理和精度信息。

跨设备浮点数不保证逐位相等。验收比较裁剪像素、各部位前 20 候选身份与顺序，并记录分数误差；不能将单个输入通过理解成所有硬件、所有输入必然完全一致。

```powershell
python scripts/check_runtime_environment.py
python scripts/compare_inference_reports.py data/outputs/cpu/results.json data/outputs/cuda/results.json --output work/parity.json
```

现有图库向量与固定候选局部特征由同一已发布图库提供给 CPU/CUDA；本次不依据待识别答案重建或调整权重。查询裁剪从 JPEG 改为 PNG 后，名次可能与 beta.4 变化，应比较同版程序、同版图库、同输入和相同候选数量的结果。
