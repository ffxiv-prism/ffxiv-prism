"""Numerical environment shared by source checkouts and portable releases."""
from importlib.metadata import PackageNotFoundError, version
import sys

from PIL import features

INFERENCE_CONTRACT = "ieee-fp32-torchvision-lossless-v1"

VERSIONS = {
    "torch": "2.11.0", "torchvision": "0.26.0", "transformers": "5.12.1",
    "numpy": "2.4.6", "pillow": "11.1.0",
}


def runtime_info():
    packages = {}
    for name in VERSIONS:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {
        "contract": INFERENCE_CONTRACT,
        "packages": packages,
        "jpeg_turbo": features.version_feature("libjpeg_turbo"),
        "processor_backend": "torchvision", "pose_device": "cpu",
        "crop_format": "png", "precision": "ieee-float32", "cuda_tf32": False,
        "cosine_tie_decimals": 6,
    }


def validate_runtime():
    info = runtime_info()
    errors = []
    for name, expected in VERSIONS.items():
        actual = info["packages"][name]
        if not actual or actual.split("+", 1)[0] != expected:
            errors.append(f"{name}: expected {expected}, found {actual or 'missing'}")
    if sys.platform == "win32" and info["jpeg_turbo"] != "3.1.0":
        errors.append("Pillow must use the locked official wheel (libjpeg-turbo 3.1.0)")
    if errors:
        raise RuntimeError("Inference environment differs from the release contract: "
                           + "; ".join(errors)
                           + ". See docs/runtime_parity_zh.md.")
    return info
