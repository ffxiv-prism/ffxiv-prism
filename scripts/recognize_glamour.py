from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


# Initialize Conda's BLAS/OpenMP runtime before the CLI imports CUDA PyTorch.
# This is required by the supported Windows environment and avoids the unsafe
# KMP_DUPLICATE_LIB_OK workaround.
np.linalg.inv(np.eye(2, dtype=np.float32))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffxiv_prism.cli.match import main


if __name__ == "__main__":
    raise SystemExit(main())
