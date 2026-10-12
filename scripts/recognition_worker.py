"""Serial JSON-lines worker. EOF releases all retained model resources."""
from __future__ import annotations

import json
import logging
import sys
import traceback
from collections import OrderedDict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
# Embedded CPython's ._pth isolation does not add the entry-point directory.
sys.path[:0] = [str(SCRIPT_DIR), str(SCRIPT_DIR.parent / "src")]
# Keep the Windows BLAS initialization identical to the one-shot entry point.
from recognize_glamour import main


def asset_state(models):
    from ffxiv_prism.model_assets import model_weight_files
    from ffxiv_prism.vision.outfit_regions import DEFAULT_POSE_MODEL
    rows = []
    for model_id in sorted({key[1] for key in models} | {DEFAULT_POSE_MODEL}):
        for weight in sorted(model_weight_files(model_id)):
            for path in [weight, *sorted(weight.parent.glob("*.json"))]:
                stat = path.stat()
                rows.append((str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino))
    return rows


def serve():
    models = OrderedDict()
    previous_assets = None
    for line in sys.stdin:
        request = json.loads(line)
        code = 1
        try:
            if previous_assets is not None and asset_state(models) != previous_assets:
                from ffxiv_prism.matching.asset_identity import asset_signature
                asset_signature.cache_clear()
                models.clear()
            code = main(request["argv"], shared_models=models, model_limit=8)
            previous_assets = asset_state(models)
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
        except Exception:
            traceback.print_exc()
        if code:
            models.clear()
        print("WORKER_RESULT " + json.dumps(dict(id=request["id"], return_code=code)), flush=True)
        if code:
            return  # A failed CUDA context must never be reused.


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    serve()
