"""Check the numerical environment before comparing devices or building a release."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ffxiv_prism.runtime_contract import validate_runtime

if __name__ == "__main__":
    print(json.dumps(validate_runtime(), ensure_ascii=False, indent=2))
