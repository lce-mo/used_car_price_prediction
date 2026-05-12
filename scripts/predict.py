from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.train_model import train


if __name__ == "__main__":
    args = list(sys.argv[1:])
    if "--predict-test" not in args:
        args.extend(["--predict-test", "true"])
    train(args)
