"""Build model-ready arrays from the raw downloads: python scripts/build_dataset.py"""

import argparse
from pathlib import Path

from surgecast.dataset import build

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    args = ap.parse_args()
    build(args.raw, args.out)
