"""Download NOAA tide-gauge data and NHC hurricane tracks.

    python scripts/download.py --start 2008 --end 2025

Water levels are fetched for every candidate station first; only stations with
enough record are kept, and the remaining products are fetched for those.
"""

import argparse
from pathlib import Path

import numpy as np

from surgecast import noaa


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2008)
    ap.add_argument("--end", type=int, default=2025)
    ap.add_argument("--root", type=Path, default=Path("data/raw"))
    ap.add_argument("--min-coverage", type=float, default=0.8)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    index = noaa.hourly_index(args.start, args.end)
    coops = args.root / "coops"

    candidates = noaa.list_stations()
    print(f"{len(candidates)} candidate stations")
    noaa.download([s["id"] for s in candidates], ["hourly_height"], index, coops, args.workers)

    selected = []
    for s in candidates:
        path = coops / s["id"] / "hourly_height.npy"
        if not path.exists():
            continue
        s["coverage"] = float(np.isfinite(np.load(path)).mean())
        if s["coverage"] >= args.min_coverage:
            selected.append(s)
    print(f"{len(selected)} stations with >= {args.min_coverage:.0%} water-level coverage")

    noaa.download([s["id"] for s in selected], ["predictions", "air_pressure", "wind"],
                  index, coops, args.workers)

    # a station is unusable without harmonic tide predictions
    selected = [s for s in selected
                if np.isfinite(np.load(coops / s["id"] / "predictions.npy")).mean() > 0.95]
    print(f"{len(selected)} stations with tide predictions")

    noaa.save_json({"start": args.start, "end": args.end, "stations": selected},
                   args.root / "stations.json")
    noaa.download_hurdat2(args.root / "hurdat2.txt")


if __name__ == "__main__":
    main()
