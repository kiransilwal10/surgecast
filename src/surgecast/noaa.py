"""Download helpers for NOAA CO-OPS (tide gauges) and NHC HURDAT2 (hurricane tracks).

Everything is cached under ``data/raw`` as one float32 ``.npy`` per station and
product, aligned to a single hourly UTC index, so the download can be resumed
and the processing step never has to touch the network.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

API = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
MDAPI = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json"
HURDAT2_URL = "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2025-092326.txt"
APP_NAME = "surgecast"

ATLANTIC_GULF_STATES = {
    "ME", "NH", "MA", "RI", "CT", "NY", "NJ", "DE", "PA", "MD", "VA", "DC",
    "NC", "SC", "GA", "FL", "AL", "MS", "LA", "TX",
}

# product name -> number of values stored per hour
PRODUCTS = {"hourly_height": 1, "predictions": 1, "air_pressure": 1, "wind": 2}

_session = requests.Session()

# The API sits behind an AWS gateway that answers 403 when hit too hard, so all
# threads share one rate limit and back off for a long time when throttled.
MIN_INTERVAL_S = 0.5
_rate_lock = threading.Lock()
_last_request = [0.0]


def _throttle() -> None:
    with _rate_lock:
        wait = _last_request[0] + MIN_INTERVAL_S - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_request[0] = time.monotonic()


def hourly_index(start_year: int, end_year: int) -> pd.DatetimeIndex:
    return pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31 23:00", freq="h")


def list_stations() -> list[dict]:
    """Tidal water-level stations on the US Atlantic and Gulf coasts."""
    r = _session.get(MDAPI, params={"type": "waterlevels"}, timeout=60)
    r.raise_for_status()
    out = []
    for s in r.json()["stations"]:
        if not s["tidal"] or s["greatlakes"] or s["state"] not in ATLANTIC_GULF_STATES:
            continue
        if not (-98.0 < s["lng"] < -66.0 and 24.0 < s["lat"] < 46.0):
            continue
        out.append({"id": s["id"], "name": s["name"], "state": s["state"],
                    "lat": float(s["lat"]), "lon": float(s["lng"])})
    return out


def _request(params: dict, retries: int = 8) -> dict:
    params = {"application": APP_NAME, "time_zone": "gmt", "units": "metric",
              "format": "json", **params}
    for attempt in range(retries):
        _throttle()
        try:
            r = _session.get(API, params=params, timeout=120)
            if r.status_code in (403, 429) or r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code}", response=r)
            return r.json()
        except (requests.RequestException, ValueError):
            if attempt == retries - 1:
                raise
            time.sleep(min(15 * 2 ** attempt, 300))
    raise RuntimeError("unreachable")


def fetch_year(station: str, product: str, year: int, datum: str = "MSL") -> pd.DataFrame | None:
    """One calendar year of hourly data, or None if the station has nothing."""
    params = {"product": product, "station": station,
              "begin_date": f"{year}0101 00:00", "end_date": f"{year}1231 23:00"}
    if product in ("hourly_height", "predictions"):
        params["datum"] = datum
    if product in ("predictions", "air_pressure", "wind"):
        params["interval"] = "h"

    js = _request(params)
    rows = js.get("predictions") if product == "predictions" else js.get("data")
    if not rows:
        return None

    df = pd.DataFrame(rows)
    df["t"] = pd.to_datetime(df["t"])
    if product == "wind":
        df["s"] = pd.to_numeric(df["s"], errors="coerce")
        df["d"] = pd.to_numeric(df["d"], errors="coerce")
        return df.set_index("t")[["s", "d"]]
    df["v"] = pd.to_numeric(df["v"], errors="coerce")
    return df.set_index("t")[["v"]]


def fetch_station_product(station: str, product: str, index: pd.DatetimeIndex,
                          out_dir: Path, datum: str = "MSL") -> Path:
    """Download all years for one station/product and save an aligned .npy."""
    path = out_dir / station / f"{product}.npy"
    if path.exists():
        return path
    width = PRODUCTS[product]
    arr = np.full((len(index), width), np.nan, dtype=np.float32)
    for year in sorted(set(index.year)):
        df = fetch_year(station, product, year, datum)
        if df is None:
            continue
        df = df[~df.index.duplicated()].reindex(index[index.year == year])
        pos = index.get_indexer(df.index)
        arr[pos] = df.to_numpy(dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr.squeeze(-1) if width == 1 else arr)
    return path


def download(stations: list[str], products: list[str], index: pd.DatetimeIndex,
             out_dir: Path, workers: int = 3) -> None:
    jobs = [(s, p) for s in stations for p in products]
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(fetch_station_product, s, p, index, out_dir): (s, p)
                   for s, p in jobs}
        for f in tqdm(as_completed(futures), total=len(futures), desc="CO-OPS"):
            s, p = futures[f]
            try:
                f.result()
            except Exception as exc:  # keep going; a missing station is not fatal
                tqdm.write(f"failed {s}/{p}: {exc}")


def download_hurdat2(path: Path) -> Path:
    if not path.exists():
        r = _session.get(HURDAT2_URL, timeout=120)
        r.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(r.text)
    return path


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
