"""Tropical-cyclone state from NHC HURDAT2 best tracks.

Each storm becomes a moving node in the graph. At forecast time t the model
sees the storm's position, intensity, size and recent motion (all from the
track up to t). It never sees where the storm goes next.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .physics import EARTH_RADIUS_KM, haversine_km

# per-storm channels stored for every hour
STORM_FIELDS = ("lat", "lon", "vmax_kt", "pmin_mb", "motion_east_kmh", "motion_north_kmh", "r34_nmi")


@dataclass
class Track:
    storm_id: str
    name: str
    times: pd.DatetimeIndex
    lat: np.ndarray
    lon: np.ndarray
    vmax: np.ndarray
    pmin: np.ndarray
    r34: np.ndarray


def _coord(tok: str) -> float:
    v = float(tok[:-1])
    return -v if tok[-1] in "SW" else v


def parse_hurdat2(path: Path, start_year: int, end_year: int) -> list[Track]:
    tracks: list[Track] = []
    lines = Path(path).read_text().splitlines()
    i = 0
    while i < len(lines):
        head = [t.strip() for t in lines[i].split(",")]
        storm_id, name, n = head[0], head[1], int(head[2])
        year = int(storm_id[4:8])
        rows = [[t.strip() for t in lines[i + 1 + k].split(",")] for k in range(n)]
        i += n + 1
        if not start_year - 1 <= year <= end_year:
            continue
        times = pd.to_datetime([r[0] + r[1] for r in rows], format="%Y%m%d%H%M")
        lat = np.array([_coord(r[4]) for r in rows])
        lon = np.array([_coord(r[5]) for r in rows])
        vmax = np.array([float(r[6]) for r in rows])
        pmin = np.array([float(r[7]) for r in rows])
        # 34-kt wind radius (storm size), mean over quadrants. 0 means no
        # tropical-storm-force wind in that quadrant; -999 means not reported.
        radii = np.array([[float(x) for x in r[8:12]] for r in rows])
        r34 = np.where((radii < 0).any(1), np.nan, radii.mean(1))
        pmin[pmin < 0] = np.nan
        vmax[vmax < 0] = np.nan
        # tracks occasionally repeat a timestamp (e.g. landfall records)
        keep = ~times.duplicated()
        tracks.append(Track(storm_id, name, times[keep], lat[keep], lon[keep],
                            vmax[keep], pmin[keep], r34[keep]))
    return tracks


def _interp(t_ns: np.ndarray, tp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    ok = np.isfinite(fp)
    if ok.sum() == 0:
        return np.full(len(t_ns), np.nan)
    return np.interp(t_ns, tp[ok], fp[ok])


def hourly_storm_states(tracks: list[Track], index: pd.DatetimeIndex,
                        station_lat: np.ndarray, station_lon: np.ndarray,
                        max_storms: int = 4, max_range_km: float = 1500.0):
    """Interpolate tracks to the hourly index and keep the storms nearest the coast.

    Returns ``states`` (T, S, len(STORM_FIELDS)) float32, ``valid`` (T, S) bool,
    and ``storm_idx`` (T, S) int, an index into ``tracks`` (-1 where empty).
    """
    T = len(index)
    states = np.zeros((T, max_storms, len(STORM_FIELDS)), dtype=np.float32)
    valid = np.zeros((T, max_storms), dtype=bool)
    storm_idx = np.full((T, max_storms), -1, dtype=np.int32)
    best_dist = np.full((T, max_storms), np.inf)

    t_all = index.asi8
    for k, tr in enumerate(tracks):
        lo = index.searchsorted(tr.times[0])
        hi = index.searchsorted(tr.times[-1], side="right")
        if hi <= lo:
            continue
        t = t_all[lo:hi]
        tp = tr.times.asi8
        lat, lon = _interp(t, tp, tr.lat), _interp(t, tp, tr.lon)
        vmax, pmin, r34 = _interp(t, tp, tr.vmax), _interp(t, tp, tr.pmin), _interp(t, tp, tr.r34)

        # causal motion: displacement over the previous 6 h of track
        lag = np.maximum(np.arange(len(t)) - 6, 0)
        hours = np.maximum(np.arange(len(t)) - lag, 1)
        me = np.deg2rad(lon - lon[lag]) * np.cos(np.deg2rad(lat)) * EARTH_RADIUS_KM / hours
        mn = np.deg2rad(lat - lat[lag]) * EARTH_RADIUS_KM / hours

        # distance to the nearest gauge decides which storms get a slot
        d = haversine_km(lat[:, None], lon[:, None], station_lat[None], station_lon[None]).min(1)
        feats = np.stack([lat, lon, vmax, pmin, me, mn, r34], axis=1)

        for j in np.flatnonzero(d < max_range_km):
            row = lo + j
            slot = int(np.argmax(best_dist[row]))  # the worst (farthest/empty) slot
            if d[j] < best_dist[row, slot]:
                best_dist[row, slot] = d[j]
                states[row, slot] = feats[j]
                valid[row, slot] = True
                storm_idx[row, slot] = k
    return states, valid, storm_idx
