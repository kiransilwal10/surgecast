"""Turn raw gauge records into model-ready tensors.

Target: the non-tidal residual r = observed water level - NOAA harmonic tide
prediction. Storm surge lives in r. The model forecasts the change
r(t0 + h) - r(t0) for h = 1..H hours, so slow offsets (sea-level rise relative
to the tidal datum epoch, seasonal cycles) cancel out and the network spends
its capacity on the weather-driven part.

Input channels per gauge and hour (all causal, nothing after t0 leaks in):

    0  surge anomaly      (r - trailing 30-day mean) / sigma
    1  water level valid  1/0
    2  astronomical tide  tide / tide_scale
    3  inverse barometer  IB(p) / sigma, the static surge pressure alone would cause
    4  pressure valid     1/0
    5  wind stress east   asinh(tau_x / 0.05 N m^-2)
    6  wind stress north  asinh(tau_y / 0.05 N m^-2)
    7  wind valid         1/0
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import physics
from .noaa import hourly_index
from .storms import STORM_FIELDS, hourly_storm_states, parse_hurdat2

CHANNELS = ("anomaly", "wl_valid", "tide", "inv_barometer", "p_valid",
            "tau_east", "tau_north", "wind_valid")
TAU_SCALE = 0.05
MAX_FILL_HOURS = 2
MEAN_WINDOW_HOURS = 30 * 24


@dataclass
class Splits:
    train: tuple[int, int] = (2008, 2020)
    val: tuple[int, int] = (2021, 2021)
    test: tuple[int, int] = (2022, 2025)


def build(raw: Path, out: Path, splits: Splits = Splits()) -> None:
    meta = json.loads((raw / "stations.json").read_text())
    stations = meta["stations"]
    index = hourly_index(meta["start"], meta["end"])
    T, N = len(index), len(stations)

    def load(sid, product):
        return np.load(raw / "coops" / sid / f"{product}.npy")

    wl = np.stack([load(s["id"], "hourly_height") for s in stations], 1)
    tide = np.stack([load(s["id"], "predictions") for s in stations], 1)
    pres = np.stack([load(s["id"], "air_pressure") for s in stations], 1)
    wind = np.stack([load(s["id"], "wind") for s in stations], 1)  # (T, N, 2)

    # light QC: the verified products are clean, but sensors still glitch
    resid = wl - tide
    resid[np.abs(resid) > 5.0] = np.nan
    pres[(pres < 880) | (pres > 1070)] = np.nan
    speed, direction = wind[..., 0], wind[..., 1]
    bad_wind = ~((speed >= 0) & (speed < 90) & (direction >= 0) & (direction <= 360))
    speed[bad_wind], direction[bad_wind] = np.nan, np.nan

    ffill = lambda a: pd.DataFrame(a).ffill(limit=MAX_FILL_HOURS).to_numpy(np.float32)
    resid_in, pres_in = ffill(resid), ffill(pres)
    speed_in, dir_in = ffill(speed), ffill(direction)

    # trailing (causal) 30-day mean separates surge from slow sea-level variability
    slow = (pd.DataFrame(resid_in).rolling(MEAN_WINDOW_HOURS, min_periods=7 * 24).mean()
            .to_numpy(np.float32))
    anomaly = resid_in - slow

    train = (index.year >= splits.train[0]) & (index.year <= splits.train[1])
    sigma = np.nanstd(anomaly[train], axis=0).astype(np.float32)
    tide_scale = np.nanstd(tide[train], axis=0).astype(np.float32)

    ib = physics.inverse_barometer(pres_in) / sigma
    tau_e, tau_n = physics.wind_stress(speed_in, dir_in)

    x = np.zeros((T, N, len(CHANNELS)), dtype=np.float32)
    x[..., 0] = anomaly / sigma
    x[..., 1] = np.isfinite(anomaly)
    x[..., 2] = tide / tide_scale
    x[..., 3] = ib
    x[..., 4] = np.isfinite(ib)
    x[..., 5] = np.arcsinh(tau_e / TAU_SCALE)
    x[..., 6] = np.arcsinh(tau_n / TAU_SCALE)
    x[..., 7] = np.isfinite(tau_e)
    x = np.nan_to_num(x, nan=0.0)

    lat = np.array([s["lat"] for s in stations])
    lon = np.array([s["lon"] for s in stations])
    tracks = parse_hurdat2(raw / "hurdat2.txt", meta["start"], meta["end"])
    storm_x, storm_valid, storm_idx = hourly_storm_states(tracks, index, lat, lon)

    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "x.npy", x)
    np.save(out / "residual.npy", resid.astype(np.float32))  # raw, NaN = missing (targets)
    np.save(out / "residual_filled.npy", resid_in)
    np.save(out / "slow.npy", slow)
    np.save(out / "tide.npy", tide.astype(np.float32))
    np.save(out / "storm_x.npy", storm_x)
    np.save(out / "storm_valid.npy", storm_valid)
    np.save(out / "storm_idx.npy", storm_idx)
    (out / "meta.json").write_text(json.dumps({
        "start": meta["start"], "end": meta["end"],
        "stations": stations, "sigma": sigma.tolist(), "tide_scale": tide_scale.tolist(),
        "channels": CHANNELS, "storm_fields": STORM_FIELDS,
        "storms": [{"id": t.storm_id, "name": t.name} for t in tracks],
        "splits": {"train": splits.train, "val": splits.val, "test": splits.test},
    }, indent=1))


class SurgeData:
    """Processed arrays plus the hourly index, loaded from ``build`` output."""

    def __init__(self, root: Path):
        root = Path(root)
        self.meta = json.loads((root / "meta.json").read_text())
        self.index = hourly_index(self.meta["start"], self.meta["end"])
        self.stations = self.meta["stations"]
        self.sigma = np.array(self.meta["sigma"], dtype=np.float32)
        self.tide_scale = np.array(self.meta["tide_scale"], dtype=np.float32)
        self.lat = np.array([s["lat"] for s in self.stations])
        self.lon = np.array([s["lon"] for s in self.stations])
        for name in ("x", "residual", "residual_filled", "slow", "tide",
                     "storm_x", "storm_valid", "storm_idx"):
            setattr(self, name, np.load(root / f"{name}.npy", mmap_mode="r"))

    @property
    def n_nodes(self) -> int:
        return len(self.stations)

    def forecast_times(self, years: tuple[int, int], history: int, horizon: int,
                       stride: int = 1) -> np.ndarray:
        """Integer positions t0 whose history and full horizon fit inside ``years``."""
        y = self.index.year
        inside = np.flatnonzero((y >= years[0]) & (y <= years[1]))
        lo = max(inside[0], history - 1 + MEAN_WINDOW_HOURS // 4)
        hi = inside[-1] - horizon
        return np.arange(lo, hi + 1, stride)


class Batcher:
    """Builds batches of forecast windows directly on the training device.

    All arrays live on the device once; a batch is pure index arithmetic, so
    there is no DataLoader and no per-sample Python work.
    """

    def __init__(self, data: SurgeData, history: int, horizon: int, device: torch.device):
        to = lambda a, dt=torch.float32: torch.as_tensor(np.array(a), dtype=dt, device=device)
        self.history, self.horizon, self.device = history, horizon, device
        self.x = to(data.x)
        self.tide = to(np.asarray(data.tide) / data.tide_scale)
        self.resid = to(data.residual)
        self.base = to(data.residual_filled)
        self.sigma = to(data.sigma)
        self.storm_x = to(data.storm_x)
        self.storm_valid = to(data.storm_valid, torch.bool)
        doy = data.index.dayofyear.to_numpy() / 365.25 * 2 * np.pi
        self.season = to(np.stack([np.sin(doy), np.cos(doy)], 1))
        self.hist_off = torch.arange(-history + 1, 1, device=device)
        self.fut_off = torch.arange(1, horizon + 1, device=device)

    def __call__(self, t0: np.ndarray | torch.Tensor) -> dict[str, torch.Tensor]:
        t0 = torch.as_tensor(t0, device=self.device, dtype=torch.long)
        th = t0[:, None] + self.hist_off
        tf = t0[:, None] + self.fut_off
        base = self.base[t0]  # (B, N) last available residual, metres
        target = self.resid[tf]  # (B, H, N)
        y = (target - base[:, None]) / self.sigma
        mask = torch.isfinite(y)
        return {
            "x": self.x[th],  # (B, T_in, N, C)
            "tide_now": self.tide[t0],  # (B, N)
            "tide_future": self.tide[tf],  # (B, H, N)
            "season": self.season[t0],  # (B, 2)
            "storm_x": self.storm_x[t0],  # (B, S, F)
            "storm_valid": self.storm_valid[t0],  # (B, S)
            "y": torch.nan_to_num(y),
            "mask": mask,
            "base": torch.nan_to_num(base),
            "t0": t0,
        }
