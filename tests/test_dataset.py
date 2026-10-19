import json

import numpy as np
import torch

from surgecast.dataset import CHANNELS, MEAN_WINDOW_HOURS, Batcher, SurgeData
from surgecast.noaa import hourly_index


def _fake_processed(root, n=4):
    index = hourly_index(2019, 2021)
    t = len(index)
    rng = np.random.default_rng(0)
    resid = rng.normal(size=(t, n)).astype(np.float32)
    resid[100, 1] = np.nan
    np.save(root / "x.npy", rng.normal(size=(t, n, len(CHANNELS))).astype(np.float32))
    np.save(root / "residual.npy", resid)
    np.save(root / "residual_filled.npy", np.nan_to_num(resid))
    np.save(root / "slow.npy", np.zeros((t, n), np.float32))
    np.save(root / "tide.npy", rng.normal(size=(t, n)).astype(np.float32))
    np.save(root / "storm_x.npy", np.zeros((t, 4, 7), np.float32))
    np.save(root / "storm_valid.npy", np.zeros((t, 4), bool))
    np.save(root / "storm_idx.npy", np.full((t, 4), -1, np.int32))
    stations = [{"id": str(i), "name": f"s{i}", "lat": 30.0 + i, "lon": -80.0} for i in range(n)]
    (root / "meta.json").write_text(json.dumps({
        "start": 2019, "end": 2021, "stations": stations,
        "sigma": [2.0] * n, "tide_scale": [1.0] * n}))
    return resid


def test_forecast_windows_stay_inside_their_years(tmp_path):
    _fake_processed(tmp_path)
    data = SurgeData(tmp_path)
    t0 = data.forecast_times((2020, 2020), history=48, horizon=24)
    years = data.index.year
    assert (years[t0] == 2020).all()
    assert (years[t0 + 24] == 2020).all()  # targets never reach into the next split
    t0 = data.forecast_times((2019, 2019), history=48, horizon=24)
    assert t0[0] >= 47 + MEAN_WINDOW_HOURS // 4


def test_batch_targets_are_future_change_in_residual(tmp_path):
    resid = _fake_processed(tmp_path)
    data = SurgeData(tmp_path)
    b = Batcher(data, history=48, horizon=24, device=torch.device("cpu"))
    t0 = np.array([90, 5000])
    batch = b(t0)
    assert batch["x"].shape == (2, 48, 4, len(CHANNELS))
    # the last history step is t0 itself, never t0 + 1
    assert torch.allclose(batch["x"][1, -1], torch.tensor(np.array(data.x[5000])))
    expected = (resid[5003] - np.nan_to_num(resid[5000])) / 2.0
    assert np.allclose(batch["y"][1, 2].numpy(), expected, atol=1e-6)
    # the missing observation at t=100 (t0=90, lead 10) is masked out of the loss
    assert not batch["mask"][0, 9, 1] and batch["mask"][0, 9, 0]
