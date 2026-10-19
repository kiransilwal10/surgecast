"""Test-set evaluation against classical baselines.

    python -m surgecast.evaluate --runs gru gru_storms gnn surgecast

Every method forecasts the non-tidal residual at t0 + h; adding the
astronomical tide turns that into a total water level, so residual error and
water-level error are the same number.

Baselines
    Tide table   NOAA's harmonic prediction plus the trailing 30-day mean
                 residual (sea-level offset). No weather information at all.
    Persistence  whatever the residual is now, it stays.
    Ridge        per-gauge linear regression on the same 48 h inputs
                 (surge, tide, pressure, wind stress) as the neural models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view

from .dataset import Batcher, SurgeData
from .train import build_model, pick_device, predict, storm_distance

LEADS = (1, 3, 6, 12, 24)
LABELS = {"gru": "GRU (no graph)", "gru_storms": "GRU + storm node",
          "gnn": "GNN (no storm node)", "surgecast": "SurgeCast"}


def _ridge_design(x_n: np.ndarray, tide_n: np.ndarray, t0: np.ndarray, history: int, horizon: int):
    win = sliding_window_view(x_n, history, axis=0)  # (T - history + 1, C, history)
    hist = win[t0 - history + 1].reshape(len(t0), -1)
    fut = sliding_window_view(tide_n, horizon)[t0 + 1]
    return np.concatenate([hist, fut, tide_n[t0, None], np.ones((len(t0), 1))], 1)


def _ridge_targets(resid_n, base_n, sigma, t0, horizon):
    fut = sliding_window_view(resid_n, horizon)[t0 + 1]
    return (fut - base_n[t0, None]) / sigma


def ridge_baseline(data: SurgeData, history: int, horizon: int, t_fit: np.ndarray,
                   t_val: np.ndarray, t_test: np.ndarray, lams=(0.1, 1.0, 10.0, 100.0)) -> np.ndarray:
    """Per-gauge ridge regression, lambda picked on the validation year. Returns (len(t_test), H, N)."""
    out = np.zeros((len(t_test), horizon, data.n_nodes), dtype=np.float32)
    tide_all = np.asarray(data.tide) / data.tide_scale
    for n in range(data.n_nodes):
        x_n, tide_n = np.asarray(data.x[:, n]), tide_all[:, n]
        resid_n, base_n = np.asarray(data.residual[:, n]), np.asarray(data.residual_filled[:, n])
        X = _ridge_design(x_n, tide_n, t_fit, history, horizon)
        Y = _ridge_targets(resid_n, base_n, data.sigma[n], t_fit, horizon)
        ok = np.isfinite(Y).all(1)
        X, Y = X[ok], Y[ok]
        Xv = _ridge_design(x_n, tide_n, t_val, history, horizon)
        Yv = _ridge_targets(resid_n, base_n, data.sigma[n], t_val, horizon)
        gram, rhs = X.T @ X, X.T @ Y
        best, best_w = np.inf, None
        for lam in lams:
            w = np.linalg.solve(gram + lam * len(X) * 1e-3 * np.eye(len(gram)), rhs)
            err = np.nanmean((Xv @ w - Yv) ** 2)
            if err < best:
                best, best_w = err, w
        out[:, :, n] = _ridge_design(x_n, tide_n, t_test, history, horizon) @ best_w
    return out


def rmse_cm(pred: np.ndarray, obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """RMSE per lead, in cm. Arrays are (B, H, N)."""
    err = np.where(mask, pred - obs, 0.0) ** 2
    return 100 * np.sqrt(err.sum((0, 2)) / np.maximum(mask.sum((0, 2)), 1))


def evaluate(runs: list[str], run_root: Path, data_dir: Path, out: Path) -> dict:
    data = SurgeData(data_dir)
    cfg0 = torch.load(run_root / runs[0] / "model.pt", map_location="cpu", weights_only=False)["config"]
    history, horizon = cfg0["history"], cfg0["horizon"]
    splits = data.meta["splits"]
    t_test = data.forecast_times(tuple(splits["test"]), history, horizon)
    t_val = data.forecast_times(tuple(splits["val"]), history, horizon, 3)
    t_fit = data.forecast_times(tuple(splits["train"]), history, horizon, 2)

    lead_idx = np.arange(1, horizon + 1)
    obs = np.asarray(data.residual)[t_test[:, None] + lead_idx]  # (B, H, N) metres
    base = np.asarray(data.residual_filled)[t_test]
    slow = np.asarray(data.slow)[t_test]
    sigma = data.sigma

    preds = {
        "Tide table": np.repeat(slow[:, None], horizon, 1),
        "Persistence": np.repeat(base[:, None], horizon, 1),
    }
    print("fitting ridge baseline")
    preds["Ridge (per gauge)"] = base[:, None] + sigma * ridge_baseline(
        data, history, horizon, t_fit, t_val, t_test)
    (run_root / "ridge").mkdir(parents=True, exist_ok=True)
    np.save(run_root / "ridge" / "test_pred.npy", preds["Ridge (per gauge)"].astype(np.float16))

    device = pick_device()
    batcher = Batcher(data, history, horizon, device)
    for name in runs:
        ckpt = torch.load(run_root / name / "model.pt", map_location="cpu", weights_only=False)
        model = build_model(ckpt["config"], data).to(device)
        model.load_state_dict(ckpt["state_dict"])
        print(f"predicting {name}")
        pred = base[:, None] + sigma * predict(model, batcher, t_test)
        np.save(run_root / name / "test_pred.npy", pred.astype(np.float16))
        preds[LABELS.get(name, name)] = pred

    mask = np.isfinite(obs) & np.isfinite(base)[:, None] & np.isfinite(slow)[:, None]
    obs0 = np.nan_to_num(obs)

    # subsets: a hurricane within 500 km at t0, and the top 1% of surge anomalies
    near = (storm_distance(data, t_test) < 500)[:, None]
    anomaly_future = obs - np.asarray(data.slow)[t_test[:, None] + lead_idx]
    train_rows = data.forecast_times(tuple(splits["train"]), history, horizon, 6)
    p99 = np.nanpercentile(np.abs(np.asarray(data.residual_filled)[train_rows]
                                  - np.asarray(data.slow)[train_rows]), 99, axis=0)
    extreme = np.abs(np.nan_to_num(anomaly_future)) > p99
    subsets = {"all": mask, "hurricane": mask & near, "extreme": mask & extreme}

    results = {"leads": list(LEADS), "n_test_windows": int(len(t_test)),
               "n_points": {k: int(v.sum()) for k, v in subsets.items()}, "rmse_cm": {}}
    for subset, m in subsets.items():
        results["rmse_cm"][subset] = {}
        for name, p in preds.items():
            r = rmse_cm(np.nan_to_num(p), obs0, m)
            results["rmse_cm"][subset][name] = {
                "by_lead": {int(l): float(r[l - 1]) for l in LEADS},
                "mean": float(np.sqrt(np.mean(r ** 2))),
            }
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(results, indent=1))
    (out / "metrics.md").write_text(markdown_tables(results))
    print(markdown_tables(results))
    return results


def markdown_tables(results: dict) -> str:
    leads = results["leads"]
    lines = []
    for subset, table in results["rmse_cm"].items():
        lines += [f"\n**{subset}** ({results['n_points'][subset]:,} gauge-hours)\n",
                  "| Model | " + " | ".join(f"{l} h" for l in leads) + " | all leads |",
                  "|---|" + "---:|" * (len(leads) + 1)]
        best = {l: min(v["by_lead"][l] for v in table.values()) for l in leads}
        for name, v in table.items():
            cells = [(f"**{v['by_lead'][l]:.1f}**" if np.isclose(v["by_lead"][l], best[l])
                      else f"{v['by_lead'][l]:.1f}") for l in leads]
            lines.append(f"| {name} | " + " | ".join(cells) + f" | {v['mean']:.1f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--run-root", type=Path, default=Path("runs"))
    ap.add_argument("--data", type=Path, default=Path("data/processed"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()
    evaluate(args.runs, args.run_root, args.data, args.out)


if __name__ == "__main__":
    main()
