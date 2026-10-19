"""README figures from the evaluation outputs: python scripts/make_figures.py"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from surgecast import viz
from surgecast.dataset import Batcher, SurgeData
from surgecast.evaluate import LABELS
from surgecast.graph import knn_edges
from surgecast.physics import haversine_km
from surgecast.storms import parse_hurdat2
from surgecast.train import build_model, pick_device

LEAD = 12  # hours ahead shown in the case studies and the influence map

# (storm, year, place, lat, lon): the nearest gauge with data during the storm is used
CASES = [
    ("HELENE", 2024, "Cedar Key, FL", 29.13, -83.03),
    ("IDALIA", 2023, "Cedar Key, FL", 29.13, -83.03),
    ("IAN", 2022, "Fort Myers, FL", 26.65, -81.87),  # right of the eye: onshore wind
    ("IAN", 2022, "St. Petersburg, FL", 27.76, -82.63),  # left of the eye: Tampa Bay drains
    ("MILTON", 2024, "Fort Myers, FL", 26.65, -81.87),
    ("BERYL", 2024, "Galveston, TX", 29.36, -94.72),
]
ANIMATION = ("HELENE", 2024, "2024-09-25 12:00", "2024-09-27 18:00", (-90.0, -79.5, 24.0, 31.5))
ANIMATION_LEADS = (6, 12)


def storm_track(data, tracks, name, year):
    """Hourly positions of one storm, interpolated from its best track."""
    tr = next(t for t in tracks if t.name == name and t.times[0].year == year)
    hours = np.arange(data.index.searchsorted(tr.times[0]),
                      data.index.searchsorted(tr.times[-1], side="right"))
    t = data.index.asi8[hours]
    return hours, np.interp(t, tr.times.asi8, tr.lat), np.interp(t, tr.times.asi8, tr.lon), tr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/processed"))
    ap.add_argument("--hurdat", type=Path, default=Path("data/raw/hurdat2.txt"))
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--figures", type=Path, default=Path("figures"))
    args = ap.parse_args()
    RUNS, FIG = args.runs, args.figures
    FIG.mkdir(exist_ok=True)
    data = SurgeData(args.data)
    results = json.loads((args.results / "metrics.json").read_text())
    tracks = parse_hurdat2(args.hurdat, data.meta["start"], data.meta["end"])

    t_test = data.forecast_times(tuple(data.meta["splits"]["test"]), 48, 24)
    preds = {LABELS[r]: np.load(RUNS / r / "test_pred.npy", mmap_mode="r")
             for r in ("gru", "gru_storms", "gnn", "surgecast") if (RUNS / r / "test_pred.npy").exists()}
    preds["Ridge (per gauge)"] = np.load(RUNS / "ridge" / "test_pred.npy", mmap_mode="r")
    resid = np.asarray(data.residual)
    base = np.asarray(data.residual_filled)
    slow = np.asarray(data.slow)

    # 1. network map with the test-period hurricanes that hit the coast
    src, dst = knn_edges(data.lat, data.lon, 6)
    shown = []
    for name, year in dict.fromkeys((c[0], c[1]) for c in CASES):
        _, tlat, tlon, _ = storm_track(data, tracks, name, year)
        shown.append((name.title(), tlat, tlon))
    viz.network_map(data.lat, data.lon, src, dst, FIG / "network.png", shown)

    # 2. error vs lead time
    order = ["Tide table", "Persistence", "Ridge (per gauge)"] + [m for m in preds if m != "Ridge (per gauge)"]
    viz.skill_curves(results, FIG / "skill_vs_lead.png", [m for m in order if m in results["rmse_cm"]["all"]])

    # 3. case studies at a fixed lead
    cases, peaks = [], []
    for name, year, place, plat, plon in CASES:
        hours, tlat, tlon, _ = storm_track(data, tracks, name, year)
        near = hours[haversine_km(tlat, tlon, plat, plon) < 500]
        lo, hi = near[0] - 24, near[-1] + 36
        window = np.arange(lo, hi)
        dist = haversine_km(plat, plon, data.lat, data.lon)
        has = np.isfinite(resid[window]).mean(0) > 0.8
        n = int(np.argmin(np.where(has, dist, np.inf)))
        rows = window - LEAD - t_test[0]
        forecasts = {
            "Tide table": slow[window - LEAD, n],
            "Persistence": base[window - LEAD, n],
        }
        for m in ("GRU (no graph)", "SurgeCast"):
            if m in preds:
                forecasts[m] = preds[m][rows, LEAD - 1, n].astype(np.float32)
        obs = resid[window, n]
        cases.append({"title": f"Hurricane {name.title()} ({year}) at {data.stations[n]['name']}, "
                               f"{data.stations[n]['state']}",
                      "times": data.index[window], "observed": obs, "forecasts": forecasts})
        i_peak = int(np.nanargmax(np.abs(obs)))
        peaks.append({"storm": f"{name.title()} {year}", "gauge": data.stations[n]["name"],
                      "observed_peak_m": float(obs[i_peak]),
                      **{f"{m} ({LEAD} h)": float(f[i_peak]) for m, f in forecasts.items()}})
    viz.case_studies(cases, FIG / "case_studies.png", LEAD)
    (args.results / "peaks.json").write_text(json.dumps(peaks, indent=1))

    # 4. animation of one storm across the whole network
    name, year, t_start, t_end, extent = ANIMATION
    frames = np.arange(data.index.get_loc(pd.Timestamp(t_start)), data.index.get_loc(pd.Timestamp(t_end)) + 1)
    hours, tlat, tlon, _ = storm_track(data, tracks, name, year)
    pos = {h: i for i, h in enumerate(hours)}
    s_lat = np.array([tlat[pos[f]] if f in pos else np.nan for f in frames])
    s_lon = np.array([tlon[pos[f]] if f in pos else np.nan for f in frames])
    in_box = ((data.lon > extent[0]) & (data.lon < extent[1]) &
              (data.lat > extent[2]) & (data.lat < extent[3]))
    obs = resid[frames] - slow[frames]
    obs[:, ~in_box] = np.nan
    fcs = {}
    for lead in ANIMATION_LEADS:
        fc = preds["SurgeCast"][frames - lead - t_test[0], lead - 1].astype(np.float32) - slow[frames - lead]
        fc[:, ~in_box] = np.nan
        fcs[f"SurgeCast, {lead} h ahead"] = fc
    viz.storm_animation(data.index[frames], data.lat, data.lon, obs, fcs, s_lat, s_lon,
                        f"Hurricane {name.title()} ({year})", FIG / "helene.gif", extent)

    # 5. which gauges does the forecast at the Helene landfall gauge listen to?
    helene = cases[[c[0] for c in CASES].index("HELENE")]
    n_target = next(i for i, s in enumerate(data.stations) if helene["title"].endswith(f"{s['name']}, {s['state']}"))
    t_peak = data.index.get_loc(helene["times"][int(np.nanargmax(np.abs(helene["observed"])))])
    device = pick_device()
    ckpt = torch.load(RUNS / "surgecast" / "model.pt", map_location="cpu", weights_only=False)
    model = build_model(ckpt["config"], data).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    batch = Batcher(data, 48, 24, device)(np.array([t_peak - LEAD]))
    x = batch["x"].clone().requires_grad_(True)
    out = model({**batch, "x": x})
    out[0, LEAD - 1, n_target].backward()
    # gradient x input over the physical channels (surge, pressure, wind stress), per gauge
    physical = [0, 3, 5, 6]
    influence = (x.grad * x)[..., physical].abs().sum((0, 1, 3)).detach().cpu().numpy()
    np.save(RUNS / "surgecast" / "helene_influence.npy", influence)
    influence[n_target] = 0.0  # the gauge's own history dominates; the map shows everyone else
    viz.influence_map(data.lat, data.lon, influence, n_target,
                      f"What the {LEAD} h Helene forecast at {data.stations[n_target]['name']} looks at",
                      FIG / "influence_helene.png", (-90.0, -76.0, 24.0, 34.0))
    print(json.dumps(peaks, indent=1))


if __name__ == "__main__":
    main()
