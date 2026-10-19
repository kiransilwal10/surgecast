"""Training loop.

    python -m surgecast.train --config configs/surgecast.yaml
    python -m surgecast.train --config configs/surgecast.yaml --set model.use_graph=false --name gru
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .dataset import Batcher, SurgeData
from .model import SurgeCast
from .physics import haversine_km


def load_config(path: Path, overrides: list[str] = ()) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    for item in overrides:
        key, value = item.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node[p]
        node[leaf] = yaml.safe_load(value)
    return cfg


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def storm_distance(data: SurgeData, t0: np.ndarray) -> np.ndarray:
    """(len(t0), N) distance in km from each gauge to the nearest active storm (inf if none)."""
    sx, sv = np.asarray(data.storm_x[t0]), np.asarray(data.storm_valid[t0])
    d = haversine_km(sx[..., 0, None], sx[..., 1, None], data.lat, data.lon)  # (B, S, N)
    d = np.where(sv[..., None], d, np.inf)
    return d.min(1)


def event_weights(data: SurgeData, t0: np.ndarray, boost: float) -> np.ndarray:
    """Oversample hours with a storm near the coast or a large surge already under way.

    Only information available at t0 is used, so this is a sampling choice and
    not a peek at the targets.
    """
    near_storm = (storm_distance(data, t0) < 500).any(1)
    big_anomaly = np.abs(np.asarray(data.x[t0, :, 0])).max(1) > 3.0
    return 1.0 + boost * (near_storm | big_anomaly)


def masked_mse(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.float()
    return ((pred - y) ** 2 * m).sum() / m.sum().clamp_min(1.0)


def build_model(cfg: dict, data: SurgeData) -> SurgeCast:
    return SurgeCast(data.lat, data.lon, in_channels=data.x.shape[-1],
                     horizon=cfg["horizon"], **cfg["model"])


@torch.no_grad()
def predict(model: SurgeCast, batcher: Batcher, t0: np.ndarray, batch_size: int = 128) -> np.ndarray:
    """Normalised forecasts (len(t0), H, N)."""
    model.eval()
    out = []
    for i in range(0, len(t0), batch_size):
        out.append(model(batcher(t0[i:i + batch_size])).float().cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def validation_loss(model: SurgeCast, batcher: Batcher, t0: np.ndarray, batch_size: int = 128) -> float:
    model.eval()
    num, den = 0.0, 0.0
    for i in range(0, len(t0), batch_size):
        b = batcher(t0[i:i + batch_size])
        m = b["mask"].float()
        num += float((((model(b) - b["y"]) ** 2) * m).sum())
        den += float(m.sum())
    return num / max(den, 1.0)


def train(cfg: dict) -> Path:
    tc = cfg["train"]
    torch.manual_seed(tc["seed"])
    np.random.seed(tc["seed"])
    device = pick_device()
    run_dir = Path(cfg.get("run_dir", "runs")) / cfg["name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    data = SurgeData(Path(cfg["data_dir"]))
    splits = data.meta["splits"]
    batcher = Batcher(data, cfg["history"], cfg["horizon"], device)
    t_train = data.forecast_times(tuple(splits["train"]), cfg["history"], cfg["horizon"])
    t_val = data.forecast_times(tuple(splits["val"]), cfg["history"], cfg["horizon"], tc["val_stride"])
    weights = torch.as_tensor(event_weights(data, t_train, tc["event_boost"]))
    print(f"{data.n_nodes} gauges | {len(t_train)} train windows "
          f"({float((weights > 1).float().mean()):.1%} event hours) | {len(t_val)} val | {device}")

    model = build_model(cfg, data).to(device)
    print(f"{sum(p.numel() for p in model.parameters()):,} parameters")
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    total = tc["epochs"] * tc["steps_per_epoch"]
    warmup = min(500, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))

    history, best, best_state = [], float("inf"), None
    for epoch in range(tc["epochs"]):
        model.train()
        t_start, running = time.time(), 0.0
        for _ in range(tc["steps_per_epoch"]):
            idx = torch.multinomial(weights, tc["batch_size"], replacement=True).numpy()
            b = batcher(t_train[idx])
            loss = masked_mse(model(b), b["y"], b["mask"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += loss.item()
        val = validation_loss(model, batcher, t_val)
        history.append({"epoch": epoch + 1, "train": running / tc["steps_per_epoch"], "val": val,
                        "seconds": time.time() - t_start})
        flag = ""
        if val < best:
            best, best_state, flag = val, copy.deepcopy(model.state_dict()), " *"
        print(f"epoch {epoch + 1:3d}  train {history[-1]['train']:.4f}  val {val:.4f}  "
              f"({history[-1]['seconds']:.0f}s){flag}", flush=True)

    torch.save({"config": cfg, "state_dict": best_state}, run_dir / "model.pt")
    (run_dir / "history.json").write_text(json.dumps(history, indent=1))
    return run_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--set", nargs="*", default=[], help="overrides like model.use_graph=false")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    if args.name:
        cfg["name"] = args.name
    train(cfg)


if __name__ == "__main__":
    main()
