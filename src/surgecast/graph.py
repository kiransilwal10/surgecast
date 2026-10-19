"""Station graph construction.

Gauges are irregularly spaced along a coastline, which is exactly the setting
where a graph beats a grid. Each gauge is connected to its k nearest
neighbours (great-circle distance), the graph is made symmetric, and every
directed edge carries the sender's position relative to the receiver. That
direction matters: coastally trapped waves travel with the coast on their right
in the northern hemisphere, so "upstream" and "downstream" neighbours are not
interchangeable.
"""

from __future__ import annotations

import numpy as np

from .physics import EARTH_RADIUS_KM, haversine_km

EDGE_SCALE_KM = 300.0


def knn_edges(lat: np.ndarray, lon: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric k-NN graph. Returns (src, dst) arrays of directed edges src -> dst."""
    n = len(lat)
    d = haversine_km(lat[:, None], lon[:, None], lat[None], lon[None])
    np.fill_diagonal(d, np.inf)
    nbrs = np.argsort(d, axis=1)[:, :k]
    adj = np.zeros((n, n), dtype=bool)
    adj[np.repeat(np.arange(n), k), nbrs.ravel()] = True
    adj |= adj.T
    dst, src = np.nonzero(adj)
    return src.astype(np.int64), dst.astype(np.int64)


def edge_features(lat: np.ndarray, lon: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Sender offset relative to the receiver: [east, north, distance] / 300 km."""
    lat0, lon0, lat1, lon1 = lat[dst], lon[dst], lat[src], lon[src]
    dx = np.deg2rad(lon1 - lon0) * np.cos(np.deg2rad(0.5 * (lat0 + lat1))) * EARTH_RADIUS_KM
    dy = np.deg2rad(lat1 - lat0) * EARTH_RADIUS_KM
    dist = haversine_km(lat0, lon0, lat1, lon1)
    return (np.stack([dx, dy, dist], axis=1) / EDGE_SCALE_KM).astype(np.float32)


def node_positions(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Unit-sphere xyz coordinates (no seam or pole artefacts, unlike raw lat/lon)."""
    la, lo = np.deg2rad(lat), np.deg2rad(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], 1).astype(np.float32)
