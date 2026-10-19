"""SurgeCast: encode-process-decode GNN over the tide-gauge network.

    encode   per-gauge GRU over the last 48 h of surge, tide, pressure and wind,
             tokenised into 6 h patches (one token per half tidal cycle)
    storms   each active hurricane is a node that sends one message to every
             gauge within 1000 km, conditioned on where the storm sits relative
             to that gauge and how strong, big and fast it is
    process  L rounds of attention message passing between neighbouring gauges
    decode   per-gauge MLP -> change in surge for each of the next H hours,
             conditioned on the known future astronomical tide
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .graph import edge_features, knn_edges, node_positions
from .layers import MLP, GraphAttentionLayer
from .physics import EARTH_RADIUS_KM

STORM_RADIUS_KM = 1000.0
STORM_EDGE_DIM = 9


def storm_edge_features(storm_x: torch.Tensor, storm_valid: torch.Tensor,
                        lat: torch.Tensor, lon: torch.Tensor):
    """Features for every (storm, gauge) pair. storm_x: (B, S, F) raw HURDAT2 state.

    Returns (B, S, N, STORM_EDGE_DIM) features and a (B, S, N) mask of live edges.
    """
    slat, slon = storm_x[..., 0, None], storm_x[..., 1, None]  # (B, S, 1)
    vmax, pmin = storm_x[..., 2, None], storm_x[..., 3, None]
    move_e, move_n, r34 = storm_x[..., 4, None], storm_x[..., 5, None], storm_x[..., 6, None]

    # storm position relative to each gauge, local tangent plane
    p1, p2 = torch.deg2rad(lat), torch.deg2rad(slat)
    dx = torch.deg2rad(slon - lon) * torch.cos(0.5 * (p1 + p2)) * EARTH_RADIUS_KM
    dy = torch.deg2rad(slat - lat) * EARTH_RADIUS_KM
    a = (torch.sin((p2 - p1) / 2) ** 2
         + torch.cos(p1) * torch.cos(p2) * torch.sin(torch.deg2rad(slon - lon) / 2) ** 2)
    dist = 2 * EARTH_RADIUS_KM * torch.asin(torch.sqrt(a.clamp(0, 1)))

    shape = dist.shape
    r34_known = torch.isfinite(r34).float().expand(shape)
    feats = torch.stack([
        dx / 500, dy / 500, dist / 500,
        torch.nan_to_num(vmax / 100).expand(shape),
        torch.nan_to_num((1013 - pmin) / 50).expand(shape),
        torch.nan_to_num(move_e / 20).expand(shape),
        torch.nan_to_num(move_n / 20).expand(shape),
        torch.nan_to_num(r34 / 150).expand(shape),
        r34_known,
    ], -1)
    live = storm_valid[..., None] & (dist < STORM_RADIUS_KM)
    return feats, live


class StormEncoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.mlp = MLP(STORM_EDGE_DIM, dim, dim)

    def forward(self, storm_x, storm_valid, lat, lon):
        feats, live = storm_edge_features(storm_x, storm_valid, lat, lon)
        return (self.mlp(feats) * live.unsqueeze(-1)).sum(1)  # (B, N, D)


class SurgeCast(nn.Module):
    def __init__(self, lat: np.ndarray, lon: np.ndarray, in_channels: int, horizon: int,
                 dim: int = 128, layers: int = 4, heads: int = 4, k: int = 6,
                 use_graph: bool = True, use_storms: bool = True, emb_dim: int = 16,
                 patch: int = 6):
        super().__init__()
        n = len(lat)
        src, dst = knn_edges(lat, lon, k)
        self.register_buffer("lat", torch.tensor(lat, dtype=torch.float32))
        self.register_buffer("lon", torch.tensor(lon, dtype=torch.float32))
        self.register_buffer("src", torch.tensor(src))
        self.register_buffer("dst", torch.tensor(dst))
        self.register_buffer("edge_attr", torch.tensor(edge_features(lat, lon, src, dst)))
        self.register_buffer("pos", torch.tensor(node_positions(lat, lon)))

        self.use_graph, self.use_storms, self.patch = use_graph, use_storms, patch
        self.patch_embed = nn.Linear(patch * in_channels, dim)
        self.temporal = nn.GRU(dim, dim, batch_first=True)
        self.node_emb = nn.Embedding(n, emb_dim)
        self.node_encoder = MLP(dim + emb_dim + 3 + 2, dim, dim)
        self.edge_encoder = MLP(self.edge_attr.shape[1], dim, dim)
        self.storms = StormEncoder(dim) if use_storms else None
        self.processor = nn.ModuleList(GraphAttentionLayer(dim, heads) for _ in range(layers))
        self.decoder = MLP(dim + horizon + 1, dim, horizon)

    def forward(self, batch: dict, return_attention: bool = False):
        x = batch["x"]
        B, T, N, C = x.shape
        tokens = x.permute(0, 2, 1, 3).reshape(B * N, T // self.patch, self.patch * C)
        _, h_last = self.temporal(self.patch_embed(tokens))
        h = h_last[-1].view(B, N, -1)

        static = torch.cat([self.node_emb.weight, self.pos], -1).expand(B, -1, -1)
        season = batch["season"][:, None].expand(-1, N, -1)
        h = self.node_encoder(torch.cat([h, static, season], -1))
        if self.storms is not None:
            h = h + self.storms(batch["storm_x"], batch["storm_valid"], self.lat, self.lon)

        e = self.edge_encoder(self.edge_attr).expand(B, -1, -1)
        attention = []
        for layer in self.processor:
            h, e, alpha = layer(h, e, self.src, self.dst, self.use_graph)
            attention.append(alpha)

        tide = torch.cat([batch["tide_future"].transpose(1, 2), batch["tide_now"][..., None]], -1)
        out = self.decoder(torch.cat([h, tide], -1)).transpose(1, 2)  # (B, H, N)
        return (out, attention) if return_attention else out
