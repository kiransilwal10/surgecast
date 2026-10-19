import numpy as np
import torch

from surgecast.layers import GraphAttentionLayer, segment_softmax
from surgecast.model import SurgeCast


def test_segment_softmax_sums_to_one_per_receiver():
    torch.manual_seed(0)
    dst = torch.tensor([0, 0, 1, 1, 1, 3])
    w = segment_softmax(torch.randn(2, 6, 4), dst, num_segments=4)
    sums = torch.zeros(2, 4, 4).index_add(1, dst, w)
    assert torch.allclose(sums[:, [0, 1, 3]], torch.ones(2, 3, 4), atol=1e-6)
    assert torch.all(sums[:, 2] == 0)  # node 2 has no incoming edges


def test_segment_softmax_is_shift_invariant_and_stable():
    dst = torch.tensor([0, 0, 0])
    logits = torch.tensor([[[1.0], [2.0], [3.0]]])
    a = segment_softmax(logits, dst, 1)
    b = segment_softmax(logits + 1e4, dst, 1)
    assert torch.allclose(a, b) and torch.isfinite(b).all()


def test_message_passing_is_permutation_equivariant():
    torch.manual_seed(0)
    n, d = 7, 16
    src = torch.tensor([0, 1, 1, 2, 3, 4, 5, 6, 2, 0])
    dst = torch.tensor([1, 0, 2, 1, 4, 3, 6, 5, 0, 2])
    layer = GraphAttentionLayer(d, heads=4).eval()
    h, e = torch.randn(2, n, d), torch.randn(2, len(src), d)
    out, _, _ = layer(h, e, src, dst)

    perm = torch.randperm(n)
    inv = torch.argsort(perm)  # new index of each old node
    out_p, _, _ = layer(h[:, perm], e, inv[src], inv[dst])
    assert torch.allclose(out_p, out[:, perm], atol=1e-5)


def _batch(b, t, n, c, h, s=2):
    torch.manual_seed(1)
    storm = torch.zeros(b, s, 7)
    storm[:, 0, :2] = torch.tensor([28.0, -82.0])
    storm[:, 0, 2:4] = torch.tensor([100.0, 960.0])
    storm[:, 0, 6] = float("nan")  # missing wind radii must not poison the output
    return {
        "x": torch.randn(b, t, n, c), "tide_now": torch.randn(b, n),
        "tide_future": torch.randn(b, h, n), "season": torch.randn(b, 2),
        "storm_x": storm, "storm_valid": torch.tensor([[True, False]] * b),
    }


def test_surgecast_shapes_and_ablation_switches():
    n = 10
    lat, lon = np.linspace(25, 35, n), np.full(n, -82.0)
    batch = _batch(3, 12, n, 8, 6)  # 12 h history = two 6 h patches
    for use_graph in (True, False):
        for use_storms in (True, False):
            m = SurgeCast(lat, lon, in_channels=8, horizon=6, dim=32, layers=2, heads=4, k=3,
                          use_graph=use_graph, use_storms=use_storms, history=12)
            out = m(batch)
            assert out.shape == (3, 6, n) and torch.isfinite(out).all()


def test_storm_only_reaches_gauges_within_range():
    n = 6
    lat = np.array([28.0, 28.5, 29.0, 40.0, 41.0, 42.0])
    lon = np.array([-82.0, -82.2, -82.4, -70.0, -70.5, -71.0])
    m = SurgeCast(lat, lon, in_channels=8, horizon=4, dim=32, layers=1, heads=4, k=2,
                  use_graph=False, use_storms=True, history=12).eval()
    batch = _batch(1, 12, n, 8, 4)
    base = m({**batch, "storm_valid": torch.zeros(1, 2, dtype=torch.bool)})
    with_storm = m(batch)
    changed = (with_storm - base).abs().amax(1)[0] > 0
    assert changed[:3].all() and not changed[3:].any()
