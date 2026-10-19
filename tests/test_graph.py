import numpy as np

from surgecast.graph import edge_features, knn_edges, node_positions


def _coast(n=12):
    lat = np.linspace(25, 40, n)
    lon = -80 + 0.3 * np.sin(np.arange(n))
    return lat, lon


def test_knn_graph_is_symmetric_without_self_loops():
    lat, lon = _coast()
    src, dst = knn_edges(lat, lon, k=3)
    pairs = set(zip(src.tolist(), dst.tolist()))
    assert all((d, s) in pairs for s, d in pairs)
    assert all(s != d for s, d in pairs)
    assert np.bincount(dst, minlength=len(lat)).min() >= 3


def test_edge_features_flip_with_direction():
    lat, lon = _coast()
    src, dst = knn_edges(lat, lon, k=2)
    f = edge_features(lat, lon, src, dst)
    rev = {(s, d): i for i, (s, d) in enumerate(zip(src, dst))}
    for i, (s, d) in enumerate(zip(src, dst)):
        j = rev[(d, s)]
        assert np.allclose(f[i, :2], -f[j, :2], atol=1e-3)  # offset reverses
        assert np.isclose(f[i, 2], f[j, 2])  # distance does not


def test_node_positions_on_unit_sphere():
    lat, lon = _coast()
    assert np.allclose(np.linalg.norm(node_positions(lat, lon), axis=1), 1.0, atol=1e-6)
