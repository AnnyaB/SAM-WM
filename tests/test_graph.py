import numpy as np

from coolworld.graph import haversine_xy, knn_graph, local_static_features


def test_local_geometry_is_finite_and_centred():
    lat = np.array([48.0, 48.001, 48.002], dtype=float)
    lon = np.array([7.8, 7.801, 7.803], dtype=float)
    elev = np.array([250.0, 260.0, np.nan], dtype=float)
    x, y = haversine_xy(lat, lon)
    features = local_static_features(lat, lon, elev)
    assert features.shape == (3, 3)
    assert np.isfinite(features).all()
    assert abs(float(x.mean())) < 1e-6
    assert abs(float(y.mean())) < 1e-6


def test_knn_graph_exposes_direction_and_distance():
    lat = np.array([48.0, 48.001, 48.002], dtype=float)
    lon = np.array([7.8, 7.801, 7.803], dtype=float)
    edge_index, edge_attr = knn_graph(lat, lon, k=2)
    assert edge_index.shape[0] == 2
    assert edge_attr.shape[1] == 3
    assert (edge_attr[:, 2] > 0).all()


def test_knn_graph_does_not_construct_dense_pairwise_distance_tensor(monkeypatch):
    real_norm = np.linalg.norm

    def guarded_norm(value, *args, **kwargs):
        if np.asarray(value).ndim >= 3:
            raise AssertionError("dense pairwise distance tensor is forbidden")
        return real_norm(value, *args, **kwargs)

    monkeypatch.setattr(np.linalg, "norm", guarded_norm)
    n = 1024
    lat = 37.33 + np.linspace(0.0, 0.08, n)
    lon = -121.95 + 0.02 * np.sin(np.linspace(0.0, 20.0, n))
    edge_index, edge_attr = knn_graph(lat, lon, k=4)

    assert edge_index.shape[1] <= n * 4
    assert edge_attr.shape == (edge_index.shape[1], 3)
    assert np.isfinite(edge_attr.numpy()).all()


def test_graph_rejects_invalid_geography():
    with np.testing.assert_raises(ValueError):
        knn_graph(np.array([0.0, np.nan]), np.array([0.0, 1.0]), k=1)
    with np.testing.assert_raises(ValueError):
        knn_graph(np.array([0.0, 91.0]), np.array([0.0, 1.0]), k=1)
