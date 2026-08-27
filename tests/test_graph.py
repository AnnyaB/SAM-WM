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
