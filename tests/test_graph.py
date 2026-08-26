import numpy as np

from coolworld.graph import knn_graph


def test_sparse_graph_unique_pairs():
    lat = np.array([48, 48.001, 48.002, 48.003])
    lon = np.array([7.8, 7.801, 7.802, 7.803])
    ei, ea = knn_graph(lat, lon, k=2)
    assert ei.shape[0] == 2 and ea.shape[1] == 2
    assert all(int(a) < int(b) for a, b in ei.t().tolist())
