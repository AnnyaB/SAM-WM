from coolworld.grid import grid_signature_from_centroids, grid_signature_from_geojson


def test_grid_signature_matches_centroid_representation() -> None:
    fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "a",
                "properties": {"tile_id": "a"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[1.0, 2.0], [2.0, 2.0], [2.0, 3.0], [1.0, 3.0], [1.0, 2.0]]],
                },
            },
            {
                "type": "Feature",
                "id": "b",
                "properties": {"tile_id": "b"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[3.0, 4.0], [4.0, 4.0], [4.0, 5.0], [3.0, 5.0], [3.0, 4.0]]],
                },
            },
        ],
    }
    sig1 = grid_signature_from_geojson(fc)
    sig2 = grid_signature_from_centroids([("a", 1.5, 2.5), ("b", 3.5, 4.5)])
    assert sig1 == sig2
