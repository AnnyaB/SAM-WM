# Source registry

All empirical source connectors are explicit and reviewable.

## FortyGuard

Base API: `https://api.fortyguard.com`

Used endpoints:

- `POST /v1/heatmap`
- `POST /v1/env_params`
- `POST /v1/satellite`
- `POST /v1/streetview`
- `POST /v1/heat_intelligence`
- `GET /v1/status/{activity_id}`

Authentication: `api-key` request header, read from the server environment.

## City of Phoenix — real implemented/planned cooling infrastructure

Cool pavement:
`https://maps.phoenix.gov/pub/rest/services/Public/STR_MainCoolPave/MapServer`

Cool corridors:
`https://maps.phoenix.gov/pub/rest/services/Public/STR_CoolCorridors/MapServer`

Shade/tree canopy:
`https://maps.phoenix.gov/pub/rest/services/Public/Shade_Study_Data_CMO_OHR/MapServer`

These official ArcGIS services support GeoJSON queries. The application fetches
runtime data from them and records the request URL + content SHA-256.


## 3D basemap / building geometry

OpenFreeMap / OpenMapTiles using data from OpenStreetMap. The UI uses real vector-tile geometry. Building extrusion is enabled only when a positive height/render-height attribute exists.

## FortyGuard heatmap 3D field

The official Quickstart's heatmap cache shows TCM tile properties including `tile_id`, `average_temperature`, `min_temperature`, and `max_temperature`. CoolWorld's observed 3D thermal layer accepts the documented `average_temperature` field and does not impute a missing value.
