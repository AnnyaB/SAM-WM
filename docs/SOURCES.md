# Sources and provenance

## Primary data

- Plein et al. Street-level weather station network in Freiburg, Germany: Curated dataset from 2022-09-01 to 2023-08-31 [L2]. Zenodo. DOI: 10.5281/zenodo.12732565. Official file MD5s used by the loader:
  - gap-filled Ta/RH CSV: `840a2f677d43b1298f50f40f0a250d98`
  - annual station statistics CSV: `4a70262921bd9a90513fe6cf25527163`

## OOD data

- Savić et al. Hourly Air Temperature Datasets from city of Novi Sad - NSUNET system. Zenodo. DOI: 10.5281/zenodo.7738094. Archive MD5: `a9e3574d500b0a621a209cc41c1d6fb8`.
- Amini et al. Comprehensive compilation and quality assessment of street-level urban air temperature measurements across European networks. Scientific Data (2026). DOI: 10.1038/s41597-026-06804-4. Dataset DOI: 10.48620/93247.

## Methodological context

- Maes et al. LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels. arXiv:2603.19312. Public code: lucas-maes/le-wm.
- C3S Data Rescue Service. Station Exchange Format (SEF) specification, used by FAIRUrbTemp.

## FortyGuard

The live integration uses the participant's own FortyGuard Temperature API credential supplied only at runtime. The repo contains no credential. Real API outputs are content-addressed locally and are excluded from Git until a sanitized evidence excerpt is intentionally added.
