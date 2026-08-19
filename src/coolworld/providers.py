from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class TemperatureProvider(Protocol):
    def heatmap(self, request: Any) -> Any: ...


@runtime_checkable
class UrbanGeometryProvider(Protocol):
    def fetch_geojson(self, layer: Any, where: str = "1=1") -> dict[str, Any]: ...


@runtime_checkable
class InterventionRegistry(Protocol):
    def list_interventions(self) -> dict[str, Any]: ...


@runtime_checkable
class CounterfactualModel(Protocol):
    def predict_set(self, state: Any, action: Any) -> Any: ...
