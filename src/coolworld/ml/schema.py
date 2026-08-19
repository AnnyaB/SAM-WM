from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class FeatureSchema:
    dynamic_features: tuple[str, ...]
    static_features: tuple[str, ...]
    action_features: tuple[str, ...]
    temperature_feature: str = "temperature_c"

    def __post_init__(self) -> None:
        if self.temperature_feature not in self.dynamic_features:
            raise ValueError("temperature_feature must be present in dynamic_features")
        if not self.dynamic_features:
            raise ValueError("dynamic_features cannot be empty")
        if not self.action_features:
            raise ValueError("action_features cannot be empty")

    @property
    def temperature_index(self) -> int:
        return self.dynamic_features.index(self.temperature_feature)

    def to_dict(self) -> dict[str, object]:
        return {
            "dynamic_features": list(self.dynamic_features),
            "static_features": list(self.static_features),
            "action_features": list(self.action_features),
            "temperature_feature": self.temperature_feature,
        }

    @classmethod
    def from_json(cls, path: Path) -> FeatureSchema:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            dynamic_features=tuple(raw["dynamic_features"]),
            static_features=tuple(raw.get("static_features", [])),
            action_features=tuple(raw["action_features"]),
            temperature_feature=str(raw.get("temperature_feature", "temperature_c")),
        )

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
