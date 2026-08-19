from __future__ import annotations

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings.

    Secrets are server-side only. `SecretStr` prevents accidental plain-text
    representation in logs or exception messages.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    fortyguard_api_key: SecretStr | None = Field(default=None, alias="FORTYGUARD_API_KEY")
    evidence_dir: Path = Field(default=Path("evidence"), alias="COOLWORLD_EVIDENCE_DIR")
    context_bundle: Path = Field(
        default=Path("data/processed/urban_thermal_sequences.npz"),
        alias="COOLWORLD_CONTEXT_BUNDLE",
    )
    context_manifest: Path = Field(
        default=Path("data/processed/urban_thermal_sequences.manifest.json"),
        alias="COOLWORLD_CONTEXT_MANIFEST",
    )
    http_timeout_seconds: float = Field(
        default=45.0,
        alias="COOLWORLD_HTTP_TIMEOUT_SECONDS",
        gt=0,
    )

    @property
    def has_fortyguard_key(self) -> bool:
        return self.fortyguard_api_key is not None and bool(
            self.fortyguard_api_key.get_secret_value().strip()
        )
