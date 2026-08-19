from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from coolworld.ml.trainer import TrainConfig, train_world_model


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    resolved = OmegaConf.to_container(cfg, resolve=True)
    manifest = train_world_model(TrainConfig(**resolved))  # type: ignore[arg-type]
    print(OmegaConf.to_yaml(manifest))


if __name__ == "__main__":
    main()
