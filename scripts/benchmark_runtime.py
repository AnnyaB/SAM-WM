from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from coolworld.ml.data import UrbanThermalSequenceDataset
from coolworld.ml.future import known_future_indices
from coolworld.ml.model import ActionConditionedJEPAWorldModel


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark SAM-WM on one real sequence without fabricating input data."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--context-len", type=int, default=12)
    parser.add_argument("--pred-len", type=int, default=6)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--output", default="outputs/runtime_benchmark.json")
    args = parser.parse_args()

    dataset = UrbanThermalSequenceDataset(
        args.dataset,
        args.manifest,
        context_len=args.context_len,
        pred_len=args.pred_len,
    )
    sample = dataset[0]
    schema = dataset.manifest.schema
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    known_idx = known_future_indices(schema)
    model = ActionConditionedJEPAWorldModel(
        len(schema.dynamic_features),
        len(schema.static_features),
        len(schema.action_features),
        len(known_idx),
        latent_dim=int(cfg["latent_dim"]),
        spatial_layers=int(cfg["spatial_layers"]),
        spatial_heads=int(cfg["spatial_heads"]),
        dropout=float(cfg["dropout"]),
    )
    model.load_state_dict(checkpoint["model"])
    device = choose_device(args.device)
    model.to(device).eval()

    batch = {key: value.unsqueeze(0).to(device) for key, value in sample.items()}
    future_known = batch["future_dynamic"][..., list(known_idx)]

    def run_once() -> None:
        model(
            batch["context_dynamic"],
            batch["context_actions"],
            batch["context_mask"],
            batch["static"],
            batch["future_actions"],
            future_known,
            future_mask=batch["future_mask"],
        )

    with torch.no_grad():
        for _ in range(args.warmup):
            run_once()
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(args.runs):
            run_once()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started

    params = sum(parameter.numel() for parameter in model.parameters())
    payload = {
        "dataset_id": dataset.manifest.dataset_id,
        "device": str(device),
        "parameters": int(params),
        "runs": args.runs,
        "mean_latency_ms": 1000.0 * elapsed / args.runs,
        "context_len": args.context_len,
        "pred_len": args.pred_len,
        "tiles": int(sample["static"].shape[0]),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
