from __future__ import annotations

import argparse
import json

from coolworld.ml.evaluate import evaluate_checkpoint


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--context-len", type=int, default=12)
    p.add_argument("--pred-len", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--output", default="outputs/eval.json")
    a = p.parse_args()
    result = evaluate_checkpoint(
        a.checkpoint,
        a.dataset,
        a.manifest,
        context_len=a.context_len,
        pred_len=a.pred_len,
        batch_size=a.batch_size,
    )
    from pathlib import Path

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
