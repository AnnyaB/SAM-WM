from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

EXPECTED = {
    "paper_suite/paper_suite_results.json": "434e2d1846c9652e07c6aef055812e4333fe99d030e384983bcb40dcae06a0f6",
    "paper_suite/FINAL_MANIFEST.json": "62f893b9895be0d227bbae130a9458a3fae67fa7420eec0141c4155fd253774f",
    "paper_suite/figures/efficiency.svg": "94e961ea4c60afb7862247b3c80d5b47713300798deb4e90968d3ea77e52c657",
    "paper_suite/figures/forecast_and_calibration.svg": "69d3f76f18a98f313daa97b4c04bc22bce3309f1a2bad7145f711bfb791e9924",
    "paper_suite/figures/forecast_trace.svg": "c1e2ddbcdb7938c2b63bb90742663ce596a3ee30e5b1879a2ad048bef6a9d186",
    "paper_suite/figures/learning_curves.svg": "be6d09a227465cc65ee5659bd7e418f6c7ac7f8b1ab6cdb3605a26859f2787b8",
    "paper_suite/figures/main_horizon_results.svg": "5a4d21b62c0389b0ba39cbdcc672345b0e48a36af524618d70fe410170fb1ea7",
    "paper_suite/figures/samwm_ablations.svg": "00b933ad209e671816fc8ec31dcda93581da1d1edd2cbad02db8634c67f98567",
    "paper_suite/runs/samwm/seed_42/best.pt": "d29e2939f86e7d6961dd16b6d2e5e20a2868d1c003825ef5c0ad2eae996f18dc",
}

TARGETS = {
    "paper_suite/paper_suite_results.json": Path("results/paper_suite/paper_suite_results.json"),
    "paper_suite/FINAL_MANIFEST.json": Path("results/paper_suite/FINAL_MANIFEST.json"),
    "paper_suite/figures/efficiency.svg": Path("results/paper_suite/figures/efficiency.svg"),
    "paper_suite/figures/forecast_and_calibration.svg": Path(
        "results/paper_suite/figures/forecast_and_calibration.svg"
    ),
    "paper_suite/figures/forecast_trace.svg": Path(
        "results/paper_suite/figures/forecast_trace.svg"
    ),
    "paper_suite/figures/learning_curves.svg": Path(
        "results/paper_suite/figures/learning_curves.svg"
    ),
    "paper_suite/figures/main_horizon_results.svg": Path(
        "results/paper_suite/figures/main_horizon_results.svg"
    ),
    "paper_suite/figures/samwm_ablations.svg": Path(
        "results/paper_suite/figures/samwm_ablations.svg"
    ),
    "paper_suite/runs/samwm/seed_42/best.pt": Path(
        "results/paper_suite/checkpoints/samwm_seed42_best.pt"
    ),
}

MODELS = (
    "samwm",
    "itransformer",
    "timemixer",
    "samwm_no_sigreg",
    "samwm_no_exchange",
    "samwm_no_mental_map",
    "samwm_no_residual",
    "samwm_no_rh",
)
SEEDS = (17, 29, 42, 73, 101)
FIGURES = (
    "efficiency",
    "forecast_and_calibration",
    "forecast_trace",
    "learning_curves",
    "main_horizon_results",
    "samwm_ablations",
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def verify_archive(zf: zipfile.ZipFile) -> None:
    names = set(zf.namelist())

    missing_runs: list[str] = []
    for model in MODELS:
        for seed in SEEDS:
            for filename in ("best.pt", "history.json"):
                member = f"paper_suite/runs/{model}/seed_{seed}/{filename}"
                if member not in names:
                    missing_runs.append(member)
    if missing_runs:
        preview = "\n".join(f"  - {name}" for name in missing_runs[:12])
        raise RuntimeError(
            f"Archive is not the completed 40-run suite; missing {len(missing_runs)} run artifacts:\n{preview}"
        )

    missing_figures: list[str] = []
    for stem in FIGURES:
        for suffix in ("pdf", "png", "svg"):
            member = f"paper_suite/figures/{stem}.{suffix}"
            if member not in names:
                missing_figures.append(member)
    if missing_figures:
        raise RuntimeError(
            "Archive is missing final publication figure outputs: " + ", ".join(missing_figures)
        )

    for member, expected in EXPECTED.items():
        if member not in names:
            raise RuntimeError(f"Required member missing: {member}")
        actual = sha256_bytes(zf.read(member))
        if actual != expected:
            raise RuntimeError(
                f"SHA-256 mismatch for {member}\nexpected: {expected}\nactual:   {actual}"
            )

    manifest = json.loads(zf.read("paper_suite/FINAL_MANIFEST.json"))
    if manifest.get("status") != "complete" or manifest.get("training_runs") != 40:
        raise RuntimeError("FINAL_MANIFEST.json does not declare the completed 40-run suite")


def import_exact_bytes(zf: zipfile.ZipFile, repo_root: Path) -> None:
    for member, relative_target in TARGETS.items():
        target = repo_root / relative_target
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = zf.read(member)
        target.write_bytes(payload)
        actual = sha256_bytes(target.read_bytes())
        expected = EXPECTED[member]
        if actual != expected:
            raise RuntimeError(f"Post-write checksum mismatch: {target}")
        print(f"OK  {relative_target}  {actual}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the completed SAM-WM Kaggle paper-suite ZIP and copy only the exact, "
            "judge-facing artifacts into the repository without redrawing or reserializing them."
        )
    )
    parser.add_argument("--zip", dest="zip_path", type=Path, required=True)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="SAM-WM repository root (default: inferred from this script).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    zip_path = args.zip_path.expanduser().resolve()
    repo_root = args.repo_root.expanduser().resolve()

    if not zip_path.is_file():
        raise SystemExit(f"Final paper-suite ZIP not found: {zip_path}")
    if not (repo_root / ".git").exists():
        raise SystemExit(f"Not a Git working tree root: {repo_root}")

    with zipfile.ZipFile(zip_path) as zf:
        verify_archive(zf)
        import_exact_bytes(zf, repo_root)

    print("\nVerified and imported exact final Kaggle artifacts.")
    print("No training was run and no figure was regenerated.")
    print("Review `git status --short` before committing the imported artifacts.")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, zipfile.BadZipFile) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
