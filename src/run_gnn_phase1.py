"""Phase 1 driver: run GNN-vs-Massey sanity check across multiple test seasons.

Usage:
    python -m src.run_gnn_phase1 --seasons 2018,2019,2021,2022,2024
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src.gnn_stage1_peer.training import run_phase1_one_season


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.FileHandler(log_path, mode="w"),
        logging.StreamHandler(sys.stdout),
    ]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=handlers)


def aggregate(per_season: list[dict], gate_threshold: float = 0.005) -> dict:
    n_pass = sum(1 for r in per_season if r["compare"]["gate_pass"])
    mean_ll_delta = sum(r["compare"]["ll_delta"] for r in per_season) / max(len(per_season), 1)
    return {
        "n_seasons": len(per_season),
        "n_pass": n_pass,
        "mean_ll_delta": mean_ll_delta,
        "gate_threshold": gate_threshold,
        "verdict": "PASS" if mean_ll_delta >= gate_threshold else "FAIL",
        "max_train_minutes": max((r["train_minutes"] for r in per_season), default=0.0),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", default="2018,2019,2021,2022,2024",
                        help="Comma-separated test seasons.")
    parser.add_argument("--data-dir", default="data/raw/march-machine-learning-2026")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    seasons = [int(s) for s in args.seasons.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "gnn_phase1_diagnostic.log"
    setup_logging(log_path)
    logging.info(f"Phase 1 sweep: seasons={seasons}, seed={args.seed}")

    per_season: list[dict] = []
    for s in seasons:
        logging.info(f"=== Season {s} ===")
        result = run_phase1_one_season(
            data_dir=Path(args.data_dir),
            season=s,
            epochs=args.epochs,
            seed=args.seed,
        )
        logging.info(
            f"Season {s}: GNN LL={result['gnn']['ll']:.4f} acc={result['gnn']['accuracy']:.3f} "
            f"vs Massey LL={result['massey']['ll']:.4f} acc={result['massey']['accuracy']:.3f} "
            f"-> ll_delta={result['compare']['ll_delta']:+.4f} "
            f"({'PASS' if result['compare']['gate_pass'] else 'FAIL'}), "
            f"train_minutes={result['train_minutes']:.1f}"
        )
        per_season.append(result)

    summary = aggregate(per_season)
    logging.info(f"=== AGGREGATE === {json.dumps(summary, indent=2)}")

    with open(output_dir / "gnn_phase1_per_season.json", "w") as f:
        json.dump(per_season, f, indent=2)
    with open(output_dir / "gnn_phase1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(output_dir / "gnn_phase1_summary.txt", "w") as f:
        f.write(f"Phase 1 sweep verdict: {summary['verdict']}\n")
        f.write(f"Mean LL delta (Massey - GNN): {summary['mean_ll_delta']:+.4f} "
                f"(gate >= +{summary['gate_threshold']:.4f})\n")
        f.write(f"Per-season passes: {summary['n_pass']}/{summary['n_seasons']}\n")
        f.write(f"Max per-season training time: {summary['max_train_minutes']:.1f} min\n")
        f.write(f"\nPer-season detail:\n")
        for r in per_season:
            f.write(
                f"  {r['season']}: GNN LL {r['gnn']['ll']:.4f} acc {r['gnn']['accuracy']:.3f} | "
                f"Massey LL {r['massey']['ll']:.4f} acc {r['massey']['accuracy']:.3f} | "
                f"delta {r['compare']['ll_delta']:+.4f} | "
                f"train_min {r['train_minutes']:.1f}\n"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
