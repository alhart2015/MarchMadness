"""Phase 2 driver: 22-season LOSO sweep for the GNN stage-1 peer.

For each holdout season in 2003-2025 (excluding 2020), train one
shared-parameter ``GNNStage1Peer`` on the other 21 seasons' tournament games
and emit:

  - ``output/pairwise_gnn_phase2.csv``  -- round-robin pairwise predictions
        over each holdout's tournament field. Same shape as
        ``output/pairwise_v4.csv``: columns ``season,team_a,team_b,p_a_wins``,
        asymmetric (``team_a < team_b``, one row per pair). Consumed by the
        LL-blend gate.
  - ``output/gnn_phase2_loso_per_holdout.json`` -- per-holdout summary dicts.
  - ``output/gnn_phase2_loso_summary.json``    -- weighted aggregate metrics.
  - ``output/gnn_phase2_loso_run.log``         -- full run log.

The driver mirrors ``src/loso_with_pairwise_for_team_history.py`` for memory
hygiene: pairwise rows are appended to the CSV per-holdout (not accumulated
in memory), and ``gc.collect()`` is called after each holdout. Per the
team-seed-residual experience, ``MM_PAIRWISE_OUT`` in
``enhanced_model_v3.py`` died on Windows for runs of ~6-20 seasons; if this
22-season loop crashes mid-run, fall back to running the failed seasons as
separate one-off invocations from the command line.

Usage:
    python -m src.run_gnn_phase2
    python -m src.run_gnn_phase2 --holdout-seasons 2024,2025
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

from src.gnn_stage1_peer.loso import run_phase2_one_holdout

DEFAULT_SEASONS: list[int] = [s for s in range(2003, 2026) if s != 2020]
DEFAULT_HOLDOUTS: list[int] = list(DEFAULT_SEASONS)


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [
        logging.FileHandler(log_path, mode="w"),
        logging.StreamHandler(sys.stdout),
    ]
    # Reset existing handlers so reruns within the same process do not
    # double-log into stale files.
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
    )


def aggregate(per_holdout: list[dict]) -> dict:
    """Weighted aggregate over per-holdout results."""
    n = len(per_holdout)
    total_games = sum(int(r["gnn"]["n"]) for r in per_holdout)
    denom = max(total_games, 1)
    weighted_ll = (
        sum(float(r["gnn"]["ll"]) * int(r["gnn"]["n"]) for r in per_holdout) / denom
    )
    weighted_acc = (
        sum(float(r["gnn"]["accuracy"]) * int(r["gnn"]["n"]) for r in per_holdout) / denom
    )
    return {
        "n_holdouts": n,
        "total_test_games": total_games,
        "weighted_mean_ll": weighted_ll,
        "weighted_mean_accuracy": weighted_acc,
        "max_train_minutes": max(
            (float(r["train_minutes"]) for r in per_holdout), default=0.0
        ),
        "total_train_minutes": sum(float(r["train_minutes"]) for r in per_holdout),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 2 LOSO sweep for the GNN stage-1 peer.",
    )
    parser.add_argument(
        "--holdout-seasons",
        default=",".join(str(s) for s in DEFAULT_HOLDOUTS),
        help=(
            "Comma-separated holdout seasons. "
            "Default: 2003-2025 minus 2020 (22 seasons)."
        ),
    )
    parser.add_argument(
        "--seasons",
        default=",".join(str(s) for s in DEFAULT_SEASONS),
        help=(
            "Comma-separated training-pool seasons (must include every "
            "holdout). Default: 2003-2025 minus 2020 (22 seasons)."
        ),
    )
    parser.add_argument(
        "--data-dir",
        default="data/raw/march-machine-learning-2026",
        help="Directory with MRegularSeasonCompactResults.csv and MNCAATourneyCompactResults.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory for pairwise CSV, per-holdout/summary JSON, and run log.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--decoder-hidden", type=int, default=128)
    parser.add_argument(
        "--encoder",
        choices=["sage", "edge_attr"],
        default="sage",
        help=(
            "Encoder variant: 'sage' (default; original GraphSAGE encoder, "
            "ignores edge_attr) or 'edge_attr' (GINE encoder consuming "
            "edge_attr; Phase 2 MARGINAL-row structural variant)."
        ),
    )
    parser.add_argument(
        "--run-tag",
        default="",
        help=(
            "Optional suffix appended to output filenames as '_<tag>'. "
            "Defaults to 'edge_attr' when --encoder edge_attr is set without "
            "an explicit tag, so the SAGE-encoder outputs are not clobbered."
        ),
    )
    args = parser.parse_args(argv)

    holdouts = [int(s) for s in args.holdout_seasons.split(",") if s.strip()]
    seasons = [int(s) for s in args.seasons.split(",") if s.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve run tag: explicit --run-tag wins; otherwise default to the
    # encoder name (only matters for non-default encoders so we don't clobber
    # the sage-encoder outputs).
    if args.run_tag:
        tag = args.run_tag
    elif args.encoder != "sage":
        tag = args.encoder
    else:
        tag = ""
    suffix = f"_{tag}" if tag else ""

    log_path = output_dir / f"gnn_phase2_loso_run{suffix}.log"
    pairwise_out = output_dir / f"pairwise_gnn_phase2{suffix}.csv"
    per_holdout_path = output_dir / f"gnn_phase2_loso_per_holdout{suffix}.json"
    summary_path = output_dir / f"gnn_phase2_loso_summary{suffix}.json"

    # Ensure clean pairwise output: we append per-holdout to keep memory flat.
    pairwise_out.unlink(missing_ok=True)

    setup_logging(log_path)
    logging.info(
        "Phase 2 LOSO sweep: holdouts=%s seed=%d epochs=%d patience=%d encoder=%s tag=%r",
        holdouts,
        args.seed,
        args.epochs,
        args.patience,
        args.encoder,
        tag,
    )
    logging.info("Pairwise output: %s", pairwise_out)

    per_holdout: list[dict] = []
    t_start = time.time()
    for holdout in holdouts:
        logging.info("=== Holdout season %d ===", holdout)
        try:
            result = run_phase2_one_holdout(
                data_dir=Path(args.data_dir),
                holdout_season=holdout,
                seasons=seasons,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
                decoder_hidden=args.decoder_hidden,
                epochs=args.epochs,
                lr=args.lr,
                patience=args.patience,
                seed=args.seed,
                emit_pairwise=True,
                encoder=args.encoder,
            )
        except Exception as exc:
            logging.exception("Holdout %d FAILED: %s", holdout, exc)
            raise

        # Append pairwise rows immediately (memory hygiene -- mirrors
        # src/loso_with_pairwise_for_team_history.py).
        pdf = result.pop("pairwise_df")
        pdf.insert(0, "season", holdout)
        pdf.to_csv(
            pairwise_out,
            mode="a",
            index=False,
            header=not pairwise_out.exists(),
        )

        # Slim down the per-holdout dict before storing -- the per-pair
        # predictions list is not needed in the JSON summary and it bloats
        # the file substantially over 22 seasons.
        result_lite = {k: v for k, v in result.items() if k != "predictions"}
        per_holdout.append(result_lite)

        logging.info(
            "Holdout %d: GNN LL=%.4f acc=%.3f n=%d train_minutes=%.2f best_epoch=%d epochs_run=%d",
            holdout,
            result["gnn"]["ll"],
            result["gnn"]["accuracy"],
            result["gnn"]["n"],
            result["train_minutes"],
            result["best_epoch"],
            result["epochs_run"],
        )

        # Free per-holdout state aggressively. Mirrors the
        # team-seed-residual driver's pattern.
        del result, pdf
        gc.collect()

    summary = aggregate(per_holdout)
    summary["wall_clock_minutes"] = (time.time() - t_start) / 60.0
    logging.info("=== AGGREGATE === %s", json.dumps(summary, indent=2))

    with open(per_holdout_path, "w") as f:
        json.dump(per_holdout, f, indent=2)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logging.info("Wrote %s", per_holdout_path)
    logging.info("Wrote %s", summary_path)
    logging.info("Wrote %s", pairwise_out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
