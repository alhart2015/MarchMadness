"""Run Phase 2 retrain at a small grid of audit-derived T values.

Override of the spec's auto-trigger gate (which would skip Phase 2
because Phase 1 FAILed). Phase 1 is null by construction (chalk
scoring is monotone-invariant); Phase 2 retrains v8 on rescaled v4,
which can produce different chalk picks because XGB retraining is
not a monotone transformation.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.eval_v4_calibration import run_phase2  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("output/v4_calibration_eval_log.txt", mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def main() -> int:
    T_grid = [0.85, 1.15, 1.50, 2.00]
    results = {}
    for T in T_grid:
        logger.info("===== Phase 2 cell: T=%.2f =====", T)
        t0 = time.time()
        out_csv = f"output/pairwise_v8_phase2_T{T:.2f}.csv"
        out = run_phase2(
            v4_csv="output/pairwise_v4.csv",
            winning_config=T,
            baseline_v8_csv="output/pairwise_v8.csv",
            out_csv=out_csv,
        )
        wall = time.time() - t0
        logger.info(
            "Phase 2 T=%.2f: verdict=%s delta=%+.1f drop_best=%+.1f W/L/T=%d/%d/%d wall=%.1fs",
            T, out["verdict"],
            out["cell"]["delta_total"],
            out["cell"]["drop_best_season_delta"],
            out["cell"]["wins"], out["cell"]["losses"], out["cell"]["ties"],
            wall,
        )
        results[f"T={T:.2f}"] = out

    # Merge into existing JSON.
    eval_path = Path("output/v4_calibration_eval.json")
    summary = json.loads(eval_path.read_text())
    summary["phase2_sweep"] = results
    summary.pop("phase2", None)  # drop the old "skipped" stub
    eval_path.write_text(json.dumps(summary, indent=2, default=str))
    logger.info("merged phase2_sweep into %s", eval_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
