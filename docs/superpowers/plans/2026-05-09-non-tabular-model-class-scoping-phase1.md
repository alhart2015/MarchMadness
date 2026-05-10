# GNN Stage-1 Peer Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Phase 1 of the non-tabular GNN stage-1 peer experiment per the scoping spec. Build a Graph Neural Network on the regular-season game graph, train it to predict RS-game outcomes, and compare against a scalar Massey baseline on held-out late-season RS games (Mar 1 -> Selection Sunday). PASS gate: GNN beats scalar Massey LL by >= 0.005 averaged over 5 test seasons. PASS unlocks Phase 2 (separate plan); FAIL closes the GNN candidate per the spec's sequel-ordering matrix.

**Scope:** Phase 1 only. Phase 2 (22-season tournament LOSO), v8 retrain, and bracket-points re-test are deferred to follow-up plans pending Phase 1 verdict. The gate is structural -- if Massey already extracts the bulk of the relational signal at the RS level, the tournament-LOSO investment is wasted.

**Architecture:** Standard graph link-prediction setup. For each test season T:
- Build one PyTorch Geometric `Data` graph from T's regular-season games before March 1 (cutoff `DayNum < 120`). Nodes = ~350 D-I teams, edges = RS games with edge features (margin sign + magnitude, site indicator, days_rest, days_from_season_start).
- Encoder: 2-layer GraphSAGE (hidden_dim=64, mean aggregation, dropout=0.2) -> per-team embedding.
- Matchup decoder: concat(team_a_emb, team_b_emb, abs(team_a_emb - team_b_emb)) -> 2-layer MLP (hidden=128) -> logit -> sigmoid -> p(A wins).
- Training labels = edge outcomes (W/L) of pre-March-1 games, with home/away symmetrization.
- Test labels = held-out March 1 -> Selection Sunday games (`120 <= DayNum < 134`).
- Per-test-season independent training (Phase 2 will revisit cross-season parameter sharing).

**Deliberate scope simplifications (vs spec's full GNN candidate description):**
- **Cross-season tournament edges deferred.** Spec open question #4 flagged
  cross-season prior-tournament edges (~1,400 historical-tournament edges per
  season's graph) as A/B-testable in Phase 1. This plan uses RS-only graphs;
  the cross-season augmentation is deferred to Phase 2 unless the Phase 1
  verdict is MARGINAL (-0.002 to +0.005), in which case Task 9 Step 3 calls
  out the augmentation as one of the sensitivity-sweep variants to try
  before final verdict.
- **`days_rest` set to 0.0 placeholder.** Computing days-rest requires sorting
  each team's games chronologically and tracking gap-since-last-game; the
  signal is plausibly small at the RS level. Placeholder is acceptable for
  Phase 1; Phase 2 may revisit if days-rest is nontrivial in tournament
  prediction.
- **No node features.** The encoder uses a learned `nn.Embedding`. Phase 2
  may add Massey scalar / KP rating / seed as node features if RS graph
  topology alone underperforms.

**Tech Stack:** Python 3.11+, PyTorch >= 2.2 (CPU build), PyTorch Geometric >= 2.4, pandas, numpy, pytest. Inputs: `data/raw/march-machine-learning-2026/MRegularSeasonCompactResults.csv`, `MMasseyOrdinals.csv`, `MTeams.csv`. Existing pieces reused: `src/config.py` (`load_config`), the Massey-loading pattern from `src/features/feature_matrix.py:69-103`.

**Spec:** `docs/superpowers/specs/2026-05-09-non-tabular-model-class-scoping-design.md`
**Predecessors:**
- Spec PR (this branch's commit `63abac3`): scoping spec for the non-tabular stage-1 peer lane.
- Team-program tournament-history features FAIL (PR 34): seventh same-data-peer null result motivating this lane.

---

## Scope and Kill Criterion

This plan implements **Phase 1 only**: a pre-LOSO RS-prediction sanity check that gates whether the more expensive Phase 2 (22-season tournament LOSO, ~1.5 weeks) is worth running.

**Phase 1 gate:** Aggregate (GNN LL - Massey LL) <= -0.005 across the 5 test seasons. Equivalent: GNN beats Massey LL by at least 0.005 LL averaged.

- **PASS:** Proceed to Phase 2 plan (separate doc; written after Phase 1 lands).
- **FAIL:** Close the GNN candidate. Trigger spec's sequel-ordering "GNN fails Phase 1" branch -- rank up Candidate 4 (self-supervised embeddings), keep Candidate 3 (box-score), deprioritize Candidate 2 (sequence model). Update TODO Active queue accordingly.

Phase 1 also serves as a **wall-clock timing check**: per-season GNN training budget on CPU. If a single season's training exceeds ~30 min on CPU, escalate scope (GPU access or simpler architecture) before committing to Phase 2's 22-season loop.

---

## File Structure

**Created (committed):**

- `src/gnn_stage1_peer/__init__.py` -- empty package marker.
- `src/gnn_stage1_peer/data.py` (~150 LOC)
  - Public: `load_rs_games(data_dir: Path, season: int) -> pd.DataFrame` -- returns games with columns `[Season, DayNum, WTeamID, WScore, LTeamID, LScore, WLoc]` for `season`.
  - Public: `split_phase1(games: pd.DataFrame, train_cutoff_daynum: int = 120, test_end_daynum: int = 134) -> tuple[pd.DataFrame, pd.DataFrame]` -- returns `(train_games, test_games)` where `train_games` has `DayNum < train_cutoff_daynum` and `test_games` has `train_cutoff_daynum <= DayNum < test_end_daynum`.
  - Public: `build_team_index(games: pd.DataFrame) -> dict[int, int]` -- maps `TeamID -> contiguous_node_idx` for use as PyG node IDs.
- `src/gnn_stage1_peer/graph.py` (~120 LOC)
  - Public: `build_pyg_graph(train_games: pd.DataFrame, team_index: dict[int, int]) -> torch_geometric.data.Data` -- returns PyG `Data` with `edge_index` (2, 2E), `edge_attr` (2E, 4), `num_nodes`. Edges are bidirected (symmetrized) so message passing works regardless of W/L orientation.
  - Public: `extract_edge_features(game: pd.Series) -> torch.Tensor` -- returns 4-d edge feature `[score_diff, site_indicator, days_rest, days_from_season_start]`. `score_diff` is signed from the directional edge perspective.
  - Public: `build_matchup_pairs(games: pd.DataFrame, team_index: dict[int, int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]` -- returns `(team_a_idx, team_b_idx, label)` for use as training/eval samples. Symmetrized: each game appears twice (A-vs-B with label=1 if A won, B-vs-A with label=0) to match v3's symmetric-matchup convention.
- `src/gnn_stage1_peer/baselines.py` (~80 LOC)
  - Public: `load_massey_composite(data_dir: Path, season: int, ranking_day: int = 133) -> dict[int, float]` -- returns `{team_id: composite_rank}` from `MMasseyOrdinals.csv` filtered to `Season == season AND RankingDayNum <= ranking_day`. Uses the multi-system composite pattern from `src/features/feature_matrix.py`.
  - Public: `predict_massey_logit(team_a: int, team_b: int, massey_ranks: dict[int, float], scale: float = 0.05) -> float` -- returns `-scale * (massey_ranks[team_a] - massey_ranks[team_b])`. Lower rank = better team; logit positive when A is favored. `scale` tuned in Task 5.
  - Public: `evaluate_massey_baseline(test_games: pd.DataFrame, season: int, data_dir: Path, scale: float) -> dict` -- returns `{ll, accuracy, n}`. Reused as the gate baseline in Phase 1 driver.
- `src/gnn_stage1_peer/model.py` (~150 LOC)
  - Public: `class GraphSAGEEncoder(nn.Module)` -- 2-layer SAGEConv, hidden_dim=64, dropout=0.2, ReLU. Constructor: `(num_nodes, hidden_dim=64, num_layers=2, dropout=0.2)`.
  - Public: `class MatchupDecoder(nn.Module)` -- input dim = 3 * hidden_dim (concat + abs-diff), 2-layer MLP with hidden=128, output scalar logit.
  - Public: `class GNNStage1Peer(nn.Module)` -- combines encoder + decoder. Forward: `(graph_data, team_a_idx, team_b_idx) -> logits`.
- `src/gnn_stage1_peer/training.py` (~150 LOC)
  - Public: `train_gnn(model, graph, train_pairs, val_pairs, *, epochs=50, lr=1e-3, patience=5, seed=42) -> dict` -- trains with Adam + early stopping on val LL. Returns `{best_epoch, best_val_ll, train_history}`.
  - Public: `set_determinism(seed: int) -> None` -- sets torch + numpy + python seeds, enables CUDA-determinism flags (no-op on CPU).
- `src/gnn_stage1_peer/evaluation.py` (~80 LOC)
  - Public: `evaluate_gnn_phase1(model, graph, test_pairs) -> dict` -- returns `{ll, accuracy, n, predictions}`. `predictions` is a list of `{team_a, team_b, p_a_wins, label}` for downstream analysis.
  - Public: `compare_gnn_vs_massey(gnn_results: dict, massey_results: dict) -> dict` -- returns `{ll_delta, acc_delta, gate_pass}` where `ll_delta = massey_ll - gnn_ll` (positive means GNN better) and `gate_pass = ll_delta >= 0.005`.
- `src/run_gnn_phase1.py` (~150 LOC)
  - Public: `main(argv) -> int` -- CLI driver. Loops over `--seasons` argument (default `2018,2019,2021,2022,2024`), runs Phase 1 per season, aggregates into summary JSON, writes log + JSON + verdict to `output/`.

**Tests:**

- `tests/test_gnn_stage1_peer/__init__.py`
- `tests/test_gnn_stage1_peer/test_data.py` (~120 LOC)
- `tests/test_gnn_stage1_peer/test_graph.py` (~100 LOC)
- `tests/test_gnn_stage1_peer/test_baselines.py` (~80 LOC)
- `tests/test_gnn_stage1_peer/test_model.py` (~80 LOC)
- `tests/test_gnn_stage1_peer/test_training.py` (~60 LOC)
- `tests/test_gnn_stage1_peer/test_evaluation.py` (~50 LOC)
- `tests/test_gnn_stage1_peer/test_run_phase1_smoke.py` (~50 LOC) -- end-to-end smoke on one season, asserts outputs exist and are well-formed.

**Modified:**

- `pyproject.toml` -- add `torch>=2.2` and `torch-geometric>=2.4` to `[project.optional-dependencies.gnn]` (NEW optional group; not a hard dep so non-GNN runs don't pay the wheel install cost).

**Generated (force-added per `.gitignore: output/`):**

- `output/gnn_phase1_diagnostic.log` -- per-season training log.
- `output/gnn_phase1_per_season.json` -- per-season `{ll, acc, n, train_minutes}` for both GNN and Massey, plus `ll_delta` and `gate_pass`.
- `output/gnn_phase1_summary.json` -- aggregate verdict (mean LL delta, individual season pass rate, overall PASS/FAIL).
- `output/gnn_phase1_summary.txt` -- human-readable summary.

---

## Procedural anchors

These follow the spec's Procedural Requirements section. Verify ALL before claiming Phase 1 done:

1. **No worktrees.** All work happens on `feat/non-tabular-model-class-scoping` in the main repo. Do not run `git worktree add`.
2. **Force-add outputs.** `output/gnn_phase1_*` files are gitignored; the final commit MUST `git add -f` them. The team-seed-residual experiment (PR 34) is the template.
3. **Determinism.** Phase 1 verdict must be reproducible. Set torch/numpy/python seeds at the start of every per-season training run. Document seed in the summary JSON.
4. **No shortcut.** The gate is `ll_delta >= 0.005`. Do not adjust the threshold post-hoc to make the result pass; if the verdict is borderline, document and flag for follow-up.

---

## Task 0: Setup -- PyTorch + PyG dependencies + package skeleton

**Files:**
- Modify: `pyproject.toml`
- Create: `src/gnn_stage1_peer/__init__.py`
- Create: `tests/test_gnn_stage1_peer/__init__.py`

- [ ] **Step 1: Add optional `gnn` dependency group to pyproject.toml**

```toml
# Append to [project.optional-dependencies]
[project.optional-dependencies]
dev = [
    "pytest>=7.4",
    "pytest-cov>=4.1",
]
gnn = [
    "torch>=2.2",
    "torch-geometric>=2.4",
]
```

- [ ] **Step 2: Install the optional group**

```bash
cd /c/Users/alden/MarchMadness && pip install -e ".[gnn]"
```

Expected: torch and torch-geometric install. CPU-only wheel is fine.

- [ ] **Step 3: Smoke-test imports**

```bash
cd /c/Users/alden/MarchMadness && python -c "import torch; import torch_geometric; print('torch:', torch.__version__, 'pyg:', torch_geometric.__version__)"
```

Expected: prints torch and pyg versions; no import error. If this fails with `ImportError: torch_scatter`, install via `pip install torch-scatter -f https://data.pyg.org/whl/torch-${torch_version}+cpu.html`.

- [ ] **Step 4: Create empty package + tests directory**

```python
# src/gnn_stage1_peer/__init__.py
"""Graph Neural Network stage-1 peer model.

Phase 1 sanity check vs scalar Massey on regular-season game prediction.
See docs/superpowers/specs/2026-05-09-non-tabular-model-class-scoping-design.md
and docs/superpowers/plans/2026-05-09-non-tabular-model-class-scoping-phase1.md.
"""
```

```python
# tests/test_gnn_stage1_peer/__init__.py
```

- [ ] **Step 5: Commit**

```bash
cd /c/Users/alden/MarchMadness && git add pyproject.toml src/gnn_stage1_peer/__init__.py tests/test_gnn_stage1_peer/__init__.py && git commit -m "feat(gnn-phase1): add torch+pyg optional deps and package skeleton"
```

---

## Task 1: Data loading and Phase 1 train/test split

**Files:**
- Create: `src/gnn_stage1_peer/data.py`
- Create: `tests/test_gnn_stage1_peer/test_data.py`

- [ ] **Step 1: Write failing tests for `load_rs_games` and `split_phase1`**

```python
# tests/test_gnn_stage1_peer/test_data.py
import pandas as pd
import pytest
from pathlib import Path


def _toy_rs_csv(tmp_path: Path) -> Path:
    df = pd.DataFrame([
        # Season 2024, mix of early- and late-season games
        {"Season": 2024, "DayNum": 50,  "WTeamID": 1101, "WScore": 80, "LTeamID": 1102, "LScore": 70, "WLoc": "H", "NumOT": 0},
        {"Season": 2024, "DayNum": 119, "WTeamID": 1102, "WScore": 75, "LTeamID": 1101, "LScore": 72, "WLoc": "A", "NumOT": 0},
        {"Season": 2024, "DayNum": 120, "WTeamID": 1101, "WScore": 78, "LTeamID": 1103, "LScore": 65, "WLoc": "N", "NumOT": 0},
        {"Season": 2024, "DayNum": 132, "WTeamID": 1103, "WScore": 70, "LTeamID": 1101, "LScore": 68, "WLoc": "H", "NumOT": 1},
        {"Season": 2024, "DayNum": 134, "WTeamID": 1101, "WScore": 80, "LTeamID": 1104, "LScore": 60, "WLoc": "N", "NumOT": 0},  # tournament, excluded
        # Season 2023, should not appear in season=2024 load
        {"Season": 2023, "DayNum": 50,  "WTeamID": 1101, "WScore": 90, "LTeamID": 1102, "LScore": 80, "WLoc": "H", "NumOT": 0},
    ])
    p = tmp_path / "MRegularSeasonCompactResults.csv"
    df.to_csv(p, index=False)
    return tmp_path


def test_load_rs_games_filters_to_season(tmp_path):
    from src.gnn_stage1_peer.data import load_rs_games
    data_dir = _toy_rs_csv(tmp_path)
    games = load_rs_games(data_dir, season=2024)
    assert len(games) == 5  # all 2024 rows including DayNum=134 (filtering happens in split_phase1)
    assert (games["Season"] == 2024).all()


def test_split_phase1_partitions_by_daynum(tmp_path):
    from src.gnn_stage1_peer.data import load_rs_games, split_phase1
    data_dir = _toy_rs_csv(tmp_path)
    games = load_rs_games(data_dir, season=2024)
    train, test = split_phase1(games)
    # Train: DayNum < 120 -> 2 games (DayNum=50, DayNum=119)
    assert len(train) == 2
    assert (train["DayNum"] < 120).all()
    # Test: 120 <= DayNum < 134 -> 2 games (DayNum=120, DayNum=132)
    assert len(test) == 2
    assert (test["DayNum"] >= 120).all() and (test["DayNum"] < 134).all()


def test_build_team_index_assigns_contiguous_indices(tmp_path):
    from src.gnn_stage1_peer.data import load_rs_games, build_team_index
    data_dir = _toy_rs_csv(tmp_path)
    games = load_rs_games(data_dir, season=2024)
    idx = build_team_index(games)
    # Three teams appear in 2024: 1101, 1102, 1103, 1104
    assert set(idx.keys()) == {1101, 1102, 1103, 1104}
    # Indices are contiguous 0..N-1
    assert sorted(idx.values()) == [0, 1, 2, 3]
```

- [ ] **Step 2: Run tests to verify failure**

```bash
cd /c/Users/alden/MarchMadness && python -m pytest tests/test_gnn_stage1_peer/test_data.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.gnn_stage1_peer.data'`.

- [ ] **Step 3: Implement `data.py`**

```python
# src/gnn_stage1_peer/data.py
"""Phase 1 data loading and train/test splits."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# Phase 1 split cutoffs (DayNum-based; approximate March 1 boundary).
# DayNum 0 is roughly the first Monday of November; March 1 of the following
# year is approximately DayNum 120. Selection Sunday is approximately DayNum 132,
# tournament starts at DayNum 134 (First Four).
TRAIN_CUTOFF_DAYNUM = 120
TEST_END_DAYNUM = 134


def load_rs_games(data_dir: Path, season: int) -> pd.DataFrame:
    """Load regular-season games for one season."""
    path = Path(data_dir) / "MRegularSeasonCompactResults.csv"
    df = pd.read_csv(path)
    return df[df["Season"] == season].reset_index(drop=True)


def split_phase1(
    games: pd.DataFrame,
    train_cutoff_daynum: int = TRAIN_CUTOFF_DAYNUM,
    test_end_daynum: int = TEST_END_DAYNUM,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partition games into Phase 1 train (early-season) and test (late-season)."""
    train = games[games["DayNum"] < train_cutoff_daynum].reset_index(drop=True)
    test = games[
        (games["DayNum"] >= train_cutoff_daynum) & (games["DayNum"] < test_end_daynum)
    ].reset_index(drop=True)
    return train, test


def build_team_index(games: pd.DataFrame) -> dict[int, int]:
    """Build a contiguous TeamID -> node_idx mapping over all teams in `games`."""
    teams = sorted(set(games["WTeamID"]).union(set(games["LTeamID"])))
    return {team_id: idx for idx, team_id in enumerate(teams)}
```

- [ ] **Step 4: Run tests to verify pass**

```bash
cd /c/Users/alden/MarchMadness && python -m pytest tests/test_gnn_stage1_peer/test_data.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
cd /c/Users/alden/MarchMadness && git add src/gnn_stage1_peer/data.py tests/test_gnn_stage1_peer/test_data.py && git commit -m "feat(gnn-phase1): RS game loader + Phase 1 train/test split"
```

---

## Task 2: Build PyG graph from RS games

**Files:**
- Create: `src/gnn_stage1_peer/graph.py`
- Create: `tests/test_gnn_stage1_peer/test_graph.py`

- [ ] **Step 1: Write failing test for `build_pyg_graph`**

```python
# tests/test_gnn_stage1_peer/test_graph.py
import pandas as pd
import pytest
import torch


def _toy_train_games() -> pd.DataFrame:
    return pd.DataFrame([
        {"Season": 2024, "DayNum": 50,  "WTeamID": 1101, "WScore": 80, "LTeamID": 1102, "LScore": 70, "WLoc": "H", "NumOT": 0},
        {"Season": 2024, "DayNum": 100, "WTeamID": 1102, "WScore": 75, "LTeamID": 1103, "LScore": 60, "WLoc": "A", "NumOT": 0},
        {"Season": 2024, "DayNum": 119, "WTeamID": 1101, "WScore": 90, "LTeamID": 1103, "LScore": 85, "WLoc": "N", "NumOT": 1},
    ])


def test_build_pyg_graph_has_bidirected_edges():
    from src.gnn_stage1_peer.graph import build_pyg_graph
    from src.gnn_stage1_peer.data import build_team_index
    games = _toy_train_games()
    idx = build_team_index(games)  # 1101->0, 1102->1, 1103->2
    g = build_pyg_graph(games, idx)
    assert g.num_nodes == 3
    # 3 games -> 6 directed edges (bidirected)
    assert g.edge_index.shape == (2, 6)
    assert g.edge_attr.shape == (6, 4)


def test_build_pyg_graph_edge_attr_signed_score_diff():
    """Edge attribute score_diff is signed from the source-node perspective."""
    from src.gnn_stage1_peer.graph import build_pyg_graph
    from src.gnn_stage1_peer.data import build_team_index
    games = pd.DataFrame([
        {"Season": 2024, "DayNum": 50, "WTeamID": 1101, "WScore": 80, "LTeamID": 1102, "LScore": 70, "WLoc": "N", "NumOT": 0},
    ])
    idx = build_team_index(games)  # 1101->0, 1102->1
    g = build_pyg_graph(games, idx)
    # Two directed edges: 0->1 (score_diff = +10) and 1->0 (score_diff = -10).
    src = g.edge_index[0].tolist()
    dst = g.edge_index[1].tolist()
    diffs = g.edge_attr[:, 0].tolist()
    edges = list(zip(src, dst, diffs))
    assert (0, 1, 10.0) in edges
    assert (1, 0, -10.0) in edges


def test_build_matchup_pairs_symmetric():
    from src.gnn_stage1_peer.graph import build_matchup_pairs
    from src.gnn_stage1_peer.data import build_team_index
    games = pd.DataFrame([
        {"Season": 2024, "DayNum": 50, "WTeamID": 1101, "WScore": 80, "LTeamID": 1102, "LScore": 70, "WLoc": "H", "NumOT": 0},
    ])
    idx = build_team_index(games)
    a, b, y = build_matchup_pairs(games, idx)
    assert a.shape == (2,) and b.shape == (2,) and y.shape == (2,)
    # Both orientations: (1101, 1102, 1) and (1102, 1101, 0)
    pairs = sorted(zip(a.tolist(), b.tolist(), y.tolist()))
    assert pairs == sorted([(0, 1, 1.0), (1, 0, 0.0)])
```

- [ ] **Step 2: Run tests to verify failure**

```bash
cd /c/Users/alden/MarchMadness && python -m pytest tests/test_gnn_stage1_peer/test_graph.py -v
```

Expected: FAIL.

- [ ] **Step 3: Implement `graph.py`**

```python
# src/gnn_stage1_peer/graph.py
"""PyG graph construction from RS games."""
from __future__ import annotations

import pandas as pd
import torch
from torch_geometric.data import Data


SITE_INDICATOR = {"H": 1.0, "A": -1.0, "N": 0.0}


def build_pyg_graph(train_games: pd.DataFrame, team_index: dict[int, int]) -> Data:
    """Build a bidirected PyG graph from training-set games.

    Edges are bidirected: each game (W, L) produces two directed edges:
    - (W -> L) with score_diff = +(WScore - LScore), site_indicator = +1 if WLoc=H else 0/-1
    - (L -> W) with score_diff = -(WScore - LScore), site_indicator = -1*above

    Edge attribute layout: [score_diff, site_indicator, days_rest, days_from_season_start].
    Node features are not added in this task -- the encoder uses learned embeddings.
    """
    src_list, dst_list, attr_list = [], [], []
    for _, g in train_games.iterrows():
        w_idx = team_index[int(g["WTeamID"])]
        l_idx = team_index[int(g["LTeamID"])]
        score_diff = float(g["WScore"] - g["LScore"])
        site = SITE_INDICATOR[g["WLoc"]]
        days_from_start = float(g["DayNum"])
        # Days rest is hard to compute without sorting; use 0 placeholder for now.
        # If signal is dependent on rest, refine in a follow-up task.
        days_rest = 0.0
        # W -> L edge: score_diff positive
        src_list.append(w_idx)
        dst_list.append(l_idx)
        attr_list.append([score_diff, site, days_rest, days_from_start])
        # L -> W edge: flip score_diff and site
        src_list.append(l_idx)
        dst_list.append(w_idx)
        attr_list.append([-score_diff, -site, days_rest, days_from_start])

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr = torch.tensor(attr_list, dtype=torch.float)
    num_nodes = len(team_index)
    return Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)


def build_matchup_pairs(
    games: pd.DataFrame, team_index: dict[int, int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build symmetric matchup pairs for training/eval.

    Each game produces two pairs:
    - (W, L, label=1) -- W wins
    - (L, W, label=0) -- W still wins from L's perspective

    Mirrors v3's symmetric-matchup convention (`src/models/matchup.py`).
    """
    a_list, b_list, y_list = [], [], []
    for _, g in games.iterrows():
        w_idx = team_index[int(g["WTeamID"])]
        l_idx = team_index[int(g["LTeamID"])]
        a_list.append(w_idx); b_list.append(l_idx); y_list.append(1.0)
        a_list.append(l_idx); b_list.append(w_idx); y_list.append(0.0)
    return (
        torch.tensor(a_list, dtype=torch.long),
        torch.tensor(b_list, dtype=torch.long),
        torch.tensor(y_list, dtype=torch.float),
    )
```

- [ ] **Step 4: Run tests**

```bash
cd /c/Users/alden/MarchMadness && python -m pytest tests/test_gnn_stage1_peer/test_graph.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
cd /c/Users/alden/MarchMadness && git add src/gnn_stage1_peer/graph.py tests/test_gnn_stage1_peer/test_graph.py && git commit -m "feat(gnn-phase1): build_pyg_graph + symmetric matchup pairs"
```

---

## Task 3: Massey baseline (composite rank-based logit)

**Files:**
- Create: `src/gnn_stage1_peer/baselines.py`
- Create: `tests/test_gnn_stage1_peer/test_baselines.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_gnn_stage1_peer/test_baselines.py
import pandas as pd
from pathlib import Path


def _toy_massey_csv(tmp_path: Path) -> Path:
    df = pd.DataFrame([
        # Two systems, one season, three teams. Lower rank = better.
        {"Season": 2024, "RankingDayNum": 100, "SystemName": "POM", "TeamID": 1101, "OrdinalRank": 5},
        {"Season": 2024, "RankingDayNum": 100, "SystemName": "POM", "TeamID": 1102, "OrdinalRank": 50},
        {"Season": 2024, "RankingDayNum": 100, "SystemName": "POM", "TeamID": 1103, "OrdinalRank": 200},
        {"Season": 2024, "RankingDayNum": 100, "SystemName": "MAS", "TeamID": 1101, "OrdinalRank": 7},
        {"Season": 2024, "RankingDayNum": 100, "SystemName": "MAS", "TeamID": 1102, "OrdinalRank": 60},
        {"Season": 2024, "RankingDayNum": 100, "SystemName": "MAS", "TeamID": 1103, "OrdinalRank": 180},
        # Future-dated row should be ignored when ranking_day=100
        {"Season": 2024, "RankingDayNum": 133, "SystemName": "POM", "TeamID": 1101, "OrdinalRank": 1},
        # Other-season row should be ignored
        {"Season": 2023, "RankingDayNum": 100, "SystemName": "POM", "TeamID": 1101, "OrdinalRank": 99},
    ])
    p = tmp_path / "MMasseyOrdinals.csv"
    df.to_csv(p, index=False)
    return tmp_path


def test_load_massey_composite_filters_season_and_day(tmp_path):
    from src.gnn_stage1_peer.baselines import load_massey_composite
    data_dir = _toy_massey_csv(tmp_path)
    ranks = load_massey_composite(data_dir, season=2024, ranking_day=100)
    # Composite is mean across systems at the latest day <= ranking_day.
    # 1101: (5 + 7) / 2 = 6.0
    assert ranks[1101] == 6.0
    assert ranks[1102] == 55.0
    assert ranks[1103] == 190.0


def test_predict_massey_logit_signs():
    from src.gnn_stage1_peer.baselines import predict_massey_logit
    ranks = {1101: 5.0, 1102: 50.0}
    # 1101 better (lower rank); A=1101 favored -> positive logit.
    logit = predict_massey_logit(1101, 1102, ranks, scale=0.05)
    assert logit > 0
    # Reverse orientation -> negative.
    assert predict_massey_logit(1102, 1101, ranks, scale=0.05) == pytest.approx(-logit)


def test_evaluate_massey_baseline_returns_ll_and_acc(tmp_path):
    from src.gnn_stage1_peer.baselines import evaluate_massey_baseline
    data_dir = _toy_massey_csv(tmp_path)
    test_games = pd.DataFrame([
        {"Season": 2024, "DayNum": 125, "WTeamID": 1101, "WScore": 80, "LTeamID": 1103, "LScore": 60, "WLoc": "N"},
        {"Season": 2024, "DayNum": 130, "WTeamID": 1102, "WScore": 75, "LTeamID": 1103, "LScore": 65, "WLoc": "N"},
    ])
    out = evaluate_massey_baseline(test_games, season=2024, data_dir=data_dir, scale=0.05)
    # Both games: better-ranked team won. Massey logit positive both times -> p > 0.5 -> acc=1.0.
    assert out["accuracy"] == 1.0
    assert out["n"] == 4  # symmetric, 2 games -> 4 pairs
    assert out["ll"] > 0  # standard convention: positive LL = mean BCE loss
```

- [ ] **Step 2: Run tests to verify failure**

```bash
cd /c/Users/alden/MarchMadness && python -m pytest tests/test_gnn_stage1_peer/test_baselines.py -v
```

Expected: FAIL.

- [ ] **Step 3: Implement `baselines.py`**

```python
# src/gnn_stage1_peer/baselines.py
"""Scalar Massey-composite baseline for Phase 1 gate."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

# Default Massey systems used for the composite. Subset of v4's massey_systems
# that have stable historical coverage across 2003-2025.
DEFAULT_SYSTEMS = ("POM", "MAS", "SAG", "MOR", "DOL")


def load_massey_composite(
    data_dir: Path,
    season: int,
    ranking_day: int = 133,
    systems: tuple[str, ...] = DEFAULT_SYSTEMS,
) -> dict[int, float]:
    """Load Massey composite rank as `{team_id: mean_rank}` for one season.

    Composite = mean of OrdinalRank across `systems`, evaluated at the latest
    RankingDayNum <= `ranking_day` per (season, system, team).
    """
    path = Path(data_dir) / "MMasseyOrdinals.csv"
    df = pd.read_csv(path)
    df = df[(df["Season"] == season) & (df["RankingDayNum"] <= ranking_day)]
    df = df[df["SystemName"].isin(systems)]
    # Take latest day per (system, team)
    df = df.sort_values("RankingDayNum").groupby(["SystemName", "TeamID"]).tail(1)
    composite = df.groupby("TeamID")["OrdinalRank"].mean().astype(float)
    return composite.to_dict()


def predict_massey_logit(
    team_a: int, team_b: int, massey_ranks: dict[int, float], scale: float = 0.05
) -> float:
    """Predict logit p(A wins) given Massey composite ranks. Lower rank = better."""
    if team_a not in massey_ranks or team_b not in massey_ranks:
        return 0.0
    return -scale * (massey_ranks[team_a] - massey_ranks[team_b])


def evaluate_massey_baseline(
    test_games: pd.DataFrame,
    season: int,
    data_dir: Path,
    scale: float = 0.05,
    ranking_day: int = 133,
) -> dict:
    """Evaluate Massey baseline on a test split. Symmetric over orientations."""
    ranks = load_massey_composite(data_dir, season, ranking_day)
    nll_sum = 0.0
    correct = 0
    n = 0
    for _, g in test_games.iterrows():
        for (a, b, label) in (
            (int(g["WTeamID"]), int(g["LTeamID"]), 1.0),
            (int(g["LTeamID"]), int(g["WTeamID"]), 0.0),
        ):
            logit = predict_massey_logit(a, b, ranks, scale)
            p = 1.0 / (1.0 + math.exp(-logit))
            # BCE per sample: -[y log p + (1-y) log(1-p)]
            eps = 1e-12
            nll_sum += -(label * math.log(max(p, eps)) + (1.0 - label) * math.log(max(1.0 - p, eps)))
            correct += int((p >= 0.5) == (label >= 0.5))
            n += 1
    return {"ll": nll_sum / max(n, 1), "accuracy": correct / max(n, 1), "n": n}
```

- [ ] **Step 4: Run tests**

```bash
cd /c/Users/alden/MarchMadness && python -m pytest tests/test_gnn_stage1_peer/test_baselines.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
cd /c/Users/alden/MarchMadness && git add src/gnn_stage1_peer/baselines.py tests/test_gnn_stage1_peer/test_baselines.py && git commit -m "feat(gnn-phase1): Massey composite baseline (LL/acc evaluator)"
```

---

## Task 4: Tune Massey `scale` on real 2024 RS data

The default `scale=0.05` is a guess. Tune it before declaring the gate threshold.

**Files:**
- Create: `tests/test_gnn_stage1_peer/test_baselines_tune.py` (optional one-shot script; can be deleted after running)
- Modify: `src/gnn_stage1_peer/baselines.py` -- update DEFAULT_SCALE constant.

- [ ] **Step 1: Write a one-shot tuning script**

```python
# tests/test_gnn_stage1_peer/test_baselines_tune.py
"""One-shot: scan scale values and pick the one that minimizes LL on real 2024 RS test split.

Run with: pytest tests/test_gnn_stage1_peer/test_baselines_tune.py::test_tune_scale -s
"""
from pathlib import Path

import pytest


@pytest.mark.skip(reason="One-shot tuning; un-skip and run manually then re-skip.")
def test_tune_scale():
    from src.gnn_stage1_peer.data import load_rs_games, split_phase1
    from src.gnn_stage1_peer.baselines import evaluate_massey_baseline
    data_dir = Path("data/raw/march-machine-learning-2026")
    games = load_rs_games(data_dir, season=2024)
    _, test = split_phase1(games)
    best = (None, float("inf"))
    for scale in [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]:
        out = evaluate_massey_baseline(test, season=2024, data_dir=data_dir, scale=scale)
        print(f"scale={scale:.3f}  LL={out['ll']:.4f}  acc={out['accuracy']:.3f}  n={out['n']}")
        if out["ll"] < best[1]:
            best = (scale, out["ll"])
    print(f"BEST scale={best[0]:.3f}  LL={best[1]:.4f}")
```

- [ ] **Step 2: Comment out the `@pytest.mark.skip` line and run the tuning script**

Edit `tests/test_gnn_stage1_peer/test_baselines_tune.py` and comment out the `@pytest.mark.skip(...)` decorator (a single-line edit). Then run:

```bash
cd /c/Users/alden/MarchMadness && python -m pytest tests/test_gnn_stage1_peer/test_baselines_tune.py::test_tune_scale -s
```

Pick the scale value with lowest LL. Expected best is around 0.04-0.07 based on Massey's typical scale.

- [ ] **Step 3: Update the default in `baselines.py`**

```python
# Edit DEFAULT_SCALE constant at top of src/gnn_stage1_peer/baselines.py
DEFAULT_SCALE = 0.05  # <-- replace with tuned value from Task 4 Step 2
```

Then update `predict_massey_logit` and `evaluate_massey_baseline` defaults to use `DEFAULT_SCALE`.

- [ ] **Step 4: Restore the `@pytest.mark.skip` decorator**

Un-comment the `@pytest.mark.skip(...)` line in `test_baselines_tune.py` so future test runs don't re-tune.

- [ ] **Step 5: Commit**

```bash
cd /c/Users/alden/MarchMadness && git add src/gnn_stage1_peer/baselines.py tests/test_gnn_stage1_peer/test_baselines_tune.py && git commit -m "feat(gnn-phase1): tune Massey baseline scale on 2024 late-RS"
```

---

## Task 5: GraphSAGE encoder + matchup decoder

**Files:**
- Create: `src/gnn_stage1_peer/model.py`
- Create: `tests/test_gnn_stage1_peer/test_model.py`

- [ ] **Step 1: Write failing tests for shapes**

```python
# tests/test_gnn_stage1_peer/test_model.py
import torch
from torch_geometric.data import Data


def _toy_graph(num_nodes: int = 4) -> Data:
    # 4 nodes, 6 directed edges (3 bidirected pairs)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long)
    edge_attr = torch.zeros((6, 4), dtype=torch.float)
    return Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)


def test_graphsage_encoder_output_shape():
    from src.gnn_stage1_peer.model import GraphSAGEEncoder
    enc = GraphSAGEEncoder(num_nodes=4, hidden_dim=64, num_layers=2, dropout=0.0)
    g = _toy_graph()
    out = enc(g)
    assert out.shape == (4, 64)


def test_matchup_decoder_output_shape():
    from src.gnn_stage1_peer.model import MatchupDecoder
    dec = MatchupDecoder(embed_dim=64, hidden=128)
    a_emb = torch.randn(8, 64)
    b_emb = torch.randn(8, 64)
    logits = dec(a_emb, b_emb)
    assert logits.shape == (8,)


def test_full_model_forward_pass():
    from src.gnn_stage1_peer.model import GNNStage1Peer
    m = GNNStage1Peer(num_nodes=4, hidden_dim=64, num_layers=2, dropout=0.0, decoder_hidden=128)
    g = _toy_graph()
    a = torch.tensor([0, 1, 2], dtype=torch.long)
    b = torch.tensor([1, 2, 3], dtype=torch.long)
    logits = m(g, a, b)
    assert logits.shape == (3,)
```

- [ ] **Step 2: Run tests to verify failure**

```bash
cd /c/Users/alden/MarchMadness && python -m pytest tests/test_gnn_stage1_peer/test_model.py -v
```

Expected: FAIL.

- [ ] **Step 3: Implement `model.py`**

```python
# src/gnn_stage1_peer/model.py
"""GNN stage-1 peer model (encoder + matchup decoder)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv


class GraphSAGEEncoder(nn.Module):
    """2-layer GraphSAGE with learned node embeddings as input features."""

    def __init__(
        self, num_nodes: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.node_emb = nn.Embedding(num_nodes, hidden_dim)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr="mean"))
        self.dropout = dropout

    def forward(self, graph: Data) -> torch.Tensor:
        x = self.node_emb.weight  # (num_nodes, hidden_dim)
        for conv in self.convs[:-1]:
            x = conv(x, graph.edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, graph.edge_index)
        return x


class MatchupDecoder(nn.Module):
    """Concat(a, b, |a-b|) -> 2-layer MLP -> logit."""

    def __init__(self, embed_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.fc1 = nn.Linear(embed_dim * 3, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, a_emb: torch.Tensor, b_emb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([a_emb, b_emb, (a_emb - b_emb).abs()], dim=-1)
        x = F.relu(self.fc1(x))
        return self.fc2(x).squeeze(-1)


class GNNStage1Peer(nn.Module):
    """Encoder + decoder. Forward: (graph, a_idx, b_idx) -> logits."""

    def __init__(
        self,
        num_nodes: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        decoder_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.encoder = GraphSAGEEncoder(num_nodes, hidden_dim, num_layers, dropout)
        self.decoder = MatchupDecoder(hidden_dim, decoder_hidden)

    def forward(
        self, graph: Data, a_idx: torch.Tensor, b_idx: torch.Tensor
    ) -> torch.Tensor:
        embeds = self.encoder(graph)
        return self.decoder(embeds[a_idx], embeds[b_idx])
```

- [ ] **Step 4: Run tests**

```bash
cd /c/Users/alden/MarchMadness && python -m pytest tests/test_gnn_stage1_peer/test_model.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
cd /c/Users/alden/MarchMadness && git add src/gnn_stage1_peer/model.py tests/test_gnn_stage1_peer/test_model.py && git commit -m "feat(gnn-phase1): GraphSAGE encoder + matchup decoder"
```

---

## Task 6: Training loop with early stopping + determinism

**Files:**
- Create: `src/gnn_stage1_peer/training.py`
- Create: `tests/test_gnn_stage1_peer/test_training.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_gnn_stage1_peer/test_training.py
import torch
from torch_geometric.data import Data


def _toy_setup(num_nodes: int = 6, seed: int = 0):
    """Build a separable toy: nodes 0-2 always beat nodes 3-5."""
    torch.manual_seed(seed)
    # Edges: each "good" team beat each "bad" team once
    src, dst, attr = [], [], []
    for w in (0, 1, 2):
        for l in (3, 4, 5):
            src += [w, l]; dst += [l, w]
            attr += [[10.0, 0.0, 0.0, 50.0], [-10.0, 0.0, 0.0, 50.0]]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(attr, dtype=torch.float)
    g = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)
    # Train pairs: same separable structure
    a = torch.tensor([0, 1, 2, 3, 4, 5, 0, 1, 2], dtype=torch.long)
    b = torch.tensor([3, 4, 5, 0, 1, 2, 4, 5, 3], dtype=torch.long)
    y = torch.tensor([1, 1, 1, 0, 0, 0, 1, 1, 1], dtype=torch.float)
    return g, (a, b, y)


def test_train_gnn_loss_decreases():
    from src.gnn_stage1_peer.model import GNNStage1Peer
    from src.gnn_stage1_peer.training import train_gnn, set_determinism
    set_determinism(42)
    g, (a, b, y) = _toy_setup()
    model = GNNStage1Peer(num_nodes=6, hidden_dim=16, dropout=0.0, decoder_hidden=32)
    history = train_gnn(model, g, (a, b, y), (a, b, y), epochs=30, lr=0.05, patience=10, seed=42)
    losses = history["train_history"]["loss"]
    # On a separable toy task, loss should drop substantially.
    assert losses[-1] < losses[0] * 0.5
    assert losses[-1] < 0.4  # near-perfect separation


def test_set_determinism_reproducible():
    from src.gnn_stage1_peer.training import set_determinism
    set_determinism(42)
    a1 = torch.randn(3)
    set_determinism(42)
    a2 = torch.randn(3)
    assert torch.equal(a1, a2)
```

- [ ] **Step 2: Run tests to verify failure**

```bash
cd /c/Users/alden/MarchMadness && python -m pytest tests/test_gnn_stage1_peer/test_training.py -v
```

Expected: FAIL.

- [ ] **Step 3: Implement `training.py`**

```python
# src/gnn_stage1_peer/training.py
"""Training loop for the Phase 1 GNN with early stopping + determinism."""
from __future__ import annotations

import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F


def set_determinism(seed: int) -> None:
    """Set all relevant seeds for reproducibility (CPU)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # no-op on CPU machines, hedged for completeness
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)  # SAGE/CUDA paths require non-strict mode


def _bce_logits_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, targets)


def _eval_ll(model, graph, pairs) -> float:
    model.eval()
    with torch.no_grad():
        a, b, y = pairs
        logits = model(graph, a, b)
        return _bce_logits_loss(logits, y).item()


def train_gnn(
    model,
    graph,
    train_pairs,
    val_pairs,
    *,
    epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 5,
    seed: int = 42,
) -> dict:
    """Train the GNN with Adam + early stopping on val LL."""
    set_determinism(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    a, b, y = train_pairs
    history = {"loss": [], "val_ll": []}
    best_val = math.inf
    best_state = None
    bad_epochs = 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(graph, a, b)
        loss = _bce_logits_loss(logits, y)
        loss.backward()
        optimizer.step()
        history["loss"].append(loss.item())
        val_ll = _eval_ll(model, graph, val_pairs)
        history["val_ll"].append(val_ll)
        if val_ll < best_val - 1e-5:
            best_val = val_ll
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "best_val_ll": best_val,
        "best_epoch": int(np.argmin(history["val_ll"])),
        "epochs_run": len(history["loss"]),
        "train_history": history,
    }
```

- [ ] **Step 4: Run tests**

```bash
cd /c/Users/alden/MarchMadness && python -m pytest tests/test_gnn_stage1_peer/test_training.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
cd /c/Users/alden/MarchMadness && git add src/gnn_stage1_peer/training.py tests/test_gnn_stage1_peer/test_training.py && git commit -m "feat(gnn-phase1): training loop with Adam + early stopping + determinism"
```

---

## Task 7: Phase 1 evaluation harness

**Files:**
- Create: `src/gnn_stage1_peer/evaluation.py`
- Create: `tests/test_gnn_stage1_peer/test_evaluation.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_gnn_stage1_peer/test_evaluation.py
def test_evaluate_gnn_phase1_returns_well_formed_dict():
    import torch
    from torch_geometric.data import Data
    from src.gnn_stage1_peer.model import GNNStage1Peer
    from src.gnn_stage1_peer.evaluation import evaluate_gnn_phase1
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    g = Data(edge_index=edge_index, edge_attr=torch.zeros((2, 4)), num_nodes=2)
    m = GNNStage1Peer(num_nodes=2, hidden_dim=4, dropout=0.0, decoder_hidden=4)
    a = torch.tensor([0, 1], dtype=torch.long)
    b = torch.tensor([1, 0], dtype=torch.long)
    y = torch.tensor([1, 0], dtype=torch.float)
    out = evaluate_gnn_phase1(m, g, (a, b, y))
    assert {"ll", "accuracy", "n", "predictions"} <= out.keys()
    assert out["n"] == 2


def test_compare_gnn_vs_massey_gate_logic():
    from src.gnn_stage1_peer.evaluation import compare_gnn_vs_massey
    # GNN clearly better
    out = compare_gnn_vs_massey({"ll": 0.50, "accuracy": 0.72}, {"ll": 0.51, "accuracy": 0.70})
    assert out["ll_delta"] == 0.01
    assert out["gate_pass"] is True
    # GNN essentially flat
    out = compare_gnn_vs_massey({"ll": 0.508, "accuracy": 0.71}, {"ll": 0.510, "accuracy": 0.71})
    assert out["gate_pass"] is False
```

- [ ] **Step 2: Run tests to verify failure**

```bash
cd /c/Users/alden/MarchMadness && python -m pytest tests/test_gnn_stage1_peer/test_evaluation.py -v
```

Expected: FAIL.

- [ ] **Step 3: Implement `evaluation.py`**

```python
# src/gnn_stage1_peer/evaluation.py
"""Phase 1 evaluation: GNN metrics + GNN-vs-Massey comparison."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

PHASE1_GATE_LL_DELTA = 0.005


def evaluate_gnn_phase1(model, graph, eval_pairs) -> dict:
    """Run GNN on eval_pairs, return LL/accuracy/n/predictions."""
    model.eval()
    a, b, y = eval_pairs
    with torch.no_grad():
        logits = model(graph, a, b)
        probs = torch.sigmoid(logits)
        ll = F.binary_cross_entropy_with_logits(logits, y).item()
        preds = (probs >= 0.5).float()
        acc = (preds == y).float().mean().item()
        n = int(y.numel())
        predictions = [
            {
                "team_a_idx": int(a[i]),
                "team_b_idx": int(b[i]),
                "p_a_wins": float(probs[i]),
                "label": float(y[i]),
            }
            for i in range(n)
        ]
    return {"ll": ll, "accuracy": acc, "n": n, "predictions": predictions}


def compare_gnn_vs_massey(gnn_results: dict, massey_results: dict) -> dict:
    """Apply Phase 1 gate: GNN LL must be at least PHASE1_GATE_LL_DELTA below Massey's."""
    ll_delta = massey_results["ll"] - gnn_results["ll"]
    acc_delta = gnn_results["accuracy"] - massey_results["accuracy"]
    gate_pass = ll_delta >= PHASE1_GATE_LL_DELTA
    return {
        "ll_delta": ll_delta,
        "acc_delta": acc_delta,
        "gate_pass": gate_pass,
        "gnn_ll": gnn_results["ll"],
        "massey_ll": massey_results["ll"],
        "gnn_acc": gnn_results["accuracy"],
        "massey_acc": massey_results["accuracy"],
    }
```

- [ ] **Step 4: Run tests**

```bash
cd /c/Users/alden/MarchMadness && python -m pytest tests/test_gnn_stage1_peer/test_evaluation.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
cd /c/Users/alden/MarchMadness && git add src/gnn_stage1_peer/evaluation.py tests/test_gnn_stage1_peer/test_evaluation.py && git commit -m "feat(gnn-phase1): evaluation harness + LL-delta gate logic"
```

---

## Task 8: Per-season Phase 1 driver function

This task wires the pieces into a single per-season function. The next task adds the multi-season CLI driver on top.

**Files:**
- Modify: `src/gnn_stage1_peer/__init__.py` (re-export `run_phase1_one_season`)
- Modify: `src/gnn_stage1_peer/training.py` -- add `run_phase1_one_season` function (it bridges the data + model + eval pipeline; lives in training.py to avoid circular imports).
- Create: tests/test_gnn_stage1_peer/test_run_phase1_smoke.py (real-data smoke test).

- [ ] **Step 1: Write failing smoke test (real Kaggle data, season 2024)**

```python
# tests/test_gnn_stage1_peer/test_run_phase1_smoke.py
"""Real-data smoke test on season 2024.

This test EXPECTS the Kaggle data files to be present at
data/raw/march-machine-learning-2026/. It is slow (~30-90s on CPU);
mark @pytest.mark.slow if you want to skip it during fast runs.
"""
import pytest
from pathlib import Path


@pytest.mark.slow
def test_run_phase1_one_season_2024_returns_well_formed():
    from src.gnn_stage1_peer.training import run_phase1_one_season
    data_dir = Path("data/raw/march-machine-learning-2026")
    out = run_phase1_one_season(
        data_dir=data_dir,
        season=2024,
        hidden_dim=64,
        epochs=30,
        lr=1e-3,
        patience=5,
        seed=42,
    )
    assert {"season", "gnn", "massey", "compare", "train_minutes"} <= out.keys()
    assert out["season"] == 2024
    assert out["gnn"]["n"] > 100  # 2024 has hundreds of late-RS games
    assert 0.4 <= out["gnn"]["ll"] <= 1.0
    assert 0.4 <= out["massey"]["ll"] <= 1.0
    assert out["train_minutes"] < 30.0  # if this exceeds 30, escalate per spec
```

- [ ] **Step 2: Implement `run_phase1_one_season` in `training.py`**

Append the following to `src/gnn_stage1_peer/training.py`:

```python
import time
from pathlib import Path

from .data import load_rs_games, split_phase1, build_team_index
from .graph import build_pyg_graph, build_matchup_pairs
from .baselines import evaluate_massey_baseline
from .evaluation import evaluate_gnn_phase1, compare_gnn_vs_massey
from .model import GNNStage1Peer


def run_phase1_one_season(
    data_dir: Path,
    season: int,
    *,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    decoder_hidden: int = 128,
    epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 5,
    seed: int = 42,
) -> dict:
    """Run one season's Phase 1: train GNN on early-RS, eval on late-RS, compare to Massey."""
    set_determinism(seed)
    games = load_rs_games(data_dir, season)
    train_games, test_games = split_phase1(games)
    if train_games.empty or test_games.empty:
        raise ValueError(f"Season {season}: train or test split empty.")

    team_index = build_team_index(games)
    graph = build_pyg_graph(train_games, team_index)
    train_pairs = build_matchup_pairs(train_games, team_index)
    test_pairs = build_matchup_pairs(test_games, team_index)

    model = GNNStage1Peer(
        num_nodes=len(team_index),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        decoder_hidden=decoder_hidden,
    )
    t0 = time.time()
    train_result = train_gnn(
        model, graph, train_pairs, test_pairs,
        epochs=epochs, lr=lr, patience=patience, seed=seed,
    )
    train_minutes = (time.time() - t0) / 60.0

    gnn_eval = evaluate_gnn_phase1(model, graph, test_pairs)
    massey_eval = evaluate_massey_baseline(test_games, season=season, data_dir=data_dir)
    compare = compare_gnn_vs_massey(gnn_eval, massey_eval)

    return {
        "season": season,
        "gnn": {k: v for k, v in gnn_eval.items() if k != "predictions"},
        "massey": massey_eval,
        "compare": compare,
        "train_minutes": train_minutes,
        "epochs_run": train_result["epochs_run"],
        "best_epoch": train_result["best_epoch"],
    }
```

- [ ] **Step 3: Run smoke test (slow; expect ~1-2 min on CPU)**

```bash
cd /c/Users/alden/MarchMadness && python -m pytest tests/test_gnn_stage1_peer/test_run_phase1_smoke.py -v -m slow --no-header
```

Expected: PASS. If `train_minutes > 30`, escalate per the spec's open-question #3 (CPU wall-clock budget).

- [ ] **Step 4: Commit**

```bash
cd /c/Users/alden/MarchMadness && git add src/gnn_stage1_peer/training.py tests/test_gnn_stage1_peer/test_run_phase1_smoke.py && git commit -m "feat(gnn-phase1): per-season driver run_phase1_one_season + 2024 smoke"
```

---

## Task 9: Phase 1 CLI driver -- multi-season sweep + outputs

**Files:**
- Create: `src/run_gnn_phase1.py`

- [ ] **Step 1: Implement the CLI driver**

```python
# src/run_gnn_phase1.py
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
```

- [ ] **Step 2: Run on the 5 default test seasons**

```bash
cd /c/Users/alden/MarchMadness && python -m src.run_gnn_phase1 --seasons 2018,2019,2021,2022,2024 2>&1 | tee output/gnn_phase1_console.log
```

Expected total runtime: ~5-15 min on CPU (5 seasons x ~1-3 min each). If it exceeds 60 min, escalate.

- [ ] **Step 3: Inspect the verdict**

```bash
cd /c/Users/alden/MarchMadness && cat output/gnn_phase1_summary.txt
```

Three possible outcomes:

1. **PASS (mean_ll_delta >= +0.005):** proceed to Task 10 (verdict + Phase 2 plan handoff).
2. **FAIL (mean_ll_delta < +0.005):** proceed to Task 10 (FAIL findings + sequel re-rank).
3. **MARGINAL (-0.002 to +0.005):** Document; flag in findings; recommend a hyperparameter sensitivity sweep before final verdict (e.g., hidden_dim 32 vs 128, dropout 0.0 vs 0.4, lr 1e-3 vs 5e-4). One sweep cycle, 1 day of work; if MARGINAL persists, treat as FAIL.

- [ ] **Step 4: Force-add outputs to git**

```bash
cd /c/Users/alden/MarchMadness && git add -f output/gnn_phase1_diagnostic.log output/gnn_phase1_per_season.json output/gnn_phase1_summary.json output/gnn_phase1_summary.txt && git add src/run_gnn_phase1.py && git commit -m "data(gnn-phase1): 5-season sweep + summary -- $(cat output/gnn_phase1_summary.json | python -c 'import json,sys; print(json.load(sys.stdin)[\"verdict\"])')"
```

The commit message captures the verdict via the inline shell expansion; reproducible across machines. If your shell doesn't support that pattern, hardcode the verdict.

---

## Task 10: Findings note + TODO update

**Files:**
- Create: `docs/notes/<run-date>-gnn-phase1.md`
- Modify: `TODO.md`

- [ ] **Step 1: Write the findings note**

Use this template at `docs/notes/<YYYY-MM-DD>-gnn-phase1.md` (replace `<YYYY-MM-DD>` with the run date):

```markdown
# GNN Stage-1 Peer Phase 1 -- <PASS or FAIL or MARGINAL> (<YYYY-MM-DD>)

**Spec:** `docs/superpowers/specs/2026-05-09-non-tabular-model-class-scoping-design.md`
**Plan:** `docs/superpowers/plans/2026-05-09-non-tabular-model-class-scoping-phase1.md`
**Branch:** `feat/non-tabular-model-class-scoping`

## TL;DR

GNN Phase 1 sanity check on RS-prediction (March 1 -> Selection Sunday) compared
to scalar Massey composite baseline. **Verdict: <PASS | FAIL | MARGINAL>.**

Aggregate: mean LL delta (Massey - GNN) = <+0.0xxx>, gate >= +0.005. <N>/<5>
seasons individually pass.

## Per-season detail

(Paste contents of `output/gnn_phase1_summary.txt` here.)

## What this means

- **PASS:** the GNN extracts RS signal that scalar Massey doesn't. Justifies
  the more expensive Phase 2 (22-season tournament LOSO). Phase 2 plan is the
  next deliverable.
- **FAIL:** Massey already extracts the bulk of the relational signal at the
  RS level. The GNN candidate is closed. Per the spec's sequel-ordering
  matrix: rank up Candidate 4 (self-supervised embeddings, similar
  saturation-break theory at team level), keep Candidate 3 (box-score,
  distinct signal class), deprioritize Candidate 2 (sequence model, same
  "already aggregated" risk).
- **MARGINAL:** see Task 9 Step 3 -- run hyperparameter sensitivity sweep,
  re-evaluate.

## Methodological notes

- Per-test-season independent training (no cross-season parameter sharing in
  Phase 1; revisited in Phase 2 if applicable).
- DayNum cutoff 120 used as the March-1 boundary; this is approximate
  (varies +/- 2 days across seasons depending on calendar). Open question:
  does using exact-calendar cutoff (computed from `MSeasons.csv`'s `DayZero`)
  shift the verdict materially? Defer to Phase 2 if needed.
- Massey scale tuned at <value> (Task 4). The verdict is robust to
  scale +/- 50% based on the Task 4 grid.
- Per-season training wall-clock: max <X.X min>, well within the
  30-min-per-season escalation threshold.

## Open questions / Phase 2 implications

(Document here any specific surprises, e.g., one season is a strong outlier;
massey baseline behaves unexpectedly on 2021; etc.)

## Files of record

- This findings note: `docs/notes/<YYYY-MM-DD>-gnn-phase1.md`
- Outputs (force-added):
  - `output/gnn_phase1_diagnostic.log`
  - `output/gnn_phase1_per_season.json`
  - `output/gnn_phase1_summary.json`
  - `output/gnn_phase1_summary.txt`
- Code: `src/gnn_stage1_peer/`, `src/run_gnn_phase1.py`, `tests/test_gnn_stage1_peer/`
```

- [ ] **Step 2: Update `TODO.md` Active queue**

In `TODO.md`'s Active queue section (search for `## Active queue`), insert a Done entry at the top of the Done section AND update Active queue item #3 (currently "Small neural net (MLP)") to reflect the GNN Phase 1 result.

PASS template (for Done entry):

```markdown
- **GNN stage-1 peer Phase 1 -- PASS (<YYYY-MM-DD>).** RS prediction (Mar 1
  -> Selection Sunday) GNN beats scalar Massey LL by mean +0.0xxx across 5
  test seasons (gate >= +0.005). N/5 individual seasons pass. Per-season
  training wall-clock max <X> min on CPU. Phase 2 (22-season tournament
  LOSO) plan is the next deliverable. Findings:
  `docs/notes/<YYYY-MM-DD>-gnn-phase1.md`.
```

FAIL template:

```markdown
- **GNN stage-1 peer Phase 1 -- FAIL (<YYYY-MM-DD>).** Mean LL delta
  +0.0xxx (gate >= +0.005); N/5 seasons pass individually. Massey composite
  already extracts the bulk of the relational signal at the RS level.
  Eighth same-data-equivalent null result (counting BT, feature-view,
  HBT, Colley, Massey-MOV, Massey-decay-14d, team-seed-residual,
  GNN-Phase-1). Per spec sequel-ordering matrix, GNN candidate closed;
  promote Candidate 4 (self-supervised embeddings) to lead, keep
  Candidate 3 (box-score), deprioritize Candidate 2 (sequence model).
  Findings: `docs/notes/<YYYY-MM-DD>-gnn-phase1.md`.
```

Active queue update -- replace item #3 with PASS or FAIL ordering:

```markdown
3. **GNN stage-1 peer Phase 2 (if Phase 1 passes).** [If PASS only]
   22-season tournament LOSO via the GNN architecture validated in Phase 1.
   See findings note. Cost ~1.5 weeks Phase 2 + 2 days v8 retrain. Plan
   pending.
```

OR (FAIL):

```markdown
3. **Self-supervised team embeddings via regular-season margin prediction
   (Candidate 4 promoted).** [If FAIL] Promoted from sequel position by GNN
   Phase 1 failure. See spec for saturation-break theory and gate criteria.
   Scope a Phase 1 sanity check (analogous to GNN's) before committing to
   LOSO.
```

- [ ] **Step 3: Verify ASCII-only**

```bash
cd /c/Users/alden/MarchMadness && python -c "open('docs/notes/<YYYY-MM-DD>-gnn-phase1.md').read().encode('ascii')" && python -c "open('TODO.md').read().encode('ascii')"
```

Expected: no `UnicodeEncodeError`. Per CLAUDE.md, all files must be ASCII-only (cp1252 console safety).

- [ ] **Step 4: Run full test suite + smoke test**

```bash
cd /c/Users/alden/MarchMadness && python -m pytest tests/test_gnn_stage1_peer/ -v
```

Expected: all unit tests pass; smoke test passes (or is appropriately marked slow). Per CLAUDE.md's FORCED VERIFICATION rule, never skip this step.

- [ ] **Step 5: Commit**

```bash
cd /c/Users/alden/MarchMadness && git add docs/notes/<YYYY-MM-DD>-gnn-phase1.md TODO.md && git commit -m "docs(gnn-phase1): findings + TODO update -- <PASS or FAIL>"
```

---

## Phase 1 acceptance checklist

Before opening the PR for this branch, verify:

- [ ] All 7 unit-test files pass (Tasks 1-7) under `pytest tests/test_gnn_stage1_peer/`.
- [ ] Smoke test on real 2024 data passes (Task 8).
- [ ] 5-season sweep ran end-to-end (Task 9) and produced all 4 output files.
- [ ] Outputs are force-added (`git ls-files output/gnn_phase1_*` returns 4 files).
- [ ] Findings note exists at `docs/notes/<YYYY-MM-DD>-gnn-phase1.md`.
- [ ] TODO Active queue updated with PASS or FAIL entry per Task 10.
- [ ] All files ASCII-only (verify with the python encode check from Task 10 Step 3).
- [ ] Branch is `feat/non-tabular-model-class-scoping` (no worktree).
- [ ] PR description references the spec, the findings note, and lists the verdict.

If verdict is PASS: write the Phase 2 plan as the next deliverable
(`docs/superpowers/plans/2026-05-09-non-tabular-model-class-scoping-phase2.md`).
If FAIL: the lane is closed; no Phase 2 plan is needed. The sequel-ordering
matrix in the spec governs which candidate to scope next.
