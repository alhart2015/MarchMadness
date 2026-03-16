"""Bracket output formatting and export."""

import pandas as pd

ROUND_NAMES = {1: "R64", 2: "R32", 3: "S16", 4: "E8", 5: "F4", 6: "Champ"}


def format_advancement_table(
    advancement_probs: dict[int, dict[int, float]],
    teams: pd.DataFrame,
) -> str:
    """Format advancement probabilities as a readable table string."""
    team_map = dict(zip(teams["TeamID"], teams["TeamName"]))
    rows = []

    for team_id, rounds in sorted(
        advancement_probs.items(),
        key=lambda x: x[1].get(6, 0),
        reverse=True,
    ):
        name = team_map.get(team_id, str(team_id))
        probs = "  ".join(
            f"{ROUND_NAMES.get(r, f'R{r}')}: {rounds.get(r, 0):.1%}"
            for r in range(1, 7)
        )
        rows.append(f"{name:25s} {probs}")

    header = f"{'Team':25s} " + "  ".join(f"{ROUND_NAMES.get(r, f'R{r}'):>6s}" for r in range(1, 7))
    return header + "\n" + "-" * len(header) + "\n" + "\n".join(rows)


def export_bracket_csv(
    advancement_probs: dict[int, dict[int, float]],
    teams: pd.DataFrame,
    output_path: str,
) -> None:
    """Export advancement probabilities to CSV."""
    team_map = dict(zip(teams["TeamID"], teams["TeamName"]))
    rows = []
    for team_id, rounds in advancement_probs.items():
        row = {"TeamID": team_id, "TeamName": team_map.get(team_id, str(team_id))}
        for r in range(1, 7):
            row[f"R{r}"] = rounds.get(r, 0.0)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("R6", ascending=False)
    df.to_csv(output_path, index=False)
