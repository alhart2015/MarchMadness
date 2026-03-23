"""Post-processing step: blend model probabilities with R64 Vegas closing lines."""
from scipy.stats import norm

SIGMA = 11.0


def blend_r64_probs(
    model_probs: dict[tuple[int, int], float],
    r64_lines: dict[tuple[int, int], float],
    weight: float = 0.35,
) -> dict[tuple[int, int], float]:
    """Blend model pairwise probabilities with R64 game-specific Vegas lines.

    Args:
        model_probs: Dict of {(team_a_id, team_b_id): P(a beats b)}.
        r64_lines: Dict of {(team_a_id, team_b_id): spread} where positive
            means team_a is favored.
        weight: Vegas weight in [0, 1]. 0 = pure model, 1 = pure Vegas.

    Returns:
        New dict with blended probabilities for games with lines,
        original probabilities for games without.
    """
    result = dict(model_probs)

    for key, model_p in model_probs.items():
        if key in r64_lines:
            spread = r64_lines[key]
            vegas_p = norm.cdf(spread / SIGMA)
            result[key] = (1 - weight) * model_p + weight * vegas_p
        elif (key[1], key[0]) in r64_lines:
            spread = r64_lines[(key[1], key[0])]
            vegas_p = 1 - norm.cdf(spread / SIGMA)
            result[key] = (1 - weight) * model_p + weight * vegas_p

    return result
