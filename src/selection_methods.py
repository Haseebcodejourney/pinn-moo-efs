"""Compromise selection methods for Pareto sets."""

from __future__ import annotations

import numpy as np

from fuzzy_rules import FuzzyRuleParams, fuzzy_preference_score, rank_with_fuzzy
from pareto_metrics import OBJECTIVE_COLS


def _objs(candidates: list[dict[str, float]]) -> np.ndarray:
    rows = []
    for row in candidates:
        try:
            f3 = row.get("f3_data_budget", row.get("f3_n_collocation", 0.0))
            vals = [float(row["f1_l2_error"]), float(row["f2_pde_residual"]), float(f3)]
        except (TypeError, ValueError, KeyError):
            continue
        if not any(np.isnan(v) for v in vals):
            rows.append(vals)
    if not rows:
        raise ValueError("No valid candidates with complete objectives.")
    return np.array(rows, dtype=float)


def select_ideal_point(candidates: list[dict[str, float]]) -> dict[str, float]:
    pts = _objs(candidates)
    ideal = pts.min(axis=0)
    dist = np.linalg.norm(pts - ideal, axis=1)
    return candidates[int(np.argmin(dist))]


def select_knee_point(candidates: list[dict[str, float]]) -> dict[str, float]:
    """Knee via maximum distance to utopia-nadir line in normalized objective space."""
    pts = _objs(candidates)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = np.where(hi - lo < 1e-12, 1.0, hi - lo)
    norm = (pts - lo) / span
    utopia = np.zeros(norm.shape[1])
    nadir = np.ones(norm.shape[1])
    line = nadir - utopia
    line = line / (np.linalg.norm(line) + 1e-12)
    distances = []
    for p in norm:
        proj = utopia + line * np.dot(p - utopia, line)
        distances.append(np.linalg.norm(p - proj))
    return candidates[int(np.argmax(distances))]


def select_nsga_first(candidates: list[dict[str, float]]) -> dict[str, float]:
    """First candidate (caller should pass rank-0 front)."""
    return candidates[0]


def select_static_fuzzy(candidates: list[dict[str, float]]) -> dict[str, float]:
    return rank_with_fuzzy(candidates, params=None)[0]


def select_evolved_fuzzy(candidates: list[dict[str, float]], params: FuzzyRuleParams) -> dict[str, float]:
    return rank_with_fuzzy(candidates, params=params)[0]


def compare_selections(
    candidates: list[dict[str, float]],
    efs_params: FuzzyRuleParams | None = None,
) -> list[dict[str, float]]:
    methods = {
        "ideal_point": select_ideal_point(candidates),
        "knee_point": select_knee_point(candidates),
        "nsga_first": select_nsga_first(candidates),
        "static_fuzzy": select_static_fuzzy(candidates),
    }
    if efs_params is not None:
        methods["evolved_fuzzy"] = select_evolved_fuzzy(candidates, efs_params)

    rows = []
    for name, row in methods.items():
        rows.append({"selection_method": name, **row})
    return rows
