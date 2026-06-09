"""Fuzzy preference rules for multi-objective PINN trade-offs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def triangular(x: float, a: float, b: float, c: float) -> float:
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a) if b > a else 0.0
    return (c - x) / (c - b) if c > b else 0.0


@dataclass
class FuzzyRuleParams:
    """Genome for evolutionary fuzzy system (18 parameters)."""

    l2_ref: float = 0.5
    res_ref: float = 0.05
    n_ref: float = 300.0
    w_accuracy_base: float = 1.0
    w_physics_base: float = 1.0
    w_data_base: float = 1.0
    alpha_l2: float = 0.8
    alpha_res: float = 1.0
    alpha_n: float = 0.6
    l2_mf: tuple[float, float, float] = (0.5, 1.0, 2.0)
    res_mf: tuple[float, float, float] = (0.5, 1.0, 2.0)
    n_mf: tuple[float, float, float] = (0.6, 1.0, 1.6)

    def to_vector(self) -> np.ndarray:
        return np.array(
            [
                self.l2_ref,
                self.res_ref,
                self.n_ref,
                self.w_accuracy_base,
                self.w_physics_base,
                self.w_data_base,
                self.alpha_l2,
                self.alpha_res,
                self.alpha_n,
                *self.l2_mf,
                *self.res_mf,
                *self.n_mf,
            ],
            dtype=float,
        )

    @classmethod
    def from_vector(cls, v: np.ndarray) -> FuzzyRuleParams:
        v = np.asarray(v, dtype=float).ravel()
        return cls(
            l2_ref=float(max(v[0], 1e-4)),
            res_ref=float(max(v[1], 1e-4)),
            n_ref=float(max(v[2], 1.0)),
            w_accuracy_base=float(max(v[3], 0.1)),
            w_physics_base=float(max(v[4], 0.1)),
            w_data_base=float(max(v[5], 0.1)),
            alpha_l2=float(max(v[6], 0.0)),
            alpha_res=float(max(v[7], 0.0)),
            alpha_n=float(max(v[8], 0.0)),
            l2_mf=(float(v[9]), float(v[10]), float(v[11])),
            res_mf=(float(v[12]), float(v[13]), float(v[14])),
            n_mf=(float(v[15]), float(v[16]), float(v[17])),
        )

    @classmethod
    def default(cls) -> FuzzyRuleParams:
        return cls()

    @classmethod
    def random(cls, rng: np.random.Generator) -> FuzzyRuleParams:
        return cls(
            l2_ref=float(rng.uniform(0.1, 1.0)),
            res_ref=float(rng.uniform(0.01, 0.2)),
            n_ref=float(rng.uniform(100, 500)),
            w_accuracy_base=float(rng.uniform(0.5, 2.0)),
            w_physics_base=float(rng.uniform(0.5, 2.0)),
            w_data_base=float(rng.uniform(0.5, 2.0)),
            alpha_l2=float(rng.uniform(0.2, 1.5)),
            alpha_res=float(rng.uniform(0.2, 1.5)),
            alpha_n=float(rng.uniform(0.2, 1.5)),
            l2_mf=tuple(rng.uniform(0.2, 2.5, size=3)),
            res_mf=tuple(rng.uniform(0.2, 2.5, size=3)),
            n_mf=tuple(rng.uniform(0.2, 2.5, size=3)),
        )


def _triangular(x: float, a: float, b: float, c: float) -> float:
    return triangular(x, a, b, c)


def fuzzy_preference_score(
    l2_error: float,
    pde_residual: float,
    data_budget: float,
    params: FuzzyRuleParams | None = None,
) -> float:
    """
    Higher score = preferred compromise.
    Mamdani-style rule aggregation with evolvable parameters.
    """
    p = params or FuzzyRuleParams.default()
    l2_n = l2_error / (p.l2_ref + 1e-12)
    res_n = pde_residual / (p.res_ref + 1e-12)
    n_n = data_budget / (p.n_ref + 1e-12)

    a, b, c = sorted(p.l2_mf)
    l2_high = _triangular(l2_n, a, b, c)
    a, b, c = sorted(p.res_mf)
    residual_high = _triangular(res_n, a, b, c)
    a, b, c = sorted(p.n_mf)
    data_high = _triangular(n_n, a, b, c)

    w_accuracy = p.w_accuracy_base + p.alpha_l2 * l2_high
    w_physics = p.w_physics_base + p.alpha_res * residual_high
    w_data = p.w_data_base + p.alpha_n * data_high

    penalties = np.array([l2_n, res_n, n_n])
    weights = np.array([w_accuracy, w_physics, w_data])
    weighted_penalty = float(np.sum(weights * penalties) / np.sum(weights))
    return 1.0 / (1.0 + weighted_penalty)


def rank_with_fuzzy(
    candidates: list[dict[str, float]],
    params: FuzzyRuleParams | None = None,
) -> list[dict[str, float]]:
    ranked = []
    for row in candidates:
        f3 = row.get("f3_data_budget", row.get("f3_n_collocation", 0.0))
        score = fuzzy_preference_score(
            row["f1_l2_error"],
            row["f2_pde_residual"],
            float(f3),
            params=params,
        )
        ranked.append({**row, "fuzzy_score": score})
    ranked.sort(key=lambda r: r["fuzzy_score"], reverse=True)
    return ranked


def select_compromise(candidates: list[dict[str, float]], use_fuzzy: bool = True) -> dict[str, float]:
    if not candidates:
        raise ValueError("No candidates to select from.")
    if use_fuzzy:
        return rank_with_fuzzy(candidates, params=None)[0]

    cols = ("f1_l2_error", "f2_pde_residual")
    f3_key = "f3_data_budget" if "f3_data_budget" in candidates[0] else "f3_n_collocation"
    pts = np.array([[c["f1_l2_error"], c["f2_pde_residual"], c.get(f3_key, 0.0)] for c in candidates])
    ideal = pts.min(axis=0)
    idx = int(np.argmin(np.linalg.norm(pts - ideal, axis=1)))
    return candidates[idx]
