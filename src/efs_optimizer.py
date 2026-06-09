"""Evolutionary Fuzzy System: GA over membership parameters and rule weights."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fuzzy_rules import FuzzyRuleParams, rank_with_fuzzy
from pareto_metrics import OBJECTIVE_COLS


def _balanced_loss(candidate: dict[str, float]) -> float:
    f3 = candidate.get("f3_data_budget", candidate.get("f3_n_collocation", 1.0))
    vals = np.array([candidate["f1_l2_error"], candidate["f2_pde_residual"], float(f3)])
    return float(np.sum(vals / (vals.max() + 1e-12)))


def efs_fitness(params: FuzzyRuleParams, pareto_rows: list[dict[str, float]]) -> float:
    """
    Fitness = negative balanced loss of top fuzzy-ranked Pareto point.
    Lower balanced loss -> higher fitness.
    """
    if not pareto_rows:
        return -1e6
    top = rank_with_fuzzy(pareto_rows, params=params)[0]
    return -_balanced_loss(top)


def _crossover(p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    mask = rng.random(len(p1)) < 0.5
    child = np.where(mask, p1, p2)
    return child


def _mutate(v: np.ndarray, rng: np.random.Generator, rate: float = 0.2, scale: float = 0.15) -> np.ndarray:
    out = v.copy()
    for i in range(len(out)):
        if rng.random() < rate:
            out[i] *= float(rng.uniform(1.0 - scale, 1.0 + scale))
    return out


def evolve_efs(
    pareto_df: pd.DataFrame,
    pop_size: int = 20,
    n_gen: int = 30,
    seed: int = 42,
) -> tuple[FuzzyRuleParams, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    pareto_rows = pareto_df.to_dict(orient="records")

    population = [FuzzyRuleParams.random(rng).to_vector() for _ in range(pop_size)]
    population[0] = FuzzyRuleParams.default().to_vector()

    history = []
    best_params = FuzzyRuleParams.default()
    best_fit = -np.inf

    for gen in range(n_gen):
        scored = []
        for vec in population:
            params = FuzzyRuleParams.from_vector(vec)
            fit = efs_fitness(params, pareto_rows)
            scored.append((fit, vec, params))
            if fit > best_fit:
                best_fit = fit
                best_params = params

        scored.sort(key=lambda x: x[0], reverse=True)
        history.append({"generation": gen, "best_fitness": best_fit})

        elites = [s[1] for s in scored[: max(2, pop_size // 5)]]
        next_pop = elites.copy()
        while len(next_pop) < pop_size:
            i, j = rng.choice(len(scored), size=2, replace=False)
            child = _mutate(_crossover(scored[i][1], scored[j][1], rng), rng)
            next_pop.append(child)
        population = next_pop

    hist_df = pd.DataFrame(history)
    return best_params, hist_df


def run_efs(
    pareto_path: Path,
    out_dir: Path,
    pop_size: int = 20,
    n_gen: int = 30,
    seed: int = 42,
) -> dict:
    pareto_df = pd.read_csv(pareto_path)
    best_params, hist = evolve_efs(pareto_df, pop_size=pop_size, n_gen=n_gen, seed=seed)

    pareto_rows = pareto_df.to_dict(orient="records")
    static_top = rank_with_fuzzy(pareto_rows, params=None)[0]
    evolved_top = rank_with_fuzzy(pareto_rows, params=best_params)[0]

    static_loss = _balanced_loss(static_top)
    evolved_loss = _balanced_loss(evolved_top)

    result = {
        "static_fuzzy_balanced_loss": static_loss,
        "evolved_fuzzy_balanced_loss": evolved_loss,
        "efs_improves": evolved_loss < static_loss,
        "best_params": best_params.to_vector().tolist(),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    hist.to_csv(out_dir / "efs_evolution.csv", index=False)
    with open(out_dir / "efs_best_params.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    comparison = pd.DataFrame(
        [
            {"method": "static_fuzzy", **static_top, "balanced_loss": static_loss},
            {"method": "evolved_fuzzy", **evolved_top, "balanced_loss": evolved_loss},
        ]
    )
    comparison.to_csv(out_dir / "efs_comparison.csv", index=False)
    return {"params": best_params, "result": result, "comparison": comparison}
