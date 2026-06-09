"""Consistent paths and logging for experiment runs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"


def run_tag(pde: str, mode: str, n_obs: int = 0, obs_noise: float = 0.0) -> str:
    noise_s = str(obs_noise).replace(".", "p")
    return f"{pde}_{mode}_n{n_obs}_s{noise_s}"


def result_csv(pde: str, mode: str, n_obs: int = 0, obs_noise: float = 0.0) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / f"{run_tag(pde, mode, n_obs, obs_noise)}.csv"


def append_run_log(row: dict) -> None:
    path = RESULTS_DIR / "all_runs_log.csv"
    df = pd.DataFrame([row])
    if path.exists():
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    df.to_csv(path, index=False)
