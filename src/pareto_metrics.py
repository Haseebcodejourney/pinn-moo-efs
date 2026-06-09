"""Pareto front quality metrics: hypervolume and spacing."""

from __future__ import annotations

import numpy as np
import pandas as pd

OBJECTIVE_COLS = ("f1_l2_error", "f2_pde_residual", "f3_data_budget")


def objective_column_names(df: pd.DataFrame) -> tuple[str, str, str]:
    f3 = "f3_data_budget" if "f3_data_budget" in df.columns else "f3_n_collocation"
    return "f1_l2_error", "f2_pde_residual", f3


def sanitize_objectives_df(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with missing or non-numeric objective values."""
    f1, f2, f3 = objective_column_names(df)
    out = df.copy()
    for col in (f1, f2, f3):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if f3 != "f3_data_budget" and "f3_data_budget" not in out.columns:
        out = out.rename(columns={f3: "f3_data_budget"})
    return out.dropna(subset=["f1_l2_error", "f2_pde_residual", "f3_data_budget"]).reset_index(drop=True)


def _as_array(points: np.ndarray | list[dict]) -> np.ndarray:
    if isinstance(points, list):
        rows = []
        for p in points:
            try:
                row = [
                    float(p.get("f1_l2_error", np.nan)),
                    float(p.get("f2_pde_residual", np.nan)),
                    float(p.get("f3_data_budget", p.get("f3_n_collocation", np.nan))),
                ]
            except (TypeError, ValueError):
                continue
            if not np.isnan(row).any():
                rows.append(row)
        return np.array(rows, dtype=float) if rows else np.empty((0, 3))
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return pts.reshape(0, 3)
    return pts[~np.isnan(pts).any(axis=1)]


def normalize_objectives(points: np.ndarray, ref: np.ndarray | None = None) -> np.ndarray:
    pts = _as_array(points)
    if len(pts) == 0:
        return pts
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    span = np.where(hi - lo < 1e-12, 1.0, hi - lo)
    normed = (pts - lo) / span
    if ref is not None:
        normed = np.vstack([normed, (ref - lo) / span])
    return normed


def hypervolume(points: np.ndarray | list[dict], ref_point: np.ndarray | None = None) -> float:
    """Monte-Carlo hypervolume estimate for 3-objective minimization."""
    pts = _as_array(points)
    if len(pts) == 0:
        return 0.0

    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    span = np.maximum(hi - lo, 1e-8)

    if ref_point is None:
        ref_point = hi + 0.1 * span
    else:
        ref_point = np.asarray(ref_point, dtype=float)

    ref_point = np.maximum(ref_point, lo + 1e-8)
    if np.any(ref_point <= lo):
        return 0.0

    n_samples = 5000
    rng = np.random.default_rng(42)
    samples = rng.uniform(lo, ref_point, size=(n_samples, pts.shape[1]))

    dominated = 0.0
    for s in samples:
        if np.any(np.all(pts <= s, axis=1)):
            dominated += 1.0

    vol_box = float(np.prod(ref_point - lo))
    return float(dominated / n_samples * vol_box)


def spacing(points: np.ndarray | list[dict]) -> float:
    """Lower spacing = more uniform Pareto distribution (Schott's metric)."""
    pts = _as_array(points)
    if len(pts) < 2:
        return 0.0

    dists = []
    for i, p in enumerate(pts):
        others = np.delete(pts, i, axis=0)
        dmin = np.min(np.linalg.norm(others - p, axis=1))
        dists.append(dmin)
    d_mean = float(np.mean(dists))
    return float(np.sqrt(np.mean([(d - d_mean) ** 2 for d in dists])))
