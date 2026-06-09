"""Generate Figure 1: method workflow diagram."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures" / "fig_workflow.png"


def _box(ax, xy, text, width=2.6, height=0.75, fc="#E8F4FD", ec="#1a5276"):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.05,rounding_size=0.08",
        linewidth=1.5,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=9, wrap=True)


def _arrow(ax, start, end):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.4,
            color="#2c3e50",
        )
    )


def main() -> None:
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 3)
    ax.axis("off")

    # Row 1: main pipeline
    _box(ax, (0.3, 1.1), "PINN training\n(λ_data, λ_pde, N_coll)", fc="#D5F5E3")
    _box(ax, (3.3, 1.1), "Objective evaluation\nf₁ L2, f₂ residual, f₃ budget", fc="#FCF3CF")
    _box(ax, (6.3, 1.1), "MOO search\nGrid / NSGA-II", fc="#FADBD8")
    _box(ax, (9.3, 1.1), "Pareto front\n+ metrics (HV, spacing)", fc="#E8DAEF")
    _box(ax, (12.0, 1.1), "EFS + selection\n(ideal, knee, fuzzy)", fc="#D6EAF8", width=1.7)

    _arrow(ax, (2.9, 1.47), (3.3, 1.47))
    _arrow(ax, (5.9, 1.47), (6.3, 1.47))
    _arrow(ax, (8.9, 1.47), (9.3, 1.47))
    _arrow(ax, (11.9, 1.47), (12.0, 1.47))

    # Inputs / outputs
    _box(ax, (0.3, 2.05), "Inputs: PDE, IC/BC,\nsparse obs (optional)", width=2.6, height=0.65, fc="#F8F9F9", ec="#7f8c8d")
    _box(ax, (6.3, 2.05), "Decision vars:\n(λ_data, λ_pde, N_coll)", width=2.6, height=0.65, fc="#F8F9F9", ec="#7f8c8d")
    _box(ax, (12.0, 2.05), "Output:\ncompromise PINN", width=1.7, height=0.65, fc="#F8F9F9", ec="#7f8c8d")

    ax.annotate("", xy=(1.6, 2.05), xytext=(1.6, 1.85), arrowprops=dict(arrowstyle="-|>", color="#7f8c8d"))
    ax.annotate("", xy=(7.6, 2.05), xytext=(7.6, 1.85), arrowprops=dict(arrowstyle="-|>", color="#7f8c8d"))
    ax.annotate("", xy=(12.85, 1.85), xytext=(12.85, 2.05), arrowprops=dict(arrowstyle="-|>", color="#7f8c8d"))

    # Baselines row
    _box(ax, (9.3, 0.15), "Baselines:\nfixed weights, GradNorm", width=2.6, height=0.6, fc="#FDFEFE", ec="#95a5a6")
    ax.annotate("", xy=(10.6, 0.75), xytext=(10.6, 1.1), arrowprops=dict(arrowstyle="-|>", color="#95a5a6", linestyle="dashed"))

    ax.set_title("Figure 1. Multi-objective PINN optimization workflow", fontsize=11, fontweight="bold", pad=12)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
