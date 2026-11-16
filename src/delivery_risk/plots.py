"""Charts for the README. Static PNGs on a light surface.

Colours carry identity only: blue is the handoff model, orange the rule of
thumb (or the flattering number), aqua the checkout model. Text stays in ink.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402

SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

_PREFERRED = ["Segoe UI", "Helvetica Neue", "Arial", "Liberation Sans", "DejaVu Sans"]
_INSTALLED = {f.name for f in font_manager.fontManager.ttflist}

plt.rcParams.update(
    {
        "font.family": next((f for f in _PREFERRED if f in _INSTALLED), "sans-serif"),
        "font.size": 9,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    }
)


def _frame(title: str, subtitle: str, size=(7.6, 3.9), left=0.09, right=0.86, legend_row=True):
    """A figure with a left-aligned title block and one plot area below it."""
    width, height = size
    fig, ax = plt.subplots(figsize=size, dpi=160)
    top_inches = 1.05 if legend_row else 0.8  # room for title, subtitle and an optional legend row
    fig.subplots_adjust(top=1 - top_inches / height, left=left, right=right, bottom=0.5 / height)
    fig.text(0.02, 1 - 0.12 / height, title, fontsize=12.5, fontweight="bold", color=INK, va="top")
    fig.text(0.02, 1 - 0.40 / height, subtitle, fontsize=9, color=INK_2, va="top")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def _line(ax, x, y, color, label):
    ax.plot(
        x, y, color=color, linewidth=2, solid_capstyle="round", solid_joinstyle="round",
        marker="o", markersize=6.5, markeredgecolor=SURFACE, markeredgewidth=1.6, label=label,
    )  # fmt: skip


def _percent_axis(ax):
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")


def _legend(ax, ncols=4):
    ax.legend(
        loc="lower left", bbox_to_anchor=(0, 1.02), ncols=ncols, frameon=False, fontsize=8.5,
        labelcolor=INK_2, handlelength=1.6, columnspacing=1.4, borderaxespad=0,
    )  # fmt: skip


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def recall_by_month(handoff: pd.DataFrame, rule: pd.DataFrame, checkout: pd.DataFrame, budget: float, path: Path):
    """Share of late orders caught each month when alerting on the riskiest `budget` of orders."""
    fig, ax = _frame(
        "Late orders caught by alerting on the riskiest 10% of orders",
        "Each month is scored by a model trained only on outcomes known before that month began.",
    )
    months = list(handoff["month"])
    _line(ax, months, handoff["recall_at_budget"], BLUE, "Model at carrier handoff")
    _line(ax, months, rule["recall_at_budget"], ORANGE, "Slack rule at handoff")
    _line(ax, months, checkout["recall_at_budget"], AQUA, "Model at checkout")
    ax.axhline(budget, color=MUTED, linewidth=1)
    ax.annotate(
        "Random alerts would catch 10%", (0, budget), xytext=(0, 5), textcoords="offset points",
        va="bottom", fontsize=8.5, color=MUTED,
    )  # fmt: skip
    ax.set_ylim(0, max(0.75, float(handoff["recall_at_budget"].max()) + 0.08))
    _percent_axis(ax)
    _legend(ax)
    _save(fig, path)


def score_vs_actual(monitoring: pd.DataFrame, path: Path):
    """Average score against the late rate that was later observed, by month."""
    fig, ax = _frame(
        "The scores rank orders well but drift as probabilities",
        "Average score for the month against the share of orders that turned out late.",
    )
    months = list(monitoring["month"])
    _line(ax, months, monitoring["mean_score"], BLUE, "Average model score")
    _line(ax, months, monitoring["actual_late_rate"], ORANGE, "Actual late rate")
    ax.set_ylim(0, float(monitoring[["mean_score", "actual_late_rate"]].max().max()) * 1.18)
    _percent_axis(ax)
    _legend(ax)
    _save(fig, path)


def leak_check(rows: list[dict], path: Path):
    """ROC AUC from a shuffled split against the walk-forward backtest, per stage."""
    fig, ax = _frame(
        "A shuffled split flatters the model",
        "ROC AUC on the same months, measured two ways. The backtest is the honest one.",
        size=(7.6, 3.1),
        left=0.2,
        right=0.94,
    )
    fig.subplots_adjust(bottom=0.2)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    for i, row in enumerate(rows):
        ax.plot([row["walk_forward"], row["random_split"]], [i, i], color=AXIS, linewidth=2, zorder=1)
        for key, color, dx, align in (("walk_forward", BLUE, -9, "right"), ("random_split", ORANGE, 9, "left")):
            ax.scatter(
                [row[key]], [i], s=95, color=color, edgecolor=SURFACE, linewidth=1.6, zorder=3,
                label={"walk_forward": "Walk-forward backtest", "random_split": "Shuffled 80/20 split"}[key]
                if i == 0 else None,
            )  # fmt: skip
            ax.annotate(
                f"{row[key]:.2f}", (row[key], i), xytext=(dx, 0), textcoords="offset points",
                ha=align, va="center", fontsize=9, color=INK,
            )  # fmt: skip
    ax.set_yticks(range(len(rows)), [r["label"] for r in rows])
    ax.tick_params(axis="y", labelcolor=INK_2, labelsize=9)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("ROC AUC (0.5 is a coin flip)")
    _legend(ax, ncols=2)
    _save(fig, path)


def importance(table: pd.DataFrame, path: Path):
    """Permutation importance for the handoff model: one colour, sorted bars."""
    table = table.sort_values("importance")
    fig, ax = _frame(
        "What the handoff model leans on",
        "Drop in average precision when one input is shuffled, on the last backtest month.",
        size=(7.6, 0.34 * len(table) + 1.5),
        left=0.3,
        right=0.9,
        legend_row=False,
    )
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_xlim(0, float(table["importance"].max()) * 1.12)
    ax.set_ylim(-0.6, len(table) - 0.4)
    thickness = 11  # points; thin bars with a rounded data end and a square baseline
    fig.canvas.draw()
    to_data = ax.transData.inverted()
    cap = (to_data.transform((thickness / 2 * fig.dpi / 72, 0)) - to_data.transform((0, 0)))[0]
    for i, value in enumerate(table["importance"]):
        value = max(float(value), 0.0)
        if value > 2 * cap:
            ax.plot([0, value - cap], [i, i], color=BLUE, linewidth=thickness, solid_capstyle="butt")
            ax.plot([value - 1.5 * cap, value - cap], [i, i], color=BLUE, linewidth=thickness, solid_capstyle="round")
        else:
            ax.plot([0, value], [i, i], color=BLUE, linewidth=thickness, solid_capstyle="butt")
        ax.annotate(
            f"{value:.3f}", (value, i), xytext=(6, 0), textcoords="offset points",
            va="center", fontsize=8.5, color=INK_2,
        )  # fmt: skip
    ax.set_yticks(range(len(table)), table["feature"])
    ax.tick_params(axis="y", labelcolor=INK_2, labelsize=9)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.2f}")
    _save(fig, path)
