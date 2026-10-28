"""Figures for the signal research; shared by the runner and the notebook.

Every function takes tidy frames (the CSVs written by ``src.run_signal_research``) and returns
the matplotlib figure; pass ``path`` to save it.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _finish(fig, path):
    fig.tight_layout()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130)
    return fig


def _order(table: pd.DataFrame, features: list[str] | None) -> list[str]:
    return features or list(dict.fromkeys(table["feature"]))


def plot_tstat_overview(summary: pd.DataFrame, t_threshold: float = 3.0, path=None):
    """NW t-stat of the mean IC for every feature x horizon, with the |t| = 3 line."""
    pivot = summary.pivot(index="feature", columns="horizon", values="nw_t").loc[_order(summary, None)]
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(pivot) + 1.5))
    pivot.plot.barh(ax=ax, width=0.8)
    for x in (-t_threshold, t_threshold):
        ax.axvline(x, color="k", ls="--", lw=0.8)
    ax.invert_yaxis()
    ax.set(xlabel="Newey-West t of mean IC (research sample)", ylabel="")
    ax.legend(title="horizon (d)")
    return _finish(fig, path)


def plot_ic_year_heatmap(by_year: pd.DataFrame, horizon: int, path=None):
    """Mean daily IC by calendar year (rows: feature) for one horizon."""
    d = by_year[(by_year["kind"] == "year") & (by_year["horizon"] == horizon)]
    pivot = d.pivot(index="feature", columns="label", values="ic_mean").loc[_order(d, None)]
    fig, ax = plt.subplots(figsize=(11, 0.35 * len(pivot) + 1.5))
    lim = float(np.nanmax(np.abs(pivot.to_numpy()))) or 0.01
    sns.heatmap(pivot, ax=ax, cmap="RdBu_r", center=0, vmin=-lim, vmax=lim, cbar_kws={"label": "mean IC"})
    ax.set(title=f"Mean IC by calendar year, h = {horizon}d (2021 is partial)", xlabel="", ylabel="")
    return _finish(fig, path)


def plot_cumulative_ic(daily_ic: pd.DataFrame, features: list[str], title: str = "", path=None):
    """Cumulative sum of the daily IC (columns: features)."""
    fig, ax = plt.subplots(figsize=(8, 4))
    daily_ic[features].fillna(0).cumsum().plot(ax=ax, lw=1)
    ax.set(title=title, ylabel="cumulative IC", xlabel="")
    return _finish(fig, path)


def plot_decile_means(deciles: pd.DataFrame, horizon: int, features: list[str] | None = None, path=None):
    """Mean h-day return by decile (non-overlapping dates), one panel per feature."""
    d = deciles[deciles["horizon"] == horizon]
    names = _order(d, features)
    cols = 4
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 2.3 * rows), sharex=True, squeeze=False)
    for ax, name in zip(axes.flat, names):
        part = d[d["feature"] == name]
        ax.bar(part["decile"], part["mean_return"] * 100, color="#4c78a8")
        ax.set_title(name, fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in list(axes.flat)[len(names):]:
        ax.axis("off")
    fig.supxlabel("decile (1 = lowest feature value)")
    fig.supylabel(f"mean {horizon}d return, %")
    return _finish(fig, path)


def plot_decay(decay: pd.DataFrame, features: list[str] | None = None, path=None):
    """IC and cumulative IC against k (trading days after entry)."""
    names = _order(decay, features)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    for i, name in enumerate(names):
        part = decay[decay["feature"] == name]
        style = {"color": colors[i % 20], "ls": "-" if i < 20 else "--", "lw": 1}
        axes[0].plot(part["k"], part["ic_mean"], label=name, **style)
        axes[1].plot(part["k"], part["cum_ic"], label=name, **style)
    axes[0].set(xlabel="k (days after entry)", ylabel="mean IC vs 1d return")
    axes[1].set(xlabel="k (days after entry)", ylabel="cumulative IC")
    axes[1].legend(fontsize=6, ncol=2)
    return _finish(fig, path)


def plot_correlation(corr: pd.DataFrame, path=None):
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(corr, ax=ax, cmap="RdBu_r", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f", annot_kws={"size": 6})
    ax.set_title("Time-averaged cross-sectional rank correlation (grouped by family)")
    return _finish(fig, path)


def plot_long_short(spread: pd.DataFrame, horizon: int, features: list[str], path=None):
    """Cumulative gross D10-D1 return on non-overlapping dates, oriented by expected sign."""
    fig, ax = plt.subplots(figsize=(8, 4))
    (spread[features].fillna(0).cumsum() * 100).plot(ax=ax, lw=1)
    ax.set(title=f"Cumulative gross long-short, h = {horizon}d (sum of period returns)", ylabel="%", xlabel="")
    return _finish(fig, path)
