"""Figures for the modelling; shared by the runner and the notebook.

Every function takes the tidy outputs of ``src.run_models`` and returns the matplotlib figure;
pass ``path`` to save it.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _finish(fig, path):
    fig.tight_layout()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130)
        plt.close(fig)
    return fig


def _columns(ic_summary: pd.DataFrame, h: int, variant: str) -> pd.DataFrame:
    """Rows of one horizon: score baselines (no variant) plus everything fitted on ``variant``."""
    return ic_summary[(ic_summary["horizon"] == h) & ic_summary["variant"].isin([variant, "none"])]


def plot_cumulative_ic(daily_ic: pd.DataFrame, ic_summary: pd.DataFrame, h: int, variant: str, path=None):
    """Cumulative sum of the daily out-of-sample IC of every predictor at one horizon."""
    rows = _columns(ic_summary, h, variant)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for _, row in rows.iterrows():
        daily_ic[row["column"]].fillna(0).cumsum().plot(ax=ax, label=f"{row['predictor']} ({row['ic_mean']:+.4f})")
    ax.axhline(0, color="k", lw=0.8)
    ax.set(title=f"Cumulative OOS IC, h = {h}d ({variant} target)", ylabel="sum of daily IC", xlabel="")
    ax.legend(fontsize=8)
    return _finish(fig, path)


def plot_ic_by_year(stability: pd.DataFrame, h: int, variant: str, path=None):
    """Heatmap of the mean OOS IC by calendar year and predictor."""
    sub = stability[(stability["kind"] == "year") & (stability["horizon"] == h)
                    & stability["variant"].isin([variant, "none"])]
    pivot = sub.pivot(index="predictor", columns="label", values="ic_mean")
    fig, ax = plt.subplots(figsize=(11, 0.5 * len(pivot) + 1.5))
    sns.heatmap(pivot, ax=ax, cmap="RdBu_r", center=0, annot=True, fmt=".3f", annot_kws={"size": 7}, cbar=False)
    ax.set(title=f"OOS IC by year, h = {h}d ({variant} target)", xlabel="", ylabel="")
    return _finish(fig, path)


def plot_coefficient_paths(coefs: pd.DataFrame, variant: str, path=None):
    """Standardised Ridge and Elastic Net coefficients per fold (test year) and horizon."""
    sub = coefs[coefs["model"].isin(["ridge", "elastic_net"]) & (coefs["variant"] == variant)]
    panels = sub[["model", "horizon"]].drop_duplicates().sort_values(["horizon", "model"]).to_numpy()
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.5), squeeze=False)
    for ax, (model, h) in zip(axes[0], panels):
        part = sub[(sub["model"] == model) & (sub["horizon"] == h)]
        part.pivot(index="fold_id", columns="feature", values="coef").plot(ax=ax, legend=False, lw=1)
        ax.axhline(0, color="k", lw=0.8)
        ax.set(title=f"{model}, h = {h}d", xlabel="fold (0 = first test year)", ylabel="coefficient")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=7)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    return _finish(fig, path)


def plot_lightgbm_importance(perm: pd.DataFrame, shap: pd.DataFrame, variant: str, path=None):
    """Mean over folds of the permutation IC drop and of mean |SHAP|, per horizon."""
    p = perm[perm["variant"] == variant].groupby(["horizon", "feature"])["ic_drop_mean"].mean().unstack(0)
    s = shap[shap["variant"] == variant].groupby(["horizon", "feature"])["mean_abs_shap"].mean().unstack(0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    p.sort_values(p.columns[0]).plot.barh(ax=axes[0])
    axes[0].set(title="permutation importance (drop in mean daily IC)", ylabel="")
    s.sort_values(s.columns[0]).plot.barh(ax=axes[1])
    axes[1].set(title="mean |SHAP| (target sd units)", ylabel="")
    return _finish(fig, path)


def plot_decile_means(decile_means: pd.DataFrame, h: int, variant: str, spread: str = "beta_hedged", path=None):
    """Mean return per prediction decile (one line per predictor) for one horizon and spread type."""
    sub = decile_means[(decile_means["horizon"] == h) & decile_means["variant"].isin([variant, "none"])
                       & (decile_means["spread"] == spread)]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sub.pivot(index="decile", columns="predictor", values="mean").plot(ax=ax, marker="o")
    ax.set(title=f"Mean {spread} return per prediction decile, h = {h}d", xlabel="decile (10 = highest prediction)")
    return _finish(fig, path)
