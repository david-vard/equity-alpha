"""Headline tables assembled from the signal-research CSVs (shared by the runner, notebook and README)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def headline_table(out_dir: str | Path) -> pd.DataFrame:
    """One row per feature x horizon: IC, significance after correction, gross long-short and turnover.

    ``ls_sharpe_gross`` is the raw D10 - D1 Sharpe (no costs); ``turnover`` is the mean one-way
    turnover of the two legs.
    """
    out_dir = Path(out_dir)
    ic = pd.read_csv(out_dir / "ic_summary.csv")
    ls = pd.read_csv(out_dir / "long_short.csv")
    to = pd.read_csv(out_dir / "turnover.csv")
    top = [c for c in to.columns if c.endswith("_turnover") and not c.startswith("D1_")][0]
    keys = ["feature", "horizon"]
    table = ic[[*keys, "family", "expected_sign", "ic_mean", "nw_t", "p_value", "p_value_holm", "p_value_bh",
                "reject_holm", "reject_bh", "reject_by", "sign_consistent"]]
    table = table.merge(ls[[*keys, "gross_raw_sharpe"]].rename(columns={"gross_raw_sharpe": "ls_sharpe_gross"}), on=keys)
    to = to.assign(turnover=(to["D1_turnover"] + to[top]) / 2)
    return table.merge(to[[*keys, "turnover"]], on=keys)


def markdown_table(table: pd.DataFrame) -> str:
    """Compact markdown rendering of ``headline_table`` for the README."""
    def p(x: float) -> str:
        return f"{x:.4f}" if x >= 1e-4 else "<0.0001"

    lines = [
        "| Feature | h | IC mean | NW t | Holm p | BH p | Gross L/S Sharpe | Turnover |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in table.itertuples():
        lines.append(
            f"| `{r.feature}` | {r.horizon} | {r.ic_mean:+.4f} | {r.nw_t:+.2f} | {p(r.p_value_holm)} | "
            f"{p(r.p_value_bh)} | {r.ls_sharpe_gross:+.2f} | {r.turnover:.2f} |"
        )
    return "\n".join(lines)


def survivors(headline: pd.DataFrame, method: str = "holm") -> pd.DataFrame:
    """Rows whose IC t-test is rejected after the given multiple-testing adjustment."""
    return headline[headline[f"reject_{method}"]].sort_values(["horizon", "nw_t"]).reset_index(drop=True)


def half_life_table(out_dir: str | Path) -> pd.DataFrame:
    """Per feature: IC at k = 0 and k = 20, cumulative IC over the window and the half-life (NaN if none)."""
    decay = pd.read_csv(Path(out_dir) / "ic_decay.csv")
    grouped = decay.groupby("feature", sort=False)
    return pd.DataFrame(
        {
            "ic_k0": grouped["ic_mean"].first(),
            "t_k0": grouped["nw_t"].first(),
            "ic_last": grouped["ic_mean"].last(),
            "cum_ic": grouped["cum_ic"].last(),
            "half_life_days": grouped["half_life"].first(),
        }
    )


def period_table(out_dir: str | Path, horizon: int, features: list[str]) -> pd.DataFrame:
    """Mean IC by sub-period and by half of the research sample for the given features and horizon."""
    stability = pd.read_csv(Path(out_dir) / "ic_stability.csv")
    part = stability[(stability["kind"].isin(["period", "half"])) & (stability["horizon"] == horizon)]
    table = part[part["feature"].isin(features)].pivot_table(index="feature", columns="label", values="ic_mean")
    return table.loc[features, ["dotcom", "gfc", "y2010_2019", "post2020", "first", "second"]]


def terminal_effect(out_dir: str | Path) -> pd.Series:
    """Largest change in mean IC and NW t when delisted-stock exits (fwd_terminal) are excluded."""
    t = pd.read_csv(Path(out_dir) / "ic_terminal_sensitivity.csv")
    return pd.Series(
        {
            "max_abs_change_ic_mean": (t["ic_mean_all"] - t["ic_mean_ex_terminal"]).abs().max(),
            "max_abs_change_nw_t": (t["nw_t_all"] - t["nw_t_ex_terminal"]).abs().max(),
        }
    )
