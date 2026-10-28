"""Tables assembled from the modelling CSVs (shared by the notebook and the README)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ORDER = ["B0_reversal", "B1_composite", "B2_fama_macbeth", "ridge", "elastic_net", "lightgbm"]


def _read(out_dir: str | Path, name: str) -> pd.DataFrame:
    return pd.read_csv(Path(out_dir) / f"{name}.csv")


def _ordered(table: pd.DataFrame) -> pd.DataFrame:
    rank = {p: i for i, p in enumerate(ORDER)}
    return (table.assign(_o=table["predictor"].map(rank))
            .sort_values(["horizon", "variant", "_o"]).drop(columns="_o").reset_index(drop=True))


def headline_table(out_dir: str | Path) -> pd.DataFrame:
    """One row per predictor x horizon x target: OOS IC, Holm, beta-hedged and rank decile spreads, turnover."""
    ic = _read(out_dir, "ic_summary")
    spreads = _read(out_dir, "decile_spreads").pivot_table(index="column", columns="spread",
                                                            values=["mean", "nw_t"])
    spreads.columns = [f"{spread}_{stat}" for stat, spread in spreads.columns]
    to = _read(out_dir, "prediction_turnover").set_index("column")
    table = ic.set_index("column")[["predictor", "horizon", "variant", "ic_mean", "ic_ir", "hit_rate", "nw_t",
                                    "p_value_holm", "reject_holm", "ic_mean_dispersion_weighted",
                                    "oos_r2_vs_zero"]]
    table = table.join(spreads[["beta_hedged_mean", "beta_hedged_nw_t", "rank_mean", "rank_nw_t"]])
    table = table.join(to[["rank_autocorrelation", "turnover_D1", "turnover_D10"]])
    return _ordered(table.reset_index())


def comparison_table(out_dir: str | Path) -> pd.DataFrame:
    """Paired daily-IC differences (model minus baseline), NW t and Holm over all comparisons."""
    return _read(out_dir, "model_comparison")[["model", "baseline", "horizon", "variant", "ic_diff_mean", "nw_t",
                                               "p_value", "p_value_holm", "reject_holm"]]


def period_table(out_dir: str | Path, variant: str) -> pd.DataFrame:
    """Mean OOS IC by sub-period and OOS half for every predictor fitted on ``variant`` (plus B0 / B1)."""
    stab = _read(out_dir, "ic_stability")
    sub = stab[(stab["kind"] != "year") & stab["variant"].isin([variant, "none"])]
    table = sub.pivot_table(index=["horizon", "predictor"], columns="label", values="ic_mean")
    return table[["gfc", "y2010_2019", "post2020", "first", "second"]]


def ablation_table(out_dir: str | Path) -> pd.DataFrame:
    """OOS IC of each family on its own and of the model without it, next to the all-feature model."""
    abl = _read(out_dir, "family_ablation")
    keys = ["model", "horizon", "variant"]
    full = abl[abl["kind"] == "all_features"].set_index(keys)["ic_mean"].rename("all_features")
    table = abl[abl["kind"] != "all_features"].pivot_table(index=[*keys, "family"], columns="kind", values="ic_mean")
    return table.join(full, on=keys)[["all_features", "single_family", "leave_one_family_out"]]


def sign_table(out_dir: str | Path, variant: str) -> pd.DataFrame:
    """Mean linear coefficient and LightGBM SHAP direction per feature, against the research prior."""
    signs = _read(out_dir, "coefficient_signs")
    signs = signs[(signs["variant"] == variant) & signs["model"].isin(["ridge", "elastic_net"])]
    table = signs.pivot_table(index="feature", columns=["model", "horizon"], values="coef_mean")
    table.columns = [f"{m}_h{h}" for m, h in table.columns]
    shap = _read(out_dir, "lightgbm_shap")
    shap = shap[shap["variant"] == variant].groupby(["feature", "horizon"])["shap_value_corr"].mean().unstack()
    shap.columns = [f"lgbm_shap_dir_h{h}" for h in shap.columns]
    prior = signs.groupby("feature")["prior_sign"].first()
    return table.join(shap).join(prior)
