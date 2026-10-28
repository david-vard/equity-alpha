import copy

import numpy as np
import pandas as pd
import pytest
import yaml

from src.backtest import DAILY_COLUMNS, MarketData, performance, run_backtest
from src.costs import CostModel
from src.portfolio import (
    PORTFOLIO_CONFIG_PATH,
    OptimiserSpec,
    Target,
    check_constraints,
    decile_builder,
    optimised_builder,
    volscaled_builder,
)
from src.signals import quantile_buckets, quantile_returns

PCFG = yaml.safe_load(open(PORTFOLIO_CONFIG_PATH))
COSTS = CostModel.from_config(PCFG, 252)
ZERO = COSTS.scenario(multiplier=0.0)
HOLDOUT = "2030-01-01"
LAG, H = 1, 5


def synthetic(T=90, N=40, seed=0, h=H, lag=LAG, start="2010-01-01"):
    """Prices for every name; panel rows only where the name is a member and traded. Name 0 leaves
    the index on day 40 but keeps trading, name 1 delists after day 47, and some days are gaps."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=T)
    P = 100 * np.cumprod(1 + rng.normal(0.0003, 0.015, (T, N)) + rng.normal(0, 0.01, (T, 1)), axis=0)
    P[48:, 1] = np.nan                                   # name 1: last price on day 47
    member = np.isfinite(P)
    member[41:, 0] = False                               # name 0: leaves the index after day 40
    gaps = rng.random((T, N)) < 0.02
    gaps[:, :2] = False
    member &= ~gaps
    prev = np.vstack([np.full((1, N), np.nan), P[:-1]])
    ret = np.where(member, P / prev - 1, np.nan)
    label = np.full((T, N), np.nan)
    last_price = np.array([P[np.flatnonzero(np.isfinite(P[:, i]))[-1], i] for i in range(N)])
    for s in range(T - lag - h):
        entry, exit_ = P[s + lag], P[s + lag + h]
        exit_ = np.where(np.isfinite(exit_), exit_, last_price)          # terminal exit at the last price
        label[s] = np.where(member[s] & np.isfinite(entry), exit_ / entry - 1, np.nan)
    score = np.where(member, rng.normal(size=(T, N)), np.nan)
    score[46:48, 1] = np.nan                              # not scored once its entry would be after delisting
    vol = np.where(member, rng.uniform(0.01, 0.03, (T, N)), np.nan)
    beta = np.where(member, rng.normal(1, 0.3, (T, N)), np.nan)
    adv = np.where(member, rng.uniform(5e7, 5e8, (T, N)), np.nan)
    return MarketData(dates, pd.Index([f"T{i}" for i in range(N)]), score, ret, label, vol, beta, adv), P


class Fixed:
    """Builder that returns fixed weights (0 for unscored names) and records the current weights."""

    def __init__(self, w):
        self.w, self.seen = np.asarray(w, dtype=float), []

    def __call__(self, data, s, current, capital_usd, costs):
        self.seen.append((s, current.copy()))
        return Target(np.where(np.isfinite(data.score[s]), self.w, 0.0))  # unscored names are sold


# --------------------------------------------------------------------------- accounting


def test_net_pnl_is_gross_minus_costs_minus_borrow_and_tranches_sum_to_the_total():
    data, _ = synthetic()
    res = run_backtest(data, decile_builder(5, 10, 2.0), COSTS, H, LAG, H, HOLDOUT)
    t = res.tranches
    np.testing.assert_allclose(t["net_pnl"], t["gross_pnl"] - t["commission"] - t["spread"] - t["impact"] - t["borrow"],
                               atol=1e-15)
    summed = t.groupby("date")[list(DAILY_COLUMNS)].sum(min_count=1)
    for c in DAILY_COLUMNS:
        np.testing.assert_allclose(summed[c].to_numpy(), res.total[c].to_numpy(), atol=1e-12, err_msg=c)
    assert (res.total["n_rebalanced"].iloc[1:] <= 1).all()          # one tranche trades per day
    assert (t["cost"] > 0).any() and (t["borrow"] > 0).any()
    stats = performance(res.total, res.first_full_date, None, H, 252)
    assert stats["n_days"] == len(res.total.loc[res.first_full_date:])


def test_gross_pnl_is_the_drifted_weights_times_returns_and_costs_follow_the_trade():
    data, _ = synthetic(T=30, N=3, seed=4)
    data.score[:] = 1.0
    data.label[:] = np.nan                                  # no residuals: P&L is purely sum w * r
    data.ret[np.isnan(data.ret)] = 0.0
    build = Fixed([1.0, -1.0, 0.0])
    res = run_backtest(data, build, COSTS, H, LAG, 1, HOLDOUT)
    hold, expected = np.zeros(3), []
    for p in range(len(data.dates)):
        r = data.ret[p]
        expected.append(hold @ r)
        hold = hold * (1 + r)
        if p >= LAG and (p - LAG) % H == 0:
            hold = np.array([1.0, -1.0, 0.0])
    np.testing.assert_allclose(res.total["gross_pnl"], expected, atol=1e-14)
    # cost of each trade: from the drifted weights to the target
    for (s, current), (_, row) in zip(build.seen, res.weights.groupby("trade_date")):
        p = s + LAG
        day = res.total.iloc[p]
        dw = np.array([1.0, -1.0, 0.0]) - current
        traded = np.abs(dw) * COSTS.aum_usd
        sig, adv = data.vol[s], data.adv[s]
        cost = (traded * (2.5e-4 + sig * np.sqrt(traded / adv))).sum() / COSTS.aum_usd
        assert day["cost"] == pytest.approx(cost, rel=1e-10)
        np.testing.assert_allclose(row.set_index("ticker").reindex(data.tickers)["trade"].fillna(0), dw, atol=1e-15)
    short = 1.0 * np.cumprod(1 + data.ret[2:, 1])           # the short leg drifts with its own returns
    assert res.total["borrow"].iloc[2] == pytest.approx(1.0 * 25e-4 / 252)
    assert res.total["borrow"].iloc[3] == pytest.approx(short[0] * 25e-4 / 252)


def test_weights_drift_exactly_with_relative_returns_without_a_rebalance():
    data, _ = synthetic(T=30, N=3, seed=5)
    data.ret[np.isnan(data.ret)] = 0.0
    data.label[:] = np.nan
    data.score[:] = np.nan
    data.score[0] = 1.0
    build = Fixed([0.6, -0.4, -0.2])
    res = run_backtest(data, build, ZERO, horizon=20, lag=LAG, n_tranches=1, holdout_start=HOLDOUT)
    # entered at the close of day 1, next trade at the close of day 21 (score day 20)
    growth = np.cumprod(1 + data.ret[2:22], axis=0)
    drifted = np.array([0.6, -0.4, -0.2]) * growth[-1]
    s, current = build.seen[1]
    assert s == 20
    np.testing.assert_allclose(current, drifted, rtol=1e-12)
    path = np.array([0.6, -0.4, -0.2]) * growth[:-1]          # end of days 2..20
    np.testing.assert_allclose(res.total["net_exposure"].iloc[2:21], path.sum(axis=1), rtol=1e-12)
    np.testing.assert_allclose(res.total["gross_exposure"].iloc[2:21], np.abs(path).sum(axis=1), rtol=1e-12)


def test_terminal_exit_closes_at_the_last_price_and_is_charged_once():
    data, P = synthetic(T=70, N=3, seed=6)
    # long name 1 (last price on day 47) against name 2 with zero returns
    data.ret[:, 2] = np.where(np.isfinite(data.ret[:, 2]), 0.0, np.nan)
    data.label[:, 2] = 0.0
    data.score[:, 2] = 1.0
    res = run_backtest(data, Fixed([0.0, 1.0, -1.0]), COSTS, H, LAG, 1, HOLDOUT)
    # entered at the close of day 46 (score day 45); the stock stops trading after day 47, so the
    # window to day 51 earns the label to its last price
    assert data.label[45, 1] == pytest.approx(P[47, 1] / P[46, 1] - 1)
    window = res.total["gross_pnl"].iloc[47:52].sum()
    assert window == pytest.approx(P[47, 1] / P[46, 1] - 1, rel=1e-10)
    t1 = res.weights[res.weights["ticker"] == "T1"]
    sold = t1[t1["weight"] == 0]
    assert len(sold) == 1 and sold["trade_date"].iloc[0] == data.dates[51] and sold["cost_usd"].iloc[0] > 0
    assert sold["trade"].iloc[0] == pytest.approx(-(1 + data.label[45, 1]), rel=1e-10)   # the whole drifted position
    assert t1[t1["trade_date"] > data.dates[51]].empty


def test_a_name_that_leaves_the_index_still_earns_its_label():
    data, P = synthetic(T=70, N=3, seed=7)
    data.ret[:, 2] = np.where(np.isfinite(data.ret[:, 2]), 0.0, np.nan)
    data.label[:, 2] = 0.0
    data.score[:] = 1.0
    res = run_backtest(data, Fixed([1.0, 0.0, -1.0]), ZERO, H, LAG, 1, HOLDOUT)
    # name 0 has no rows after day 40; the window from the close of day 36 to day 41 books its label
    assert res.total["gross_pnl"].iloc[37:42].sum() == pytest.approx(P[41, 0] / P[36, 0] - 1, rel=1e-10)


# --------------------------------------------------------------------------- reconciliation with the research spread


def test_decile_ls_with_zero_costs_reproduces_the_research_d10_minus_d1_spread():
    data, _ = synthetic(T=120, N=60, seed=8)
    q, min_names = 5, 10
    res = run_backtest(data, decile_builder(q, min_names, 2.0), ZERO, H, LAG, 1, HOLDOUT)
    score = pd.DataFrame(data.score, index=data.dates)
    label = pd.DataFrame(data.label, index=data.dates)
    deciles = quantile_returns(quantile_buckets(score, q, min_names), label, q)
    spread = deciles[q] - deciles[1]
    checked = 0
    for s in range(0, len(data.dates) - LAG - H, H):
        p = s + LAG
        pnl = res.total["gross_pnl"].iloc[p + 1: p + H + 1].sum()
        assert pnl == pytest.approx(spread.iloc[s], abs=1e-12)
        checked += 1
    assert checked >= 20 and res.total[["cost", "borrow"]].to_numpy().sum() == 0


# --------------------------------------------------------------------------- no lookahead


def perturbed_after(data: MarketData, t: int, seed: int) -> MarketData:
    rng = np.random.default_rng(seed)
    out = copy.deepcopy(data)
    for name in ("score", "ret", "vol", "beta", "adv"):
        a = getattr(out, name)
        a[t + 1:] = np.where(np.isfinite(a[t + 1:]), a[t + 1:] * rng.uniform(-3, 3, a[t + 1:].shape), np.nan)
        a[t + 1:, :3] = np.nan                     # also remove some inputs outright
    ends = np.arange(len(out.dates)) + LAG + H
    out.label[ends > t] = rng.normal(0, 1, out.label[ends > t].shape)   # labels that end after t
    return out


def small_pcfg():
    pcfg = copy.deepcopy(PCFG)
    pcfg["risk_model"]["window"] = 20
    return pcfg


BUILDERS = {
    "decile_ls": lambda: decile_builder(5, 10, 2.0),
    "neutral_volscaled": lambda: volscaled_builder(small_pcfg(), 10, 252),
    "optimised": lambda: optimised_builder(small_pcfg(), OptimiserSpec.from_config(PCFG, H, 10.0, 1.0),
                                           lambda s: 0.02, 10),
}


@pytest.mark.parametrize("construction", list(BUILDERS))
def test_weights_and_pnl_up_to_t_ignore_every_input_after_t(construction):
    T, t = (60, 45) if construction != "optimised" else (45, 38)
    data, _ = synthetic(T=T, N=40 if construction != "optimised" else 15, seed=9)
    a = run_backtest(data, BUILDERS[construction](), COSTS, H, LAG, H, HOLDOUT)
    b = run_backtest(perturbed_after(data, t, 1), BUILDERS[construction](), COSTS, H, LAG, H, HOLDOUT)
    cols = list(DAILY_COLUMNS) + ["market_return"]
    pd.testing.assert_frame_equal(a.total.iloc[: t + 1][cols], b.total.iloc[: t + 1][cols])
    wa = a.weights[a.weights["trade_date"] <= data.dates[t]].reset_index(drop=True)
    wb = b.weights[b.weights["trade_date"] <= data.dates[t]].reset_index(drop=True)
    pd.testing.assert_frame_equal(wa, wb)
    assert len(wa) > 0
    assert not a.total.iloc[t + 1:]["gross_pnl"].equals(b.total.iloc[t + 1:]["gross_pnl"])


def test_optimised_backtest_weights_satisfy_the_constraints():
    data, _ = synthetic(T=45, N=15, seed=10)
    spec = OptimiserSpec.from_config(PCFG, H, 10.0, 1.0)
    seen = []
    inner = optimised_builder(small_pcfg(), spec, lambda s: 0.02, 10)

    def build(data, s, current, capital_usd, costs):
        target = inner(data, s, current, capital_usd, costs)
        seen.append((s, current, target))
        return target

    res = run_backtest(data, build, COSTS, H, LAG, H, HOLDOUT)
    assert seen and sum(res.fallbacks.values()) == sum(t.fallback is not None for _, _, t in seen)
    for s, current, target in seen:
        w = target.weights
        assert abs(w.sum()) <= 1e-6 and np.abs(w).sum() <= 2 + 1e-6 and np.abs(w).max() <= 0.02 + 1e-6
        assert abs(np.nan_to_num(data.beta[s]) @ w) <= 0.05 + 1e-6
        free = w != 0
        cap = 0.05 * data.adv[s, free] / (COSTS.aum_usd / H)
        assert np.all(np.abs(w - current)[free] <= cap + 1e-6)


# --------------------------------------------------------------------------- holdout guard


def test_engine_refuses_a_holdout_date():
    data, _ = synthetic(T=40, N=20, start="2021-11-15")
    assert data.dates.max() >= pd.Timestamp("2022-01-01")
    with pytest.raises(ValueError, match="holdout"):
        run_backtest(data, decile_builder(5, 10, 2.0), COSTS, H, LAG, H, "2022-01-01")
    run_backtest(data, decile_builder(5, 10, 2.0), COSTS, H, LAG, H, "2022-03-01")


def test_marketdata_rejects_misaligned_arrays():
    data, _ = synthetic(T=20, N=5)
    with pytest.raises(ValueError):
        MarketData(data.dates, data.tickers, data.score[:-1], data.ret, data.label, data.vol, data.beta, data.adv)


# --------------------------------------------------------------------------- runner on a synthetic panel

from src.models import load_models_config  # noqa: E402
from src.signals import load_research_config  # noqa: E402

RCFG = load_research_config()
MCFG = load_models_config(rcfg=RCFG)
SIGNALS = ["lightgbm_h5_rank", "B0_reversal_h5", "B1_composite_h5", "lightgbm_h21_rank"]


def write_runner_inputs(root, poison_after=None, seed=12):
    """Panel, predictions, IC table and configs for a 25-name universe, 2004 to mid-2008.

    With ``poison_after`` every panel value and prediction dated after it, and every label whose
    window ends after it, is replaced by garbage."""
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2004-01-01", "2008-06-30")
    tickers = [f"T{i}" for i in range(25)]
    T, N = len(dates), len(tickers)
    P = 50 * np.cumprod(1 + rng.normal(0.0002, 0.015, (T, N)) + rng.normal(0, 0.008, (T, 1)), axis=0)
    prev = np.vstack([np.full((1, N), np.nan), P[:-1]])
    wide = {"ret_1d": P / prev - 1,
            "vol_63d": rng.uniform(0.01, 0.03, (T, N)),
            "beta_252d": rng.normal(1, 0.3, (T, N)),
            "dollar_volume_21d": rng.uniform(5e7, 5e8, (T, N))}
    for h in (5, 21):
        lab = np.full((T, N), np.nan)
        lab[: T - 1 - h] = P[1 + h:] / P[1: T - h] - 1
        wide[f"fwd_ret_{h}d"] = lab
    for c in MCFG["features"]["columns"]:
        wide[c] = rng.normal(size=(T, N))
    wide["z_ret_5d"] = -(wide["fwd_ret_5d"] * 20 + rng.normal(size=(T, N)))   # a planted reversal signal
    wide["z_ret_5d"] = np.nan_to_num(wide["z_ret_5d"])
    junk = np.random.default_rng(seed + 1)          # separate stream: clean values stay identical
    if poison_after is not None:
        late = dates > pd.Timestamp(poison_after)
        cut = int(np.flatnonzero(~late)[-1])
        for name, a in wide.items():
            a[late] = junk.normal(0, 1e3, a[late].shape)
            if name.startswith("fwd_ret_"):
                h = int(name.split("_")[2].rstrip("d"))
                ends = np.arange(T) + 1 + h
                a[ends > cut] = junk.normal(0, 1e3, a[ends > cut].shape)
    index = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    panel = pd.DataFrame({k: v.reshape(-1).astype("float32") for k, v in wide.items()}, index=index)
    panel.to_parquet(root / "panel.parquet")

    oos = panel.index.get_level_values("date") >= pd.Timestamp("2007-01-01")
    base = -panel.loc[oos, "z_ret_5d"].to_numpy()
    preds = pd.DataFrame({c: (base + rng.normal(0, 1, oos.sum())).astype("float32") for c in SIGNALS},
                         index=panel.index[oos])
    if poison_after is not None:
        late = preds.index.get_level_values("date") > pd.Timestamp(poison_after)
        preds.loc[late, SIGNALS] = junk.normal(0, 1e3, (late.sum(), len(SIGNALS))).astype("float32")
    preds["fold_id"] = (preds.index.get_level_values("date").year - 2007).astype("int16")
    preds.to_parquet(root / "oof.parquet")
    pd.DataFrame([{"model": "lightgbm", "horizon": h, "variant": "rank", "fold_id": f, "validation_ic": 0.02}
                  for h in (5, 21) for f in (0, 1)]).to_csv(root / "hyper.csv", index=False)

    rcfg = copy.deepcopy(RCFG)
    rcfg.update(holdout_start="2008-03-01", min_names_per_date=10, quantiles=5)
    (root / "research.yaml").write_text(yaml.safe_dump(rcfg))
    pcfg = copy.deepcopy(PCFG)
    pcfg["sample"]["split_date"] = "2007-04-02"
    pcfg["risk_model"]["window"] = 60
    pcfg["constructions"]["optimised"]["lambda_risk"] = [10.0]
    pcfg["constructions"]["optimised"]["lambda_tc"] = [1.0, 2.0]
    pcfg["costs"]["scenarios"] = {"cost_multipliers": [0.0, 1.0], "aum_usd": [100000000]}
    (root / "portfolio.yaml").write_text(yaml.safe_dump(pcfg, sort_keys=False))
    return root


def runner_args(root, *extra):
    return ["--config", str(root / "portfolio.yaml"), "--research-config", str(root / "research.yaml"),
            "--panel", str(root / "panel.parquet"), "--predictions", str(root / "oof.parquet"),
            "--ic-table", str(root / "hyper.csv"), "--out", str(root / "reports"),
            "--data-out", str(root / "processed"), "--allow-dirty-config", *extra]


def test_tuning_never_reads_a_return_after_the_tuning_window(tmp_path):
    from src.run_portfolio import run

    tuning_end = pd.bdate_range("2007-01-01", "2007-04-01")[-1]       # last trading day before the split
    clean = write_runner_inputs(tmp_path / "clean")
    poisoned = write_runner_inputs(tmp_path / "poisoned", poison_after=tuning_end)
    a = run(runner_args(clean, "--tune", "--signals", "lightgbm_h5_rank,B0_reversal_h5"))
    b = run(runner_args(poisoned, "--tune", "--signals", "lightgbm_h5_rank,B0_reversal_h5"))
    pd.testing.assert_frame_equal(a["lambda_grid"], b["lambda_grid"])
    grid = a["lambda_grid"]
    assert len(grid) == 4 and set(grid["signal"]) == {"lightgbm_h5_rank", "B0_reversal_h5"}
    assert (pd.to_datetime(grid["end"]) <= tuning_end).all() and grid["n_days"].min() > 30
    assert len(a["selected_hyperparameters"]) == 2


def test_evaluate_runs_with_frozen_lambdas_and_writes_the_declared_outputs(tmp_path):
    import json

    from src.run_portfolio import run

    root = write_runner_inputs(tmp_path / "eval")
    with pytest.raises(SystemExit, match="--tune"):
        run(runner_args(root, "--evaluate", "--signals", "B0_reversal_h5"))
    run(runner_args(root, "--tune", "--signals", "B0_reversal_h5"))
    tables = run(runner_args(root, "--evaluate", "--signals", "B0_reversal_h5"))
    summary = tables["summary"]
    assert set(summary["construction"]) == {"decile_ls", "neutral_volscaled", "optimised"}
    assert set(summary["scenario"]) == {"base", "cost_x0"} and set(summary["window"]) == {"tuning", "evaluation", "full"}
    base = summary.query("scenario == 'base' and window == 'evaluation'")
    assert (pd.to_datetime(base["start"]) >= pd.Timestamp("2007-04-02")).all()
    free = summary.query("scenario == 'cost_x0' and window == 'full'").set_index("construction")
    paid = summary.query("scenario == 'base' and window == 'full'").set_index("construction")
    assert (free.loc[["decile_ls", "neutral_volscaled"], "net_return_ann"]
            > paid.loc[["decile_ls", "neutral_volscaled"], "net_return_ann"]).all()
    meta = json.loads((root / "reports" / "run_metadata.json").read_text())
    assert meta["mode"] == "evaluate" and pd.Timestamp(meta["last_engine_date"]) < pd.Timestamp("2008-03-01")
    assert meta["selected_lambdas"]["B0_reversal_h5"][0] == 10.0
    daily = pd.read_parquet(root / "processed" / "portfolio_daily_returns.parquet")
    assert pd.to_datetime(daily["date"]).max() < pd.Timestamp("2008-03-01")
    assert (root / "processed" / "portfolio_weights.parquet").exists()
    assert (root / "reports" / "figures" / "cumulative_net_B0_reversal_h5.png").exists()

    other = yaml.safe_load(open(root / "portfolio.yaml"))
    other["costs"]["commission_bps"] = 1.0                 # the frozen tuning no longer matches the design
    (root / "portfolio.yaml").write_text(yaml.safe_dump(other, sort_keys=False))
    with pytest.raises(SystemExit, match="another portfolio config"):
        run(runner_args(root, "--evaluate", "--signals", "B0_reversal_h5"))


def test_runner_raises_if_a_holdout_date_reaches_the_engine(tmp_path, monkeypatch):
    from src import run_portfolio

    root = write_runner_inputs(tmp_path / "holdout")
    leaky = pd.DatetimeIndex(pd.bdate_range("2008-02-01", "2008-03-10"))
    monkeypatch.setattr(run_portfolio, "calendar_extension", lambda *a, **k: leaky)
    with pytest.raises(AssertionError, match="holdout"):
        run_portfolio.run(runner_args(root, "--evaluate", "--signals", "B0_reversal_h5", "--constructions", "decile_ls"))
