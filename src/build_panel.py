"""Build the modelling panel from cached Sharadar data.

Usage (from the repository root):
    python -m src.build_panel [--config configs/features.yaml]

Writes ``<output.dir>/<output.name>.parquet`` (gitignored) and a small text summary to
``reports/panel_summary.txt``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.data import get_delistings, get_sp500_membership, get_universe_panel
from src.features import PRICE_FIELDS, build_panel, delisting_info
from src.utils import REPO_ROOT, load_config, membership_to_mask


def load_inputs(cfg: dict[str, Any]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Dense price history for every ever-member ticker, plus the point-in-time member mask."""
    data = cfg["data"]
    prices = get_universe_panel(
        data["start"], data["end"], PRICE_FIELDS, data["refresh"], apply_membership=False
    )
    membership = get_sp500_membership(data["start"], data["end"], data["refresh"])
    reference = prices["closeadj"]
    return prices, membership_to_mask(membership, reference.index, reference.columns)


def summarize_panel(panel: pd.DataFrame, terminal: pd.DataFrame | None = None) -> str:
    """Date range, universe size per date, rows, missingness, and terminal-price labels.

    ``terminal`` is the ``delisting_info`` table; it adds a breakdown of the labels that exit a
    delisted stock at its last price by delisting reason.
    """
    dates = panel.index.get_level_values("date")
    per_date = panel.groupby(level="date").size()
    feature_cols = [c for c in panel.columns if not c.startswith(("z_", "fwd_"))]
    label_cols = [c for c in panel.columns if c.startswith("fwd_ret_") and not c.endswith("_xs")]
    missing = (panel[feature_cols + label_cols].isna().mean() * 100).round(2)
    lines = [
        f"date range      : {dates.min().date()} to {dates.max().date()} ({dates.nunique()} dates)",
        f"rows            : {len(panel):,}  (unique tickers in panel: "
        f"{panel.index.get_level_values('ticker').nunique()})",
        f"tickers per date: min {per_date.min()}, mean {per_date.mean():.1f}, max {per_date.max()}",
        f"columns         : {len(feature_cols)} raw features, {len(feature_cols)} z-scored, "
        f"{len(label_cols)} labels (+ {len(label_cols)} cross-sectional)",
        "missing % by raw feature / label:",
    ]
    lines += [f"  {name:<24}{pct:>7.2f}" for name, pct in missing.items()]
    flags = [c for c in panel.columns if c.startswith("fwd_terminal_")]
    if flags:
        lines.append("labels that exit a delisted stock at its last price (rows, % of rows):")
        for flag in flags:
            hit = panel[flag] == 1
            lines.append(f"  {flag:<24}{int(hit.sum()):>7,}{hit.mean() * 100:>8.3f}")
        if terminal is not None:
            longest = max(flags, key=lambda c: int(c.split("_")[-1].rstrip("d")))
            tickers = panel.index.get_level_values("ticker")[(panel[longest] == 1).to_numpy()]
            by_reason = terminal["reason"].reindex(tickers).value_counts()
            lines.append(f"  {longest} by delisting reason: " + ", ".join(f"{k} {v:,}" for k, v in by_reason.items()))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> pd.DataFrame:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=Path, default=None, help="YAML config path")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    prices, member_mask = load_inputs(cfg)
    delisting_cfg = cfg["labels"].get("delisting", {"enabled": False})
    delistings = get_delistings(cfg["data"]["end"], cfg["data"]["refresh"]) if delisting_cfg["enabled"] else None
    panel = build_panel(prices, member_mask, cfg, delistings)
    terminal = (
        delisting_info(prices["closeadj"], delistings, delisting_cfg) if delisting_cfg["enabled"] else None
    )

    out_dir = REPO_ROOT / cfg["output"]["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{cfg['output']['name']}.parquet"
    panel.to_parquet(out_path)

    summary = summarize_panel(panel, terminal)
    reports_dir = REPO_ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    (reports_dir / "panel_summary.txt").write_text(summary + "\n")
    print(f"saved {out_path.relative_to(REPO_ROOT)}")
    print(summary)
    return panel


if __name__ == "__main__":
    main()
