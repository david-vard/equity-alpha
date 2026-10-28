"""Shared helpers: config loading, run provenance, cross-sectional preprocessing, panel reshaping."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "features.yaml"


def _require_positive_ints(name: str, values: Any) -> list[int]:
    if isinstance(values, int) and not isinstance(values, bool):
        values = [values]
    if (
        not isinstance(values, list)
        or not values
        or not all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in values)
    ):
        raise ValueError(f"config '{name}' must be a positive integer or non-empty list of them")
    return values


def validate_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Check structure and value ranges; return the config unchanged."""
    for section in ("data", "universe", "features", "preprocessing", "labels", "output"):
        if section not in cfg:
            raise ValueError(f"config is missing the '{section}' section")

    start, end = pd.Timestamp(cfg["data"]["start"]), pd.Timestamp(cfg["data"]["end"])
    if start >= end:
        raise ValueError("config data.start must be before data.end")
    if cfg["data"]["max_gap_days"] < 0:
        raise ValueError("config data.max_gap_days must be >= 0")

    features = cfg["features"]
    if not 0 < features["min_obs_frac"] <= 1:
        raise ValueError("config features.min_obs_frac must be in (0, 1]")
    for key in (
        "reversal_windows",
        "momentum_windows",
        "volatility_windows",
        "beta_window",
        "dollar_volume_window",
        "high_window",
        "trend_windows",
    ):
        _require_positive_ints(f"features.{key}", features[key])
    lookback, skip = features["momentum_12_1"]["lookback"], features["momentum_12_1"]["skip"]
    if not 0 < skip < lookback:
        raise ValueError("config features.momentum_12_1 needs 0 < skip < lookback")
    _require_positive_ints("universe.liquidity.window", cfg["universe"]["liquidity"]["window"])
    min_dollar_volume = cfg["universe"]["liquidity"]["min_avg_dollar_volume"]
    if not isinstance(min_dollar_volume, (int, float)) or min_dollar_volume < 0:
        raise ValueError("config universe.liquidity.min_avg_dollar_volume must be a number >= 0")

    low, high = cfg["preprocessing"]["winsor_limits"]
    if not 0 <= low < high <= 1:
        raise ValueError("config preprocessing.winsor_limits must satisfy 0 <= low < high <= 1")

    _require_positive_ints("labels.horizons", cfg["labels"]["horizons"])
    if cfg["labels"]["lag"] < 0:
        raise ValueError("config labels.lag must be >= 0")
    delisting = cfg["labels"].get("delisting")
    if delisting is not None and delisting.get("enabled"):
        if not isinstance(delisting.get("grace_days"), int) or delisting["grace_days"] < 0:
            raise ValueError("config labels.delisting.grace_days must be an integer >= 0")
        haircut = delisting.get("haircut", {})
        if "default" not in haircut or not all(
            isinstance(v, (int, float)) and -1 <= v <= 1 for v in haircut.values()
        ):
            raise ValueError("config labels.delisting.haircut needs a 'default' and values in [-1, 1]")
    return cfg


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate a YAML config (defaults to configs/features.yaml)."""
    with open(path or DEFAULT_CONFIG_PATH) as handle:
        return validate_config(yaml.safe_load(handle))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str, repo_root: Path = REPO_ROOT) -> str:
    """Stdout of a git command run in ``repo_root``; empty if git fails."""
    done = subprocess.run(["git", *args], cwd=repo_root, capture_output=True, text=True)
    return done.stdout.strip() if done.returncode == 0 else ""


def provenance(
    config: Path, features_config: Path | None, panel_file: Path, allow_dirty: bool,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Identify a run: config commit and hash, code revision, feature-config and panel hashes.

    Refuses to run unless the design config is committed and unmodified (the design is fixed
    before results); the state of the rest of the code is recorded, not enforced. ``repo_root``
    is only overridden by tests.
    """
    resolved = config.resolve()
    rel = str(resolved.relative_to(repo_root)) if resolved.is_relative_to(repo_root) else str(config)
    sha = git("log", "-1", "--format=%H", "--", rel, repo_root=repo_root)
    dirty = git("status", "--porcelain", "--", rel, repo_root=repo_root)
    if (not sha or dirty) and not allow_dirty:
        raise SystemExit(
            f"{rel} must be committed and unmodified before the analysis runs; "
            "commit it or pass --allow-dirty-config."
        )
    features_path = features_config or DEFAULT_CONFIG_PATH
    return {
        "config_commit": sha or None,
        "config_dirty": bool(dirty) or not sha,
        "config_sha256": file_sha256(config),
        "code_commit": git("rev-parse", "HEAD", repo_root=repo_root) or None,
        "code_dirty": bool(git("status", "--porcelain", "--", "src", "configs", repo_root=repo_root)),
        "features_config_sha256": file_sha256(features_path),
        "panel_sha256": file_sha256(panel_file),
        "panel_bytes": panel_file.stat().st_size,
    }


def warmup_days(cfg: dict[str, Any]) -> int:
    """Trading days of history the longest-lookback feature needs before it is defined."""
    features = cfg["features"]
    windows = [
        *features["reversal_windows"],
        *features["momentum_windows"],
        features["momentum_12_1"]["lookback"],
        *features["volatility_windows"],
        features["beta_window"],
        features["dollar_volume_window"],
        features["high_window"],
        *features["trend_windows"],
        cfg["universe"]["liquidity"]["window"],
    ]
    return max(windows)


def winsorize_by_date(frame: pd.DataFrame, limits: tuple[float, float]) -> pd.DataFrame:
    """Clip each row (date) of a date x ticker frame at its own quantiles."""
    lower = frame.quantile(limits[0], axis=1)
    upper = frame.quantile(limits[1], axis=1)
    return frame.clip(lower=lower, upper=upper, axis=0)


def zscore_by_date(frame: pd.DataFrame) -> pd.DataFrame:
    """Standardise each row (date) to mean 0 and sample std 1; constant rows become NaN."""
    mean = frame.mean(axis=1)
    std = frame.std(axis=1)
    return frame.sub(mean, axis=0).div(std.where(std > 0), axis=0)


def membership_to_mask(
    membership: pd.DataFrame, index: pd.DatetimeIndex, columns: pd.Index
) -> pd.DataFrame:
    """Boolean date x ticker frame from long ``date, ticker`` membership rows.

    Membership is on calendar days; dates that are not in ``index`` (weekends,
    holidays) and tickers that are not in ``columns`` are ignored.
    """
    mask = np.zeros((len(index), len(columns)), dtype=bool)
    rows = index.get_indexer(membership["date"])
    cols = columns.get_indexer(membership["ticker"])
    keep = (rows >= 0) & (cols >= 0)
    mask[rows[keep], cols[keep]] = True
    return pd.DataFrame(mask, index=index, columns=columns)


