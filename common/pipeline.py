"""
Shared data / feature / evaluation pipeline for Weeks 3-5.

Reuses the conventions established in Week 1 (yfinance Adj Close, log returns,
rolling mean/vol, momentum features) and Week 2 (walk-forward evaluation,
zero / rolling-mean baselines, Ridge). Week 2's rolling features leaked the
current day into its own 20-day window (df["r_t"].rolling(20).mean() includes
r_t) and predicted same-day return r_t rather than next-day return. This
module fixes that: every feature at row t is built only from information
available at the close of day t-1, and the target is the return realized on
day t.

Universe / date range: 5-asset universe from Week 1's "Long Term" window.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Config (shared across Weeks 3-5 for reproducibility)
# ---------------------------------------------------------------------------
TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
START_DATE = "2015-01-01"
END_DATE = "2024-01-01"
SEED = 42

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
PRICE_CACHE = DATA_DIR / "prices_5asset_2015_2024.csv"


def set_seed(seed: int = SEED) -> None:
    """Fix seeds for numpy / random / torch (if installed) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_price_panel(
    tickers: list[str] = TICKERS,
    start: str = START_DATE,
    end: str = END_DATE,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Download Adj Close for `tickers` between [start, end) and return a wide
    DataFrame (index=date, columns=tickers). Cached to disk after first
    download since yfinance calls are slow / rate-limited and we want
    reproducible offline reruns.
    """
    if use_cache and PRICE_CACHE.exists():
        prices = pd.read_csv(PRICE_CACHE, index_col=0, parse_dates=True)
        if set(tickers).issubset(prices.columns):
            return prices[tickers]

    raw = yf.download(tickers, start=start, end=end, auto_adjust=False)
    prices = raw["Adj Close"] if "Adj Close" in raw.columns.get_level_values(0) else raw["Close"]
    prices = prices[tickers].ffill().dropna()

    DATA_DIR.mkdir(exist_ok=True)
    prices.to_csv(PRICE_CACHE)
    return prices


# ---------------------------------------------------------------------------
# Feature engineering (no lookahead)
# ---------------------------------------------------------------------------
FEATURE_COLS = ["r_lag1", "r_lag2", "roll_mean_20", "roll_vol_20", "mom_5"]


def build_asset_features(price: pd.Series) -> pd.DataFrame:
    """
    Build the no-lookahead feature/target table for a single asset's price
    series. Every feature at row t uses only information available at the
    close of day t-1; the target is the log return realized ON day t.

        r_t            = log(price_t / price_{t-1})           [target]
        r_lag1          = r_{t-1}                              (known at close t-1)
        r_lag2          = r_{t-2}
        roll_mean_20    = mean(r_{t-1}, ..., r_{t-20})
        roll_vol_20     = std(r_{t-1}, ..., r_{t-20})
        mom_5           = price_{t-1} / price_{t-6} - 1
    """
    log_ret = np.log(price / price.shift(1))

    df = pd.DataFrame(index=price.index)
    df["target"] = log_ret  # r_t, realized on day t -- what we predict

    lagged_ret = log_ret.shift(1)  # everything below only ever looks at day t-1 and earlier
    df["r_lag1"] = lagged_ret
    df["r_lag2"] = log_ret.shift(2)
    df["roll_mean_20"] = lagged_ret.rolling(window=20, min_periods=20).mean()
    df["roll_vol_20"] = lagged_ret.rolling(window=20, min_periods=20).std()
    df["mom_5"] = price.shift(1) / price.shift(6) - 1

    return df.dropna()


def build_panel(
    prices: pd.DataFrame, tickers: list[str] = TICKERS
) -> pd.DataFrame:
    """
    Build a long-format panel across all assets: columns
    [date, ticker, r_lag1, r_lag2, roll_mean_20, roll_vol_20, mom_5, target].
    This is the pooled cross-sectional table used to train a single model
    across all assets, and to compute cross-sectional (rank-)IC per date.
    """
    frames = []
    for tk in tickers:
        f = build_asset_features(prices[tk])
        f["ticker"] = tk
        f["date"] = f.index
        frames.append(f)
    panel = pd.concat(frames, axis=0, ignore_index=True)
    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)
    return panel[["date", "ticker"] + FEATURE_COLS + ["target"]]


# ---------------------------------------------------------------------------
# Walk-forward splitting
# ---------------------------------------------------------------------------
def walk_forward_splits(
    dates: np.ndarray,
    min_train_days: int = 500,
    test_days: int = 63,
    expanding: bool = True,
):
    """
    Yield (train_dates, test_dates) pairs walking forward through unique
    sorted `dates`, non-overlapping test segments of `test_days` trading
    days each, advancing strictly forward in time every step (no lookahead:
    a given window's test dates are always strictly after its train dates).

    expanding=True -> training window grows each step (matches Week 2's
    TimeSeriesSplit convention). expanding=False -> fixed-size rolling window
    of `min_train_days`.
    """
    unique_dates = np.array(sorted(pd.unique(dates)))
    n = len(unique_dates)

    start = min_train_days
    while start + test_days <= n:
        test_idx = slice(start, start + test_days)
        if expanding:
            train_idx = slice(0, start)
        else:
            train_idx = slice(max(0, start - min_train_days), start)
        yield unique_dates[train_idx], unique_dates[test_idx]
        start += test_days


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson correlation between predicted and realized returns (pooled)."""
    if len(y_true) < 2 or np.std(y_pred) == 0:
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def rank_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank correlation between predicted and realized returns."""
    if len(y_true) < 2 or np.std(y_pred) == 0:
        return np.nan
    corr, _ = spearmanr(y_true, y_pred)
    return float(corr)


def daily_cross_sectional_rank_ic(dates: pd.Series, y_true: np.ndarray, y_pred: np.ndarray) -> pd.Series:
    """
    Cross-sectional rank-IC per date: on each date, rank-correlate predicted
    vs realized returns ACROSS the asset universe. This is the IC that
    matters for portfolio construction (are relative rankings right on a
    given day), as opposed to the pooled time-series IC above.
    """
    df = pd.DataFrame({"date": dates, "y_true": y_true, "y_pred": y_pred})
    out = df.groupby("date").apply(
        lambda g: rank_ic(g["y_true"].values, g["y_pred"].values) if len(g) >= 3 else np.nan
    )
    return out.dropna()
