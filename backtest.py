"""
backtest.py
-----------
Standard cross-sectional factor evaluation:

1. Information Coefficient (IC): each week, rank-correlate that week's
   signal against next week's realized return across the universe.
   IC is computed cross-sectionally (across tickers at a fixed date), not
   across time -- this is the right axis for a factor meant to pick
   winners/losers within a universe at each point in time.

2. Quantile long/short portfolio: each week, go long the top signal
   quantile and short the bottom quantile, equal-weighted, held one week.

3. Performance stats: annualized return/vol/Sharpe, max drawdown, hit rate.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def compute_ic_series(signal: pd.DataFrame, fwd_returns: pd.DataFrame) -> pd.Series:
    ic = {}
    for date in signal.index:
        if date not in fwd_returns.index:
            continue
        s = signal.loc[date].dropna()
        r = fwd_returns.loc[date].dropna()
        common = s.index.intersection(r.index)
        if len(common) < 3:
            continue
        corr, _ = spearmanr(s[common], r[common])
        if not np.isnan(corr):
            ic[date] = corr
    return pd.Series(ic).sort_index()


def quantile_long_short(signal: pd.DataFrame, fwd_returns: pd.DataFrame,
                         n_quantiles: int = 5) -> pd.Series:
    """
    Equal-weighted long top-bucket / short bottom-bucket portfolio.

    Uses a fixed top-k / bottom-k selection by rank rather than pd.qcut --
    with a small universe, qcut's bin-edge logic frequently fails on tied
    ranks, silently dropping periods. Rank-based top-k/bottom-k is
    equivalent for evenly-sized buckets and degrades gracefully for small
    universes (always keeps at least 1 name per side).
    """
    port_rets = {}
    for date in signal.index:
        if date not in fwd_returns.index:
            continue
        s = signal.loc[date].dropna()
        r = fwd_returns.loc[date].dropna()
        common = s.index.intersection(r.index)
        if len(common) < max(4, n_quantiles):
            continue
        s, r = s[common], r[common]
        k = max(1, len(s) // n_quantiles)
        ordered = s.sort_values()
        bottom_names = ordered.index[:k]
        top_names = ordered.index[-k:]
        port_rets[date] = r[top_names].mean() - r[bottom_names].mean()
    return pd.Series(port_rets).sort_index()


def pooled_ic(signal: pd.DataFrame, fwd_returns: pd.DataFrame) -> float:
    """
    Pooled (panel-wide) Spearman IC: rank-correlate signal against forward
    return across ALL ticker-week observations at once, rather than
    averaging noisy single-week cross-sectional correlations. More stable
    when the cross-section (universe size) is small relative to the time
    series length -- the standard workaround quant researchers use for
    narrow universes.
    """
    s = signal.stack().dropna()
    r = fwd_returns.stack().dropna()
    common = s.index.intersection(r.index)
    if len(common) < 10:
        return float("nan")
    corr, _ = spearmanr(s.loc[common], r.loc[common])
    return corr


def performance_stats(returns: pd.Series, periods_per_year: int = 52) -> dict:
    returns = returns.dropna()
    mean, std = returns.mean(), returns.std()
    sharpe = (mean / std) * np.sqrt(periods_per_year) if std > 0 else np.nan
    cum = (1 + returns).cumprod()
    drawdown = (cum - cum.cummax()) / cum.cummax()
    return {
        "n_periods": len(returns),
        "mean_weekly_return": mean,
        "ann_return": (1 + mean) ** periods_per_year - 1,
        "ann_vol": std * np.sqrt(periods_per_year),
        "sharpe": sharpe,
        "max_drawdown": drawdown.min(),
        "hit_rate": (returns > 0).mean(),
    }


def ic_stats(ic_series: pd.Series, periods_per_year: int = 52) -> dict:
    ic_series = ic_series.dropna()
    mean, std = ic_series.mean(), ic_series.std()
    icir = (mean / std) * np.sqrt(periods_per_year) if std > 0 else np.nan
    return {
        "mean_ic": mean,
        "ic_std": std,
        "icir_annualized": icir,
        "pct_weeks_positive": (ic_series > 0).mean(),
    }
