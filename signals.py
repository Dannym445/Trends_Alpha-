"""
signals.py
----------
Turns raw Google Trends levels into a cross-sectional alpha signal.

Design choice: use each ticker's own trailing z-score (level relative to
its own recent history), not the raw 0-100 level. Raw levels aren't
comparable across keywords (Trends rescales each query independently), so
comparing "Tesla" = 62 to "Peloton" = 40 directly is meaningless. Z-scoring
against each series' own trailing window fixes that and also captures the
"surprise" component (spikes above/below the recent normal) that is more
likely to be informative than the level itself.
"""

import pandas as pd


def trailing_zscore(trends: pd.DataFrame, window: int) -> pd.DataFrame:
    roll_mean = trends.rolling(window, min_periods=max(4, window // 4)).mean()
    roll_std = trends.rolling(window, min_periods=max(4, window // 4)).std()
    z = (trends - roll_mean) / roll_std.replace(0, pd.NA)
    return z


def week_over_week_change(trends: pd.DataFrame) -> pd.DataFrame:
    return trends.pct_change()


def build_signal(trends: pd.DataFrame, z_window: int = 26) -> pd.DataFrame:
    """
    Composite signal = trailing z-score of search interest.
    Returns a DataFrame aligned to `trends`' index/columns, NaNs where
    there isn't enough history yet to compute a window.
    """
    z = trailing_zscore(trends, z_window)
    return z
