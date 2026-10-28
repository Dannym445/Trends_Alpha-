"""
data_fetch.py
-------------
Pulls two aligned weekly panels:
  1. Google Trends search interest per keyword   (0-100 scale)
  2. Adjusted close prices per ticker -> weekly forward returns

Live pulls use `pytrends` and `yfinance`. Both require outbound internet
access. If either is unavailable (no network, rate-limited, package not
installed) this module automatically falls back to a SYNTHETIC panel so the
rest of the pipeline can still be built, tested, and demoed offline.

IMPORTANT: the synthetic fallback is clearly labeled everywhere it's used.
It embeds a small, deliberately weak lead-lag relationship (trend surprise
-> next week's return) purely so the backtest below has something non-zero
to show end-to-end. It is NOT a claim about real markets. Swap in live data
(see fetch_universe(use_live=True)) before drawing any real conclusions.
"""

import numpy as np
import pandas as pd


def fetch_google_trends(keywords, timeframe="today 5-y", geo="US"):
    """Attempt a real pytrends pull; return None on any failure."""
    try:
        from pytrends.request import TrendReq

        pytrends = TrendReq(hl="en-US", tz=360)
        # pytrends allows max 5 keywords per request -> batch if needed
        frames = []
        for i in range(0, len(keywords), 5):
            batch = keywords[i:i + 5]
            pytrends.build_payload(batch, timeframe=timeframe, geo=geo)
            df = pytrends.interest_over_time()
            if "isPartial" in df.columns:
                df = df.drop(columns=["isPartial"])
            frames.append(df)
        out = pd.concat(frames, axis=1)
        if out.empty:
            return None
        return out.resample("W").mean()
    except Exception as e:
        print(f"[data_fetch] Live Google Trends pull failed ({e!r}).")
        return None


def fetch_prices(tickers, period="5y"):
    """Attempt a real yfinance pull; return None on any failure."""
    try:
        import yfinance as yf

        df = yf.download(tickers, period=period, progress=False)["Adj Close"]
        if isinstance(df, pd.Series):
            df = df.to_frame(tickers[0])
        if df.empty:
            return None
        return df.resample("W").last()
    except Exception as e:
        print(f"[data_fetch] Live price pull failed ({e!r}).")
        return None


def _synthetic_panel(universe: dict, weeks=260, embedded_ic=0.12, seed=7):
    """
    Build a linked synthetic (trends, forward_returns) panel for offline use.

    trend_t is a bounded, autocorrelated random walk on [0, 100].
    forward_return_{t+1} for each ticker is a noisy function of that
    ticker's own standardized trend surprise, with a small enough
    coefficient that the resulting cross-sectional IC lands around
    `embedded_ic` -- roughly the magnitude you'd hope for from a real,
    usable-but-not-magic alt-data signal.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=weeks, freq="W")
    weeks = len(dates)  # pandas can return periods-1 for some end-date/freq combos
    tickers = list(universe.keys())

    trend_data, ret_data = {}, {}
    for i, (tkr, kw) in enumerate(universe.items()):
        r = np.random.default_rng(seed + i)
        level = 45 + 12 * np.sin(np.linspace(0, 5 * np.pi, weeks) + i * 0.7)
        rw = np.cumsum(r.normal(0, 1.4, weeks))
        noise = r.normal(0, 6, weeks)
        spikes = r.binomial(1, 0.025, weeks) * r.normal(28, 8, weeks)
        trend = np.clip(level + rw * 0.25 + noise + spikes, 0, 100)
        trend_data[kw] = trend

        z = (trend - pd.Series(trend).rolling(26, min_periods=8).mean().values) / \
            (pd.Series(trend).rolling(26, min_periods=8).std().values + 1e-6)
        z = np.nan_to_num(z)
        idio_vol = r.uniform(0.03, 0.06)
        fwd_ret = np.empty(weeks)
        fwd_ret[:] = np.nan
        fwd_ret[:-1] = embedded_ic * 0.4 * z[:-1] * idio_vol + r.normal(0, idio_vol, weeks - 1)
        ret_data[tkr] = fwd_ret

    trends_df = pd.DataFrame(trend_data, index=dates)
    fwd_returns_df = pd.DataFrame(ret_data, index=dates)
    return trends_df, fwd_returns_df, tickers


def fetch_universe(universe: dict, timeframe="today 5-y", geo="US", use_live=True):
    """
    Returns (trends_df, fwd_returns_df, is_synthetic).

    trends_df:      weekly search interest, columns = keywords
    fwd_returns_df: weekly forward returns aligned to the SAME dates as
                     trends_df (fwd_returns_df.loc[t] = return realized
                     from week t to week t+1 -> already shifted, no
                     look-ahead needed downstream)
    """
    keywords = list(universe.values())
    tickers = list(universe.keys())

    trends_df, prices_df = None, None
    if use_live:
        trends_df = fetch_google_trends(keywords, timeframe=timeframe, geo=geo)
        prices_df = fetch_prices(tickers)

    if trends_df is not None and prices_df is not None:
        # align keyword columns -> ticker columns
        trends_df = trends_df.rename(columns={v: k for k, v in universe.items()})
        rets = prices_df.pct_change()
        fwd_returns_df = rets.shift(-1)  # week t's forward return
        common_idx = trends_df.index.intersection(fwd_returns_df.index)
        trends_df = trends_df.loc[common_idx, tickers]
        fwd_returns_df = fwd_returns_df.loc[common_idx, tickers]
        return trends_df, fwd_returns_df, False

    print("[data_fetch] Falling back to SYNTHETIC data (offline demo mode). "
          "Set use_live=True with pytrends+yfinance installed and network "
          "access to run this on real data.")
    trends_df, fwd_returns_df, _ = _synthetic_panel(universe)
    trends_df.columns = tickers  # rename keyword cols -> ticker cols for consistency
    return trends_df, fwd_returns_df, True
