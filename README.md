# Trends-Alpha: An Alternative-Data Signal from Google Search Interest

A small, self-contained quant research project that tests whether **Google
Trends search interest** in a company's core product carries predictive
information about that company's **next-week stock return**, and turns any
signal found into a systematic cross-sectional long/short strategy.

![Results (synthetic demo data)](outputs/trends_alpha_results.png)

*Chart above is from the offline synthetic demo, not real market data. Run `python main.py --live` for real results.*

## 1. Thesis

Search volume for a brand or product ("iPhone", "Peloton", "DraftKings") is
a real-time, freely available proxy for consumer attention and demand —
often available before that demand shows up in quarterly earnings, guidance,
or sell-side estimates. If search interest surprises to the upside relative
to its own recent history, that *may* precede outperformance relative to
peers over the following week, and vice versa.

This is a classic "alternative data" alpha thesis — the same category as
credit-card panel data, satellite parking-lot counts, or app download
rankings, but free and easy to pull, which makes it a good learning project
even though any edge here is likely small and partially arbitraged away
already.

## 2. Data

| Series | Source | Frequency | Notes |
|---|---|---|---|
| Search interest | Google Trends (`pytrends`) | Weekly, 0–100 scale | One keyword per ticker, mapped in `config.py` |
| Prices | Yahoo Finance (`yfinance`) | Daily → resampled weekly | Adjusted close, converted to weekly returns |

**Offline mode:** this sandbox has no internet access, so `data_fetch.py`
automatically falls back to a synthetic panel when the live pulls fail
(`fetch_universe(..., use_live=True)` tries real data first). The synthetic
generator embeds a small, deliberately weak lead-lag relationship purely so
the pipeline has something non-trivial to show end-to-end — **it is not a
claim about real markets.** Run with `--live` (and `pip install pytrends
yfinance`) on a machine with internet access to get real results; expect the
real signal to be noisier and possibly weaker than the demo output.

## 3. Signal construction (`signals.py`)

Raw Trends levels aren't comparable across keywords — Google rescales each
query's series independently to its own 0–100 range — so the signal is each
ticker's **trailing z-score**: how far this week's search interest sits from
its own trailing 26-week mean, in standard-deviation units. This captures
"surprise" relative to a name's own normal search baseline rather than
absolute popularity.

## 4. Evaluation (`backtest.py`)

Two standard cross-sectional factor tests:

- **Information Coefficient (IC):** each week, Spearman-rank-correlate the
  signal against the *next* week's realized return across the universe.
  Reported both as a weekly time series (mean, IC-IR) and as a **pooled IC**
  across all ticker-weeks at once — the pooled version is more statistically
  stable when the universe is small relative to the sample's time depth,
  which is the case here (16 names).
- **Quantile long/short portfolio:** each week, equal-weight long the
  top-ranked names and short the bottom-ranked names (bucket size = universe
  size ÷ `N_QUANTILES`), hold one week, roll forward. Reported as annualized
  return, volatility, Sharpe, max drawdown, and hit rate.

## 5. Results (offline synthetic demo)

See `outputs/results_summary.txt` and `outputs/trends_alpha_results.png`
for the latest run. Headline numbers from the demo run:

- Pooled IC ≈ 0.05, annualized IC-IR ≈ 1.0
- Long/short Sharpe ≈ 1.2, with a real-looking drawdown period

Take these as a **methodology demonstration**, not a result — they come from
data where a small effect was deliberately embedded so the pipeline has
something to detect.

## 6. How to run

```bash
pip install -r requirements.txt

# offline demo (synthetic fallback, no internet needed)
python main.py

# real data (requires internet)
python main.py --live
```

Outputs land in `outputs/`: a 3-panel chart (equity curve, weekly IC,
rolling IC) and a text summary of all stats.

## 7. Honest limitations / what a reviewer will ask about

- **Small universe (16 names).** Per-week cross-sectional IC is noisy with
  only ~16 points per correlation — that's why the pooled IC is reported
  alongside it. A real version of this project should widen the universe
  (S&P 500 consumer names, mapped to keywords programmatically via each
  company's product line) to get statistically meaningful per-period IC.
- **Point-in-time integrity.** Google Trends data is *re-indexed* on every
  pull — historical values can shift slightly depending on the pull date and
  window. A production version needs to snapshot pulls and never look back
  at data through a later, re-normalized lens.
- **Keyword selection is a researcher degree of freedom.** Trying many
  keywords per ticker and keeping the best-performing one is a direct path
  to overfitting/data-snooping. The keyword map should be chosen
  *ex ante* on economic logic, not backtested performance.
- **Transaction costs and capacity are ignored.** Weekly turnover on a
  long/short book is real cost; small-cap "meme" names with heavy Trends
  variance also tend to have wider spreads.
- **No regime/robustness testing.** A real writeup would split into
  train/test periods, check performance pre- and post-2022, and test
  sensitivity to the z-score window and holding period.

## 8. Natural extensions

- Combine multiple alt-data sources (App Store rankings, Reddit/X mention
  volume, job postings) into a multi-factor composite signal.
- Use daily (not weekly) Trends granularity for shorter-horizon signals.
- Add a proper walk-forward IC/Sharpe significance test (block bootstrap,
  given autocorrelated weekly returns).
- Sector-neutralize the cross-sectional ranks before forming quantiles, so
  the strategy isn't secretly a sector-momentum bet.

## Project structure

```
config.py        universe, keywords, parameters
data_fetch.py     live pulls (pytrends/yfinance) + synthetic offline fallback
signals.py        z-score signal construction
backtest.py       IC computation, quantile long/short, performance stats
main.py           orchestrates the run, produces charts + summary
outputs/          results_summary.txt + trends_alpha_results.png
```
