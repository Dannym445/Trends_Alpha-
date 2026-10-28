"""
main.py
-------
End-to-end run of the Trends-Alpha pipeline:

    Google Trends  -->  z-score signal  -->  cross-sectional IC
                                          -->  quantile long/short backtest
                                          -->  charts + written summary

Usage:
    python main.py                # offline demo (synthetic fallback)
    python main.py --live         # attempt real pytrends + yfinance pull
"""

import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import UNIVERSE, TIMEFRAME, GEO, Z_WINDOW, N_QUANTILES, PERIODS_PER_YEAR
from data_fetch import fetch_universe
from signals import build_signal
from backtest import compute_ic_series, quantile_long_short, performance_stats, ic_stats, pooled_ic


def run(use_live: bool, outdir: str = "outputs"):
    trends_df, fwd_returns_df, is_synthetic = fetch_universe(
        UNIVERSE, timeframe=TIMEFRAME, geo=GEO, use_live=use_live
    )

    signal = build_signal(trends_df, z_window=Z_WINDOW)

    ic_series = compute_ic_series(signal, fwd_returns_df)
    ls_returns = quantile_long_short(signal, fwd_returns_df, n_quantiles=N_QUANTILES)

    perf = performance_stats(ls_returns, periods_per_year=PERIODS_PER_YEAR)
    ic_summary = ic_stats(ic_series, periods_per_year=PERIODS_PER_YEAR)
    ic_summary["pooled_ic"] = pooled_ic(signal, fwd_returns_df)

    # ---- charts ----
    cum = (1 + ls_returns).cumprod()

    fig, axes = plt.subplots(3, 1, figsize=(9, 11))

    axes[0].plot(cum.index, cum.values, color="#1f6feb", linewidth=1.8)
    axes[0].set_title("Long/Short Quantile Portfolio -- Cumulative Growth of $1")
    axes[0].set_ylabel("Growth of $1")
    axes[0].grid(alpha=0.3)

    axes[1].bar(ic_series.index, ic_series.values, width=5, color="#2ea043")
    axes[1].axhline(ic_summary["mean_ic"], color="black", linestyle="--", linewidth=1,
                     label=f"mean IC = {ic_summary['mean_ic']:.3f}")
    axes[1].set_title("Weekly Cross-Sectional Information Coefficient (Spearman)")
    axes[1].set_ylabel("IC")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(alpha=0.3)

    rolling_ic = ic_series.rolling(12, min_periods=6).mean()
    axes[2].plot(rolling_ic.index, rolling_ic.values, color="#8957e5", linewidth=1.5)
    axes[2].axhline(0, color="gray", linewidth=0.8)
    axes[2].set_title("12-Week Rolling Mean IC")
    axes[2].set_ylabel("Rolling IC")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig_path = f"{outdir}/trends_alpha_results.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    # ---- written summary ----
    lines = []
    lines.append("TRENDS-ALPHA -- RESULTS SUMMARY")
    lines.append("=" * 40)
    lines.append(f"Data source: {'SYNTHETIC (offline demo)' if is_synthetic else 'LIVE (pytrends + yfinance)'}")
    lines.append(f"Universe: {', '.join(UNIVERSE.keys())}")
    lines.append(f"Weeks of data: {len(signal)}")
    lines.append("")
    lines.append("Information Coefficient (signal vs next-week return)")
    lines.append("-" * 40)
    for k, v in ic_summary.items():
        lines.append(f"  {k:22s}: {v:.4f}" if isinstance(v, float) else f"  {k:22s}: {v}")
    lines.append("")
    lines.append(f"Long/Short Quantile Portfolio (top {N_QUANTILES}-tile minus bottom)")
    lines.append("-" * 40)
    for k, v in perf.items():
        lines.append(f"  {k:22s}: {v:.4f}" if isinstance(v, float) else f"  {k:22s}: {v}")
    summary_text = "\n".join(lines)

    with open(f"{outdir}/results_summary.txt", "w") as f:
        f.write(summary_text)

    print(summary_text)
    print(f"\nChart saved to {fig_path}")
    return signal, ic_series, ls_returns, perf, ic_summary, is_synthetic


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true",
                         help="Attempt a real pytrends + yfinance pull instead of the offline synthetic demo.")
    args = parser.parse_args()
    run(use_live=args.live)
