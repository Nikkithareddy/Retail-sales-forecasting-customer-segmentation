"""Run the full analysis and write tables and figures to reports/.

Usage:  python -m src.run_analysis
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import data, eda, forecast, segment

REPORTS, FIG = Path("reports"), Path("reports/figures")
sns.set_theme(style="whitegrid")


def save(fig, name):
    fig.tight_layout(); fig.savefig(FIG / name, dpi=150); plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    data.download()
    sales, cancels = data.clean(data.load_raw())

    # ---------------- EDA ----------------
    monthly = eda.monthly_summary(sales)
    monthly.round(0).to_csv(REPORTS / "monthly_summary.csv")
    daily = eda.daily_revenue(sales)
    anomalies = eda.find_anomalies(daily)
    anomalies.round(2).to_csv(REPORTS / "anomalies.csv")
    print("Monthly revenue (GBP):\n", monthly["revenue"].round(0).to_string())
    print("\nWeekday mean revenue:\n", eda.weekday_profile(sales).round(0).to_string())
    print(f"\nAnomalous trading days: {len(anomalies)}\n", anomalies.head(8).round(1).to_string())

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].bar([d.strftime("%b %y") for d in monthly.index], monthly["revenue"] / 1e3, color="#2b6cb0")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].set_title("Monthly revenue"); axes[0].set_ylabel("GBP thousands"); axes[0].set_xlabel("")
    eda.weekday_profile(sales).plot.bar(ax=axes[1], color="#2b6cb0", rot=0)
    axes[1].set_title("Mean revenue by weekday"); axes[1].set_ylabel("GBP")
    eda.hourly_profile(sales).plot.bar(ax=axes[2], color="#2b6cb0", rot=0)
    axes[2].set_title("Revenue by hour of day"); axes[2].set_xlabel("hour")
    save(fig, "eda_seasonality.png")

    fig, ax = plt.subplots(figsize=(12, 4))
    daily.plot(ax=ax, lw=0.8, color="#2b6cb0")
    ax.scatter(anomalies.index, anomalies["revenue"], color="#c53030", zorder=3, s=25, label="anomaly")
    ax.set_title("Daily revenue with anomalous days flagged"); ax.set_ylabel("GBP"); ax.legend()
    save(fig, "eda_anomalies.png")

    # ---------------- Forecasting ----------------
    y = data.weekly_series(sales, "Quantity")
    y.to_frame().to_csv(REPORTS / "weekly_units.csv")
    res = forecast.run_forecast(y)
    print(f"\nARIMA order chosen: {res['order']}")
    print(res["metrics"].round(1).to_string(index=False))
    res["metrics"].round(2).to_csv(REPORTS / "forecast_metrics.csv", index=False)
    pd.DataFrame({"actual": res["actual"], **{k: v for k, v in res["predictions"].items()}}).round(0).to_csv(
        REPORTS / "forecast_predictions.csv")

    fig, ax = plt.subplots(figsize=(11, 4.5))
    y.iloc[:-forecast.N_TEST].plot(ax=ax, color="#4a5568", label="training weeks")
    res["actual"].plot(ax=ax, color="black", lw=2.2, marker="o", label="actual (hold-out)")
    for (name, p), col in zip(res["predictions"].items(), ["#2b6cb0", "#c05621", "#718096"]):
        p.plot(ax=ax, color=col, marker="o", ls="--", label=name)
    ax.margins(x=0.03)
    ax.set_title("Weekly units sold: walk-forward forecasts on the 8-week hold-out")
    ax.set_ylabel("units"); ax.legend(fontsize=8)
    save(fig, "forecast_holdout.png")

    # ---------------- Segmentation ----------------
    rfm = segment.build_rfm(sales, cancels)
    Z = segment.scale_rfm(rfm)
    sil = segment.silhouette_by_k(Z)
    k = segment.choose_k(sil)
    seg = segment.fit_segments(rfm, Z, k)
    names = segment.label_clusters(seg)
    prof = segment.profile(seg, names)
    omni, pair = segment.test_segments(seg)
    sil.round(4).to_csv(REPORTS / "silhouette_by_k.csv", index=False)
    prof.to_csv(REPORTS / "segment_profile.csv")
    omni.to_csv(REPORTS / "hypothesis_tests_overall.csv", index=False)
    pair.to_csv(REPORTS / "hypothesis_tests_pairwise.csv", index=False)
    seg.assign(segment=seg["cluster"].map(names)).round(3).to_csv(REPORTS / "customer_segments.csv")
    print(f"\nCustomers: {len(rfm)} | silhouette by k:\n{sil.round(3).to_string(index=False)}\nChosen k = {k}")
    print("\nSegment profile:\n", prof.to_string())
    print("\nKruskal-Wallis:\n", omni.round(4).to_string(index=False))
    print("\nPairwise (adjusted p):\n", pair.round(4).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sil.plot(x="k", y="silhouette", marker="o", ax=axes[0], legend=False, color="#2b6cb0")
    axes[0].axvline(k, ls="--", color="#c53030"); axes[0].set_title("Silhouette score by number of clusters")
    axes[0].set_ylabel("silhouette")
    sil.plot(x="k", y="inertia", marker="o", ax=axes[1], legend=False, color="#2b6cb0")
    axes[1].set_title("Inertia (elbow view)")
    save(fig, "silhouette.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    plot_df = seg.assign(segment=seg["cluster"].map(names))
    order = prof["segment"].tolist()
    for ax, col, ttl in zip(axes, RFM_COLS := ["recency", "frequency", "monetary"],
                            ["Recency (days since last order)", "Frequency (orders)", "Monetary (GBP)"]):
        sns.boxplot(data=plot_df, x="segment", y=col, order=order, ax=ax, color="#90cdf4", showfliers=False)
        ax.set_title(ttl); ax.set_xlabel(""); ax.tick_params(axis="x", rotation=25)
    save(fig, "segment_profiles.png")


if __name__ == "__main__":
    main()
