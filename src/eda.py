"""Exploratory analysis helpers: seasonality, trends and anomalies."""
import numpy as np
import pandas as pd


def monthly_summary(sales: pd.DataFrame) -> pd.DataFrame:
    m = sales.groupby(sales["InvoiceDate"].dt.to_period("M")).agg(
        revenue=("Revenue", "sum"), units=("Quantity", "sum"),
        orders=("InvoiceNo", "nunique"), customers=("CustomerID", "nunique"))
    m.index = m.index.to_timestamp()
    return m


def daily_revenue(sales: pd.DataFrame) -> pd.Series:
    return sales.groupby(sales["InvoiceDate"].dt.normalize())["Revenue"].sum()


def weekday_profile(sales: pd.DataFrame) -> pd.Series:
    daily = daily_revenue(sales)
    prof = daily.groupby(daily.index.dayofweek).mean()
    prof.index = prof.index.map({0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"})
    return prof


def hourly_profile(sales: pd.DataFrame) -> pd.Series:
    return sales.groupby(sales["InvoiceDate"].dt.hour)["Revenue"].sum()


def find_anomalies(daily: pd.Series, window: int = 15, threshold: float = 3.0) -> pd.DataFrame:
    """Flag trading days whose revenue is far from the local level.

    Works on weekday-adjusted log revenue: compare each day with the centred rolling median,
    scale the residuals by the median absolute deviation (robust to the anomalies themselves).
    """
    logr = np.log(daily)
    # Sundays are always quieter than weekdays, so remove the weekday effect first;
    # otherwise ordinary Sundays would be flagged as anomalies.
    weekday_effect = logr.groupby(logr.index.dayofweek).transform("mean") - logr.mean()
    adj = logr - weekday_effect
    local = adj.rolling(window, center=True, min_periods=window // 2).median()
    resid = adj - local
    mad = np.median(np.abs(resid.dropna() - np.median(resid.dropna())))
    z = resid / (1.4826 * mad)
    out = pd.DataFrame({"revenue": daily, "local_level": np.exp(local + weekday_effect), "robust_z": z})
    return out[out["robust_z"].abs() > threshold].sort_values("robust_z", key=np.abs, ascending=False)
