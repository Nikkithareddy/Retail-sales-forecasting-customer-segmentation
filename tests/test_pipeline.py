import numpy as np
import pandas as pd

from src import data, eda, forecast, segment


def raw_rows():
    # invoice, stock, qty, date, price, customer
    rows = [
        ("1001", "85123A", 6, "2011-03-01 10:00", 2.5, 1.0),
        ("1002", "85123A", 100, "2011-03-02 10:00", 2.5, 2.0),      # later cancelled in full
        ("C1003", "85123A", -100, "2011-03-02 10:05", 2.5, 2.0),
        ("1004", "POST", 1, "2011-03-03 10:00", 18.0, 1.0),          # postage, not a product
        ("1005", "22423", 3, "2011-03-04 11:00", 0.0, 3.0),          # zero price
        ("1006", "22423", 4, "2011-03-05 12:00", 4.0, np.nan),       # no customer id, still demand
    ]
    return pd.DataFrame(rows, columns=["InvoiceNo", "StockCode", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID"])\
        .assign(InvoiceDate=lambda d: pd.to_datetime(d["InvoiceDate"]), Description="x", Country="UK")


def test_clean_removes_cancellations_non_products_and_reversed_orders():
    sales, cancels = data.clean(raw_rows())
    assert set(sales["InvoiceNo"]) == {"1001", "1006"}      # 1002 reversed, 1004 postage, 1005 free
    assert len(cancels) == 1
    assert (sales["Revenue"] == sales["Quantity"] * sales["UnitPrice"]).all()


def test_rfm_values():
    sales, cancels = data.clean(raw_rows())
    rfm = segment.build_rfm(sales, cancels)
    assert list(rfm.index) == [1.0]                          # only identified, still-valid customers
    assert rfm.loc[1.0, "frequency"] == 1 and rfm.loc[1.0, "monetary"] == 15.0


def test_find_anomalies_flags_a_spike():
    idx = pd.date_range("2011-01-03", periods=60, freq="D")
    base = pd.Series(1000.0 + np.random.default_rng(0).normal(0, 20, 60), index=idx)
    base.iloc[30] = 20000.0
    flagged = eda.find_anomalies(base)
    assert idx[30] in flagged.index


def test_walk_forward_forecasts_have_one_value_per_holdout_week():
    idx = pd.date_range("2011-01-10", periods=40, freq="W-MON")
    y = pd.Series(100 + np.arange(40) * 2.0 + np.random.default_rng(1).normal(0, 3, 40), index=idx)
    res = forecast.run_forecast(y, n_test=6)
    assert len(res["actual"]) == 6
    assert all(len(p) == 6 for p in res["predictions"].values())
    assert set(res["metrics"]["model"]) == {"ARIMA", "Regression (lags + trend)", "Naive (last week)"}
    assert res["metrics"]["RMSE"].min() < 50


def test_kmeans_segments_and_labels():
    rng = np.random.default_rng(2)
    rfm = pd.DataFrame({"recency": rng.integers(1, 300, 300), "frequency": rng.integers(1, 20, 300),
                        "monetary": rng.uniform(50, 5000, 300)})
    rfm["avg_items_per_order"] = rng.uniform(5, 200, 300); rfm["cancel_rate"] = rng.uniform(0, 0.3, 300)
    Z = segment.scale_rfm(rfm)
    sil = segment.silhouette_by_k(Z, range(2, 6))
    k = segment.choose_k(sil)
    seg = segment.fit_segments(rfm, Z, k)
    names = segment.label_clusters(seg)
    assert seg["cluster"].nunique() == k and len(set(names.values())) == k
    omni, pair = segment.test_segments(seg)
    assert len(omni) == 2 and (pair["p_adjusted"].between(0, 1)).all()
