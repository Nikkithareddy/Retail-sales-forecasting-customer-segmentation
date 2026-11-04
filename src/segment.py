"""Customer segmentation: RFM features, K-Means with silhouette-based k, hypothesis tests."""
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

SEED = 42
RFM = ["recency", "frequency", "monetary"]


def build_rfm(sales: pd.DataFrame, cancels: pd.DataFrame) -> pd.DataFrame:
    """One row per identified customer.

    recency   = days since last purchase (relative to the day after the last transaction)
    frequency = number of distinct invoices
    monetary  = total revenue
    Two extra descriptors NOT used for clustering are kept for testing the segments:
    avg_items_per_order and cancel_rate.
    """
    s = sales.dropna(subset=["CustomerID"]).copy()
    snapshot = s["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)
    g = s.groupby("CustomerID")
    rfm = pd.DataFrame({
        "recency": (snapshot - g["InvoiceDate"].max()).dt.days,
        "frequency": g["InvoiceNo"].nunique(),
        "monetary": g["Revenue"].sum(),
        "total_items": g["Quantity"].sum(),
    })
    rfm["avg_items_per_order"] = rfm["total_items"] / rfm["frequency"]
    n_cancel = cancels.dropna(subset=["CustomerID"]).groupby("CustomerID")["InvoiceNo"].nunique()
    rfm["cancelled_orders"] = n_cancel.reindex(rfm.index).fillna(0)
    rfm["cancel_rate"] = rfm["cancelled_orders"] / (rfm["frequency"] + rfm["cancelled_orders"])
    return rfm.drop(columns="total_items")


def scale_rfm(rfm: pd.DataFrame) -> np.ndarray:
    """Log-transform (RFM values are heavily right-skewed) then standardise."""
    return StandardScaler().fit_transform(np.log1p(rfm[RFM]))


def silhouette_by_k(Z: np.ndarray, k_range=range(2, 9)) -> pd.DataFrame:
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Z)
        rows.append({"k": k, "silhouette": silhouette_score(Z, km.labels_), "inertia": km.inertia_})
    return pd.DataFrame(rows)


def choose_k(sil: pd.DataFrame, min_k: int = 3):
    """Highest silhouette among k >= min_k.

    k=2 usually just splits active from lapsed customers, which is too coarse to act on,
    so the search starts at 3. The full table is still reported.
    """
    cand = sil[sil["k"] >= min_k]
    return int(cand.sort_values("silhouette", ascending=False).iloc[0]["k"])


def fit_segments(rfm: pd.DataFrame, Z: np.ndarray, k: int) -> pd.DataFrame:
    labels = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(Z)
    out = rfm.copy()
    out["cluster"] = labels
    return out


def label_clusters(seg: pd.DataFrame) -> dict:
    """Name each cluster by value tier (rank of mean spend) and recency of its customers."""
    prof = seg.groupby("cluster")[["recency", "monetary"]].mean()
    k = len(prof)
    tiers = {3: ["High-value", "Mid-value", "Low-value"],
             4: ["Top-value", "High-value", "Mid-value", "Low-value"]}.get(k)
    ranked = prof["monetary"].sort_values(ascending=False).index.tolist()
    names = {}
    for i, c in enumerate(ranked):
        tier = tiers[i] if tiers else f"Value tier {i + 1}"
        r = prof.loc[c, "recency"]
        recency = "recent" if r <= 30 else "cooling" if r <= 90 else "lapsed"
        names[c] = f"{tier}, {recency}"
    return names


def profile(seg: pd.DataFrame, names: dict) -> pd.DataFrame:
    g = seg.groupby("cluster")
    p = g.agg(customers=("recency", "size"), mean_recency_days=("recency", "mean"),
              mean_frequency=("frequency", "mean"), mean_monetary=("monetary", "mean"),
              total_revenue=("monetary", "sum"),
              median_items_per_order=("avg_items_per_order", "median"),
              mean_cancel_rate=("cancel_rate", "mean"))
    p["pct_customers"] = 100 * p["customers"] / p["customers"].sum()
    p["pct_revenue"] = 100 * p["total_revenue"] / p["total_revenue"].sum()
    p.insert(0, "segment", p.index.map(names))
    return p.round(3).sort_values("total_revenue", ascending=False)


def test_segments(seg: pd.DataFrame, variables=("avg_items_per_order", "cancel_rate")) -> tuple:
    """Kruskal-Wallis across segments, then Bonferroni-adjusted pairwise Mann-Whitney.

    Tested on variables that were NOT used to build the clusters. Testing differences in
    recency, frequency or monetary would be circular, because the clusters were chosen to
    separate exactly those.
    """
    omnibus, pairwise = [], []
    clusters = sorted(seg["cluster"].unique())
    n = len(seg)
    for var in variables:
        groups = [seg.loc[seg["cluster"] == c, var].values for c in clusters]
        H, p = stats.kruskal(*groups)
        omnibus.append({"variable": var, "H": H, "p_value": p, "epsilon_squared": H / (n - 1)})
        pairs = list(combinations(clusters, 2))
        for a, b in pairs:
            U, pp = stats.mannwhitneyu(seg.loc[seg["cluster"] == a, var], seg.loc[seg["cluster"] == b, var])
            pairwise.append({"variable": var, "segment_a": a, "segment_b": b,
                             "p_adjusted": min(1.0, pp * len(pairs))})
    return pd.DataFrame(omnibus), pd.DataFrame(pairwise)
