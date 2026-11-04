"""Download, load and clean the UCI Online Retail transactions."""
import urllib.request
from pathlib import Path

import pandas as pd

URL = ("https://raw.githubusercontent.com/databricks/Spark-The-Definitive-Guide/"
       "master/data/retail-data/all/online-retail-dataset.csv")
DATA_PATH = Path("data/online_retail.csv")


def download(path: Path = DATA_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print(f"Downloading {URL}")
        urllib.request.urlretrieve(URL, path)
    return path


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"InvoiceNo": str, "StockCode": str})
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="%m/%d/%Y %H:%M")
    return df


def clean(df: pd.DataFrame):
    """Split raw rows into valid sales and cancellations.

    Sales: positive quantity and price, real product stock codes (5 digits first), not a
    cancellation invoice. Non-product codes (POST, BANK CHARGES, AMAZONFEE, ...) are
    postage, fees and manual adjustments, not product demand.
    Cancellations: invoices starting with "C". Returned separately because they are used
    for the cancellation-rate feature, not for demand.
    """
    df = df.copy()
    df["is_cancel"] = df["InvoiceNo"].str.startswith("C")
    is_product = df["StockCode"].str.match(r"^\d{5}", na=False)

    sales = df[~df["is_cancel"] & is_product & (df["Quantity"] > 0) & (df["UnitPrice"] > 0)].copy()
    sales["Revenue"] = sales["Quantity"] * sales["UnitPrice"]
    sales = sales.drop_duplicates()

    cancels = df[df["is_cancel"] & is_product].copy()
    sales = drop_reversed(sales, cancels)
    return sales, cancels


def drop_reversed(sales: pd.DataFrame, cancels: pd.DataFrame) -> pd.DataFrame:
    """Remove sales that were later cancelled in full.

    A cancelled order leaves two rows: the original sale and a negative "C" invoice. If
    only the cancellation were dropped, the original sale would still count as demand
    (for example a 74,215-unit order placed and cancelled within minutes). Each cancellation
    is matched to an earlier sale with the same customer, product and quantity. Rows
    without a CustomerID cannot be matched and are left alone.
    """
    key = ["CustomerID", "StockCode", "Quantity"]
    c = cancels.dropna(subset=["CustomerID"]).copy()
    c["Quantity"] = -c["Quantity"]
    n_cancel = c.groupby(key).size().rename("n_cancel").reset_index()

    s = sales.sort_values("InvoiceDate").copy()
    has_id = s["CustomerID"].notna()
    s.loc[has_id, "_rank"] = s[has_id].groupby(key).cumcount()
    s = s.merge(n_cancel, on=key, how="left")
    s["n_cancel"] = s["n_cancel"].fillna(0)
    reversed_row = s["_rank"].notna() & (s["_rank"] < s["n_cancel"])
    return s[~reversed_row].drop(columns=["_rank", "n_cancel"]).sort_index()


def weekly_series(sales: pd.DataFrame, column: str = "Quantity") -> pd.Series:
    """Weekly total (weeks start on Monday) over full trading weeks only.

    The shop closes between Christmas and New Year (no sales 24 Dec to 3 Jan), and the
    data ends part-way through a week, so the series starts at the first full trading
    week of January and stops at the last complete week.
    """
    week = sales["InvoiceDate"].dt.to_period("W-SUN").dt.start_time
    s = sales.groupby(week)[column].sum()
    s.index = pd.DatetimeIndex(s.index)
    first_full = pd.Timestamp("2011-01-10")
    last_full = (sales["InvoiceDate"].max().normalize() - pd.Timedelta(days=sales["InvoiceDate"].max().weekday()) - pd.Timedelta(days=7))
    s = s.loc[first_full:last_full]
    s = s.asfreq("W-MON")
    s.name = column
    return s
