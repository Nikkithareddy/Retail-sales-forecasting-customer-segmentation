"""Weekly demand forecasting: ARIMA vs a lag-feature regression baseline vs naive.

Validation is walk-forward on a hold-out period: for each held-out week the model is
fitted on all earlier weeks only and predicts that one week, so no future data leaks.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

N_TEST = 8


def choose_d(train: pd.Series) -> int:
    """Difference once if the Augmented Dickey-Fuller test cannot reject a unit root."""
    return 0 if adfuller(train.values)[1] < 0.05 else 1


def select_arima_order(train: pd.Series, max_p: int = 3, max_q: int = 2):
    """Pick (p, d, q): d from the ADF test, then p and q by lowest AIC on the training weeks."""
    d = choose_d(train)
    rows = []
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                rows.append({"p": p, "d": d, "q": q, "aic": ARIMA(train, order=(p, d, q)).fit().aic})
            except Exception:
                continue
    table = pd.DataFrame(rows).sort_values("aic").reset_index(drop=True)
    best = table.iloc[0]
    return (int(best.p), int(best.d), int(best.q)), table


def arima_walk_forward(y: pd.Series, order, n_test: int = N_TEST) -> pd.Series:
    preds = []
    for i in range(len(y) - n_test, len(y)):
        fit = ARIMA(y.iloc[:i], order=order).fit()
        preds.append(float(fit.forecast(1).iloc[0]))
    return pd.Series(preds, index=y.index[-n_test:], name="ARIMA")


def lag_features(y: pd.Series) -> pd.DataFrame:
    """Features known one week ahead: previous weeks' demand, their 4-week mean, and trend."""
    X = pd.DataFrame({"lag1": y.shift(1), "lag2": y.shift(2), "lag4": y.shift(4),
                      "mean4": y.shift(1).rolling(4).mean(),
                      "trend": np.arange(len(y))}, index=y.index)
    return X


def regression_walk_forward(y: pd.Series, n_test: int = N_TEST) -> pd.Series:
    X = lag_features(y)
    preds = []
    for i in range(len(y) - n_test, len(y)):
        train = X.iloc[:i].dropna()
        model = LinearRegression().fit(train, y.loc[train.index])
        preds.append(float(model.predict(X.iloc[[i]])[0]))
    return pd.Series(preds, index=y.index[-n_test:], name="Regression (lags + trend)")


def naive_walk_forward(y: pd.Series, n_test: int = N_TEST) -> pd.Series:
    """Predict next week = this week. Any useful model should beat this."""
    return y.shift(1).iloc[-n_test:].rename("Naive (last week)")


def score(actual: pd.Series, predictions: dict) -> pd.DataFrame:
    rows = []
    for name, p in predictions.items():
        rows.append({"model": name,
                     "MAE": mean_absolute_error(actual, p),
                     "RMSE": float(np.sqrt(mean_squared_error(actual, p))),
                     "MAPE_%": float(np.mean(np.abs((actual - p) / actual)) * 100)})
    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)


def run_forecast(y: pd.Series, n_test: int = N_TEST):
    train = y.iloc[:-n_test]
    order, aic_table = select_arima_order(train)
    actual = y.iloc[-n_test:]
    preds = {
        "ARIMA": arima_walk_forward(y, order, n_test),
        "Regression (lags + trend)": regression_walk_forward(y, n_test),
        "Naive (last week)": naive_walk_forward(y, n_test),
    }
    return {"order": order, "aic_table": aic_table, "actual": actual,
            "predictions": preds, "metrics": score(actual, preds)}
