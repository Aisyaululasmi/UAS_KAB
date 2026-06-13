import numpy as np
import pandas as pd


def min_max(series: pd.Series) -> pd.Series:
    span = series.max() - series.min()
    if pd.isna(span) or span == 0:
        return pd.Series(0.5, index=series.index)
    return (series - series.min()) / span


def safe_mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < 1e-9, np.nan, np.abs(y_true))
    return float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100)


def directional_accuracy(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2:
        return 0.0
    return float(np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))) * 100)


def business_days_after(last_date, periods: int) -> pd.DatetimeIndex:
    start = pd.to_datetime(last_date) + pd.offsets.BDay(1)
    return pd.bdate_range(start=start, periods=periods)
