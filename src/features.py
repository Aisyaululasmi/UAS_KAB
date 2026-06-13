import numpy as np
import pandas as pd

from .config import FEATURE_COLUMNS, HORIZON_DAYS, PROCESSED_DIR, TARGET_DAYS


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_diff = df["High"].diff()
    low_diff = -df["Low"].diff()
    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0.0)
    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0.0)
    atr = _atr(df, period).replace(0, np.nan)
    plus_di = 100 * plus_dm.rolling(period).mean() / atr
    minus_di = 100 * minus_dm.rolling(period).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.rolling(period).mean()


def make_features_one(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("Date").copy()
    df["return_1d"] = df["Close"].pct_change()
    df["return_5d"] = df["Close"].pct_change(5)
    df["return_10d"] = df["Close"].pct_change(10)
    df["return_20d"] = df["Close"].pct_change(20)
    df["log_return_1d"] = np.log(df["Close"] / df["Close"].shift(1))
    df["price_change"] = df["Close"] - df["Open"]
    df["high_low_spread"] = (df["High"] - df["Low"]) / df["Close"]
    df["open_close_spread"] = (df["Close"] - df["Open"]) / df["Open"]
    for win in [5, 10, 20, 50, 100]:
        df[f"sma_{win}"] = df["Close"].rolling(win).mean() / df["Close"] - 1
    df["ema_12"] = df["Close"].ewm(span=12, adjust=False).mean() / df["Close"] - 1
    df["ema_26"] = df["Close"].ewm(span=26, adjust=False).mean() / df["Close"] - 1
    df["rsi_14"] = _rsi(df["Close"])
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    df["macd"] = macd / df["Close"]
    df["macd_signal"] = macd.ewm(span=9, adjust=False).mean() / df["Close"]
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    df["momentum_10"] = df["Close"] - df["Close"].shift(10)
    df["roc_10"] = df["Close"].pct_change(10)
    df["roc_20"] = df["Close"].pct_change(20)
    df["roc_60"] = df["Close"].pct_change(60)

    low_14 = df["Low"].rolling(14).min()
    high_14 = df["High"].rolling(14).max()
    price_range_14 = (high_14 - low_14).replace(0, np.nan)
    df["stoch_k_14"] = 100 * (df["Close"] - low_14) / price_range_14
    df["stoch_d_3"] = df["stoch_k_14"].rolling(3).mean()
    df["williams_r_14"] = -100 * (high_14 - df["Close"]) / price_range_14

    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    tp_sma_20 = typical_price.rolling(20).mean()
    mean_dev_20 = (typical_price - tp_sma_20).abs().rolling(20).mean()
    df["cci_20"] = (typical_price - tp_sma_20) / (0.015 * mean_dev_20.replace(0, np.nan))

    bb_mid = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    df["bb_percent_b_20"] = (df["Close"] - (bb_mid - 2 * bb_std)) / (4 * bb_std).replace(0, np.nan)
    df["bb_bandwidth_20"] = (4 * bb_std) / bb_mid.replace(0, np.nan)

    donchian_high_20 = df["High"].rolling(20).max()
    donchian_low_20 = df["Low"].rolling(20).min()
    df["donchian_pos_20"] = (df["Close"] - donchian_low_20) / (donchian_high_20 - donchian_low_20).replace(0, np.nan)

    for win in [10, 20, 60]:
        df[f"rolling_std_{win}"] = df["return_1d"].rolling(win).std()
    df["atr_14"] = _atr(df) / df["Close"]
    df["adx_14"] = _adx(df)
    df["sharpe_20"] = df["return_1d"].rolling(20).mean() / df["return_1d"].rolling(20).std().replace(0, np.nan)
    df["sharpe_60"] = df["return_1d"].rolling(60).mean() / df["return_1d"].rolling(60).std().replace(0, np.nan)
    df["volume_change"] = df["Volume"].pct_change()
    df["volume_sma_20"] = df["Volume"].rolling(20).mean() / df["Volume"] - 1
    volume_mean_20 = df["Volume"].rolling(20).mean()
    volume_std_20 = df["Volume"].rolling(20).std()
    df["volume_zscore_20"] = (df["Volume"] - volume_mean_20) / volume_std_20.replace(0, np.nan)
    df["dollar_volume_20"] = np.log1p((df["Close"] * df["Volume"]).rolling(20).mean())
    direction = np.sign(df["Close"].diff()).fillna(0)
    df["obv"] = (direction * df["Volume"]).cumsum()
    df["obv"] = df["obv"].pct_change(20)
    df["close_to_52w_high"] = df["Close"] / df["Close"].rolling(252).max() - 1
    df["close_to_52w_low"] = df["Close"] / df["Close"].rolling(252).min() - 1
    rolling_low_252 = df["Close"].rolling(252).min()
    rolling_high_252 = df["Close"].rolling(252).max()
    df["price_position_252d"] = (
        (df["Close"] - rolling_low_252)
        / (rolling_high_252 - rolling_low_252).replace(0, np.nan)
    ).fillna(0.5)
    df["target_return_20d"] = df["Close"].shift(-TARGET_DAYS) / df["Close"] - 1
    df["target_return_60d"] = df["Close"].shift(-60) / df["Close"] - 1
    df["target_return_120d"] = df["Close"].shift(-HORIZON_DAYS) / df["Close"] - 1
    df["drawdown"] = df["Close"] / df["Close"].cummax() - 1
    df["volatility_60d"] = df["return_1d"].rolling(60).std()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


def make_modeling_dataset(all_df: pd.DataFrame) -> pd.DataFrame:
    features = [make_features_one(g) for _, g in all_df.groupby("Ticker")]
    out = pd.concat(features, ignore_index=True)
    out.to_csv(PROCESSED_DIR / "features_all.csv", index=False)
    modeling = out.dropna(subset=FEATURE_COLUMNS + ["target_return_20d", "target_return_60d", "target_return_120d"]).copy()
    modeling.to_csv(PROCESSED_DIR / "modeling_dataset.csv", index=False)
    return modeling
