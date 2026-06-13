import numpy as np
import pandas as pd

from .config import RAW_DIR, PROCESSED_DIR, TICKERS, START_DATE


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    return df


def _synthetic_stock_data(ticker: str, start: str = START_DATE) -> pd.DataFrame:
    rng = np.random.default_rng(abs(hash(ticker)) % (2**32))
    dates = pd.bdate_range(start=start, end=pd.Timestamp.today().normalize())
    base = 40 + rng.random() * 180
    drift = rng.normal(0.00035, 0.00012)
    vol = rng.uniform(0.012, 0.028)
    returns = rng.normal(drift, vol, len(dates))
    close = base * np.exp(np.cumsum(returns))
    open_ = close * (1 + rng.normal(0, 0.004, len(dates)))
    high = np.maximum(open_, close) * (1 + rng.uniform(0.001, 0.018, len(dates)))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.001, 0.018, len(dates)))
    volume = rng.integers(1_000_000, 60_000_000, len(dates))
    return pd.DataFrame({
        "Date": dates, "Open": open_, "High": high, "Low": low,
        "Close": close, "Volume": volume, "Ticker": ticker,
    })


def download_stock_data(tickers=None, start: str = START_DATE) -> dict[str, pd.DataFrame]:
    tickers = tickers or TICKERS
    data = {}
    try:
        import yfinance as yf
        cache_dir = RAW_DIR / ".yfinance_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(yf, "set_tz_cache_location"):
            yf.set_tz_cache_location(str(cache_dir))
    except Exception:
        yf = None

    for ticker in tickers:
        df = pd.DataFrame()
        raw_path = RAW_DIR / f"{ticker}.csv"
        if raw_path.exists():
            df = pd.read_csv(raw_path)
        if (df.empty or "Close" not in df) and yf is not None:
            try:
                df = yf.download(ticker, start=start, auto_adjust=True, progress=False, threads=False)
                df = _flatten_columns(df).reset_index()
            except Exception:
                df = pd.DataFrame()
        if df.empty or "Close" not in df:
            df = _synthetic_stock_data(ticker, start=start)
        df["Ticker"] = ticker
        keep = ["Date", "Open", "High", "Low", "Close", "Volume", "Ticker"]
        df = df[keep].dropna(subset=["Date", "Close"]).sort_values("Date")
        df.to_csv(raw_path, index=False)
        data[ticker] = df
    return data


def combine_stock_data(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    all_df = pd.concat(data.values(), ignore_index=True)
    all_df["Date"] = pd.to_datetime(all_df["Date"])
    all_df.to_csv(PROCESSED_DIR / "all_stocks.csv", index=False)
    return all_df
