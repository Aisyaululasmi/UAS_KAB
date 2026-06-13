from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PREDICTIONS_DIR = DATA_DIR / "predictions"
OUTPUTS_DIR = ROOT / "outputs"
TABLES_DIR = OUTPUTS_DIR / "tables"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = ROOT / "reports"

TICKERS = ["AAPL", "AXP", "KO", "BAC", "CVX", "MCO", "OXY", "CB", "KHC", "GOOGL"]
COMPANY_NAMES = {
    "AAPL": "Apple Inc.",
    "AXP": "American Express Co.",
    "KO": "Coca-Cola Co.",
    "BAC": "Bank of America Corp.",
    "CVX": "Chevron Corp.",
    "MCO": "Moody's Corporation",
    "OXY": "Occidental Petroleum Corp.",
    "CB": "Chubb Ltd.",
    "KHC": "Kraft Heinz Co.",
    "GOOGL": "Alphabet Inc. Class A",
}
SECTORS = {
    "AAPL": "Technology",
    "AXP": "Financial Services",
    "KO": "Consumer Staples",
    "BAC": "Financial Services",
    "CVX": "Energy",
    "MCO": "Financial Services",
    "OXY": "Energy",
    "CB": "Insurance",
    "KHC": "Consumer Staples",
    "GOOGL": "Communication Services",
}

START_DATE = "2018-01-01"
HORIZON_DAYS = 120
TARGET_DAYS = 20
TOTAL_CAPITAL_IDR = 1_000_000_000_000
RANDOM_STATE = 42

FEATURE_COLUMNS = [
    "return_1d", "return_5d", "return_10d", "return_20d", "log_return_1d",
    "price_change", "high_low_spread", "open_close_spread", "sma_5", "sma_10",
    "sma_20", "sma_50", "sma_100", "ema_12", "ema_26", "rsi_14", "macd",
    "macd_signal", "macd_hist", "momentum_10", "roc_10", "roc_20", "roc_60",
    "stoch_k_14", "stoch_d_3", "williams_r_14", "cci_20", "bb_percent_b_20",
    "bb_bandwidth_20", "donchian_pos_20", "rolling_std_10", "rolling_std_20",
    "rolling_std_60", "atr_14", "adx_14", "sharpe_20", "sharpe_60",
    "volume_change", "volume_sma_20", "volume_zscore_20", "dollar_volume_20",
    "obv", "close_to_52w_high", "close_to_52w_low", "price_position_252d",
]

def ensure_dirs() -> None:
    for path in [
        RAW_DIR, PROCESSED_DIR, PREDICTIONS_DIR, TABLES_DIR, FIGURES_DIR,
        FIGURES_DIR / "price_history", FIGURES_DIR / "forecast",
        FIGURES_DIR / "risk", FIGURES_DIR / "portfolio", REPORTS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
