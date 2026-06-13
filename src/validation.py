import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error

from .config import SECTORS
from .utils import directional_accuracy


def add_equal_weight_benchmark(modeling: pd.DataFrame) -> pd.DataFrame:
    out = modeling.copy()
    benchmark = out.groupby("Date")["target_return_120d"].mean().rename("benchmark_return_120d")
    out = out.merge(benchmark, left_on="Date", right_index=True, how="left")
    out["target_excess_return_120d"] = out["target_return_120d"] - out["benchmark_return_120d"]
    return out


def _metric_row(ticker: str, year: int, y_true, y_pred, benchmark_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    benchmark_pred = np.asarray(benchmark_pred, dtype=float)
    active = y_pred - benchmark_pred
    actual_active = y_true - benchmark_pred
    return {
        "ticker": ticker,
        "validation_year": year,
        "rows": len(y_true),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
        "hit_rate_vs_equal_weight": float(np.mean(actual_active > 0)),
        "avg_actual_return_120d": float(np.mean(y_true)),
        "avg_predicted_return_120d": float(np.mean(y_pred)),
        "avg_equal_weight_return_120d": float(np.mean(benchmark_pred)),
        "avg_actual_excess_return_120d": float(np.mean(actual_active)),
        "avg_predicted_excess_return_120d": float(np.mean(active)),
    }


def walk_forward_validation(modeling: pd.DataFrame, min_train_rows: int = 500) -> pd.DataFrame:
    """Fast walk-forward robustness check.

    The main model metrics are produced by the trained RF/CatBoost pipeline. This
    supplemental check avoids expensive repeated retraining and tests whether
    historical 120D return tendency plus observable momentum beats an internal
    equal-weight benchmark across the latest validation years.
    """
    rows = []
    years = sorted(pd.to_datetime(modeling["Date"]).dt.year.unique())[-3:]
    for ticker, one in modeling.groupby("Ticker"):
        one = one.sort_values("Date").copy()
        one["year"] = pd.to_datetime(one["Date"]).dt.year
        for year in years:
            train = one[one["year"] < year]
            valid = one[one["year"] == year]
            if len(train) < min_train_rows or len(valid) < 40:
                continue
            y_valid = valid["target_return_120d"]
            historical_mean = float(train["target_return_120d"].tail(504).mean())
            momentum_col = "return_60d" if "return_60d" in valid.columns else "roc_60"
            momentum_signal = valid[momentum_col].fillna(0).clip(-0.30, 0.30)
            risk_penalty = valid["volatility_60d"].fillna(valid["volatility_60d"].median()).clip(0, 0.08)
            pred = 0.65 * historical_mean + 0.45 * momentum_signal - 0.50 * risk_penalty
            rows.append(_metric_row(ticker, year, y_valid, pred, valid["benchmark_return_120d"]))
    return pd.DataFrame(rows)


def summarize_walk_forward(walk_forward: pd.DataFrame) -> pd.DataFrame:
    if walk_forward.empty:
        return pd.DataFrame()
    return (
        walk_forward.groupby("ticker", as_index=False)
        .agg(
            validation_windows=("validation_year", "count"),
            avg_mae=("mae", "mean"),
            avg_rmse=("rmse", "mean"),
            avg_directional_accuracy=("directional_accuracy", "mean"),
            avg_hit_rate_vs_equal_weight=("hit_rate_vs_equal_weight", "mean"),
            avg_actual_excess_return_120d=("avg_actual_excess_return_120d", "mean"),
        )
    )


def benchmark_comparison(stock_ranking: pd.DataFrame, portfolio: pd.DataFrame) -> pd.DataFrame:
    selected = portfolio.merge(
        stock_ranking[["ticker", "expected_return_6m", "expected_excess_return_6m"]],
        on="ticker",
        how="left",
        suffixes=("", "_ranking"),
    )
    universe_return = float(stock_ranking["expected_return_6m"].mean())
    top5_equal = stock_ranking.sort_values("ranking_score", ascending=False).head(5)
    top5_equal_return = float(top5_equal["expected_return_6m"].mean())
    portfolio_return = float((selected["final_weight"] * selected["expected_return_6m"]).sum())
    portfolio_excess = portfolio_return - universe_return
    rows = [
        {
            "strategy": "Risk-aware selected portfolio",
            "expected_return_6m": portfolio_return,
            "expected_excess_vs_equal_weight_universe": portfolio_excess,
            "notes": "Bobot mengikuti allocation score, risiko, dan cap portofolio.",
        },
        {
            "strategy": "Equal-weight all 10 candidates",
            "expected_return_6m": universe_return,
            "expected_excess_vs_equal_weight_universe": 0.0,
            "notes": "Benchmark internal karena tidak memakai data SPY eksternal.",
        },
        {
            "strategy": "Equal-weight top 5 by ranking score",
            "expected_return_6m": top5_equal_return,
            "expected_excess_vs_equal_weight_universe": top5_equal_return - universe_return,
            "notes": "Baseline sederhana tanpa optimasi bobot.",
        },
    ]
    return pd.DataFrame(rows)


def stress_test_results(all_df: pd.DataFrame, portfolio: pd.DataFrame) -> pd.DataFrame:
    weights = portfolio.set_index("ticker")["final_weight"].to_dict()
    scenarios = [
        ("COVID crash proxy", "2020-02-19", "2020-03-23"),
        ("2022 rate shock proxy", "2022-01-03", "2022-10-14"),
        ("Recent test window", "2024-12-11", "2025-12-12"),
    ]
    rows = []
    prices = all_df.pivot_table(index="Date", columns="Ticker", values="Close").sort_index()
    prices.index = pd.to_datetime(prices.index)
    for name, start, end in scenarios:
        window = prices.loc[pd.to_datetime(start):pd.to_datetime(end)]
        if len(window) < 2:
            continue
        stock_returns = window.iloc[-1] / window.iloc[0] - 1
        portfolio_return = sum(weights.get(ticker, 0.0) * stock_returns.get(ticker, np.nan) for ticker in weights)
        equal_weight_return = float(stock_returns.mean())
        worst_stock = stock_returns.idxmin()
        rows.append({
            "scenario": name,
            "start_date": start,
            "end_date": end,
            "portfolio_return": float(portfolio_return),
            "equal_weight_universe_return": equal_weight_return,
            "excess_vs_equal_weight": float(portfolio_return - equal_weight_return),
            "worst_stock": worst_stock,
            "worst_stock_return": float(stock_returns[worst_stock]),
        })
    return pd.DataFrame(rows)


def sector_exposure(portfolio: pd.DataFrame) -> pd.DataFrame:
    out = portfolio.copy()
    out["sector"] = out["ticker"].map(SECTORS).fillna("Other")
    return (
        out.groupby("sector", as_index=False)["final_weight"]
        .sum()
        .rename(columns={"final_weight": "portfolio_weight"})
        .sort_values("portfolio_weight", ascending=False)
    )
