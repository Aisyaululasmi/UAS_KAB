import warnings

import numpy as np
import pandas as pd

from src.config import (
    FEATURE_COLUMNS, HORIZON_DAYS, PREDICTIONS_DIR, TABLES_DIR, TICKERS,
    ensure_dirs,
)
from src.data_loader import combine_stock_data, download_stock_data
from src.features import make_features_one, make_modeling_dataset
from src.models_ml import (
    predict_latest_positive_probability,
    predict_latest_return,
    train_models_for_ticker,
)
from src.models_timesfm import timesfm_forecast
from src.portfolio import allocate_portfolio, allocate_portfolio_aggressive, final_recommendation, rank_stocks
from src.report_tables import generate_report
from src.utils import business_days_after
from src.validation import (
    add_equal_weight_benchmark,
    benchmark_comparison,
    sector_exposure,
    stress_test_results,
    summarize_walk_forward,
    walk_forward_validation,
)
from src.visualization import save_forecast_figures, save_portfolio_chart, save_price_history
from src.visualization import generate_all_forecast_panels

warnings.filterwarnings("ignore")


def historical_summary(all_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ticker, one in all_df.groupby("Ticker"):
        f = make_features_one(one)
        rows.append({
            "ticker": ticker,
            "start_date": pd.to_datetime(one["Date"]).min().date(),
            "end_date": pd.to_datetime(one["Date"]).max().date(),
            "rows": len(one),
            "last_close": one.sort_values("Date")["Close"].iloc[-1],
            "historical_return_6m": one.sort_values("Date")["Close"].pct_change(120).iloc[-1],
            "volatility_60d": f["volatility_60d"].iloc[-1],
            "max_drawdown": f["drawdown"].min(),
        })
    return pd.DataFrame(rows)


def _timesfm_shaped_path(last_close: float, tf_path: np.ndarray, final_price: float) -> np.ndarray:
    tf_path = np.asarray(tf_path, dtype=float)
    if len(tf_path) == 0 or not np.isfinite(tf_path).all() or tf_path[-1] == 0:
        return np.linspace(last_close, final_price, HORIZON_DAYS)
    scale_end = final_price / tf_path[-1]
    progress = np.linspace(0, 1, len(tf_path))
    scale = scale_end ** progress
    path = tf_path * scale
    path[0] = 0.70 * path[0] + 0.30 * last_close
    path[-1] = final_price
    return path


def _project_to_120d(ret: float, source_horizon: int) -> float:
    periods = HORIZON_DAYS / source_horizon
    projected = (1 + ret) ** periods - 1
    return float(np.clip(projected, -0.40, 0.60))


def _hybrid_120d_return(r20: float, r60: float, r120: float) -> float:
    return float(
        0.25 * _project_to_120d(r20, 20)
        + 0.25 * _project_to_120d(r60, 60)
        + 0.50 * np.clip(r120, -0.40, 0.60)
    )


def _dynamic_ensemble_weights(ticker_metrics: list[dict]) -> dict[str, float]:
    metrics = pd.DataFrame(ticker_metrics)
    rf_rmse = float(metrics.query("model == 'RandomForest'")["rmse"].iloc[0])
    cat_rmse = float(metrics.query("model == 'CatBoostRegressor'")["rmse"].iloc[0])
    inv_rf = 1 / max(rf_rmse, 1e-9)
    inv_cat = 1 / max(cat_rmse, 1e-9)
    tabular_total = inv_rf + inv_cat
    return {
        "timesfm": 0.30,
        "rf": 0.70 * inv_rf / tabular_total,
        "catboost": 0.70 * inv_cat / tabular_total,
    }


def build_forecast_rows(
    ticker: str,
    one_raw: pd.DataFrame,
    rf_ret: float,
    cat_ret: float,
    tf_prices,
    tf_label: str,
    ensemble_weights: dict[str, float],
):
    last = one_raw.sort_values("Date").iloc[-1]
    last_close = float(last["Close"])
    dates = business_days_after(last["Date"], HORIZON_DAYS)
    tf_path = np.asarray(tf_prices, dtype=float)
    rf_final_price = last_close * (1 + rf_ret)
    cat_final_price = last_close * (1 + cat_ret)
    rf_path = _timesfm_shaped_path(last_close, tf_path, rf_final_price)
    cat_path = _timesfm_shaped_path(last_close, tf_path, cat_final_price)
    ensemble = (
        ensemble_weights["timesfm"] * tf_path
        + ensemble_weights["rf"] * rf_path
        + ensemble_weights["catboost"] * cat_path
    )
    history = one_raw.sort_values("Date").tail(252)
    hist_rows = pd.DataFrame({
        "date": pd.to_datetime(history["Date"]),
        "ticker": ticker,
        "actual_price": history["Close"].values,
        "timesfm_forecast": np.nan,
        "rf_forecast": np.nan,
        "catboost_forecast": np.nan,
        "ensemble_forecast": history["Close"].values,
    })
    future_rows = pd.DataFrame({
        "date": dates,
        "ticker": ticker,
        "actual_price": np.nan,
        "timesfm_forecast": tf_path,
        "rf_forecast": rf_path,
        "catboost_forecast": cat_path,
        "ensemble_forecast": ensemble,
    })
    return pd.concat([hist_rows, future_rows], ignore_index=True), {
        "ticker": ticker,
        "latest_price": last_close,
        "predicted_price_6m": float(ensemble[-1]),
        "expected_return_6m": float(ensemble[-1] / last_close - 1),
        "timesfm_return_6m": float(tf_path[-1] / last_close - 1),
        "rf_return_6m": float(rf_final_price / last_close - 1),
        "catboost_return_6m": float(cat_final_price / last_close - 1),
        "timesfm_weight": ensemble_weights["timesfm"],
        "rf_weight": ensemble_weights["rf"],
        "catboost_weight": ensemble_weights["catboost"],
        "timesfm_label": tf_label,
    }


def main() -> None:
    ensure_dirs()
    raw_data = download_stock_data(TICKERS)
    all_df = combine_stock_data(raw_data)
    save_price_history(all_df)
    modeling = make_modeling_dataset(all_df)
    modeling = add_equal_weight_benchmark(modeling)
    walk_forward = walk_forward_validation(modeling)
    walk_forward_summary = summarize_walk_forward(walk_forward)

    metrics_rows = []
    classifier_metrics_rows = []
    test_prediction_frames = []
    feature_importance_rows = []
    forecast_frames = []
    summary_rows = []
    timesfm_labels = set()
    for ticker in TICKERS:
        one_model = modeling[modeling["Ticker"] == ticker].sort_values("Date")
        if len(one_model) < 260:
            continue
        (
            rf_model,
            cat_model,
            cat_cls_model,
            horizon_models,
            ticker_metrics,
            classifier_metrics,
            test_predictions,
            importance_rows,
        ) = train_models_for_ticker(modeling, ticker)
        metrics_rows.extend(ticker_metrics)
        classifier_metrics_rows.append(classifier_metrics)
        test_prediction_frames.append(test_predictions)
        feature_importance_rows.extend(importance_rows)
        latest = one_model.tail(1)
        rf_ret_20d = predict_latest_return(horizon_models[20]["rf"], latest)
        rf_ret_60d = predict_latest_return(horizon_models[60]["rf"], latest)
        rf_ret_120d = predict_latest_return(horizon_models[120]["rf"], latest)
        cat_ret_20d = predict_latest_return(horizon_models[20]["cat"], latest)
        cat_ret_60d = predict_latest_return(horizon_models[60]["cat"], latest)
        cat_ret_120d = predict_latest_return(horizon_models[120]["cat"], latest)
        rf_ret_hybrid = _hybrid_120d_return(rf_ret_20d, rf_ret_60d, rf_ret_120d)
        cat_ret_hybrid = _hybrid_120d_return(cat_ret_20d, cat_ret_60d, cat_ret_120d)
        cat_positive_prob = predict_latest_positive_probability(cat_cls_model, latest)
        one_raw = raw_data[ticker].sort_values("Date")
        tf_prices, tf_label = timesfm_forecast(one_raw["Close"])
        timesfm_labels.add(tf_label)
        ensemble_weights = _dynamic_ensemble_weights(ticker_metrics)
        f_rows, s_row = build_forecast_rows(
            ticker,
            one_raw,
            rf_ret_hybrid,
            cat_ret_hybrid,
            tf_prices,
            tf_label,
            ensemble_weights,
        )
        latest_features = make_features_one(one_raw).sort_values("Date").iloc[-1]
        s_row.update({
            "volatility_60d": float(latest_features["volatility_60d"]),
            "drawdown": float(latest_features["drawdown"]),
            "max_drawdown": float(make_features_one(one_raw)["drawdown"].min()),
            "price_position_252d": float(latest_features["price_position_252d"]),
            "avg_volume_60d": float(one_raw["Volume"].tail(60).mean()),
            "directional_accuracy": float(pd.DataFrame(ticker_metrics).query("model == 'Ensemble'")["directional_accuracy"].iloc[0]),
            "catboost_positive_prob": cat_positive_prob,
            "rf_return_20d": rf_ret_20d,
            "rf_return_60d": rf_ret_60d,
            "rf_return_120d": rf_ret_120d,
            "catboost_return_20d": cat_ret_20d,
            "catboost_return_60d": cat_ret_60d,
            "catboost_return_120d": cat_ret_120d,
        })
        forecast_frames.append(f_rows)
        summary_rows.append(s_row)

    model_metrics = pd.DataFrame(metrics_rows)
    classifier_metrics = pd.DataFrame(classifier_metrics_rows)
    test_predictions_df = pd.concat(test_prediction_frames, ignore_index=True)
    feature_importance = pd.DataFrame(feature_importance_rows)
    forecasts = pd.concat(forecast_frames, ignore_index=True)
    pred_summary = pd.DataFrame(summary_rows)
    benchmark_expected_return = float(pred_summary["expected_return_6m"].mean())
    pred_summary["benchmark_expected_return_6m"] = benchmark_expected_return
    pred_summary["expected_excess_return_6m"] = (
        pred_summary["expected_return_6m"] - pred_summary["benchmark_expected_return_6m"]
    )
    stock_ranking = rank_stocks(pred_summary)
    portfolio = allocate_portfolio(stock_ranking)
    aggressive_portfolio = allocate_portfolio_aggressive(stock_ranking)
    recommendation = final_recommendation(portfolio)
    hist_summary = historical_summary(all_df)
    benchmark_df = benchmark_comparison(stock_ranking, portfolio)
    stress_df = stress_test_results(all_df, portfolio)
    sector_df = sector_exposure(portfolio)

    model_metrics.to_csv(TABLES_DIR / "model_metrics.csv", index=False)
    classifier_metrics.to_csv(TABLES_DIR / "catboost_classifier_metrics.csv", index=False)
    classifier_metrics.to_csv(TABLES_DIR / "classification_metrics.csv", index=False)
    test_predictions_df.to_csv(TABLES_DIR / "test_predictions.csv", index=False)
    feature_importance.to_csv(TABLES_DIR / "feature_importance.csv", index=False)
    forecasts.to_csv(TABLES_DIR / "forecasts.csv", index=False)
    forecasts.to_csv(PREDICTIONS_DIR / "forecasts.csv", index=False)
    stock_ranking.to_csv(TABLES_DIR / "stock_ranking.csv", index=False)
    portfolio.to_csv(TABLES_DIR / "portfolio_allocation.csv", index=False)
    aggressive_portfolio.to_csv(TABLES_DIR / "portfolio_allocation_aggressive.csv", index=False)
    recommendation.to_csv(TABLES_DIR / "final_recommendation.csv", index=False)
    hist_summary.to_csv(TABLES_DIR / "historical_summary.csv", index=False)
    pred_summary.to_csv(TABLES_DIR / "forecast_summary.csv", index=False)
    pred_summary.to_csv(PREDICTIONS_DIR / "forecast_summary.csv", index=False)
    walk_forward.to_csv(TABLES_DIR / "walk_forward_results.csv", index=False)
    walk_forward_summary.to_csv(TABLES_DIR / "model_validation_summary.csv", index=False)
    benchmark_df.to_csv(TABLES_DIR / "benchmark_comparison.csv", index=False)
    stress_df.to_csv(TABLES_DIR / "stress_test_results.csv", index=False)
    sector_df.to_csv(TABLES_DIR / "sector_exposure.csv", index=False)

    future_forecasts = forecasts[forecasts["actual_price"].isna()].copy()
    future_forecasts["Forecast_Day"] = future_forecasts.groupby("ticker").cumcount() + 1
    forecast_daily_120d = future_forecasts.rename(columns={
        "date": "Date",
        "ticker": "Ticker",
        "timesfm_forecast": "TimesFM_Predicted_Price",
        "rf_forecast": "RF_Predicted_Price",
        "catboost_forecast": "CatBoost_Predicted_Price",
        "ensemble_forecast": "Ensemble_Predicted_Price",
    })
    last_price_map = pred_summary.set_index("ticker")["latest_price"]
    forecast_daily_120d["Last_Price"] = forecast_daily_120d["Ticker"].map(last_price_map)
    forecast_daily_120d["RF_Predicted_Return"] = (
        forecast_daily_120d["RF_Predicted_Price"] / forecast_daily_120d["Last_Price"]
        - 1
    )
    forecast_daily_120d["CatBoost_Predicted_Return"] = (
        forecast_daily_120d["CatBoost_Predicted_Price"] / forecast_daily_120d["Last_Price"]
        - 1
    )
    forecast_daily_120d["Ensemble_Predicted_Return"] = (
        forecast_daily_120d["Ensemble_Predicted_Price"] / forecast_daily_120d["Last_Price"]
        - 1
    )
    forecast_daily_120d.to_csv(TABLES_DIR / "forecast_daily_120d.csv", index=False)

    forecast_summary_6m = pred_summary.rename(columns={
        "ticker": "Ticker",
        "latest_price": "Last_Price",
        "predicted_price_6m": "Predicted_Price_6M",
        "timesfm_return_6m": "TimesFM_Return_6M",
        "rf_return_6m": "RF_Return_6M",
        "catboost_return_6m": "CatBoost_Return_6M",
        "expected_return_6m": "Ensemble_Return_6M",
        "catboost_positive_prob": "Buy_Probability",
        "benchmark_expected_return_6m": "Benchmark_Expected_Return_6M",
        "expected_excess_return_6m": "Expected_Excess_Return_6M",
    })
    forecast_summary_6m["Projected_Profit_Per_1B"] = forecast_summary_6m["Ensemble_Return_6M"] * 1_000_000_000
    model_returns = forecast_summary_6m[["TimesFM_Return_6M", "RF_Return_6M", "CatBoost_Return_6M"]]
    forecast_summary_6m["Forecast_Consistency"] = (model_returns.gt(0).sum(axis=1) / 3).round(4)
    forecast_summary_6m.to_csv(TABLES_DIR / "forecast_summary_6m.csv", index=False)

    risk_metrics = stock_ranking.rename(columns={
        "ticker": "Ticker",
        "volatility_60d": "Volatility_60D",
        "max_drawdown": "Max_Drawdown",
        "risk_score": "Risk_Score",
    })[["Ticker", "Volatility_60D", "Max_Drawdown", "Risk_Score"]].copy()
    risk_metrics["Risk_Level"] = pd.cut(
        risk_metrics["Risk_Score"],
        bins=[-0.01, 0.33, 0.66, 1.01],
        labels=["Low", "Medium", "High"],
    )
    risk_metrics.to_csv(TABLES_DIR / "risk_metrics.csv", index=False)

    stock_selection_matrix = stock_ranking.rename(columns={
        "rank": "Rank",
        "ticker": "Ticker",
        "expected_return_6m": "Ensemble_Return_6M",
        "catboost_positive_prob": "Buy_Probability",
        "volatility_60d": "Volatility_60D",
        "max_drawdown": "Max_Drawdown",
        "price_position": "Price_Position",
        "buy_lowest_score": "Buy_Lowest_Score",
        "hold_highest_score": "Hold_Highest_Score",
        "risk_score": "Risk_Score",
        "expected_excess_return_6m": "Expected_Excess_Return_6M",
        "ranking_score": "Final_Score",
        "signal": "Final_Signal",
        "selected": "Selected",
    })
    stock_selection_matrix["Hold_Probability"] = 1 - stock_selection_matrix["Buy_Probability"]
    stock_selection_matrix["Reason"] = stock_selection_matrix.apply(
        lambda row: f"{row['Final_Signal']} dengan return {row['Ensemble_Return_6M']:.2%}, prob BUY {row['Buy_Probability']:.2%}, risk {row['Risk_Score']:.3f}.",
        axis=1,
    )
    stock_selection_matrix.to_csv(TABLES_DIR / "stock_selection_matrix.csv", index=False)
    stock_selection_matrix[stock_selection_matrix["Selected"]].to_csv(TABLES_DIR / "top5_selected_stocks.csv", index=False)

    portfolio_summary = pd.DataFrame([{
        "scenario": "balanced",
        "total_capital_idr": portfolio["allocated_amount_idr"].sum(),
        "expected_portfolio_return_6m": (portfolio["final_weight"] * portfolio["expected_return_6m"]).sum(),
        "projected_profit_idr": portfolio["expected_profit_idr"].sum(),
        "selected_tickers": ", ".join(portfolio["ticker"].tolist()),
    }, {
        "scenario": "aggressive",
        "total_capital_idr": aggressive_portfolio["allocated_amount_idr"].sum(),
        "expected_portfolio_return_6m": (aggressive_portfolio["final_weight"] * aggressive_portfolio["expected_return_6m"]).sum(),
        "projected_profit_idr": aggressive_portfolio["expected_profit_idr"].sum(),
        "selected_tickers": ", ".join(aggressive_portfolio["ticker"].tolist()),
    }])
    portfolio_summary.to_csv(TABLES_DIR / "portfolio_summary.csv", index=False)

    save_forecast_figures(forecasts)
    save_portfolio_chart(portfolio)
    generate_all_forecast_panels(TICKERS, all_df, test_predictions_df, forecast_daily_120d, model_metrics)
    generate_report(
        hist_summary,
        model_metrics,
        stock_ranking,
        portfolio,
        recommendation,
        ", ".join(sorted(timesfm_labels)),
        classifier_metrics,
        walk_forward_summary,
        benchmark_df,
        stress_df,
        sector_df,
    )

    print("Pipeline selesai.")
    print(f"Output tabel: {TABLES_DIR}")
    print(f"Laporan: outputs/laporan_uas.md")
    print("Saham terpilih:", ", ".join(recommendation["ticker"].tolist()))


if __name__ == "__main__":
    main()
