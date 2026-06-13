import pandas as pd
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import FIGURES_DIR


def save_price_history(all_df: pd.DataFrame) -> None:
    for ticker, one in all_df.groupby("Ticker"):
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(pd.to_datetime(one["Date"]), one["Close"], linewidth=1.2)
        ax.set_title(f"Historical Close Price - {ticker}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Close Price (USD)")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "price_history" / f"{ticker}_price_history.png", dpi=140)
        plt.close(fig)


def save_forecast_figures(forecasts: pd.DataFrame) -> None:
    for ticker, one in forecasts.groupby("ticker"):
        fig, ax = plt.subplots(figsize=(9, 4.5))
        plot = one.sort_values("date")
        if "actual_price" in plot:
            ax.plot(pd.to_datetime(plot["date"]), plot["actual_price"], label="Actual", linewidth=1.2)
        ax.plot(pd.to_datetime(plot["date"]), plot["ensemble_forecast"], label="Ensemble", linewidth=1.5)
        ax.set_title(f"Forecast 6 Bulan - {ticker}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "forecast" / f"{ticker}_forecast.png", dpi=140)
        plt.close(fig)


def save_portfolio_chart(portfolio: pd.DataFrame) -> None:
    if portfolio.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.pie(portfolio["final_weight"], labels=portfolio["ticker"], autopct="%1.1f%%", startangle=90)
    ax.set_title("Alokasi Portofolio")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "portfolio" / "portfolio_allocation.png", dpi=140)
    plt.close(fig)


def plot_forecast_panel(
    ticker,
    history_df,
    test_df,
    forecast_df,
    metrics_dict,
    output_path,
    price_col="Close",
    actual_col="Actual",
    predicted_col="Predicted",
    forecast_col="Forecast",
):
    output_path = str(output_path)
    import os

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    history_df = history_df.copy()
    test_df = test_df.copy()
    forecast_df = forecast_df.copy()
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    ax1, ax2 = axes

    if not test_df.empty and {"Date", actual_col, predicted_col}.issubset(test_df.columns):
        test_df["Date"] = pd.to_datetime(test_df["Date"])
        ax1.plot(test_df["Date"], test_df[actual_col], label="Aktual", linewidth=1.8)
        ax1.plot(test_df["Date"], test_df[predicted_col], label="Prediksi", linestyle="--", linewidth=1.8)
    else:
        ax1.text(0.5, 0.5, "Data test aktual/prediksi belum tersedia", ha="center", va="center", transform=ax1.transAxes)

    ax1.set_title(f"{ticker} - Aktual vs Prediksi (Test)")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price (USD)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    def fmt_num(x, digits=4):
        try:
            if x is None or np.isnan(x):
                return "N/A"
        except Exception:
            if x is None:
                return "N/A"
        return f"{float(x):.{digits}f}"

    def fmt_pct(x):
        try:
            if x is None or np.isnan(x):
                return "N/A"
        except Exception:
            if x is None:
                return "N/A"
        x = float(x)
        return f"{x * 100:.2f}%" if 0 <= x <= 1 else f"{x:.2f}%"

    mse = metrics_dict.get("MSE", metrics_dict.get("mse"))
    rmse = metrics_dict.get("RMSE", metrics_dict.get("rmse"))
    mae = metrics_dict.get("MAE", metrics_dict.get("mae"))
    dstat = metrics_dict.get("Directional_Accuracy", metrics_dict.get("directional_accuracy"))
    metric_text = f"MSE   : {fmt_num(mse)}\nRMSE  : {fmt_num(rmse)}\nMAE   : {fmt_num(mae)}\nDSTAT : {fmt_pct(dstat)}"
    ax1.text(0.02, 0.95, metric_text, transform=ax1.transAxes, verticalalignment="top", fontsize=10, bbox=dict(boxstyle="round", alpha=0.15))

    if not history_df.empty and "Date" in history_df.columns:
        history_df["Date"] = pd.to_datetime(history_df["Date"])
        if price_col not in history_df.columns:
            price_col = "Adj Close" if "Adj Close" in history_df.columns else "Close"
        ax2.plot(history_df["Date"], history_df[price_col], label="Historis", linewidth=1.5)

    if not forecast_df.empty and "Date" in forecast_df.columns:
        forecast_df["Date"] = pd.to_datetime(forecast_df["Date"])
        if forecast_col not in forecast_df.columns:
            forecast_col = "Ensemble_Predicted_Price" if "Ensemble_Predicted_Price" in forecast_df.columns else "Forecast"
        if forecast_col in forecast_df.columns:
            ax2.plot(forecast_df["Date"], forecast_df[forecast_col], label="Forecast (120D)", linewidth=2.0)
            upper = forecast_df[forecast_col] * 1.03
            lower = forecast_df[forecast_col] * 0.97
            ax2.fill_between(forecast_df["Date"], lower, upper, alpha=0.2, label="+/-3% band")
            ax2.axvline(forecast_df["Date"].iloc[0], linestyle=":", alpha=0.8, label="Awal Forecast")
            if not history_df.empty and price_col in history_df.columns:
                last_actual_price = history_df[price_col].dropna().iloc[-1]
                final_forecast_price = forecast_df[forecast_col].dropna().iloc[-1]
                trend = ((final_forecast_price / last_actual_price) - 1) * 100
                ax2.text(0.05, 0.05, f"Trend: {trend:+.2f}%", transform=ax2.transAxes, fontsize=11, fontweight="bold")
    else:
        ax2.text(0.5, 0.5, "Data forecast belum tersedia", ha="center", va="center", transform=ax2.transAxes)

    ax2.set_title(f"{ticker} - Full History + Forecast")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Price (USD)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_all_forecast_panels(tickers, historical_df, test_predictions_df, forecast_daily_df, metrics_df, output_dir=FIGURES_DIR):
    output_dir = FIGURES_DIR if output_dir is None else output_dir
    for ticker in tickers:
        history_ticker = historical_df[historical_df["Ticker"] == ticker].copy()
        test_ticker = test_predictions_df[test_predictions_df["Ticker"] == ticker].copy() if not test_predictions_df.empty else pd.DataFrame()
        if "Ticker" in forecast_daily_df.columns:
            forecast_ticker = forecast_daily_df[forecast_daily_df["Ticker"] == ticker].copy()
        else:
            forecast_ticker = pd.DataFrame()
        metrics_dict = {}
        if metrics_df is not None and not metrics_df.empty:
            metric_row = metrics_df[(metrics_df["ticker"] == ticker) & (metrics_df["model"] == "Ensemble")]
            if not metric_row.empty:
                metrics_dict = metric_row.iloc[0].to_dict()
                if "rmse" in metrics_dict:
                    metrics_dict["mse"] = float(metrics_dict["rmse"]) ** 2
        plot_forecast_panel(
            ticker=ticker,
            history_df=history_ticker,
            test_df=test_ticker,
            forecast_df=forecast_ticker,
            metrics_dict=metrics_dict,
            output_path=FIGURES_DIR / f"forecast_panel_{ticker}.png",
            price_col="Adj Close" if "Adj Close" in history_ticker.columns else "Close",
            actual_col="Actual",
            predicted_col="Predicted",
            forecast_col="Ensemble_Predicted_Price",
        )
