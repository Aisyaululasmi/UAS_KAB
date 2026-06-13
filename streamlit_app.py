from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from components.charts import bar_return, line_forecast, pie_allocation
from components.metrics_cards import summary_metrics
from components.tables import show_table

def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "outputs" / "tables").exists():
            return parent
    return current.parents[1]


ROOT = find_project_root()
TABLES_DIR = ROOT / "outputs" / "tables"
FIGURES_DIR = ROOT / "outputs" / "figures"
TOTAL_CAPITAL = 1_000_000_000_000
TICKER_ORDER = ["AAPL", "AXP", "KO", "BAC", "CVX", "MCO", "OXY", "CB", "KHC", "GOOGL"]

st.set_page_config(page_title="UAS SKAB Stock Forecasting", layout="wide")


def apply_light_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --primary: #1f6feb;
            --primary-dark: #164a9f;
            --ink: #172033;
            --muted: #667085;
            --line: #e6ebf2;
            --soft: #f6f8fb;
            --panel: #ffffff;
            --accent: #0f766e;
        }

        html, body, [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] > .main,
        .stApp {
            background:
                linear-gradient(180deg, #f7fbff 0%, #ffffff 34%, #f8fafc 100%) !important;
            color: #111111 !important;
        }

        header[data-testid="stHeader"],
        div[data-testid="stToolbar"],
        div[data-testid="stDecoration"],
        section[data-testid="stSidebar"],
        section[data-testid="stSidebar"] > div {
            background: #ffffff !important;
            color: #111111 !important;
            border-right: 1px solid var(--line);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2.5rem;
            max-width: 1420px;
        }

        .app-hero {
            background: linear-gradient(135deg, #ffffff 0%, #eef6ff 58%, #f5fffb 100%) !important;
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1.35rem 1.55rem 1.25rem;
            margin-bottom: 1.1rem;
            box-shadow: 0 16px 40px rgba(31, 111, 235, 0.08);
            text-align: center;
        }

        .app-hero h1 {
            color: #111111 !important;
            font-size: clamp(1.65rem, 2vw, 2.35rem);
            line-height: 1.16;
            margin: 0 0 .75rem 0;
            letter-spacing: 0;
            text-align: center;
        }

        .identity-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: .65rem;
        }

        .identity-item {
            background: rgba(255, 255, 255, 0.92) !important;
            border: 1px solid #dde8f5;
            border-radius: 8px;
            padding: .72rem .85rem;
        }

        .identity-label {
            color: #111111 !important;
            display: block;
            font-size: .78rem;
            margin-bottom: .18rem;
        }

        .identity-value {
            color: #111111 !important;
            font-weight: 700;
            font-size: .95rem;
        }

        .stApp p, .stApp li, .stApp label,
        [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] *,
        [data-testid="caption"], [data-testid="stCaptionContainer"], small {
            color: #111111 !important;
        }

        h1, h2, h3, h4, h5, h6 {
            color: #111111 !important;
            letter-spacing: 0;
        }

        div[data-testid="stMetric"] {
            background: #ffffff !important;
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: .85rem .95rem;
            box-shadow: 0 10px 26px rgba(15, 23, 42, 0.05);
        }

        div[data-testid="stMetricLabel"] p {
            color: #111111 !important;
            font-size: .82rem;
        }

        div[data-testid="stMetricValue"],
        div[data-testid="stMetricValue"] * {
            color: #111111 !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            background: #ffffff !important;
            gap: .35rem;
            border-bottom: 1px solid var(--line);
        }

        .stTabs [data-baseweb="tab"] {
            background: #ffffff !important;
            border: 1px solid var(--line);
            border-bottom: 0;
            border-radius: 8px 8px 0 0;
            color: #111111 !important;
            padding: .55rem .85rem;
        }

        .stTabs [aria-selected="true"] {
            color: #111111 !important;
            background: #eef6ff !important;
            font-weight: 700;
        }

        [data-testid="stSelectbox"] label,
        [data-testid="stRadio"] label,
        [data-testid="stExpander"] summary,
        [data-baseweb="select"] span,
        [role="option"] {
            color: #111111 !important;
        }

        [data-baseweb="select"] > div,
        [data-baseweb="popover"] > div {
            background-color: #ffffff !important;
            color: #111111 !important;
            border-color: #d8e0ea !important;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.04);
            background: #ffffff !important;
            color: #111111 !important;
        }

        div[data-testid="stDataFrame"] *,
        div[data-testid="stDataFrame"] [role="grid"],
        div[data-testid="stDataFrame"] [role="row"],
        div[data-testid="stDataFrame"] [role="gridcell"],
        div[data-testid="stDataFrame"] [role="columnheader"] {
            background-color: #ffffff !important;
            color: #111111 !important;
            border-color: #e6ebf2 !important;
        }

        div[data-testid="stDataFrame"] [role="columnheader"],
        div[data-testid="stDataFrame"] [data-testid="stDataFrameResizable"] {
            background-color: #eef6ff !important;
            color: #111111 !important;
            font-weight: 700 !important;
        }

        [data-testid="stTable"] {
            background: #ffffff !important;
            color: #111111 !important;
            border: 1px solid var(--line);
            border-radius: 8px;
            overflow: hidden;
        }

        [data-testid="stTable"] table,
        [data-testid="stTable"] thead,
        [data-testid="stTable"] tbody,
        [data-testid="stTable"] tr,
        [data-testid="stTable"] th,
        [data-testid="stTable"] td {
            background: #ffffff !important;
            color: #111111 !important;
        }

        [data-testid="stTable"] th {
            background: #eef6ff !important;
            color: #111111 !important;
            font-weight: 700;
        }

        div[data-testid="stAlert"],
        div[data-testid="stAlert"] *,
        [data-testid="stNotification"],
        [data-testid="stNotification"] * {
            background-color: #ffffff !important;
            color: #111111 !important;
        }

        button,
        button *,
        input,
        textarea,
        [data-baseweb="input"],
        [data-baseweb="textarea"],
        [data-baseweb="button"],
        [data-baseweb="menu"],
        [data-baseweb="menu"] *,
        [data-baseweb="popover"] *,
        [role="listbox"],
        [role="listbox"] *,
        [role="option"],
        [role="option"] * {
            background-color: #ffffff !important;
            color: #111111 !important;
            border-color: #d8e0ea !important;
        }

        button[kind="primary"],
        button[kind="primary"] * {
            background-color: #1f6feb !important;
            color: #ffffff !important;
        }

        div[data-testid="stImage"] img {
            border-radius: 8px;
            border: 1px solid var(--line);
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.06);
            background: #ffffff !important;
        }

        .stAlert {
            border-radius: 8px;
        }

        @media (max-width: 900px) {
            .identity-grid {
                grid-template-columns: 1fr;
            }
            .app-hero {
                padding: 1rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


apply_light_theme()


@st.cache_data
def load_csv(name: str) -> pd.DataFrame:
    path = TABLES_DIR / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def available_tickers(*frames: pd.DataFrame) -> list[str]:
    tickers = set()
    for frame in frames:
        if frame.empty:
            continue
        for column in ["ticker", "Ticker"]:
            if column in frame.columns:
                tickers.update(frame[column].dropna().astype(str).unique())
    ordered = [ticker for ticker in TICKER_ORDER if ticker in tickers]
    ordered.extend(sorted(ticker for ticker in tickers if ticker not in ordered))
    return ordered


def show_image(path: Path, caption: str, missing: str | None = None) -> None:
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    elif missing:
        st.info(missing)


def first_existing_path(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def style_plotly(fig):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="#ffffff",
        font={"family": "Arial, sans-serif", "color": "#172033"},
        title={"font": {"size": 20, "color": "#172033"}},
        margin={"l": 24, "r": 24, "t": 70, "b": 38},
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eef2f7", linecolor="#d8e0ea")
    fig.update_yaxes(showgrid=True, gridcolor="#eef2f7", linecolor="#d8e0ea")
    return fig


def show_ticker_visuals(ticker: str) -> None:
    panel_path = FIGURES_DIR / f"forecast_panel_{ticker}.png"
    price_history_path = FIGURES_DIR / "price_history" / f"{ticker}_price_history.png"
    forecast_path = FIGURES_DIR / "forecast" / f"{ticker}_forecast.png"

    show_image(
        panel_path,
        f"Panel evaluasi dan forecast 120 hari - {ticker}",
        f"Forecast panel {ticker} belum tersedia. Jalankan `python main.py`.",
    )

    col1, col2 = st.columns(2)
    with col1:
        show_image(price_history_path, f"Riwayat harga - {ticker}")
    with col2:
        show_image(forecast_path, f"Forecast detail - {ticker}")


st.markdown(
    """
    <div class="app-hero">
        <h1>Dashboard Rekomendasi Portofolio Saham<br>
        Ujian Akhir Semester Mata Kuliah Kecerdasan Artifisial pada Bisnis</h1>
        <div class="identity-grid">
            <div class="identity-item">
                <span class="identity-label">Nama</span>
                <span class="identity-value">Aisya Ulul Asmi</span>
            </div>
            <div class="identity-item">
                <span class="identity-label">NIM</span>
                <span class="identity-value">25/564969/PPA/07123</span>
            </div>
            <div class="identity-item">
                <span class="identity-label">Model</span>
                <span class="identity-value">Ensemble TimesFM 2.5 200M PyTorch + RandomForest + CatBoost.</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

metrics_df = load_csv("model_metrics.csv")
classifier_df = load_csv("catboost_classifier_metrics.csv")
classification_df = load_csv("classification_metrics.csv")
forecast_df = load_csv("forecasts.csv")
forecast_summary_6m_df = load_csv("forecast_summary_6m.csv")
test_predictions_df = load_csv("test_predictions.csv")
ranking_df = load_csv("stock_ranking.csv")
portfolio_df = load_csv("portfolio_allocation.csv")
portfolio_aggressive_df = load_csv("portfolio_allocation_aggressive.csv")
portfolio_summary_df = load_csv("portfolio_summary.csv")
recommendation_df = load_csv("final_recommendation.csv")
historical_df = load_csv("historical_summary.csv")
risk_df = load_csv("risk_metrics.csv")
selection_df = load_csv("top5_selected_stocks.csv")
feature_importance_df = load_csv("feature_importance.csv")
daily_forecast_df = load_csv("forecast_daily_120d.csv")
validation_summary_df = load_csv("model_validation_summary.csv")
walk_forward_df = load_csv("walk_forward_results.csv")
benchmark_df = load_csv("benchmark_comparison.csv")
stress_df = load_csv("stress_test_results.csv")
sector_df = load_csv("sector_exposure.csv")
conservative_portfolio_df = load_csv("portfolio_allocation_conservative.csv")
scenario_summary_df = load_csv("scenario_summary.csv")
final_decision_df = load_csv("final_decision_summary.csv")
confidence_df = load_csv("recommendation_confidence_levels.csv")
transaction_cost_df = load_csv("transaction_cost_sensitivity.csv")
external_benchmark_proxy_df = load_csv("external_benchmark_proxy.csv")
monthly_backtest_df = load_csv("monthly_rebalance_backtest_summary.csv")
monthly_backtest_curve_df = load_csv("monthly_rebalance_equity_curve.csv")
fundamental_proxy_df = load_csv("fundamental_valuation_proxy.csv")
explainability_global_df = load_csv("explainability_global_top_features.csv")
explainability_ticker_df = load_csv("explainability_ticker_top_features.csv")
drawdown_analysis_df = load_csv("portfolio_drawdown_analysis.csv")
governance_df = load_csv("governance_monitoring_plan.csv")

if all(df.empty for df in [metrics_df, forecast_df, ranking_df, portfolio_df, recommendation_df]):
    st.warning("CSV output belum tersedia. Jalankan `python main.py` terlebih dahulu.")

tickers = available_tickers(forecast_df, forecast_summary_6m_df, ranking_df, historical_df)

tab_summary, tab_final, tab_forecast, tab_visuals, tab_metrics, tab_reco, tab_portfolio, tab_risk = st.tabs(
    [
        "Executive Summary",
        "Final Decision",
        "Forecasting",
        "Visualisasi",
        "Model Evaluation",
        "BUY/HOLD Recommendation",
        "Portfolio Allocation",
        "Risk Analysis",
    ]
)

with tab_summary:
    st.header("Executive Summary")
    summary_metrics(TOTAL_CAPITAL, portfolio_df)
    if not portfolio_summary_df.empty:
        st.subheader("Perbandingan Skenario")
        show_table(portfolio_summary_df)
    if not benchmark_df.empty:
        st.subheader("Benchmark Comparison")
        show_table(benchmark_df)
    if not external_benchmark_proxy_df.empty:
        st.subheader("External Benchmark Proxy")
        show_table(external_benchmark_proxy_df)
    if not monthly_backtest_df.empty:
        st.subheader("Monthly Rebalance Backtest")
        show_table(monthly_backtest_df)
        if not monthly_backtest_curve_df.empty and {
            "Date", "selected_monthly_rebalance_equity", "universe_equal_weight_equity"
        }.issubset(monthly_backtest_curve_df.columns):
            curve = monthly_backtest_curve_df.copy()
            curve["Date"] = pd.to_datetime(curve["Date"])
            fig = px.line(
                curve,
                x="Date",
                y=["selected_monthly_rebalance_equity", "universe_equal_weight_equity"],
                title="Monthly Rebalance Equity Curve",
            )
            st.plotly_chart(style_plotly(fig), use_container_width=True)
    st.subheader("Rekomendasi Final")
    show_table(recommendation_df)
    portfolio_image = FIGURES_DIR / "portfolio" / "portfolio_allocation.png"
    portfolio_image = first_existing_path(portfolio_image, FIGURES_DIR / "portfolio_allocation.png")
    show_image(portfolio_image, "Visualisasi alokasi portofolio")
    st.subheader("Ringkasan Historis")
    show_table(historical_df)

with tab_final:
    st.header("Final Decision Summary")
    if not final_decision_df.empty:
        show_table(final_decision_df)
    if not scenario_summary_df.empty:
        st.subheader("Scenario Summary")
        show_table(scenario_summary_df)
        if {"scenario", "expected_return_6m", "projected_profit_idr"}.issubset(scenario_summary_df.columns):
            fig = px.bar(
                scenario_summary_df,
                x="scenario",
                y="expected_return_6m",
                color="scenario",
                title="Expected Return per Scenario",
            )
            st.plotly_chart(style_plotly(fig), use_container_width=True)
    if not confidence_df.empty:
        st.subheader("Confidence Level Rekomendasi")
        show_table(confidence_df)
    if not transaction_cost_df.empty:
        st.subheader("Transaction Cost and Slippage")
        show_table(transaction_cost_df)

with tab_forecast:
    st.header("Forecasting Harga Saham")
    if tickers:
        selected_ticker = st.selectbox("Pilih Ticker", tickers)
        show_ticker_visuals(selected_ticker)
        if not forecast_summary_6m_df.empty and "Ticker" in forecast_summary_6m_df.columns:
            ticker_summary = forecast_summary_6m_df[forecast_summary_6m_df["Ticker"] == selected_ticker]
            if not ticker_summary.empty:
                row = ticker_summary.iloc[0]
                st.markdown(
                    f"""
                    **Interpretasi Singkat:** Saham **{selected_ticker}** memiliki expected return ensemble
                    **{row.get('Ensemble_Return_6M', 0):.2%}** untuk horizon 6 bulan. Panel visual membandingkan prediksi
                    pada data test dengan forecast 120 trading days ke depan.
                    """
                )
                summary_columns = [
                    "Ticker",
                    "Last_Price",
                    "Predicted_Price_6M",
                    "Ensemble_Return_6M",
                    "TimesFM_Return_6M",
                    "RF_Return_6M",
                    "CatBoost_Return_6M",
                    "Buy_Probability",
                    "Forecast_Consistency",
                ]
                st.subheader("Ringkasan Forecast 6 Bulan")
                show_table(ticker_summary[[c for c in summary_columns if c in ticker_summary.columns]])

        ticker_df = forecast_df[forecast_df["ticker"] == selected_ticker].copy()
        if "date" in ticker_df.columns:
            st.plotly_chart(line_forecast(ticker_df, selected_ticker), use_container_width=True)
        show_table(ticker_df)
        if not daily_forecast_df.empty:
            daily_column = "ticker" if "ticker" in daily_forecast_df.columns else "Ticker" if "Ticker" in daily_forecast_df.columns else None
            if daily_column:
                st.subheader("Forecast Daily 120 Trading Days")
                show_table(daily_forecast_df[daily_forecast_df[daily_column] == selected_ticker])
    else:
        st.info("Data forecast belum tersedia. Jalankan `python main.py`.")

with tab_visuals:
    st.header("Galeri Visualisasi")
    if tickers:
        visual_mode = st.radio("Jenis visualisasi", ["Forecast Panel", "Price History", "Forecast Detail"], horizontal=True)
        image_dir = {
            "Forecast Panel": FIGURES_DIR,
            "Price History": FIGURES_DIR / "price_history",
            "Forecast Detail": FIGURES_DIR / "forecast",
        }[visual_mode]
        suffix = {
            "Forecast Panel": "forecast_panel_{ticker}.png",
            "Price History": "{ticker}_price_history.png",
            "Forecast Detail": "{ticker}_forecast.png",
        }[visual_mode]

        cols = st.columns(2)
        for idx, ticker in enumerate(tickers):
            path = image_dir / suffix.format(ticker=ticker)
            with cols[idx % 2]:
                show_image(path, f"{visual_mode} - {ticker}")
    else:
        st.info("Belum ada ticker yang dapat divisualisasikan.")

with tab_metrics:
    st.header("Evaluasi Model")
    show_table(metrics_df)
    st.subheader("Evaluasi CatBoostClassifier")
    show_table(classifier_df)
    if not classification_df.empty:
        st.subheader("Classification Metrics Tambahan")
        show_table(classification_df)
    if not validation_summary_df.empty:
        st.subheader("Walk-Forward Robustness Summary")
        show_table(validation_summary_df)
    if not walk_forward_df.empty:
        with st.expander("Detail Walk-Forward Results"):
            show_table(walk_forward_df)
    if not metrics_df.empty and {"model", "mape"}.issubset(metrics_df.columns):
        fig = px.bar(metrics_df, x="model", y="mape", color="ticker", title="Perbandingan MAPE Model")
        st.plotly_chart(style_plotly(fig), use_container_width=True)
    if not feature_importance_df.empty:
        st.subheader("Feature Importance")
        model_options = sorted(feature_importance_df["model"].dropna().unique()) if "model" in feature_importance_df else []
        ticker_options = sorted(feature_importance_df["ticker"].dropna().unique()) if "ticker" in feature_importance_df else []
        col1, col2 = st.columns(2)
        selected_model = col1.selectbox("Model", model_options, key="fi_model") if model_options else None
        selected_fi_ticker = col2.selectbox("Ticker", ticker_options, key="fi_ticker") if ticker_options else None
        fi_view = feature_importance_df.copy()
        if selected_model:
            fi_view = fi_view[fi_view["model"] == selected_model]
        if selected_fi_ticker:
            fi_view = fi_view[fi_view["ticker"] == selected_fi_ticker]
        if {"feature", "importance"}.issubset(fi_view.columns):
            fi_top = fi_view.sort_values("importance", ascending=False).head(20)
            fig = px.bar(fi_top, x="importance", y="feature", orientation="h", title="Top 20 Feature Importance")
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(style_plotly(fig), use_container_width=True)
        show_table(fi_view.head(100))
    if not explainability_global_df.empty:
        st.subheader("Explainability - Global Top Features")
        show_table(explainability_global_df)
        if {"feature", "importance"}.issubset(explainability_global_df.columns):
            fig = px.bar(
                explainability_global_df.head(15),
                x="importance",
                y="feature",
                color="model" if "model" in explainability_global_df.columns else None,
                orientation="h",
                title="Top Explainability Features",
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(style_plotly(fig), use_container_width=True)
    if not explainability_ticker_df.empty:
        st.subheader("Explainability - Selected Ticker Detail")
        show_table(explainability_ticker_df)

with tab_reco:
    st.header("Ranking dan Sinyal BUY/HOLD")
    show_table(ranking_df)
    if not ranking_df.empty and {"ticker", "expected_return_6m"}.issubset(ranking_df.columns):
        st.plotly_chart(bar_return(ranking_df), use_container_width=True)
    if not selection_df.empty:
        st.subheader("Top 5 Selected Stocks")
        show_table(selection_df)
    if not confidence_df.empty:
        st.subheader("Confidence Level")
        show_table(confidence_df)
    if not fundamental_proxy_df.empty:
        st.subheader("Fundamental and Valuation Proxy")
        show_table(fundamental_proxy_df)
    if not recommendation_df.empty:
        st.subheader("Rekomendasi Final")
        for _, row in recommendation_df.iterrows():
            ticker = row.get("ticker", row.get("Ticker", ""))
            reason = row.get("reason", "")
            signal = row.get("signal", "")
            expected_return = row.get("expected_return_6m", 0)
            st.markdown(f"**{ticker} - {signal}:** expected return {expected_return:.2%}. {reason}")

with tab_portfolio:
    st.header("Alokasi Portofolio")
    portfolio_image = first_existing_path(
        FIGURES_DIR / "portfolio" / "portfolio_allocation.png",
        FIGURES_DIR / "portfolio_allocation.png",
    )
    show_image(portfolio_image, "Grafik alokasi portofolio")
    st.subheader("Conservative Scenario")
    show_table(conservative_portfolio_df)
    if not conservative_portfolio_df.empty and {"ticker", "allocated_amount_idr"}.issubset(conservative_portfolio_df.columns):
        conservative_plot = conservative_portfolio_df[conservative_portfolio_df["allocated_amount_idr"] > 0].copy()
        st.plotly_chart(pie_allocation(conservative_plot), use_container_width=True)
    st.subheader("Balanced Scenario")
    show_table(portfolio_df)
    if not portfolio_df.empty and {"ticker", "allocated_amount_idr"}.issubset(portfolio_df.columns):
        st.plotly_chart(pie_allocation(portfolio_df), use_container_width=True)
    st.subheader("Aggressive Scenario")
    show_table(portfolio_aggressive_df)
    if not portfolio_aggressive_df.empty and {"ticker", "allocated_amount_idr"}.issubset(portfolio_aggressive_df.columns):
        st.plotly_chart(pie_allocation(portfolio_aggressive_df), use_container_width=True)
    if not transaction_cost_df.empty:
        st.subheader("Transaction Cost and Slippage")
        show_table(transaction_cost_df)
    if not scenario_summary_df.empty:
        st.subheader("Scenario Summary")
        show_table(scenario_summary_df)

with tab_risk:
    st.header("Risk Analysis")
    show_table(risk_df)
    if not drawdown_analysis_df.empty:
        st.subheader("Historical Drawdown Analysis")
        show_table(drawdown_analysis_df)
    if not risk_df.empty and {"Ticker", "Risk_Score"}.issubset(risk_df.columns):
        fig = px.bar(risk_df, x="Ticker", y="Risk_Score", color="Risk_Level", title="Risk Score per Saham")
        st.plotly_chart(style_plotly(fig), use_container_width=True)
    if not sector_df.empty:
        st.subheader("Sector Exposure")
        show_table(sector_df)
        if {"sector", "portfolio_weight"}.issubset(sector_df.columns):
            fig = px.bar(sector_df, x="sector", y="portfolio_weight", title="Bobot Portofolio per Sektor")
            st.plotly_chart(style_plotly(fig), use_container_width=True)
    if not stress_df.empty:
        st.subheader("Stress Test Historis")
        show_table(stress_df)
        if {"scenario", "portfolio_return", "equal_weight_universe_return"}.issubset(stress_df.columns):
            stress_long = stress_df.melt(
                id_vars=["scenario"],
                value_vars=["portfolio_return", "equal_weight_universe_return"],
                var_name="series",
                value_name="return",
            )
            fig = px.bar(stress_long, x="scenario", y="return", color="series", barmode="group", title="Stress Test vs Equal-Weight Universe")
            st.plotly_chart(style_plotly(fig), use_container_width=True)
    st.subheader("Stock Selection Matrix")
    show_table(load_csv("stock_selection_matrix.csv"))
