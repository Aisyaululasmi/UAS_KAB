import plotly.express as px


COLOR_SEQUENCE = ["#1f6feb", "#0f766e", "#f59e0b", "#dc2626", "#7c3aed", "#0891b2", "#64748b"]


def apply_chart_theme(fig):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="#ffffff",
        font={"family": "Arial, sans-serif", "color": "#172033"},
        title={"font": {"size": 19, "color": "#172033"}, "x": 0, "xanchor": "left", "y": 0.98, "yanchor": "top"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.10, "xanchor": "left", "x": 0},
        margin={"l": 24, "r": 24, "t": 115, "b": 38},
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eef2f7", linecolor="#d8e0ea")
    fig.update_yaxes(showgrid=True, gridcolor="#eef2f7", linecolor="#d8e0ea")
    return fig


def line_forecast(df, ticker):
    y_cols = [c for c in ["actual_price", "timesfm_forecast", "rf_forecast", "catboost_forecast", "ensemble_forecast"] if c in df.columns]
    fig = px.line(df, x="date", y=y_cols, title=f"Historical vs Forecast - {ticker}", color_discrete_sequence=COLOR_SEQUENCE)
    fig.update_traces(line={"width": 2})
    return apply_chart_theme(fig)


def bar_return(df):
    color = "signal" if "signal" in df.columns else None
    fig = px.bar(df, x="ticker", y="expected_return_6m", color=color, title="Expected Return 6 Bulan", color_discrete_sequence=COLOR_SEQUENCE)
    return apply_chart_theme(fig)


def pie_allocation(df):
    fig = px.pie(df, names="ticker", values="allocated_amount_idr", title="Bobot Alokasi Portofolio", color_discrete_sequence=COLOR_SEQUENCE)
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return apply_chart_theme(fig)
