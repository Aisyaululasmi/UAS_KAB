import streamlit as st


def summary_metrics(total_capital, portfolio_df):
    total_profit = portfolio_df["expected_profit_idr"].sum() if "expected_profit_idr" in portfolio_df else 0
    selected_count = len(portfolio_df)
    expected_return = 0
    if {"final_weight", "expected_return_6m"}.issubset(portfolio_df.columns):
        expected_return = (portfolio_df["final_weight"] * portfolio_df["expected_return_6m"]).sum()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Modal Investasi", f"Rp{total_capital:,.0f}")
    col2.metric("Saham Terpilih", selected_count)
    col3.metric("Expected Return", f"{expected_return:.2%}")
    col4.metric("Estimasi Keuntungan", f"Rp{total_profit:,.0f}")
