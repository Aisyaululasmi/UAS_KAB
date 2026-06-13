from html import escape

import pandas as pd
import streamlit as st


DISPLAY_COLUMN_NAMES = {
    "directional_accuracy": "DSTAT",
    "Directional_Accuracy": "DSTAT",
    "Directional Accuracy": "DSTAT",
    "avg_directional_accuracy": "avg_DSTAT",
    "norm_directional_accuracy": "norm_DSTAT",
}


def _format_cell(value):
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:,.4f}"
    if isinstance(value, int):
        return f"{value:,}"
    return escape(str(value))


def _render_html_table(df: pd.DataFrame) -> str:
    headers = "".join(f"<th>{escape(str(column))}</th>" for column in df.columns)
    rows = []
    for _, row in df.iterrows():
        cells = "".join(f"<td>{_format_cell(value)}</td>" for value in row)
        rows.append(f"<tr>{cells}</tr>")

    return f"""
    <div class="light-table-wrap">
        <table class="light-table">
            <thead><tr>{headers}</tr></thead>
            <tbody>{''.join(rows)}</tbody>
        </table>
    </div>
    """


def show_table(df, message="Data belum tersedia."):
    if df.empty:
        st.info(message)
    else:
        df = df.rename(columns=DISPLAY_COLUMN_NAMES)
        st.markdown(
            """
            <style>
            .light-table-wrap {
                background: #ffffff !important;
                border: 1px solid #d8e0ea;
                border-radius: 8px;
                margin: .45rem 0 1rem;
                max-height: 620px;
                overflow: auto;
                width: 100%;
            }

            .light-table {
                background: #ffffff !important;
                border-collapse: collapse;
                color: #111111 !important;
                font-size: .88rem;
                min-width: 100%;
                width: max-content;
            }

            .light-table thead th {
                background: #eef6ff !important;
                border-bottom: 1px solid #cfd9e6;
                color: #111111 !important;
                font-weight: 700;
                position: sticky;
                top: 0;
                z-index: 2;
            }

            .light-table th,
            .light-table td {
                background: #ffffff !important;
                border-bottom: 1px solid #edf1f6;
                color: #111111 !important;
                padding: .5rem .65rem;
                text-align: left;
                vertical-align: top;
                white-space: nowrap;
            }

            .light-table tbody tr:nth-child(even) td {
                background: #f8fafc !important;
            }

            .light-table tbody tr:hover td {
                background: #eef6ff !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(_render_html_table(df), unsafe_allow_html=True)
