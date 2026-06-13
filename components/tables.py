import streamlit as st


DISPLAY_COLUMN_NAMES = {
    "directional_accuracy": "DSTAT",
    "Directional_Accuracy": "DSTAT",
    "Directional Accuracy": "DSTAT",
    "avg_directional_accuracy": "avg_DSTAT",
    "norm_directional_accuracy": "norm_DSTAT",
}


def show_table(df, message="Data belum tersedia."):
    if df.empty:
        st.info(message)
    else:
        df = df.rename(columns=DISPLAY_COLUMN_NAMES)
        row_count = len(df)
        height = min(620, max(180, 38 * min(row_count + 1, 14)))
        st.dataframe(df, use_container_width=True, height=height)
