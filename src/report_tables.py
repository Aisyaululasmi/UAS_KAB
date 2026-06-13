from pathlib import Path
import pandas as pd

from .config import OUTPUTS_DIR, REPORTS_DIR, TOTAL_CAPITAL_IDR


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_Tidak ada data._"
    view = df.copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.6g}")
    headers = list(view.columns)
    rows = view.astype(str).values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def generate_report(
    historical_summary: pd.DataFrame,
    model_metrics: pd.DataFrame,
    stock_ranking: pd.DataFrame,
    portfolio: pd.DataFrame,
    recommendation: pd.DataFrame,
    timesfm_note: str,
    classifier_metrics: pd.DataFrame | None = None,
    validation_summary: pd.DataFrame | None = None,
    benchmark_comparison: pd.DataFrame | None = None,
    stress_tests: pd.DataFrame | None = None,
    sector_exposure: pd.DataFrame | None = None,
) -> str:
    total_profit = portfolio["expected_profit_idr"].sum()
    port_return = (portfolio["final_weight"] * portfolio["expected_return_6m"]).sum()
    selected = ", ".join(recommendation["ticker"].tolist())
    text = f"""# Laporan UAS Kecerdasan Artifisial pada Bisnis

## Analisis Pemilihan 5 Saham Warren Buffett Portfolio

Pipeline ini memilih 5 saham dari AAPL, AXP, KO, BAC, CVX, MCO, OXY, CB, KHC, dan GOOGL menggunakan ensemble TimesFM 2.5, RandomForestRegressor, CatBoostRegressor, dan CatBoostClassifier.

## Ringkasan Eksekutif

- Modal investasi: Rp{TOTAL_CAPITAL_IDR:,.0f}
- Lima saham terpilih: {selected}
- Expected return portofolio 6 bulan: {port_return:.2%}
- Proyeksi keuntungan 6 bulan: Rp{total_profit:,.0f}
- Status TimesFM: {timesfm_note}

## Ringkasan Data Historis

{_markdown_table(historical_summary)}

## Evaluasi Model

{_markdown_table(model_metrics)}

## Evaluasi CatBoostClassifier

{_markdown_table(classifier_metrics if classifier_metrics is not None else pd.DataFrame())}

## Validasi Walk-Forward

Validasi tambahan dilakukan dengan pendekatan walk-forward tahunan. Model dilatih menggunakan data sebelum tahun validasi, lalu diuji pada data tahun validasi. Validasi ini membantu melihat apakah performa model stabil pada beberapa periode, bukan hanya pada satu test set terakhir.

{_markdown_table(validation_summary if validation_summary is not None else pd.DataFrame())}

## Benchmark Comparison

Karena pipeline ini tidak mengambil benchmark eksternal seperti SPY, benchmark internal yang digunakan adalah equal-weight dari 10 saham kandidat. Tujuannya adalah melihat apakah portofolio terpilih memberi expected return lebih baik daripada membeli seluruh kandidat dengan bobot sama.

{_markdown_table(benchmark_comparison if benchmark_comparison is not None else pd.DataFrame())}

## Visualisasi Aktual vs Prediksi dan Forecast 120 Hari

Setiap saham divisualisasikan dalam dua panel. Panel kiri menampilkan aktual vs prediksi pada data test, sedangkan panel kanan menampilkan full history dan forecast 120 trading days dengan confidence band +/-3%.

![Forecast Panel AAPL](figures/forecast_panel_AAPL.png)
![Forecast Panel AXP](figures/forecast_panel_AXP.png)
![Forecast Panel KO](figures/forecast_panel_KO.png)
![Forecast Panel BAC](figures/forecast_panel_BAC.png)
![Forecast Panel CVX](figures/forecast_panel_CVX.png)
![Forecast Panel MCO](figures/forecast_panel_MCO.png)
![Forecast Panel OXY](figures/forecast_panel_OXY.png)
![Forecast Panel CB](figures/forecast_panel_CB.png)
![Forecast Panel KHC](figures/forecast_panel_KHC.png)
![Forecast Panel GOOGL](figures/forecast_panel_GOOGL.png)

Interpretasi singkat: panel ini membantu membandingkan apakah model hanya kuat pada angka metrik atau juga dapat mengikuti pola harga secara visual. Forecast band +/-3% menunjukkan ruang ketidakpastian sederhana sehingga keputusan investasi tidak dibaca sebagai prediksi pasti.

## Ranking Saham

{_markdown_table(stock_ranking)}

## Alokasi Portofolio

{_markdown_table(portfolio)}

## Sector Exposure

{_markdown_table(sector_exposure if sector_exposure is not None else pd.DataFrame())}

## Stress Test Historis

Stress test ini menggunakan beberapa periode historis yang tersedia pada dataset, seperti proxy COVID crash, proxy rate shock 2022, dan recent test window. Tujuannya bukan memprediksi krisis masa depan, tetapi memberi gambaran sensitivitas portofolio pada periode pasar yang menekan.

{_markdown_table(stress_tests if stress_tests is not None else pd.DataFrame())}

## Rekomendasi Final

{_markdown_table(recommendation)}

## Catatan Risiko

Hasil ini merupakan simulasi akademik dan bukan nasihat investasi. Keputusan aktual perlu mempertimbangkan fundamental perusahaan, kondisi makroekonomi, risiko nilai tukar USD/IDR, biaya transaksi, pajak, dan sentimen pasar.
"""
    for path in [OUTPUTS_DIR / "laporan_uas.md", REPORTS_DIR / "laporan_uas.md"]:
        Path(path).write_text(text, encoding="utf-8")
    return text
