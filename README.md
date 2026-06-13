# UAS KAB - Dashboard Rekomendasi Portofolio Saham

Repository ini berisi aplikasi Streamlit untuk Ujian Akhir Semester mata kuliah Kecerdasan Artifisial pada Bisnis. Aplikasi ini digunakan untuk menampilkan hasil rekomendasi portofolio saham berbasis model forecasting dan analisis pendukung investasi.

Dashboard ini dibuat oleh:

- Nama: Aisya Ulul Asmi
- NIM: 25/564969/PPA/07123
- Mata kuliah: Kecerdasan Artifisial pada Bisnis
- Model: Ensemble TimesFM 2.5 200M PyTorch + RandomForest + CatBoost

## Deskripsi Proyek

Proyek ini bertujuan untuk membantu proses pemilihan saham dan penyusunan rekomendasi portofolio berdasarkan hasil prediksi return, risiko, evaluasi model, dan simulasi alokasi dana. Sistem tidak hanya menampilkan hasil forecasting, tetapi juga menyajikan analisis pendukung seperti benchmark proxy, confidence level, skenario portofolio, sensitivitas biaya transaksi, drawdown, sector exposure, dan stress test.

Dashboard membaca hasil model yang sudah tersedia di folder `outputs/`. Proses training dan forecasting tidak dijalankan ulang saat aplikasi dibuka di Streamlit Cloud, sehingga aplikasi lebih ringan dan fokus pada penyajian hasil akhir.

## Fitur Dashboard

- Executive summary hasil rekomendasi portofolio.
- Rekomendasi final saham yang dipilih.
- Visualisasi historical price dan forecast.
- Evaluasi model menggunakan metrik error dan DSTAT.
- Rekomendasi BUY/HOLD beserta confidence level.
- Simulasi alokasi portofolio untuk skenario conservative, balanced, dan aggressive.
- Analisis risiko, drawdown, sector exposure, stress test, dan transaction cost sensitivity.
- Tabel hasil analisis yang sudah diperbarui dari task terakhir.

## Struktur Repository

```text
.
|-- streamlit_app.py
|-- requirements.txt
|-- README.md
|-- .gitignore
|-- .streamlit/
|   `-- config.toml
|-- components/
|   |-- charts.py
|   |-- metrics.py
|   `-- tables.py
`-- outputs/
    |-- figures/
    `-- tables/
```

Keterangan:

- `streamlit_app.py`: file utama aplikasi Streamlit.
- `components/`: komponen pendukung untuk chart, tabel, dan tampilan metrik.
- `outputs/tables/`: file CSV hasil model dan analisis.
- `outputs/figures/`: gambar visualisasi yang ditampilkan di dashboard.
- `.streamlit/config.toml`: konfigurasi tema dashboard.
- `requirements.txt`: daftar dependency untuk deployment.

## Cara Menjalankan Secara Lokal

Pastikan Python sudah terpasang, lalu jalankan:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Setelah itu buka URL lokal yang muncul di terminal, biasanya:

```text
http://localhost:8501
```

## Cara Deploy ke Streamlit Cloud

1. Upload seluruh isi repository ini ke GitHub.
2. Buka Streamlit Community Cloud.
3. Pilih repository GitHub, misalnya `Aisyaululasmi/UAS_KAB`.
4. Pilih branch yang digunakan, misalnya `main`.
5. Isi main file path dengan:

```text
streamlit_app.py
```

6. Klik deploy.

Jika file `streamlit_app.py` berada di dalam folder tertentu, maka main file path harus mengikuti lokasi folder tersebut. Namun jika file sudah berada langsung di root repository, cukup gunakan `streamlit_app.py`.

## Catatan Model

Model utama yang digunakan adalah ensemble TimesFM 2.5 200M PyTorch, RandomForest, dan CatBoost. TimesFM digunakan sebagai komponen utama forecasting time series, sedangkan RandomForest dan CatBoost digunakan sebagai model pendukung untuk estimasi return, validasi sinyal, dan pembentukan rekomendasi portofolio.
