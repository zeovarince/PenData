# K-Nearest Neighbors (KNN) Imputation
## Penanganan Missing Values: K-Nearest Neighbors (KNN) Imputation

Setelah data dinormalisasi agar memiliki skala yang seimbang, kita sekarang dapat menangani masalah *missing values* (data yang hilang). Salah satu metode yang paling cerdas dan akurat untuk imputasi adalah menggunakan **K-Nearest Neighbors (KNN)**.

## Konsep Dasar KNN

Ide dasar dari KNN sangat intuitif: *"Beritahu saya siapa tetangga terdekat Anda, maka saya akan tahu siapa Anda."*

Jika sebuah data kehilangan nilai pada fitur tertentu (misal: JMK), KNN akan mencari sejumlah **$K$ data lain yang paling mirip (terdekat)** dengannya berdasarkan fitur-fitur lain yang tersedia (seperti IPK dan PO). Nilai yang hilang kemudian diisi menggunakan rata-rata dari nilai-nilai tetangga terdekat tersebut.

## Kelebihan vs Kekurangan KNN Imputation

| Aspek         | KNN Imputer                   | Mean/Median Imputation  |
| ------------- | ----------------------------- | ----------------------- |
| Akurasi       | Tinggi (konteks-aware)        | Rendah (global average) |
| Kompleksitas  | O(n²) lambat untuk data besar | O(1) sangat cepat       |
| Memori        | Butuh seluruh dataset         | Hanya statistik kolom   |
| Preprocessing | Wajib normalisasi             | Tidak perlu             |


## Tujuan, Kapan, dan Mengapa Menggunakan KNN Imputation

**1. Tujuan Menggunakan KNN Imputation:**
Tujuan utama metode ini adalah untuk memperkirakan dan mengisi nilai yang hilang (*missing value*) pada sebuah data dengan memanfaatkan informasi dari tetangga terdekatnya. Alih-alih menebak secara buta, KNN mencoba merekonstruksi nilai yang hilang berdasarkan kemiripan karakteristik multidimensional dari data tersebut terhadap data lainnya di dalam dataset.

**2. Kapan Harus Menggunakan KNN Imputation?**
* **Setelah Data Dinormalisasi:** Ini adalah **syarat mutlak**. Karena KNN bekerja berdasarkan perhitungan jarak (seperti *Euclidean*), fitur dengan skala besar akan mendominasi perhitungan jika data tidak dinormalisasi terlebih dahulu.
* **Terdapat Pola Relasi Antar Fitur:** Digunakan ketika kamu yakin bahwa fitur-fitur dalam dataset memiliki hubungan satu sama lain (misalnya, ada korelasi logis antara IPK, Penghasilan Orang Tua, dan Jumlah Mata Kuliah).
* **Missing Value Bersifat Acak (MAR / MCAR):** Sangat efektif jika data yang hilang terjadi secara acak (*Missing at Random*), bukan karena suatu pola kesalahan sistematis yang disengaja.

**3. Mengapa Menggunakan KNN (Dibandingkan Metode Lain)?**
Cara paling tradisional untuk mengisi data yang kosong adalah menggunakan nilai **Rata-rata (Mean)** atau **Nilai Tengah (Median)** dari kolom tersebut. Namun, metode tradisional ini memiliki kelemahan fatal: *mereka mengabaikan konteks*.

* **Contoh Kasus:** Misalkan kita memiliki data kosong pada fitur "Gaji" untuk seseorang yang berumur 22 tahun dan baru lulus kuliah. Jika kita menggunakan metode *Mean*, sistem mungkin akan mengisi gajinya dengan Rp 10.000.000 (karena rata-rata gaji seluruh karyawan di perusahaan termasuk direktur). Ini tentu tidak masuk akal.
* **Keunggulan KNN:** Dengan KNN, algoritma akan melihat *konteks*. KNN akan mencari tetangga terdekat (misalnya karyawan lain yang berumur mirip dan memiliki tingkat pendidikan yang sama). KNN kemudian akan mengambil rata-rata gaji dari *tetangga-tetangga yang mirip tersebut*, misalnya Rp 4.000.000. Hasil prediksi ini jauh lebih akurat, logis, dan menjaga variansi asli dari dataset.

## Rumus Perhitungan Jarak (Euclidean Distance)

Untuk mencari "tetangga terdekat", kita menghitung jarak antar data. Rumus jarak yang paling umum adalah **Jarak Euclidean**:

$$d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2 + ... + (n_2 - n_1)^2}$$

*Di mana $d$ adalah jarak, dan $x, y, n$ adalah fitur-fitur yang digunakan.*

---

## Perhitungan Manual KNN (Studi Kasus Dataset Mahasiswa)

Agar perhitungan jarak akurat dan tidak didominasi oleh fitur yang besar, kita **WAJIB** menggunakan data yang sudah dinormalisasi (misalnya hasil **Min-Max Scaling** sebelumnya).

**Tabel Data (Fitur X Menggunakan Hasil Min-Max, Target Y menggunakan Nilai Asli JMK):**

| ID | IPK (Norm) | PO (Norm) | JMK (Asli) |
|:---:|:---:|:---:|:---:|
| 1 | 0.0 | 0.0 | 2 |
| 2 | 0.5 | 0.5 | 3 |
| 3 | 1.0 | 0.0 | 2 |
| 4 | 0.0 | 0.0 | 3 |
| 5 | 0.5 | 0.5 | 2 |
| 6 | 1.0 | 1.0 | 3 |
| **7** | **0.0** | **0.5** | **?** |

Kita ingin mencari nilai JMK untuk **ID 7**, dan kita tentukan **$K = 3$** (mencari 3 tetangga terdekat).

**Langkah 1: Hitung Jarak Euclidean dari ID 7 ke semua data lain**
* $d(7, 1) = \sqrt{(0.0 - 0.0)^2 + (0.5 - 0.0)^2} = \sqrt{0 + 0.25} = 0.5$
* $d(7, 2) = \sqrt{(0.0 - 0.5)^2 + (0.5 - 0.5)^2} = \sqrt{0.25 + 0} = 0.5$
* $d(7, 3) = \sqrt{(0.0 - 1.0)^2 + (0.5 - 0.0)^2} = \sqrt{1.0 + 0.25} = \sqrt{1.25} \approx 1.118$
* $d(7, 4) = \sqrt{(0.0 - 0.0)^2 + (0.5 - 0.0)^2} = \sqrt{0 + 0.25} = 0.5$
* $d(7, 5) = \sqrt{(0.0 - 0.5)^2 + (0.5 - 0.5)^2} = \sqrt{0.25 + 0} = 0.5$
* $d(7, 6) = \sqrt{(0.0 - 1.0)^2 + (0.5 - 1.0)^2} = \sqrt{1.0 + 0.25} = \sqrt{1.25} \approx 1.118$

**Langkah 2: Tentukan K Tetangga Terdekat (K=3)**
Jarak terkecil adalah $0.5$. Karena ada empat data dengan jarak $0.5$ (ID 1, 2, 4, 5), kita bisa mengambil tiga yang pertama sebagai tetangga:
1.  **Tetangga 1:** ID 1 (Jarak = 0.5, JMK = 2)
2.  **Tetangga 2:** ID 2 (Jarak = 0.5, JMK = 3)
3.  **Tetangga 3:** ID 4 (Jarak = 0.5, JMK = 3)

**Langkah 3: Hitung Rata-rata Nilai Tetangga**
$$JMK_{prediksi} = \frac{JMK_1 + JMK_2 + JMK_4}{3}$$
$$JMK_{prediksi} = \frac{2 + 3 + 3}{3} = \frac{8}{3} \approx 2.67$$

Jadi, berdasarkan perhitungan KNN ($K=3$), nilai JMK untuk data ID 7 yang hilang diimputasi menjadi **2.67**.

---

## Implementasi Kode Python (KNN Imputation)

Di Python, kita bisa menggunakan modul bawaan `KNNImputer` dari `scikit-learn` yang sudah dioptimalkan untuk pekerjaan ini.

```{code-cell} ipython3
import pandas as pd
from sklearn.impute import KNNImputer

# 1. Menyiapkan data (menggunakan data yang SUDAH dinormalisasi Min-Max)
data = {
    'IPK_Norm': [0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.0],
    'PO_Norm': [0.0, 0.5, 0.0, 0.0, 0.5, 1.0, 0.5],
    'JMK': [2, 3, 2, 3, 2, 3, None] # None mewakili missing value
}
df_knn = pd.DataFrame(data)

print("Data Sebelum Imputasi:")
display(df_knn)

# 2. Proses KNN Imputation
# n_neighbors adalah K. Kita set K=3
imputer = KNNImputer(n_neighbors=3)

# Lakukan imputasi dan kembalikan ke dalam DataFrame
df_imputed_array = imputer.fit_transform(df_knn)
df_hasil_knn = pd.DataFrame(df_imputed_array, columns=df_knn.columns)

print("\nData Setelah Imputasi KNN (Perhatikan baris indeks 6):")
display(df_hasil_knn)
```

---