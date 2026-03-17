---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Weighted K-Nearest Neighbors (WKNN) Imputation

## Imputasi Missing Values dengan Weighted K-Nearest Neighbors (WKNN)

Meskipun KNN standar cukup baik, metode tersebut memiliki kelemahan: *semua tetangga dianggap memiliki pengaruh yang sama*, tidak peduli apakah tetangga itu sangat dekat atau berada di batas terjauh.

Untuk memperbaikinya, kita menggunakan **Weighted K-Nearest Neighbors (WKNN)**. Dalam WKNN, setiap tetangga diberikan **bobot (weight)**. Tetangga yang jaraknya lebih dekat akan memiliki pengaruh (bobot) yang lebih besar terhadap nilai prediksi dibandingkan tetangga yang lebih jauh.

## Konsep dan Rumus Matematis WKNN

**1. Menghitung Jarak (Euclidean Distance):**
Sama seperti KNN, kita mencari jarak antar data menggunakan data yang sudah dinormalisasi.
$$d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2 + ... + (n_2 - n_1)^2}$$

**2. Menghitung Bobot (Weight):**
Bobot untuk setiap tetangga dihitung sebagai kebalikan dari kuadrat jaraknya. Semakin kecil jaraknya, semakin besar bobotnya.
$$W_i = \frac{1}{d_i^2}$$

**3. Memprediksi Nilai (Rata-rata Terbobot):**
Nilai yang hilang (Y) dihitung dengan mengalikan target masing-masing tetangga dengan bobotnya, lalu dibagi dengan total bobot.
$$Y_{prediksi} = \frac{\sum_{i=1}^{k} (W_i \cdot Y_i)}{\sum_{i=1}^{k} W_i}$$

---

## Perhitungan Manual WKNN (Studi Kasus Data Mahasiswa)

Kita menggunakan data yang **sudah dinormalisasi Min-Max** untuk fitur IPK dan PO, sedangkan target JMK menggunakan nilai aslinya.

**Tabel Data:**

| ID | IPK (Norm) | PO (Norm) | JMK (Asli) |
|:---:|:---:|:---:|:---:|
| 1 | 0.0 | 0.0 | 2 |
| 2 | 0.5 | 0.5 | 3 |
| 3 | 1.0 | 0.0 | 2 |
| 4 | 0.0 | 0.0 | 3 |
| 5 | 0.5 | 0.5 | 2 |
| 6 | 1.0 | 1.0 | 3 |
| **7** | **0.0** | **0.5** | **?** |

Kita ingin mencari nilai JMK untuk **ID 7** dengan **$K = 3$**.

**Langkah 1: Hitung Jarak Euclidean dari ID 7 (0.0, 0.5)**
* $d(7, 1) = \sqrt{(0.0 - 0.0)^2 + (0.5 - 0.0)^2} = \sqrt{0.25} = 0.5$
* $d(7, 2) = \sqrt{(0.0 - 0.5)^2 + (0.5 - 0.5)^2} = \sqrt{0.25} = 0.5$
* $d(7, 3) = \sqrt{(0.0 - 1.0)^2 + (0.5 - 0.0)^2} \approx 1.118$
* $d(7, 4) = \sqrt{(0.0 - 0.0)^2 + (0.5 - 0.0)^2} = 0.5$
* $d(7, 5) = \sqrt{(0.0 - 0.5)^2 + (0.5 - 0.5)^2} = 0.5$
* $d(7, 6) = \sqrt{(0.0 - 1.0)^2 + (0.5 - 1.0)^2} \approx 1.118$

**Langkah 2: Tentukan K=3 Tetangga Terdekat dan Hitung Bobotnya**
Dari hasil di atas, ada 4 data (ID 1, 2, 4, 5) yang memiliki jarak identik yaitu $0.5$. Algoritma umumnya akan mengambil 3 data pertama yang ditemukan, yaitu **ID 1, ID 2, dan ID 4**.

Kita hitung bobot ($W = \frac{1}{d^2}$) untuk ketiga tetangga tersebut:
* **Tetangga 1 (ID 1):** $W_1 = \frac{1}{0.5^2} = \frac{1}{0.25} = 4$
* **Tetangga 2 (ID 2):** $W_2 = \frac{1}{0.5^2} = \frac{1}{0.25} = 4$
* **Tetangga 3 (ID 4):** $W_3 = \frac{1}{0.5^2} = \frac{1}{0.25} = 4$

**Langkah 3: Hitung Rata-Rata Terbobot (JMK Prediksi)**
Nilai JMK dari tetangga: $Y_1 = 2$, $Y_2 = 3$, $Y_3 = 3$.

$$JMK_{prediksi} = \frac{(W_1 \cdot Y_1) + (W_2 \cdot Y_2) + (W_3 \cdot Y_3)}{W_1 + W_2 + W_3}$$

$$JMK_{prediksi} = \frac{(4 \cdot 2) + (4 \cdot 3) + (4 \cdot 3)}{4 + 4 + 4}$$

$$JMK_{prediksi} = \frac{8 + 12 + 12}{12} = \frac{32}{12} \approx 2.67$$

*(Catatan: Karena dalam dataset spesifik ini jarak ketiga tetangga terdekat ke ID 7 adalah identik (sama-sama 0.5), maka bobotnya pun sama besar. Itulah mengapa hasil akhir WKNN di dataset ini bernilai persis sama dengan KNN standar).*

---

## Implementasi Kode Python (WKNN Imputation)

Pada `scikit-learn`, kita dapat menggunakan modul `KNNImputer` dan mengubah parameter `weights` dari 'uniform' menjadi `'distance'` untuk mengaktifkan algoritma **WKNN**.

```{code-cell} ipython3
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from IPython.display import display

# 1. Memuat Data
df = pd.read_excel("Book1.xlsx", sheet_name="Sheet1")
df['JMK'] = pd.to_numeric(df['JMK'], errors='coerce')

# 2. Normalisasi Data (Hanya IPK dan PO) dengan Min-Max
# Kita harus menormalisasi kolom fitur sebelum menghitung jarak
scaler = MinMaxScaler()
df_scaled = df.copy()
df_scaled[['IPK', 'PO']] = scaler.fit_transform(df[['IPK', 'PO']])

print("Data Setelah Dinormalisasi (Siap Dihitung Jaraknya):")
display(df_scaled)

# 3. Proses Imputasi WKNN
# n_neighbors=3, weights='distance' adalah kunci WKNN
wknn_imputer = KNNImputer(n_neighbors=3, weights='distance')

# Lakukan imputasi dan kembalikan ke dalam DataFrame
df_imputed_array = wknn_imputer.fit_transform(df_scaled[['IPK', 'PO', 'JMK']])
df_hasil_wknn = pd.DataFrame(df_imputed_array, columns=['IPK_Norm', 'PO_Norm', 'JMK_Imputed'])

# Menggabungkan hasil imputasi dengan ID asli
df_final = pd.concat([df[['ID']], df_hasil_wknn], axis=1)

print("\nData Setelah Imputasi WKNN (Fokus pada ID 7):")
display(df_final)
```
> Pada kode di atas, kita melakukan normalisasi pada fitur IPK dan PO terlebih dahulu, kemudian menggunakan `KNNImputer` dengan `weights='distance'` untuk melakukan imputasi berdasarkan WKNN. Hasilnya akan menunjukkan nilai JMK yang sudah diimputasi untuk ID 7 berdasarkan bobot jarak dari tetangga terdekatnya.

## Hasil wknn dengan berbagai metode normalisasi
### A. Hasil WKNN Menggunakan Min-Max Scaling

Pada Min-Max, skala data berada di rentang $0.0$ hingga $1.0$. Posisi ID 7 berada di koordinat $(0.0, 0.5)$.

**1. Perhitungan Jarak Euclidean dari ID 7:**
* $d(7, 1) = \sqrt{(0.0 - 0.0)^2 + (0.5 - 0.0)^2} = 0.5$
* $d(7, 2) = \sqrt{(0.0 - 0.5)^2 + (0.5 - 0.5)^2} = 0.5$
* $d(7, 3) = \sqrt{(0.0 - 1.0)^2 + (0.5 - 0.0)^2} \approx 1.118$
* $d(7, 4) = 0.5$
* $d(7, 5) = 0.5$
* $d(7, 6) \approx 1.118$

**2. Tiga Tetangga Terdekat & Bobot:** Tiga data pertama yang terdekat adalah **ID 1, ID 2, dan ID 4** dengan jarak sama ($0.5$).
Bobot ($W = 1/d^2$): $W_1 = W_2 = W_4 = \frac{1}{0.5^2} = 4$

**3. Prediksi:**
$$JMK_{prediksi} = \frac{(4 \cdot 2) + (4 \cdot 3) + (4 \cdot 3)}{4 + 4 + 4} = \frac{32}{12} \approx \mathbf{2.6667}$$

---

### B. Hasil WKNN Menggunakan Z-Score Standardization (Kejutan!)

Z-Score mendistorsi jarak karena pembagian standar deviasi yang berbeda untuk tiap kolom. Posisi ID 7 berada di $(-1.0289, 0.4082)$. Hal ini mengubah peta jarak secara keseluruhan!

**1. Perhitungan Jarak Euclidean dari ID 7:**
* $d(7, 1) \approx 1.4289$ *(JMK = 2)*
* $d(7, 2) \approx 1.2005$ *(JMK = 3)*
* $d(7, 3) \approx 2.7939$ *(JMK = 2)*
* $d(7, 4) \approx 1.4289$ *(JMK = 3)*
* $d(7, 5) \approx 1.2005$ *(JMK = 2)*
* $d(7, 6) \approx 2.7939$ *(JMK = 3)*

**2. Tiga Tetangga Terdekat & Bobot:** Tetangga terdekatnya berubah menjadi **ID 2, ID 5, dan ID 1** dengan jarak berbeda!
* $W_2 = \frac{1}{(1.2005)^2} \approx 0.6939$
* $W_5 = \frac{1}{(1.2005)^2} \approx 0.6939$
* $W_1 = \frac{1}{(1.4289)^2} \approx 0.4898$

**3. Prediksi:**
$$JMK_{prediksi} = \frac{(0.6939 \cdot 3) + (0.6939 \cdot 2) + (0.4898 \cdot 2)}{0.6939 + 0.6939 + 0.4898} = \frac{4.4491}{1.8776} \approx \mathbf{2.3696}$$

---

### C. Hasil WKNN Menggunakan Decimal Scaling

Pada Decimal Scaling, skala data ID 7 menjadi jauh lebih kecil, yaitu $(0.2, 0.3)$.

**1. Perhitungan Jarak Euclidean dari ID 7:**
* $d(7, 1) = \sqrt{(0.2 - 0.2)^2 + (0.3 - 0.2)^2} = \sqrt{0 + 0.01} = 0.1$
* $d(7, 2) = \sqrt{(0.2 - 0.3)^2 + (0.3 - 0.3)^2} = \sqrt{0.01 + 0} = 0.1$
* $d(7, 3) = \sqrt{(0.2 - 0.4)^2 + (0.3 - 0.2)^2} = \sqrt{0.05} \approx 0.2236$
* $d(7, 4) = 0.1$
* $d(7, 5) = 0.1$
* $d(7, 6) \approx 0.2236$

**2. Tiga Tetangga Terdekat & Bobot:** Jarak sangat kecil, sehingga bobotnya menjadi sangat besar! Tetangga terdekat: **ID 1, ID 2, dan ID 4**.
Bobot ($W = 1/d^2$): $W_1 = W_2 = W_4 = \frac{1}{0.1^2} = \frac{1}{0.01} = 100$

**3. Prediksi:**
$$JMK_{prediksi} = \frac{(100 \cdot 2) + (100 \cdot 3) + (100 \cdot 3)}{100 + 100 + 100} = \frac{800}{300} \approx \mathbf{2.6667}$$

---

### D. Hasil WKNN Menggunakan Mean Normalization

Pada Mean Normalization, rata-rata digeser ke nol. Koordinat ID 7 menjadi $(-0.4285, 0.1428)$.

**1. Perhitungan Jarak Euclidean dari ID 7:**
* $d(7, 1) = \sqrt{(-0.4285 - (-0.4285))^2 + (0.1428 - (-0.3571))^2} = \sqrt{0 + 0.5^2} = 0.5$
* $d(7, 2) = \sqrt{(-0.4285 - 0.0714)^2 + (0.1428 - 0.1428)^2} = \sqrt{(-0.5)^2 + 0} = 0.5$
* $d(7, 3) = \sqrt{(-0.4285 - 0.5714)^2 + (0.1428 - (-0.3571))^2} = \sqrt{(-1.0)^2 + 0.5^2} = \sqrt{1.25} \approx 1.118$
* $d(7, 4) = 0.5$
* $d(7, 5) = 0.5$
* $d(7, 6) \approx 1.118$

**2. Tiga Tetangga Terdekat & Bobot:** Tetangga terdekat: **ID 1, ID 2, dan ID 4**.
Bobot ($W = 1/d^2$): $W_1 = W_2 = W_4 = \frac{1}{0.5^2} = 4$

**3. Prediksi:**
$$JMK_{prediksi} = \frac{(4 \cdot 2) + (4 \cdot 3) + (4 \cdot 3)}{4 + 4 + 4} = \frac{32}{12} \approx \mathbf{2.6667}$$

---

## Kesimpulan

ini membuktikan kaidah emas dalam pemrosesan data:
> **"Algoritma berbasis jarak seperti KNN/WKNN sangat dipengaruhi oleh teknik transformasi data yang digunakan."**

Dalam kasus ini:
1. Normalisasi linier (Min-Max, Decimal, Mean) memberikan komposisi peta jarak yang proporsinya dipertahankan. Bobot jarak berbeda-beda (Min-Max/Mean $W=4$, Decimal $W=100$), namun ketiganya menebak JMK ID 7 adalah **2.67**.
2. Z-Score Standardization merusak proporsi jarak asli dan menemukan tetangga yang *berbeda*, sehingga tebakannya menjadi **2.36**.

## Implementasi Kode Eksekusi 4 Metode Sekaligus

Berikut adalah *script* Python untuk mengotomatiskan komputasi ini:

```{code-cell} ipython3
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

df = pd.read_excel("Book1.xlsx", sheet_name="Sheet1")
df['JMK'] = pd.to_numeric(df['JMK'], errors='coerce')

# Siapkan imputer WKNN
wknn_imputer = KNNImputer(n_neighbors=3, weights='distance')

print("=== HASIL IMPUTASI WKNN (PREDIKSI ID 7) ===")

# 1. Min-Max
df_minmax = df.copy()
df_minmax[['IPK', 'PO']] = MinMaxScaler().fit_transform(df[['IPK', 'PO']])
res_minmax = wknn_imputer.fit_transform(df_minmax[['IPK', 'PO', 'JMK']])
print(f"1. Prediksi JMK (Min-Max)  : {res_minmax[6, 2]:.4f}")

# 2. Z-Score
df_zscore = df.copy()
df_zscore[['IPK', 'PO']] = StandardScaler().fit_transform(df[['IPK', 'PO']])
res_zscore = wknn_imputer.fit_transform(df_zscore[['IPK', 'PO', 'JMK']])
print(f"2. Prediksi JMK (Z-Score)  : {res_zscore[6, 2]:.4f}")

# 3. Decimal Scaling
df_decimal = df.copy()
for col in ['IPK', 'PO']:
    max_abs = df[col].abs().max()
    j = np.ceil(np.log10(max_abs)) if max_abs > 0 else 0
    df_decimal[col] = df[col] / (10**j)
res_decimal = wknn_imputer.fit_transform(df_decimal[['IPK', 'PO', 'JMK']])
print(f"3. Prediksi JMK (Decimal)  : {res_decimal[6, 2]:.4f}")

# 4. Mean Normalization
df_mean = df.copy()
for col in ['IPK', 'PO']:
    df_mean[col] = (df[col] - df[col].mean()) / (df[col].max() - df[col].min())
res_mean = wknn_imputer.fit_transform(df_mean[['IPK', 'PO', 'JMK']])
print(f"4. Prediksi JMK (Mean Norm): {res_mean[6, 2]:.4f}")
```