# Describe Data

## Analisis Statistik Deskriptif

**Describing Data** atau deskripsi data merupakan tahapan krusial dalam analisis untuk menyajikan informasi mentah menjadi sajian yang lebih terstruktur dan mudah dipahami. Proses ini bertujuan untuk merangkum karakteristik utama dari dataset melalui berbagai parameter statistik guna memberikan gambaran menyeluruh sebelum menarik kesimpulan lebih lanjut.

### 1. Mean (Rata-Rata)
Nilai rata-rata hitung dari sekumpulan data.
$$ \bar{x} = \frac{\sum_{i=1}^{n} x_i}{n} $$

### 2. Median (Nilai Tengah)
Nilai yang membagi data menjadi dua bagian yang sama besar setelah diurutkan.
* Jika $n$ ganjil: $Med = x_{(\frac{n+1}{2})}$
* Jika $n$ genap: $Med = \frac{x_{(\frac{n}{2})} + x_{(\frac{n}{2} + 1)}}{2}$

### 3. Modus (Mode)
Nilai yang paling sering muncul atau memiliki frekuensi tertinggi dalam distribusi data.

### 4. Ukuran Sebaran (Quartil & Rentang)
Membagi data terurut menjadi empat bagian sama besar.
* **Quartil ($Q_i$):** Nilai pembatas data (25%, 50%, 75%).
* **Rentang Interquartil (IQR):** Selisih antara Quartil atas dan Quartil bawah.
$$ IQR = Q_3 - Q_1 $$



### 5. Variansi dan Standar Deviasi
Mengukur seberapa jauh sebaran titik data dari nilai rata-rata (*mean*).
* **Variansi ($s^2$):**
$$ s^2 = \frac{\sum_{i=1}^{n} (x_i - \bar{x})^2}{n - 1} $$
* **Standar Deviasi ($s$):**
$$ s = \sqrt{s^2} $$



### 6. Skewness (Kemencengan)
Ukuran ketidaksimetrisan sebuah distribusi. Nilai ini menunjukkan apakah data cenderung menumpuk di sisi kiri atau kanan.
$$ Skewness = \frac{n}{(n-1)(n-2)} \sum \left( \frac{x_i - \bar{x}}{s} \right)^3 $$



---

### Implementasi pada Data `sepal_length`

Berikut adalah kode Python untuk menghitung parameter statistik di atas menggunakan pustaka `pandas` dan `scipy`:

```{code-cell}
:tags: [hide-input]
import pandas as pd
from scipy import stats

# Memuat dataset
df = pd.read_csv("IRIS.csv")

# Perhitungan statistik khusus fitur sepal_length
print("--- STATISTIK SEPAL LENGTH ---")
print("Jumlah data      :", df['sepal_length'].count())
print("Rata-rata        :", df['sepal_length'].mean())
print("Nilai minimal    :", df['sepal_length'].min())
print("Q1 (25%)         :", df['sepal_length'].quantile(0.25))
print("Q2 (Median)      :", df['sepal_length'].quantile(0.5))
print("Q3 (75%)         :", df['sepal_length'].quantile(0.75))
print("Nilai Maksimal   :", df['sepal_length'].max())

# Menghitung Modus menggunakan scipy
mode_res = stats.mode(df['sepal_length'], keepdims=True)
print(f"Nilai Modus      : {mode_res.mode[0]} (Muncul {mode_res.count[0]} kali)")

# Variansi, Standar Deviasi, dan Skewness
print("Kemencengan      :", "{0:.6f}".format(round(df['sepal_length'].skew(), 6)))
print("Standar Deviasi  :", "{0:.2f}".format(round(df['sepal_length'].std(), 2)))
print("Variansi         :", "{0:.2f}".format(round(df['sepal_length'].var(), 2)))
```

---

### Output Statistik (Dataset Iris)

Berdasarkan eksekusi kode di atas, berikut adalah ringkasan nilai statistik untuk kolom `sepal_length`:

| Atribut | Nilai Statistik | Keterangan |
| :--- | :--- | :--- |
| **Jumlah Data** | 150 | Data lengkap, tidak ada *missing value*. |
| **Mean** | 5.84 | Rata-rata panjang kelopak bunga. |
| **Median** | 5.80 | Nilai tengah distribusi data. |
| **Modus** | 5.0 | Nilai yang paling banyak muncul (10 kali). |
| **Q1 (25%)** | 5.10 | 25% data memiliki nilai di bawah 5.1. |
| **Q3 (75%)** | 6.40 | 75% data memiliki nilai di bawah 6.4. |
| **Standar Deviasi** | 0.83 | Sebaran data cukup rapat di sekitar rata-rata. |
| **Skewness** | 0.314911 | Menceng positif ringan (data cenderung simetris). |

---