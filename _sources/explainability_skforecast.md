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

# Explainability pada Model Forecasting dengan Skforecast

> **Sumber Referensi:** [https://skforecast.org/0.15.1/user_guides/explainability.html](https://skforecast.org/0.15.1/user_guides/explainability.html)  
---

## 1. Analisis Prediksi Tentang Apa?

Kasus pada halaman explainability skforecast melakukan **prediksi permintaan listrik (Electricity Demand)** di negara bagian Victoria, Australia.

Dataset yang digunakan adalah **`vic_electricity`** — dataset yang berasal dari paket R `tsibbledata` — yang berisi rekaman setengah-jam (half-hourly) konsumsi listrik selama beberapa tahun. Kolom-kolom utamanya adalah:

| Kolom | Keterangan |
|---|---|
| `Demand` | Permintaan listrik dalam satuan Megawatt (MW) — **target prediksi** |
| `Temperature` | Suhu di Melbourne, ibu kota Victoria (°C) — variabel eksogen |
| `Holiday` | Indikator apakah hari tersebut adalah hari libur nasional |

**Tujuan prediksinya:** Memprediksi berapa besar permintaan listrik pada jam-jam ke depan berdasarkan pola historis konsumsi listrik dan faktor suhu. Setelah model dibangun, analisis dilanjutkan dengan teknik **explainability** untuk menjawab pertanyaan: *"Faktor apa yang paling berpengaruh terhadap prediksi demand listrik?"*

---

## 2. Bentuk Data Training (Input dan Output)

### Konsep Umum

Model yang digunakan adalah `ForecasterAutoreg` dari library skforecast — sebuah **recursive multi-step forecaster**. Model ini menggunakan pendekatan **autoregressive**, yaitu nilai masa lalu dari variabel target digunakan sebagai fitur input untuk memprediksi nilai berikutnya.

### Input (Fitur / Prediktor)

Data training memiliki dua jenis input:

**a) Lag Features (dari variabel target `Demand`)**

Model dikonfigurasi dengan `lags = 7`, artinya 7 nilai historis Demand digunakan sebagai fitur:

| Nama Fitur | Arti |
|---|---|
| `lag_1` | Nilai Demand 1 langkah waktu sebelumnya |
| `lag_2` | Nilai Demand 2 langkah waktu sebelumnya |
| `lag_3` | Nilai Demand 3 langkah waktu sebelumnya |
| `lag_4` | Nilai Demand 4 langkah waktu sebelumnya |
| `lag_5` | Nilai Demand 5 langkah waktu sebelumnya |
| `lag_6` | Nilai Demand 6 langkah waktu sebelumnya |
| `lag_7` | Nilai Demand 7 langkah waktu sebelumnya |

**b) Exogenous Variable (Variabel Eksogen)**

| Nama Fitur | Arti |
|---|---|
| `Temperature` | Suhu saat waktu yang diprediksi |

Jadi total ada **8 kolom input** dalam matriks training (`X_train`): 7 lag + 1 variabel eksogen.

### Output (Target)

| Nama | Arti |
|---|---|
| `y_train` | Nilai `Demand` pada langkah waktu saat ini (yang ingin diprediksi) |

### Ilustrasi Matriks Training

Berikut ilustrasi bagaimana baris data training terbentuk dari deret waktu:

```
Waktu    Demand   → lag_1  lag_2  lag_3  lag_4  lag_5  lag_6  lag_7  Temp  | y (target)
t=8      D8       → D7     D6     D5     D4     D3     D2     D1     T8    | D8
t=9      D9       → D8     D7     D6     D5     D4     D3     D2     T9    | D9
t=10     D10      → D9     D8     D7     D6     D5     D4     D3     T10   | D10
...
```

Matriks training ini dibuat secara otomatis oleh skforecast menggunakan metode:

```python
X_train, y_train = forecaster.create_train_X_y(
    y    = data_train['Demand'],
    exog = data_train['Temperature']
)
```

### Kode Pembuatan dan Pelatihan Model

```python
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from lightgbm import LGBMRegressor
from skforecast.datasets import fetch_dataset
from skforecast.ForecasterAutoreg import ForecasterAutoreg

# Load dataset
data = fetch_dataset(name='vic_electricity')
data = data[['Demand', 'Temperature', 'Holiday']]

# Split data
end_train = '2012-12-31 23:30:00'
data_train = data.loc[:end_train]
data_test  = data.loc[end_train:]

# Buat dan latih forecaster
forecaster = ForecasterAutoreg(
    regressor = LGBMRegressor(random_state=123, verbose=-1),
    lags      = 7
)
forecaster.fit(
    y    = data_train['Demand'],
    exog = data_train['Temperature']
)
```

---

## 3. Apa Itu Lag?

**Lag** adalah nilai historis dari suatu deret waktu yang digunakan sebagai fitur (prediktor) dalam model machine learning untuk forecasting.

### Analogi Sederhana

Bayangkan kamu ingin memprediksi seberapa ramai sebuah warung kopi besok pagi. Secara alami, kamu akan melihat:
- Seberapa ramai warung kopi **kemarin pagi** (lag 1)
- Seberapa ramai warung kopi **2 hari lalu pagi** (lag 2)
- Dan seterusnya...

Itulah lag — kamu *"menoleh ke belakang"* untuk memahami pola, lalu menggunakannya untuk prediksi ke depan.

### Definisi Teknis

Untuk deret waktu $y_t$, lag ke-$k$ didefinisikan sebagai:

$$y_{t-k}$$

Artinya, nilai variabel target pada $k$ langkah waktu sebelum waktu $t$.

### Contoh Konkret pada Kasus Ini

Dengan `lags = 7`, model menggunakan 7 nilai Demand sebelumnya sebagai fitur:

```
y_t  = f(y_{t-1}, y_{t-2}, y_{t-3}, y_{t-4}, y_{t-5}, y_{t-6}, y_{t-7}, Temperature_t)
       ───────────────────────────────────────────────────────────────────────────────
       lag_1    lag_2    lag_3    lag_4    lag_5    lag_6    lag_7    variabel eksogen
```

Karena data bersifat **half-hourly** (30 menit sekali):
- `lag_1` = 30 menit lalu
- `lag_2` = 1 jam lalu
- `lag_7` = 3,5 jam lalu

### Kenapa Lag Penting?

- Deret waktu seringkali **bergantung pada nilainya sendiri di masa lalu** (disebut **autokorelasi**)
- Lag menangkap pola seperti: "jika 1 jam lalu demand tinggi, kemungkinan sekarang juga masih tinggi"
- Pemilihan jumlah lag yang tepat sangat mempengaruhi performa model

---

## 4. Proses Analisis yang Dilakukan

Analisis pada tutorial ini terdiri dari **empat tahap utama**, yaitu:

### Tahap 1: Persiapan Data dan Pelatihan Model

1. **Load dataset** `vic_electricity` menggunakan `fetch_dataset()`
2. **Split data** menjadi data training (sebelum 31 Des 2012) dan data testing (setelahnya)
3. **Buat forecaster** menggunakan `ForecasterAutoreg` dengan regressor `LGBMRegressor` (LightGBM — model berbasis pohon keputusan)
4. **Latih model** dengan memanggil `forecaster.fit()`, menggunakan `Demand` sebagai target dan `Temperature` sebagai variabel eksogen

### Tahap 2: Feature Importance (Bawaan Model)

Metode pertama untuk memahami model adalah **feature importance bawaan** LightGBM.

```python
# Mendapatkan feature importance dari model
feature_importances = forecaster.get_feature_importances()
```

LightGBM menghitung seberapa sering sebuah fitur digunakan sebagai titik split di pohon keputusan. Fitur dengan split lebih banyak dianggap lebih penting.

**Insight yang bisa diperoleh:** Apakah lag terdekat (lag_1, lag_2) lebih berpengaruh daripada lag yang jauh (lag_7)? Apakah suhu lebih penting dari nilai historis demand?

### Tahap 3: Permutation Importance

**Permutation Importance** adalah metode yang lebih robust dibanding feature importance bawaan model. Cara kerjanya:

1. Hitung performa model pada data asli (misalnya MAE)
2. Acak (*permute*) nilai satu fitur secara acak
3. Hitung lagi performa model — performa yang turun = fitur itu penting
4. Ulangi untuk setiap fitur

```python
from sklearn.inspection import permutation_importance

X_train, y_train = forecaster.create_train_X_y(
    y    = data_train['Demand'],
    exog = data_train['Temperature']
)

r = permutation_importance(
    estimator   = forecaster.regressor,
    X           = X_train,
    y           = y_train,
    n_repeats   = 3,
    max_samples = 0.5,
    random_state = 123
)

importances = pd.DataFrame({
    'feature'         : X_train.columns,
    'mean_importance' : r.importances_mean,
    'std_importance'  : r.importances_std
}).sort_values('mean_importance', ascending=False)
```

**Keunggulan:** Tidak bergantung pada struktur model, sehingga lebih adil dan dapat dibandingkan antar model yang berbeda.

### Tahap 4: SHAP Values (SHapley Additive exPlanations)

SHAP adalah metode yang paling komprehensif. Ia menghitung **kontribusi masing-masing fitur terhadap setiap prediksi individu**, bukan hanya rata-rata global.

**Konsep dasar SHAP:** Berasal dari teori permainan (game theory). Setiap fitur dianggap sebagai "pemain" dalam tim, dan SHAP menghitung kontribusi adil setiap pemain terhadap "kemenangan" (hasil prediksi) berdasarkan semua kombinasi fitur yang mungkin.

```python
import shap

shap.initjs()

# Buat explainer untuk model berbasis pohon
explainer   = shap.TreeExplainer(forecaster.regressor)
shap_values = explainer.shap_values(X_train)

# Summary plot — menampilkan distribusi kontribusi setiap fitur
shap.summary_plot(shap_values, X_train)

# Force plot — menjelaskan satu prediksi spesifik
shap.force_plot(
    explainer.expected_value,
    shap_values[0, :],
    X_train.iloc[0, :]
)
```

**Visualisasi yang dihasilkan:**

| Plot | Fungsi |
|---|---|
| **Summary Plot** | Menampilkan distribusi nilai SHAP semua fitur di seluruh data — seberapa besar dan ke arah mana kontribusinya |
| **Force Plot** | Menjelaskan satu prediksi tunggal — fitur mana yang mendorong prediksi naik atau turun |
| **Dependence Plot** | Hubungan antara nilai satu fitur dan nilai SHAP-nya |

### Tahap 5: Partial Dependence Plot (PDP)

PDP menunjukkan **bagaimana perubahan nilai satu fitur mempengaruhi prediksi**, dengan semua fitur lain dipegang konstan (dirata-ratakan).

```python
from sklearn.inspection import PartialDependenceDisplay

fig, ax = plt.subplots(figsize=(10, 4))
PartialDependenceDisplay.from_estimator(
    estimator = forecaster.regressor,
    X         = X_train,
    features  = ['Temperature', 'lag_1'],
    ax        = ax
)
plt.tight_layout()
plt.show()
```

**Contoh insight dari PDP:**
- Jika grafik PDP untuk `Temperature` berbentuk U, berarti demand listrik tinggi saat suhu sangat panas (AC menyala) maupun sangat dingin (pemanas menyala)
- Jika PDP untuk `lag_1` linear naik, berarti semakin tinggi demand 30 menit lalu, semakin tinggi pula prediksi demand sekarang

---

## Ringkasan Alur Analisis

```
Dataset vic_electricity (Demand, Temperature)
            │
            ▼
    Split Train / Test
            │
            ▼
  ForecasterAutoreg (LGBMRegressor, lags=7)
  Input: lag_1 ... lag_7 + Temperature
  Output: Demand (prediksi)
            │
            ▼
    ┌───────┴────────┐
    │                │
    ▼                ▼
Feature Importance   SHAP Values
(built-in LightGBM)  (TreeExplainer)
    │                │
    ▼                ▼
Permutation      Summary Plot
Importance       Force Plot
    │
    ▼
Partial Dependence Plot (PDP)
```

---

## Kesimpulan

Analisis explainability pada kasus forecasting permintaan listrik Victoria ini mengajarkan bahwa:

1. **Model tidak cukup hanya akurat** — kita perlu memahami *mengapa* model membuat prediksi tersebut
2. **Lag paling dekat** (lag_1, lag_2) cenderung menjadi fitur paling penting karena demand listrik bersifat sangat autokorelasi (nilai sekarang sangat bergantung pada nilai sesaat sebelumnya)
3. **Suhu (Temperature)** juga menjadi prediktor penting karena cuaca panas/dingin mendorong penggunaan AC dan pemanas
4. Teknik seperti **SHAP** membantu membuka "kotak hitam" model ensemble seperti LightGBM, sehingga keputusan model bisa dipercaya dan dijelaskan kepada stakeholder

---

*Referensi: skforecast.org/0.15.1/user_guides/explainability.html*
