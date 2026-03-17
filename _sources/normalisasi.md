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

# Normalisasi Data
## Transformasi Data: Normalisasi (Data Normalization)

Dalam dataset dunia nyata, fitur (kolom) sering kali memiliki rentang nilai (*scale*) yang sangat berbeda. Misalnya, fitur "Umur" mungkin berkisar antara 0 hingga 100, sedangkan fitur "Gaji" bisa berkisar dari jutaan hingga miliaran. 

Perbedaan skala ini menjadi masalah besar bagi algoritma *machine learning* yang berbasis perhitungan jarak (seperti WKNN). Fitur dengan rentang angka besar akan mendominasi perhitungan jarak, mengabaikan fitur dengan angka kecil.

Untuk mengatasi ini, kita melakukan **Normalisasi Data**: proses menskalakan nilai-nilai atribut dari data mentah agar jatuh dalam rentang kecil yang spesifik tanpa mengubah karakteristik dasar dari distribusi data tersebut.

Berikut adalah 4 metode normalisasi beserta penerapannya pada dataset mahasiswa.
*(Catatan: Nilai `?` pada kolom **JMK** di baris 7 dianggap sebagai Missing Value atau NaN, dan diabaikan dalam penentuan batas hitung).*

---

## A. Min-Max Scaling (Normalisasi Min-Max)

**Konsep:**
Min-Max Scaling mentransformasikan data secara linear sehingga semua nilai jatuh ke dalam rentang spesifik, umumnya $[0, 1]$. 

**Rumus Matematis:**
$$X_{norm} = \frac{X - X_{min}}{X_{max} - X_{min}}$$

**Tabel Data Awal:**

| ID | IPK | PO | JMK |
|:---:|:---:|:---:|:---:|
| 1 | 2 | 200000 | 2 |
| 2 | 3 | 300000 | 3 |
| 3 | 4 | 200000 | 2 |
| 4 | 2 | 200000 | 3 |
| 5 | 3 | 300000 | 2 |
| 6 | 4 | 400000 | 3 |
| 7 | 2 | 300000 | ? |

**Perhitungan Manual (Contoh untuk Data ID 2):**
* **IPK (Nilai = 3):** $Min = 2$, $Max = 4$
  * $X_{norm} = \frac{3 - 2}{4 - 2} = 0.5$
* **PO (Nilai = 300000):** $Min = 200000$, $Max = 400000$
  * $X_{norm} = \frac{300000 - 200000}{400000 - 200000} = 0.5$

**Tabel Hasil Normalisasi Min-Max:**

| ID | IPK | PO | JMK |
|:---:|:---:|:---:|:---:|
| 1 | 0.0 | 0.0 | 0.0 |
| 2 | 0.5 | 0.5 | 1.0 |
| 3 | 1.0 | 0.0 | 0.0 |
| 4 | 0.0 | 0.0 | 1.0 |
| 5 | 0.5 | 0.5 | 0.0 |
| 6 | 1.0 | 1.0 | 1.0 |
| 7 | 0.0 | 0.5 | NaN |

```{code-cell} ipython3
import pandas as pd
import numpy as np
from IPython.display import display

# Membaca data dan mengubah '?' menjadi NaN
df = pd.read_csv("Book1.xlsx - Sheet1.csv")
df['JMK'] = pd.to_numeric(df['JMK'], errors='coerce')

# Proses Min-Max Scaling manual
df_minmax = df.copy()
kolom_numerik = ['IPK', 'PO', 'JMK']

for col in kolom_numerik:
    x_min = df[col].min()
    x_max = df[col].max()
    df_minmax[col] = (df[col] - x_min) / (x_max - x_min)

print("Hasil Min-Max Scaling:")
display(df_minmax)

```
---

## B. Z-Score Standardization (Standardisasi Z-Score)

**Konsep:**
Berbeda dengan Min-Max yang membatasi data dalam rentang pasti, Z-Score memusatkan data agar memiliki **rata-rata ($\mu$) = 0** dan **standar deviasi ($\sigma$) = 1**. Nilai Z-score merepresentasikan seberapa jauh sebuah nilai menyimpang dari rata-ratanya.

**Rumus Matematis:**
$$Z = \frac{X - \mu}{\sigma}$$

**Kapan Harus Menggunakannya:**
* Jika dataset memiliki banyak *outlier* ekstrem (Z-score jauh lebih kebal terhadap outlier dibandingkan Min-Max).
* Sangat dianjurkan untuk algoritma yang mengasumsikan data terdistribusi normal (seperti Regresi Linear, Regresi Logistik, dan SVM).

**Tabel Data Awal:**

| ID | IPK | PO | JMK |
|:---:|:---:|:---:|:---:|
| 1 | 2 | 200000 | 2 |
| 2 | 3 | 300000 | 3 |
| 3 | 4 | 200000 | 2 |
| 4 | 2 | 200000 | 3 |
| 5 | 3 | 300000 | 2 |
| 6 | 4 | 400000 | 3 |
| 7 | 2 | 300000 | ? |

**Perhitungan Manual (Contoh untuk Data ID 2):**
* **IPK (Nilai = 3):** $\mu = 2.857$, $\sigma = 0.833$
  * $Z = \frac{3 - 2.857}{0.833} \approx 0.1715$
* **PO (Nilai = 300000):** $\mu = 271428.57$, $\sigma = 69980.25$
  * $Z = \frac{300000 - 271428.57}{69980.25} \approx 0.4082$

**Tabel Hasil Normalisasi Z-Score:**

| ID | IPK | PO | JMK |
|:---:|:---:|:---:|:---:|
| 1 | -1.02899 | -1.02062 | -1.0 |
| 2 | 0.17150 | 0.40825 | 1.0 |
| 3 | 1.37199 | -1.02062 | -1.0 |
| 4 | -1.02899 | -1.02062 | 1.0 |
| 5 | 0.17150 | 0.40825 | -1.0 |
| 6 | 1.37199 | 1.83712 | 1.0 |
| 7 | -1.02899 | 0.40825 | NaN |

```{code-cell} ipython3
import pandas as pd

# Membaca data
df = pd.read_csv("Book1.xlsx - Sheet1.csv")
df['JMK'] = pd.to_numeric(df['JMK'], errors='coerce')

df_zscore = df.copy()
kolom_numerik = ['IPK', 'PO', 'JMK']

for col in kolom_numerik:
    mean = df[col].mean()
    # ddof=0 digunakan untuk Standar Deviasi Populasi
    std = df[col].std(ddof=0) 
    df_zscore[col] = (df[col] - mean) / std

print("Hasil Z-Score Standardization:")
display(df_zscore)
```

---
## C. Decimal Scaling

**Konsep:**
Decimal Scaling menskalakan data dengan cara memindahkan letak titik desimal dari nilai atribut. Jumlah pergeseran titik desimal bergantung pada nilai absolut maksimum dari fitur tersebut. Tujuannya adalah membuat nilai absolut maksimum dari seluruh data pada atribut tersebut menjadi kurang dari 1.

**Rumus Matematis:**
$$X_{norm} = \frac{X}{10^j}$$
Di mana $j$ adalah bilangan bulat terkecil sedemikian sehingga $\max(|X_{norm}|) < 1$.

**Kapan Harus Menggunakannya:**
* Sangat berguna ketika kita hanya ingin mengecilkan angka yang sangat besar menjadi desimal yang mudah dibaca secara cepat tanpa komputasi rumit (seperti rata-rata atau standar deviasi).
* Mempertahankan struktur digit angka aslinya (hanya bergeser koma).

**Tabel Data Awal:**

| ID | IPK | PO | JMK |
|:---:|:---:|:---:|:---:|
| 1 | 2 | 200000 | 2 |
| 2 | 3 | 300000 | 3 |
| 3 | 4 | 200000 | 2 |
| 4 | 2 | 200000 | 3 |
| 5 | 3 | 300000 | 2 |
| 6 | 4 | 400000 | 3 |
| 7 | 2 | 300000 | ? |

**Perhitungan Manual (Contoh untuk Data ID 2):**
* **IPK (Nilai = 3):** Nilai mutlak maksimum dari seluruh data IPK adalah $4$. Karena $4 < 10^1$, maka $j = 1$.
  * $X_{norm} = \frac{3}{10^1} = 0.3$
* **PO (Nilai = 300000):** Nilai mutlak maksimum dari seluruh data PO adalah $400000$. Karena $400000 < 10^6$, maka $j = 6$.
  * $X_{norm} = \frac{300000}{10^6} = \frac{300000}{1000000} = 0.3$

**Tabel Hasil Decimal Scaling:**

| ID | IPK | PO | JMK |
|:---:|:---:|:---:|:---:|
| 1 | 0.2 | 0.2 | 0.2 |
| 2 | 0.3 | 0.3 | 0.3 |
| 3 | 0.4 | 0.2 | 0.2 |
| 4 | 0.2 | 0.2 | 0.3 |
| 5 | 0.3 | 0.3 | 0.2 |
| 6 | 0.4 | 0.4 | 0.3 |
| 7 | 0.2 | 0.3 | NaN |

```{code-cell} ipython3
import pandas as pd
import numpy as np

# Membaca data
df = pd.read_csv("Book1.xlsx - Sheet1.csv")
df['JMK'] = pd.to_numeric(df['JMK'], errors='coerce')

df_decimal = df.copy()
kolom_numerik = ['IPK', 'PO', 'JMK']

for col in kolom_numerik:
    max_abs = df[col].abs().max()
    # Mencari nilai pangkat j
    j = np.ceil(np.log10(max_abs)) if max_abs > 0 else 0
    df_decimal[col] = df[col] / (10**j)

print("Hasil Decimal Scaling:")
display(df_decimal)
```

---
## D. Mean Normalization

**Konsep:**
Mirip dengan Min-Max Scaling, namun kita mengurangkan nilai asli dengan nilai rata-ratanya ($\mu$), bukan nilai minimum. Kemudian hasilnya dibagi dengan rentang data (maksimum dikurangi minimum). Hal ini akan memusatkan rata-rata data menjadi 0 dan membuat sebaran datanya merata di sekitar angka nol.

**Rumus Matematis:**
$$X_{norm} = \frac{X - \mu}{X_{max} - X_{min}}$$

**Kapan Harus Menggunakannya:**
* Sering digunakan dalam algoritma *Machine Learning* yang menggunakan *Gradient Descent* atau sistem rekomendasi (misal: *Collaborative Filtering*) di mana menggeser nilai rata-rata menjadi titik tengah (nol) sangat membantu dan mempercepat proses komputasi bobot.
* Cocok sebagai alternatif Min-Max jika kita juga ingin rata-rata data tersentralisasi.

**Tabel Data Awal:**

| ID | IPK | PO | JMK |
|:---:|:---:|:---:|:---:|
| 1 | 2 | 200000 | 2 |
| 2 | 3 | 300000 | 3 |
| 3 | 4 | 200000 | 2 |
| 4 | 2 | 200000 | 3 |
| 5 | 3 | 300000 | 2 |
| 6 | 4 | 400000 | 3 |
| 7 | 2 | 300000 | ? |

**Perhitungan Manual (Contoh untuk Data ID 2):**
* **IPK (Nilai = 3):** $\mu = 2.857$, $Min = 2$, $Max = 4$
  * $X_{norm} = \frac{3 - 2.857}{4 - 2} = \frac{0.143}{2} \approx 0.0714$
* **PO (Nilai = 300000):** $\mu = 271428.57$, $Min = 200000$, $Max = 400000$
  * $X_{norm} = \frac{300000 - 271428.57}{400000 - 200000} = \frac{28571.43}{200000} \approx 0.1428$

**Tabel Hasil Mean Normalization:**

| ID | IPK | PO | JMK |
|:---:|:---:|:---:|:---:|
| 1 | -0.42857 | -0.35714 | -0.5 |
| 2 | 0.07143 | 0.14286 | 0.5 |
| 3 | 0.57143 | -0.35714 | -0.5 |
| 4 | -0.42857 | -0.35714 | 0.5 |
| 5 | 0.07143 | 0.14286 | -0.5 |
| 6 | 0.57143 | 0.64286 | 0.5 |
| 7 | -0.42857 | 0.14286 | NaN |

```{code-cell} ipython3
import pandas as pd

# Membaca data
df = pd.read_csv("Book1.xlsx - Sheet1.csv")
df['JMK'] = pd.to_numeric(df['JMK'], errors='coerce')

df_mean_norm = df.copy()
kolom_numerik = ['IPK', 'PO', 'JMK']

for col in kolom_numerik:
    mean = df[col].mean()
    x_min = df[col].min()
    x_max = df[col].max()
    df_mean_norm[col] = (df[col] - mean) / (x_max - x_min)

print("Hasil Mean Normalization:")
display(df_mean_norm)
```