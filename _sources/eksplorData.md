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

# Eksplorasi Data IRIS

Eksplorasi data merupakan langkah awal yang krusial dalam analisis data. Tahap ini bertujuan untuk memahami karakteristik, struktur, pola, dan kualitas dari suatu dataset sebelum melanjutkan ke pemodelan atau analisis yang lebih mendalam. 

Melalui eksplorasi ini, kita dapat mengenali isi data secara menyeluruh untuk menentukan metode analisis yang paling sesuai. Dalam catatan ini, saya menggunakan studi kasus **Dataset IRIS Flower**.

> **Sumber Data:** [Download IRIS.csv](./IRIS.csv)

## Persiapan Library

Sebelum memulai analisis, kita perlu memastikan pustaka (*library*) yang dibutuhkan sudah terpasang:

```{code-cell}
---
tags: [remove-output]
---
%pip install pandas seaborn matplotlib
```

## Pratinjau Dataset

Dataset IRIS Flower terdiri dari 5 fitur dengan total 150 baris data. Berikut adalah tampilan data yang telah dimuat:

```{code-cell}
:tags: [hide-input]
import pandas as pd

# Membaca data
df = pd.read_csv("IRIS.csv")
# Mengatur index agar mulai dari 1 
df.index = df.index + 1
# Menampilkan data
df
```

---
## Dimensi Data

Dimensi data memberitahu kita seberapa besar dataset yang sedang kita hadapi (Jumlah Baris vs Jumlah Kolom).

```{code-cell}
df.shape
```
## Struktur dan Tipe Data

Kita perlu memastikan setiap kolom memiliki tipe data yang sesuai (misalnya: angka sebagai float, label sebagai object/string).

```{code-cell}
df.info()
```
- Sepal/Petal Length & Width: Berupa float64 (Desimal), sudah tepat untuk perhitungan statistik.
- Species: Berupa object (Kategorikal), sebagai target klasifikasi.

## Pengecekan Data Duplikat

Data yang ganda dapat menyebabkan bias pada model.

```{code-cell}
df.duplicated().sum()
```

## Pengecekan Missing Value (Data Kosong)

Data yang kosong bisa merusak akurasi model. Kita harus memastikan setiap sel terisi.

```{code-cell}
df.isnull().sum()
```

Analisis: Jika hasilnya 0 untuk semua kolom, berarti dataset IRIS ini sangat bersih dan tidak memerlukan tahap pengisian data kosong (imputation).


## Deteksi Outlier (Pencilan Data)

Untuk mendeteksi outlier kita menggunakan tools orange data mining, dengan langkah sebagai berikut:
1. Buka Orange Data Mining.
2. Pilih widget "File" untuk memuat dataset IRIS.csv.
3. Sambungkan widget "File" ke widget "Outliers".
4. Lalu berikan 2 data table untuk melihat data yang dianggap outlier dan inlears.

![outlierIRIS](img/outlieriris.png)

Diatas adalah hasil dari deteksi outlier dengan menggunakan Orange data mining, dimana terdapat 15 data yang dianggap sebagai outlier, dan 135 data yang dianggap inlears. Namun, untuk dataset IRIS ini, kita tidak akan menghapus data outlier karena jumlahnya relatif kecil dan dapat memberikan informasi penting tentang variasi dalam data.

## Ringkasan Statistik

Untuk memahami sebaran angka pada setiap fitur numerik, berikut adalah ringkasan statistik deskriptif dari dataset tersebut:

```{code-cell}
:tags: [hide-input]
# Menampilkan ringkasan statistik (Mean, Std Dev, Min, Max, dll)
df.describe()
```

> *Catatan: Gunakan tombol "Click to show" pada sel di atas jika ingin melihat kode Python yang digunakan untuk memproses tabel ini.*

---
## Kesimpulan Eksplorasi Data

Berdasarkan seluruh tahapan eksplorasi data yang telah dilakukan, berikut adalah kesimpulan utamanya:

- **Kualitas dan Integritas Data:**
   * **Data Sangat Bersih:** Tidak ditemukan adanya *missing values* (nilai kosong) pada seluruh fitur.
   * **Dataset Seimbang (*Balanced*):** Dataset memiliki distribusi kelas yang sempurna (masing-masing spesies memiliki 50 sampel).

- **Struktur dan Dimensi:**
   * Dataset memiliki dimensi **150 baris dan 5 kolom**.
   * Seluruh fitur numerik didefinisikan sebagai `float` dan target sebagai `object`.