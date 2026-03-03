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

# Mengukur Jarak

## Similarity vs Dissimilarity

Dalam data mining, kita perlu mengukur kedekatan antar objek data:
* **Similarity (Kemiripan):** Ukuran numerik yang menunjukkan seberapa serupa dua objek. Semakin tinggi nilainya, semakin mirip objek tersebut (Range $[0,1]$).
* **Dissimilarity (Ketidakmiripan):** Ukuran numerik yang menunjukkan perbedaan antar objek. Semakin rendah nilainya, semakin mirip objek tersebut (Minimum $0$).



---

## Pengukuran Jarak Berdasarkan Tipe Atribut

### Atribut Numerik (Minkowski Distance)

Formula umum untuk menghitung jarak pada data numerik:

$$
d(i, j) = \sqrt[h]{\sum_{f=1}^{p} |x_{if} - x_{jf}|^h}
$$

Beberapa kasus khusus dari Minkowski Distance adalah sebagai berikut.

Dataset contoh (dua data pertama Iris):

- $x_1 = (5.1, 3.5, 1.4, 0.2)$  
- $x_2 = (4.9, 3.0, 1.4, 0.2)$  

---

### 1. Manhattan Distance ($h=1$)

Rumus:

$$
d(i,j) = \sum_{f=1}^{p} |x_{if} - x_{jf}|
$$

Perhitungan:

$$
|5.1 - 4.9| = 0.2
$$
$$
|3.5 - 3.0| = 0.5
$$
$$
|1.4 - 1.4| = 0
$$
$$
|0.2 - 0.2| = 0
$$

$$
d = 0.2 + 0.5 + 0 + 0 = \mathbf{0.7}
$$

#### Dissimilarity Matrix (Manhattan)

Karena hanya ada dua objek, maka matriks ketidakmiripan berbentuk:

$$
D =
\begin{bmatrix}
0 & 0.7 \\
0.7 & 0
\end{bmatrix}
$$

![manhattan](img/DMatriksManhattan.png)

**Penjelasan:**  
- Elemen diagonal bernilai 0 karena jarak suatu objek dengan dirinya sendiri adalah 0.  
- Matriks bersifat simetris ($d(i,j) = d(j,i)$).  
- Nilai 0.7 menunjukkan tingkat ketidakmiripan antara $x_1$ dan $x_2$ berdasarkan Manhattan Distance.

---

### 2. Euclidean Distance ($h=2$)

Rumus:

$$
d(i,j) = \sqrt{\sum_{f=1}^{p} (x_{if} - x_{jf})^2}
$$

Perhitungan:

$$
(0.2)^2 = 0.04
$$
$$
(0.5)^2 = 0.25
$$
$$
(0)^2 = 0
$$
$$
(0)^2 = 0
$$

$$
0.04 + 0.25 + 0 + 0 = 0.29
$$

$$
d = \sqrt{0.29} \approx \mathbf{0.538}
$$

#### Dissimilarity Matrix (Euclidean)

$$
D =
\begin{bmatrix}
0 & 0.538 \\
0.538 & 0
\end{bmatrix}
$$

![euclidean](img/DmatriksEuclidean.png)

**Penjelasan:**  
- Matriks tetap simetris dan diagonal bernilai nol.  
- Nilai 0.538 lebih kecil dibanding Manhattan karena metode Euclidean menggunakan akar kuadrat dari jumlah kuadrat selisih.  
- Euclidean Distance sering digunakan dalam clustering seperti K-Means.

---

### 3. Minkowski Distance ($h=3$)

Rumus:

$$
d(i, j) = \sqrt[3]{\sum_{f=1}^{p} |x_{if} - x_{jf}|^3}
$$

Perhitungan:

$$
|0.2|^3 = 0.008
$$
$$
|0.5|^3 = 0.125
$$
$$
|0|^3 = 0
$$
$$
|0|^3 = 0
$$

$$
0.008 + 0.125 + 0 + 0 = 0.133
$$

$$
d = \sqrt[3]{0.133} \approx \mathbf{0.510}
$$

#### Dissimilarity Matrix (Minkowski, $h=3$)

$$
D =
\begin{bmatrix}
0 & 0.510 \\
0.510 & 0
\end{bmatrix}
$$

**Penjelasan:**  
- Nilai jarak bergantung pada parameter $h$.  
- Semakin besar $h$, jarak akan semakin dipengaruhi oleh selisih terbesar antar atribut.  
- Matriks tetap simetris dan digunakan sebagai input dalam algoritma clustering berbasis jarak.

---

### 4. Supremum Distance ($h \to \infty$)

Rumus:

$$
d(i,j) = \max_{f} |x_{if} - x_{jf}|
$$

Perhitungan:

$$
\max(0.2, 0.5, 0, 0) = \mathbf{0.5}
$$

#### Dissimilarity Matrix (Supremum)

$$
D =
\begin{bmatrix}
0 & 0.5 \\
0.5 & 0
\end{bmatrix}
$$

**Penjelasan:**  
- Supremum Distance hanya mempertimbangkan selisih terbesar antar atribut.  
- Matriks tetap simetris dengan diagonal bernilai nol.  
- Cocok digunakan ketika fokus analisis hanya pada perbedaan maksimum antar fitur.


### Atribut Biner (Binary)
Untuk variabel biner tidak simetris (*asymmetric binary*), kita menggunakan **Jaccard Coefficient**:
$$Sim_{Jaccard} = \frac{q}{q + r + s}$$
Di mana:
- $q$: jumlah atribut yang keduanya bernilai 1.
- $r$: jumlah atribut objek $i$ bernilai 1, objek $j$ bernilai 0.
- $s$: jumlah atribut objek $i$ bernilai 0, objek $j$ bernilai 1.

### Cosine Similarity
Sering digunakan untuk membandingkan kemiripan antar dokumen (vektor frekuensi kata):
$$\cos(d_1, d_2) = \frac{d_1 \cdot d_2}{||d_1|| ||d_2||}$$

---

## Implementasi Perhitungan Jarak dengan Python

Gunakan kode berikut untuk menghitung jarak antar baris pada dataset IRIS:

```{code-cell}
:tags: [hide-input]

import pandas as pd
import numpy as np
from scipy.spatial import distance

# Memuat data
df = pd.read_csv("IRIS.csv")

# Mengambil dua objek data pertama (fitur numerik)
obj1 = df.iloc[0, 0:4].values
obj2 = df.iloc[1, 0:4].values

# Menghitung berbagai jenis jarak
dist_euc = distance.euclidean(obj1, obj2)
dist_man = distance.cityblock(obj1, obj2)
dist_mink = distance.minkowski(obj1, obj2, p=3)

print(f"Objek 1: {obj1}")
print(f"Objek 2: {obj2}")
print("-" * 35)
print(f"Jarak Euclidean (h=2) : {dist_euc:.4f}")
print(f"Jarak Manhattan (h=1) : {dist_man:.4f}")
print(f"Jarak Minkowski (h=3) : {dist_mink:.4f}")
```

## Atribut Ordinal

Atribut ordinal mirip dengan atribut nominal, namun memiliki urutan (*ranking*) yang bermakna (contoh: Kecil, Sedang, Besar).

**Langkah Mengukur Jarak Ordinal:**
1. Mengganti nilai $f$ dengan urutannya (ranking): $r_{if} \in \{1, \dots, M_f\}$.
2. Melakukan normalisasi agar rentang nilai berada di $[0, 1]$ menggunakan formula:
   $$z_{if} = \frac{r_{if} - 1}{M_f - 1}$$
3. Hitung jarak menggunakan **Euclidean Distance** atau metrik numerik lainnya pada nilai $z_{if}$ tersebut.



---

## Atribut Campuran (Mixed Types)

Dalam database nyata, seringkali kita menemukan berbagai tipe data sekaligus (Nominal, Biner, Numerik, Ordinal). Untuk menghitung jarak antar objek dengan atribut campuran, kita menggunakan pembobotan:

$$d(i, j) = \frac{\sum_{f=1}^{p} \delta_{ij}^{(f)} d_{ij}^{(f)}}{\sum_{f=1}^{p} \delta_{ij}^{(f)}}$$

* **Jika $f$ biner/nominal:** $d_{ij}^{(f)} = 0$ jika $x_{if} = x_{jf}$, selain itu $d_{ij}^{(f)} = 1$.
* **Jika $f$ numerik:** Gunakan jarak yang dinormalisasi.
* **Jika $f$ ordinal:** Gunakan rangking yang telah dinormalisasi ($z_{if}$).

---

## Cosine Similarity

Metode ini sangat populer dalam pengolahan teks (*Text Mining*). Dokumen dinyatakan sebagai vektor berdasarkan kemunculan kata-kata (term-frequency).

**Formula Cosine Similarity:**

$$\cos(d_1, d_2) = \frac{d_1 \cdot d_2}{\|d_1\| \|d_2\|}$$

Di mana:
- $d_1 \cdot d_2$: Perkalian titik (*dot product*) dari dua vektor.
- $\|d\|$: Panjang atau norma dari vektor.



**Karakteristik:**
* Jika sudut antar vektor 0°, maka $\cos(0) = 1$ (sangat mirip).
* Jika sudut antar vektor 90°, maka $\cos(90) = 0$ (tidak ada kemiripan).

---