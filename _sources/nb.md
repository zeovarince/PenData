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

# Naive Bayes
## Analisis Data Menggunakan Naive Bayes
Tugas ini berfokus pada implementasi algoritma klasifikasi Gaussian Naive Bayes menggunakan library scikit-learn dari Python. Sesuai instruksi, penulisan script tidak dilakukan secara mandiri di IDE biasa, melainkan diintegrasikan ke dalam lingkungan KNIME Analytics Platform dengan memanfaatkan eksekusi melalui node Python Script.

## Karakteristik Dataset
Data yang digunakan untuk pelatihan dan pengujian model adalah dataset ObesityDataSet.csv.
> **Sumber Data:** [Download ObesityDataSet.csv](./ObesityDataSet.csv)
---

| Parameter Utama | Detail Informasi |
|:---|:---|
| **Total Baris Data** | 2.111 baris |
| **Jumlah Fitur (Atribut)** | 16 kolom (Kombinasi data kategorikal dan numerik) |
| **Rincian Fitur** | `Gender`, `Age`, `Height`, `Weight`, `family_history_with_overweight`, `FAVC`, `FCVC`, `NCP`, `CAEC`, `SMOKE`, `CH2O`, `SCC`, `FAF`, `TUE`, `CALC`, `MTRANS` |
| **Target Klasifikasi (Label)** | `NObeyesdad` (Tingkat Obesitas) |
| **Kategori Kelas** | • `Insufficient_Weight`<br>• `Normal_Weight`<br>• `Overweight_Level_I`<br>• `Overweight_Level_II`<br>• `Obesity_Type_I`<br>• `Obesity_Type_II`<br>• `Obesity_Type_III` |
| **Kondisi Distribusi Data** | ✨ **Cukup Seimbang (*Relatively Balanced*)**<br>Sebaran kelas cukup merata dan terdistribusi dengan baik, berada di rentang 272 hingga 351 sampel untuk masing-masing kategorinya. |

--- 
## Preprocessing (Transformasi Data)

Dataset *Obesity* memiliki banyak atribut dengan tipe data campuran (kategorikal berupa teks dan numerik). Oleh karena itu, pada tahap *preprocessing*, dilakukan dua proses utama yaitu **Encoding** dan **Normalisasi** agar data siap diproses oleh model Gaussian Naive Bayes menggunakan `scikit-learn`.

### 1. Proses Encoding Data
Karena algoritma pada Python tidak bisa memproses data berupa teks secara langsung (seperti "Male"/"Female" atau "Yes"/"No"), dilakukan teknik *Label Encoding* menggunakan `LabelEncoder` untuk mengubah data kategorikal menjadi format numerik. Teknik ini tepat digunakan pada fitur bertipe **ordinal** dan **biner** yang memiliki urutan atau hanya dua kategori.

Sebagai contoh representasi:
* **Gender**: `Female` menjadi 0, `Male` menjadi 1
* **SMOKE (Merokok)**: `no` menjadi 0, `yes` menjadi 1
* **NObeyesdad (Target)**: Diubah dari format teks tingkat obesitas menjadi rentang angka 0 hingga 6.

### 2. Proses Normalisasi Data
Setelah semua data berupa angka, skala datanya masih sangat berbeda (contoh: Umur puluhan tahun, Berat Badan seratusan kilogram, Tinggi Badan berupa angka desimal 1.70). Normalisasi dilakukan menggunakan `MinMaxScaler` yang akan mengubah seluruh rentang angka menjadi skala **0 hingga 1**, sehingga distribusi setiap fitur dapat dimodelkan dengan lebih baik oleh Gaussian Naive Bayes melalui estimasi *mean* dan varians masing-masing fitur.

> **Catatan:** Pada Gaussian Naive Bayes, model mengestimasi *mean* dan varians setiap fitur secara **independen** untuk menghitung probabilitas. Berbeda dengan algoritma berbasis jarak seperti KNN, Gaussian NB tidak sensitif terhadap dominasi skala fitur — namun normalisasi tetap disarankan agar asumsi distribusi Gaussian terpenuhi dengan lebih baik.

## Seleksi Atribut untuk Demonstrasi Manual

Seleksi atribut bertujuan untuk menentukan fitur-fitur yang relevan dan akan digunakan dalam proses demonstrasi perhitungan probabilitas Naive Bayes secara manual. Dari 17 atribut yang tersedia, dipilih 4 atribut utama yang mewakili empat tipe data yang berbeda (Numerik, Nominal, Ordinal, dan Biner).

### Atribut Yang Digunakan

| No | Atribut  | Tipe Data | Alasan Penggunaan                                          |
| -- | -------- | --------- | ---------------------------------------------------------- |
| 1  | Weight   | Numerik   | Mewakili data angka kontinu dengan rentang nilai yang lebar.|
| 2  | MTRANS   | Nominal   | Menunjukkan alat transportasi tanpa urutan tingkatan.      |
| 3  | CAEC     | Ordinal   | Menunjukkan frekuensi makan dengan tingkatan yang jelas.   |
| 4  | Gender   | Biner     | Mewakili jenis kelamin dengan dua kategori.                |

### Atribut Yang Tidak Digunakan

Atribut lainnya seperti `Age`, `Height`, `family_history`, hingga `NObeyesdad` tidak digunakan dalam demonstrasi manual ini untuk menyederhanakan proses, meskipun dalam analisis komputasional seluruh atribut dapat dilibatkan.

---

## Hasil Transformasi Data

Transformasi bertujuan untuk mengubah seluruh atribut ke dalam bentuk numerik dan memiliki skala yang sebanding (0 sampai 1), sehingga distribusi setiap fitur dapat dimodelkan dengan baik oleh Gaussian Naive Bayes melalui estimasi *mean* dan varians.

### 1. Encoding Data Kategorikal

Encoding dilakukan untuk mengubah atribut bertipe kategori (teks) menjadi angka.

**Gender (Biner) :**
Menggunakan *binary encoding* karena hanya memiliki dua kategori.
- **Female : 1**
- **Male : 0**

**MTRANS (Nominal) :**
Karena `MTRANS` bersifat nominal (tidak memiliki urutan), idealnya digunakan *One-Hot Encoding* untuk menghindari pemberian urutan palsu antar kategori. Namun untuk keperluan demonstrasi manual ini, diberikan label angka unik sebagai representasi sederhana.
- **Public_Transportation : 0**
- **Walking : 1**
- **Automobile : 2**
- **Motorbike : 3**
- **Bike : 4**

**CAEC (Ordinal) :**
Diubah menjadi angka berdasarkan tingkatan intensitasnya (0 sampai 3). *Label Encoding* tepat digunakan di sini karena kategorinya memiliki urutan yang bermakna.
- **no : 0**
- **Sometimes : 1**
- **Frequently : 2**
- **Always : 3**

## Implementasi Alur Kerja (Workflow) KNIME

Proses analisis ini disusun menggunakan *node-based programming* di KNIME Analytics Platform. Berdasarkan desain arsitektur yang dirancang, berikut adalah 5 komponen utama yang digunakan dalam *workflow*:

![Workflow KNIME Naive Bayes](img/naviebayes.png)


1. **CSV Reader**: Berfungsi sebagai *data ingestion* untuk mengimpor file `ObesityDataSet.csv` ke dalam *workspace* KNIME.
2. **Column Filter**: Node ini digunakan untuk menyaring fitur sejak awal. Hanya 5 kolom yang diteruskan ke tahap selanjutnya, yaitu 4 atribut prediktor (`Weight`, `MTRANS`, `CAEC`, `Gender`) dan 1 atribut target (`NObeyesdad`).
3. **Partitioning**: Node ini membagi data yang sudah difilter menjadi dua jalur secara *stratified*. Jalur atas menghasilkan **80% data latih (training)**, dan jalur bawah menghasilkan **20% data uji (testing)**.
4. **Python Script**: Merupakan *engine* komputasi utama. Node ini menerima data latih dan uji, melakukan *preprocessing* (Label Encoding & Normalisasi), melatih model **Gaussian Naive Bayes** menggunakan `scikit-learn`, dan menghasilkan tabel data uji yang telah dilengkapi dengan kolom prediksi.
5. **Scorer**: Diletakkan di akhir alur untuk mengevaluasi performa model. Node ini membandingkan kolom target asli (`NObeyesdad`) dengan kolom prediksi keluaran Python Script untuk menghasilkan nilai Akurasi dan *Confusion Matrix* secara visual.

### 1. Analisis Akurasi (Accuracy Statistics)
![Accuracy Statistics Naive Bayes](img/accstk.png)

Berdasarkan metrik evaluasi `Accuracy Statistics`, model Gaussian Naive Bayes yang dibangun menggunakan 4 atribut terpilih mendapatkan **Overall Accuracy sebesar 0.6028 (60.28%)**. Artinya, dari 423 data uji, model berhasil memprediksi dengan benar sekitar 255 data.

Jika dibedah lebih dalam berdasarkan performa per kelas (Tingkat Obesitas):
* **Performa Tertinggi (Excellent):** Kelas `Obesity_Type_III` memiliki nilai *Recall* 0.985 (98.5%) dan *Precision* 1.0 (100%). Artinya, model hampir tidak pernah meleset sedikitpun dalam mendeteksi seseorang yang berada di kategori obesitas ekstrem ini.
* **Performa Terendah (Poor):** Kelas `Overweight_Level_II` memiliki *Recall* sangat rendah yaitu 0.172 (17.2%). Model sangat kesulitan dan sering salah dalam mengklasifikasikan kelas ini.

### 2. Analisis Confusion Matrix
Confusion Matrix memberikan visualisasi detail mengenai letak kesalahan prediksi (Misclassification) yang dilakukan oleh model:
![Confusion Matrix Naive Bayes](img/cmnb.png)

* **Diagonal Utama (Prediksi Benar / True Positive):** Angka yang membentang dari kiri atas ke kanan bawah (45, 24, 39, 56, 64, 17, 10) mewakili jumlah prediksi yang tepat. Contoh puncaknya ada di baris ke-5, di mana 64 dari 65 data `Obesity_Type_III` berhasil ditebak dengan benar.
* **Sebaran Kesalahan (False Positive / False Negative):** * Pada baris `Overweight_Level_I` (aslinya Overweight 1), terdapat **29 data** yang disalahartikan oleh model sebagai `Obesity_Type_I`.
  * Pada baris `Overweight_Level_II` (aslinya Overweight 2), terdapat **26 data** yang ditebak sebagai `Obesity_Type_I` dan **17 data** ditebak sebagai `Obesity_Type_II`.

### 3. Insight dan Kesimpulan Evaluasi
Mengapa akurasi berada di angka 60.28% dan mengapa model kebingungan di kelas *Overweight*? Hal ini dapat dijelaskan secara analitis:

1. **Efek Penghapusan Fitur Tinggi Badan (Height):** Pada demonstrasi ini, kita hanya menggunakan 4 fitur (`Weight`, `MTRANS`, `CAEC`, `Gender`). Tanpa adanya atribut `Height` (Tinggi), model kehilangan instrumen untuk menghitung rasio indeks massa tubuh secara matematis. Model hanya mengandalkan `Weight` (Berat Mutlak) sebagai patokan utama.
2. **Irisan Berat Badan:** Sangat mudah bagi model menebak `Obesity_Type_III` karena berat badan mereka pasti secara mutlak sangat tinggi. Namun, rentang berat badan antara orang `Overweight` dan `Obesity_Type_I` sangat beririsan (overlap). Orang yang pendek dengan berat 70kg bisa masuk obesitas, sedangkan orang tinggi dengan berat 70kg mungkin hanya overweight. Karena tinggi badannya tidak di-input, wajar jika Naive Bayes kebingungan dan sering salah tebak di area tengah ini.

Secara keseluruhan, algoritma Gaussian Naive Bayes telah berjalan dengan sangat baik dan logis sesuai dengan data probabilitas yang diberikan kepadanya.

## Implementasi Script Python (Node Python Script)
Di dalam node Python Script, dilakukan pengolahan data menggunakan 4 atribut terpilih (Weight, MTRANS, CAEC, Gender). Berikut adalah source code lengkap yang diimplemetasikan 

```{code-cell}
import knime.scripting.io as knio
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# 1. SETUP DATA INPUT
# Mengambil data dari port input KNIME
df_train = knio.input_tables[0].to_pandas()
df_test = knio.input_tables[1].to_pandas()

# Menentukan fitur yang digunakan dan kolom target
selected_features = ['Weight', 'MTRANS', 'CAEC', 'Gender']
target_col = 'NObeyesdad'

# 2. ENCODING & PREPROCESSING
# Menggabungkan data untuk memastikan konsistensi label angka
df_all = pd.concat([df_train, df_test], axis=0)

# Melakukan encoding pada kolom teks
cols_to_encode = ['MTRANS', 'CAEC', 'Gender', target_col]
for col in cols_to_encode:
    le = LabelEncoder()
    df_all[col] = le.fit_transform(df_all[col].astype(str))

# Memisahkan kembali data latih dan data uji
df_train_proc = df_all.iloc[:len(df_train), :]
df_test_proc = df_all.iloc[len(df_train):, :]

X_train = df_train_proc[selected_features].values
y_train = df_train_proc[target_col].values
X_test = df_test_proc[selected_features].values

# 3. NORMALISASI DATA (Min-Max Scaling)
scaler = MinMaxScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

# 4. TRAINING & PREDIKSI (Gaussian Naive Bayes)
model = GaussianNB()
model.fit(X_train_norm, y_train)

# Melakukan prediksi pada 20% data testing
y_pred_encoded = model.predict(X_test_norm)

# 5. DECODE HASIL PREDIKSI
# Mengembalikan hasil prediksi angka menjadi teks klasifikasi obesitas
le_target = LabelEncoder()
le_target.fit(df_all[target_col].astype(str))
y_pred_text = le_target.inverse_transform(y_pred_encoded)

# 6. OUTPUT KE KNIME
df_test['Prediction_Naive_Bayes'] = y_pred_text
knio.output_tables[0] = knio.Table.from_pandas(df_test)
```

## Demonstrasi Perhitungan Manual Naive Bayes

Untuk memahami cara kerja algoritma di balik layar (sebelum dieksekusi oleh KNIME), berikut adalah demonstrasi perhitungan probabilitas klasifikasi Naive Bayes secara manual menggunakan 4 atribut terpilih (`Weight`, `MTRANS`, `CAEC`, `Gender`).

Algoritma Naive Bayes beroperasi berdasarkan Teorema Bayes:
$$P(C \mid X) = \frac{P(X \mid C) \cdot P(C)}{P(X)}$$

Karena nilai penyebut $P(X)$ selalu konstan untuk semua kelas, perhitungan dapat disederhanakan dengan mencari nilai probabilitas gabungan terbesar (Maximum A Posteriori):
$$Posterior \propto Prior \times Likelihood$$
$$P(C_i \mid X) \propto P(C_i) \cdot \prod_{k=1}^{n} P(x_k \mid C_i)$$

### 1. Skenario Data Uji (Data Dummy)
Misalkan kita memiliki satu data pasien baru yang belum diketahui tingkat obesitasnya, dengan nilai atribut yang sudah melalui tahap *preprocessing* (Encoding & Normalisasi) sebagai berikut:
* **Gender** = 1 (Female)
* **MTRANS** = 0 (Public_Transportation)
* **CAEC** = 1 (Sometimes)
* **Weight** = 0.45 (Sudah dinormalisasi)

### 2. Menghitung Probabilitas Prior $P(C)$
Langkah pertama adalah menghitung seberapa sering masing-masing kelas target (Tingkat Obesitas) muncul di dalam data *training*.
$$P(C_i) = \frac{\text{Jumlah kemunculan kelas } C_i}{\text{Total seluruh data latih}}$$
*Contoh:* Jika dari total 1.688 data training terdapat 280 data kelas `Obesity_Type_I`, maka Probabilitas Prior $P(\text{Obesity\_Type\_I}) = \frac{280}{1688} = 0.165$. Hal ini dihitung untuk ke-7 kelas obesitas.

### 3. Menghitung Likelihood $P(X \mid C)$
Langkah kedua adalah menghitung probabilitas setiap nilai atribut pada masing-masing kelas target.

**A. Untuk Fitur Kategorikal (Gender, MTRANS, CAEC):**
Dihitung menggunakan probabilitas diskrit (frekuensi kemunculan).

$$P(Gender=1 \mid C_i) = \frac{\text{Jumlah Female pada kelas } C_i}{\text{Total data pada kelas } C_i}$$

**B. Untuk Fitur Numerik Kontinu (Weight):**
Karena `Weight` adalah data numerik yang diasumsikan berdistribusi normal (Gaussian), probabilitasnya tidak dihitung dari frekuensi, melainkan menggunakan fungsi *Probability Density Function* (PDF):

$$P(Weight=0.45 \mid C_i) = \frac{1}{\sqrt{2\pi \sigma^2_{C_i}}} \exp \left( -\frac{(0.45 - \mu_{C_i})^2}{2\sigma^2_{C_i}} \right)$$
*Di mana:*

* $\mu_{C_i}$ = Rata-rata (mean) atribut Weight pada kelas $C_i$

* $\sigma^2_{C_i}$ = Varians atribut Weight pada kelas $C_i$

### 4. Menghitung Probabilitas Posterior $P(C \mid X)$
Langkah terakhir adalah mengalikan nilai *Prior* dengan seluruh nilai *Likelihood* fitur untuk masing-masing kelas.

*Misal untuk kelas `Obesity_Type_I`:*

$$Posterior = P(Obesity\_Type\_I) \times P(Gender=1 \mid Obesity\_Type\_I) \times P(MTRANS=0 \mid Obesity\_Type\_I) \times P(CAEC=1 \mid Obesity\_Type\_I) \times P(Weight=0.45 \mid Obesity\_Type\_I)$$

Perhitungan ini diulang untuk ke-7 kelas obesitas. Kelas dengan nilai *Posterior* tertinggi (paling maksimum) akan dipilih oleh model sebagai hasil akhir tebakan tingkat obesitas untuk pasien tersebut.