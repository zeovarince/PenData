# Memahami Data

## Data Understanding
- Mengumpulkan data awal dan dilanjutkan dengan kegiatan untuk mendapat data yang lazim serta,
- Identifikasi data yang berkualitas
- Pemahaman data sangat diperlukan untuk mendeteksi bagian yang menarik dari data sehingga dapat membangun hipotesa terhadap informasi yang tersembunyi

## Pentingnya memahami data
- Membantu memilih teknik data mining yang tepat
- Meningkat akurasi model prediksi
- Menghindari kesalahan interpretasi
- Memastikan hasil yang dapat diandalkan dan dapat ditindaklanjuti

## Klasifikasi Jenis Data dalam Data Mining

Dalam ekosistem *Big Data*, kita akan berhadapan dengan berbagai format data. Memahami jenis data ini sangat penting karena setiap jenis memerlukan algoritma dan alat (*tools*) yang berbeda untuk diolah.

| Jenis Data | Karakteristik Utama | Format/Contoh File | Contoh Nyata |
| :--- | :--- | :--- | :--- |
| **Terstruktur (*Structured*)** | Memiliki skema tetap, rapi dalam baris dan kolom. Sangat mudah dicari dan diolah. | SQL, Excel, CSV | Data transaksi bank, tabel induk mahasiswa. |
| **Tidak Terstruktur (*Unstructured*)** | Tidak memiliki format atau struktur internal yang tetap. | File mentah | Kumpulan file acak, dokumen PDF tanpa format baku. |
| **Bahasa Alami (*Natural Language*)** | Data berbentuk teks komunikasi manusia. Membutuhkan teknik NLP. | Dokumen Teks, Chat | Tweet di X, ulasan produk, isi email. |
| **Dibangkitkan Mesin (*Machine-Generated*)** | Data otomatis hasil sensor atau log perangkat tanpa input manual manusia. | Log files, JSON | Log server, riwayat pencarian web, sensor suhu. |
| **Multimedia (Audio, Video, Citra)** | Data berupa sinyal suara, piksel gambar, atau urutan bingkai video. | JPG, MP3, MP4 | Foto rontgen, rekaman CCTV, *face recognition*. |
| **Streaming** | Data yang masuk secara terus-menerus dan *real-time* dalam volume besar. | Data Aliran (*Streams*) | Data bursa saham, navigasi GPS, data *smartwatch*. |
| **Berbasis Graf (*Graph-based*)** | Menitikberatkan pada hubungan atau koneksi antar entitas (simpul & sisi). | Graph Database | Jaringan pertemanan Facebook, rute penerbangan. |


## Mengapa Perbedaan Ini Penting?

* **Penyimpanan:** Data terstruktur disimpan di *Database Relasional* (MySQL), sedangkan data tidak terstruktur biasanya di *Data Lake* (Hadoop/NoSQL).
* **Analisis:** Data citra (gambar) membutuhkan *Deep Learning* (CNN), sedangkan data terstruktur bisa diolah dengan algoritma statistik sederhana.
* **Persiapan:** Data teks (*Natural Language*) membutuhkan tahap *Cleaning* khusus seperti penghapusan kata sambung (*stopword removal*) sebelum bisa diproses.

## Komponen Utama Memahami Data


| Komponen | Penjelasan |
| -------- | -------- |
| Pengumpulan Data Awal    | Mengidentifikasi dan mengumpulkan semua sumber data yang relevan (database, CSV, API, dll.)    |
| Deskripsi Data | Memahami struktur data: jumlahbaris/kolom, jenis variabel (numerik,kategorikal), nama atribut, contoh data. |
| Eksploraasi Data (Exploratory Data Analysis/ EDA) | Mencari pola, tren, korelasi, atau anomali menggunakan statistik deskriptif dan visualisasi(Grafik, Histogram, Scatter plot.) |
| Kualitas | Memeriksa kelengkapan, keakuratan, dan konsistensi data. Meliputi pengecekan data hilang (missing value), data duplikat, outlier atau nilai tidak wajar, kesalahan format data, inkonsistensi penulisan, serta kesesuaian tipe data dengan atribut yang digunakan. |

## Atribut Data

Memahami tipe atribut adalah kunci untuk menentukan metode statistik atau algoritma yang akan digunakan. Berikut adalah klasifikasi atribut berdasarkan skala pengukurannya:

| Jenis Atribut | Karakteristik | Contoh |
| :--- | :--- | :--- |
| **Nominal (Kategorikal)** | Hanya sebagai label/nama, tidak memiliki urutan atau nilai numerik. | Jenis kelamin, warna, ID karyawan, nama kota. |
| **Biner (*Binary*)** | Kasus khusus nominal yang hanya memiliki dua kemungkinan nilai (0 atau 1). | Ya/Tidak, Lulus/Gagal, Positif/Negatif. |
| **Ordinal** | Memiliki urutan atau tingkatan yang bermakna, namun jarak antar nilai tidak tetap. | Tingkat pendidikan (SD, SMP, SMA), tingkat kepuasan pelanggan. |
| **Numerik (*Quantitative*)** | Berbentuk angka yang dapat dioperasikan secara matematis. | Berat badan, suhu, pendapatan, jumlah stok barang. |

## Seleksi Fitur (Feature Selection)

Seleksi fitur adalah proses memilih subset atribut yang paling relevan untuk digunakan dalam pembangunan model. 

### Mengapa perlu Seleksi Fitur?
* **Reduksi Dimensi:** Mengurangi jumlah kolom yang tidak perlu.
* **Menghilangkan Noise:** Membuang data yang tidak memiliki hubungan dengan target prediksi.
* **Efisiensi:** Mempercepat proses komputasi.