# Visualisasi Data IRIS

Visualisasi data merupakan proses mengonversi informasi mentah menjadi representasi grafis, seperti diagram, plot, atau gambar, dengan tujuan utama menyederhanakan kompleksitas data. Melalui visualisasi, kita dapat mendeteksi pola, tren, hingga korelasi antar variabel secara instan yang mungkin sulit ditemukan jika hanya mengamati barisan angka dalam tabel statistik.Dalam eksplorasi dataset Iris kali ini, proses visualisasi dilakukan menggunakan Orange Data Mining. Orange adalah perangkat lunak berbasis visual programming yang memungkinkan pengguna melakukan analisis data secara interaktif tanpa harus menulis baris kode (coding). Kita cukup menghubungkan berbagai widget untuk menghasilkan visualisasi yang mendalam dan dinamis.

## Histogram

Histogram digunakan untuk melihat persebaran frekuensi nilai pada fitur sepal_length yang dikelompokkan berdasarkan spesies.

![visualHistogram](img/visualH.png)

- Iris-setosa (Biru): Memiliki distribusi yang cenderung terkumpul di area nilai rendah, dengan puncak frekuensi pada kisaran 5.0 cm. Ini menunjukkan Setosa memiliki kelopak yang paling pendek.
- Iris-versicolor (Merah): Berada di rentang menengah, dengan puncak distribusi di sekitar 5.6 cm hingga 5.9 cm.
- Iris-virginica (Hijau): Memiliki sebaran yang paling luas dan menempati area nilai tertinggi, dengan puncak frekuensi di atas 6.3 cm.
- Analisis: Terdapat tumpang tindih (overlap) yang signifikan antara Versicolor dan Virginica, yang menandakan bahwa fitur sepal_length saja tidak cukup kuat untuk memisahkan kedua spesies ini secara sempurna.

## Scatter Plot

Scatter plot memetakan hubungan antara sepal_length (sumbu X) dan sepal_width (sumbu Y) untuk melihat pola pengelompokan spesies.

![visualScatterPlot](img/visualSP.png)

- Pemisahan Kelompok: Spesies Iris-setosa membentuk kelompok (cluster) yang terpisah jauh dari dua spesies lainnya. Hal ini menunjukkan karakteristik kelopak Setosa yang pendek namun cenderung lebar.
- Identifikasi Outlier: Titik-titik data yang berada jauh di luar kelompok utamanya dapat diidentifikasi secara visual sebagai outlier atau pencilan yang perlu diperhatikan dalam analisis lebih lanjut.