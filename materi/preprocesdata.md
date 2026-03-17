# Preprocessing Data

## Preprocessing Data dalam Penambangan Data

**Preprocessing Data** adalah langkah awal yang sangat penting dalam siklus *Data Mining*, yang berfokus pada transformasi data mentah menjadi format yang lebih bersih, terstruktur, dan siap untuk menganalisis pola melalui algoritma *Machine Learning*. 

Salah satu prinsip kunci yang mendasari pentingnya tahap ini adalah **"Garbage In, Garbage Out" (GIGO)**. Tidak peduli seberapa canggih algoritma pemodelan yang digunakan, jika data yang dimasukkan berkualitas rendah (misalnya, mengandung nilai kosong, *noise*, atau ketidakkonsistenan), maka hasil yang diperoleh akan menjadi wawasan atau prediksi yang tidak akurat.

Secara rinci, tahapan *Preprocessing Data* terdiri dari empat proses utama, yaitu:

### A. Pembersihan Data
Data di dunia nyata sering kali tidak lengkap, berisik, dan rentan terhadap kesalahan. Proses ini bertujuan untuk "memperbaiki" data tersebut.
* **Menangani Missing Values:** Ketidakhadiran nilai pada atribut tertentu bisa disebabkan oleh kesalahan input, kerusakan sensor, atau data yang memang tidak tersedia. Beberapa metode untuk mengatasinya meliputi:
    * Menghapus *record* (baris) atau atribut (kolom) jika nilai yang hilang terlalu banyak.
    * Imputasi menggunakan ukuran tendensi sentral seperti Mean, Median, atau Modus.
    * Imputasi berbasis pemodelan prediksi, seperti menggunakan algoritma regresi atau **K-Nearest Neighbors (KNN) / Weighted KNN (WKNN)** untuk memprediksi nilai yang hilang berdasarkan kemiripan data.
* **Menangani Noisy Data & Outliers:** *Noise* adalah variasi acak atau kesalahan dalam variabel yang diukur, sedangkan *outlier* adalah data yang menyimpang jauh dari distribusi normal. Teknik penanganannya bisa menggunakan *Binning*, *Clustering* (untuk mendeteksi pencilan), atau regresi linear.

### B. Integrasi Data
Dalam analisis berskala besar, data jarang berasal dari satu sumber tunggal. Integrasi data adalah proses menggabungkan data dari berbagai basis data, *data cube*, atau *flat file* ke dalam satu *data store* yang koheren.
* Tantangan utama di tahap ini adalah **Entity Identification Problem** (memastikan bahwa entitas yang sama dari sumber yang berbeda dapat dikenali, misalnya mencocokkan "customer_id" dari database A dengan "cust_number" dari database B) dan menangani redundansi fitur.

### C. Transformasi Data
Tahap ini mengubah atau mengkonsolidasikan data ke dalam bentuk yang paling optimal untuk algoritma *data mining*.
* **Normalisasi / Standardisasi:** Algoritma berbasis jarak (seperti KNN, K-Means) sangat sensitif terhadap skala data. Normalisasi mengubah nilai numerik ke dalam skala yang seragam (misalnya: 0 hingga 1), sehingga fitur dengan rentang nilai besar (seperti gaji) tidak mendominasi fitur dengan rentang kecil (seperti umur).
* **Diskritisasi:** Mengubah atribut kontinu menjadi interval diskrit atau atribut nominal. Contoh: Mengubah rentang umur 1-100 menjadi kategori "Anak-anak", "Remaja", "Dewasa", dan "Lansia".
* **Konstruksi Fitur:** Menciptakan atribut baru dari himpunan atribut yang sudah ada untuk membantu meningkatkan akurasi model.

### D. Reduksi Data
Dataset modern sering kali berukuran sangat besar (bisa mencapai terabyte). Melakukan analisis kompleks pada data sebesar ini sangat memakan waktu. *Reduksi Data* bertujuan untuk mendapatkan representasi dataset yang jauh lebih kecil, namun tetap mampu menghasilkan output analitis yang sama (atau hampir sama).
* **Pengurangan Dimensi:** Mengurangi jumlah fitur/atribut menggunakan teknik seperti PCA (*Principal Component Analysis*).
* **Pengurangan Numeris:** Mengganti data asli dengan model parametrik yang lebih kecil (hanya menyimpan parameter modelnya saja) atau menggunakan metode non-parametrik seperti *sampling* atau histogram.

### Tujuan Utama Melakukan Data Preprocessing

Secara teknis dan analitis, *data preprocessing* bukan sekadar langkah "bersih-bersih" biasa. Tahapan ini memiliki tujuan fundamental yang berdampak langsung pada performa algoritma *machine learning* dan *data mining*:

1. **Meningkatkan Kualitas Data (Data Quality):** Kualitas data diukur dari beberapa dimensi, yaitu akurasi, kelengkapan (*completeness*), konsistensi, ketepatan waktu (*timeliness*), dan validitas. Preprocessing memastikan data yang masuk memenuhi standar kualitas ini sebelum dianalisis.
2. **Mencegah Bias dan *Overfitting*:** Data dunia nyata sering kali mengandung *noise* atau pola acak yang tidak relevan. Jika data ini langsung dilatih, model bisa mengalami *overfitting* (terlalu menghafal data latih termasuk *noise*-nya) sehingga gagal melakukan prediksi pada data baru.
3. **Mempercepat Konvergensi Algoritma:** Banyak algoritma optimasi (seperti *Gradient Descent* pada *Neural Networks* atau Regresi Logistik) akan bekerja jauh lebih lambat jika skala fitur-fiturnya tidak seimbang. Teknik transformasi seperti Normalisasi memastikan algoritma mencapai titik optimal (konvergen) dengan lebih cepat dan efisien.
4. **Menghindari *Curse of Dimensionality* (Kutukan Dimensi):** Dataset dengan ratusan atau ribuan fitur/kolom dapat membuat model kebingungan dan menurunkan akurasi algoritma berbasis jarak (seperti KNN). Preprocessing melalui teknik *dimensionality reduction* bertujuan untuk mengekstraksi fitur yang benar-benar bermakna saja.
5. **Memenuhi Asumsi Matematis Algoritma:** Setiap algoritma memiliki "syarat dan ketentuan" matematis. Misalnya:
    * **KNN dan K-Means:** Sangat sensitif terhadap skala data (membutuhkan Normalisasi/Standardisasi).
    * **Naive Bayes:** Asumsi independensi antar fitur dan sering kali bekerja lebih baik dengan data diskrit.
    * **Regresi Linear:** Membutuhkan penanganan *outlier* yang ketat karena garis regresi sangat mudah ditarik oleh nilai ekstrem.

---

### Kapan Data Preprocessing Dilakukan? (Pendekatan CRISP-DM)

Dalam industri *Data Science*, standar proses yang paling banyak digunakan adalah **CRISP-DM** (*Cross-Industry Standard Process for Data Mining*). Metodologi ini membagi proyek penambangan data ke dalam enam fase utama. 



*Data Preprocessing* secara eksklusif terjadi pada fase **Data Preparation (Persiapan Data)**. Menariknya, para praktisi data mengakui bahwa **fase ini memakan sekitar 60% hingga 80% dari total waktu dan upaya** dalam keseluruhan proyek penambangan data.

Berikut adalah posisi *Data Preprocessing* dalam alur CRISP-DM:

1. **Business Understanding (Pemahaman Bisnis):** Menentukan tujuan proyek dari perspektif bisnis.
2. **Data Understanding (Pemahaman Data):** Mengumpulkan data mentah, mengeksplorasi data awal (EDA - *Exploratory Data Analysis*), dan mengidentifikasi masalah kualitas data (mendeteksi adanya *missing values* atau *outlier*).
3. **Data Preparation (Fase Data Preprocessing):**  **Di sinilah preprocessing dilakukan secara penuh.** Data mentah dibersihkan, diintegrasikan, diubah formatnya (transformasi/normalisasi), dan direduksi jika perlu. Output dari fase ini adalah dataset final yang bersih dan siap dimodelkan.
4. **Modeling (Pemodelan):** Dataset hasil *preprocessing* mulai dimasukkan ke dalam algoritma *machine learning* (seperti WKNN, Decision Tree, atau SVM). Jika model gagal berjalan, analis sering kali harus mundur kembali ke fase *Data Preparation* untuk memperbaiki datanya.
5. **Evaluation (Evaluasi):** Mengevaluasi apakah model sudah menjawab tujuan bisnis.
6. **Deployment (Penyebaran):** Menerapkan model ke dalam sistem produksi (*production*).

Dengan memahami alur ini, terlihat jelas bahwa *preprocessing* adalah jembatan kritis. Data mentah tidak akan pernah bisa melompat langsung ke tahap *Modeling* tanpa melalui "pabrik pemrosesan" di tahap *Data Preparation*.