# Analisis Sentimen Ulasan Aplikasi Gojek

Proyek ini melakukan analisis sentimen terhadap ulasan pengguna aplikasi Gojek di Google Play Store menggunakan teknik NLP. Pipeline mencakup scraping data, preprocessing teks, ekstraksi fitur TF-IDF, dan klasifikasi dengan Logistic Regression.

## Struktur Proyek

```
analisis_sentimen_gojek/
├── Scraping_Review_Google_Play_Store.ipynb  # Scraping & preprocessing dataset
├── NPL_Review_Gojek.ipynb                   # Pelatihan & evaluasi model
├── gojek_reviews_balanced.csv               # Dataset final (4000 ulasan)
└── requirements.txt                         # Dependensi library
```

## Alur Pipeline

### 1. Scraping Data (`Scraping_Review_Google_Play_Store.ipynb`)

- Mengambil **1.673.024 ulasan** dari Google Play Store menggunakan `google-play-scraper` (bahasa Indonesia, region ID)
- **Cleaning teks:** lowercase, hapus URL, emoji, angka, simbol, kata < 3 huruf
- **Labeling:** skor ≤ 2 → Negatif (`-1`), skor > 2 → Positif (`1`)
- **Balancing:** 2000 sampel per kelas → **4000 total**
- Output: `gojek_reviews_balanced.csv`

### 2. Pemodelan NLP (`NPL_Review_Gojek.ipynb`)

- **Preprocessing:** stopword removal (NLTK Indonesian), tokenisasi, filtering panjang kata
- **Ekstraksi Fitur:** TF-IDF berbasis karakter n-gram (`char_wb`, ngram range 3–5) → 25.781 fitur
- **Split:** 80% train (3200), 20% test (800), stratified
- **Model:** Logistic Regression (`C=3.0`, solver `liblinear`, `class_weight='balanced'`)

## Hasil Evaluasi

| Metrik    | Negatif | Positif | Macro Avg |
| --------- | ------- | ------- | --------- |
| Precision | 0.84    | 0.87    | 0.86      |
| Recall    | 0.88    | 0.83    | 0.85      |
| F1-Score  | 0.86    | 0.85    | 0.85      |

**Akurasi: 85.50%**

**Confusion Matrix:**

```
              Prediksi
              Negatif  Positif
Aktual Negatif   351       49
       Positif    67      333
```

## Cara Menjalankan

Proyek ini dijalankan di **Google Colab**. Dataset disimpan di Google Drive.

1. Upload `gojek_reviews_balanced.csv` ke Google Drive
2. Buka `NPL_Review_Gojek.ipynb` di Google Colab
3. Mount Google Drive dan sesuaikan `file_path`
4. Jalankan sel secara berurutan

### Contoh Prediksi

```python
predict_review("aplikasi gojek sangat membantu dan drivernya ramah")
# Sentimen: Positif

predict_review("aplikasi sering error dan saldo saya tidak masuk")
# Sentimen: Negatif
```

## Dataset

File `gojek_reviews_balanced.csv` berisi 2 kolom:

| Kolom        | Deskripsi                           |
| ------------ | ----------------------------------- |
| `clean_text` | Teks ulasan setelah proses cleaning |
| `label`      | `-1` = Negatif, `1` = Positif       |

## Teknologi

- **Platform:** Google Colab
- **Bahasa:** Python 3.12
- **Library Utama:** `pandas`, `scikit-learn`, `nltk`, `google-play-scraper`, `emoji`
