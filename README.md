# Perbandingan Decision Tree dan K-Nearest Neighbor

## 1. Tujuan

Membandingkan dua algoritma klasifikasi—**Decision Tree** dan **K-Nearest Neighbor (K-NN)**—dalam memprediksi keputusan pembelian pelanggan berdasarkan usia dan pendapatan.

## 2. Dataset

- **Jumlah data:** 96
- **Fitur:** `age`, `income`
- **Target:** `will_buy` (1 = membeli, 0 = tidak)

## 3. Proses Analisis

- Visualisasi distribusi usia, pendapatan, dan keputusan pembelian
- Pembagian data (80% training, 20% testing) dengan stratifikasi
- **Decision Tree:** `max_depth=5`, hasil divisualisasikan
- **K-NN:** uji `k` dari 1–20, pilih yang akurasinya tertinggi
- Evaluasi dengan confusion matrix dan classification report

## 4. Hasil

- Akurasi kedua model: **95%**
- Decision Tree mudah dijelaskan
- K-NN lebih fleksibel terhadap pola data

## 5. Insight Bisnis

- Pendapatan lebih berpengaruh dari usia dalam keputusan pembelian
- Model dapat membantu segmentasi dan strategi pemasaran

## 6. Jalankan Proyek

```bash
python decision_tree_vs_knn.py
```

## 7. Dependensi

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```
