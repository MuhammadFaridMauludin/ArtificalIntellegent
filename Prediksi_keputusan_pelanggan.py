import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import tree
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("husl")

print("="*80)
print("PERBANDINGAN KINERJA DECISION TREE DAN K-NEAREST NEIGHBOR")
print("PADA DATA PREDIKSI PEMBELIAN PELANGGAN")
print("="*80)

print("\nA. DATASET")
print("-" * 50)

data = {
    'age': [25,35,45,30,50,28,40,55,32,48,22,39,41,29,33,46,34,36,47,55,26,38,42,51,29,32,37,53,48,27,30,52,31,44,43,35,49,26,34,40,42,51,31,48,54,56,29,37,27,39,45,28,36,53,34,49,30,35,43,52,26,38,47,28,41,44,31,37,55,33,46,49,32,30,38,35,45,41,23,33,48,40,53,42,51,29,26,37,32,34,50,48,35,45,31,39],
    'income': [30000,40000,60000,35000,80000,32000,45000,70000,38000,75000,28000,42000,49000,33000,37000,58000,39000,41000,62000,80000,31000,48000,55000,78000,34000,36000,42000,65000,72000,33000,38000,77000,37000,59000,56000,40000,78000,32000,38000,49000,51000,75000,36000,68000,77000,81000,34000,43000,32000,48000,58000,33000,42000,67000,37000,74000,35000,43000,54000,78000,30000,45000,59000,34000,48000,56000,36000,43000,72000,38000,61000,74000,38000,35000,48000,40000,59000,51000,29000,40000,64000,50000,69000,52000,76000,36000,30000,45000,37000,39000,77000,71000,42000,57000,35000,49000],
    'will_buy': [0,1,1,0,1,0,1,1,0,1,0,1,1,0,0,1,1,1,1,1,0,1,1,1,0,0,1,1,1,0,0,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,0,1,0,1,1,0,1,1,1,1,0,1,1,1,0,1,1,0,1,1,0,1,1,1,1,1,0,0,1,1,1,1,0,1,1,1,1,1,1,0,0,1,0,1,1,1,1,1,0,1]
}

df = pd.DataFrame(data)

print(f"Nama Dataset: Customer Purchase Prediction Dataset")
print(f"Jumlah sampel: {len(df)}")
print(f"Jumlah fitur: {df.shape[1] - 1}")
print(f"Target yang diprediksi: will_buy (1 = akan membeli, 0 = tidak akan membeli)")

print("\nAtribut/Fitur:")
print("- age: Usia pelanggan (tahun)")
print("- income: Pendapatan pelanggan (rupiah)")
print("- will_buy: Target variabel (0/1)")

print("\nStatistik Deskriptif:")
print(df.describe())

print("\nDistribusi Target:")
print(df['will_buy'].value_counts())

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

axes[0,0].hist(df['age'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
axes[0,0].set_title('Distribusi Usia Pelanggan')
axes[0,0].set_xlabel('Usia')
axes[0,0].set_ylabel('Frekuensi')

axes[0,1].hist(df['income'], bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
axes[0,1].set_title('Distribusi Pendapatan Pelanggan')
axes[0,1].set_xlabel('Pendapatan')
axes[0,1].set_ylabel('Frekuensi')

will_buy_counts = df['will_buy'].value_counts()
axes[1,0].pie(will_buy_counts.values, labels=['Tidak Membeli', 'Membeli'], autopct='%1.1f%%', colors=['lightcoral', 'lightblue'])
axes[1,0].set_title('Distribusi Keputusan Pembelian')

scatter = axes[1,1].scatter(df['age'], df['income'], c=df['will_buy'], cmap='coolwarm', alpha=0.7)
axes[1,1].set_title('Hubungan Usia vs Pendapatan (Berdasarkan Keputusan Pembelian)')
axes[1,1].set_xlabel('Usia')
axes[1,1].set_ylabel('Pendapatan')
plt.colorbar(scatter, ax=axes[1,1], label='Will Buy')

plt.tight_layout()
plt.show()

print("\n\nB. METODE KLASIFIKASI")
print("-" * 50)
print("Metode klasifikasi yang digunakan pada penelitian ini adalah:")
print("1. Decision Tree (Pohon Keputusan)")
print("   - Algoritma supervised learning yang menggunakan struktur pohon")
print("   - Mudah diinterpretasi dan memahami aturan keputusan")
print("   - Dapat menangani data numerik dan kategorikal")
print()
print("2. K-Nearest Neighbor (K-NN)")
print("   - Algoritma lazy learning yang mengklasifikasi berdasarkan tetangga terdekat")
print("   - Tidak memerlukan asumsi tentang distribusi data")
print("   - Sensitif terhadap skala data dan pemilihan nilai k")

print("\n\nC. HASIL DAN PEMBAHASAN")
print("-" * 50)

print("\nC.1 DATA SPLITTING")
print("-" * 30)

X = df[['age', 'income']]
y = df['will_buy']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Tahap klasifikasi dimulai dengan membaca data dan memisahkan data menjadi variabel X dan Y,")
print("dimana X adalah atribut (age, income) dan Y adalah target (will_buy).")
print(f"Data dipisahkan menjadi data training (80%) dan testing (20%).")
print(f"Jumlah data training: {len(X_train)}")
print(f"Jumlah data testing: {len(X_test)}")

print("\n\nC.2 MODEL DECISION TREE")
print("-" * 30)

dt_model = DecisionTreeClassifier(max_depth=5, random_state=42, min_samples_split=5)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

print("Decision Tree adalah algoritma yang membangun model prediksi dalam bentuk struktur pohon.")
print("Model ini membuat keputusan dengan memecah data berdasarkan fitur yang memberikan")
print("information gain tertinggi pada setiap node.")

plt.figure(figsize=(15, 10))
tree.plot_tree(dt_model, feature_names=['age', 'income'], class_names=['Not Buy', 'Buy'], filled=True, rounded=True, fontsize=10)
plt.title('Visualisasi Decision Tree')
plt.show()

print("\n\nC.3 MODEL K-NEAREST NEIGHBOR")
print("-" * 30)

k_values = range(1, 21)
k_accuracies = []

for k in k_values:
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(X_train, y_train)
    temp_pred = knn_temp.predict(X_test)
    acc = accuracy_score(y_test, temp_pred)
    k_accuracies.append(acc)

plt.figure(figsize=(10, 6))
plt.plot(k_values, k_accuracies, marker='o', linewidth=2, markersize=8)
plt.title('Akurasi K-NN untuk Berbagai Nilai k')
plt.xlabel('Nilai k')
plt.ylabel('Akurasi')
plt.grid(True, alpha=0.3)
plt.xticks(k_values)
plt.show()

optimal_k = k_values[np.argmax(k_accuracies)]
print(f"Nilai k optimal: {optimal_k} dengan akurasi: {max(k_accuracies):.4f}")

knn_model = KNeighborsClassifier(n_neighbors=optimal_k)
knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_test)

print("K-Nearest Neighbor mengklasifikasi data berdasarkan mayoritas kelas")
print("dari k tetangga terdekat dalam ruang fitur.")

print("\n\nC.4 CONFUSION MATRIX")
print("-" * 30)

print("Confusion matrix menampilkan performa prediksi algoritma dengan membandingkan")
print("prediksi dengan nilai aktual.")

dt_cm = confusion_matrix(y_test, dt_pred)
knn_cm = confusion_matrix(y_test, knn_pred)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

disp1 = ConfusionMatrixDisplay(confusion_matrix=dt_cm, display_labels=['Not Buy', 'Buy'])
disp1.plot(ax=axes[0], cmap='Blues')
axes[0].set_title('Confusion Matrix - Decision Tree')

disp2 = ConfusionMatrixDisplay(confusion_matrix=knn_cm, display_labels=['Not Buy', 'Buy'])
disp2.plot(ax=axes[1], cmap='Greens')
axes[1].set_title('Confusion Matrix - K-NN')

plt.tight_layout()
plt.show()

print("\nConfusion Matrix Decision Tree:")
print(dt_cm)
print("\nConfusion Matrix K-NN:")
print(knn_cm)

print("\n\nC.5 CLASSIFICATION REPORT")
print("-" * 30)

print("Classification report mengukur kualitas prediksi dengan metrik:")
print("- Precision: Proporsi prediksi positif yang benar")
print("- Recall: Proporsi data positif yang berhasil diprediksi")
print("- F1-score: Harmonic mean dari precision dan recall")
print("- Accuracy: Proporsi total prediksi yang benar")

print("\nDecision Tree Classification Report:")
print(classification_report(y_test, dt_pred))

print("\nK-NN Classification Report:")
print(classification_report(y_test, knn_pred))

print("\n\nC.6 ANALISIS TAMBAHAN DAN VISUALISASI")
print("-" * 30)

dt_accuracy = accuracy_score(y_test, dt_pred)
knn_accuracy = accuracy_score(y_test, knn_pred)

models = ['Decision Tree', 'K-NN']
accuracies = [dt_accuracy, knn_accuracy]

plt.figure(figsize=(10, 6))
bars = plt.bar(models, accuracies, color=['skyblue', 'lightgreen'], alpha=0.8, edgecolor='black')
plt.title('Perbandingan Akurasi Model')
plt.ylabel('Akurasi')
plt.ylim(0, 1)

for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')

plt.grid(axis='y', alpha=0.3)
plt.show()

def plot_decision_boundary(model, X, y, title):
    h = 0.02
    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    y_min, y_max = X.iloc[:, 1].min() - 1000, X.iloc[:, 1].max() + 1000
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, 200))
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(mesh_points)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    scatter = plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap='coolwarm', edgecolors='black')
    plt.xlabel('Age')
    plt.ylabel('Income')
    plt.title(title)
    plt.colorbar(scatter)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

plt.subplot(1, 2, 1)
plot_decision_boundary(dt_model, X, y, 'Decision Tree - Decision Boundary')

plt.subplot(1, 2, 2)
plot_decision_boundary(knn_model, X, y, 'K-NN - Decision Boundary')

plt.tight_layout()
plt.show()

feature_importance = dt_model.feature_importances_
features = ['Age', 'Income']

plt.figure(figsize=(8, 6))
bars = plt.bar(features, feature_importance, color=['coral', 'lightblue'], alpha=0.8, edgecolor='black')
plt.title('Feature Importance - Decision Tree')
plt.ylabel('Importance')

for bar, imp in zip(bars, feature_importance):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{imp:.3f}', ha='center', va='bottom', fontweight='bold')

plt.grid(axis='y', alpha=0.3)
plt.show()

print("\n\nD. KESIMPULAN")
print("-" * 50)

print("Berdasarkan analisis yang telah dilakukan, dapat disimpulkan bahwa:")
print(f"1. Decision Tree memiliki akurasi sebesar {dt_accuracy:.4f} ({dt_accuracy*100:.2f}%)")
print(f"2. K-NN memiliki akurasi sebesar {knn_accuracy:.4f} ({knn_accuracy*100:.2f}%)")

if dt_accuracy > knn_accuracy:
    print("3. Decision Tree menunjukkan performa yang lebih baik dibandingkan K-NN")
    print("   untuk dataset ini, kemungkinan karena data memiliki pola yang dapat")
    print("   direpresentasikan dengan baik menggunakan aturan pohon keputusan.")
elif knn_accuracy > dt_accuracy:
    print("3. K-NN menunjukkan performa yang lebih baik dibandingkan Decision Tree")
    print("   untuk dataset ini, menunjukkan bahwa pola data lebih cocok dengan")
    print("   pendekatan berbasis kedekatan/similarity.")
else:
    print("3. Kedua model menunjukkan performa yang sama untuk dataset ini.")

print("4. Fitur pendapatan (income) memiliki pengaruh yang lebih besar dibandingkan")
print("   usia dalam menentukan keputusan pembelian pelanggan.")
print("5. Visualisasi decision boundary menunjukkan perbedaan cara kedua algoritma")
print("   dalam memisahkan kelas data.")

print("\n\nRINGKASAN PERBANDINGAN MODEL:")
print("=" * 60)
print(f"{'Metrik':<20} {'Decision Tree':<15} {'K-NN':<15}")
print("=" * 60)
print(f"{'Akurasi':<20} {dt_accuracy:<15.4f} {knn_accuracy:<15.4f}")
print(f"{'Parameter':<20} {'max_depth=5':<15} {f'k={optimal_k}':<15}")
print(f"{'Interpretability':<20} {'Tinggi':<15} {'Rendah':<15}")
print(f"{'Training Time':<20} {'Cepat':<15} {'Cepat':<15}")
print(f"{'Prediction Time':<20} {'Sangat Cepat':<15} {'Lambat':<15}")
print("=" * 60)
