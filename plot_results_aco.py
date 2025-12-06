import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('aco_smart_results.csv')

# Setup plot
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")

# Scatter plot: Sumbu X = Jumlah Fitur, Sumbu Y = Validasi Akurasi
# Warna titik = Jumlah Semut
scatter = sns.scatterplot(
    data=df, 
    x='Selected_Feats', 
    y='Val_Acc', 
    hue='Ants', 
    palette='viridis', 
    s=100, 
    alpha=0.8
)

# Tandai Pemenang
best_row = df.iloc[0] # Karena sudah di-sort, baris pertama adalah juara
plt.scatter(best_row['Selected_Feats'], best_row['Val_Acc'], color='red', s=200, marker='*', label='Best Model')

# Annotasi
plt.annotate(f"Best: {best_row['Val_Acc']:.4f}\n(85 Feats)", 
             (best_row['Selected_Feats'], best_row['Val_Acc']),
             xytext=(best_row['Selected_Feats']+10, best_row['Val_Acc']+0.005),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.title('Optimization Landscape: Accuracy vs. Efficiency', fontsize=14)
plt.xlabel('Number of Selected Features (Efficiency)', fontsize=12)
plt.ylabel('Validation Accuracy (Performance)', fontsize=12)
plt.legend(title='Num Ants')
plt.grid(True, linestyle='--', alpha=0.7)

# Simpan gambar
plt.tight_layout()
plt.savefig('optimization_result.png', dpi=300)
print("Grafik disimpan sebagai 'optimization_result.png'")
plt.show()