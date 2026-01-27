import matplotlib.pyplot as plt
import numpy as np

# Deneyden elde ettiğin gerçek sonuçlar
# (Konsol çıktısından aldığımız değerler)
scores = {
    'Original Graph': 0.6096,
    'Random Graph': 0.3194
}

# Grafik Ayarları
labels = list(scores.keys())
values = list(scores.values())
colors = ['#2ecc71', '#e74c3c'] # Yeşil (Başarılı) ve Kırmızı (Başarısız)

plt.figure(figsize=(8, 6))
bars = plt.bar(labels, values, color=colors, width=0.5)

# Eksenler ve Başlıklar (IEEE Standardı)
plt.ylabel('F1 Score (Macro)', fontsize=12)
plt.title('Impact of Graph Structure on Model Performance\n(Structural Leakage Test)', fontsize=14)
plt.ylim(0, 0.8) # Y eksenini biraz yukarıda tutalım ki barlar sıkışmasın

# Barların üzerine değerleri yaz
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{height:.4f}',
             ha='center', va='bottom', fontsize=12, fontweight='bold')

# Izgara çizgileri (Okunabilirlik için)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Kaydet
plt.savefig('problem1_structural_leakage.png', dpi=300)
print("✅ Grafik kaydedildi: problem1_structural_leakage.png")
plt.show()