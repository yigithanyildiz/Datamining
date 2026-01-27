import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
import numpy as np

# =====================
# 1. AYARLAR & YÜKLEME
# =====================
device = torch.device("cpu") # GPU varsa "cuda" yapabilirsiniz
print(f"Using device: {device}")

try:
    data = torch.load("amazon_office_graph.pt", weights_only=False)
    data = data.to(device)
except FileNotFoundError:
    print("HATA: 'amazon_office_graph.pt' bulunamadı.")
    exit()

# --- BİLİMSEL KONTROL 1: Feature Leakage Önlemi (GÜÇLÜ MASK) ---
# Modelin "puana" veya "metne" bakıp kopya çekmesini engellemek için
# tüm özellikleri siliyoruz. Ona "kimliksiz" düğümler veriyoruz.
# Böylece model sadece "Kiminle Arkadaş?" (Structure) bilgisine bakmak ZORUNDA kalır.
print("🛑 Feature Masking Uygulanıyor (Modelin gözü bağlanıyor)...")
data.x = torch.ones((data.num_nodes, 10)).to(device) 
num_features = 10  # Yapay feature boyutu

# =====================
# 2. MODEL MİMARİSİ (GCN)
# =====================
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Giriş boyutunu 10 yaptık (Maskelenmiş featurelar)
        self.conv1 = GCNConv(num_features, 32) 
        self.conv2 = GCNConv(32, 16)
        self.conv3 = GCNConv(16, 2) # Output: 2 sınıf (Memnun / Değil)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        x = self.conv3(x, edge_index)
        return x

# =====================
# 3. TRAIN / EVAL (Hocanın İstediği Accuracy Formatı)
# =====================
def train_and_eval(edge_index, exp_name, epochs=60):
    
    # --- BİLİMSEL KONTROL 2: Class Imbalance Çözümü ---
    # Sınıfları dengeliyoruz (50 Memnun - 50 Memnun Değil)
    valid_mask = data.y != -1
    valid_indices = valid_mask.nonzero(as_tuple=True)[0]
    labels = data.y[valid_indices]

    neg_indices = valid_indices[labels == 0]
    pos_indices = valid_indices[labels == 1]

    # Azınlık sınıfı kadar çoğunluktan al
    min_count = min(len(neg_indices), len(pos_indices))
    
    perm_neg = torch.randperm(len(neg_indices))[:min_count]
    perm_pos = torch.randperm(len(pos_indices))[:min_count]

    balanced_indices = torch.cat([neg_indices[perm_neg], pos_indices[perm_pos]])
    
    # Train/Test Split
    perm = torch.randperm(len(balanced_indices))
    train_size = int(0.8 * len(balanced_indices))
    
    train_idx = balanced_indices[perm[:train_size]]
    test_idx = balanced_indices[perm[train_size:]]

    print(f"\n[{exp_name}] Eğitim Başlıyor... (Test Seti: {len(test_idx)} Kişi)")

    model = GCN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    # Eğitim
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, edge_index)
        loss = criterion(out[train_idx], data.y[train_idx])
        loss.backward()
        optimizer.step()

    # Test (SADECE ACCURACY)
    model.eval()
    with torch.no_grad():
        out = model(data.x, edge_index)
        pred = out.argmax(dim=1)
        
        # Basit Doğruluk Hesabı
        correct = (pred[test_idx] == data.y[test_idx]).sum().item()
        total = test_idx.size(0)
        acc = correct / total

    return acc

# =====================
# 4. DENEYLER
# =====================

print("Deney 1: Orijinal Graph (Gerçek İlişkiler)...")
acc_orig = train_and_eval(data.edge_index, "ORİJİNAL")
print(f"✅ Orijinal Graph -> Accuracy: {acc_orig:.4f}")

print("\nDeney 2: Random Graph (Bozuk İlişkiler)...")
# Kenarları kopyala ve karıştır
num_edges = data.edge_index.size(1)
random_src = torch.randint(0, data.num_nodes, (num_edges,), device=device)
random_dst = torch.randint(0, data.num_nodes, (num_edges,), device=device)
random_edge_index = torch.stack([random_src, random_dst]).to(device)

acc_rand = train_and_eval(random_edge_index, "RANDOM")
print(f"❌ Random Graph   -> Accuracy: {acc_rand:.4f}")

# =====================
# 5. GRAFİK ÇİZİMİ (RAPOR İÇİN)
# =====================
print("\nGrafik oluşturuluyor...")

labels = ['Original Graph\n(Structure Intact)', 'Random Graph\n(Structure Broken)']
values = [acc_orig, acc_rand]
colors = ['#3498db', '#e74c3c'] # Mavi ve Kırmızı

plt.figure(figsize=(8, 6))
bars = plt.bar(labels, values, color=colors, width=0.5)

# Değerleri yaz
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.ylabel('Test Accuracy Score')
plt.title('Proof of Structural Dependency\n(Feature Masked Test)')
plt.ylim(0, 1.0)
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.savefig('experiment_1_structural_leakage.png', dpi=300)
print(f"✅ Grafik kaydedildi: experiment_1_structural_leakage.png")
print("\nSONUÇ: Eğer Mavi bar Kırmızıdan yüksekse, modelin yapıya bağımlı olduğu kanıtlanmıştır.")
plt.show()