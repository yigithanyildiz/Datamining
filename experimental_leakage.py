import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.metrics import f1_score
import numpy as np

# =====================
# 1. AYARLAR & YÜKLEME
# =====================
device = torch.device("cpu") # GPU varsa "cuda" yapabilirsiniz
print(f"Using device: {device}")

# Daha önce data_prep.py ile oluşturduğun GERÇEK Amazon verisi
try:
    data = torch.load("amazon_office_graph.pt", weights_only=False)
    data = data.to(device)
except FileNotFoundError:
    print("HATA: 'amazon_office_graph.pt' bulunamadı. Önce data_prep.py çalıştırılmalı.")
    exit()

# --- BİLİMSEL KONTROL 1: Feature Leakage Önlemi ---
# Modelin cevabı "x" özelliklerinden kopya çekmemesi için,
# ortalama puan (rating) bilgisini siliyoruz/sıfırlıyoruz.
# Sadece "yorum sayısı" gibi yapısal özellikler kalıyor.
if data.x.shape[1] > 0:
    data.x[:, 0] = 0.0 
    print("✅ Bilimsel Kontrol: Feature leakage önlendi (Rating sütunu maskelendi).")

num_nodes = data.num_nodes
num_features = data.x.shape[1]

# =====================
# 2. MODEL MİMARİSİ (GCN)
# =====================
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Hidden layer sayısını ve nöron sayısını gerçek veri için artırdık
        self.conv1 = GCNConv(num_features, 64) 
        self.conv2 = GCNConv(64, 32)
        self.conv3 = GCNConv(32, 2) # Output: 2 sınıf (Memnun / Değil)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.conv3(x, edge_index)
        return x

# =====================
# 3. TRAIN / EVAL (Dengeli Veri İle)
# =====================
def train_and_eval(edge_index, exp_name, epochs=100):
    
    # --- BİLİMSEL KONTROL 2: Class Imbalance Çözümü ---
    # Gerçek hayatta memnuniyet oranı %90'dır. Modeli "yapıyı" öğrenmeye zorlamak için
    # eğitim setini 50% Memnun - 50% Memnun Değil olacak şekilde dengeliyoruz.
    
    valid_mask = data.y != -1
    valid_indices = valid_mask.nonzero(as_tuple=True)[0]
    labels = data.y[valid_indices]

    neg_indices = valid_indices[labels == 0]
    pos_indices = valid_indices[labels == 1]

    # Azınlık sınıfı kadar çoğunluktan al (Undersampling)
    min_count = min(len(neg_indices), len(pos_indices))
    
    # Rastgele seçim (Reproducibility için seed eklenebilir)
    perm_neg = torch.randperm(len(neg_indices))[:min_count]
    perm_pos = torch.randperm(len(pos_indices))[:min_count]

    # Dengeli index listesi
    balanced_indices = torch.cat([neg_indices[perm_neg], pos_indices[perm_pos]])
    
    # Train/Test Split (%80 - %20)
    perm = torch.randperm(len(balanced_indices))
    train_size = int(0.8 * len(balanced_indices))
    
    train_idx = balanced_indices[perm[:train_size]]
    test_idx = balanced_indices[perm[train_size:]]

    print(f"\n[{exp_name}] Veri Seti: {min_count*2} Örnek (Dengeli). Train: {len(train_idx)}, Test: {len(test_idx)}")

    # Modeli başlat
    model = GCN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # Eğitim Döngüsü
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, edge_index)
        loss = criterion(out[train_idx], data.y[train_idx])
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            pass # İstersen buraya print koyabilirsin

    # Test (F1 Score - Gerçek Performans)
    model.eval()
    with torch.no_grad():
        out = model(data.x, edge_index)
        pred = out.argmax(dim=1)
        
        y_true = data.y[test_idx].cpu().numpy()
        y_pred = pred[test_idx].cpu().numpy()
        
        acc = (y_pred == y_true).mean()
        f1 = f1_score(y_true, y_pred, average='macro') 

    return acc, f1

# =====================
# 4. DENEYLER
# =====================

print("Deney 1: Orijinal Graph Eğitiliyor...")
acc_orig, f1_orig = train_and_eval(data.edge_index, "ORİJİNAL")
print(f"✅ Orijinal Graph -> Accuracy: {acc_orig:.4f} | F1 Score: {f1_orig:.4f}")

print("\nDeney 2: Random Graph (Yapı Bozulmuş) Eğitiliyor...")
# Kenarları kopyala ve karıştır
edge_index_rand = data.edge_index.clone()
num_edges = edge_index_rand.size(1)

# Tamamen rastgele bağlantılar üret (Structure Destroyed)
random_src = torch.randint(0, num_nodes, (num_edges,), device=device)
random_dst = torch.randint(0, num_nodes, (num_edges,), device=device)
random_edge_index = torch.stack([random_src, random_dst]).to(device)

acc_rand, f1_rand = train_and_eval(random_edge_index, "RANDOM")
print(f"❌ Random Graph   -> Accuracy: {acc_rand:.4f} | F1 Score: {f1_rand:.4f}")

# =====================
# 5. SONUÇ YORUMU
# =====================
print("\n" + "="*40)
print("DENEY SONUCU VE YORUMU")
print("="*40)
print(f"Orijinal F1 Score: {f1_orig:.4f}")
print(f"Random F1 Score  : {f1_rand:.4f}")
print(f"Fark (Structure Impact): {f1_orig - f1_rand:.4f}")

if f1_orig > f1_rand + 0.05:
    print("\n✅ BAŞARILI: Model, müşteri memnuniyetini tahmin etmek için")
    print("gerçekten GRAPH YAPISINI (yani arkadaşlık/ürün ilişkilerini) kullanıyor.")
    print("IEEE abstract'ındaki 'intrinsic structures' iddiasını test etmeye hazırız.")
else:
    print("\n⚠️ DİKKAT: Model yapıdan yeterince bilgi alamadı.")
    print("Node feature'ları zenginleştirmemiz veya learning rate ayarı yapmamız gerekebilir.")