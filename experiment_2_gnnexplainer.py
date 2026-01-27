import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.explain import Explainer, GNNExplainer
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx, k_hop_subgraph

# =====================
# 1. AYARLAR & YÜKLEME
# =====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

try:
    data = torch.load("amazon_office_graph.pt", weights_only=False)
    data = data.to(device)
except FileNotFoundError:
    print("HATA: 'amazon_office_graph.pt' bulunamadı.")
    exit()

# --- BİLİMSEL KONTROL: Feature Leakage Önlemi ---
# (Step 1'deki gibi puanı siliyoruz ki model yapıya baksın)
if data.x.shape[1] > 0:
    data.x[:, 0] = 0.0 

num_features = data.x.shape[1]
num_classes = 2

# =====================
# 2. MODELİ HAZIRLA (Step 1'deki Modelin Aynısı)
# =====================
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(num_features, 64)
        self.conv2 = GCNConv(64, 32)
        self.conv3 = GCNConv(32, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        return x

print("Model hazırlanıyor ve hızlıca eğitiliyor...")
model = GCN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# Explanation için modelin biraz eğitilmiş olması lazım
# (Step 1'de kanıtladığımız yapıyı öğrensin)
model.train()
# Sadece valid user'ları al
valid_mask = data.y != -1
train_idx = torch.where(valid_mask)[0]

# Hızlı eğitim (Explanation testi için)
for epoch in range(100):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[train_idx], data.y[train_idx])
    loss.backward()
    optimizer.step()

# =====================
# 3. HEDEF KULLANICI SEÇİMİ
# =====================
model.eval()
out = model(data.x, data.edge_index)
pred = out.argmax(dim=1)

# Modelin "Memnun (1)" dediği ve Gerçekte de "Memnun (1)" olan birini bul
target_node = -1
for i in train_idx:
    if data.y[i] == 1 and pred[i] == 1:
        # Biraz bağlantısı olan birini seçelim ki görsel güzel olsun (degree > 2)
        degree = (data.edge_index[0] == i).sum().item()
        if degree > 2 and degree < 20: # Çok kalabalık da olmasın
            target_node = i.item()
            break

if target_node == -1:
    print("Uygun hedef kullanıcı bulunamadı, rastgele biri seçiliyor.")
    target_node = train_idx[0].item()

print(f"\n🎯 Hedef Kullanıcı Node ID: {target_node}")
print(f"Gerçek Etiket: {data.y[target_node].item()} | Tahmin: {pred[target_node].item()}")

# =====================
# 4. GNNEXPLAINER (Single-Objective)
# =====================
print("\n🔍 GNNExplainer Çalıştırılıyor (Single-Objective: Fidelity)...")

explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=200),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=dict(
        mode='multiclass_classification',
        task_level='node',
        return_type='raw',
    ),
)

# Açıklamayı üret
explanation = explainer(
    x=data.x,
    edge_index=data.edge_index,
    index=target_node
)

# =====================
# 5. GÖRSELLEŞTİRME VE ANALİZ
# =====================
# Önemli kenarları seç (Threshold: 0.5 üstü)
edge_mask = explanation.edge_mask
important_edges_mask = edge_mask > 0.5
num_important = important_edges_mask.sum().item()

print(f"\n📊 Analiz Sonuçları:")
print(f"Toplam Komşuluk (Edge): {edge_mask.shape[0]}")
print(f"Explanation İçin Seçilen Edge Sayısı: {num_important}")

# Görselleştirme (Subgraph)
# Sadece hedef node ve onun 2-hop komşularını al
subset, sub_edge_index, mapping, _ = k_hop_subgraph(
    target_node, 2, data.edge_index, relabel_nodes=True
)

# NetworkX'e çevir
data_sub = torch.load("amazon_office_graph.pt", weights_only=False) # Featurelar için tekrar yükle
g = to_networkx(data, to_undirected=True)
sub_g = g.subgraph(subset.tolist())

plt.figure(figsize=(10, 8))
pos = nx.spring_layout(sub_g, seed=42)

# Tüm düğümleri gri çiz
nx.draw_networkx_nodes(sub_g, pos, node_size=100, node_color='#bdc3c7', alpha=0.5)
# Hedef düğümü Kırmızı çiz
nx.draw_networkx_nodes(sub_g, pos, nodelist=[target_node], node_size=300, node_color='#e74c3c')
# Tüm kenarları silik çiz
nx.draw_networkx_edges(sub_g, pos, alpha=0.1)

# --- Explanation Edges ---
# GNNExplainer'ın "önemli" dediği kenarları bulup üstüne çizelim
# (Mapping işlemi karmaşık olduğu için burada basitleştirilmiş görselleştirme yapıyoruz)
# Bu demo görselidir, raporda "önemli kenar sayısı" verisini kullanacağız.

plt.title(f"GNNExplainer Result for User {target_node}\nSelected Edges: {num_important}", fontsize=14)
plt.axis('off')
plt.savefig("gnn_explainer_result.png")
print("✅ Görsel kaydedildi: gnn_explainer_result.png")

print("\n--- YORUM ---")
if num_important < 2:
    print("👉 SONUÇ: Explanation çok 'Sparse' (Seyrek). Belki yetersiz bilgi veriyor.")
    print("Farklı bir run yaparsak sonuç değişebilir (Instability).")
elif num_important > 10:
    print("👉 SONUÇ: Explanation çok 'Dense' (Yoğun). Okunabilirliği düşük.")
    print("Yönetici (Manager) bu açıklamayı anlamaz (User Requirement Conflict).")
else:
    print("👉 SONUÇ: Makul bir açıklama. Ancak sadece 'Fidelity' odaklı.")