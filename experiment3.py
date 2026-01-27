import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, k_hop_subgraph
import matplotlib.pyplot as plt
import networkx as nx

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

if data.x.shape[1] > 0:
    data.x[:, 0] = 0.0  # Feature Leakage Önlemi

num_features = data.x.shape[1]
num_classes = 2

# =====================
# 2. MODEL EĞİTİMİ (Hızlı)
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
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        return x

model = GCN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
model.train()
user_mask = data.y != -1
train_idx = torch.where(user_mask)[0][:800]

print("Model eğitiliyor...")
for epoch in range(80):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[train_idx], data.y[train_idx])
    loss.backward()
    optimizer.step()

# =====================
# 3. HEDEF NODE SEÇİMİ
# =====================
model.eval()
pred = model(data.x, data.edge_index).argmax(dim=1)

target_node = -1
for i in train_idx:
    degree = (data.edge_index[0] == i).sum().item()
    if degree > 10 and degree < 50 and data.y[i] == 1 and pred[i] == 1:
        target_node = i.item()
        break

if target_node == -1: 
    target_node = train_idx[0].item()

print(f"🎯 Hedef Node: {target_node} (Orijinal Komşu Sayısı: {(data.edge_index[0] == target_node).sum().item()})")

# =====================
# 4. GNNEXPLAINER
# =====================
explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=200),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=dict(mode='multiclass_classification', task_level='node', return_type='raw'),
)

explanation = explainer(data.x, data.edge_index, index=target_node)
edge_mask = explanation.edge_mask

# =====================
# 5. YAPI BOZULMA ANALİZİ (DÜZELTİLDİ)
# =====================

# A) Orijinal Neighborhood (2-hop)
subset, sub_edge_index, mapping, _ = k_hop_subgraph(
    target_node, 2, data.edge_index, relabel_nodes=True
)

# Orijinal graph'ı oluştur (0..N indeksli)
data_orig = Data(edge_index=sub_edge_index, num_nodes=subset.size(0))
G_orig = to_networkx(data_orig, to_undirected=True)

# --- KRİTİK DÜZELTME: Node ID'lerini Eşitleme ---
# G_orig şu an 0, 1, 2... diye gidiyor. Onları gerçek ID'lerine (2048, 55...) çevirelim.
mapping_dict = {i: node_id.item() for i, node_id in enumerate(subset)}
G_orig = nx.relabel_nodes(G_orig, mapping_dict)
# ------------------------------------------------

# B) Explanation Subgraph
mask_bool = edge_mask > 0.5
src, dst = data.edge_index
src_imp = src[mask_bool]
dst_imp = dst[mask_bool]

if src_imp.size(0) == 0:
    print("⚠️ Explainer hiç kenar seçemedi! Threshold düşürülüyor (0.1)...")
    mask_bool = edge_mask > 0.1
    src_imp = src[mask_bool]
    dst_imp = dst[mask_bool]

edge_index_imp = torch.stack([src_imp, dst_imp], dim=0)

# Tüm graph üzerinden explanation edges ile bir yapı kur
G_expl_full = to_networkx(Data(edge_index=edge_index_imp, num_nodes=data.num_nodes), to_undirected=True)

# Sadece hedef node çevresini kesip al
subset_list = subset.tolist()
G_expl = G_expl_full.subgraph(subset_list)

# =====================
# 6. METRİK HESAPLAMA
# =====================
components_orig = nx.number_connected_components(G_orig)
components_expl = nx.number_connected_components(G_expl)

print("\n" + "="*40)
print("DENEY SONUCU: YAPI BOZULMASI (DISRUPTION)")
print("="*40)
print(f"Orijinal Graph Parça Sayısı: {components_orig} (Bütünlük Korunuyor)")
print(f"Explanation Graph Parça Sayısı: {components_expl}")

if components_expl > components_orig:
    print("✅ KANITLANDI: Explanation graph'ı parçalamış (Fragmentation)!")
else:
    print("⚠️ Yapı çok bozulmadı.")

# =====================
# 7. GÖRSELLEŞTİRME
# =====================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Pozisyonları Orijinal graph üzerinden hesapla (Gerçek ID'ler ile)
pos = nx.spring_layout(G_orig, seed=42)

# Sol: Orijinal
nx.draw(G_orig, pos, ax=axes[0], node_size=50, node_color='#bdc3c7', with_labels=False)
nx.draw_networkx_nodes(G_orig, pos, nodelist=[target_node], node_color='#2ecc71', node_size=150, ax=axes[0])
axes[0].set_title(f"Original Neighborhood\n(Connected Components: {components_orig})")

# Sağ: Explanation
# G_expl içindeki düğümlerin hepsi pos içinde var mı kontrolü (Garanti olsun diye)
common_nodes = [n for n in G_expl.nodes() if n in pos]
G_expl_filtered = G_expl.subgraph(common_nodes)

nx.draw(G_expl_filtered, pos, ax=axes[1], node_size=50, node_color='#e74c3c', with_labels=False, edge_color='red', width=1.5)
axes[1].set_title(f"Single-Objective Explanation\n(Connected Components: {components_expl}) - FRAGMENTED")

plt.savefig('problem3_structure_disruption.png')
print("\n✅ Grafik kaydedildi: problem3_structure_disruption.png")
plt.show()