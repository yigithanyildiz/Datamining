import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.utils import k_hop_subgraph, to_networkx
import networkx as nx
import matplotlib.pyplot as plt

# =====================
# 1. AYARLAR & YÜKLEME
# =====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

try:
    data = torch.load("amazon_office_graph.pt", weights_only=False).to(device)
except FileNotFoundError:
    print("HATA: 'amazon_office_graph.pt' bulunamadı.")
    exit()

# Feature Leakage Önlemi
if data.x.shape[1] > 0: data.x[:, 0] = 0.0 
num_features = data.x.shape[1]
num_classes = 2

# =====================
# 2. MODEL EĞİTİMİ
# =====================
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(num_features, 64)
        self.conv2 = GCNConv(64, 32)
        self.conv3 = GCNConv(32, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.conv3(x, edge_index)

print("Model eğitiliyor...")
model = GCN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
model.train()
train_idx = torch.where(data.y != -1)[0]

for epoch in range(100):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[train_idx], data.y[train_idx])
    loss.backward()
    optimizer.step()

# =====================
# 3. HEDEF SEÇİMİ (DISRUPTION TESTİ İÇİN)
# =====================
# Sadece 1. dereceden (doğrudan) komşusu 15-50 arası olanları seçelim.
# Çok büyük olursa görselleşmez, çok küçük olursa parçalanmaz.
degrees = (data.edge_index[0].unsqueeze(1) == train_idx).sum(dim=0)
candidates = train_idx[(degrees > 15) & (degrees < 50)]

if len(candidates) > 0:
    target_node = candidates[torch.randint(0, len(candidates), (1,)).item()].item()
else:
    target_node = train_idx[0].item()

print(f"\n🎯 Hedef Node: {target_node}")

# =====================
# 4. GNNEXPLAINER & SUBGRAPH ANALİZİ (DÜZELTİLMİŞ KISIM)
# =====================
model.eval()
explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=200),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=dict(mode='multiclass_classification', task_level='node', return_type='raw'),
)

print("Açıklama üretiliyor...")
explanation = explainer(data.x, data.edge_index, index=target_node)

# --- KRİTİK DÜZELTME: Sadece 2-Hop Subgraph'a Bakıyoruz ---
# Model 2 katmanlı olduğu için "Görüş Alanı" (Receptive Field) 2-hop'tur.
# Analizi tüm grafikte değil, sadece bu alanda yapmalıyız.
subset, sub_edge_index, mapping, edge_mask_in_subgraph = k_hop_subgraph(
    target_node, num_hops=2, edge_index=data.edge_index
)

print(f"📍 Analiz Alanı (2-Hop Neighborhood): {len(subset)} Node")

# ORİJİNAL DURUM (Maskesiz)
G_orig = nx.Graph()
# Node'ları ID'leriyle ekle ki karışmasın
G_orig.add_nodes_from(subset.cpu().numpy())
G_orig.add_edges_from(sub_edge_index.t().cpu().numpy())

# EXPLANATION DURUMU (Maskeli)
# Sadece subgraph içindeki kenarların önem skorlarını alıyoruz
sub_weights = explanation.edge_mask[edge_mask_in_subgraph]
THRESHOLD = 0.5
mask = sub_weights > THRESHOLD
filtered_edges = sub_edge_index[:, mask] # Sadece önemli kenarlar kalıyor

G_expl = nx.Graph()
G_expl.add_nodes_from(subset.cpu().numpy()) # Node'lar aynı kalıyor (silinmiyor)
G_expl.add_edges_from(filtered_edges.t().cpu().numpy()) # Sadece seçilen kenarlar ekleniyor

# =====================
# 5. SONUÇLARI HESAPLA
# =====================
comp_orig = nx.number_connected_components(G_orig)
comp_expl = nx.number_connected_components(G_expl)

print("\n" + "="*40)
print("DENEY SONUCU: YAPI BOZULMASI (DISRUPTION)")
print("="*40)
print(f"Analyzed Graph Size (Nodes): {len(subset)}")
print(f"Original Connected Components: {comp_orig} (Bütünlük Tam)")
print(f"Explanation Connected Components: {comp_expl}")

if comp_expl > comp_orig * 2:
    print(f"✅ KANITLANDI: Mahalle {comp_expl} parçaya bölündü (Fragmentation)!")
    print("   -> Bu, açıklamanın bağlamı kopardığını (Loss of Context) gösterir.")
else:
    print("⚠️ Yeterince parçalanma olmadı, threshold'u artırmayı dene.")

# =====================
# 6. GÖRSELLEŞTİRME (OPSİYONEL - RAPOR İÇİN)
# =====================
plt.figure(figsize=(12, 5))

# Sol: Orijinal
plt.subplot(1, 2, 1)
pos = nx.spring_layout(G_orig, seed=42)
nx.draw(G_orig, pos, node_size=20, node_color='#bdc3c7', alpha=0.5)
nx.draw_networkx_nodes(G_orig, pos, nodelist=[target_node], node_color='red', node_size=100)
plt.title(f"Original Neighborhood\n({comp_orig} Component)")

# Sağ: Explanation
plt.subplot(1, 2, 2)
nx.draw(G_expl, pos, node_size=20, node_color='#bdc3c7', alpha=0.5) # Node'lar silik
nx.draw_networkx_edges(G_expl, pos, edge_color='#e74c3c', width=2) # Kenarlar belirgin
nx.draw_networkx_nodes(G_expl, pos, nodelist=[target_node], node_color='red', node_size=100)
plt.title(f"Explanation Output\n({comp_expl} Components - Fragmented)")

plt.savefig("experiment3_disruption_proof.png", dpi=300)
print("✅ Grafik kaydedildi: experiment3_disruption_proof.png")
plt.show()