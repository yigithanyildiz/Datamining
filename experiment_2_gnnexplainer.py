import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.explain import Explainer, GNNExplainer
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# =====================
# AYARLAR
# =====================
THRESHOLD = 0.5  # Hangi kenarların "önemli" sayılacağı eşiği
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. VERİYİ YÜKLE
try:
    data = torch.load("amazon_office_graph.pt", weights_only=False)
    data = data.to(device)
except FileNotFoundError:
    print("HATA: 'amazon_office_graph.pt' bulunamadı.")
    exit()

# Feature Leakage Önlemi
if data.x.shape[1] > 0: data.x[:, 0] = 0.0 
num_features = data.x.shape[1]
num_classes = 2

# 2. MODELİ KUR & EĞİT
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

print("Model hazırlanıyor ve eğitiliyor...")
model = GCN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
model.train()
train_idx = torch.where(data.y != -1)[0]

for epoch in range(80):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[train_idx], data.y[train_idx])
    loss.backward()
    optimizer.step()

# 3. ZORLU HEDEF SEÇİMİ (High-Degree Node)
# Çatışmayı kanıtlamak için "Başı Kalabalık" birini seçiyoruz.
print("\n🔍 Karmaşık açıklama üretecek aday aranıyor...")
degrees = (data.edge_index[0].unsqueeze(1) == train_idx).sum(dim=0)
# 20-60 arası bağlantısı olanlar idealdir
candidates = train_idx[(degrees > 20) & (degrees < 60)]

if len(candidates) > 0:
    target_node = candidates[torch.randint(0, len(candidates), (1,)).item()].item()
    target_degree = degrees[(train_idx == target_node).nonzero(as_tuple=True)[0]].item()
    print(f"🎯 HEDEF SEÇİLDİ: Node {target_node} (Arkadaş Sayısı: {target_degree})")
else:
    target_node = train_idx[0].item()
    print(f"⚠️ Uygun aday bulunamadı, varsayılan: {target_node}")

# 4. GNNEXPLAINER ÇALIŞTIR
model.eval()
explainer = Explainer(
    model=model, algorithm=GNNExplainer(epochs=200),
    explanation_type='model', node_mask_type='attributes', edge_mask_type='object',
    model_config=dict(mode='multiclass_classification', task_level='node', return_type='raw'),
)
explanation = explainer(data.x, data.edge_index, index=target_node)
edge_mask = explanation.edge_mask

# 5. AĞ GÖRSELLEŞTİRME (Network Visualization)
print("\nGrafik çiziliyor...")
selected_indices = (edge_mask > THRESHOLD).nonzero(as_tuple=True)[0]

# NetworkX Grafiği (Yönsüz)
G_exp = nx.Graph()
src = data.edge_index[0][selected_indices].cpu().numpy()
dst = data.edge_index[1][selected_indices].cpu().numpy()
weights = edge_mask[selected_indices].cpu().detach().numpy()

for u, v, w in zip(src, dst, weights):
    G_exp.add_edge(u, v, weight=w)

# Hedef node'u kesin ekle
if target_node not in G_exp.nodes(): G_exp.add_node(target_node)

# Çizim Ayarları
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G_exp, seed=42, k=0.6) # Düğümleri ferahlat

# Düğümler
nx.draw_networkx_nodes(G_exp, pos, nodelist=[target_node], node_color='#e74c3c', node_size=1200, label=f'Target User')
neighbors = [n for n in G_exp.nodes() if n != target_node]
nx.draw_networkx_nodes(G_exp, pos, nodelist=neighbors, node_color='#3498db', node_size=600, label='Factors')

# Kenarlar
edge_weights_viz = [G_exp[u][v]['weight'] * 3 for u, v in G_exp.edges()]
nx.draw_networkx_edges(G_exp, pos, width=edge_weights_viz, edge_color='#34495e', alpha=0.8)

# Etiketler
nx.draw_networkx_labels(G_exp, pos, font_size=10, font_color='white', font_weight='bold')

visual_edge_count = G_exp.number_of_edges()
plt.title(f"Single-Objective GNNExplainer Output\n(Fixed Output: {visual_edge_count} Edges)", fontsize=14)
plt.legend()
plt.axis('off')

filename = "problem2_conflict_network.png"
plt.savefig(filename, dpi=300)
print(f"✅ Görsel kaydedildi: {filename}")
print(f"👉 Bu grafikte {visual_edge_count} kenar var. Bunu 'Yöneticiye çok, Analiste az' diye sunacaksın.")
plt.show()