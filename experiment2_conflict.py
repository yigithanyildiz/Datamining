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
data = torch.load("amazon_office_graph.pt", weights_only=False)
data = data.to(device)

if data.x.shape[1] > 0:
    data.x[:, 0] = 0.0  # Feature Leakage Önlemi

num_features = data.x.shape[1]
num_classes = 2

# =====================
# 2. MODELİ KUR (Eğitilmiş Halini Varsayıyoruz)
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
        x = self.conv3(x, edge_index)
        return x

model = GCN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# Hızlı tekrar eğitim
model.train()
user_mask = data.y != -1
train_idx = torch.where(user_mask)[0][:1000]
for epoch in range(80):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[train_idx], data.y[train_idx])
    loss.backward()
    optimizer.step()

# =====================
# 3. HEDEF NODE ANALİZİ (Node 677)
# =====================
# Senin loglarında bulduğun 'Degree 89' olan node'u kullanalım
target_node = 677 
print(f"🎯 Hedef Node: {target_node} inceleniyor...")

explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=200),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=dict(mode='multiclass_classification', task_level='node', return_type='raw'),
)

explanation = explainer(data.x, data.edge_index, index=target_node)

# Seçilen Önemli Kenarlar (Threshold > 0.5)
mask = explanation.edge_mask
selected_edges_count = (mask > 0.5).sum().item()
total_neighbors = (data.edge_index[0] == target_node).sum().item() # Doğrudan komşular

print(f"\n📊 GNNExplainer Sonuçları:")
print(f"Toplam Komşu Sayısı: {total_neighbors}")
print(f"Önemli Diye Seçilen Kenar Sayısı: {selected_edges_count}")

# =====================
# 4. USER REQUIREMENT CONFLICT ANALİZİ
# =====================
# Senaryo:
# Manager: En fazla 5 kenar okuyabilir. (Sparsity odaklı)
# Analyst: Tüm detayları ister. (Fidelity odaklı)

manager_limit = 5
analyst_limit = 50

print("\n⚡ KULLANICI ÇATIŞMASI ANALİZİ (User Conflict Test):")

# Manager Memnuniyeti
if selected_edges_count > manager_limit:
    print(f"❌ MANAGER: 'Bu açıklama çok karışık! Ben max {manager_limit} istedim, sen {selected_edges_count} verdin.'")
    print("   -> Manager Satisfaction: DÜŞÜK")
else:
    print(f"✅ MANAGER: 'Teşekkürler, {selected_edges_count} kenar tam bana göre.'")

# Analyst Memnuniyeti
if selected_edges_count > 10:
    print(f"✅ ANALYST: 'Güzel, {selected_edges_count} kenar ile detaylı bir analiz yapabilirim.'")
    print("   -> Analyst Satisfaction: YÜKSEK")
else:
    print(f"❌ ANALYST: 'Bu ne? Sadece {selected_edges_count} kenar var, detaylar kaybolmuş!'")

# =====================
# 5. GRAFİK: TEK TİP ÇÖZÜMÜN SORUNU
# =====================
# Bu grafik, tek bir explanation'ın (GNNExplainer çıktısının) 
# farklı kullanıcıları nasıl tatmin edemediğini gösterir.

users = ['Manager\n(İster: <5 Edge)', 'Analyst\n(İster: >10 Edge)', 'Customer\n(İster: Basit)']
# Skorlama mantığı (Basit simülasyon)
# Explanation size (örneğin 20) Manager için kötü (0.2), Analyst için iyi (0.9)
size = selected_edges_count

# Basit bir memnuniyet fonksiyonu uyduralım
manager_score = max(0, 1 - (size - 5)/20) if size > 5 else 1.0
analyst_score = min(1.0, size / 15)
customer_score = max(0, 1 - (size - 3)/10) if size > 3 else 1.0

scores = [manager_score, analyst_score, customer_score]

plt.figure(figsize=(8, 6))
bars = plt.bar(users, scores, color=['#e74c3c', '#2ecc71', '#f1c40f'])
plt.ylabel('Kullanıcı Memnuniyet Skoru (0-1)')
plt.title(f'Problem 2: User Requirement Conflict\n(Explanation Size: {size} Edges)')
plt.ylim(0, 1.1)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{height:.2f}', ha='center', va='bottom', fontweight='bold')

plt.axhline(y=0.5, color='gray', linestyle='--')
plt.savefig('problem2_user_conflict.png')
print("\n✅ Grafik kaydedildi: problem2_user_conflict.png")
print("Bu grafik, GNNExplainer'ın Analyst'i mutlu ederken Manager'ı mutsuz ettiğini kanıtlar.")
plt.show()