import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.explain import Explainer, GNNExplainer
import matplotlib.pyplot as plt
import numpy as np
import math

# AYARLAR & MODEL (Standart Split ile)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = torch.load("amazon_office_graph.pt", weights_only=False).to(device)
if data.x.shape[1] > 0: data.x[:, 0] = 0.0 
num_features = data.x.shape[1]
num_classes = 2

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

model = GCN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
model.train()
# Dinamik Split
train_idx = torch.where(data.y != -1)[0][:int(len(torch.where(data.y != -1)[0])*0.8)]

for epoch in range(100):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[train_idx], data.y[train_idx])
    loss.backward()
    optimizer.step()

# HEDEF SEÇİMİ (Rastgelelik)
# İstatistiksel olarak "Ortalama üzeri" bağlantısı olanları filtrele
all_degrees = (data.edge_index[0].unsqueeze(1) == train_idx).sum(dim=0).float()
mean_degree = all_degrees.mean().item()
candidates = train_idx[all_degrees > mean_degree] # Ortalamanın üstündekiler

target_node = candidates[torch.randint(0, len(candidates), (1,)).item()].item()
target_degree = (data.edge_index[0] == target_node).sum().item()
print(f"🎯 Hedef Node: {target_node} (Derece: {target_degree})")

# EXPLAINER
explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=200),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=dict(mode='multiclass_classification', task_level='node', return_type='raw'),
)
explanation = explainer(data.x, data.edge_index, index=target_node)
# Dinamik Threshold: Top-25%
threshold = torch.quantile(explanation.edge_mask, 0.75).item()
selected_edges = (explanation.edge_mask > threshold).sum().item()
print(f"📊 Model Çıktısı: {selected_edges} kenar")

# --- BİLİMSEL SKORLAMA ---

# 1. MANAGER: MILLER'S LAW (7 ± 2 Rule)
# Bilişsel psikolojide kısa süreli hafıza limiti 5-9 arasıdır.
# Biz "Cognitive Load Limit" olarak alt sınır olan 5'i referans alıyoruz.
COGNITIVE_LIMIT = 5 
SIGMA = 10.0 # Tolerans (Gaussian genişliği)

if selected_edges <= COGNITIVE_LIMIT:
    manager_score = 1.0
else:
    # Gaussian Decay ile bilimsel düşüş
    diff = selected_edges - COGNITIVE_LIMIT
    manager_score = math.exp(- (diff**2) / (2 * SIGMA**2))

# 2. ANALYST: INFORMATION RECALL (Bilgi Kapsama)
# Analist, grafiğin en az %50'sinin (Majority Context) korunmasını ister.
# Bu keyfi bir sayı değil, "Majority Voting" mantığıdır.
REQUIRED_COVERAGE = int(target_degree * 0.50) 
if REQUIRED_COVERAGE < 5: REQUIRED_COVERAGE = 5 # Minimum mantıklı sınır

if selected_edges >= REQUIRED_COVERAGE:
    analyst_score = 1.0
else:
    analyst_score = selected_edges / REQUIRED_COVERAGE

print("\n🧮 BİLİMSEL SKORLAR:")
print(f"   -> Manager (Miller's Law Limit: {COGNITIVE_LIMIT}): {manager_score:.2f}")
print(f"   -> Analyst (Coverage Target >{REQUIRED_COVERAGE}): {analyst_score:.2f}")

# GRAFİK (Aynı kalabilir, veri artık bilimsel)
users = ['Manager\n(Minimizer)', 'Analyst\n(Maximizer)']
scores = [manager_score, analyst_score]
colors = ['#e74c3c' if s < 0.6 else '#2ecc71' for s in scores]
plt.figure(figsize=(8, 6))
plt.bar(users, scores, color=colors, width=0.5)
plt.title(f'Scientific Conflict Analysis\n(Miller\'s Law vs Information Recall)')
plt.ylabel('Utility Score')
plt.savefig('problem2_ideal_conflict.png')
print("✅ Grafik oluşturuldu.")