import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.explain import Explainer, GNNExplainer
import matplotlib.pyplot as plt
import numpy as np
import math

# =====================
# 1. AYARLAR & MODEL (Standart Kısım)
# =====================
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
train_idx = torch.where(data.y != -1)[0]
for epoch in range(100):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[train_idx], data.y[train_idx])
    loss.backward()
    optimizer.step()

# =====================
# 2. HEDEF SEÇİMİ
# =====================
degrees = (data.edge_index[0].unsqueeze(1) == train_idx).sum(dim=0)
candidates = train_idx[(degrees > 30) & (degrees < 60)]
if len(candidates) > 0:
    target_node = candidates[torch.randint(0, len(candidates), (1,)).item()].item()
else:
    target_node = train_idx[0].item()

# =====================
# 3. GNNEXPLAINER ÇALIŞTIR
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
selected_edges = (explanation.edge_mask > 0.4).sum().item()
print(f"📊 Model Çıktısı: {selected_edges} kenar")

# =====================
# 4. İDEAL SKORLAMA YÖNTEMİ (UTILITY FUNCTIONS)
# =====================

# --- MANAGER: GAUSSIAN DECAY (Çan Eğrisi Düşüşü) ---
# Bilimsel Açıklama: İnsanlar limit aşılınca hemen nefret etmez.
# Tolerans yavaş yavaş azalır, sonra hızla düşer.
MANAGER_LIMIT = 5
SIGMA = 10.0 # Hoşgörü katsayısı (Ne kadar geniş o kadar hoşgörülü)

if selected_edges <= MANAGER_LIMIT:
    # Limitin altındaysa mükemmel
    manager_score = 1.0
else:
    # Limit aşıldıysa Gaussian Düşüş başlar
    # Formül: exp( - (fark)^2 / (2 * sigma^2) )
    diff = selected_edges - MANAGER_LIMIT
    manager_score = math.exp(- (diff**2) / (2 * SIGMA**2))

# --- ANALYST: SATURATION CURVE (Doygunluk Eğrisi) ---
# Bilimsel Açıklama: 15 isterim ama 14 de olur (%93). 
# 15'ten sonrası benim için fark etmez (1.0).
ANALYST_TARGET = 15

if selected_edges >= ANALYST_TARGET:
    analyst_score = 1.0
else:
    # Hedefe ne kadar yaklaştık?
    # Basit lineer oran analist için mantıklıdır.
    analyst_score = selected_edges / ANALYST_TARGET

print("\n🧮 İDEAL SKORLAR (Bilimsel):")
print(f"   -> Manager (Gaussian Decay): {manager_score:.2f}")
print(f"      (Limit: {MANAGER_LIMIT}, Aşan Miktar: {max(0, selected_edges - MANAGER_LIMIT)})")
print(f"   -> Analyst (Saturation): {analyst_score:.2f}")
print(f"      (Target: {ANALYST_TARGET})")

# =====================
# 5. GRAFİK
# =====================
users = ['Manager\n(Minimizer)', 'Analyst\n(Maximizer)']
scores = [manager_score, analyst_score]
colors = ['#e74c3c' if s < 0.6 else '#2ecc71' for s in scores]

plt.figure(figsize=(8, 6))
bars = plt.bar(users, scores, color=colors, width=0.5)

plt.ylabel('User Utility Score (0-1)')
plt.title(f'Ideal Conflict Quantification\n(Model Output: {selected_edges} edges)')
plt.ylim(0, 1.1)
plt.axhline(0.6, color='gray', linestyle='--', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    # Emojili durum
    if height > 0.85: txt = "Excellent 🤩"
    elif height > 0.6: txt = "Acceptable 🙂"
    else: txt = "Poor 😡"
    
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{txt}\n({height:.2f})', ha='center', va='bottom', fontweight='bold')

plt.savefig('problem2_ideal_conflict.png')
print("\n✅ Grafik kaydedildi: problem2_ideal_conflict.png")
plt.show()