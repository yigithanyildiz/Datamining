import torch
import random
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import matplotlib.pyplot as plt  # <--- EKLENDİ (Grafik için)
import numpy as np

# =====================
# Load graph
# =====================
device = torch.device("cpu")
print("Using device:", device)

try:
    data = torch.load("amazon_office_graph.pt", weights_only=False)
    data = data.to(device)
except FileNotFoundError:
    print("HATA: 'amazon_office_graph.pt' bulunamadı.")
    exit()

# =====================
# GCN Model
# =====================
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Feature sayısı 2 değilse burayı data.num_features yapabilirsin
        self.conv1 = GCNConv(data.num_features, 16) 
        self.conv2 = GCNConv(16, 2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# =====================
# Train / Eval Function
# =====================
def train_and_eval(edge_index, label="Graph"):
    print(f"\n--- Training on {label} ---")
    model = GCN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    user_mask = data.y != -1
    user_indices = user_mask.nonzero(as_tuple=True)[0]

    # Basitçe %80 train, %20 test ayıralım
    perm = torch.randperm(len(user_indices))
    train_idx = user_indices[perm[:int(0.8 * len(perm))]]
    test_idx = user_indices[perm[int(0.8 * len(perm)):]]

    # Training
    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, edge_index)
        loss = F.cross_entropy(out[train_idx], data.y[train_idx])
        loss.backward()
        optimizer.step()

    # Evaluation (ACCURACY HESABI - HOCANIN İSTEDİĞİ YER)
    model.eval()
    with torch.no_grad():
        out = model(data.x, edge_index)
        pred = out.argmax(dim=1)
        # İŞTE BURASI ACCURACY:
        acc = (pred[test_idx] == data.y[test_idx]).float().mean().item()
    
    return acc

# =====================
# 1️⃣ Original Graph
# =====================
acc_original = train_and_eval(data.edge_index, "Original Graph")
print(f"Original Graph Accuracy: {acc_original:.4f}")

# =====================
# 2️⃣ Randomized Graph (Structural Leakage Test)
# =====================
edge_index = data.edge_index.clone()
num_edges = edge_index.size(1)
num_nodes = data.num_nodes

# Kenarları rastgele karıştır (Structure'ı yok et)
random_src = torch.randint(0, num_nodes, (num_edges,))
random_dst = torch.randint(0, num_nodes, (num_edges,))
random_edge_index = torch.stack([random_src, random_dst]).to(device)

acc_random = train_and_eval(random_edge_index, "Randomized Graph")
print(f"Randomized Graph Accuracy: {acc_random:.4f}")

# =====================
# 3️⃣ PLOT RESULTS (RAPOR İÇİN GRAFİK)
# =====================
print("\nGrafik çiziliyor...")
methods = ['Original Graph', 'Random Graph']
accuracies = [acc_original, acc_random]
colors = ['#3498db', '#e74c3c'] # Mavi (İyi), Kırmızı (Kötü)

plt.figure(figsize=(8, 6))
bars = plt.bar(methods, accuracies, color=colors, width=0.5)

# Barların üstüne sayıları yazalım
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.2f}", ha='center', fontweight='bold')

plt.ylabel('Test Accuracy')
plt.title('Impact of Structural Information on Model Performance')
plt.ylim(0, 1.0) # 0 ile 1 arası olsun
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Kaydet
plt.savefig('experiment_1_structural_leakage.png', dpi=300)
print("✅ Grafik kaydedildi: experiment_1_structural_leakage.png")
plt.show()