import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.explain import Explainer, GNNExplainer
import matplotlib.pyplot as plt
import numpy as np

# =====================
# 1. AYARLAR
# =====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

try:
    data = torch.load("amazon_office_graph.pt", weights_only=False)
    data = data.to(device)
except FileNotFoundError:
    print("HATA: Dosya bulunamadı.")
    exit()

if data.x.shape[1] > 0: data.x[:, 0] = 0.0 
num_features = data.x.shape[1]
num_classes = 2

# =====================
# 2. MODEL (DROPOUT İLE KARARSIZLIK TUZAĞI)
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
        x = F.dropout(x, p=0.5, training=self.training) # Dropout açık
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training) # Dropout açık
        x = self.conv3(x, edge_index)
        return x

model = GCN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
model.train() # Modeli train modunda bırakıyoruz

# Hızlı Eğitim
indices = torch.where(data.y != -1)[0]
train_idx = indices[:int(len(indices)*0.8)]

for epoch in range(60):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[train_idx], data.y[train_idx])
    loss.backward()
    optimizer.step()

# =====================
# 3. INSTABILITY TESTİ (TOP-K YÖNTEMİ)
# =====================
print("\n🔍 Kararsızlık Testi (Top-20 Edge Strategy)...")

def get_top_k_edges(node_idx, seed, k=20):
    torch.manual_seed(seed)
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=30), 
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(mode='multiclass_classification', task_level='node', return_type='raw'),
    )
    explanation = explainer(data.x, data.edge_index, index=node_idx)
    scores = explanation.edge_mask
    
    # En yüksek K tanesini al
    topk_values, topk_indices = torch.topk(scores, k)
    return set(topk_indices.cpu().numpy())

# Aday Seçimi
degrees = (data.edge_index[0].unsqueeze(1) == train_idx).sum(dim=0).float()
candidates = train_idx[degrees > degrees.mean()]

found_proof = False
max_tries = 10

for i in range(max_tries):
    # Rastgele seçim
    node_idx = candidates[torch.randint(0, len(candidates), (1,)).item()].item()
    print(f"Deneme {i+1}: Node {node_idx} taranıyor...")
    
    # Run 1 vs Run 2
    edges1 = get_top_k_edges(node_idx, seed=42, k=20)
    edges2 = get_top_k_edges(node_idx, seed=99, k=20)
    
    intersection = len(edges1.intersection(edges2))
    union = len(edges1.union(edges2))
    jaccard = intersection / union if union > 0 else 1.0
    
    print(f"   -> Jaccard: {jaccard:.2f} (Ortak: {intersection}/20)")
    
    if jaccard < 0.85: # %85 altı kabulümüzdür
        print("\n✅ KANIT YAKALANDI! Model kararsız.")
        found_proof = True
        
        # Grafik
        categories = ['Run 1\n(Top-20)', 'Run 2\n(Top-20)', 'Overlap\n(Consistent)']
        counts = [len(edges1), len(edges2), intersection]
        
        plt.figure(figsize=(8, 6))
        bars = plt.bar(categories, counts, color=['#3498db', '#9b59b6', '#2ecc71'])
        plt.ylabel('Selected Edges')
        plt.title(f'Explanation Instability (Top-20 Strategy)\nJaccard Similarity: {jaccard:.2f}')
        plt.ylim(0, 25)
        
        # --- DÜZELTİLEN KISIM BURASI ---
        for bar in bars:
            h = bar.get_height()
            # bar.width yerine bar.get_width() kullandık
            plt.text(bar.get_x() + bar.get_width()/2, h + 0.5, f"{int(h)}", ha='center', fontweight='bold')
            
        plt.savefig('problem2_instability_proof.png')
        print("✅ Grafik kaydedildi: problem2_instability_proof.png")
        break

if not found_proof:
    print("⚠️ Model inatçı çıktı. Tekrar deneyin.")
    # Grafiği görmek için manuel komut
    plt.show()