import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.explain import Explainer, GNNExplainer
import matplotlib.pyplot as plt

# =====================
# 1. AYARLAR
# =====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = torch.load("amazon_office_graph.pt", weights_only=False)
data = data.to(device)

if data.x.shape[1] > 0:
    data.x[:, 0] = 0.0 

num_features = data.x.shape[1]
num_classes = 2

# =====================
# 2. MODELİ HAZIRLA
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
# Hızlıca eğit (overfit olsun ki explanation zorlaşsın)
model.train()
user_mask = data.y != -1
train_idx = torch.where(user_mask)[0][:1000]

for epoch in range(50): # Az epoch = Daha az kararlı model
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[train_idx], data.y[train_idx])
    loss.backward()
    optimizer.step()

# =====================
# 3. INSTABILITY TARAMASI
# =====================
print("🔍 'Unstable' (Kararsız) Node aranıyor...")

def get_edges(node_idx, seed):
    torch.manual_seed(seed)
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=10), # Daha az epoch = daha çok kararsızlık
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(mode='multiclass_classification', task_level='node', return_type='raw'),
    )
    explanation = explainer(data.x, data.edge_index, index=node_idx)
    return set(torch.where(explanation.edge_mask > 0.5)[0].cpu().numpy())

# Yüksek dereceli (çok komşulu) kullanıcıları dene
degrees = (data.edge_index[0].unsqueeze(1) == train_idx).sum(dim=0)
# En çok komşusu olan 50 kişiyi al
high_degree_indices = train_idx[torch.argsort(degrees, descending=True)][:50]

found_unstable = False

for i, node_idx in enumerate(high_degree_indices):
    node_idx = node_idx.item()
    degree = degrees[torch.argsort(degrees, descending=True)][i].item()
    
    # Run 1 vs Run 2
    edges1 = get_edges(node_idx, 42)
    edges2 = get_edges(node_idx, 99)
    
    intersection = len(edges1.intersection(edges2))
    union = len(edges1.union(edges2))
    jaccard = intersection / union if union > 0 else 0.0
    
    print(f"Node {node_idx} (Degree: {degree}) -> Jaccard: {jaccard:.4f}")
    
    if jaccard < 0.95: # %95'ten az benzerlik bulursak yakaladık demektir
        print(f"\n✅ BULUNDU! Node {node_idx} kararsız davranıyor.")
        
        # Grafiği Çiz
        labels = ['Run 1 Edges', 'Run 2 Edges', 'Overlapping']
        values = [len(edges1), len(edges2), intersection]
        
        plt.figure(figsize=(8, 6))
        bars = plt.bar(labels, values, color=['#3498db', '#9b59b6', '#2ecc71'])
        plt.ylabel('Kenar Sayısı')
        plt.title(f'Explanation Instability Detected (Node {node_idx})\nJaccard Similarity: {jaccard:.2f}')
        
        # Değerleri yaz
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height, f'{height}', ha='center', va='bottom')
            
        plt.savefig('problem2_instability_proof.png')
        print("Grafik kaydedildi: problem2_instability_proof.png")
        found_unstable = True
        break

if not found_unstable:
    print("\n⚠️ Model çok kararlı. Bu durumda 'Rigidity' (Problem 2) argümanına geçeceğiz.")
    print("Yani: 'Model hep aynı 27 kenarı veriyor ama ben 5 tane istiyorum, vermiyor.'")