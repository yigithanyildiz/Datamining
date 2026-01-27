import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

data = torch.load("amazon_office_graph.pt", weights_only=False)
data = data.to(device)


user_mask = data.y != -1
user_indices = torch.where(user_mask)[0]


perm = torch.randperm(len(user_indices))
train_size = int(0.8 * len(user_indices))

train_idx = user_indices[perm[:train_size]]
test_idx = user_indices[perm[train_size:]]

print("Train users:", len(train_idx))
print("Test users:", len(test_idx))


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

model = GCN(
    in_channels=data.x.size(1),
    hidden_channels=32,
    out_channels=2
).to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()


def train():
    model.train()
    optimizer.zero_grad()

    out = model(data.x, data.edge_index)
    loss = criterion(out[train_idx], data.y[train_idx])

    loss.backward()
    optimizer.step()

    return loss.item()


def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)

    correct = (pred[test_idx] == data.y[test_idx]).sum()
    acc = int(correct) / len(test_idx)

    return acc


epochs = 100

for epoch in range(1, epochs + 1):
    loss = train()
    acc = test()

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Test Acc: {acc:.4f}")

print("Training finished.")
