
import json
import pandas as pd
import torch
from torch_geometric.data import Data

JSON_PATH = "Office_Products_5.json"  
GRAPH_SAVE_PATH = "amazon_office_graph.pt"

def load_amazon_json(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)


print("Loading JSON...")
df = load_amazon_json(JSON_PATH)
print("Total reviews:", len(df))



def label_satisfaction(rating):
    if rating >= 4:
        return 1   # satisfied
    elif rating <= 2:
        return 0   # unsatisfied
    else:
        return -1  # neutral (drop)


df["label"] = df["overall"].apply(label_satisfaction)


df = df[df["label"] != -1]
print("After removing neutral reviews:", len(df))



df = df[["reviewerID", "asin", "overall", "label", "helpful"]]

user_ids = df["reviewerID"].unique()
product_ids = df["asin"].unique()

user_map = {uid: i for i, uid in enumerate(user_ids)}
product_map = {pid: i + len(user_ids) for i, pid in enumerate(product_ids)}

num_nodes = len(user_ids) + len(product_ids)
print("Users:", len(user_ids))
print("Products:", len(product_ids))
print("Total nodes:", num_nodes)

edges = []
edge_weights = []

for _, row in df.iterrows():
    u = user_map[row["reviewerID"]]
    p = product_map[row["asin"]]


    edges.append([u, p])
    edges.append([p, u])

    helpful = row["helpful"]
    weight = helpful[0] / helpful[1] if helpful[1] > 0 else 0.0

    edge_weights.append(weight)
    edge_weights.append(weight)


edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
edge_attr = torch.tensor(edge_weights, dtype=torch.float)

print("Total edges:", edge_index.shape[1])

x = torch.zeros((num_nodes, 2))

for uid in user_ids:
    idx = user_map[uid]
    user_reviews = df[df["reviewerID"] == uid]
    x[idx, 0] = 0.0
    x[idx, 1] = len(user_reviews)

for pid in product_ids:
    idx = product_map[pid]
    product_reviews = df[df["asin"] == pid]
    x[idx, 0] = 0.0
    x[idx, 1] = len(product_reviews)


y = torch.full((num_nodes,), -1, dtype=torch.long)

for uid in user_ids:
    idx = user_map[uid]
    user_reviews = df[df["reviewerID"] == uid]
    y[idx] = user_reviews["label"].mode()[0]


data = Data(
    x=x,
    edge_index=edge_index,
    edge_attr=edge_attr,
    y=y
)

print(data)


torch.save(data, GRAPH_SAVE_PATH)
print(f"Graph saved to {GRAPH_SAVE_PATH}")
