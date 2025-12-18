import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x, adj):
        # x: (N, in_features)
        # adj: (N, N) - normalized adjacency
        
        # H = A * X * W
        # Step 1: X * W
        support = self.linear(x)
        
        # Step 2: A * support
        # Use sparse mm if adj is sparse
        if adj.is_sparse:
            out = torch.sparse.mm(adj, support)
        else:
            out = torch.mm(adj, support)
        
        return out + self.bias

class SimpleGNN(nn.Module):
    def __init__(self, num_node_features, hidden_dim, num_classes=1):
        super(SimpleGNN, self).__init__()
        self.conv1 = GCNLayer(num_node_features, hidden_dim)
        self.conv2 = GCNLayer(hidden_dim, hidden_dim)
        self.conv3 = GCNLayer(hidden_dim, hidden_dim)
        
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, adj, batch_idx):
        # x: (Total_Nodes, Features)
        # adj: (Total_Nodes, Total_Nodes) - Block diagonal
        # batch_idx: (Total_Nodes,) - indicating which graph the node belongs to
        
        x = self.conv1(x, adj)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, adj)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv3(x, adj)
        x = F.relu(x)
        
        # Global Mean Pooling
        # We need to aggregate nodes belonging to the same graph
        # Scatter add or simple loop? 
        # Since we don't have scatter_add from PyG, we implement a simple loop or matrix mult
        
        # Efficient pooling using index_add_ or similar
        batch_size = batch_idx.max().item() + 1
        
        # Create pooling matrix? No, too big.
        # Use simple loop for now (or scatter_add if implemented manually)
        
        # Manual scatter mean
        # out: (Batch_Size, Hidden)
        out = torch.zeros(batch_size, x.size(1), device=x.device)
        counts = torch.zeros(batch_size, 1, device=x.device)
        
        out.index_add_(0, batch_idx, x)
        counts.index_add_(0, batch_idx, torch.ones_like(batch_idx, dtype=torch.float, device=x.device).unsqueeze(1))
        
        out = out / (counts + 1e-6)
        
        x = self.fc1(out)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return torch.sigmoid(x)
