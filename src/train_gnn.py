import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import os
import sys

# Add path to import model
sys.path.append(os.path.dirname(__file__))
from gnn_model import SimpleGNN

# Constants
DATASET_PATH = "analysis_results/processed_graphs.pt"
BATCH_SIZE = 32
EPOCHS = 50
LR = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GraphListDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

def collate_fn(batch):
    # Batch is a list of dicts: {'x': ..., 'adj': ..., 'y': ...}
    
    # 1. Concatenate Node Features
    x_list = [item['x'] for item in batch]
    x_batch = torch.cat(x_list, dim=0)
    
    # 2. Create Block Diagonal Adjacency
    # This is the tricky part. 
    # We construct indices and values for a sparse tensor, then convert to dense if needed (or keep sparse)
    # Our model currently assumes dense mm, but for batching, dense block diagonal is HUGE.
    # We MUST use sparse mm in the model or handle it carefully.
    # Let's modify the model to accept sparse adj? 
    # Or just construct a big sparse tensor here.
    
    # Let's try sparse construction
    indices = []
    values = []
    
    offset = 0
    batch_idx = []
    
    for i, item in enumerate(batch):
        adj = item['adj'] # (N, N)
        num_nodes = adj.shape[0]
        
        # Get indices of non-zero elements
        # adj is dense tensor from dataset builder
        rows, cols = torch.nonzero(adj, as_tuple=True)
        vals = adj[rows, cols]
        
        rows += offset
        cols += offset
        
        indices.append(torch.stack([rows, cols]))
        values.append(vals)
        
        # Batch Index
        batch_idx.append(torch.full((num_nodes,), i, dtype=torch.long))
        
        offset += num_nodes
    
    total_nodes = offset
    indices = torch.cat(indices, dim=1)
    values = torch.cat(values)
    
    adj_batch = torch.sparse_coo_tensor(indices, values, (total_nodes, total_nodes))
    
    # 3. Concatenate Labels
    y_batch = torch.cat([item['y'] for item in batch])
    
    # 4. Concatenate Batch Index
    batch_idx_batch = torch.cat(batch_idx)
    
    return x_batch, adj_batch, y_batch, batch_idx_batch

def train(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for x, adj, y, batch_idx in loader:
        x, adj, y, batch_idx = x.to(DEVICE), adj.to(DEVICE), y.to(DEVICE), batch_idx.to(DEVICE)
        
        optimizer.zero_grad()
        out = model(x, adj, batch_idx).squeeze()
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * y.size(0)
        all_preds.extend(out.detach().cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        
    avg_loss = total_loss / len(loader.dataset)
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except: auc = 0.5
    return avg_loss, auc

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, adj, y, batch_idx in loader:
            x, adj, y, batch_idx = x.to(DEVICE), adj.to(DEVICE), y.to(DEVICE), batch_idx.to(DEVICE)
            
            out = model(x, adj, batch_idx).squeeze()
            loss = criterion(out, y)
            
            total_loss += loss.item() * y.size(0)
            all_preds.extend(out.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
    avg_loss = total_loss / len(loader.dataset)
    try:
        auc = roc_auc_score(all_labels, all_preds)
        acc = accuracy_score(all_labels, np.array(all_preds) > 0.5)
    except: 
        auc = 0.5
        acc = 0.5
    return avg_loss, auc, acc

def main():
    print(f"Using device: {DEVICE}")
    
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset not found: {DATASET_PATH}")
        return
        
    print("Loading dataset...")
    data_list = torch.load(DATASET_PATH)
    print(f"Loaded {len(data_list)} graphs.")
    
    # Split
    train_size = int(0.8 * len(data_list))
    val_size = int(0.1 * len(data_list))
    test_size = len(data_list) - train_size - val_size
    
    train_data, val_data, test_data = random_split(data_list, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(GraphListDataset(train_data), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(GraphListDataset(val_data), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(GraphListDataset(test_data), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # Model
    # Determine input dim from first sample
    sample = data_list[0]
    num_features = sample['x'].shape[1]
    
    model = SimpleGNN(num_features, hidden_dim=64).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss()
    
    print("Starting training...")
    best_auc = 0
    for epoch in range(EPOCHS):
        train_loss, train_auc = train(model, train_loader, criterion, optimizer)
        val_loss, val_auc, val_acc = evaluate(model, val_loader, criterion)
        
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {train_loss:.4f} - Train AUC: {train_auc:.4f} - Val Loss: {val_loss:.4f} - Val AUC: {val_auc:.4f} - Val Acc: {val_acc:.4f}")
        
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), "analysis_results/gnn_model.pth")
            
    print(f"Best Val AUC: {best_auc:.4f}")
    
    # Test
    model.load_state_dict(torch.load("analysis_results/gnn_model.pth"))
    test_loss, test_auc, test_acc = evaluate(model, test_loader, criterion)
    print(f"Test Results - Loss: {test_loss:.4f} - AUC: {test_auc:.4f} - Acc: {test_acc:.4f}")

if __name__ == "__main__":
    main()
