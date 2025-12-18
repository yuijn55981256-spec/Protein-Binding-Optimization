import torch
import sys
import os
import csv
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src/module2_model_building")))
try:
    from gnn_model import SimpleGNN
except ImportError:
    print("Could not import SimpleGNN. Check path.")
    sys.exit(1)

import argparse

def main():
    parser = argparse.ArgumentParser(description="Run GNN inference")
    parser.add_argument("--model", default="analysis_results/gnn_model.pth", help="Path to model file")
    parser.add_argument("--dataset", default="analysis_results/homolog_inference_graphs.pt", help="Path to dataset .pt file")
    parser.add_argument("--output", default="analysis_results/homolog_predictions.csv", help="Path to output CSV file")
    args = parser.parse_args()

    model_path = args.model
    dataset_path = args.dataset
    output_path = args.output
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        return

    print("Loading dataset...")
    # Load with map_location to cpu to avoid cuda issues if running on cpu machine
    # But we want to use cuda if available
    data_list = torch.load(dataset_path)
    print(f"Loaded {len(data_list)} graphs.")
    
    if len(data_list) == 0:
        print("No graphs to process.")
        return

    # Load Model
    # Need to know input dim
    sample = data_list[0]
    num_features = sample['x'].shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = SimpleGNN(num_features, hidden_dim=64).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Model not found: {model_path}")
        return
        
    model.eval()
    
    results = []
    
    print("Running inference...")
    with torch.no_grad():
        for i, data in enumerate(tqdm(data_list)):
            x = data['x'].to(device)
            adj = data['adj'].to(device)
            
            # Batch Index (all zeros for single graph)
            batch_idx = torch.zeros(x.size(0), dtype=torch.long, device=device)
            
            try:
                score = model(x, adj, batch_idx).item()
            except RuntimeError as e:
                print(f"Error processing graph {i}: {e}")
                continue
            
            # Get metadata
            protein = data.get('protein_path', 'Unknown')
            ligand = data.get('ligand_path', 'Unknown')
            
            results.append({
                "protein": os.path.basename(protein),
                "ligand": os.path.basename(ligand),
                "score": score,
                "protein_full_path": protein,
                "ligand_full_path": ligand
            })
            
    # Sort by score (descending)
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # Save
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["protein", "ligand", "score", "protein_full_path", "ligand_full_path"])
        writer.writeheader()
        writer.writerows(results)
        
    print(f"Predictions saved to {output_path}")
    
    # Print top 10
    print("\nTop 10 Candidates:")
    for r in results[:10]:
        print(f"{r['protein']} : {r['score']:.4f}")

if __name__ == "__main__":
    main()
