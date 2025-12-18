import os
import torch
import numpy as np
import csv
from tqdm import tqdm
import math

# Constants
ELEMENTS = {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'P': 4, 'OTHER': 5}
NUM_NODE_FEATURES = len(ELEMENTS) + 1 # +1 for is_ligand flag
DISTANCE_THRESHOLD = 4.5

def one_hot_element(element):
    vec = [0] * len(ELEMENTS)
    idx = ELEMENTS.get(element.upper(), ELEMENTS['OTHER'])
    vec[idx] = 1
    return vec

def parse_pdb_atoms(pdb_path):
    atoms = []
    if not os.path.exists(pdb_path):
        return atoms
    
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                element = line[76:78].strip()
                if not element:
                    element = line[12:14].strip() # Fallback to atom name
                    element = ''.join([c for c in element if c.isalpha()])[:1]
                
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    atoms.append({'element': element, 'coords': (x, y, z), 'is_ligand': 0})
                except: pass
    return atoms

def parse_mol2_atoms(mol2_path):
    atoms = []
    if not os.path.exists(mol2_path):
        return atoms
        
    with open(mol2_path, 'r') as f:
        lines = f.readlines()
        
    in_atom = False
    for line in lines:
        if line.startswith("@<TRIPOS>ATOM"):
            in_atom = True
            continue
        if line.startswith("@<TRIPOS>BOND"):
            break
            
        if in_atom:
            parts = line.split()
            if len(parts) >= 6:
                try:
                    x = float(parts[2])
                    y = float(parts[3])
                    z = float(parts[4])
                    atom_type = parts[5]
                    element = atom_type.split('.')[0]
                    atoms.append({'element': element, 'coords': (x, y, z), 'is_ligand': 1})
                except: pass
    return atoms

def build_graph(protein_path, ligand_path, label):
    # 1. Parse Atoms
    # For protein, we only want the pocket. 
    # Since we don't have a pocket file for all, we might need to use the full protein 
    # BUT full protein is too big for GNN.
    # Luckily, the manifest has 'pocket_path' which is the pocket.pdb!
    # Wait, let me check the manifest columns again.
    
    # If pocket_path exists, use it. Else use protein_path but filter by distance to ligand?
    # The manifest has 'pocket_path'.
    
    p_atoms = parse_pdb_atoms(protein_path)
    l_atoms = parse_mol2_atoms(ligand_path)
    
    if not p_atoms or not l_atoms:
        print(f"Parse failed. P: {len(p_atoms) if p_atoms else 0}, L: {len(l_atoms) if l_atoms else 0}")
        return None

    # Filter protein atoms to keep only those near ligand (e.g. 10A)
    # This effectively extracts the pocket on the fly
    l_coords = np.array([a['coords'] for a in l_atoms])
    p_coords = np.array([a['coords'] for a in p_atoms])
    
    # Calculate min distance from each protein atom to ANY ligand atom
    # Optimization: Use simple bounding box check first
    l_min = l_coords.min(axis=0) - 6.0
    l_max = l_coords.max(axis=0) + 6.0
    
    # Filter by bounding box
    in_box_indices = np.where(
        (p_coords[:, 0] >= l_min[0]) & (p_coords[:, 0] <= l_max[0]) &
        (p_coords[:, 1] >= l_min[1]) & (p_coords[:, 1] <= l_max[1]) &
        (p_coords[:, 2] >= l_min[2]) & (p_coords[:, 2] <= l_max[2])
    )[0]
    
    p_atoms_filtered = [p_atoms[i] for i in in_box_indices]
    
    if len(p_atoms_filtered) == 0:
        print(f"Filter removed all atoms. Box: {l_min} - {l_max}")
        return None

    # Refine by exact distance if needed, but box is usually good enough to reduce size
    # Let's do exact distance for safety if still too large
    if len(p_atoms_filtered) > 1000:
        # print(f"Large pocket: {len(p_atoms_filtered)} atoms. Keeping as is.")
        pass

    all_atoms = p_atoms_filtered + l_atoms
    
    # 2. Node Features
    x = []
    coords = []
    for atom in all_atoms:
        feat = one_hot_element(atom['element'])
        feat.append(atom['is_ligand'])
        x.append(feat)
        coords.append(atom['coords'])
    
    x = torch.tensor(x, dtype=torch.float)
    coords = torch.tensor(coords, dtype=torch.float)
    
    # 3. Adjacency (Distance based)
    # Pairwise distance
    # Memory efficient way:
    num_nodes = len(all_atoms)
    if num_nodes > 6000: # Increased Safety cap
        print(f"Graph too large: {num_nodes} nodes.")
        return None
        
    # dist matrix: (N, N)
    # Using broadcasting
    r = torch.sum(coords**2, dim=1).view(-1, 1)
    dist_sq = r + r.t() - 2.0 * torch.mm(coords, coords.t())
    # Clamp to 0 to avoid numerical errors
    dist_sq = torch.clamp(dist_sq, min=0.0)
    dist = torch.sqrt(dist_sq)
    
    # Adjacency: 1 if dist < threshold, 0 otherwise
    # Self loops included? Usually yes for GCN.
    adj = (dist < DISTANCE_THRESHOLD).float()
    
    # Normalize Adjacency for GCN: D^-0.5 * A * D^-0.5
    # A_hat = A + I
    adj = adj + torch.eye(num_nodes)
    
    degrees = torch.sum(adj, dim=1)
    d_inv_sqrt = torch.pow(degrees, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    
    adj_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    
    return {
        'x': x,
        'adj': adj_norm,
        'y': torch.tensor([label], dtype=torch.float),
        'protein_path': protein_path,
        'ligand_path': ligand_path
    }

def windows_to_wsl_path(win_path):
    if not win_path:
        return win_path
    # Handle E:\ -> /mnt/e/
    if win_path[1:3] == ':\\':
        drive = win_path[0].lower()
        path = win_path[3:].replace('\\', '/')
        return f"/mnt/{drive}/{path}"
    return win_path.replace('\\', '/')

def process_manifest(manifest_path, label, output_list, output_path=None):
    # Manifest path itself might need conversion if passed from Windows context, 
    # but here we assume the script is run with correct relative path or absolute path.
    # However, the CONTENT of the manifest has Windows paths.
    
    if not os.path.exists(manifest_path):
        print(f"Manifest not found: {manifest_path}")
        return

    with open(manifest_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        rows = list(reader)
        
    # Load existing if resuming
    processed_paths = set()
    if output_path and os.path.exists(output_path):
        try:
            existing_data = torch.load(output_path)
            output_list.extend(existing_data)
            for d in existing_data:
                # Assuming protein_path is stored in data object
                # We need to match what's in the manifest. 
                # The data object has 'protein_path' which is WSL path.
                # Manifest has Windows path.
                # Let's store the basename for easier matching or convert.
                p_path = d.get('protein_path', '')
                if p_path:
                    processed_paths.add(os.path.basename(p_path))
            print(f"Resuming: Loaded {len(existing_data)} graphs. Skipping {len(processed_paths)} processed files.", flush=True)
            success_count = len(existing_data)
        except Exception as e:
            print(f"Failed to load existing dataset for resuming: {e}", flush=True)

    print(f"Processing {len(rows)} samples from {manifest_path}...", flush=True)
    
    for i, row in enumerate(rows):
        # if i >= 10: break # Debug limit
        
        # Prefer pocket path if available
        protein_win = row.get('pocket_path')
        if not protein_win or protein_win == 'N/A':
            protein_win = row.get('protein_path')
            
        ligand_win = row.get('ligand_path')
        
        if not protein_win: continue

        # Normalize path for basename check
        p_win_norm = protein_win.replace('\\', '/')
        if os.path.basename(p_win_norm) in processed_paths:
            continue

        # List of known crashing file identifiers
        crash_ids = ["AHB17899", "AMY95822", "ANH22049", "AUD40029", "KAE9588988", "KAG4924607", "KAG5003670"]
        if any(crash_id in protein_win for crash_id in crash_ids):
            print(f"Skipping known crashing file: {os.path.basename(protein_win)}", flush=True)
            continue

        print(f"Processing: {os.path.basename(protein_win) if protein_win else 'Unknown'}", flush=True)
        
        if protein_win and ligand_win:
            protein = windows_to_wsl_path(protein_win)
            ligand = windows_to_wsl_path(ligand_win)
            
            # Debug first few failures
            if not os.path.exists(protein):
                print(f"File not found: {protein} (Orig: {protein_win})")
                pass

            try:
                data = build_graph(protein, ligand, label)
                if data:
                    output_list.append(data)
                    success_count += 1
                else:
                    print(f"Failed to build graph for {os.path.basename(protein)}")
            except Exception as e:
                print(f"Exception building graph for {os.path.basename(protein)}: {e}")
                import traceback
                traceback.print_exc()
        
        # Checkpoint every 10 samples
        if (i + 1) % 10 == 0 and output_path:
            torch.save(output_list, output_path)
            print(f"Checkpoint saved: {len(output_list)} graphs.", flush=True)

    print(f"Successfully built {success_count} graphs from this manifest.")

def main():
    pos_manifest = "analysis_results/pdbbind_glycan_subset.tsv"
    neg_manifest = "analysis_results/pdbbind_negative_subset.tsv"
    output_file = "analysis_results/processed_graphs.pt"
    
    dataset = []
    
    # Process Positives (Label 1)
    if os.path.exists(pos_manifest):
        process_manifest(pos_manifest, 1, dataset, output_path=output_file)
    
    # Process Negatives (Label 0)
    if os.path.exists(neg_manifest):
        process_manifest(neg_manifest, 0, dataset, output_path=output_file)
        
    print(f"Total graphs processed: {len(dataset)}")
    torch.save(dataset, output_file)
    print(f"Saved dataset to {output_file}")

if __name__ == "__main__":
    main()
