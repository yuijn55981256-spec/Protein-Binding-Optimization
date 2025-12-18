import os
import subprocess
import numpy as np

def run_gmx_cmd(cmd, inputs, cwd):
    try:
        # Use Popen to pipe input
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd, text=True)
        stdout, stderr = p.communicate(input=inputs)
        if p.returncode != 0:
            print(f"Error running command: {' '.join(cmd)}")
            print(stderr)
            return False
        return True
    except Exception as e:
        print(f"Exception running command: {e}")
        return False

def parse_xvg(filepath):
    data = []
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith(('#', '@')): continue
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    data.append([float(p) for p in parts])
                except: pass
    return np.array(data)

def analyze_dir(name, path, is_control=False):
    print(f"\nAnalyzing {name} in {path}...")
    
    # 1. H-Bonds
    hbond_file = "hbnum.xvg"
    if not os.path.exists(os.path.join(path, hbond_file)):
        print("  Calculating H-bonds...")
        # Assuming Group 1=Protein, Group 13=Ligand (need to verify groups if possible, but 1 and 13 are common for GROMACS if Ligand is at end)
        # Better to check index.ndx if exists, but let's try standard first.
        # Actually, for 1WGT, ligand might be different group.
        # Let's try to list groups first? No, just try 1 and 13. If fails, we'll see.
        # For 1WGT, ligand is likely "UNL" or "NAG".
        cmd = ["gmx", "hbond", "-s", "md_0_1.tpr", "-f", "md_0_1.xtc", "-num", hbond_file]
        # Input: 1 (Protein) and 13 (Ligand) - Wait, if 13 doesn't exist?
        # Let's assume Protein is 1. Ligand might be "LIG" or "NAG".
        # We can try to use "Protein" and "Ligand" names if make_ndx was used.
        # But gmx hbond prompts for two groups.
        run_gmx_cmd(cmd, "1\n13\n", path)
    
    hb_data = parse_xvg(os.path.join(path, hbond_file))
    if hb_data is not None and len(hb_data) > 0:
        hb = hb_data[:, 1] # 2nd column is number of H-bonds
        print(f"  H-Bonds: Mean={np.mean(hb):.2f}, Max={np.max(hb):.0f}, Zero_Frac={np.sum(hb==0)/len(hb)*100:.1f}%")
    else:
        print("  Failed to get H-bond data.")

    # 2. RMSF
    rmsf_file = "rmsf.xvg"
    if not os.path.exists(os.path.join(path, rmsf_file)):
        print("  Calculating RMSF...")
        cmd = ["gmx", "rmsf", "-s", "md_0_1.tpr", "-f", "md_0_1.xtc", "-o", rmsf_file, "-res"]
        run_gmx_cmd(cmd, "1\n", path) # Group 1 = Protein
        
    rmsf_data = parse_xvg(os.path.join(path, rmsf_file))
    if rmsf_data is not None and len(rmsf_data) > 0:
        resids = rmsf_data[:, 0]
        flucts = rmsf_data[:, 1]
        print(f"  RMSF: Mean={np.mean(flucts):.3f} nm, Max={np.max(flucts):.3f} nm")
        
        # Identify key residues (Trp, Tyr, etc.)
        # We don't have the sequence map here, but we can list low RMSF residues.
        # Or if we assume WGA sequence...
        # WGA has 4 hevein domains. Key residues are often Cys (disulfides) and aromatics.
        # Let's print residues with RMSF < 0.05 nm (very stable).
        stable_mask = flucts < 0.05
        stable_res = resids[stable_mask]
        print(f"  Stable Residues (RMSF < 0.5 A): {len(stable_res)} residues")
        # Print a few examples
        if len(stable_res) > 0:
            print(f"  Examples: {stable_res[:5]}")
            
        # Check for high fluctuations
        high_mask = flucts > 0.2
        high_res = resids[high_mask]
        print(f"  Flexible Residues (RMSF > 2.0 A): {len(high_res)} residues")

def main():
    # b047
    analyze_dir("b047 (Candidate)", "md_simulation")
    
    # 1WGT
    analyze_dir("1WGT-NAG (Control)", "md_simulation_1wgt", is_control=True)

if __name__ == "__main__":
    main()
