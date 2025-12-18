#!/bin/bash
# Script to generate missing RMSF and H-bond statistics
# Run this in your WSL environment where GROMACS is installed.

# Activate Conda Environment (Adjust path if needed)
source /root/miniconda/etc/profile.d/conda.sh
conda activate colabfold

set -e

# --- Configuration ---
# Adjust these paths if necessary
BASE_DIR="/mnt/e/MyProject/MyPROJECT/WGA_Homolog_Screening_Clean"
B047_DIR="$BASE_DIR/md_simulation"
W1GT_DIR="$BASE_DIR/md_simulation_1wgt"

echo "=================================================="
echo "Recovering Missing MD Data"
echo "=================================================="

# --- 1. b047 (Candidate) - Calculate RMSF ---
echo ""
echo "Processing b047 (Candidate)..."
cd "$B047_DIR"

if [ ! -f "rmsf.xvg" ]; then
    echo "  Calculating RMSF for Protein residues..."
    # Group 1 is usually Protein. We select 1 for calculation and 1 for output alignment.
    # Use -res to get per-residue RMSF
    # Note: Using md_0_1.tpr and md_0_1.xtc as per directory listing
    echo -e "1\n" | gmx rmsf -s md_0_1.tpr -f md_0_1.xtc -o rmsf.xvg -res > rmsf.log 2>&1
    if [ -f "rmsf.xvg" ]; then
        echo "  [OK] RMSF generated: $B047_DIR/rmsf.xvg"
    else
        echo "  [FAIL] RMSF calculation failed. Check rmsf.log"
    fi
else
    echo "  [SKIP] rmsf.xvg already exists."
fi

# --- 2. 1WGT (Control) - Calculate H-bonds and RMSF ---
echo ""
echo "Processing 1WGT-NAG (Control)..."
cd "$W1GT_DIR"

# Note: 1WGT used a fixed index file. We need to be careful with group numbers.
# Usually Group 1 is Protein, Group 13 is Ligand (or similar).
# We will try to use 'Protein' and 'LIG' names if possible, or standard numbers.

if [ ! -f "hbnum.xvg" ]; then
    echo "  Calculating H-bonds (Protein-Ligand)..."
    # Try to find group numbers for Protein and Ligand
    # We'll assume 1 (Protein) and 13 (Ligand) based on standard GROMACS output, 
    # but for safety we will try to select by name if interactive, but here we use echo.
    # Let's assume standard: 1=Protein, 13=Ligand. 
    # If this fails, the user might need to check index.ndx manually.
    
    # Using '1' (Protein) and '13' (Ligand)
    echo -e "1\n13\n" | gmx hbond -s md.tpr -f complex_matched.xtc -n index_fixed.ndx -num hbnum.xvg > hbond.log 2>&1
    
    if [ -f "hbnum.xvg" ]; then
        echo "  [OK] H-bonds generated: $W1GT_DIR/hbnum.xvg"
    else
        echo "  [FAIL] H-bond calculation failed. Check hbond.log"
        echo "         (Group IDs might be wrong. Check index_fixed.ndx)"
    fi
else
    echo "  [SKIP] hbnum.xvg already exists."
fi

if [ ! -f "rmsf.xvg" ]; then
    echo "  Calculating RMSF for Protein residues..."
    # Use Group 1 (Protein)
    # Note: md.tpr has extra atoms. Using _GMXMMPBSA_COM_FIXED.pdb as reference structure which should match complex_matched.xtc
    echo -e "1\n" | gmx rmsf -s _GMXMMPBSA_COM_FIXED.pdb -f complex_matched.xtc -n index_fixed.ndx -o rmsf.xvg -res > rmsf.log 2>&1
    if [ -f "rmsf.xvg" ]; then
        echo "  [OK] RMSF generated: $W1GT_DIR/rmsf.xvg"
    else
        echo "  [FAIL] RMSF calculation failed. Check rmsf.log"
    fi
else
    echo "  [SKIP] rmsf.xvg already exists."
fi

echo ""
echo "=================================================="
echo "Recovery Complete."
echo "Please check the output files in:"
echo "  - $B047_DIR"
echo "  - $W1GT_DIR"
echo "=================================================="
