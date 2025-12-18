# Technical Report: WGA Homolog Screening & Validation

## 1. Introduction
Wheat Germ Agglutinin (WGA) is a widely used lectin, but its application is often limited by variable binding affinity. This project aimed to identify novel WGA homologs with superior ligand binding properties using a computational screening pipeline.

## 2. Methodology

### 2.1 Graph Neural Network (GNN) Screening
We constructed a custom Graph Neural Network to predict protein-ligand binding affinity.
*   **Input**: Protein 3D structures converted into graph representations (nodes=atoms, edges=bonds/contacts).
*   **Training**: Trained on the PDBbind dataset.
*   **Inference**: Screened ~6,000 homologous sequences obtained via BLAST+.

### 2.2 Machine Learning Refinement
To improve ranking precision, we developed a Random Forest model incorporating specific interaction features extracted by PLIP (Protein-Ligand Interaction Profiler).
*   **Features**: Hydrogen bonds, Hydrophobic interactions, Water bridges, Pi-stacking.
*   **Result**: The improved model corrected false positives from the initial GNN screen.

### 2.3 Molecular Dynamics (MD) Simulation
Top candidates were validated using 10ns MD simulations (GROMACS 2022).
*   **Force Field**: Amber99SB-ILDN (Protein) + GAFF2 (Ligand).
*   **Analysis**: RMSD, RMSF, Hydrogen Bond analysis, and MM/PBSA binding energy calculation.

## 3. Results & Discussion

### 3.1 Candidate Identification
The screening pipeline identified `b047` as a top candidate. Initial docking suggested potential steric clashes, but energy minimization revealed a highly favorable binding conformation.

### 3.2 Stability Analysis (The "Anchoring Effect")
MD simulations revealed a distinct mechanism for the enhanced affinity of `b047`:
*   **Rigid Binding Pocket**: RMSF analysis showed that the binding pocket of `b047` maintains high rigidity (Max RMSF < 0.7 nm), whereas the wild-type (`1WGT`) pocket exhibits significant fluctuations (> 2.4 nm).
*   **Enhanced H-Bonding**: `b047` maintains a denser hydrogen bond network (Mean=0.54) compared to the wild-type (Mean=0.23), effectively "anchoring" the ligand.

### 3.3 Binding Energy
MM/PBSA calculations confirmed the stability findings:
*   **b047**: -18.17 kcal/mol (Strong Binding)
*   **1WGT**: +0.34 kcal/mol (Weak/Unstable)

## 4. Conclusion
This project successfully demonstrated the power of combining deep learning (GNN) with physics-based simulation (MD) for protein engineering. The identified candidate `b047` shows promise for high-affinity applications.
