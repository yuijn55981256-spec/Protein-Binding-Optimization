# WGA Homolog Screening Project

## Project Overview
This project focuses on the computational screening and validation of Wheat Germ Agglutinin (WGA) homologs for enhanced ligand binding affinity. By combining Graph Neural Networks (GNN), Machine Learning (Random Forest), and Molecular Dynamics (MD) simulations, we successfully identified a candidate protein (`b047`) with significantly improved binding properties compared to the wild-type (`1WGT`).

## Key Achievements
*   **High-Throughput Screening**: Processed ~6,000 homologous sequences using a custom GNN pipeline.
*   **Machine Learning Optimization**: Developed an improved Random Forest model based on PLIP interaction features, achieving higher precision in candidate ranking.
*   **Molecular Dynamics Validation**: Performed 10ns MD simulations (GROMACS) and MM/PBSA analysis, confirming a **-18.17 kcal/mol** binding energy for the top candidate (vs. +0.34 kcal/mol for wild-type).
*   **Mechanism Discovery**: Identified the "Anchoring Effect" where a rigidified binding pocket and enhanced hydrogen bond network lock the ligand in place.

## Repository Structure
*   `scripts/`: Core computational pipelines.
    *   `inference_gnn.py`: GNN model inference logic.
    *   `train_improved_model.py`: Random Forest training with PLIP features.
    *   `generate_missing_stats.sh`: Automated shell script for GROMACS data recovery.
    *   `calculate_extra_stats.py`: Statistical analysis of MD trajectories.
*   `docs/`: Technical reports and methodology.
*   `results/`: Summary of key findings and validation metrics.

## Technology Stack
*   **Bioinformatics**: GROMACS, AmberTools (MMPBSA), PLIP, BLAST+
*   **Machine Learning**: PyTorch (GNN), Scikit-learn (Random Forest)
*   **Languages**: Python, Bash

## Contact
[Your Name/Email]
