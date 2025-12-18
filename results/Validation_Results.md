# Validation Results Summary

## 1. Binding Energy (MM/PBSA)
| System | Binding Energy (kcal/mol) | Stability |
| :--- | :--- | :--- |
| **b047 (Candidate)** | **-18.17** | **High** |
| 1WGT (Wild Type) | +0.34 | Low |

## 2. Hydrogen Bond Stability
| Metric | b047 | 1WGT | Improvement |
| :--- | :--- | :--- | :--- |
| Mean H-Bonds | **0.54** | 0.23 | **2.3x** |
| Max H-Bonds | **4** | 3 | +1 |
| Zero H-Bond Time | **57.0%** | 80.5% | -23.5% |

## 3. Structural Stability (RMSF)
| Metric | b047 | 1WGT | Interpretation |
| :--- | :--- | :--- | :--- |
| Max Pocket RMSF | **0.68 nm** | 2.43 nm | b047 pocket is significantly more rigid. |
| Ligand RMSD | **0.85 Å** | 16.0 Å | Ligand is locked in b047, drifts in 1WGT. |
