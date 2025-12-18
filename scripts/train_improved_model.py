import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, 
    average_precision_score
)
from sklearn.linear_model import LogisticRegression
import joblib
from pathlib import Path

# Feature columns from PLIP (matching training_dataset_merged.tsv)
FEATURE_COLS = [
    "plip_hbond_count", "plip_saltbridge_count", "plip_hydrophobic_count", 
    "plip_pi_stack_count", "plip_pi_cation_count", "plip_water_bridge_count", 
    "plip_metal_complex_count"
]

def load_dataset(path: str):
    """Load dataset and return DataFrame"""
    print(f"Loading data from {path}...", flush=True)
    df = pd.read_csv(path, sep="\t")
    print(f"Columns: {list(df.columns)[:15]}...", flush=True)
    
    # Use label_active as the target (already binary)
    if 'label_active' not in df.columns:
        raise ValueError("Column 'label_active' not found in dataset")
    
    # Ensure features exist and fill NaNs
    for col in FEATURE_COLS:
        if col not in df.columns:
            print(f"Warning: {col} not found, filling with 0", flush=True)
            df[col] = 0.0
        df[col] = df[col].fillna(0.0)
    
    # Clean up label
    df['label'] = df['label_active'].fillna(0).astype(int)
    
    return df

def evaluate_model(y_true, y_pred, y_probs, model_name):
    """Calculate metrics for a model"""
    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_probs) if len(np.unique(y_true)) > 1 else 0.0,
        "auprc": average_precision_score(y_true, y_probs) if len(np.unique(y_true)) > 1 else 0.0
    }
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Improved ML Evaluation with Independent Test Set")
    parser.add_argument("--data", required=True, help="Path to dataset (training_dataset_merged.tsv)")
    parser.add_argument("--out_dir", default="results", help="Output directory")
    args = parser.parse_args()
    
    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    df = load_dataset(args.data)
    print(f"Total samples: {len(df)}", flush=True)
    print(f"Positive samples: {df['label'].sum()} ({df['label'].mean()*100:.1f}%)", flush=True)
    print(f"Unique proteins: {df['protein_pdb_id'].nunique()}", flush=True)
    
    # 1. Split Independent Test Set (GroupShuffleSplit by Protein ID)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(df, df['label'], groups=df['protein_pdb_id']))
    
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]
    
    print(f"\nSplit sizes:", flush=True)
    print(f"Train: {len(train_df)} samples, {train_df['protein_pdb_id'].nunique()} proteins", flush=True)
    print(f"Test:  {len(test_df)} samples, {test_df['protein_pdb_id'].nunique()} proteins", flush=True)
    
    X_train = train_df[FEATURE_COLS].values
    y_train = train_df['label'].values
    X_test = test_df[FEATURE_COLS].values
    y_test = test_df['label'].values
    
    # 2. Train RF Model (PLIP Features)
    print("\nTraining Random Forest (PLIP Features)...", flush=True)
    rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=1)
    rf_model.fit(X_train, y_train)
    
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    rf_preds = rf_model.predict(X_test)
    rf_metrics = evaluate_model(y_test, rf_preds, rf_probs, "RF (PLIP)")
    
    # 3. Baseline: Vina Score (Single Feature)
    print("Training Baseline (Vina Score)...", flush=True)
    results = [{'name': 'RF (PLIP)', 'metrics': rf_metrics}]
    
    if 'affinity' in df.columns:
        X_vina_train = train_df[['affinity']].values
        X_vina_test = test_df[['affinity']].values
        
        vina_model = LogisticRegression(class_weight='balanced', max_iter=1000)
        vina_model.fit(X_vina_train, y_train)
        
        vina_probs = vina_model.predict_proba(X_vina_test)[:, 1]
        vina_preds = vina_model.predict(X_vina_test)
        vina_metrics = evaluate_model(y_test, vina_preds, vina_probs, "Baseline (Vina)")
        results.append({'name': 'Baseline (Vina)', 'metrics': vina_metrics})
    else:
        print("Warning: 'affinity' column not found, skipping Vina baseline.", flush=True)
    
    # 4. Save Results
    all_metrics = [r['metrics'] for r in results]
    pd.DataFrame(all_metrics).to_csv(f"{args.out_dir}/evaluation_metrics.csv", index=False)
    
    print("\n" + "="*60, flush=True)
    print("Evaluation Results (Independent Test Set):", flush=True)
    print("="*60, flush=True)
    print(pd.DataFrame(all_metrics).to_string(index=False), flush=True)
    
    # Save model
    joblib.dump(rf_model, f"{args.out_dir}/improved_rf_model.joblib")
    print(f"\nImproved model saved to {args.out_dir}/improved_rf_model.joblib", flush=True)

if __name__ == "__main__":
    main()
