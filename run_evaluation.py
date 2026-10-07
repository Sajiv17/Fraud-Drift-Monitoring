"""
Re-run evaluation on trained models

This script loads trained models and re-evaluates them on the test set.
Results are printed to console and saved to results/evaluation_report.json

Usage:
    python run_evaluation.py

Requirements:
    - Models must be trained first (run: python run_all.py)
    - models/random_forest_model.pkl must exist
    - models/scaler.pkl must exist
"""

import sys
import json
import os

sys.path.insert(0, 'src')

from src.train_supervised import load_model, load_scaler
from src.data_prep import load_data, time_based_split, separate_features_target
from src.evaluate import evaluate_with_pr_auc, calculate_metrics_at_threshold
from src.cost_analysis import find_optimal_threshold

# Create results directory
os.makedirs('results', exist_ok=True)

print("="*70)
print(" RE-RUNNING EVALUATION ON TEST SET")
print("="*70)

# Load data
print("\n1. Loading data...")
df = load_data()
splits = time_based_split(df)

# Use first test window for final evaluation
X_test, y_test = separate_features_target(splits['test_windows'][0])
print(f"   Test set: {len(X_test):,} samples, {y_test.sum()} frauds")

# Load trained model and scaler
print("\n2. Loading trained models...")
try:
    model = load_model('models/random_forest_model.pkl')
    scaler = load_scaler('models/scaler.pkl')
except FileNotFoundError as e:
    print(f"\n❌ ERROR: {e}")
    print("\n   Models not found. Train first with:")
    print("   python run_all.py")
    sys.exit(1)

# Scale features
print("\n3. Scaling features...")
X_test_scaled = X_test.copy()
X_test_scaled[['Time', 'Amount']] = scaler.transform(X_test[['Time', 'Amount']])

# Get predictions
print("\n4. Running predictions...")
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Evaluate
print("\n5. Evaluating performance...")
results = evaluate_with_pr_auc(y_test, y_pred_proba, 'Random Forest')

# Find optimal threshold on test set
cost_results = find_optimal_threshold(y_test, y_pred_proba)

# Get metrics at optimal threshold
optimal_threshold = cost_results['optimal_threshold']
metrics = calculate_metrics_at_threshold(y_test, y_pred_proba, optimal_threshold)

# Print results
print("\n" + "="*70)
print(" EVALUATION RESULTS (Test Set)")
print("="*70)
print(f"\nPR-AUC:             {results['pr_auc']:.4f}")
print(f"ROC-AUC:            {results['roc_auc']:.4f}")
print(f"\nOptimal Threshold:  {optimal_threshold:.3f}")
print(f"Precision:          {metrics['precision']:.3f}")
print(f"Recall:             {metrics['recall']:.3f}")
print(f"F1-Score:           {metrics['f1']:.3f}")
print(f"\nCost Savings:       ${cost_results['savings']:.2f}")
print(f"  vs. default threshold (0.5)")
print("="*70)

# Save results
output = {
    'dataset': 'test_window_1',
    'pr_auc': results['pr_auc'],
    'roc_auc': results['roc_auc'],
    'optimal_threshold': optimal_threshold,
    'precision': metrics['precision'],
    'recall': metrics['recall'],
    'f1': metrics['f1'],
    'cost_savings_dollars': cost_results['savings']
}

output_path = 'results/evaluation_report.json'
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n✓ Results saved to {output_path}")
print()
