"""
Master Pipeline Script
Runs the complete fraud detection system end-to-end

Usage:
    python run_all.py

This script will:
1. Load and split data (time-based)
2. Train supervised models (Logistic Regression, Random Forest)
3. Train unsupervised model (Isolation Forest)
4. Evaluate with PR-AUC
5. Optimize threshold using cost analysis
6. Monitor drift across test windows
7. Generate alert reports
8. Simulate artificial drift (BONUS)
9. Save all results
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, 'src')

# Import all modules
from config import *
from data_prep import load_data, time_based_split, separate_features_target, scale_features
from imbalance import apply_class_weights, apply_smote
from train_supervised import train_logistic_regression, train_random_forest, get_feature_importance, save_scaler
from train_unsupervised import train_isolation_forest, predict_isolation_forest
from evaluate import evaluate_with_pr_auc, plot_pr_curve, compare_models, calculate_metrics_at_threshold
from cost_analysis import find_optimal_threshold, plot_cost_vs_threshold, compare_thresholds
from drift import monitor_feature_drift, detect_score_drift, track_performance_degradation
from alerts import determine_alert_level, check_retrain_trigger, generate_alert_report
from simulate_drift import simulate_concept_drift

import json


def main():
    print("\n" + "="*70)
    print(" FRAUD DETECTION WITH DRIFT MONITORING - FULL PIPELINE")
    print("="*70)
    print()

    # Create directories
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ========================================================================
    # STEP 1: LOAD AND PREPARE DATA
    # ========================================================================
    print("\n" + "="*70)
    print(" STEP 1: DATA LOADING AND PREPARATION")
    print("="*70)

    df = load_data()
    splits = time_based_split(df)

    train_df = splits['train']
    val_df = splits['val']
    test_windows = splits['test_windows']

    X_train, y_train = separate_features_target(train_df)
    X_val, y_val = separate_features_target(val_df)

    # Scale features
    scaled = scale_features(X_train, X_val)
    X_train_scaled = scaled['train']
    X_val_scaled = scaled['val']
    scaler = scaled['scaler']

    # Save scaler
    save_scaler(scaler)

    print(f"\n✓ Data prepared:")
    print(f"  Train: {len(X_train):,} samples, {y_train.sum()} frauds")
    print(f"  Val:   {len(X_val):,} samples, {y_val.sum()} frauds")
    print(f"  Test:  {sum(len(w) for w in test_windows):,} samples in {len(test_windows)} windows")

    # ========================================================================
    # STEP 2: TRAIN MODELS
    # ========================================================================
    print("\n" + "="*70)
    print(" STEP 2: MODEL TRAINING")
    print("="*70)

    # Logistic Regression (baseline)
    lr_model = train_logistic_regression(X_train_scaled, y_train)

    # Random Forest (main model with class weights)
    rf_model = train_random_forest(X_train_scaled, y_train, save=True)

    # Isolation Forest (unsupervised)
    if_model = train_isolation_forest(X_train_scaled)

    # ========================================================================
    # STEP 3: EVALUATE MODELS
    # ========================================================================
    print("\n" + "="*70)
    print(" STEP 3: MODEL EVALUATION")
    print("="*70)

    # Get predictions on validation set
    lr_proba = lr_model.predict_proba(X_val_scaled)[:, 1]
    rf_proba = rf_model.predict_proba(X_val_scaled)[:, 1]
    if_pred = predict_isolation_forest(if_model, X_val_scaled)

    # Evaluate
    lr_results = evaluate_with_pr_auc(y_val, lr_proba, 'Logistic Regression')
    rf_results = evaluate_with_pr_auc(y_val, rf_proba, 'Random Forest')
    if_results = {
        'model': 'Isolation Forest',
        'pr_auc': 'N/A (unsupervised)',
        'roc_auc': 'N/A'
    }

    # Compare models
    results_dict = {
        'Logistic Regression': lr_results,
        'Random Forest': rf_results
    }
    comparison_df = compare_models(results_dict)

    # Plot PR curve for best model (Random Forest)
    pr_curve_path = os.path.join(FIGURES_DIR, 'pr_curves.png')
    plot_pr_curve(y_val, rf_proba, 'Random Forest', save_path=pr_curve_path)
    plt.close()

    # ========================================================================
    # STEP 4: COST ANALYSIS AND THRESHOLD OPTIMIZATION
    # ========================================================================
    print("\n" + "="*70)
    print(" STEP 4: COST ANALYSIS")
    print("="*70)

    # Find optimal threshold
    cost_results = find_optimal_threshold(y_val, rf_proba)

    # Plot cost vs threshold
    cost_plot_path = os.path.join(FIGURES_DIR, 'cost_vs_threshold.png')
    plot_cost_vs_threshold(cost_results['results_df'], save_path=cost_plot_path)
    plt.close()

    # Compare specific thresholds
    optimal_threshold = cost_results['optimal_threshold']
    threshold_comparison = compare_thresholds(y_val, rf_proba, thresholds=[0.3, optimal_threshold, 0.5, 0.7])

    # ========================================================================
    # STEP 5: DRIFT MONITORING
    # ========================================================================
    print("\n" + "="*70)
    print(" STEP 5: DRIFT MONITORING")
    print("="*70)

    # Get feature importance for drift monitoring
    feature_importance_df = get_feature_importance(rf_model, X_train.columns, top_n=TOP_N_FEATURES)
    # Exclude Time from drift monitoring (it increases monotonically by design)
    top_features = [f for f in feature_importance_df['feature'].tolist() if f != 'Time']

    print(f"\nMonitoring top {len(top_features)} features for drift (excluding Time):")
    print(", ".join(top_features[:5]) + "...")

    # Baseline PR-AUC
    baseline_pr_auc = rf_results['pr_auc']

    # Monitor each test window
    window_results = []

    for i, test_window_df in enumerate(test_windows, 1):
        print(f"\n--- Monitoring Window {i} ---")

        X_test, y_test = separate_features_target(test_window_df)
        X_test_scaled = X_test.copy()
        X_test_scaled[['Time', 'Amount']] = scaler.transform(X_test[['Time', 'Amount']])

        # Get predictions
        test_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
        test_pr_auc = average_precision_score(y_test, test_proba)

        # Feature drift detection
        drift_df = monitor_feature_drift(X_train, X_test, top_features)
        max_psi = drift_df['psi'].max()
        drift_features_count = (drift_df['psi_alert'] == 'RED').sum()

        # Performance tracking
        perf_results = track_performance_degradation(baseline_pr_auc, test_pr_auc)

        # Determine overall alert
        drift_metrics = {
            'psi_alert': drift_df['psi_alert'].value_counts().idxmax(),  # Most common alert
            'performance_alert': perf_results['alert']
        }
        overall_alert = determine_alert_level(drift_metrics)

        # Store results
        window_data = {
            'window': i,
            'pr_auc': test_pr_auc,
            'pr_auc_drop_pct': perf_results['drop_pct'] * 100,
            'max_psi': max_psi,
            'drift_features_count': drift_features_count,
            'alert_level': overall_alert
        }
        window_results.append(window_data)

        print(f"  PR-AUC: {test_pr_auc:.4f}")
        print(f"  Alert: {overall_alert}")

    # Check retrain trigger
    window_alerts = [w['alert_level'] for w in window_results]
    retrain_triggered = check_retrain_trigger(window_alerts)

    # Generate alert report
    alert_report = generate_alert_report(window_results)
    print("\n" + alert_report)

    # ========================================================================
    # STEP 6: BONUS - DRIFT SIMULATION
    # ========================================================================
    print("\n" + "="*70)
    print(" STEP 6: BONUS - ARTIFICIAL DRIFT SIMULATION")
    print("="*70)

    # Use first test window for simulation
    X_sim, y_sim = separate_features_target(test_windows[0])
    drift_simulation = simulate_concept_drift(X_sim, y_sim, rf_model, scaler, drift_magnitude=0.5)

    # ========================================================================
    # STEP 7: SAVE RESULTS
    # ========================================================================
    print("\n" + "="*70)
    print(" STEP 7: SAVING RESULTS")
    print("="*70)

    # Save metrics
    metrics_df = pd.DataFrame([
        {
            'Model': 'Random Forest',
            'PR-AUC': rf_results['pr_auc'],
            'ROC-AUC': rf_results['roc_auc'],
            'Optimal Threshold': optimal_threshold,
            'Cost Savings': f"${cost_results['savings']:.2f}"
        },
        {
            'Model': 'Logistic Regression',
            'PR-AUC': lr_results['pr_auc'],
            'ROC-AUC': lr_results['roc_auc'],
            'Optimal Threshold': '-',
            'Cost Savings': '-'
        }
    ])

    metrics_csv_path = os.path.join(RESULTS_DIR, 'metrics.csv')
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"✓ Metrics saved to {metrics_csv_path}")

    # Save drift report
    drift_report_data = {
        'baseline_pr_auc': baseline_pr_auc,
        'windows': window_results,
        'retrain_triggered': retrain_triggered,
        'drift_simulation': drift_simulation
    }

    drift_json_path = os.path.join(RESULTS_DIR, 'drift_report.json')
    with open(drift_json_path, 'w') as f:
        json.dump(drift_report_data, f, indent=2)
    print(f"✓ Drift report saved to {drift_json_path}")

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print(" ✅ PIPELINE COMPLETE!")
    print("="*70)
    print(f"\n📊 Results Summary:")
    print(f"  Best Model: Random Forest")
    print(f"  PR-AUC: {rf_results['pr_auc']:.4f}")
    print(f"  Optimal Threshold: {optimal_threshold:.3f}")
    print(f"  Cost Savings: ${cost_results['savings']:.2f}")
    print(f"  Drift Status: {'⚠️ RETRAIN NEEDED' if retrain_triggered else '✅ Stable'}")
    print(f"  BONUS Complete: ✅ Drift simulation run")
    print(f"\n📁 Outputs:")
    print(f"  Models: {MODELS_DIR}/")
    print(f"  Figures: {FIGURES_DIR}/")
    print(f"  Metrics: {metrics_csv_path}")
    print(f"  Drift Report: {drift_json_path}")
    print()


if __name__ == '__main__':
    # Import additional requirements
    from sklearn.metrics import average_precision_score

    main()
