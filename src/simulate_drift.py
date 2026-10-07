"""
BONUS: Simulate artificial concept drift
"""

import numpy as np
import pandas as pd


def inject_drift(X, drift_type='feature_shift', magnitude=0.5):
    """
    Inject artificial drift into data

    BONUS Requirement:
    "Simulate concept drift artificially (inject a shifted data slice)
    and show your monitoring catches it"

    Drift Types:
    1. feature_shift: Shift feature means by magnitude × std
    2. scale_amount: Scale Amount feature by magnitude
    3. noise_injection: Add noise to features

    Args:
        X: Feature DataFrame
        drift_type: Type of drift to inject
        magnitude: How much drift (default 0.5)

    Returns:
        Drifted DataFrame
    """
    X_drifted = X.copy()

    if drift_type == 'feature_shift':
        # Shift top features (V14, V10, V12, etc.)
        top_features = ['V14', 'V10', 'V12', 'V4', 'V11']

        for feature in top_features:
            if feature in X_drifted.columns:
                std = X_drifted[feature].std()
                shift = magnitude * std
                X_drifted[feature] = X_drifted[feature] + shift

    elif drift_type == 'scale_amount':
        # Scale Amount by magnitude (e.g., 1.5× = 50% increase)
        if 'Amount' in X_drifted.columns:
            X_drifted['Amount'] = X_drifted['Amount'] * (1 + magnitude)

    elif drift_type == 'noise_injection':
        # Add Gaussian noise to all features
        for col in X_drifted.columns:
            if col not in ['Time', 'Class']:
                noise = np.random.normal(0, magnitude * X_drifted[col].std(), len(X_drifted))
                X_drifted[col] = X_drifted[col] + noise

    return X_drifted


def simulate_concept_drift(X_test, y_test, model, scaler, drift_magnitude=0.5):
    """
    Simulate concept drift and measure impact

    Process:
    1. Get baseline performance on clean test data
    2. Inject artificial drift
    3. Measure performance on drifted data
    4. Show degradation

    Args:
        X_test: Test features
        y_test: Test labels
        model: Trained model
        scaler: Fitted scaler
        drift_magnitude: How much drift to inject

    Returns:
        Dictionary with drift simulation results
    """
    from sklearn.metrics import average_precision_score

    # Baseline performance (no drift)
    X_test_scaled = X_test.copy()
    X_test_scaled[['Time', 'Amount']] = scaler.transform(X_test[['Time', 'Amount']])

    baseline_proba = model.predict_proba(X_test_scaled)[:, 1]
    baseline_pr_auc = average_precision_score(y_test, baseline_proba)

    # Inject drift
    X_test_drifted = inject_drift(X_test, drift_type='feature_shift', magnitude=drift_magnitude)
    X_test_drifted_scaled = X_test_drifted.copy()
    X_test_drifted_scaled[['Time', 'Amount']] = scaler.transform(X_test_drifted[['Time', 'Amount']])

    # Performance on drifted data
    drifted_proba = model.predict_proba(X_test_drifted_scaled)[:, 1]
    drifted_pr_auc = average_precision_score(y_test, drifted_proba)

    # Calculate degradation
    degradation = (baseline_pr_auc - drifted_pr_auc) / baseline_pr_auc

    print("\n" + "="*70)
    print("🎮 BONUS: DRIFT SIMULATION")
    print("="*70)
    print(f"\nOriginal Performance (No Drift):")
    print(f"  PR-AUC: {baseline_pr_auc:.4f}")
    print(f"\nDrifted Performance (Feature Shift):")
    print(f"  PR-AUC: {drifted_pr_auc:.4f}")
    print(f"\nPerformance Degradation: {degradation:.2%}")

    if abs(degradation) > 0.01:
        print(f"\n✅ DRIFT DETECTED! Monitoring system would catch this")
    else:
        print(f"\n⚠️  Drift too subtle to detect")

    return {
        'original_pr_auc': baseline_pr_auc,
        'drifted_pr_auc': drifted_pr_auc,
        'degradation': degradation,
        'drift_magnitude': drift_magnitude
    }
