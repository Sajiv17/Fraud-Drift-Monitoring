"""
Drift detection methods: PSI, KS Test, Score Distribution
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score
from config import (
    PSI_THRESHOLD_GREEN, PSI_THRESHOLD_AMBER,
    KS_P_VALUE_THRESHOLD,
    PR_AUC_DROP_THRESHOLD_GREEN, PR_AUC_DROP_THRESHOLD_AMBER,
    TOP_N_FEATURES
)


def calculate_psi(baseline_dist, monitoring_dist, bins=10):
    """
    Calculate PSI (Population Stability Index)

    PSI measures distribution shift between two time periods

    How it works:
    1. Bin both distributions into same bins
    2. Calculate % of data in each bin for both
    3. PSI = Σ (% new - % old) × ln(% new / % old)

    Thresholds:
    - PSI < 0.1:  🟢 GREEN (stable)
    - 0.1-0.25:   🟡 AMBER (moderate drift)
    - PSI > 0.25: 🔴 RED (significant drift)

    Args:
        baseline_dist: Training/baseline feature values
        monitoring_dist: Monitoring window feature values
        bins: Number of bins (default 10)

    Returns:
        PSI value
    """
    # Create bins from baseline
    _, bin_edges = np.histogram(baseline_dist, bins=bins)

    # Bin both distributions
    baseline_counts, _ = np.histogram(baseline_dist, bins=bin_edges)
    monitoring_counts, _ = np.histogram(monitoring_dist, bins=bin_edges)

    # Convert to percentages (add small epsilon to avoid division by zero)
    epsilon = 1e-10
    baseline_pct = (baseline_counts + epsilon) / len(baseline_dist)
    monitoring_pct = (monitoring_counts + epsilon) / len(monitoring_dist)

    # Calculate PSI
    psi = np.sum((monitoring_pct - baseline_pct) * np.log(monitoring_pct / baseline_pct))

    return psi


def calculate_ks_test(baseline_dist, monitoring_dist):
    """
    KS Test (Kolmogorov-Smirnov Test)

    Statistical test for distribution difference

    Returns:
        - ks_statistic: Magnitude of difference (0 to 1)
        - p_value: Probability distributions are the same
          - p < 0.05: Significant drift detected 🔴
          - p > 0.05: No significant difference 🟢

    Args:
        baseline_dist: Training/baseline feature values
        monitoring_dist: Monitoring window feature values

    Returns:
        Dictionary with ks_statistic and p_value
    """
    ks_statistic, p_value = stats.ks_2samp(baseline_dist, monitoring_dist)

    return {
        'ks_statistic': ks_statistic,
        'p_value': p_value,
        'significant_drift': p_value < KS_P_VALUE_THRESHOLD
    }


def monitor_feature_drift(X_baseline, X_monitoring, feature_names, top_n=TOP_N_FEATURES):
    """
    Monitor drift across multiple features

    Args:
        X_baseline: Baseline feature DataFrame
        X_monitoring: Monitoring window feature DataFrame
        feature_names: List of features to monitor
        top_n: Number of top features to check

    Returns:
        DataFrame with PSI and KS test results per feature
    """
    results = []

    for feature in feature_names[:top_n]:
        baseline_vals = X_baseline[feature].values
        monitoring_vals = X_monitoring[feature].values

        psi = calculate_psi(baseline_vals, monitoring_vals)
        ks_result = calculate_ks_test(baseline_vals, monitoring_vals)

        results.append({
            'feature': feature,
            'psi': psi,
            'ks_statistic': ks_result['ks_statistic'],
            'ks_p_value': ks_result['p_value'],
            'psi_alert': 'RED' if psi > PSI_THRESHOLD_AMBER else ('AMBER' if psi > PSI_THRESHOLD_GREEN else 'GREEN'),
            'ks_alert': 'RED' if ks_result['significant_drift'] else 'GREEN'
        })

    return pd.DataFrame(results)


def detect_score_drift(baseline_scores, monitoring_scores):
    """
    Detect drift in model prediction scores

    Checks if model confidence is shifting:
    - Scores becoming more extreme (toward 0 or 1)
    - Scores clustering differently

    Args:
        baseline_scores: Model scores on baseline data
        monitoring_scores: Model scores on monitoring window

    Returns:
        Dictionary with score drift metrics
    """
    ks_result = calculate_ks_test(baseline_scores, monitoring_scores)

    return {
        'score_mean_baseline': baseline_scores.mean(),
        'score_mean_monitoring': monitoring_scores.mean(),
        'score_std_baseline': baseline_scores.std(),
        'score_std_monitoring': monitoring_scores.std(),
        'ks_statistic': ks_result['ks_statistic'],
        'ks_p_value': ks_result['p_value'],
        'significant_drift': ks_result['significant_drift']
    }


def track_performance_degradation(baseline_pr_auc, monitoring_pr_auc):
    """
    Track PR-AUC performance degradation

    Detects concept drift through performance drops

    Thresholds:
    - Drop < 5%:   🟢 GREEN
    - Drop 5-10%:  🟡 AMBER
    - Drop > 10%:  🔴 RED

    Args:
        baseline_pr_auc: PR-AUC on baseline/validation data
        monitoring_pr_auc: PR-AUC on monitoring window

    Returns:
        Dictionary with performance metrics
    """
    drop_pct = abs(monitoring_pr_auc - baseline_pr_auc) / baseline_pr_auc

    if drop_pct < PR_AUC_DROP_THRESHOLD_GREEN:
        alert = 'GREEN'
    elif drop_pct < PR_AUC_DROP_THRESHOLD_AMBER:
        alert = 'AMBER'
    else:
        alert = 'RED'

    return {
        'baseline_pr_auc': baseline_pr_auc,
        'monitoring_pr_auc': monitoring_pr_auc,
        'drop_pct': drop_pct,
        'alert': alert
    }
