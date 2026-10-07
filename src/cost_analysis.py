"""
Business cost analysis and threshold optimization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import FN_COST_PER_TRANSACTION, FP_COST_PER_TRANSACTION, FIGURES_DIR
import os


os.makedirs(FIGURES_DIR, exist_ok=True)


def calculate_cost(y_true, y_pred_proba, threshold=0.5, fn_cost=FN_COST_PER_TRANSACTION, fp_cost=FP_COST_PER_TRANSACTION):
    """
    Calculate business cost at specific threshold

    Cost Formula:
    - False Negative (missed fraud): Lose the transaction amount (~$122 avg)
    - False Positive (block legit): Review cost (~$5 assumed)

    Total Cost = (FN count × FN cost) + (FP count × FP cost)

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        threshold: Classification threshold
        fn_cost: Cost per false negative
        fp_cost: Cost per false positive

    Returns:
        Dictionary with cost breakdown
    """
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Count errors
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()

    # Calculate costs
    fn_total_cost = fn * fn_cost
    fp_total_cost = fp * fp_cost
    total_cost = fn_total_cost + fp_total_cost

    return {
        'threshold': threshold,
        'fn_count': int(fn),
        'fp_count': int(fp),
        'fn_cost': fn_total_cost,
        'fp_cost': fp_total_cost,
        'total_cost': total_cost
    }


def find_optimal_threshold(y_true, y_pred_proba, fn_cost=FN_COST_PER_TRANSACTION, fp_cost=FP_COST_PER_TRANSACTION,
                           thresholds=None):
    """
    Find threshold that minimizes business cost

    Why optimize threshold?
    - Default threshold (0.5) may not be optimal for business
    - Different costs for FN vs FP mean we should adjust threshold
    - Lower threshold = catch more fraud but more false alarms
    - Higher threshold = fewer false alarms but miss more fraud

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        fn_cost: Cost per false negative
        fp_cost: Cost per false positive
        thresholds: List of thresholds to try (default: 0.1 to 0.9)

    Returns:
        DataFrame with cost analysis
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.01)

    results = []

    for threshold in thresholds:
        cost_data = calculate_cost(y_true, y_pred_proba, threshold, fn_cost, fp_cost)
        results.append(cost_data)

    results_df = pd.DataFrame(results)

    # Find optimal threshold (minimum total cost)
    optimal_idx = results_df['total_cost'].idxmin()
    optimal_threshold = results_df.loc[optimal_idx, 'threshold']
    optimal_cost = results_df.loc[optimal_idx, 'total_cost']

    # Compare with default threshold (0.5)
    default_cost_data = calculate_cost(y_true, y_pred_proba, 0.5, fn_cost, fp_cost)
    default_cost = default_cost_data['total_cost']

    savings = default_cost - optimal_cost

    print("\n" + "="*70)
    print("COST-BASED THRESHOLD OPTIMIZATION")
    print("="*70)
    print(f"\nCost Parameters:")
    print(f"  FN cost (missed fraud): ${fn_cost:.2f} per transaction")
    print(f"  FP cost (false alarm):  ${fp_cost:.2f} per transaction")
    print(f"\nResults:")
    print(f"  Default threshold (0.5): Total cost = ${default_cost:,.2f}")
    print(f"  Optimal threshold ({optimal_threshold:.3f}): Total cost = ${optimal_cost:,.2f}")
    print(f"  💰 SAVINGS: ${savings:,.2f}")

    return {
        'results_df': results_df,
        'optimal_threshold': optimal_threshold,
        'optimal_cost': optimal_cost,
        'default_cost': default_cost,
        'savings': savings
    }


def plot_cost_vs_threshold(results_df, save_path=None):
    """
    Plot cost vs threshold

    Args:
        results_df: DataFrame from find_optimal_threshold
        save_path: Where to save figure (optional)

    Returns:
        Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Total cost
    ax1.plot(results_df['threshold'], results_df['total_cost'], linewidth=2, color='#FF6B6B')
    optimal_idx = results_df['total_cost'].idxmin()
    ax1.axvline(x=results_df.loc[optimal_idx, 'threshold'], color='green', linestyle='--',
                label=f"Optimal: {results_df.loc[optimal_idx, 'threshold']:.3f}")
    ax1.axvline(x=0.5, color='gray', linestyle='--', label='Default: 0.5')
    ax1.set_xlabel('Threshold', fontsize=12)
    ax1.set_ylabel('Total Cost ($)', fontsize=12)
    ax1.set_title('Total Cost vs Threshold', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: FN vs FP costs
    ax2.plot(results_df['threshold'], results_df['fn_cost'], linewidth=2, label='FN Cost (missed fraud)', color='red')
    ax2.plot(results_df['threshold'], results_df['fp_cost'], linewidth=2, label='FP Cost (false alarms)', color='orange')
    ax2.axvline(x=results_df.loc[optimal_idx, 'threshold'], color='green', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Threshold', fontsize=12)
    ax2.set_ylabel('Cost ($)', fontsize=12)
    ax2.set_title('FN vs FP Cost', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Cost analysis plot saved to {save_path}")

    return fig


def compare_thresholds(y_true, y_pred_proba, thresholds=[0.3, 0.476, 0.5, 0.7]):
    """
    Compare specific thresholds

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        thresholds: List of thresholds to compare

    Returns:
        DataFrame with comparison
    """
    from evaluate import calculate_metrics_at_threshold

    results = []

    for threshold in thresholds:
        metrics = calculate_metrics_at_threshold(y_true, y_pred_proba, threshold)
        cost_data = calculate_cost(y_true, y_pred_proba, threshold)

        results.append({
            'Threshold': threshold,
            'Precision': f"{metrics['precision']:.3f}",
            'Recall': f"{metrics['recall']:.3f}",
            'F1': f"{metrics['f1']:.3f}",
            'FP': metrics['fp'],
            'FN': metrics['fn'],
            'Total Cost': f"${cost_data['total_cost']:,.0f}"
        })

    comparison_df = pd.DataFrame(results)

    print("\n" + "="*70)
    print("THRESHOLD COMPARISON")
    print("="*70)
    print(comparison_df.to_string(index=False))

    return comparison_df
