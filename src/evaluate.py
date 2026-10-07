"""
Model evaluation with Precision-Recall AUC
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    f1_score
)
import os
from config import FIGURES_DIR


os.makedirs(FIGURES_DIR, exist_ok=True)


def evaluate_with_pr_auc(y_true, y_pred_proba, model_name='Model'):
    """
    Evaluate model using Precision-Recall AUC

    Why PR-AUC instead of ROC-AUC?
    - ROC-AUC can be misleadingly optimistic with severe imbalance
    - ROC includes True Negative Rate (we have tons of TNs)
    - PR-AUC focuses only on Precision and Recall for fraud class
    - More realistic evaluation for fraud detection

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities (not 0/1 predictions!)
        model_name: Name for display

    Returns:
        Dictionary of metrics
    """
    # PR-AUC (primary metric)
    pr_auc = average_precision_score(y_true, y_pred_proba)

    # ROC-AUC (for comparison)
    roc_auc = roc_auc_score(y_true, y_pred_proba)

    # Get predictions at default threshold (0.5)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    print(f"\n{model_name} Performance:")
    print(f"  PR-AUC:  {pr_auc:.4f} ⭐ (Primary metric)")
    print(f"  ROC-AUC: {roc_auc:.4f} (For reference)")

    return {
        'model': model_name,
        'pr_auc': pr_auc,
        'roc_auc': roc_auc
    }


def plot_pr_curve(y_true, y_pred_proba, model_name='Model', save_path=None):
    """
    Plot Precision-Recall curve

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        model_name: Name for legend
        save_path: Where to save figure (optional)

    Returns:
        Figure object
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, linewidth=2, label=f'{model_name} (PR-AUC = {pr_auc:.4f})')
    ax.axhline(y=y_true.mean(), color='gray', linestyle='--', label=f'Baseline ({y_true.mean():.4f})')

    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ PR curve saved to {save_path}")

    return fig


def compare_models(results_dict):
    """
    Compare multiple models

    Args:
        results_dict: {model_name: {'pr_auc': ..., 'roc_auc': ...}}

    Returns:
        DataFrame with comparison
    """
    comparison_df = pd.DataFrame(results_dict).T
    comparison_df = comparison_df.sort_values('pr_auc', ascending=False)

    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    print(comparison_df.to_string())

    return comparison_df


def get_confusion_matrix(y_true, y_pred):
    """
    Get confusion matrix with labels

    Returns:
        DataFrame with confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)

    cm_df = pd.DataFrame(
        cm,
        index=['Actual Legit', 'Actual Fraud'],
        columns=['Predicted Legit', 'Predicted Fraud']
    )

    return cm_df


def calculate_metrics_at_threshold(y_true, y_pred_proba, threshold=0.5):
    """
    Calculate metrics at specific threshold

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        threshold: Classification threshold (default 0.5)

    Returns:
        Dictionary of metrics
    """
    y_pred = (y_pred_proba >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'threshold': threshold,
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
