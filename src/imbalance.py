"""
Imbalance handling strategies
"""

from imblearn.over_sampling import SMOTE
from config import RANDOM_SEED


def apply_class_weights(model_params):
    """
    Strategy: Use class weights to penalize minority class errors more

    Pros:
    - No data modification
    - Simple and fast
    - Built into most sklearn models

    Cons:
    - May not work well for extreme imbalance

    Returns:
        Model parameters with class_weight='balanced'
    """
    params = model_params.copy()
    params['class_weight'] = 'balanced'
    return params


def apply_smote(X_train, y_train):
    """
    Strategy: SMOTE (Synthetic Minority Oversampling Technique)

    Creates synthetic fraud examples by interpolating between real fraud cases

    CRITICAL: Apply ONLY to training set AFTER split to avoid data leakage!

    Pros:
    - Balances classes without duplicating data
    - Creates diverse synthetic samples

    Cons:
    - Can create unrealistic samples
    - Increases training time
    - May cause overfitting if used incorrectly

    Args:
        X_train: Training features
        y_train: Training labels

    Returns:
        X_resampled, y_resampled
    """
    print("\nApplying SMOTE...")
    print(f"  Before: {len(y_train):,} samples, {y_train.sum()} frauds ({y_train.mean():.4%})")

    smote = SMOTE(random_state=RANDOM_SEED)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    print(f"  After:  {len(y_resampled):,} samples, {y_resampled.sum()} frauds ({y_resampled.mean():.4%})")
    print("  ✓ Classes balanced!")

    return X_resampled, y_resampled


def compare_strategies():
    """
    Returns comparison of imbalance handling strategies
    """
    strategies = {
        'Baseline (No handling)': {
            'description': 'Use data as-is',
            'pros': 'Simple, no bias from resampling',
            'cons': 'Model predicts all legitimate',
            'when_to_use': 'Baseline comparison only'
        },
        'Class Weights': {
            'description': 'Penalize minority class errors more',
            'pros': 'No data modification, fast',
            'cons': 'May not work for extreme imbalance',
            'when_to_use': 'Good starting point'
        },
        'SMOTE': {
            'description': 'Create synthetic minority samples',
            'pros': 'Balances classes, diverse samples',
            'cons': 'May create unrealistic data, slower',
            'when_to_use': 'Apply to train set only'
        },
        'Isolation Forest': {
            'description': 'Unsupervised anomaly detection',
            'pros': 'No labels needed, catches outliers',
            'cons': 'May miss subtle fraud patterns',
            'when_to_use': 'Compare with supervised'
        }
    }
    return strategies
