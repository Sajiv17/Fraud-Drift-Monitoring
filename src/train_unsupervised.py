"""
Unsupervised learning: Isolation Forest for anomaly detection
"""

from sklearn.ensemble import IsolationForest
from config import IF_PARAMS


def train_isolation_forest(X_train):
    """
    Train Isolation Forest (unsupervised anomaly detector)

    How it works:
    - Randomly partitions data using binary trees
    - Anomalies (fraud) are isolated faster with fewer splits
    - No fraud labels needed!

    Why use it?
    - Doesn't need labeled fraud examples
    - Good at catching truly unusual transactions
    - Can complement supervised models

    Limitations:
    - May miss subtle fraud patterns
    - Contamination parameter must be set manually

    Args:
        X_train: Training features (labels not needed!)

    Returns:
        Trained Isolation Forest model
    """
    print("\n" + "="*70)
    print("Training Isolation Forest (Unsupervised)")
    print("="*70)

    print(f"\nHyperparameters:")
    print(f"  contamination: {IF_PARAMS['contamination']} (expected fraud rate)")

    model = IsolationForest(**IF_PARAMS)
    model.fit(X_train)

    print("✓ Isolation Forest trained")
    print("\nNote: Isolation Forest predicts:")
    print("  -1 = anomaly (fraud)")
    print("  +1 = normal (legitimate)")

    return model


def predict_isolation_forest(model, X):
    """
    Get predictions from Isolation Forest

    Converts -1/+1 predictions to 0/1 format

    Args:
        model: Trained Isolation Forest
        X: Features to predict

    Returns:
        Binary predictions (0 = legit, 1 = fraud)
    """
    predictions = model.predict(X)
    # Convert -1 (anomaly) to 1 (fraud), +1 (normal) to 0 (legit)
    binary_predictions = (predictions == -1).astype(int)
    return binary_predictions


def get_anomaly_scores(model, X):
    """
    Get anomaly scores (higher = more anomalous)

    Args:
        model: Trained Isolation Forest
        X: Features

    Returns:
        Anomaly scores
    """
    # decision_function returns negative scores (lower = more anomalous)
    # We flip the sign so higher = more anomalous
    scores = -model.decision_function(X)
    return scores
