"""
Supervised learning models: Logistic Regression and Random Forest
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
from config import LR_PARAMS, RF_PARAMS, RF_MODEL_PATH, SCALER_PATH


def train_logistic_regression(X_train, y_train):
    """
    Train Logistic Regression baseline

    Why use it?
    - Simple, interpretable baseline
    - Fast to train
    - Good for understanding feature importance
    - Linear decision boundary

    Args:
        X_train: Training features
        y_train: Training labels

    Returns:
        Trained model
    """
    print("\n" + "="*70)
    print("Training Logistic Regression (Baseline)")
    print("="*70)

    model = LogisticRegression(**LR_PARAMS)
    model.fit(X_train, y_train)

    print("✓ Logistic Regression trained")
    return model


def train_random_forest(X_train, y_train, save=True):
    """
    Train Random Forest classifier

    Why use it?
    - Ensemble of decision trees
    - Handles non-linear patterns well
    - Provides feature importance
    - Robust to outliers
    - Usually better than logistic regression

    Args:
        X_train: Training features
        y_train: Training labels
        save: Whether to save the model (default True)

    Returns:
        Trained model
    """
    print("\n" + "="*70)
    print("Training Random Forest (Main Model)")
    print("="*70)

    print(f"\nHyperparameters:")
    print(f"  n_estimators: {RF_PARAMS['n_estimators']}")
    print(f"  max_depth: {RF_PARAMS['max_depth']}")
    print(f"  class_weight: {RF_PARAMS['class_weight']}")

    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train)

    print("✓ Random Forest trained")

    if save:
        joblib.dump(model, RF_MODEL_PATH)
        print(f"✓ Model saved to {RF_MODEL_PATH}")

    return model


def get_feature_importance(model, feature_names, top_n=10):
    """
    Extract feature importance from trained model

    Args:
        model: Trained Random Forest or similar
        feature_names: List of feature names
        top_n: Number of top features to return

    Returns:
        DataFrame with feature importance
    """
    import pandas as pd

    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        return importance_df.head(top_n)
    else:
        return None


def save_scaler(scaler, path=SCALER_PATH):
    """Save fitted scaler for later use"""
    joblib.dump(scaler, path)
    print(f"✓ Scaler saved to {path}")


def load_model(model_path=RF_MODEL_PATH):
    """Load saved model"""
    model = joblib.load(model_path)
    print(f"✓ Model loaded from {model_path}")
    return model


def load_scaler(scaler_path=SCALER_PATH):
    """Load saved scaler"""
    scaler = joblib.load(scaler_path)
    print(f"✓ Scaler loaded from {scaler_path}")
    return scaler
