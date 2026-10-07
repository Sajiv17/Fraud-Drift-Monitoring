"""
Data loading, splitting, and scaling
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from config import DATA_PATH, TRAIN_RATIO, VAL_RATIO, N_TEST_WINDOWS


def load_data():
    """Load credit card fraud dataset"""
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"✓ Loaded {len(df):,} transactions")
    print(f"✓ Fraud rate: {df['Class'].mean():.4%}")
    return df


def time_based_split(df, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, n_test_windows=N_TEST_WINDOWS):
    """
    Split data by time (NOT random!)

    Why time-based split?
    - In production, we train on PAST data and deploy on FUTURE data
    - Random split unrealistically mixes past and future together
    - Time-based split simulates real deployment with drift

    Args:
        df: DataFrame with 'Time' column
        train_ratio: Fraction for training (default 0.6)
        val_ratio: Fraction for validation (default 0.15)
        n_test_windows: Number of test windows for monitoring (default 3)

    Returns:
        dict with 'train', 'val', 'test_windows' DataFrames
    """
    df_sorted = df.sort_values('Time').reset_index(drop=True)
    n = len(df_sorted)

    # Calculate split indices
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df_sorted[:train_end]
    val_df = df_sorted[train_end:val_end]
    test_df = df_sorted[val_end:]

    # Split test into monitoring windows
    test_window_size = len(test_df) // n_test_windows
    test_windows = []

    for i in range(n_test_windows):
        start_idx = i * test_window_size
        if i == n_test_windows - 1:
            # Last window gets remaining data
            window = test_df[start_idx:]
        else:
            end_idx = (i + 1) * test_window_size
            window = test_df[start_idx:end_idx]
        test_windows.append(window)

    print(f"\n✓ Time-based split complete:")
    print(f"  Train: {len(train_df):,} samples ({train_ratio:.0%})")
    print(f"  Val:   {len(val_df):,} samples ({val_ratio:.0%})")
    print(f"  Test:  {len(test_df):,} samples split into {n_test_windows} windows")

    return {
        'train': train_df,
        'val': val_df,
        'test_windows': test_windows
    }


def separate_features_target(df):
    """
    Separate features (X) and target (y)

    Args:
        df: DataFrame with 'Class' column

    Returns:
        X (features), y (target)
    """
    X = df.drop('Class', axis=1)
    y = df['Class']
    return X, y


def scale_features(X_train, X_val=None, X_test=None):
    """
    Scale Time and Amount features
    (V1-V28 are already scaled from PCA)

    Args:
        X_train: Training features
        X_val: Validation features (optional)
        X_test: Test features (optional)

    Returns:
        Scaled features and fitted scaler
    """
    scaler = StandardScaler()

    X_train_scaled = X_train.copy()
    X_train_scaled[['Time', 'Amount']] = scaler.fit_transform(X_train[['Time', 'Amount']])

    result = {'train': X_train_scaled, 'scaler': scaler}

    if X_val is not None:
        X_val_scaled = X_val.copy()
        X_val_scaled[['Time', 'Amount']] = scaler.transform(X_val[['Time', 'Amount']])
        result['val'] = X_val_scaled

    if X_test is not None:
        X_test_scaled = X_test.copy()
        X_test_scaled[['Time', 'Amount']] = scaler.transform(X_test[['Time', 'Amount']])
        result['test'] = X_test_scaled

    return result
