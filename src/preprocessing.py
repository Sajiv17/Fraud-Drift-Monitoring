"""
Data Preprocessing Module

This module handles:
1. Time-based data splitting (train/val/test)
2. Data preparation for modeling
3. Creating monitoring windows

WHY TIME-BASED SPLIT?
In real life, we train a model on past data and deploy it on future data.
A random split mixes past and future together, which is unrealistic.
Time-based split simulates real deployment and lets us detect drift.
"""

import pandas as pd
import numpy as np


def time_based_split(df, train_ratio=0.6, val_ratio=0.15, n_test_windows=3):
    """
    Split data by time into train, validation, and test sets.

    WHY THIS FUNCTION EXISTS:
    Fraud patterns change over time. We want to train on OLD data and test
    on NEW data to simulate real-world deployment where we predict future frauds.

    Parameters:
    -----------
    df : pandas.DataFrame
        The full dataset with a 'Time' column
    train_ratio : float
        Fraction of data to use for training (default: 0.6 = 60%)
    val_ratio : float
        Fraction for validation (default: 0.15 = 15%)
    n_test_windows : int
        Number of test windows for drift monitoring (default: 3)

    Returns:
    --------
    dict : Contains train_df, val_df, test_windows (list of DataFrames)

    Example:
    --------
    With 100,000 transactions:
    - Train: 0-60,000 (60%)
    - Val: 60,000-75,000 (15%)
    - Test: 75,000-100,000 (25%, split into 3 windows of ~8,333 each)
    """

    # STEP 1: Sort by time (must be chronological!)
    # If we don't sort, "future" data might leak into "past" data
    df_sorted = df.sort_values('Time').reset_index(drop=True)

    n_total = len(df_sorted)

    # STEP 2: Calculate split points
    # These are the row indices where we cut the data
    train_end = int(n_total * train_ratio)
    val_end = int(n_total * (train_ratio + val_ratio))

    # STEP 3: Split the data
    train_df = df_sorted.iloc[:train_end].copy()
    val_df = df_sorted.iloc[train_end:val_end].copy()
    test_df = df_sorted.iloc[val_end:].copy()

    # STEP 4: Split test into monitoring windows
    # This simulates monitoring the model over time
    test_window_size = len(test_df) // n_test_windows
    test_windows = []

    for i in range(n_test_windows):
        start_idx = i * test_window_size
        # Last window gets all remaining rows
        end_idx = (i + 1) * test_window_size if i < n_test_windows - 1 else len(test_df)
        window = test_df.iloc[start_idx:end_idx].copy()
        test_windows.append(window)

    # STEP 5: Print summary statistics
    print("="*60)
    print("TIME-BASED DATA SPLIT SUMMARY")
    print("="*60)

    print(f"\nTotal transactions: {n_total:,}")
    print(f"\nTime range:")
    print(f"  Start: {df_sorted['Time'].min():,.0f} seconds")
    print(f"  End:   {df_sorted['Time'].max():,.0f} seconds")
    print(f"  Span:  {(df_sorted['Time'].max() - df_sorted['Time'].min()) / 3600:.1f} hours")

    print(f"\n{'Set':<15} {'Size':<12} {'Frauds':<8} {'Fraud %':<10} {'Time Range (hours)'}")
    print("-"*60)

    # Helper function to print set info
    def print_set_info(name, dataset):
        size = len(dataset)
        frauds = dataset['Class'].sum()  # Class=1 means fraud
        fraud_pct = (frauds / size * 100) if size > 0 else 0
        time_start = dataset['Time'].min() / 3600
        time_end = dataset['Time'].max() / 3600
        print(f"{name:<15} {size:<12,} {frauds:<8} {fraud_pct:<10.3f} {time_start:.1f} - {time_end:.1f}")

    print_set_info("Train", train_df)
    print_set_info("Validation", val_df)

    for i, window in enumerate(test_windows, 1):
        print_set_info(f"Test Window {i}", window)

    print("="*60)

    # STEP 6: Sanity checks
    assert len(train_df) + len(val_df) + sum(len(w) for w in test_windows) == n_total, \
        "Data split error: row count mismatch!"

    assert train_df['Time'].max() <= val_df['Time'].min(), \
        "Data split error: train time overlaps with val!"

    assert val_df['Time'].max() <= test_windows[0]['Time'].min(), \
        "Data split error: val time overlaps with test!"

    print("\\n✓ All sanity checks passed!")
    print("✓ No time overlap between train/val/test")
    print("✓ Data is properly ordered chronologically\\n")

    return {
        'train': train_df,
        'val': val_df,
        'test_windows': test_windows,
        'split_info': {
            'train_ratio': train_ratio,
            'val_ratio': val_ratio,
            'n_test_windows': n_test_windows,
            'train_size': len(train_df),
            'val_size': len(val_df),
            'test_sizes': [len(w) for w in test_windows]
        }
    }


def separate_features_target(df, target_col='Class'):
    """
    Separate features (X) from target (y).

    WHY THIS FUNCTION?
    ML models need X (input features) and y (what to predict) as separate arrays.

    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset with features and target
    target_col : str
        Name of the target column (default: 'Class')

    Returns:
    --------
    X : pandas.DataFrame
        Features only (everything except target)
    y : pandas.Series
        Target values (0 or 1)
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    return X, y


def get_fraud_statistics(df):
    """
    Calculate fraud statistics for a dataset.

    Useful for understanding each split and monitoring drift.

    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset with 'Class' column

    Returns:
    --------
    dict : Statistics about fraud in this dataset
    """
    total = len(df)
    frauds = df['Class'].sum()
    fraud_rate = frauds / total if total > 0 else 0

    return {
        'total_transactions': total,
        'fraud_count': int(frauds),
        'legit_count': int(total - frauds),
        'fraud_rate': fraud_rate,
        'fraud_percentage': fraud_rate * 100,
        'imbalance_ratio': (total - frauds) / frauds if frauds > 0 else float('inf')
    }


def print_split_summary(splits):
    """
    Print a nice summary of the data splits.

    Parameters:
    -----------
    splits : dict
        Output from time_based_split()
    """
    print("\\n" + "="*60)
    print("FRAUD STATISTICS BY SPLIT")
    print("="*60 + "\\n")

    for name, df in [('Train', splits['train']),
                     ('Validation', splits['val'])]:
        stats = get_fraud_statistics(df)
        print(f"{name}:")
        print(f"  Total: {stats['total_transactions']:,}")
        print(f"  Frauds: {stats['fraud_count']} ({stats['fraud_percentage']:.3f}%)")
        print(f"  Imbalance ratio: {stats['imbalance_ratio']:.1f}:1")
        print()

    for i, window in enumerate(splits['test_windows'], 1):
        stats = get_fraud_statistics(window)
        print(f"Test Window {i}:")
        print(f"  Total: {stats['total_transactions']:,}")
        print(f"  Frauds: {stats['fraud_count']} ({stats['fraud_percentage']:.3f}%)")
        print(f"  Imbalance ratio: {stats['imbalance_ratio']:.1f}:1")
        print()


if __name__ == "__main__":
    # Test the functions
    print("Testing preprocessing functions...\\n")

    # Load data
    try:
        df = pd.read_csv('../data/creditcard.csv')
        print(f"✓ Loaded {len(df):,} transactions")

        # Test time-based split
        splits = time_based_split(df)

        # Print fraud statistics
        print_split_summary(splits)

        print("\\n✓ All preprocessing functions work correctly!")

    except FileNotFoundError:
        print("✗ Dataset not found at ../data/creditcard.csv")
        print("  Please download it from Kaggle first.")
