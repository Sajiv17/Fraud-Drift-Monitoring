"""
Configuration file for fraud detection project
Contains all hyperparameters, paths, and constants
"""

import os

# ============================================================================
# PATHS
# ============================================================================
DATA_DIR = 'data'
MODELS_DIR = 'models'
RESULTS_DIR = 'results'
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')

DATA_PATH = os.path.join(DATA_DIR, 'creditcard.csv')
METRICS_PATH = os.path.join(RESULTS_DIR, 'metrics.csv')

# Model save paths
RF_MODEL_PATH = os.path.join(MODELS_DIR, 'random_forest_model.pkl')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')

# ============================================================================
# DATA SPLITTING
# ============================================================================
TRAIN_RATIO = 0.6
VAL_RATIO = 0.15
N_TEST_WINDOWS = 3  # For drift monitoring

# ============================================================================
# RANDOM SEEDS (for reproducibility)
# ============================================================================
RANDOM_SEED = 42

# ============================================================================
# MODEL HYPERPARAMETERS
# ============================================================================

# Random Forest
RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 50,
    'min_samples_leaf': 20,
    'random_state': RANDOM_SEED,
    'n_jobs': -1,
    'class_weight': 'balanced'  # Handle imbalance
}

# Logistic Regression
LR_PARAMS = {
    'max_iter': 1000,
    'random_state': RANDOM_SEED,
    'class_weight': 'balanced'  # Handle imbalance
}

# Isolation Forest (unsupervised)
IF_PARAMS = {
    'contamination': 0.00173,  # 0.173% fraud rate
    'random_state': RANDOM_SEED,
    'n_jobs': -1
}

# ============================================================================
# COST ANALYSIS
# ============================================================================

# False Negative cost: lose the transaction amount (average ~$122)
FN_COST_PER_TRANSACTION = 122.21  # Average fraud amount

# False Positive cost: review cost (assumed)
FP_COST_PER_TRANSACTION = 5.00

# ============================================================================
# DRIFT DETECTION THRESHOLDS
# ============================================================================

# PSI (Population Stability Index)
PSI_THRESHOLD_GREEN = 0.1   # < 0.1 = stable
PSI_THRESHOLD_AMBER = 0.25  # 0.1-0.25 = moderate drift
# > 0.25 = significant drift (RED)

# KS Test (Kolmogorov-Smirnov)
KS_P_VALUE_THRESHOLD = 0.05  # p < 0.05 = significant drift

# PR-AUC Performance Drop
PR_AUC_DROP_THRESHOLD_GREEN = 0.05   # < 5% drop
PR_AUC_DROP_THRESHOLD_AMBER = 0.10   # 5-10% drop
# > 10% drop = RED

# Alert logic
RETRAIN_TRIGGER_CONSECUTIVE_RED = 2  # Retrain after 2 consecutive RED windows

# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================
TOP_N_FEATURES = 10  # Monitor top N features for drift
