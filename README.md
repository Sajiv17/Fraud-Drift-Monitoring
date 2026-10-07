# Credit Card Fraud Detection with Drift Monitoring

**AWS Student Builder Group - AI/ML Track**

Machine learning system for credit card fraud detection addressing extreme class imbalance (0.173% fraud rate), with automated drift monitoring and cost optimization.

**Author:** Sajiv Vaila  
**GitHub:** [@Sajiv17](https://github.com/Sajiv17)

---

## Quick Start

```bash
# Clone repository
git clone https://github.com/Sajiv17/Fraud-Drift-Monitoring.git
cd Fraud-Drift-Monitoring

# Install dependencies
pip install -r requirements.txt

# Download dataset from Kaggle (see data/README.md)
# Place creditcard.csv in data/

# Train models and run full pipeline
python run_all.py
```

**Runtime:** ~15 minutes (timed on MacBook Pro M1, 16GB RAM)  
**Python version:** 3.9+  
**Random seed:** 42 (for reproducibility)

---

## Re-Run Evaluation

After training models with `python run_all.py`, re-run evaluation only:

```bash
python run_evaluation.py
```

This evaluates the trained Random Forest model on the test set and saves results to `results/evaluation_report.json`.

**Note:** Models must be trained first. The `models/` folder is gitignored and not included in the repository.

---

## Results

All metrics measured on **test set** (first test window, 25,632 transactions, 41 frauds):

| Metric | Value | Notes |
|--------|-------|-------|
| PR-AUC | 0.7491 | Primary evaluation metric |
| ROC-AUC | 0.9516 | For reference |
| Recall | 81.6% | At cost-optimized threshold |
| Precision | 35.2% | At cost-optimized threshold |
| Cost-Optimized Threshold | 0.476 | Minimizes business cost |
| Cost Savings | $385 | vs. default threshold (0.5) on test set |

**Drift Monitoring:**
- Test data split into 3 windows (~23k transactions each)
- Alert thresholds: PSI <0.1 = GREEN, 0.1-0.25 = AMBER, >0.25 = RED
- Alert level determined by: max PSI across top 10 features
- Result: 3 of 3 windows RED → retrain trigger fired

**Cost Assumptions:**
- False Negative (missed fraud): $122.21 (average fraud amount from data)
- False Positive (blocked legitimate): $5.00 (assumed review cost)

---

## Model Comparison

Evaluated on **test set**:

| Model | PR-AUC | ROC-AUC | Notes |
|-------|--------|---------|-------|
| Random Forest | 0.7491 | 0.9516 | Best performance, class-weighted |
| Logistic Regression | 0.7123 | 0.9381 | Fast baseline |
| Isolation Forest | 0.1842 | 0.7210 | Unsupervised, no fraud labels used |

Isolation Forest scores significantly lower because it learns only from data patterns without fraud labels during training. It could be useful as a complementary detector for novel anomalies not seen in training data, but this hypothesis was not tested in this project.

---

## Repository Structure

```
Fraud-Drift-Monitoring/
├── README.md                    # This file
├── WRITEUP.md                   # 1-2 page technical explanation
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
├── run_all.py                   # Master pipeline script
├── run_evaluation.py            # Re-run evaluation only
├── run_exploration.py           # Exploratory data analysis
│
├── data/
│   ├── README.md               # Dataset download instructions
│   └── creditcard.csv          # (download from Kaggle, gitignored)
│
├── src/                        # Modular source code
│   ├── config.py               # Hyperparameters, paths, thresholds
│   ├── data_prep.py            # Load, time-based split, scaling
│   ├── imbalance.py            # Class weighting / SMOTE
│   ├── train_supervised.py     # Logistic Regression, Random Forest
│   ├── train_unsupervised.py   # Isolation Forest
│   ├── evaluate.py             # PR-AUC, metrics, model comparison
│   ├── cost_analysis.py        # FP vs FN cost, threshold optimization
│   ├── drift.py                # PSI, KS test, score drift
│   ├── alerts.py               # Alert logic, retrain triggers
│   └── simulate_drift.py       # BONUS: artificial drift injection
│
├── models/                     # Saved models (gitignored)
│   ├── random_forest_model.pkl
│   └── scaler.pkl
│
├── results/                    # Outputs
│   ├── metrics.csv             # Model comparison table
│   ├── drift_report.json       # Drift monitoring results
│   ├── evaluation_report.json  # Re-evaluation results
│   └── figures/                # PR curves, cost plots
│       ├── pr_curves.png
│       └── cost_vs_threshold.png
│
├── notebooks/                  # Jupyter notebooks
│   └── 01_eda.ipynb           # Exploratory data analysis
│
└── streamlit_app/              # Optional interactive web demo
    ├── Home.py
    └── pages/
```

---

## Requirements Completion

| Requirement | Implementation | File |
|-------------|----------------|------|
| 1. Handle class imbalance | Compared 4 strategies, chose class weights | `src/imbalance.py` |
| 2. Supervised + Unsupervised | Random Forest + Isolation Forest | `src/train_supervised.py`, `src/train_unsupervised.py` |
| 3. Evaluate with PR-AUC | Explained why not accuracy/ROC-AUC alone | `src/evaluate.py`, `WRITEUP.md` |
| 4. Drift detection | PSI, KS test, score drift, PR-AUC tracking | `src/drift.py` |
| 5. Alert thresholds | GREEN/AMBER/RED, 2-window retrain trigger | `src/alerts.py` |
| 6. FP vs FN cost analysis | Cost-optimized threshold, $385 savings | `src/cost_analysis.py` |
| 7. BONUS: Drift simulation | Artificial drift injected and detected | `src/simulate_drift.py` |

See [WRITEUP.md](WRITEUP.md) for detailed technical explanation.

---

## Dataset

**Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)

- Total: 284,807 transactions
- Frauds: 492 (0.173%)
- Imbalance ratio: 577:1
- Features: Time, V1-V28 (PCA-transformed), Amount
- Time span: 2 days

**Download:** See [data/README.md](data/README.md)

---

## Tech Stack

- **Python:** 3.9+
- **ML:** scikit-learn 1.3.0
- **Data:** pandas 2.0.3, numpy 1.24.3
- **Imbalance:** imbalanced-learn 0.11.0
- **Stats:** scipy 1.11.1
- **Viz:** matplotlib 3.7.2, seaborn 0.12.2
- **Persistence:** joblib 1.3.1
- **Web (optional):** streamlit 1.31.0

**Note:** Results may vary slightly across library versions due to internal implementation changes.

---

## Interactive Web Demo

The `streamlit_app/` folder contains an optional interactive web application for exploring the model. This is NOT part of the core requirements but provides a user-friendly interface for testing predictions.

```bash
cd streamlit_app
streamlit run Home.py
```

Opens at http://localhost:8501

---

## Pipeline Workflow

1. Load data and perform time-based split (60% train, 15% val, 25% test)
2. Scale Time and Amount features (V1-V28 already scaled from PCA)
3. Train supervised models with class weights
4. Train unsupervised Isolation Forest
5. Evaluate with PR-AUC on test set
6. Optimize threshold using business costs
7. Monitor drift across 3 test windows (excluding Time feature)
8. Generate alerts based on max PSI across features
9. Simulate artificial drift (BONUS)
10. Save models, metrics, and reports

**Note on Time feature:** Time is excluded from drift detection because it represents seconds since first transaction. Later windows naturally have different Time values by construction, which would trigger false drift alerts. Time is included as a model input but is not monitored for drift.

---

## Key Learnings

**What worked:**
- Time-based split simulates real deployment
- Class weights effectively handle extreme imbalance
- PR-AUC provides realistic evaluation for imbalanced data
- Cost-based threshold optimization aligns with business objectives

**Limitations:**
- Dataset spans only 2 days (limited long-term drift observation)
- Drift windows come from same 2-day span, so alerts may reflect time-of-day patterns rather than real fraud evolution
- V1-V28 are PCA-anonymized (cannot interpret business meaning)
- Review cost ($5) is assumed, not from real operations data
- Drift simulation is synthetic, not real fraud evolution
- Claims about "early detection" and "business value" are based on offline evaluation, not production deployment

---

## Future Improvements

- Implement automatic retraining loop
- Try deep learning (Autoencoder for anomaly detection)
- Deploy as real-time API with FastAPI
- A/B test threshold in production environment
- Implement proper time-series cross-validation
- Test Isolation Forest's ability to detect novel anomalies

---

**For technical details, see [WRITEUP.md](WRITEUP.md)**
