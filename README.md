# Credit Card Fraud Detection with Drift Monitoring

**AWS Student Builder Group - AI/ML Track**

Machine learning system for credit card fraud detection addressing extreme class imbalance (0.173% fraud rate), with automated drift monitoring and cost optimization.

**Author:** Sajiv Vaila  
**Contact:** sajiv.vaila2025@vitstudent.ac.in  
**GitHub:** [@Sajiv17](https://github.com/Sajiv17)

---

## Quick Start

```bash
# Clone repository
git clone https://github.com/Sajiv17/fraud-drift-monitoring.git
cd fraud-drift-monitoring

# Install dependencies
pip install -r requirements.txt

# Download dataset from Kaggle (see data/README.md)
# Place creditcard.csv in data/

# Train models and run full pipeline
python run_all.py
```

**Runtime:** ~15 minutes on standard laptop (MacBook Pro M1, 16GB RAM)  
**Python version:** 3.8+  
**Random seed:** 42 (for reproducibility)

---

## Re-Run Evaluation

After training models with `python run_all.py`, re-run evaluation only:

```bash
# Re-evaluate trained models on validation set
python -c "
from src.train_supervised import load_model, load_scaler
from src.data_prep import load_data, time_based_split, separate_features_target
from src.evaluate import evaluate_with_pr_auc, plot_pr_curve
from src.cost_analysis import find_optimal_threshold
import os

# Load data
df = load_data()
splits = time_based_split(df)
X_val, y_val = separate_features_target(splits['val'])

# Load trained model and scaler
model = load_model('models/random_forest_model.pkl')
scaler = load_scaler('models/scaler.pkl')

# Scale features
X_val_scaled = X_val.copy()
X_val_scaled[['Time', 'Amount']] = scaler.transform(X_val[['Time', 'Amount']])

# Evaluate
y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
results = evaluate_with_pr_auc(y_val, y_pred_proba, 'Random Forest')
cost_results = find_optimal_threshold(y_val, y_pred_proba)

print(f'\nFinal Results:')
print(f'PR-AUC: {results[\"pr_auc\"]:.4f}')
print(f'Optimal Threshold: {cost_results[\"optimal_threshold\"]:.3f}')
print(f'Cost Savings: ${cost_results[\"savings\"]:.2f}')
"
```

**Note:** Models must be trained first (`python run_all.py`). The `models/` folder is gitignored, so trained models are not included in the repository.

---

## Results

| Metric | Value | Notes |
|--------|-------|-------|
| PR-AUC | 0.7491 | Primary evaluation metric |
| Recall | 81.6% | At optimal threshold (0.476) |
| Precision | 35.2% | At optimal threshold (0.476) |
| Optimal Threshold | 0.476 | vs. default 0.5 |
| Cost Savings | $385 | vs. default threshold on validation set |
| Drift Windows | 3 RED alerts | All 3 test windows triggered retrain |
| BONUS | Complete | Artificial drift simulation validated |

**Cost assumptions:**
- False Negative (missed fraud): $122.21 (average fraud transaction amount from data)
- False Positive (blocked legitimate): $5.00 (assumed review cost)

**Drift monitoring:**
- Test data split into 3 windows (each ~8% of dataset)
- Thresholds: PSI <0.1 = GREEN, 0.1-0.25 = AMBER, >0.25 = RED
- KS test: p-value <0.05 = significant drift
- Retrain triggered after 2 consecutive RED windows

---

## Model Comparison

| Model | PR-AUC | ROC-AUC | Notes |
|-------|--------|---------|-------|
| Random Forest | 0.7491 | 0.9516 | Best performance, class-weighted |
| Logistic Regression | 0.7123 | 0.9381 | Fast baseline |
| Isolation Forest | 0.6842 | 0.9102 | Unsupervised, lower PR-AUC due to no fraud labels during training |

Isolation Forest scores lower because it learns only from data patterns without fraud labels, making it less precise at identifying fraud. However, it's useful as a complementary detector for truly novel anomalies.

---

## Repository Structure

```
fraud-drift-monitoring/
├── README.md                    # This file
├── WRITEUP.md                   # 1-2 page technical explanation
├── requirements.txt             # Python dependencies
├── run_all.py                   # Master pipeline script
├── run_exploration.py           # Exploratory data analysis
│
├── data/
│   ├── README.md               # Dataset download instructions
│   └── creditcard.csv          # (download from Kaggle, not committed)
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
├── models/                     # Saved models (gitignored, train first)
├── results/                    # Metrics, drift reports, figures
├── notebooks/                  # Jupyter notebooks for EDA
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
| 6. FP vs FN cost analysis | Optimal threshold 0.476, $385 savings | `src/cost_analysis.py` |
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

- Python 3.8+
- scikit-learn 1.3.0
- pandas 2.0.3
- numpy 1.24.3
- imbalanced-learn 0.11.0
- scipy 1.11.1
- matplotlib 3.7.2
- seaborn 0.12.2
- streamlit 1.31.0 (optional, for web demo)
- joblib 1.3.1

**Note:** Results may vary slightly across library versions due to internal implementation changes.

---

## Interactive Web Demo (Optional)

The `streamlit_app/` folder contains an optional interactive web application for exploring the model. This is NOT part of the core requirements but provides a user-friendly interface for testing predictions.

```bash
cd streamlit_app
streamlit run Home.py
```

Opens at http://localhost:8501

---

## Pipeline Workflow

1. Load data and perform time-based split (60% train, 15% val, 25% test)
2. Scale Time and Amount features
3. Train supervised models with class weights
4. Train unsupervised Isolation Forest
5. Evaluate with PR-AUC
6. Optimize threshold using business costs
7. Monitor drift across 3 test windows
8. Generate alerts and check retrain triggers
9. Simulate artificial drift (BONUS)
10. Save models, metrics, and reports

---

## Key Learnings

**What worked:**
- Time-based split simulates real deployment
- Class weights effectively handle extreme imbalance
- PR-AUC provides realistic evaluation for imbalanced data
- Multi-method drift detection catches issues early
- Cost-based threshold optimization demonstrates business value

**Limitations:**
- Dataset spans only 2 days (limited long-term drift observation)
- V1-V28 are PCA-anonymized (cannot interpret business meaning)
- Review cost ($5) is assumed, not from real operations data
- Drift simulation is synthetic, not real fraud evolution

---

## Future Improvements

- Implement automatic retraining loop
- Try deep learning (Autoencoder for anomaly detection)
- Deploy as real-time API with FastAPI
- A/B test threshold in production environment
- Implement proper time-series cross-validation

---

**For technical details, see [WRITEUP.md](WRITEUP.md)**
