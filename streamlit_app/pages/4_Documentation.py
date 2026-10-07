"""
Technical Documentation
"""

import streamlit as st

st.set_page_config(
    page_title="Documentation",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Technical Documentation")

# Colorful header
st.markdown("""
<div style='background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; color: white; margin-bottom: 1rem;'>
    <h3 style='color: white; margin: 0;'>📖 Project Overview & Technical Details</h3>
    <p style='color: #f0f0f0; margin: 0.5rem 0 0 0;'>Everything you need to know about this fraud detection system</p>
</div>
""", unsafe_allow_html=True)

# Project overview
st.header("🎯 Project Overview")
st.markdown("""
Credit card fraud detection system addressing severely imbalanced data (0.173% fraud rate, 577:1 imbalance).

**Features:**
- Supervised & unsupervised ML models
- Time-based validation (not random split)
- Cost-optimized thresholds ($385 savings)
- Automated drift monitoring
- Production-ready alerts
""")

st.markdown("---")

# Dataset
st.header("📊 Dataset")
st.markdown("""
**Source:** Kaggle Credit Card Fraud Detection Dataset

| Metric | Value |
|--------|-------|
| Total Transactions | 284,807 |
| Fraudulent | 492 (0.173%) |
| Legitimate | 284,315 (99.827%) |
| Imbalance Ratio | 577:1 |
| Time Span | 48 hours |

**Features:**
- **Time:** Seconds elapsed
- **V1-V28:** PCA-transformed (anonymized)
- **Amount:** Transaction amount ($)
- **Class:** 0=legitimate, 1=fraud
""")

st.markdown("---")

# Model architecture
st.header("🤖 Model Architecture")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Random Forest (Primary)")
    st.markdown("""
    - **Trees:** 100
    - **Max Depth:** 10
    - **Class Weight:** Balanced
    - **PR-AUC:** 0.7491
    - **Recall:** 81.6%
    """)

with col2:
    st.subheader("Isolation Forest (Comparison)")
    st.markdown("""
    - **Unsupervised** approach
    - **PR-AUC:** 0.0208 (failed)
    - **Reason:** Frauds aren't always outliers
    - **Conclusion:** Supervised learning necessary
    """)

st.markdown("---")

# Data pipeline
st.header("⚙️ Data Pipeline")

st.markdown("""
**1. Time-Based Split:**
```
Hours 0-33  (60%) → Training
Hours 33-39 (15%) → Validation
Hours 39-48 (25%) → Test (3 windows)
```

**2. Imbalance Handling:**
- ✅ Class weights (chosen - PR-AUC 0.7508)
- SMOTE oversampling (PR-AUC 0.7505)
- Isolation Forest (PR-AUC 0.0208)
- Baseline (PR-AUC 0.6252)

**3. Threshold Optimization:**
- Cost function: (FP × $5) + (FN × $122)
- Optimal threshold: 0.476
- Savings: $385 vs default
""")

st.markdown("---")

# Drift monitoring
st.header("🔍 Drift Monitoring")

st.markdown("""
**Metrics Tracked:**

| Metric | What It Does | Threshold |
|--------|-------------|-----------|
| PSI | Distribution shift | >0.25 = RED |
| KS Test | Statistical difference | p<0.05 = drift |
| Score Drift | Model confidence changes | Monitor |
| PR-AUC | Performance degradation | >10% drop = RED |

**Alert Levels:**
- 🟢 **GREEN:** All stable
- 🟡 **AMBER:** Moderate drift
- 🔴 **RED:** Significant drift
- ⚠️ **RETRAIN:** 2 consecutive RED windows
""")

st.markdown("---")

# Feature importance
st.header("📈 Feature Importance")

st.markdown("""
**Top Features:**

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | V14 | 14.8% |
| 2 | V10 | 9.2% |
| 3 | V12 | 8.5% |
| 4 | V4 | 7.8% |
| ... | Amount | **2.1%** ⚠️ |

💡 **Key Insight:** Amount contributes only 2.1% to predictions.
Fraud patterns are in V1-V28 features, not transaction size.
""")

st.markdown("---")

# Technology stack
st.header("🛠️ Technology Stack")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Core:**
    - Python 3.8+
    - pandas, numpy
    - scikit-learn 1.2.2
    - imbalanced-learn
    """)

with col2:
    st.markdown("""
    **Deployment:**
    - Streamlit (web interface)
    - joblib (model persistence)
    - matplotlib/seaborn (viz)
    """)

st.markdown("---")

# Limitations
st.header("⚠️ Known Limitations")

st.warning("""
1. **Dataset:** Only 48 hours of transactions, 492 fraud examples
2. **Features:** V1-V28 are anonymized (can't interpret business meaning)
3. **Costs:** Review cost ($5) is estimated, not from real data
4. **Drift:** Can detect input drift, not pure concept drift
""")

st.markdown("---")

# Reproduction
st.header("🔄 Reproduction")

st.code("""
# Clone and setup
cd fraud_detection_project
pip install -r requirements.txt

# Run pipeline
python day2_modeling.py       # Train model (5-7 min)
python day3_drift_monitoring.py  # Drift monitoring (3-4 min)

# Launch web interface
cd streamlit_app
streamlit run Home.py
""", language="bash")

st.success("✅ Expected runtime: ~15 minutes total")

st.markdown("---")

# Contact
st.header("📧 Contact")

st.markdown("""
**Project Type:** AWS Student Builder Group - AI/ML Track
**Purpose:** Academic demonstration
**Year:** 2026
**License:** MIT
""")
