"""
Model Performance Metrics and Evaluation
"""

import streamlit as st
import pandas as pd
import json
import os
from PIL import Image

st.set_page_config(
    page_title="Model Performance",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Model Performance")

# Add colorful header
st.markdown("""
<div style='background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; color: white; margin-bottom: 1rem;'>
    <h3 style='color: white; margin: 0;'>📈 Performance Metrics & Evaluation</h3>
    <p style='color: #f0f0f0; margin: 0.5rem 0 0 0;'>Comprehensive model evaluation on severely imbalanced data</p>
</div>
""", unsafe_allow_html=True)

# Load metrics
metrics_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'results/modeling_metrics.json')

try:
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    # Performance Summary
    st.header("Performance Summary")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("PR-AUC", f"{metrics['pr_auc']:.4f}")
        st.caption("Primary metric for imbalanced classification")
    with col2:
        st.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
        st.caption("Included for comparison")
    with col3:
        best_f1 = metrics.get('threshold_comparison', {}).get('best_f1', {}).get('f1', 0)
        st.metric("F1 Score (Best)", f"{best_f1:.4f}")
        st.caption("Best F1 achieved")

    st.markdown("---")

    # Optimal threshold metrics
    st.header("Optimal Threshold Performance")
    st.markdown(f"**Threshold:** {metrics['optimal_threshold']:.3f}")

    st.info("💡 **Note:** Detailed threshold analysis and metrics available in Cost Analysis page")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Best F1 Threshold", f"{metrics.get('threshold_comparison', {}).get('best_f1', {}).get('threshold', 0):.3f}")
        st.caption(f"F1 Score: {metrics.get('threshold_comparison', {}).get('best_f1', {}).get('f1', 0):.4f}")
    with col2:
        default_cost = metrics.get('threshold_comparison', {}).get('default_0.5', {}).get('cost', 0)
        st.metric("Default Threshold Cost", f"${default_cost:.2f}")
        st.caption("At threshold 0.5")

    st.markdown("---")

    # Strategy comparison
    st.header("Imbalance Handling Strategy Comparison")
    st.markdown("Comparison of different approaches to handle the 577:1 class imbalance")

    # Load strategy comparison if available
    strategy_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'results/strategy_comparison.json')

    if os.path.exists(strategy_path):
        with open(strategy_path, 'r') as f:
            strategies = json.load(f)

        # Convert to DataFrame
        strategy_df = pd.DataFrame(strategies).T
        strategy_df.index.name = 'Strategy'

        # Display table
        st.dataframe(
            strategy_df.style.format({
                'pr_auc': '{:.4f}',
                'roc_auc': '{:.4f}',
                'f1': '{:.4f}',
                'precision': '{:.3f}',
                'recall': '{:.3f}'
            }).highlight_max(subset=['pr_auc'], color='lightgreen'),
            use_container_width=True
        )

        st.markdown("""
        **Key Findings:**
        - Class weighting achieved best PR-AUC (0.7508)
        - SMOTE close second (0.7505)
        - Isolation Forest failed with severe imbalance (PR-AUC 0.0208)
        - Baseline approach predicts all legitimate (useless)
        """)
    else:
        st.info("Strategy comparison data not available")

    st.markdown("---")

    # Visualizations
    st.header("Performance Curves")

    col1, col2 = st.columns(2)

    # PR curve
    pr_curve_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'results/figures/evaluation_curves.png')
    if os.path.exists(pr_curve_path):
        with col1:
            st.subheader("Precision-Recall Curve")
            image = Image.open(pr_curve_path)
            st.image(image, use_container_width=True)

    # Strategy comparison chart
    strategy_chart_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'results/figures/strategy_comparison.png')
    if os.path.exists(strategy_chart_path):
        with col2:
            st.subheader("Strategy Comparison")
            image = Image.open(strategy_chart_path)
            st.image(image, use_container_width=True)

    st.markdown("---")

    # Technical details
    with st.expander("Technical Details"):
        st.markdown("""
        **Model Architecture**
        - Algorithm: Random Forest Classifier
        - Trees: 100
        - Max depth: 10
        - Min samples split: 50
        - Class weight: balanced

        **Training Data**
        - Total transactions: 284,807
        - Frauds: 492 (0.173%)
        - Legitimate: 284,315 (99.827%)
        - Imbalance ratio: 577:1

        **Validation Strategy**
        - Time-based split (not random)
        - Training: 60% (hours 0-33)
        - Validation: 15% (hours 33-39)
        - Test: 25% (3 windows, hours 39-48)

        **Feature Engineering**
        - V1-V28: PCA-transformed features (used as-is)
        - Time: Scaled using StandardScaler
        - Amount: Scaled using StandardScaler
        - Total features: 30

        **Why PR-AUC over ROC-AUC?**

        With 99.83% legitimate transactions, ROC-AUC is misleadingly optimistic.
        It includes true negative rate, and we have massive true negatives.
        PR-AUC focuses only on precision and recall for the fraud class, providing
        a realistic performance measure for severe imbalance.

        Our ROC-AUC: 0.9854 (looks great!)
        Our PR-AUC: 0.7491 (realistic assessment)
        """)

except FileNotFoundError:
    st.error("Model metrics not found. Train the model first by running `day2_modeling.py`")
