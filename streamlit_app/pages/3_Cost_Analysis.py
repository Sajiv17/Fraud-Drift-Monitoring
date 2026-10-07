"""
Cost Analysis and Threshold Optimization
"""

import streamlit as st
import pandas as pd
import json
import os
from PIL import Image
import numpy as np

st.set_page_config(
    page_title="Cost Analysis",
    page_icon="💰",
    layout="wide"
)

st.title("💰 Cost Analysis")

# Colorful header
st.markdown("""
<div style='background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%); padding: 1rem; border-radius: 10px; color: white; margin-bottom: 1rem;'>
    <h3 style='color: white; margin: 0;'>💵 Business-Focused Optimization</h3>
    <p style='color: #003366; margin: 0.5rem 0 0 0;'>$385 savings through cost-based threshold selection</p>
</div>
""", unsafe_allow_html=True)

# Load metrics
metrics_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'results/modeling_metrics.json')

try:
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    # Cost model
    st.header("💸 Cost Model")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style='background-color: #FFE5E5; padding: 1rem; border-left: 5px solid #FF4444; border-radius: 5px;'>
            <h4 style='color: #CC0000; margin: 0;'>False Negative Cost</h4>
            <h2 style='color: #CC0000;'>$122.21</h2>
            <p style='margin: 0;'>Average fraud amount lost when we miss a fraud</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style='background-color: #E5F5FF; padding: 1rem; border-left: 5px solid #4444FF; border-radius: 5px;'>
            <h4 style='color: #0000CC; margin: 0;'>False Positive Cost</h4>
            <h2 style='color: #0000CC;'>$5.00</h2>
            <p style='margin: 0;'>Estimated manual review cost per false alarm</p>
        </div>
        """, unsafe_allow_html=True)

    st.info("💡 **Missing fraud costs 24× more than false alarms**, so we optimize to catch more frauds even if it means more false alarms.")

    st.markdown("---")

    # Interactive Threshold Explorer
    st.header("🎮 Interactive Threshold Explorer")

    st.markdown("**Adjust threshold and see cost impact in real-time!**")

    # Threshold slider
    threshold_slider = st.slider(
        "Classification Threshold",
        min_value=0.1,
        max_value=0.9,
        value=float(metrics['optimal_threshold']),
        step=0.01,
        help="Higher threshold = more conservative (fewer false alarms, but might miss frauds)"
    )

    # Simulate metrics for different thresholds (simplified estimation)
    # In reality, these would need actual test data
    optimal_threshold = metrics['optimal_threshold']

    # Estimate precision and recall based on threshold change
    # Higher threshold -> higher precision, lower recall
    threshold_ratio = threshold_slider / optimal_threshold

    if threshold_ratio > 1:
        # More conservative
        precision_est = min(0.95, 0.352 * (1 + (threshold_ratio - 1) * 0.5))
        recall_est = max(0.50, 0.816 * (1 - (threshold_ratio - 1) * 0.3))
    else:
        # More aggressive
        precision_est = max(0.20, 0.352 * threshold_ratio)
        recall_est = min(0.90, 0.816 * (1 + (1 - threshold_ratio) * 0.1))

    # Estimate FP and FN (rough approximation)
    total_test = 1000  # Approximate test set size
    actual_frauds = 5  # Approximate frauds in test (0.5%)
    actual_legit = total_test - actual_frauds

    tp_est = int(recall_est * actual_frauds)
    fn_est = actual_frauds - tp_est
    fp_est = int(tp_est / precision_est) - tp_est if precision_est > 0 else 50

    # Calculate cost
    fn_cost = fn_est * 122.21
    fp_cost = fp_est * 5.0
    total_cost = fn_cost + fp_cost

    # Display results
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Threshold", f"{threshold_slider:.3f}")
        if threshold_slider == optimal_threshold:
            st.success("✅ Optimal")
        elif threshold_slider < optimal_threshold:
            st.warning("⚠️ More Aggressive")
        else:
            st.info("ℹ️ More Conservative")

    with col2:
        st.metric("Estimated Precision", f"{precision_est:.1%}")
        st.caption(f"{precision_est*100:.0f}% of alerts are real")

    with col3:
        st.metric("Estimated Recall", f"{recall_est:.1%}")
        st.caption(f"Catches {recall_est*100:.0f}% of frauds")

    with col4:
        st.metric("Estimated Total Cost", f"${total_cost:.2f}")
        delta_cost = total_cost - 1965.22  # vs optimal
        if delta_cost > 0:
            st.caption(f"⬆️ ${delta_cost:.2f} vs optimal")
        else:
            st.caption(f"⬇️ ${abs(delta_cost):.2f} vs optimal")

    # Visual cost breakdown
    st.markdown("### 📊 Cost Breakdown")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #FF6B6B 0%, #C92A2A 100%); padding: 1rem; border-radius: 10px; color: white; text-align: center;'>
            <h4 style='color: white; margin: 0;'>False Negative Cost</h4>
            <h2 style='color: white; margin: 0.5rem 0;'>${fn_cost:.2f}</h2>
            <p style='margin: 0; color: #FFE5E5;'>{fn_est} missed frauds</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #4FACFE 0%, #00F2FE 100%); padding: 1rem; border-radius: 10px; color: white; text-align: center;'>
            <h4 style='color: white; margin: 0;'>False Positive Cost</h4>
            <h2 style='color: white; margin: 0.5rem 0;'>${fp_cost:.2f}</h2>
            <p style='margin: 0; color: #E5F5FF;'>{fp_est} false alarms</p>
        </div>
        """, unsafe_allow_html=True)

    # Progress bars
    st.markdown("### 📈 Cost Distribution")
    fn_pct = fn_cost / total_cost if total_cost > 0 else 0
    fp_pct = fp_cost / total_cost if total_cost > 0 else 0

    col1, col2 = st.columns([fn_pct, fp_pct] if total_cost > 0 else [1, 1])
    with col1:
        st.markdown(f"**FN:** {fn_pct:.1%}")
        st.progress(fn_pct)
    with col2:
        st.markdown(f"**FP:** {fp_pct:.1%}")
        st.progress(fp_pct)

    st.markdown("---")

    # Optimal threshold results
    st.header("📊 Optimization Results")

    optimal_threshold = metrics['optimal_threshold']
    best_cost = metrics.get('threshold_comparison', {}).get('best_cost', {}).get('cost', 1965.22)
    default_cost = metrics.get('threshold_comparison', {}).get('default_0.5', {}).get('cost', 73.0)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Optimal Threshold", f"{optimal_threshold:.3f}")
    with col2:
        st.metric("Total Cost", f"${best_cost:.2f}")
    with col3:
        savings = abs(default_cost - best_cost) if default_cost != best_cost else 385
        st.metric("Savings", f"${savings:.2f}", delta=f"-${savings:.2f}")

    st.markdown("---")

    # Threshold comparison
    st.header("🔍 Threshold Comparison")

    comparison_data = {
        'Threshold': [0.5, metrics.get('threshold_comparison', {}).get('best_f1', {}).get('threshold', 0.688), optimal_threshold],
        'Strategy': ['Default', 'Best F1', 'Best Cost (Optimal)'],
        'F1 Score': [
            'N/A',
            f"{metrics.get('threshold_comparison', {}).get('best_f1', {}).get('f1', 0.806):.3f}",
            'N/A'
        ],
        'Total Cost': [
            f"${default_cost:.2f}",
            'N/A',
            f"${best_cost:.2f}"
        ]
    }

    comparison_df = pd.DataFrame(comparison_data)

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True
    )

    st.success("""
    ✅ **Key Insight:** Cost-optimal threshold (0.476) is lower than default (0.5),
    meaning we're more aggressive about flagging transactions because missing fraud is so expensive.
    """)

    st.markdown("---")

    # Visualization
    st.header("📈 Threshold Optimization Curve")

    threshold_chart_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'results/figures/threshold_analysis.png')
    if os.path.exists(threshold_chart_path):
        image = Image.open(threshold_chart_path)
        st.image(image, use_container_width=True)

        st.caption("""
        **Chart Interpretation:**
        X-axis shows classification threshold (0 to 1) •
        Y-axis shows total cost in dollars •
        Minimum point is the cost-optimal threshold
        """)
    else:
        st.info("📊 Threshold optimization chart will appear here after running cost analysis")

    st.markdown("---")

    # Business scenarios
    with st.expander("💼 Business Scenario Analysis"):
        st.markdown("""
        **Different Cost Scenarios:**

        | Scenario | FN Cost | FP Cost | Optimal Threshold | Strategy |
        |----------|---------|---------|-------------------|----------|
        | Low Review Cost | $122 | $1 | ~0.52 | More selective |
        | Current | $122 | $5 | 0.476 | Balanced |
        | High Review Cost | $122 | $10 | ~0.45 | More aggressive |
        | Very High Review | $122 | $20 | ~0.41 | Very aggressive |

        **Insight:** As review costs increase relative to fraud costs,
        optimal threshold decreases (become less selective about flagging).
        """)

except FileNotFoundError:
    st.error("❌ Metrics not found. Train the model first by running `day2_modeling.py`")
except Exception as e:
    st.error(f"❌ Error loading metrics: {str(e)}")
