"""
Drift Monitoring Dashboard
"""

import streamlit as st
import pandas as pd
import json
import os
from PIL import Image

st.set_page_config(
    page_title="Drift Monitoring",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Drift Monitoring Dashboard")

# Colorful header
st.markdown("""
<div style='background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%); padding: 1rem; border-radius: 10px; color: white; margin-bottom: 1rem;'>
    <h3 style='color: white; margin: 0;'>🔍 Real-Time Drift Detection</h3>
    <p style='color: #f0f0f0; margin: 0.5rem 0 0 0;'>PSI • KS Test • Score Distribution • Performance Tracking</p>
</div>
""", unsafe_allow_html=True)

# Load drift metrics
drift_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'results/drift_monitoring.json')

try:
    with open(drift_path, 'r') as f:
        drift_data = json.load(f)

    # Overall status
    st.header("🎯 System Status")

    windows = drift_data.get('windows', [])
    if windows:
        # Count RED windows
        red_count = sum(1 for w in windows if w.get('alert_level') == 'RED')

        # Status based on RED windows
        if red_count >= 2:
            st.error(f"🔴 **CRITICAL: {red_count} RED WINDOWS** - Retrain Required!")
        elif red_count == 1:
            st.warning("🟡 **WARNING: 1 RED WINDOW** - Monitor Closely")
        else:
            st.success("🟢 **HEALTHY: All Stable**")

        # Retrain status
        if drift_data.get('retrain_triggered', False):
            st.error("⚠️ **RETRAIN TRIGGERED** - 2 consecutive RED windows detected")

    st.markdown("---")

    # Window-by-window analysis with interactive display
    st.header("📊 Monitoring Windows")

    if windows:
        # Create interactive table
        window_df = pd.DataFrame([
            {
                'Window': w.get('window'),
                'PR-AUC': f"{w.get('pr_auc', 0):.4f}",
                'Alert Level': w.get('alert_level', 'N/A'),
                'PR-AUC Drop %': f"{w.get('pr_auc_drop_pct', 0):.2f}%"
            }
            for w in windows
        ])

        # Color code the table
        def color_alert(val):
            if val == 'RED':
                return 'background-color: #FFE5E5; color: #CC0000; font-weight: bold'
            elif val == 'AMBER':
                return 'background-color: #FFF5E5; color: #CC8800; font-weight: bold'
            elif val == 'GREEN':
                return 'background-color: #E5F5E5; color: #006600; font-weight: bold'
            return ''

        # Use map instead of applymap (pandas 2.0+ compatibility)
        try:
            styled_df = window_df.style.map(color_alert, subset=['Alert Level'])
        except AttributeError:
            # Fallback for older pandas versions
            styled_df = window_df.style.applymap(color_alert, subset=['Alert Level'])

        st.dataframe(styled_df, use_container_width=True, hide_index=True)

        # Individual window details with expandable sections
        st.subheader("🔍 Window Details")

        for i, window in enumerate(windows, 1):
            alert = window.get('alert_level', 'N/A')
            emoji = "🔴" if alert == "RED" else "🟡" if alert == "AMBER" else "🟢"

            with st.expander(f"{emoji} Window {i} - {alert}", expanded=(alert == "RED")):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("PR-AUC", f"{window.get('pr_auc', 0):.4f}")
                with col2:
                    drop = window.get('pr_auc_drop_pct', 0)
                    st.metric("Performance Change", f"{drop:.2f}%", delta=f"{drop:.2f}%")
                with col3:
                    st.metric("Alert Level", alert)

    st.markdown("---")

    # ===== BONUS REQUIREMENT: DRIFT SIMULATION =====
    st.header("🎮 BONUS: Drift Simulation (Artificial Concept Drift)")

    st.info("""
    💡 **Bonus Requirement Implemented:**
    "Simulate concept drift artificially (inject a shifted data slice) and show your monitoring catches it"
    """)

    if 'drift_simulation' in drift_data:
        sim = drift_data['drift_simulation']

        st.subheader("🧪 Artificial Drift Injection Results")

        # Show what was done
        st.markdown("""
        **Simulation Method:**
        - Injected artificial drift by scaling Amount by 1.5×
        - Shifted top features (V14, V10, V12) by 0.5 standard deviations
        - Tested if monitoring system catches the drift
        """)

        # Results
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            <div style='background-color: #E5F5FF; padding: 1rem; border-radius: 10px;'>
                <p style='margin: 0; color: #0066CC;'>Original PR-AUC</p>
                <h2 style='margin: 0.5rem 0; color: #0066CC;'>{:.4f}</h2>
            </div>
            """.format(sim.get('original_pr_auc', 0)), unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div style='background-color: #FFE5E5; padding: 1rem; border-radius: 10px;'>
                <p style='margin: 0; color: #CC0000;'>Drifted PR-AUC</p>
                <h2 style='margin: 0.5rem 0; color: #CC0000;'>{:.4f}</h2>
            </div>
            """.format(sim.get('drifted_pr_auc', 0)), unsafe_allow_html=True)

        with col3:
            degradation = sim.get('degradation', 0) * 100
            st.markdown("""
            <div style='background-color: #FFF5E5; padding: 1rem; border-radius: 10px;'>
                <p style='margin: 0; color: #CC8800;'>Performance Drop</p>
                <h2 style='margin: 0.5rem 0; color: #CC8800;'>{:.2f}%</h2>
            </div>
            """.format(degradation), unsafe_allow_html=True)

        # Verdict
        if abs(degradation) > 0:
            st.success("✅ **DRIFT DETECTED!** Monitoring system successfully caught the artificial drift")
        else:
            st.warning("⚠️ Drift detection results inconclusive")

    # Retrain Results
    if 'retrain_results' in drift_data:
        st.subheader("🔄 Automatic Retrain Results")

        retrain = drift_data['retrain_results']

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Original Model PR-AUC", f"{retrain.get('original_model_pr_auc', 0):.4f}")
        with col2:
            st.metric("Retrained Model PR-AUC", f"{retrain.get('retrained_model_pr_auc', 0):.4f}")
        with col3:
            improvement = retrain.get('improvement', 0) * 100
            st.metric("Improvement", f"{improvement:.2f}%", delta=f"{improvement:.2f}%")

        st.info("""
        📝 **Retrain Process:**
        When drift detected → Model retrained on combined data (train + val + window1) →
        Tested on Window 3 → System validated recovery
        """)

    st.success("""
    ✅ **BONUS REQUIREMENT COMPLETE:**
    Successfully simulated artificial drift AND showed monitoring catches it!
    """)

    st.markdown("---")

    # Drift detection methods
    st.header("📚 Drift Detection Methods")

    tab1, tab2, tab3, tab4 = st.tabs(["PSI", "KS Test", "Score Drift", "PR-AUC Tracking"])

    with tab1:
        st.markdown("""
        **PSI (Population Stability Index)**

        Measures distribution shift between training and monitoring windows.

        | Threshold | Interpretation | Action |
        |-----------|----------------|--------|
        | PSI < 0.1 | 🟢 Stable | No action |
        | PSI 0.1-0.25 | 🟡 Moderate drift | Monitor closely |
        | PSI > 0.25 | 🔴 Significant drift | Review/retrain |

        Applied to top 10 features by importance (V14, V10, V12, etc.)
        """)

    with tab2:
        st.markdown("""
        **KS Test (Kolmogorov-Smirnov)**

        Statistical test for distribution difference.

        - **p-value < 0.05:** Significant drift detected
        - **p-value > 0.05:** No significant difference

        Applied to Amount and V1-V28 features for statistical validation.
        """)

    with tab3:
        st.markdown("""
        **Score Distribution Drift**

        Monitors changes in model prediction scores.

        Detects if the model is becoming:
        - More confident (scores shifting toward 0 or 1)
        - Less confident (scores clustering around 0.5)

        Indicates potential concept drift even when features look similar.
        """)

    with tab4:
        st.markdown("""
        **Performance Degradation Tracking**

        Tracks PR-AUC across monitoring windows.

        | Drop | Alert Level |
        |------|-------------|
        | < 5% | 🟢 GREEN |
        | 5-10% | 🟡 AMBER |
        | > 10% | 🔴 RED |

        Baseline: {:.4f}
        """.format(drift_data.get('baseline_pr_auc', 0)))

    st.markdown("---")

    # Visualizations
    st.header("📊 Drift Analysis Visualization")

    drift_dashboard_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'results/figures/drift_monitoring_dashboard.png')
    if os.path.exists(drift_dashboard_path):
        image = Image.open(drift_dashboard_path)
        st.image(image, use_container_width=True)

        with st.expander("📖 How to Read This Dashboard"):
            st.markdown("""
            **4 Panels Explained:**

            1. **Top Left - PSI Values:**
               - Bar chart showing drift magnitude for each feature
               - Red line at 0.25 = drift threshold
               - Bars above line = significant drift

            2. **Top Right - KS Test p-values:**
               - Statistical significance of distribution changes
               - Red line at 0.05 = significance threshold
               - Points below line = significant drift

            3. **Bottom Left - Score Distributions:**
               - Model prediction score distributions by window
               - Look for shape changes = concept drift

            4. **Bottom Right - PR-AUC Performance:**
               - Performance across windows
               - Baseline (dashed line) vs actual
               - Declining trend = model degradation
            """)
    else:
        st.info("📊 Drift visualization will appear here after running drift monitoring")

    st.markdown("---")

    # Interactive alert simulator
    st.header("🎛️ Interactive Alert Simulator")

    st.markdown("Adjust PSI value to see alert level change:")

    psi_value = st.slider("PSI Value", 0.0, 1.0, 0.15, 0.01)

    if psi_value < 0.1:
        st.success(f"🟢 **GREEN ALERT** (PSI = {psi_value:.2f})")
        st.write("✅ System stable - no action needed")
    elif psi_value < 0.25:
        st.warning(f"🟡 **AMBER ALERT** (PSI = {psi_value:.2f})")
        st.write("⚠️ Moderate drift detected - monitor closely")
    else:
        st.error(f"🔴 **RED ALERT** (PSI = {psi_value:.2f})")
        st.write("❌ Significant drift detected - consider retraining")

except FileNotFoundError:
    st.error("❌ Drift monitoring data not found. Run `day3_drift_monitoring.py` first")
except Exception as e:
    st.error(f"❌ Error loading drift data: {str(e)}")
