"""
Fraud Detection System - Live Prediction Interface
"""

import streamlit as st
import pandas as pd
import joblib
import json
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Page config
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model and data
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models/random_forest_model.pkl')
    scaler_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models/scaler.pkl')
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/creditcard.csv')
    metrics_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results/modeling_metrics.json')

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    df = pd.read_csv(data_path)

    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    return model, scaler, df, metrics

try:
    model, scaler, df, metrics = load_model()
    threshold = metrics['optimal_threshold']

    # Header with colored badge
    st.title("🛡️ Credit Card Fraud Detection System")
    st.markdown("""
    <div style='background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; color: white; margin-bottom: 1rem;'>
        <h3 style='color: white; margin: 0;'>🚀 Real-Time Fraud Detection</h3>
        <p style='color: #f0f0f0; margin: 0.5rem 0 0 0;'>Machine learning with drift monitoring • 0.173% fraud rate • 577:1 imbalance</p>
    </div>
    """, unsafe_allow_html=True)

    # System stats (real metrics) - Colorful cards
    st.markdown("### 📊 Model Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; text-align: center;'>
            <p style='color: #E0E0E0; margin: 0; font-size: 0.9rem;'>PR-AUC</p>
            <h2 style='color: white; margin: 0.5rem 0;'>{:.3f}</h2>
        </div>
        """.format(metrics['pr_auc']), unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 1rem; border-radius: 10px; text-align: center;'>
            <p style='color: #F0F0F0; margin: 0; font-size: 0.9rem;'>ROC-AUC</p>
            <h2 style='color: white; margin: 0.5rem 0;'>{:.3f}</h2>
        </div>
        """.format(metrics['roc_auc']), unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 1rem; border-radius: 10px; text-align: center;'>
            <p style='color: #003366; margin: 0; font-size: 0.9rem;'>Model</p>
            <h3 style='color: white; margin: 0.5rem 0; font-size: 1.2rem;'>{}</h3>
        </div>
        """.format(metrics.get('model', 'Random Forest')), unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); padding: 1rem; border-radius: 10px; text-align: center;'>
            <p style='color: #663300; margin: 0; font-size: 0.9rem;'>Threshold</p>
            <h2 style='color: white; margin: 0.5rem 0;'>{:.3f}</h2>
        </div>
        """.format(threshold), unsafe_allow_html=True)

    st.markdown("---")

    # Main prediction interface - INTERACTIVE!
    st.header("🎯 Interactive Fraud Prediction")

    # Tabs for different modes
    tab1, tab2, tab3 = st.tabs(["🎲 Real Transactions", "💰 Custom Amount", "🎮 Interactive Sliders"])

    with tab1:
        st.subheader("Select a Real Transaction")

        # Get fraud and legitimate samples
        fraud_samples = df[df['Class'] == 1].sample(n=5, random_state=42)
        legit_samples = df[df['Class'] == 0].sample(n=5, random_state=42)
        test_samples = pd.concat([fraud_samples, legit_samples]).reset_index(drop=True)

        # Display selection
        selected_idx = st.selectbox(
            "Transaction",
            range(len(test_samples)),
            format_func=lambda i: f"Transaction {i+1}: ${test_samples.iloc[i]['Amount']:.2f} ({'Fraud' if test_samples.iloc[i]['Class'] == 1 else 'Legitimate'})"
        )

        transaction = test_samples.iloc[selected_idx]

        if st.button("🔍 Predict", type="primary", key="btn1"):
            # Prepare features
            feature_cols = [c for c in df.columns if c != 'Class']
            X = pd.DataFrame([transaction[feature_cols]])
            X_scaled = X.copy()
            X_scaled[['Time', 'Amount']] = scaler.transform(X[['Time', 'Amount']])

            # Predict
            fraud_prob = model.predict_proba(X_scaled)[0][1]
            is_fraud = fraud_prob >= threshold
            actual_label = transaction['Class']

            # Big visual result
            if is_fraud:
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #FF6B6B 0%, #C92A2A 100%); padding: 2rem; border-radius: 15px; text-align: center; color: white; margin: 1rem 0;'>
                    <h1 style='color: white; margin: 0; font-size: 3rem;'>🚨 FRAUD DETECTED</h1>
                    <h2 style='color: #FFE5E5; margin: 1rem 0;'>{fraud_prob:.1%} Fraud Probability</h2>
                    <p style='color: #FFF; font-size: 1.2rem; margin: 0;'>⛔ BLOCK TRANSACTION</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #51CF66 0%, #2F9E44 100%); padding: 2rem; border-radius: 15px; text-align: center; color: white; margin: 1rem 0;'>
                    <h1 style='color: white; margin: 0; font-size: 3rem;'>✅ LEGITIMATE</h1>
                    <h2 style='color: #E5FFE5; margin: 1rem 0;'>{fraud_prob:.1%} Fraud Probability</h2>
                    <p style='color: #FFF; font-size: 1.2rem; margin: 0;'>✓ APPROVE TRANSACTION</p>
                </div>
                """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Actual Label", "Fraud" if actual_label == 1 else "Legitimate")
            with col2:
                if (is_fraud and actual_label == 1) or (not is_fraud and actual_label == 0):
                    st.success("**✅ Correct Prediction**")
                else:
                    st.error("**❌ Incorrect Prediction**")

    with tab2:
        st.subheader("Enter Custom Transaction Amount")
        st.info("💡 Note: Uses average V1-V28 feature values from the dataset")

        col1, col2 = st.columns([3, 1])
        with col1:
            amount = st.number_input(
                "Transaction Amount ($)",
                min_value=0.0,
                max_value=25000.0,
                value=100.0,
                step=10.0
            )
        with col2:
            st.write("")
            st.write("")
            predict_btn = st.button("🔍 Predict", type="primary", key="btn2")

        if predict_btn:
            # Build transaction
            transaction_dict = {'Time': 86400}
            for i in range(1, 29):
                transaction_dict[f'V{i}'] = df[f'V{i}'].mean()
            transaction_dict['Amount'] = amount

            # Prepare features
            feature_cols = [c for c in df.columns if c != 'Class']
            X = pd.DataFrame([transaction_dict])[feature_cols]
            X_scaled = X.copy()
            X_scaled[['Time', 'Amount']] = scaler.transform(X[['Time', 'Amount']])

            # Predict
            fraud_prob = model.predict_proba(X_scaled)[0][1]
            is_fraud = fraud_prob >= threshold

            # Big visual result
            if is_fraud:
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #FF6B6B 0%, #C92A2A 100%); padding: 2rem; border-radius: 15px; text-align: center; color: white; margin: 1rem 0;'>
                    <h1 style='color: white; margin: 0; font-size: 3rem;'>🚨 FRAUD DETECTED</h1>
                    <h2 style='color: #FFE5E5; margin: 1rem 0;'>{fraud_prob:.1%} Fraud Probability</h2>
                    <p style='color: #FFF; font-size: 1.2rem; margin: 0;'>⛔ BLOCK TRANSACTION</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #51CF66 0%, #2F9E44 100%); padding: 2rem; border-radius: 15px; text-align: center; color: white; margin: 1rem 0;'>
                    <h1 style='color: white; margin: 0; font-size: 3rem;'>✅ LEGITIMATE</h1>
                    <h2 style='color: #E5FFE5; margin: 1rem 0;'>{fraud_prob:.1%} Fraud Probability</h2>
                    <p style='color: #FFF; font-size: 1.2rem; margin: 0;'>✓ APPROVE TRANSACTION</p>
                </div>
                """, unsafe_allow_html=True)

            st.info("**Note:** This uses average feature values. Real fraud patterns depend on V1-V28 features, not just amount.")

    with tab3:
        st.subheader("🎮 Interactive Feature Exploration")
        st.info("Adjust transaction features and see fraud probability change in real-time!")

        col1, col2 = st.columns(2)
        with col1:
            amount_slider = st.slider("Transaction Amount ($)", 0.0, 5000.0, 100.0, 10.0)
            time_slider = st.slider("Time (hours)", 0, 48, 24, 1)

        with col2:
            v14_slider = st.slider("V14 (Top Feature)", -20.0, 20.0, df['V14'].mean(), 0.1)
            v10_slider = st.slider("V10 (2nd Feature)", -20.0, 20.0, df['V10'].mean(), 0.1)

        # Build transaction with slider values
        transaction_dict = {'Time': time_slider * 3600}
        for i in range(1, 29):
            if i == 14:
                transaction_dict[f'V{i}'] = v14_slider
            elif i == 10:
                transaction_dict[f'V{i}'] = v10_slider
            else:
                transaction_dict[f'V{i}'] = df[f'V{i}'].mean()
        transaction_dict['Amount'] = amount_slider

        # Prepare and predict
        feature_cols = [c for c in df.columns if c != 'Class']
        X = pd.DataFrame([transaction_dict])[feature_cols]
        X_scaled = X.copy()
        X_scaled[['Time', 'Amount']] = scaler.transform(X[['Time', 'Amount']])

        fraud_prob = model.predict_proba(X_scaled)[0][1]
        is_fraud = fraud_prob >= threshold

        # Real-time display
        st.markdown("### 🎯 Real-Time Prediction")

        # Progress bar for probability
        st.progress(fraud_prob)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Fraud Probability", f"{fraud_prob:.1%}")
        with col2:
            st.metric("Threshold", f"{threshold:.3f}")
        with col3:
            if is_fraud:
                st.error("🚨 **FRAUD**")
            else:
                st.success("✅ **LEGIT**")

        # Visual indicator
        if fraud_prob > threshold:
            danger_level = min(100, int((fraud_prob - threshold) / (1 - threshold) * 100))
            st.markdown(f"""
            <div style='background-color: #FFE5E5; padding: 1rem; border-left: 5px solid #FF4444; border-radius: 5px;'>
                <h4 style='color: #CC0000; margin: 0;'>⚠️ High Fraud Risk: {danger_level}%</h4>
            </div>
            """, unsafe_allow_html=True)
        else:
            safe_level = min(100, int((threshold - fraud_prob) / threshold * 100))
            st.markdown(f"""
            <div style='background-color: #E5F5E5; padding: 1rem; border-left: 5px solid #44AA44; border-radius: 5px;'>
                <h4 style='color: #006600; margin: 0;'>✅ Low Fraud Risk: {safe_level}% safe</h4>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Quick stats
    with st.expander("📊 Quick Statistics"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Dataset Size", f"{len(df):,} transactions")
        with col2:
            st.metric("Fraud Rate", f"{(df['Class'].sum() / len(df) * 100):.3f}%")
        with col3:
            st.metric("Imbalance Ratio", "577:1")

except FileNotFoundError as e:
    st.error(f"Error loading model or data: {e}")
    st.info("Make sure you've trained the model by running `day2_modeling.py` first.")
