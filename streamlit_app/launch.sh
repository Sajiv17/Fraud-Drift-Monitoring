#!/bin/bash

# Launch script for Fraud Detection Streamlit App

echo "=========================================="
echo " Fraud Detection System - Web Interface"
echo "=========================================="
echo ""

# Check if we're in correct directory
if [ ! -f "Home.py" ]; then
    echo "❌ Error: Must run from streamlit_app directory"
    echo ""
    echo "Run:"
    echo "  cd /Users/sajivvaila/fraud_detection_project/streamlit_app"
    echo "  ./launch.sh"
    exit 1
fi

# Check Python version
python3 --version > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "❌ Python 3 not found"
    echo "Install Python 3.8+ first"
    exit 1
fi

# Check if dependencies installed
python3 -c "import streamlit" > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "📦 Installing dependencies..."
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Installation failed"
        exit 1
    fi
fi

# Check if model exists
if [ ! -f "../models/random_forest_model.pkl" ]; then
    echo "❌ Model not found: ../models/random_forest_model.pkl"
    echo ""
    echo "Train the model first:"
    echo "  cd /Users/sajivvaila/fraud_detection_project"
    echo "  python3 day2_modeling.py"
    exit 1
fi

# Check if data exists
if [ ! -f "../data/creditcard.csv" ]; then
    echo "❌ Dataset not found: ../data/creditcard.csv"
    echo ""
    echo "Download the Kaggle dataset first"
    exit 1
fi

echo "✅ All dependencies ready"
echo ""
echo "🚀 Starting Streamlit app..."
echo ""
echo "The app will open in your browser at:"
echo "   http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop"
echo ""
echo "=========================================="
echo ""

# Launch Streamlit
streamlit run Home.py
