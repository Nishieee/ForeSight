#!/bin/bash

# ForeSight Project Runner Script

echo "================================="
echo "ForeSight - AI Early Warning System"
echo "================================="
echo ""

# Activate virtual environment
source venv/bin/activate

# Check if models exist
if [ ! -f "models/lightgbm_model.pkl" ]; then
    echo "🔧 Training models and generating data..."
    echo "This will take 5-10 minutes..."
    echo ""
    python train_pipeline.py
    echo ""
    echo "✅ Models trained successfully!"
else
    echo "✅ Models already exist. Skipping training."
fi

echo ""
echo "🚀 Starting Streamlit Dashboard..."
echo "Dashboard will open at: http://localhost:8501"
echo ""

# Start Streamlit
streamlit run app/app.py

