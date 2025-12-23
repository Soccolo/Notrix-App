#!/bin/bash

# ========================================
#    Financial Command Center - Launcher
# ========================================

# Navigate to app directory (Git Bash uses forward slashes)
cd "/c/Users/crist/OneDrive/Documents/Notrix/Notrix App"

echo "========================================"
echo "   Financial Command Center"
echo "========================================"
echo ""

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Run the Streamlit app
echo ""
echo "Starting the app..."
streamlit run app.py
