#!/bin/bash

# ========================================
#    Financial Command Center - Launcher
#    (with Virtual Environment)
# ========================================

# Navigate to app directory (Git Bash uses forward slashes)
APP_DIR="/c/Users/crist/OneDrive/Documents/Notrix/Notrix App"
cd "$APP_DIR"

echo "========================================"
echo "   Financial Command Center"
echo "========================================"
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
    echo "Virtual environment created!"
    echo ""
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/Scripts/activate

# Install/update requirements
echo "Installing dependencies..."
pip install -r requirements.txt --upgrade --quiet

# Download TextBlob data if needed (for sentiment analysis)
python -m textblob.download_corpora lite 2>/dev/null

echo ""
echo "Starting the app..."
echo "App will open at: http://localhost:8501"
echo ""

# Run the Streamlit app
streamlit run app.py
