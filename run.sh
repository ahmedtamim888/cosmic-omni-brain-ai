#!/bin/bash

# 🌌 OMNI-BRAIN BINARY AI - Ultimate Adaptive Strategy Builder
# Startup script

echo "🌌 Starting OMNI-BRAIN BINARY AI..."
echo "🔧 Checking dependencies..."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ to continue."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.8"

if [[ $(echo "$PYTHON_VERSION < $REQUIRED_VERSION" | bc -l) -eq 1 ]]; then
    echo "❌ Python $PYTHON_VERSION detected. Please upgrade to Python $REQUIRED_VERSION or higher."
    exit 1
fi

echo "✅ Python $PYTHON_VERSION detected"

# Install dependencies if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "📦 Installing/checking dependencies..."
    python3 -m pip install -r requirements.txt --quiet
    if [ $? -eq 0 ]; then
        echo "✅ Dependencies ready"
    else
        echo "⚠️ Some dependencies may be missing, but continuing..."
    fi
fi

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "⚙️ Creating .env configuration file..."
    cp .env.example .env
    echo "✅ Configuration file created"
fi

echo "🚀 Launching COSMIC AI..."
echo ""

# Run the main application
python3 main.py

echo ""
echo "🛑 COSMIC AI stopped"