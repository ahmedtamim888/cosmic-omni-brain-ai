#!/bin/bash
# ULTRA GOD MODE AI STARTUP SCRIPT

echo "🧬 STARTING ULTRA GOD MODE AI TRADING BOT..."
echo "🚀 TRANSCENDENT MARKET DOMINATION ACTIVATED"
echo "💎 BEYOND MORTAL COMPREHENSION - INFINITE PRECISION"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated"
fi

# Load environment variables
if [ -f ".env" ]; then
    export $(cat .env | grep -v '#' | xargs)
    echo "✅ Environment variables loaded"
fi

# Start the application
python ultra_god_mode_app.py
