#!/bin/bash

echo "🔐 Starting Private OMNI-BRAIN BINARY AI..."
echo "🛡️ AUTHORIZED ACCESS ONLY - User ID: 7700105638"

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3."
    exit 1
fi

# Check if requests is installed
python3 -c "import requests" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📦 Installing requests library..."
    python3 -m pip install requests --break-system-packages --quiet
fi

echo "🚀 Launching private bot..."
echo "📱 Telegram Bot: @Ghost_Em_bot"
echo "🔐 Only user ID 7700105638 can access this bot"
echo "🚫 All other users will receive an 'Access Denied' message"
echo ""

# Stop any existing bot processes
pkill -f "python3.*bot.py" 2>/dev/null || true

# Run the private bot
python3 private_bot.py