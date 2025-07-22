#!/bin/bash

echo "🌌 Starting OMNI-BRAIN BINARY AI Telegram Bot..."

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

echo "🚀 Launching bot..."
echo "📱 Telegram Token: 7604218758:AAHJj2zMDTfVwyJHpLClVCDzukNr2Psj-38"
echo "💬 Search for your bot on Telegram and send /start"
echo ""

# Run the bot
python3 working_bot.py