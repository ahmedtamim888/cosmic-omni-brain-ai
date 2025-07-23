#!/bin/bash

echo "🔥 Starting God-Tier Trading Bot - Error-Free Version 🔥"
echo "⚡ Installing dependencies..."

# Install dependencies
pip3 install python-telegram-bot opencv-python numpy --quiet

# Check if installation was successful
if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
else
    echo "⚠️ Some dependencies may have failed - continuing anyway"
fi

echo ""
echo "🚀 Launching God-Tier Trading Bot..."
echo "📱 Send chart screenshots to receive signals"
echo "💎 Only 95%+ confidence or God Mode signals"
echo ""

# Run the bot
python3 god_tier_bot_fixed_final.py