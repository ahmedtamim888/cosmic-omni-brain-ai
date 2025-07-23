#!/bin/bash

echo "🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥"
echo "🚀 STARTING ULTIMATE GOD-TIER TRADING BOT"
echo "⚡ Ultra-Accurate Binary Options Analysis"
echo "💎 95%+ Confidence or NO TRADE"
echo "🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python3."
    exit 1
fi

# Check if required packages are installed
echo "📦 Checking dependencies..."
python3 -c "import telegram, cv2, numpy, PIL" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Installing missing dependencies..."
    pip3 install python-telegram-bot opencv-python numpy pillow pandas scikit-learn --quiet
fi

echo "✅ Dependencies ready"
echo ""

# Start the bot
echo "🚀 Launching God-Tier Bot..."
echo "📱 Send chart screenshots to your Telegram bot"
echo "⚡ Press Ctrl+C to stop"
echo ""

python3 god_tier_bot_fixed.py