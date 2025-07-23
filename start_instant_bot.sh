#!/bin/bash

echo "⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡"
echo "🚀 INSTANT GOD-TIER TRADING BOT"
echo "⚡ ZERO DELAY - LIGHTNING FAST SIGNALS"
echo "💎 No lag, no delays, pure speed"
echo "⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found"
    exit 1
fi

# Quick dependency check
echo "⚡ Checking dependencies..."
python3 -c "import telegram, cv2, numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Installing dependencies..."
    pip3 install python-telegram-bot opencv-python numpy --quiet
fi

echo "✅ Ready for instant signals"
echo ""

# Launch instant bot
echo "⚡ LAUNCHING INSTANT BOT..."
echo "📱 Send charts for ZERO DELAY signals"
echo "🚀 Press Ctrl+C to stop"
echo ""

python3 instant_god_tier_bot.py