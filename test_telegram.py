#!/usr/bin/env python3
"""
Test script for COSMIC AI Telegram Bot
"""

import sys
import time
from datetime import datetime
from telegram_bot_pro import start_telegram_bot, send_signal_to_telegram, get_bot_stats

def test_bot():
    """Test the Telegram bot functionality"""
    print("🧠 COSMIC AI Telegram Bot Test")
    print("=" * 50)
    
    # Start the bot
    print("1. Starting Telegram Bot...")
    success = start_telegram_bot()
    if success:
        print("✅ Bot started successfully!")
    else:
        print("❌ Failed to start bot")
        return False
    
    # Wait for bot to initialize
    print("2. Waiting for bot initialization...")
    time.sleep(3)
    
    # Check bot status
    print("3. Checking bot status...")
    stats = get_bot_stats()
    print(f"   Bot Running: {stats['is_running']}")
    print(f"   Subscribers: {stats['subscribers']}")
    
    # Test sending a signal
    print("4. Testing signal sending...")
    test_signal = {
        'signal': 'CALL',
        'confidence': 92.4,
        'reasoning': 'Breakout + Momentum Surge',
        'strategy': 'BREAKOUT_CONTINUATION',
        'market_psychology': 'STRONG_BULLISH_SENTIMENT',
        'timeframe': '1M'
    }
    
    result = send_signal_to_telegram(test_signal)
    if result:
        print("✅ Test signal sent successfully!")
    else:
        print("⚠️ No subscribers found or signal failed")
    
    print("\n🚀 Bot is ready! Use these commands:")
    print("   /start - Initialize bot")
    print("   /subscribe - Get signals")
    print("   /status - Check status")
    print("   /help - Get help")
    
    return True

if __name__ == "__main__":
    try:
        test_bot()
        print("\n✅ Test completed. Bot is running in background.")
        print("🔗 Bot Token: 8155744845:AAGQ6s8RMSpCAjO16Mzb1xl8_otLnXh3OKQ")
        print("📱 Search @YourBotUsername on Telegram to start using!")
        
        # Keep alive for testing
        print("\nPress Ctrl+C to stop...")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping test...")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)