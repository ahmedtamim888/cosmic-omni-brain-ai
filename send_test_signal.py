#!/usr/bin/env python3
"""
Send a test signal to verify Telegram bot functionality
"""

import time
from telegram_bot_pro import send_signal_to_telegram, get_bot_stats

def send_test_signal():
    """Send a test trading signal"""
    print("🧠 Sending Test COSMIC AI Signal...")
    
    # Check bot status first
    stats = get_bot_stats()
    print(f"Bot Status: {'Running' if stats['is_running'] else 'Stopped'}")
    print(f"Subscribers: {stats['subscribers']}")
    
    if not stats['is_running']:
        print("❌ Bot is not running. Please start the bot first.")
        return False
    
    if stats['subscribers'] == 0:
        print("⚠️ No subscribers found. Please subscribe to the bot first using /start and /subscribe")
        return False
    
    # Create test signal
    test_signal = {
        'signal': 'CALL',
        'confidence': 94.2,
        'reasoning': 'Strong bullish breakout with high momentum',
        'strategy': 'BREAKOUT_CONTINUATION',
        'market_psychology': 'BULLISH_MOMENTUM',
        'timeframe': '1M'
    }
    
    # Send the signal
    result = send_signal_to_telegram(test_signal)
    
    if result:
        print("✅ Test signal sent successfully!")
        print("📱 Check your Telegram for the signal message")
    else:
        print("❌ Failed to send test signal")
    
    return result

if __name__ == "__main__":
    send_test_signal()