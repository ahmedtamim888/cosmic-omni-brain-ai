#!/usr/bin/env python3
"""
Quick Telegram Bot Status Checker
"""

import requests
import json
from config import Config

def check_bot_status():
    """Check if the Telegram bot is responding"""
    try:
        url = f"https://api.telegram.org/bot{Config.TELEGRAM_BOT_TOKEN}/getMe"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('ok'):
                result = data['result']
                print("✅ TELEGRAM BOT STATUS: ONLINE")
                print(f"🤖 Bot Name: {result['first_name']}")
                print(f"📛 Username: @{result['username']}")
                print(f"🆔 Bot ID: {result['id']}")
                print(f"👥 Can Join Groups: {result.get('can_join_groups', False)}")
                return True
            else:
                print("❌ Bot API Error:", data.get('description', 'Unknown error'))
                return False
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        return False

def send_test_message(chat_id):
    """Send a test message to check if bot can send messages"""
    try:
        url = f"https://api.telegram.org/bot{Config.TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {
            'chat_id': chat_id,
            'text': '🧠 **COSMIC AI BOT TEST**\n\n✅ Bot is working correctly!\n🚀 Ready to analyze charts!\n\n📸 Send me a chart screenshot to test!',
            'parse_mode': 'Markdown'
        }
        
        response = requests.post(url, data=data, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('ok'):
                print("✅ Test message sent successfully!")
                return True
            else:
                print("❌ Failed to send message:", result.get('description'))
                return False
        else:
            print(f"❌ HTTP Error sending message: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error sending test message: {e}")
        return False

if __name__ == "__main__":
    print("🧠 COSMIC AI Telegram Bot - Status Check")
    print("=" * 45)
    
    # Check bot status
    bot_ok = check_bot_status()
    
    if bot_ok:
        print("\n📱 BOT INFORMATION:")
        print("🔗 To use your bot:")
        print("   1. Open Telegram")
        print("   2. Search: @Itis_ghost_bot")
        print("   3. Send: /start")
        print("   4. Upload chart screenshot")
        print("   5. Get instant AI analysis!")
        
        print("\n🚀 FEATURES:")
        print("   • Advanced pattern recognition")
        print("   • Market psychology analysis") 
        print("   • Professional signal formatting")
        print("   • 85%+ confidence threshold")
        print("   • Real-time analysis (5-10 seconds)")
        
        # Optional: Send test message (uncomment and add chat ID)
        # test_chat_id = "YOUR_CHAT_ID_HERE"  # Replace with actual chat ID
        # print(f"\n📤 Sending test message to {test_chat_id}...")
        # send_test_message(test_chat_id)
        
    else:
        print("\n❌ Bot is not responding correctly.")
        print("🔧 Troubleshooting:")
        print("   1. Check internet connection")
        print("   2. Verify bot token in config.py")
        print("   3. Ensure bot process is running:")
        print("      ps aux | grep telegram")
    
    print("\n" + "=" * 45)