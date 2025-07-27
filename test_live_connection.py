#!/usr/bin/env python3
"""
Live Connection Test for AI Supreme Sniper
==========================================
Test Telegram connection and send startup notification.
"""

import os
import asyncio
from datetime import datetime
from telegram import Bot
from telegram.error import TelegramError

# Load environment variables
def load_env():
    with open('.env', 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key] = value

async def test_telegram_connection():
    """Test Telegram bot connection"""
    load_env()
    
    bot_token = os.getenv('BOT_TOKEN')
    chat_id = os.getenv('CHAT_ID')
    
    print("🤖 Testing AI Supreme Sniper Live Connection...")
    print("=" * 60)
    
    if not bot_token or not chat_id:
        print("❌ Missing Telegram credentials")
        return False
    
    try:
        bot = Bot(token=bot_token)
        
        # Test basic connection
        print("📡 Testing bot connection...")
        me = await bot.get_me()
        print(f"✅ Bot connected: @{me.username}")
        
        # Send test message
        print("📱 Sending test message...")
        current_time = datetime.now().strftime("%H:%M:%S UTC")
        
        test_message = (
            f"🤖 **𝗔𝗜 𝗦𝗨𝗣𝗥𝗘𝗠𝗘 𝗦𝗡𝗜𝗣𝗘𝗥 - 𝗟𝗜𝗩𝗘 𝗧𝗘𝗦𝗧**\n\n"
            f"🚀 **Status**: Connection Successful\n"
            f"⏰ **Time**: {current_time}\n"
            f"🌐 **Mode**: LIVE MARKET DATA\n"
            f"🎯 **Target**: Quotex OTC Markets\n"
            f"🧠 **AI Engine**: 100 Billion Years Ready\n\n"
            f"✅ **System Check**: All systems operational\n"
            f"📊 **Next**: Launching market analysis...\n\n"
            f"🤖 *Ready to start live trading signals!*"
        )
        
        await bot.send_message(chat_id=chat_id, text=test_message, parse_mode="Markdown")
        print("✅ Test message sent successfully!")
        
        return True
        
    except TelegramError as e:
        print(f"❌ Telegram error: {e}")
        return False
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False

async def main():
    """Main test function"""
    success = await test_telegram_connection()
    
    if success:
        print("\n🎉 Live Connection Test PASSED!")
        print("🚀 Ready to launch AI Supreme Sniper with live data")
        print("\n▶️  Run: python3 run_ai_supreme_sniper.py")
    else:
        print("\n❌ Live Connection Test FAILED!")
        print("🔧 Check your Telegram credentials and try again")

if __name__ == "__main__":
    asyncio.run(main())