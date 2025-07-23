#!/usr/bin/env python3
"""
Quick Telegram bot test with the provided token
"""

import asyncio
import os
from datetime import datetime

# Set the environment variable
os.environ['TELEGRAM_BOT_TOKEN'] = '8155744845:AAGQ6s8RMSpCAjO16Mzb1xl8_otLnXh3OKQ'

async def test_bot():
    print("🚀 Testing Telegram Bot Connection...")
    print("=" * 50)
    
    try:
        from telegram import Bot
        
        # Create bot instance
        bot = Bot(token='8155744845:AAGQ6s8RMSpCAjO16Mzb1xl8_otLnXh3OKQ')
        
        # Test connection
        print("🔄 Testing bot connection...")
        bot_info = await bot.get_me()
        
        print(f"✅ Bot connected successfully!")
        print(f"📱 Bot name: {bot_info.first_name}")
        print(f"🤖 Bot username: @{bot_info.username}")
        print(f"🆔 Bot ID: {bot_info.id}")
        
        print("\n📋 Next Steps:")
        print("1. Start a chat with your bot on Telegram")
        print(f"2. Search for: @{bot_info.username}")
        print("3. Send /start to your bot")
        print("4. Send any message to get your chat ID")
        print("5. Visit this URL to get your chat ID:")
        print(f"   https://api.telegram.org/bot8155744845:AAGQ6s8RMSpCAjO16Mzb1xl8_otLnXh3OKQ/getUpdates")
        
        print("\n🎉 Your Telegram bot is ready to receive trading signals!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing bot: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_bot())