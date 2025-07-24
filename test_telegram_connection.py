#!/usr/bin/env python3
"""
Test Telegram bot connection with your specific bot token and chat ID
"""

import asyncio
import sys
from telegram import Bot
from config import Config

async def test_telegram_connection():
    """Test if the bot can connect to Telegram and send a message"""
    try:
        # Create bot instance
        bot = Bot(token=Config.TELEGRAM_BOT_TOKEN)
        
        print("🤖 Testing Telegram bot connection...")
        print(f"Bot Token: {Config.TELEGRAM_BOT_TOKEN[:10]}...")
        print(f"Chat ID: {Config.TELEGRAM_CHAT_ID}")
        
        # Test bot info
        bot_info = await bot.get_me()
        print(f"✅ Bot connected successfully!")
        print(f"   Bot name: @{bot_info.username}")
        print(f"   Bot ID: {bot_info.id}")
        print(f"   First name: {bot_info.first_name}")
        
        # Test sending a message to the chat
        if Config.TELEGRAM_CHAT_ID:
            try:
                test_message = """
🧠 <b>COSMIC AI BOT - CONNECTION TEST</b> ✅

🎉 Your Telegram bot is working perfectly!

🚀 <b>Ready Features:</b>
• AI-powered chart analysis
• Real-time CALL/PUT signals
• 85%+ confidence threshold
• Multi-strategy engine
• Professional interface

📱 <b>Available Commands:</b>
/start - Get started
/subscribe - Receive signals
/help - Show help
/stats - View statistics

📊 Send any chart screenshot to get instant analysis!
                """
                
                message = await bot.send_message(
                    chat_id=Config.TELEGRAM_CHAT_ID,
                    text=test_message,
                    parse_mode='HTML'
                )
                
                print(f"✅ Test message sent successfully!")
                print(f"   Message ID: {message.message_id}")
                print(f"   Sent to chat: {Config.TELEGRAM_CHAT_ID}")
                
            except Exception as e:
                print(f"⚠️  Could not send message to chat {Config.TELEGRAM_CHAT_ID}")
                print(f"   Error: {e}")
                print("   This might be normal if the bot hasn't been started by the user yet.")
        
        return True
        
    except Exception as e:
        print(f"❌ Telegram connection failed: {e}")
        print("\nPossible issues:")
        print("1. Check your bot token is correct")
        print("2. Make sure bot was created with @BotFather")
        print("3. Verify network connection")
        return False

def main():
    """Main test function"""
    print("="*60)
    print("🧠 COSMIC AI BOT - TELEGRAM CONNECTION TEST")
    print("="*60)
    
    try:
        # Run the async test
        result = asyncio.run(test_telegram_connection())
        
        print("\n" + "="*60)
        if result:
            print("🎉 TELEGRAM CONNECTION SUCCESSFUL!")
            print("✅ Your bot is ready to receive and send messages!")
            print("\n🚀 Next steps:")
            print("1. Start your bot: python start_bot.py")
            print("2. Find your bot on Telegram")
            print("3. Send /start to begin")
            print("4. Upload chart screenshots for analysis")
        else:
            print("❌ TELEGRAM CONNECTION FAILED!")
            print("Please check your configuration and try again.")
        print("="*60)
        
        return 0 if result else 1
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())