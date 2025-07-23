#!/usr/bin/env python3
"""
Telegram Bot Setup Script for Ultra-Accurate Trading Bot
"""

import os
import sys
import asyncio
from typing import Optional

def print_banner():
    """Print setup banner"""
    print("=" * 60)
    print("🚀 ULTRA-ACCURATE TRADING BOT - TELEGRAM SETUP")
    print("=" * 60)
    print()

def print_instructions():
    """Print setup instructions"""
    print("📋 SETUP INSTRUCTIONS:")
    print()
    print("1. Create a Telegram Bot:")
    print("   • Open Telegram and search for @BotFather")
    print("   • Send /newbot command")
    print("   • Choose a name for your bot (e.g., 'My Trading Bot')")
    print("   • Choose a username (must end with 'bot', e.g., 'mytradingbot')")
    print("   • Copy the bot token (looks like: 123456789:ABCdefGHIjklMNOpqrsTUVwxyz)")
    print()
    print("2. Get your Chat ID:")
    print("   • Send a message to your bot")
    print("   • Visit: https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates")
    print("   • Look for 'chat':{'id': YOUR_CHAT_ID}")
    print()
    print("3. Set Environment Variables:")
    print("   • TELEGRAM_BOT_TOKEN=your_bot_token_here")
    print("   • TELEGRAM_CHAT_IDS=your_chat_id_here (comma-separated for multiple)")
    print()

def get_user_input(prompt: str, default: str = None) -> str:
    """Get user input with optional default"""
    if default:
        user_input = input(f"{prompt} [{default}]: ").strip()
        return user_input if user_input else default
    else:
        return input(f"{prompt}: ").strip()

def validate_token(token: str) -> bool:
    """Basic token validation"""
    if not token:
        return False
    parts = token.split(':')
    return len(parts) == 2 and parts[0].isdigit() and len(parts[1]) > 20

def validate_chat_id(chat_id: str) -> bool:
    """Basic chat ID validation"""
    try:
        int(chat_id)
        return True
    except ValueError:
        return False

async def test_telegram_connection(token: str, chat_ids: list):
    """Test Telegram bot connection"""
    try:
        # Import here to handle missing dependencies
        from communication.telegram_bot import TelegramBot
        
        print("\n🔄 Testing Telegram connection...")
        
        # Create bot instance
        bot = TelegramBot(token)
        bot.chat_ids = chat_ids
        
        # Test connection
        if await bot.test_connection():
            print("✅ Telegram bot connection successful!")
            
            # Send test message
            test_signal = {
                'action': 'NO_TRADE',
                'confidence': 0.85,
                'strategy': 'SETUP_TEST',
                'reason': 'Testing Telegram bot setup - connection successful!',
                'timestamp': __import__('datetime').datetime.now()
            }
            
            await bot.send_signal(test_signal)
            print("✅ Test signal sent successfully!")
            return True
        else:
            print("❌ Telegram bot connection failed!")
            return False
            
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Install required packages: pip install python-telegram-bot")
        return False
    except Exception as e:
        print(f"❌ Error testing connection: {e}")
        return False

def create_env_file(token: str, chat_ids: str):
    """Create .env file with configuration"""
    try:
        with open('.env', 'w') as f:
            f.write(f"# Ultra-Accurate Trading Bot Configuration\n")
            f.write(f"TELEGRAM_BOT_TOKEN={token}\n")
            f.write(f"TELEGRAM_CHAT_IDS={chat_ids}\n")
        print("✅ .env file created successfully!")
        return True
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")
        return False

def create_shell_export_commands(token: str, chat_ids: str):
    """Create shell export commands"""
    print("\n📝 SHELL EXPORT COMMANDS:")
    print("Copy and paste these commands in your terminal:")
    print()
    print(f"export TELEGRAM_BOT_TOKEN='{token}'")
    print(f"export TELEGRAM_CHAT_IDS='{chat_ids}'")
    print()

async def main():
    """Main setup function"""
    print_banner()
    print_instructions()
    
    print("🔧 INTERACTIVE SETUP:")
    print()
    
    # Get bot token
    while True:
        token = get_user_input("Enter your Telegram bot token")
        if validate_token(token):
            break
        else:
            print("❌ Invalid token format. Please try again.")
    
    # Get chat IDs
    chat_ids_input = []
    while True:
        chat_id = get_user_input("Enter a chat ID (or press Enter to finish)")
        if not chat_id:
            if chat_ids_input:
                break
            else:
                print("❌ You must enter at least one chat ID.")
                continue
        
        if validate_chat_id(chat_id):
            chat_ids_input.append(chat_id)
            print(f"✅ Added chat ID: {chat_id}")
        else:
            print("❌ Invalid chat ID format. Please try again.")
    
    chat_ids_str = ','.join(chat_ids_input)
    chat_ids_list = [int(cid) for cid in chat_ids_input]
    
    print(f"\n📋 CONFIGURATION SUMMARY:")
    print(f"Bot Token: {token[:20]}...")
    print(f"Chat IDs: {chat_ids_str}")
    print()
    
    # Test connection
    connection_ok = await test_telegram_connection(token, chat_ids_list)
    
    if connection_ok:
        # Create configuration files
        create_env_file(token, chat_ids_str)
        create_shell_export_commands(token, chat_ids_str)
        
        print("\n🎉 SETUP COMPLETE!")
        print("Your Telegram bot is ready to send trading signals!")
        print()
        print("Next steps:")
        print("1. Run the trading bot: python main.py")
        print("2. Check the logs for trading signals")
        print("3. Receive signals in your Telegram chat")
        
    else:
        print("\n⚠️  SETUP INCOMPLETE")
        print("Please check your token and chat IDs, then try again.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Setup cancelled by user.")
    except Exception as e:
        print(f"\n❌ Setup error: {e}")
        sys.exit(1)