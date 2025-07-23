#!/usr/bin/env python3
"""
Quick test script for Telegram bot functionality
"""

import asyncio
import os
import sys
from datetime import datetime

def test_imports():
    """Test if all required imports work"""
    print("🧪 Testing imports...")
    
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✅ Pandas imported successfully")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    try:
        from communication.telegram_bot import TelegramBot
        print("✅ TelegramBot imported successfully")
    except ImportError as e:
        print(f"❌ TelegramBot import failed: {e}")
        return False
    
    try:
        from utils.config import Config
        print("✅ Config imported successfully")
    except ImportError as e:
        print(f"❌ Config import failed: {e}")
        return False
    
    return True

async def test_telegram_bot():
    """Test Telegram bot functionality"""
    print("\n📱 Testing Telegram bot...")
    
    try:
        from communication.telegram_bot import TelegramBot
        from utils.config import Config
        
        # Create config
        config = Config()
        
        # Check configuration
        config_status = config.validate_config()
        print(f"Configuration valid: {config_status['valid']}")
        print(f"Telegram configured: {config_status['telegram_configured']}")
        print(f"Chat IDs count: {config_status['chat_ids_count']}")
        
        if config_status['issues']:
            print("Configuration issues:")
            for issue in config_status['issues']:
                print(f"  • {issue}")
        
        # Create bot
        bot = TelegramBot(config.TELEGRAM_TOKEN)
        
        # Test connection
        connection_ok = await bot.test_connection()
        print(f"Connection test: {'✅ OK' if connection_ok else '❌ Failed'}")
        
        # Create test signal
        test_signal = {
            'action': 'CALL',
            'confidence': 0.95,
            'strategy': 'TEST_STRATEGY',
            'reason': 'This is a test signal to verify Telegram bot functionality',
            'timestamp': datetime.now(),
            'next_candle_prediction': {
                'direction': 'UP',
                'strength': 'STRONG'
            },
            'volume_condition': 'rising',
            'trend_alignment': 'bullish_trend'
        }
        
        # Send test signal
        print("\n📤 Sending test signal...")
        await bot.send_signal(test_signal)
        print("✅ Test signal sent (check logs or Telegram)")
        
        return True
        
    except Exception as e:
        print(f"❌ Telegram bot test failed: {e}")
        return False

async def test_main_bot():
    """Test main bot initialization"""
    print("\n🤖 Testing main bot...")
    
    try:
        from main import UltraAccurateTradingBot
        
        # Create bot instance
        bot = UltraAccurateTradingBot()
        print("✅ Main bot created successfully")
        
        # Test configuration
        config_status = bot.config.validate_config()
        if not config_status['valid']:
            print("⚠️  Configuration issues found:")
            for issue in config_status['issues']:
                print(f"   • {issue}")
        else:
            print("✅ Configuration is valid")
        
        return True
        
    except Exception as e:
        print(f"❌ Main bot test failed: {e}")
        return False

def print_environment_info():
    """Print environment information"""
    print("\n🔧 Environment Information:")
    print(f"Python version: {sys.version}")
    print(f"Current directory: {os.getcwd()}")
    
    # Check environment variables
    telegram_token = os.getenv('TELEGRAM_BOT_TOKEN', 'Not set')
    telegram_chats = os.getenv('TELEGRAM_CHAT_IDS', 'Not set')
    
    print(f"TELEGRAM_BOT_TOKEN: {'Set' if telegram_token != 'Not set' else 'Not set'}")
    print(f"TELEGRAM_CHAT_IDS: {'Set' if telegram_chats != 'Not set' else 'Not set'}")
    
    # Check .env file
    env_exists = os.path.exists('.env')
    print(f".env file exists: {'Yes' if env_exists else 'No'}")

async def main():
    """Main test function"""
    print("=" * 60)
    print("🚀 ULTRA-ACCURATE TRADING BOT - QUICK TEST")
    print("=" * 60)
    
    # Print environment info
    print_environment_info()
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed! Run: pip install -r requirements.txt")
        return
    
    # Test Telegram bot
    telegram_ok = await test_telegram_bot()
    
    # Test main bot
    main_bot_ok = await test_main_bot()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY:")
    print(f"Imports: {'✅ OK' if True else '❌ Failed'}")
    print(f"Telegram Bot: {'✅ OK' if telegram_ok else '❌ Failed'}")
    print(f"Main Bot: {'✅ OK' if main_bot_ok else '❌ Failed'}")
    
    if telegram_ok and main_bot_ok:
        print("\n🎉 All tests passed! Your bot is ready to trade.")
        print("\nNext steps:")
        print("1. Configure Telegram: python setup_telegram.py")
        print("2. Run the bot: python main.py")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
        print("\nTroubleshooting:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Configure Telegram: python setup_telegram.py")
        print("3. Check logs for detailed error messages")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Test cancelled by user.")
    except Exception as e:
        print(f"\n❌ Test error: {e}")
        sys.exit(1)