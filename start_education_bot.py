#!/usr/bin/env python3
"""
Startup script for Educational Telegram Bot
Handles environment setup and graceful bot startup
"""

import os
import sys
import subprocess
import signal
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'telegram',
        'requests',
        'sqlite3'  # This is built-in, but we'll check anyway
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'sqlite3':
                import sqlite3
            elif package == 'telegram':
                import telegram
            elif package == 'requests':
                import requests
        except ImportError:
            if package != 'sqlite3':  # sqlite3 is built-in
                missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
    else:
        print("✅ All dependencies are installed!")

def check_configuration():
    """Check bot configuration"""
    print("🔧 Checking configuration...")
    
    try:
        from config import Config
        
        if not Config.TELEGRAM_BOT_TOKEN or Config.TELEGRAM_BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
            print("❌ Bot token not configured!")
            print("Please update config.py with your bot token from @BotFather")
            print("\nSteps to get a bot token:")
            print("1. Message @BotFather on Telegram")
            print("2. Use /newbot command")
            print("3. Follow the instructions")
            print("4. Copy the token to config.py")
            return False
        
        print("✅ Bot token is configured!")
        
        # Check if chat ID is configured (optional)
        if Config.TELEGRAM_CHAT_ID:
            print("✅ Chat ID is configured!")
        else:
            print("ℹ️  Chat ID not set (this is optional)")
        
        return True
        
    except ImportError as e:
        print(f"❌ Configuration error: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    
    directories = ['uploads', 'logs']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    print("✅ Directories created!")

def run_tests():
    """Run basic tests before starting the bot"""
    print("🧪 Running basic tests...")
    
    try:
        # Import the test module
        from test_education_bot import test_bot_functionality
        
        # Run basic functionality test
        if test_bot_functionality():
            print("✅ Basic tests passed!")
            return True
        else:
            print("❌ Basic tests failed!")
            return False
            
    except ImportError:
        print("⚠️  Test module not found, skipping tests...")
        return True
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def start_bot():
    """Start the educational bot"""
    print("🚀 Starting Educational Telegram Bot...")
    print("Press Ctrl+C to stop the bot\n")
    
    try:
        from telegram_education_bot import EducationalBot
        import asyncio
        
        # Create and run the bot
        bot = EducationalBot()
        asyncio.run(bot.run())
        
    except KeyboardInterrupt:
        print("\n👋 Bot stopped by user")
    except Exception as e:
        print(f"\n❌ Bot error: {e}")
        print("\nTroubleshooting tips:")
        print("- Check your internet connection")
        print("- Verify bot token is correct")
        print("- Check if bot is blocked by Telegram")
        print("- Review logs for detailed error information")

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    print("\n🛑 Received shutdown signal, stopping bot...")
    sys.exit(0)

def main():
    """Main startup function"""
    print("🎓 Educational Telegram Bot - Startup Script")
    print("=" * 50)
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Step 1: Check dependencies
    try:
        check_dependencies()
    except Exception as e:
        print(f"❌ Dependency check failed: {e}")
        return
    
    # Step 2: Check configuration
    if not check_configuration():
        return
    
    # Step 3: Create directories
    try:
        create_directories()
    except Exception as e:
        print(f"❌ Directory creation failed: {e}")
        return
    
    # Step 4: Run tests (optional)
    if not run_tests():
        response = input("\n⚠️  Tests failed. Continue anyway? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("Startup cancelled.")
            return
    
    # Step 5: Start the bot
    print("\n" + "=" * 50)
    start_bot()

if __name__ == "__main__":
    main()