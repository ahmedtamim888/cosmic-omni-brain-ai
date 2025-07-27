#!/usr/bin/env python3
"""
Supreme Sniper Bot Launcher
===========================
This script handles environment setup and launches the Supreme Sniper Bot
with proper error handling and logging.
"""

import os
import sys
import logging
from pathlib import Path

def setup_environment():
    """Load environment variables from .env file if it exists"""
    env_file = Path('.env')
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
        print("✅ Environment variables loaded from .env file")
    else:
        print("⚠️  No .env file found. Make sure to set environment variables manually.")

def check_requirements():
    """Check if all required environment variables are set"""
    required_vars = ['EMAIL', 'PASSWORD', 'BOT_TOKEN', 'CHAT_ID']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set these variables in your .env file or environment.")
        return False
    
    print("✅ All required environment variables are set")
    return True

def main():
    """Main launcher function"""
    print("🚀 Supreme Sniper Bot Launcher")
    print("=" * 50)
    
    # Setup environment
    setup_environment()
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Import and run the bot
    try:
        from supreme_sniper_bot import SupremeSniperBot
        print("🎯 Starting Supreme Sniper Bot...")
        bot = SupremeSniperBot()
        bot.run()
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user")
    except Exception as e:
        logging.error(f"❌ Bot crashed: {e}")
        print(f"❌ Bot crashed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()