#!/usr/bin/env python3
"""
🧠 COSMIC AI Binary Trading Signal Bot - Startup Script

This script starts the complete Telegram bot with AI-powered binary trading signals.
It includes both the web interface and the Telegram bot functionality.

Features:
- Professional Telegram bot with interactive commands
- AI-powered chart analysis 
- Real-time binary trading signals (CALL/PUT)
- 85%+ confidence threshold
- Multiple trading strategies
- Beautiful web interface for chart uploads
"""

import os
import sys
import logging
import asyncio
import threading
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_welcome():
    """Print welcome message and bot information"""
    print("\n" + "="*60)
    print("🧠 COSMIC AI BINARY TRADING SIGNAL BOT 🚀")
    print("="*60)
    print("📊 AI-Powered Chart Analysis")
    print("🎯 Real-time CALL/PUT Signals") 
    print("📱 Professional Telegram Interface")
    print("🌐 Modern Web Dashboard")
    print("="*60)
    print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

def check_dependencies():
    """Check if all required dependencies are installed"""
    try:
        import flask
        import cv2
        import numpy
        import telegram
        import requests
        import matplotlib
        from PIL import Image
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Run: pip install -r requirements.txt")
        return False

def check_configuration():
    """Check bot configuration"""
    try:
        from config import Config
        
        print("\n📋 Configuration Check:")
        print(f"🤖 Bot Token: {'✅ Set' if Config.TELEGRAM_BOT_TOKEN else '❌ Missing'}")
        print(f"💬 Chat ID: {'✅ Set' if Config.TELEGRAM_CHAT_ID else '⚠️  Optional (can be set later)'}")
        print(f"🎯 Confidence Threshold: {Config.CONFIDENCE_THRESHOLD}%")
        print(f"📊 Signal Timeframe: {Config.SIGNAL_TIMEFRAME}")
        print(f"🧠 Strategies Available: {len(Config.STRATEGIES)}")
        
        if not Config.TELEGRAM_BOT_TOKEN:
            print("\n❌ ERROR: TELEGRAM_BOT_TOKEN is required!")
            print("Please set your bot token in config.py or environment variable")
            return False
            
        return True
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def start_web_interface():
    """Start the Flask web interface in a separate thread"""
    try:
        from app import app
        print("🌐 Starting web interface on http://localhost:5000")
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    except Exception as e:
        logger.error(f"Web interface error: {e}")

def start_telegram_bot():
    """Start the Telegram bot"""
    try:
        from telegram_bot_pro import start_telegram_bot as start_bot
        
        print("🤖 Starting Telegram bot...")
        
        # Start the bot using the existing function
        success = start_bot()
        
        if success:
            print("✅ Telegram bot started successfully!")
            
            # Keep the main thread alive
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 Stopping Telegram bot...")
        else:
            print("❌ Failed to start Telegram bot")
            
    except Exception as e:
        logger.error(f"Telegram bot error: {e}")
        print(f"❌ Failed to start Telegram bot: {e}")

def show_usage_instructions():
    """Show usage instructions for the bot"""
    print("\n📖 HOW TO USE THE BOT:")
    print("-" * 40)
    print("1. 🌐 WEB INTERFACE:")
    print("   - Open http://localhost:5000 in your browser")
    print("   - Upload chart screenshots for analysis")
    print("   - Get instant CALL/PUT signals")
    print("")
    print("2. 📱 TELEGRAM BOT:")
    print("   - Find your bot on Telegram")
    print("   - Send /start to begin")
    print("   - Upload chart images directly")
    print("   - Use /subscribe for signal alerts")
    print("")
    print("3. 🎯 SUPPORTED BROKERS:")
    print("   - Quotex, Binomo, Pocket Option")
    print("   - IQ Option, Olymp Trade, ExpertOption")
    print("   - Any broker with candlestick charts")
    print("")
    print("4. 📊 SIGNAL TYPES:")
    print("   - CALL 📈 (price will go up)")
    print("   - PUT 📉 (price will go down)")
    print("   - NO_TRADE ⏸️ (market uncertainty)")

def main():
    """Main function to start the bot"""
    print_welcome()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check configuration
    if not check_configuration():
        sys.exit(1)
    
    print("\n🚀 Starting COSMIC AI Bot...")
    
    try:
        # Start web interface in a separate thread
        web_thread = threading.Thread(target=start_web_interface, daemon=True)
        web_thread.start()
        
        # Give web server time to start
        time.sleep(2)
        
        show_usage_instructions()
        
        print("\n✅ Bot is now running!")
        print("Press Ctrl+C to stop the bot")
        
        # Start Telegram bot (this will block)
        start_telegram_bot()
        
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down COSMIC AI Bot...")
        print("Thanks for using our service!")
    except Exception as e:
        logger.error(f"Startup error: {e}")
        print(f"❌ Failed to start bot: {e}")

if __name__ == "__main__":
    main()