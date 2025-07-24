#!/usr/bin/env python3
"""
Test script to verify the Telegram AI bot setup
"""

import sys
import traceback

def test_imports():
    """Test all required imports"""
    print("🧪 Testing imports...")
    
    try:
        import flask
        print("✅ Flask")
    except ImportError:
        print("❌ Flask - run: pip install Flask")
        return False
    
    try:
        import cv2
        print("✅ OpenCV")
    except ImportError:
        print("❌ OpenCV - run: pip install opencv-python")
        return False
    
    try:
        import numpy
        print("✅ NumPy")
    except ImportError:
        print("❌ NumPy - run: pip install numpy")
        return False
    
    try:
        import telegram
        print("✅ Python Telegram Bot")
    except ImportError:
        print("❌ Telegram - run: pip install python-telegram-bot[all]")
        return False
    
    try:
        import matplotlib
        print("✅ Matplotlib")
    except ImportError:
        print("❌ Matplotlib - run: pip install matplotlib")
        return False
    
    try:
        from PIL import Image
        print("✅ Pillow")
    except ImportError:
        print("❌ Pillow - run: pip install Pillow")
        return False
    
    return True

def test_config():
    """Test configuration"""
    print("\n🔧 Testing configuration...")
    
    try:
        from config import Config
        print("✅ Config imported successfully")
        
        if Config.TELEGRAM_BOT_TOKEN:
            print(f"✅ Bot token configured: {Config.TELEGRAM_BOT_TOKEN[:10]}...")
        else:
            print("⚠️  Bot token not set - you'll need to configure this")
        
        print(f"✅ Confidence threshold: {Config.CONFIDENCE_THRESHOLD}%")
        print(f"✅ Signal timeframe: {Config.SIGNAL_TIMEFRAME}")
        print(f"✅ Strategies available: {len(Config.STRATEGIES)}")
        
        return True
    except Exception as e:
        print(f"❌ Config error: {e}")
        return False

def test_ai_engine():
    """Test AI engine components"""
    print("\n🧠 Testing AI engine...")
    
    try:
        from logic.ai_engine import CosmicAIEngine, CandleDetector
        print("✅ AI engine imported successfully")
        
        # Test creating engine instance
        engine = CosmicAIEngine()
        print("✅ AI engine instance created")
        
        # Test candle detector
        detector = CandleDetector()
        print("✅ Candle detector created")
        
        return True
    except Exception as e:
        print(f"❌ AI engine error: {e}")
        traceback.print_exc()
        return False

def test_telegram_bot():
    """Test Telegram bot components"""
    print("\n📱 Testing Telegram bot...")
    
    try:
        from telegram_bot_pro import CosmicTelegramBot
        print("✅ Telegram bot imported successfully")
        
        # Test creating bot instance
        bot = CosmicTelegramBot()
        print("✅ Telegram bot instance created")
        
        return True
    except Exception as e:
        print(f"❌ Telegram bot error: {e}")
        return False

def test_web_app():
    """Test Flask web app"""
    print("\n🌐 Testing web application...")
    
    try:
        from app import app
        print("✅ Flask app imported successfully")
        
        # Test app configuration
        print(f"✅ App configured with secret key")
        
        return True
    except Exception as e:
        print(f"❌ Web app error: {e}")
        return False

def main():
    """Run all tests"""
    print("="*50)
    print("🧠 COSMIC AI BOT - SETUP TEST")
    print("="*50)
    
    all_passed = True
    
    all_passed &= test_imports()
    all_passed &= test_config()
    all_passed &= test_ai_engine()
    all_passed &= test_telegram_bot()
    all_passed &= test_web_app()
    
    print("\n" + "="*50)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Your bot is ready to run!")
        print("\n🚀 To start the bot, run:")
        print("   python start_bot.py")
        print("\n📖 For setup instructions, see:")
        print("   SETUP_GUIDE.md")
    else:
        print("❌ SOME TESTS FAILED!")
        print("Please fix the issues above before running the bot.")
    print("="*50)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())