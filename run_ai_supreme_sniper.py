#!/usr/bin/env python3
"""
AI Supreme Sniper Bot Launcher
==============================
Launch the 100 Billion Years AI Trading Engine with advanced setup.
"""

import os
import sys
import logging
from pathlib import Path

def setup_ai_environment():
    """Setup AI engine environment variables"""
    env_file = Path('.env')
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
        print("🤖 AI Environment variables loaded")
    else:
        print("⚠️  No .env file found. AI engine requires proper configuration.")

def check_ai_requirements():
    """Check AI engine requirements"""
    required_vars = ['EMAIL', 'PASSWORD', 'BOT_TOKEN', 'CHAT_ID']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing AI engine variables: {', '.join(missing_vars)}")
        return False
    
    print("✅ AI engine requirements satisfied")
    return True

def display_ai_banner():
    """Display AI engine startup banner"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║        🤖 SUPREME AI SNIPER — 100 BILLION YEARS ENGINE      ║
║                                                              ║
║  • Advanced Market Psychology Analysis                       ║
║  • Dynamic Strategy Tree Construction                        ║
║  • Adaptive Confidence Thresholds                           ║
║  • Secret AI Engines for Different Markets                  ║
║  • Next 1M Candle Predictions with Precise Timing          ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)

def main():
    """Main AI launcher function"""
    display_ai_banner()
    
    print("🚀 Initializing AI Supreme Sniper Engine...")
    print("=" * 60)
    
    # Setup environment
    setup_ai_environment()
    
    # Check requirements
    if not check_ai_requirements():
        print("\n💡 Setup Instructions:")
        print("1. Create .env file with your credentials")
        print("2. Add EMAIL, PASSWORD, BOT_TOKEN, CHAT_ID")
        print("3. Run this launcher again")
        sys.exit(1)
    
    print("\n🧠 AI Features Activated:")
    print("• ✅ Market Psychology Detection")
    print("• ✅ Dynamic Strategy Trees")
    print("• ✅ Momentum Shift Analysis")
    print("• ✅ Trap Zone Detection")
    print("• ✅ Support/Resistance Psychology")
    print("• ✅ Engulfing Pattern Recognition")
    print("• ✅ Breakout Continuation Logic")
    print("• ✅ Exhaustion Reversal Detection")
    print("• ✅ Next Candle Timing Precision")
    
    print("\n⚙️  AI Configuration:")
    print(f"• Confidence Threshold: 85%")
    print(f"• AI Lookback: 50 candles")
    print(f"• Data Window: 100 candles")
    print(f"• Signal Spacing: 15 seconds")
    print(f"• Timeframes: 1M, 2M")
    
    # Launch AI engine
    try:
        print("\n🎯 Launching AI Supreme Sniper...")
        print("🤖 Connecting to 100 Billion Years AI Engine...")
        
        from supreme_ai_sniper import SupremeAISniper
        ai_engine = SupremeAISniper()
        
        print("🌐 AI Engine connected to markets")
        print("📡 Real-time psychology analysis active")
        print("🧠 Dynamic strategy trees building...")
        print("\n🚀 AI Supreme Sniper is now LIVE!")
        print("📱 Monitor your Telegram for AI signals")
        print("\n⚠️  Press Ctrl+C to stop the AI engine")
        
        ai_engine.run_ai_engine()
        
    except KeyboardInterrupt:
        print("\n\n🛑 AI Engine stopped by user")
        print("🤖 100 Billion Years AI Engine shutting down...")
    except Exception as e:
        logging.error(f"❌ AI Engine crashed: {e}")
        print(f"❌ AI Engine error: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Check your internet connection")
        print("2. Verify Quotex credentials")
        print("3. Ensure Chrome browser is installed")
        print("4. Check Telegram bot token and chat ID")
        sys.exit(1)

if __name__ == "__main__":
    main()