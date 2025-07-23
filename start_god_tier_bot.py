#!/usr/bin/env python3
"""
Startup script for the Ultimate God-Tier Trading Bot
"""

import sys
import signal
import asyncio
import logging

def signal_handler(sig, frame):
    print('\n👋 God-Tier Bot stopped by user')
    sys.exit(0)

async def main():
    """Main startup function"""
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("🔥" * 60)
    print("🚀 STARTING ULTIMATE GOD-TIER TRADING BOT")
    print("⚡ Beyond human comprehension")
    print("🧬 100 billion year strategy evolution")
    print("💎 No repaint, no lag, no mercy")
    print("🔥" * 60)
    print()
    
    try:
        # Import and start the bot
        from god_tier_trading_bot import UltimateGodTierTradingBot
        
        bot = UltimateGodTierTradingBot()
        await bot.start_bot()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed:")
        print("pip3 install --break-system-packages opencv-python pandas numpy python-telegram-bot pillow")
    except Exception as e:
        print(f"❌ Bot error: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 God-Tier Bot stopped")
    except Exception as e:
        print(f"❌ Startup error: {e}")
        sys.exit(1)