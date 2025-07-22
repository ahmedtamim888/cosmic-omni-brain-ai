#!/usr/bin/env python3
"""
COSMIC AI Binary Signal Bot - Telegram Runner
Simple script to start the Telegram bot for direct image analysis
"""

import sys
import os
import logging

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from telegram_handler import main

if __name__ == "__main__":
    print("🧠 COSMIC AI Binary Signal Bot - Telegram Mode")
    print("=" * 50)
    print("📱 Starting Telegram bot for direct image analysis...")
    print("📸 Users can send chart screenshots and get instant signals!")
    print("🚀 Bot Token: 8155744845:AAGQ6s8RMSpCAjO16Mzb1xl8_otLnXh3OKQ")
    print("=" * 50)
    print()
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Telegram bot stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)