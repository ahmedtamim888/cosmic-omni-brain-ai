#!/usr/bin/env python3

"""
Real Quotex AI Supreme Sniper Launcher
======================================
🚀 Cloudflare Bypass Technology
🧠 100 Billion Years AI Strategy Engine
⚡ Live Market Data Access
"""

import sys
import os
import asyncio
import logging

def print_banner():
    """Print the AI Supreme Sniper banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║    🤖 REAL QUOTEX AI SUPREME SNIPER BOT 🤖                                 ║
║                                                                              ║
║    ⚡ Live Quotex Platform Access                                            ║
║    🧠 100 Billion Years AI Strategy Engine                                   ║
║    💎 Advanced Cloudflare Bypass Technology                                  ║
║    🎯 Real-time Market Psychology Analysis                                    ║
║    📊 Dynamic Strategy Tree Construction                                      ║
║    🚀 Next-candle Prediction System                                          ║
║                                                                              ║
║    Status: LOADING REAL MARKET CONNECTION...                                ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

🔧 INITIALIZATION SEQUENCE:
"""
    print(banner)

def check_environment():
    """Check environment and dependencies"""
    print("🔍 Checking environment variables...")
    
    required_vars = ['EMAIL', 'PASSWORD', 'BOT_TOKEN', 'CHAT_ID']
    
    try:
        with open('.env', 'r') as f:
            env_content = f.read()
            
        missing_vars = []
        for var in required_vars:
            if f"{var}=" not in env_content:
                missing_vars.append(var)
                
        if missing_vars:
            print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
            return False
        else:
            print("✅ Environment variables loaded")
            return True
            
    except FileNotFoundError:
        print("❌ .env file not found")
        return False

def check_dependencies():
    """Check required dependencies"""
    print("📦 Checking dependencies...")
    
    required_packages = [
        'pandas', 'numpy', 'telegram', 'selenium', 
        'cloudscraper', 'fake_useragent'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    # Special check for undetected_chromedriver
    try:
        import undetected_chromedriver
    except ImportError:
        missing_packages.append('undetected-chromedriver')
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("💡 Install with: pip install --break-system-packages " + " ".join(missing_packages))
        return False
    else:
        print("✅ All dependencies available")
        return True

async def main():
    """Main launcher function"""
    print_banner()
    
    # Check environment
    if not check_environment():
        print("\n❌ Environment check failed. Please configure .env file.")
        return
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Dependency check failed. Please install required packages.")
        return
    
    print("🚀 Starting Real Quotex AI Supreme Sniper...")
    print("🌐 Attempting Cloudflare bypass...")
    print("📡 Connecting to live Quotex platform...\n")
    
    try:
        # Import and run the main bot
        from real_quotex_ai_sniper import main as run_sniper
        await run_sniper()
        
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logging.error(f"Launcher error: {e}")

if __name__ == "__main__":
    asyncio.run(main())