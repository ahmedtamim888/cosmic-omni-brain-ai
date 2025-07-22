#!/usr/bin/env python3
"""
🔥 GHOST TRANSCENDENCE CORE ∞ vX - Startup Script
Simple launcher with configuration options
"""

import os
import sys
import argparse
import logging
from pathlib import Path

def setup_environment():
    """Setup environment variables and paths"""
    # Create necessary directories
    directories = ['logs', 'temp', 'static/uploads']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    # Set default environment variables if not set
    env_defaults = {
        'FLASK_ENV': 'production',
        'LOG_LEVEL': 'INFO',
        'PORT': '5000',
        'SECRET_KEY': 'ghost_transcendence_core_infinity_vx'
    }
    
    for key, value in env_defaults.items():
        if key not in os.environ:
            os.environ[key] = value

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_modules = [
        'flask', 'opencv-python', 'numpy', 'pillow', 
        'python-telegram-bot', 'scikit-image'
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            if module == 'opencv-python':
                import cv2
            elif module == 'python-telegram-bot':
                import telegram
            elif module == 'scikit-image':
                import skimage
            else:
                __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print("❌ Missing required dependencies:")
        for module in missing_modules:
            print(f"   - {module}")
        print("\n💡 Install missing dependencies with:")
        print(f"   pip install {' '.join(missing_modules)}")
        return False
    
    return True

def print_banner():
    """Print the Ghost Transcendence Core banner"""
    banner = """
🔥 ══════════════════════════════════════════════════════════════ 🔥

    ██████╗ ██╗  ██╗ ██████╗ ███████╗████████╗
    ██╔══██╗██║  ██║██╔═══██╗██╔════╝╚══██╔══╝
    ██████╔╝███████║██║   ██║███████╗   ██║   
    ██╔══██╗██╔══██║██║   ██║╚════██║   ██║   
    ██████╔╝██║  ██║╚██████╔╝███████║   ██║   
    ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝   ╚═╝   

        TRANSCENDENCE CORE ∞ vX
    🎯 The God-Level AI Trading Bot

👻 No-Loss Logic Builder
⚡ Manipulation Resistant  
🧠 Infinite Intelligence
🎯 Dynamic Strategy Creation

🔥 ══════════════════════════════════════════════════════════════ 🔥
"""
    print(banner)

def main():
    """Main startup function"""
    parser = argparse.ArgumentParser(
        description='🔥 Ghost Transcendence Core ∞ vX - AI Trading Bot'
    )
    parser.add_argument(
        '--port', '-p', 
        type=int, 
        default=int(os.environ.get('PORT', 5000)),
        help='Port to run the web server on (default: 5000)'
    )
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='Run in debug mode'
    )
    parser.add_argument(
        '--host',
        default='0.0.0.0',
        help='Host to bind to (default: 0.0.0.0)'
    )
    parser.add_argument(
        '--no-telegram',
        action='store_true',
        help='Disable Telegram bot'
    )
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Setup environment
    print("🔧 Setting up environment...")
    setup_environment()
    
    # Check dependencies
    print("📦 Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    # Set configuration
    os.environ['PORT'] = str(args.port)
    os.environ['LOG_LEVEL'] = args.log_level
    
    if args.debug:
        os.environ['FLASK_ENV'] = 'development'
    
    if args.no_telegram:
        os.environ['DISABLE_TELEGRAM'] = 'true'
    
    print("✅ Environment setup complete")
    print(f"🌐 Starting Ghost Transcendence Core on http://{args.host}:{args.port}")
    
    if not args.no_telegram:
        telegram_token = os.environ.get('TELEGRAM_BOT_TOKEN')
        if telegram_token and telegram_token != 'your_bot_token_here':
            print("📱 Telegram bot will be activated")
        else:
            print("⚠️  Telegram bot token not configured (set TELEGRAM_BOT_TOKEN)")
    
    print("🎯 No-Loss Logic Builder - ACTIVE")
    print("👻 Manipulation Resistance - MAXIMUM")
    print("⚡ Infinite Intelligence - ENGAGED")
    print("\n" + "="*60)
    
    # Import and run the main app
    try:
        from app import app, logger
        logger.info("🔥 GHOST TRANSCENDENCE CORE ∞ vX STARTUP COMPLETE")
        
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug,
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n\n👻 Ghost Transcendence Core shutting down...")
        print("🎯 All trades dominated. See you in the markets!")
    except Exception as e:
        print(f"\n❌ Startup failed: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()