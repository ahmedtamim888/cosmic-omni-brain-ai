#!/usr/bin/env python3
"""
🔥 GHOST TRANSCENDENCE CORE ∞ vX - Startup Script
Enhanced startup script with Telegram bot support
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path

def print_banner():
    """Print the Ghost Transcendence Core banner"""
    banner = """
🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥

     ██████╗ ██╗  ██╗ ██████╗ ███████╗████████╗
    ██╔════╝ ██║  ██║██╔═══██╗██╔════╝╚══██╔══╝
    ██║  ███╗███████║██║   ██║███████╗   ██║   
    ██║   ██║██╔══██║██║   ██║╚════██║   ██║   
    ╚██████╔╝██║  ██║╚██████╔╝███████║   ██║   
     ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝   ╚═╝   

 ████████╗██████╗  █████╗ ███╗   ██╗███████╗ ██████╗███████╗███╗   ██╗██████╗ 
 ╚══██╔══╝██╔══██╗██╔══██╗████╗  ██║██╔════╝██╔════╝██╔════╝████╗  ██║██╔══██╗
    ██║   ██████╔╝███████║██╔██╗ ██║███████╗██║     █████╗  ██╔██╗ ██║██║  ██║
    ██║   ██╔══██╗██╔══██║██║╚██╗██║╚════██║██║     ██╔══╝  ██║╚██╗██║██║  ██║
    ██║   ██║  ██║██║  ██║██║ ╚████║███████║╚██████╗███████╗██║ ╚████║██████╔╝
    ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝ ╚═════╝╚══════╝╚═╝  ╚═══╝╚═════╝ 

         ██████╗ ██████╗ ██████╗ ███████╗    ∞    ██╗   ██╗██╗  ██╗
        ██╔════╝██╔═══██╗██╔══██╗██╔════╝         ██║   ██║╚██╗██╔╝
        ██║     ██║   ██║██████╔╝█████╗           ██║   ██║ ╚███╔╝ 
        ██║     ██║   ██║██╔══██╗██╔══╝           ╚██╗ ██╔╝ ██╔██╗ 
        ╚██████╗╚██████╔╝██║  ██║███████╗          ╚████╔╝ ██╔╝ ██╗
         ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝           ╚═══╝  ╚═╝  ╚═╝

🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥

             🎯 THE GOD-LEVEL AI TRADING BOT
        ⚡ Infinite Intelligence • No-Loss Logic Builder
          🧠 Dynamic Strategy Creation • Manipulation Resistant
                👻 Ghost Transcendence Core Activated

🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥
"""
    print(banner)

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'flask', 'opencv-python', 'numpy', 'pillow', 
        'python-telegram-bot', 'scikit-image', 'matplotlib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
            elif package == 'python-telegram-bot':
                import telegram
            elif package == 'scikit-image':
                import skimage
            else:
                __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        
        for package in missing_packages:
            print(f"📦 Installing {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        
        print("✅ All dependencies installed!")
    else:
        print("✅ All dependencies are installed!")

def setup_environment():
    """Setup environment variables"""
    print("🌍 Setting up environment...")
    
    # Default environment variables
    defaults = {
        'FLASK_ENV': 'production',
        'FLASK_DEBUG': 'false',
        'HOST': '0.0.0.0',
        'PORT': '5000',
        'SECRET_KEY': 'ghost_transcendence_core_infinity_key',
        'TELEGRAM_BOT_TOKEN': '7604218758:AAHJj2zMDTfVwyJHpLClVCDzukNr2Psj-38',
        'API_URL': 'http://localhost:5000',
        'LOG_LEVEL': 'INFO'
    }
    
    for key, value in defaults.items():
        if key not in os.environ:
            os.environ[key] = value
            print(f"🔧 Set {key}={value}")
        else:
            print(f"✅ {key} already configured")

def start_flask_app(host='0.0.0.0', port=5000, debug=False):
    """Start the Flask application"""
    print(f"🚀 Starting Flask app on {host}:{port}")
    
    # Import and run the app
    try:
        from app import app
        app.run(host=host, port=port, debug=debug, threaded=True)
    except Exception as e:
        print(f"❌ Flask startup error: {e}")
        return False
    
    return True

def start_telegram_bot():
    """Start the Telegram bot in a separate process"""
    print("📱 Starting Telegram bot...")
    
    try:
        # Check if bot token is configured
        token = os.environ.get('TELEGRAM_BOT_TOKEN')
        if not token or token == 'your_bot_token_here':
            print("⚠️  Telegram bot token not configured, skipping bot startup")
            return False
        
        # Start bot in subprocess
        bot_process = subprocess.Popen([
            sys.executable, 'telegram_bot.py'
        ], env=os.environ.copy())
        
        print("✅ Telegram bot started successfully!")
        return bot_process
        
    except Exception as e:
        print(f"❌ Telegram bot startup error: {e}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='🔥 Ghost Transcendence Core ∞ vX - Ultimate AI Trading Bot'
    )
    
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind Flask app')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind Flask app')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--no-telegram', action='store_true', help='Disable Telegram bot')
    parser.add_argument('--telegram-only', action='store_true', help='Run only Telegram bot')
    parser.add_argument('--check-deps', action='store_true', help='Check dependencies only')
    parser.add_argument('--install-deps', action='store_true', help='Install missing dependencies')
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Check dependencies if requested
    if args.check_deps or args.install_deps:
        check_dependencies()
        if args.check_deps:
            return
    
    # Setup environment
    setup_environment()
    
    # Set environment variables from args
    os.environ['HOST'] = args.host
    os.environ['PORT'] = str(args.port)
    os.environ['FLASK_DEBUG'] = 'true' if args.debug else 'false'
    
    if args.no_telegram:
        os.environ['DISABLE_TELEGRAM'] = 'true'
    
    try:
        if args.telegram_only:
            # Run only Telegram bot
            print("📱 Running in Telegram-only mode...")
            bot_process = start_telegram_bot()
            if bot_process:
                print("📱 Telegram bot is running. Press Ctrl+C to stop.")
                try:
                    bot_process.wait()
                except KeyboardInterrupt:
                    print("\n🛑 Shutting down Telegram bot...")
                    bot_process.terminate()
        else:
            # Run Flask app (with or without Telegram bot)
            if not args.no_telegram:
                bot_process = start_telegram_bot()
                if bot_process:
                    print("📱 Telegram bot started in background")
                    time.sleep(2)  # Give bot time to start
            
            print("🚀 Starting Flask application...")
            print(f"🌐 Web interface: http://{args.host}:{args.port}")
            print(f"🔗 API endpoint: http://{args.host}:{args.port}/analyze")
            
            if not args.no_telegram:
                print("📱 Telegram bot: Active (send charts for analysis)")
            
            print("\n🎯 Ghost Transcendence Core is ready to dominate the charts!")
            print("👻 Send chart screenshots and watch the AI magic happen...")
            print("\n🛑 Press Ctrl+C to stop")
            
            # Start Flask app
            start_flask_app(args.host, args.port, args.debug)
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down Ghost Transcendence Core...")
    except Exception as e:
        print(f"❌ Startup error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()