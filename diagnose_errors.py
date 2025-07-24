#!/usr/bin/env python3
"""
Comprehensive diagnostic script to identify and fix bot errors
"""

import sys
import traceback
import os
import subprocess

def check_python_version():
    """Check Python version compatibility"""
    print("🐍 Python Version Check:")
    version = sys.version_info
    print(f"   Current: Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required!")
        return False
    else:
        print("✅ Python version compatible")
        return True

def check_virtual_environment():
    """Check if virtual environment is active"""
    print("\n🔧 Virtual Environment Check:")
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    
    if in_venv:
        print("✅ Virtual environment is active")
        print(f"   Path: {sys.prefix}")
        return True
    else:
        print("⚠️  Virtual environment not detected")
        print("   Consider running: source venv/bin/activate")
        return True  # Not critical, but recommended

def check_critical_imports():
    """Check all critical imports with detailed error reporting"""
    print("\n📦 Critical Imports Check:")
    
    critical_packages = {
        'telegram': 'python-telegram-bot[all]>=20.0',
        'flask': 'Flask>=2.0.0', 
        'cv2': 'opencv-python>=4.5.0',
        'numpy': 'numpy>=1.20.0',
        'PIL': 'Pillow>=8.0.0',
        'requests': 'requests>=2.25.0',
        'matplotlib': 'matplotlib>=3.5.0',
        'skimage': 'scikit-image>=0.19.0',
        'scipy': 'scipy>=1.7.0'
    }
    
    failed_imports = []
    
    for package, requirement in critical_packages.items():
        try:
            if package == 'cv2':
                import cv2
                print(f"✅ OpenCV: {cv2.__version__}")
            elif package == 'PIL':
                from PIL import Image
                print(f"✅ Pillow: {Image.__version__}")
            elif package == 'skimage':
                import skimage
                print(f"✅ Scikit-image: {skimage.__version__}")
            elif package == 'telegram':
                import telegram
                print(f"✅ Python-Telegram-Bot: {telegram.__version__}")
            else:
                module = __import__(package)
                version = getattr(module, '__version__', 'unknown')
                print(f"✅ {package.capitalize()}: {version}")
                
        except ImportError as e:
            print(f"❌ {package}: {e}")
            failed_imports.append((package, requirement))
        except Exception as e:
            print(f"⚠️  {package}: {e}")
    
    if failed_imports:
        print(f"\n🔧 To fix missing packages, run:")
        for package, requirement in failed_imports:
            print(f"   pip install {requirement}")
    
    return len(failed_imports) == 0

def check_bot_configuration():
    """Check bot configuration and credentials"""
    print("\n⚙️  Bot Configuration Check:")
    
    try:
        from config import Config
        print("✅ Config file loaded successfully")
        
        # Check bot token
        if Config.TELEGRAM_BOT_TOKEN and len(Config.TELEGRAM_BOT_TOKEN) > 20:
            print(f"✅ Bot token configured: {Config.TELEGRAM_BOT_TOKEN[:10]}...")
        else:
            print("❌ Bot token missing or invalid")
            return False
            
        # Check chat ID
        if Config.TELEGRAM_CHAT_ID:
            print(f"✅ Chat ID configured: {Config.TELEGRAM_CHAT_ID}")
        else:
            print("⚠️  Chat ID not set (optional for basic functionality)")
            
        # Check other settings
        print(f"✅ Confidence threshold: {Config.CONFIDENCE_THRESHOLD}%")
        print(f"✅ Signal timeframe: {Config.SIGNAL_TIMEFRAME}")
        print(f"✅ Strategies: {len(Config.STRATEGIES)} configured")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        traceback.print_exc()
        return False

def check_file_structure():
    """Check if all required files exist"""
    print("\n📁 File Structure Check:")
    
    required_files = [
        'config.py',
        'start_bot.py', 
        'telegram_bot_pro.py',
        'app.py',
        'requirements.txt',
        'logic/ai_engine.py'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✅ {file_path} ({size:,} bytes)")
        else:
            print(f"❌ {file_path} - Missing!")
            missing_files.append(file_path)
    
    return len(missing_files) == 0

def check_bot_components():
    """Test individual bot components"""
    print("\n🤖 Bot Components Check:")
    
    try:
        # Test AI Engine
        from logic.ai_engine import CosmicAIEngine, CandleDetector
        engine = CosmicAIEngine()
        detector = CandleDetector()
        print("✅ AI Engine: Working")
        
        # Test Telegram Bot
        from telegram_bot_pro import CosmicTelegramBot
        bot = CosmicTelegramBot()
        print("✅ Telegram Bot: Working")
        
        # Test Web App
        from app import app
        print("✅ Flask Web App: Working")
        
        return True
        
    except Exception as e:
        print(f"❌ Component error: {e}")
        traceback.print_exc()
        return False

def check_network_connectivity():
    """Check network connectivity for Telegram API"""
    print("\n🌐 Network Connectivity Check:")
    
    try:
        import requests
        response = requests.get("https://api.telegram.org", timeout=10)
        if response.status_code == 200:
            print("✅ Telegram API accessible")
            return True
        else:
            print(f"⚠️  Telegram API response: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Network error: {e}")
        return False

def check_permissions():
    """Check file and directory permissions"""
    print("\n🔒 Permissions Check:")
    
    # Check upload directory
    upload_dir = 'uploads'
    if not os.path.exists(upload_dir):
        try:
            os.makedirs(upload_dir)
            print(f"✅ Created uploads directory: {upload_dir}")
        except Exception as e:
            print(f"❌ Cannot create uploads directory: {e}")
            return False
    else:
        print(f"✅ Uploads directory exists: {upload_dir}")
    
    # Check write permissions
    try:
        test_file = os.path.join(upload_dir, 'test_write.tmp')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        print("✅ Write permissions OK")
        return True
    except Exception as e:
        print(f"❌ Write permission error: {e}")
        return False

def provide_solutions(failed_checks):
    """Provide specific solutions for failed checks"""
    print("\n" + "="*60)
    print("🔧 SOLUTIONS FOR DETECTED ISSUES:")
    print("="*60)
    
    if 'imports' in failed_checks:
        print("\n📦 Fix Import Issues:")
        print("   pip install -r requirements.txt")
        print("   # or install specific packages as shown above")
    
    if 'config' in failed_checks:
        print("\n⚙️  Fix Configuration Issues:")
        print("   1. Check your bot token in config.py")
        print("   2. Ensure token format: 'BOTID:TOKEN_STRING'")
        print("   3. Get token from @BotFather on Telegram")
    
    if 'files' in failed_checks:
        print("\n📁 Fix Missing Files:")
        print("   1. Ensure all required files are present")
        print("   2. Re-download the bot if files are missing")
    
    if 'network' in failed_checks:
        print("\n🌐 Fix Network Issues:")
        print("   1. Check your internet connection")
        print("   2. Verify firewall settings")
        print("   3. Try using VPN if Telegram is blocked")
    
    if 'permissions' in failed_checks:
        print("\n🔒 Fix Permission Issues:")
        print("   chmod 755 .")
        print("   mkdir -p uploads")
        print("   chmod 777 uploads")

def main():
    """Run comprehensive diagnostics"""
    print("="*60)
    print("🔍 COSMIC AI BOT - COMPREHENSIVE DIAGNOSTICS")
    print("="*60)
    
    failed_checks = []
    
    # Run all checks
    if not check_python_version():
        failed_checks.append('python')
    
    check_virtual_environment()
    
    if not check_critical_imports():
        failed_checks.append('imports')
    
    if not check_bot_configuration():
        failed_checks.append('config')
    
    if not check_file_structure():
        failed_checks.append('files')
    
    if not check_bot_components():
        failed_checks.append('components')
    
    if not check_network_connectivity():
        failed_checks.append('network')
    
    if not check_permissions():
        failed_checks.append('permissions')
    
    # Summary
    print("\n" + "="*60)
    if not failed_checks:
        print("🎉 ALL DIAGNOSTICS PASSED!")
        print("✅ Your bot should work perfectly!")
        print("\n🚀 To start your bot:")
        print("   python start_bot.py")
    else:
        print("❌ ISSUES DETECTED!")
        print(f"   Failed checks: {', '.join(failed_checks)}")
        provide_solutions(failed_checks)
    
    print("="*60)
    
    return 0 if not failed_checks else 1

if __name__ == "__main__":
    sys.exit(main())