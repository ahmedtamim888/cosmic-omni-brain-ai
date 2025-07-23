#!/usr/bin/env python3
"""
Installation Script for Ultra-Accurate Trading Bot
"""

import os
import sys
import subprocess
import platform

def print_banner():
    """Print installation banner"""
    print("=" * 60)
    print("🚀 ULTRA-ACCURATE TRADING BOT - INSTALLATION")
    print("=" * 60)
    print()

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def install_core_dependencies():
    """Install core dependencies"""
    print("\n📦 Installing core dependencies...")
    
    core_packages = [
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
        "aiohttp>=3.8.0",
        "matplotlib>=3.5.0",
        "python-telegram-bot>=20.0",
        "python-dotenv>=0.19.0"
    ]
    
    for package in core_packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            return False
    
    return True

def install_optional_dependencies():
    """Install optional dependencies"""
    print("\n📦 Installing optional dependencies...")
    
    optional_packages = [
        ("tensorflow>=2.8.0", "Deep learning support"),
        ("torch>=1.11.0", "PyTorch support"),
        ("yfinance>=0.1.70", "Yahoo Finance data"),
        ("plotly>=5.0.0", "Interactive charts"),
        ("rich>=12.0.0", "Rich console output")
    ]
    
    for package, description in optional_packages:
        try:
            print(f"Installing {package} ({description})...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Failed to install {package} (optional): {e}")
            print(f"   Continuing without {description}")

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    directories = [
        "logs",
        "data/cache",
        "models",
        "backups"
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Created directory: {directory}")
        except Exception as e:
            print(f"❌ Failed to create directory {directory}: {e}")
            return False
    
    return True

def create_sample_env():
    """Create sample .env file"""
    print("\n📝 Creating sample .env file...")
    
    env_content = """# Ultra-Accurate Trading Bot Configuration
# Copy this file to .env and fill in your values

# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN=YOUR_TELEGRAM_BOT_TOKEN_HERE
TELEGRAM_CHAT_IDS=YOUR_CHAT_ID_HERE

# Trading Configuration
CONFIDENCE_THRESHOLD=0.95
GOD_MODE_THRESHOLD=0.97

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/trading_bot.log

# Data Source (synthetic for demo)
DATA_SOURCE=synthetic
"""
    
    try:
        if not os.path.exists('.env'):
            with open('.env.sample', 'w') as f:
                f.write(env_content)
            print("✅ Created .env.sample file")
            print("   Copy to .env and configure your settings")
        else:
            print("✅ .env file already exists")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env.sample: {e}")
        return False

def test_imports():
    """Test if core imports work"""
    print("\n🧪 Testing imports...")
    
    test_imports = [
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("sklearn", "Scikit-learn"),
        ("matplotlib", "Matplotlib"),
        ("aiohttp", "aiohttp"),
    ]
    
    failed_imports = []
    
    for module, name in test_imports:
        try:
            __import__(module)
            print(f"✅ {name} import successful")
        except ImportError as e:
            print(f"❌ {name} import failed: {e}")
            failed_imports.append(name)
    
    # Test Telegram (optional)
    try:
        import telegram
        print("✅ Telegram bot library import successful")
    except ImportError:
        print("⚠️  Telegram bot library not available (install python-telegram-bot)")
        failed_imports.append("Telegram")
    
    return len(failed_imports) == 0, failed_imports

def print_next_steps():
    """Print next steps after installation"""
    print("\n🎉 INSTALLATION COMPLETE!")
    print()
    print("📋 NEXT STEPS:")
    print("1. Configure Telegram bot:")
    print("   python setup_telegram.py")
    print()
    print("2. Run the trading bot:")
    print("   python main.py")
    print()
    print("3. Check logs:")
    print("   tail -f logs/trading_bot.log")
    print()
    print("📚 DOCUMENTATION:")
    print("• README.md - General information")
    print("• .env.sample - Configuration template")
    print("• requirements.txt - All dependencies")
    print()

def main():
    """Main installation function"""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_core_dependencies():
        print("\n❌ Core dependency installation failed!")
        sys.exit(1)
    
    # Install optional dependencies
    install_optional_dependencies()
    
    # Create directories
    if not create_directories():
        print("\n❌ Directory creation failed!")
        sys.exit(1)
    
    # Create sample config
    if not create_sample_env():
        print("\n❌ Configuration file creation failed!")
        sys.exit(1)
    
    # Test imports
    success, failed = test_imports()
    if not success:
        print(f"\n⚠️  Some imports failed: {', '.join(failed)}")
        print("The bot may still work with limited functionality.")
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Installation cancelled by user.")
    except Exception as e:
        print(f"\n❌ Installation error: {e}")
        sys.exit(1)