#!/usr/bin/env python3
"""
🔧 TELEGRAM TRADING BOT SETUP SCRIPT
Automated installation and configuration
"""

import subprocess
import sys
import os
import platform
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False

def install_system_dependencies():
    """Install system-level dependencies"""
    system = platform.system().lower()
    
    if system == "linux":
        print("🐧 Detected Linux system")
        
        # Update package list
        run_command("sudo apt update", "Updating package list")
        
        # Install tesseract and required packages
        dependencies = [
            "tesseract-ocr",
            "tesseract-ocr-eng",
            "libtesseract-dev",
            "python3-pip",
            "python3-dev",
            "libgl1-mesa-glx",
            "libglib2.0-0"
        ]
        
        for dep in dependencies:
            run_command(f"sudo apt install -y {dep}", f"Installing {dep}")
            
    elif system == "darwin":  # macOS
        print("🍎 Detected macOS system")
        
        # Check if Homebrew is installed
        if run_command("which brew", "Checking Homebrew"):
            run_command("brew install tesseract", "Installing Tesseract via Homebrew")
        else:
            print("❌ Homebrew not found. Please install Homebrew first:")
            print("   /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
            return False
            
    elif system == "windows":
        print("🪟 Detected Windows system")
        print("⚠️ Manual installation required for Windows:")
        print("   1. Download Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki")
        print("   2. Install Tesseract and add to PATH")
        print("   3. Run: pip install -r requirements.txt")
        return False
    
    return True

def install_python_dependencies():
    """Install Python packages"""
    print("🐍 Installing Python dependencies...")
    
    # Upgrade pip first
    run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip")
    
    # Install requirements
    if os.path.exists("requirements.txt"):
        return run_command(f"{sys.executable} -m pip install -r requirements.txt", "Installing Python packages")
    else:
        # Install packages manually if requirements.txt doesn't exist
        packages = [
            "python-telegram-bot==20.7",
            "Pillow==10.1.0", 
            "pytesseract==0.3.10",
            "opencv-python==4.8.1.78",
            "numpy==1.24.3",
            "python-dotenv==1.0.0",
            "requests==2.31.0"
        ]
        
        for package in packages:
            if not run_command(f"{sys.executable} -m pip install {package}", f"Installing {package}"):
                return False
        return True

def create_directories():
    """Create necessary directories"""
    directories = ["temp_images", "logs"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    return True

def test_installation():
    """Test if all components are working"""
    print("🧪 Testing installation...")
    
    try:
        # Test Python imports
        import telegram
        import PIL
        import pytesseract
        import cv2
        import numpy
        print("✅ All Python packages imported successfully")
        
        # Test Tesseract
        version = pytesseract.get_tesseract_version()
        print(f"✅ Tesseract version: {version}")
        
        # Test chart checker
        from chart_checker import ChartChecker
        checker = ChartChecker()
        print("✅ Chart checker initialized successfully")
        
        print("🎉 Installation test completed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("""
🤖 TELEGRAM TRADING BOT SETUP
==============================

This script will install all required dependencies for the Telegram Trading Bot.

⚠️  You may need to enter your password for system-level installations.
""")
    
    # Confirm before proceeding
    response = input("Do you want to continue? (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("❌ Setup cancelled by user")
        return
    
    setup_steps = [
        ("Installing system dependencies", install_system_dependencies),
        ("Installing Python dependencies", install_python_dependencies), 
        ("Creating directories", create_directories),
        ("Testing installation", test_installation)
    ]
    
    success_count = 0
    
    for step_name, step_function in setup_steps:
        print(f"\n🔧 {step_name}")
        if step_function():
            success_count += 1
        else:
            print(f"❌ {step_name} failed")
            break
    
    if success_count == len(setup_steps):
        print(f"""
🎉 SETUP COMPLETED SUCCESSFULLY!

✅ All {len(setup_steps)} setup steps completed
✅ Bot is ready to run

🚀 NEXT STEPS:
1. Edit bot.py if you want to change the bot token
2. Run the bot: python3 bot.py
3. Send /start to your bot in Telegram
4. Send a trading chart screenshot to test

📊 SUPPORTED PLATFORMS:
• Quotex
• TradingView  
• MetaTrader (MT4/MT5)
• Binomo
• IQ Option
• And more...

Happy trading! 📈📉
""")
    else:
        print(f"""
❌ SETUP INCOMPLETE

Only {success_count}/{len(setup_steps)} steps completed successfully.
Please check the error messages above and try again.

💡 COMMON ISSUES:
• Permissions: Run with sudo if needed
• Internet: Check your internet connection
• System: Ensure you're on a supported OS

Need help? Check the documentation or try manual installation.
""")

if __name__ == "__main__":
    main()