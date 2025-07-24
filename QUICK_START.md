# 🚀 TELEGRAM TRADING BOT - QUICK START

## 📦 INSTALLATION COMMANDS

### 1. Install Required Dependencies
```bash
# Install all Python packages
pip3 install --break-system-packages python-telegram-bot==20.7 Pillow==10.1.0 pytesseract==0.3.10 opencv-python==4.8.1.78 numpy==1.24.3 python-dotenv==1.0.0 requests==2.31.0

# Install Tesseract OCR (Linux/Ubuntu)
sudo apt update && sudo apt install -y tesseract-ocr tesseract-ocr-eng

# For macOS (with Homebrew)
# brew install tesseract
```

### 2. Verify Installation
```bash
# Test that all components work
python3 -c "from chart_checker import ChartChecker; print('✅ Chart Checker OK')"
python3 -c "import telegram; print('✅ Telegram Bot Library OK')"
tesseract --version
```

## 🤖 RUNNING THE BOT

### 1. Start the Bot
```bash
# Your bot token is already configured in bot.py
python3 bot.py
```

### 2. Test the Bot
1. **Find your bot on Telegram**: Search for your bot using the username you created with @BotFather
2. **Send `/start`**: This will show the welcome message
3. **Send a chart screenshot**: Upload any trading chart image
4. **Get results**: Bot will respond with signal or validation error

## 📊 BOT TOKEN (ALREADY CONFIGURED)

Your bot token `8288385434:AAG_RVKnlXDWBZNN38Q3IEfSQXIgxwPlsU0` is already set in `bot.py`

## 🔧 BOT COMMANDS

- `/start` - Welcome message and instructions
- `/help` - Detailed help guide  
- `/status` - Check bot status

## ✅ CHART VALIDATION

**✅ VALID CHARTS** (will get signals):
- Quotex platform screenshots
- TradingView charts
- MetaTrader (MT4/MT5) 
- Binomo charts
- IQ Option charts
- Any trading platform with visible text

**❌ INVALID IMAGES** (will be rejected):
- Random photos
- Non-trading content
- Blurry images
- Text documents

## 📈 SIGNAL RESPONSES

**For Valid Charts:**
```
✅ Chart detected. Signal: 📈 CALL

📊 Analysis Details:
🔍 Keywords detected: 5
🎯 Confidence: 85.3%
⏰ Time: 14:30:25

💡 Detected elements: quotex, eur/usd, investment...
```

**For Invalid Images:**
```
⚠️ This is not a valid chart. Please send a real chart screenshot

🔍 What I'm looking for:
• Trading platform interfaces
• Currency pairs (EUR/USD, GBP/USD, etc.)
• Trading terms (Price, Time, Investment)
```

## 🐛 TROUBLESHOOTING

### Bot not starting?
```bash
# Check if all packages are installed
pip3 list | grep telegram
pip3 list | grep pytesseract

# Install missing packages
pip3 install --break-system-packages python-telegram-bot pytesseract
```

### "Tesseract not found" error?
```bash
# Install Tesseract OCR
sudo apt install tesseract-ocr  # Linux
brew install tesseract           # macOS
```

### Bot not responding to messages?
- Check internet connection
- Verify bot token is correct
- Ensure bot is still running in terminal
- Check for error messages in console

## 📁 FILE STRUCTURE

```
📂 Your Project/
├── 🤖 bot.py                 # Main bot (YOUR TOKEN INCLUDED)
├── 🔍 chart_checker.py       # OCR validation module  
├── 📋 requirements.txt       # Dependencies list
├── 🔧 setup_bot.py          # Automated installer
├── 📖 README_TELEGRAM_BOT.md # Complete documentation
└── 📂 temp_images/           # Temp folder (auto-created)
```

## ⚡ ONE-COMMAND SETUP

```bash
# Install everything and start bot
pip3 install --break-system-packages python-telegram-bot pytesseract Pillow opencv-python numpy requests python-dotenv && python3 bot.py
```

## 🎯 READY TO TRADE!

1. **✅ Dependencies installed**
2. **✅ Bot token configured** 
3. **✅ Run `python3 bot.py`**
4. **✅ Send chart screenshots**
5. **✅ Get trading signals**

**Your bot is ready! Send trading charts and get instant signals! 📈📉**

---

## ⚠️ IMPORTANT NOTES

- **Token Security**: Your bot token is in the code. Keep it private!
- **Demo Signals**: These are educational signals, not trading advice
- **Test First**: Always test with demo charts before real trading
- **Risk Warning**: Trading involves financial risk

**Happy Trading! 🤖📊**