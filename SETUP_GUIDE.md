# 🧠 COSMIC AI Binary Trading Signal Bot - Setup Guide

A professional Telegram bot that analyzes candlestick chart screenshots and provides AI-powered binary trading signals (CALL/PUT) with 85%+ confidence.

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.8+ installed
- Telegram account
- BotFather access on Telegram

### 2. Create Your Telegram Bot

1. **Contact BotFather on Telegram:**
   - Open Telegram and search for `@BotFather`
   - Send `/newbot` command
   - Follow instructions to create your bot
   - **Save the bot token** (looks like: `1234567890:ABCDEF...`)

2. **Get Your Bot Username:**
   - Note down your bot's username (e.g., `@YourTradingBot`)

### 3. Installation

1. **Clone/Download the project:**
   ```bash
   # Navigate to the project directory
   cd cosmic-ai-binary-bot
   ```

2. **Install dependencies:**
   ```bash
   # Create virtual environment (recommended)
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install required packages
   pip install -r requirements.txt
   ```

3. **Configure your bot:**
   - Open `config.py`
   - Replace the `TELEGRAM_BOT_TOKEN` with your bot token from BotFather
   ```python
   TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"
   ```

### 4. Run the Bot

```bash
# Activate virtual environment (if not already active)
source venv/bin/activate

# Start the bot
python start_bot.py
```

## 🎯 Features

### AI-Powered Analysis
- **Chart Recognition:** Advanced computer vision to detect candlestick patterns
- **Multi-Strategy Engine:** 5 different trading strategies
- **Pattern Detection:** Identifies 8+ candlestick patterns
- **Market Psychology:** Analyzes market sentiment and behavior

### Telegram Bot Commands
- `/start` - Welcome message and bot introduction
- `/subscribe` - Get real-time signal alerts
- `/unsubscribe` - Stop receiving signals
- `/status` - Check bot status
- `/stats` - View trading statistics
- `/help` - Detailed help and instructions
- `/analyze` - Chart analysis guide

### Supported Brokers
- ✅ Quotex
- ✅ Binomo  
- ✅ Pocket Option
- ✅ IQ Option
- ✅ Olymp Trade
- ✅ ExpertOption
- ✅ Any broker with candlestick charts

### Signal Types
- 📈 **CALL** - Price expected to go UP
- 📉 **PUT** - Price expected to go DOWN  
- ⏸️ **NO_TRADE** - Market uncertainty, avoid trading

## 🛠️ Configuration Options

Edit `config.py` to customize:

```python
# AI Engine Settings
CONFIDENCE_THRESHOLD = 85.0      # Minimum confidence for signals
MIN_CANDLES_REQUIRED = 6         # Minimum candles needed
MAX_CANDLES_ANALYZED = 8         # Maximum candles to analyze

# Trading Settings
SIGNAL_TIMEFRAME = "1M"          # 1-minute signals
TIMEZONE = timezone(timedelta(hours=6))  # Your timezone

# Image Processing
MIN_IMAGE_WIDTH = 300            # Minimum image width
MIN_IMAGE_HEIGHT = 200           # Minimum image height
SUPPORTED_FORMATS = ['jpg', 'jpeg', 'png', 'bmp']

# Strategy Weights and Thresholds
STRATEGIES = {
    'BREAKOUT_CONTINUATION': {'weight': 1.2, 'min_confidence': 80},
    'REVERSAL_PLAY': {'weight': 1.1, 'min_confidence': 85},
    'MOMENTUM_SHIFT': {'weight': 1.0, 'min_confidence': 75},
    'TRAP_FADE': {'weight': 1.3, 'min_confidence': 90},
    'EXHAUSTION_REVERSAL': {'weight': 1.15, 'min_confidence': 82}
}
```

## 🌐 Web Interface

The bot also includes a modern web interface:

1. **Access:** http://localhost:5000
2. **Features:**
   - Drag & drop chart upload
   - Real-time analysis results
   - Beautiful dark theme UI
   - Mobile responsive design

## 📱 How to Use

### For Chart Analysis:

1. **Take a screenshot** of your broker's candlestick chart
2. **Send the image** to your Telegram bot or upload via web interface
3. **Get instant analysis** with CALL/PUT signals
4. **Follow the recommendation** with the given confidence level

### Best Practices:

- ✅ Use clear, high-quality chart screenshots
- ✅ Include at least 6-8 recent candlesticks
- ✅ Ensure candlesticks are clearly visible
- ✅ Only trade signals with 85%+ confidence
- ❌ Don't trade during major news events
- ❌ Don't risk more than you can afford to lose

## 🔧 Troubleshooting

### Common Issues:

1. **Bot Token Error:**
   - Verify your bot token is correct
   - Ensure no extra spaces in config.py

2. **Dependencies Error:**
   - Run: `pip install -r requirements.txt`
   - Use Python 3.8 or higher

3. **Chart Not Recognized:**
   - Ensure image is clear and bright
   - Include more candlesticks in the screenshot
   - Try different image formats (PNG, JPG)

4. **Low Confidence Signals:**
   - Market might be uncertain
   - Try analyzing during active trading hours
   - Look for clearer chart patterns

### Getting Help:

If you encounter issues:
1. Check the logs for error messages
2. Ensure all dependencies are installed
3. Verify your bot token configuration
4. Make sure your image meets quality requirements

## ⚠️ Disclaimer

**This bot is for educational purposes only. Binary trading involves significant risk and may not be suitable for all investors. Past performance does not guarantee future results. Please trade responsibly and never risk more than you can afford to lose.**

## 🎉 Enjoy Trading!

Your COSMIC AI Binary Trading Signal Bot is now ready to help you analyze charts and make informed trading decisions. Good luck and trade safely! 🚀

---

**Need support?** Check the logs and configuration first, then review this guide for troubleshooting tips.