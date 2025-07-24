# 🤖 Telegram Binary Trading Bot

A professional Telegram bot that analyzes trading chart screenshots and provides binary options signals. Uses advanced OCR (Optical Character Recognition) to validate real trading charts and avoid fake images.

## 🌟 Features

- **📊 Chart Validation**: Advanced OCR analysis to detect real trading charts
- **🔍 Multi-Platform Support**: Quotex, TradingView, MetaTrader, Binomo, IQ Option, etc.
- **📈 Signal Generation**: Provides CALL/PUT signals for validated charts
- **🛡️ Fake Image Detection**: Rejects non-trading images and random screenshots
- **⚡ Fast Processing**: Async processing for quick responses
- **🎯 Professional UI**: Clean, user-friendly Telegram interface
- **📝 Detailed Logging**: Comprehensive logging for debugging and monitoring

## 📋 Requirements

### System Requirements
- **Operating System**: Linux (Ubuntu/Debian), macOS, or Windows
- **Python**: 3.8 or higher
- **Tesseract OCR**: For text extraction from images
- **Memory**: At least 1GB RAM
- **Storage**: 500MB free space

### Python Dependencies
- `python-telegram-bot==20.7`
- `Pillow==10.1.0`
- `pytesseract==0.3.10`
- `opencv-python==4.8.1.78`
- `numpy==1.24.3`
- `python-dotenv==1.0.0`
- `requests==2.31.0`

## 🚀 Quick Installation

### Option 1: Automated Setup (Recommended)

```bash
# 1. Clone or download the bot files
# 2. Run the automated setup script
python3 setup_bot.py

# 3. Follow the prompts and enter your password when requested
# 4. The script will install everything automatically
```

### Option 2: Manual Installation

#### For Linux (Ubuntu/Debian):
```bash
# 1. Update system packages
sudo apt update

# 2. Install Tesseract OCR
sudo apt install -y tesseract-ocr tesseract-ocr-eng libtesseract-dev

# 3. Install Python dependencies
sudo apt install -y python3-pip python3-dev libgl1-mesa-glx libglib2.0-0

# 4. Install Python packages
pip3 install -r requirements.txt
```

#### For macOS:
```bash
# 1. Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Install Tesseract
brew install tesseract

# 3. Install Python packages
pip3 install -r requirements.txt
```

#### For Windows:
1. Download and install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
2. Add Tesseract to your system PATH
3. Install Python packages: `pip install -r requirements.txt`

## ⚙️ Configuration

### Bot Token Setup
1. Create a new bot on Telegram:
   - Message @BotFather on Telegram
   - Send `/newbot` and follow instructions
   - Copy your bot token

2. Update the bot token in `bot.py`:
   ```python
   BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"
   ```

   Or use environment variable (recommended for security):
   ```bash
   export TELEGRAM_BOT_TOKEN="your_bot_token_here"
   ```

## 🏃‍♂️ Running the Bot

### Basic Usage
```bash
# Start the bot
python3 bot.py
```

### Running in Background
```bash
# Run in background (Linux/macOS)
nohup python3 bot.py &

# Or use screen
screen -S trading_bot
python3 bot.py
# Press Ctrl+A then D to detach
```

### Using systemd (Linux Production)
```bash
# Create service file
sudo nano /etc/systemd/system/trading-bot.service

# Add this content:
[Unit]
Description=Telegram Trading Bot
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/path/to/bot
ExecStart=/usr/bin/python3 /path/to/bot/bot.py
Restart=always

[Install]
WantedBy=multi-user.target

# Enable and start service
sudo systemctl enable trading-bot
sudo systemctl start trading-bot
```

## 📱 Bot Usage

### Commands
- `/start` - Welcome message and instructions
- `/help` - Detailed help and usage guide
- `/status` - Check bot status and capabilities

### How to Use
1. **Start the bot**: Send `/start` to your bot on Telegram
2. **Send chart screenshot**: Upload a trading chart image
3. **Get analysis**: Bot analyzes and responds with signal or error
4. **Follow signals**: Use the provided CALL/PUT signals for trading

### Supported Chart Types
✅ **Valid Charts:**
- Quotex platform screenshots
- TradingView chart images
- MetaTrader (MT4/MT5) charts
- Binomo platform charts
- IQ Option charts
- Other forex/binary trading platforms

❌ **Invalid Images:**
- Random photos
- Non-trading screenshots
- Blurry or unreadable images
- Text documents
- Memes or other content

## 🔧 Architecture

### File Structure
```
telegram-trading-bot/
├── bot.py                 # Main Telegram bot logic
├── chart_checker.py       # OCR chart validation module
├── requirements.txt       # Python dependencies
├── setup_bot.py          # Automated setup script
├── README_TELEGRAM_BOT.md # This documentation
├── temp_images/           # Temporary image storage
└── logs/                  # Log files (if enabled)
```

### Core Components

#### 1. TradingBot Class (`bot.py`)
- **Telegram Integration**: Handles all bot interactions
- **Message Processing**: Manages commands and image uploads
- **Async Operations**: Non-blocking image processing
- **Error Handling**: Comprehensive error management
- **Signal Generation**: Creates trading signals for valid charts

#### 2. ChartChecker Class (`chart_checker.py`)
- **OCR Processing**: Extracts text from chart images
- **Keyword Analysis**: Validates trading-related content
- **Image Preprocessing**: Enhances images for better OCR
- **Confidence Scoring**: Calculates chart validity confidence
- **Multi-Platform Support**: Recognizes various trading platforms

## 🎯 Chart Validation Logic

### Keyword Detection
The bot searches for trading-related keywords in images:

**Platform Names:**
- quotex, tradingview, metatrader, mt4, mt5, binomo, iqoption, etc.

**Currency Pairs:**
- eur/usd, gbp/usd, usd/jpy, btc/usdt, etc.

**Trading Terms:**
- payout, investment, call, put, price, time, expiry, etc.

**Technical Indicators:**
- rsi, macd, support, resistance, volume, etc.

### Confidence Scoring
- **Minimum Keywords**: 2+ trading keywords required
- **Platform Bonus**: +30% for known platform names
- **Currency Bonus**: +20% for currency pairs
- **Trading Terms**: +20% for trading terminology
- **Length Penalty**: -10% for very short text (likely false positive)

### Validation Thresholds
- **Minimum Keywords Found**: 2
- **Minimum Confidence Score**: 30%
- **OCR Text Length**: At least 10 characters

## 📊 Signal Generation

### Signal Types
- **📈 CALL/UP**: Buy signal for upward price movement
- **📉 PUT/DOWN**: Sell signal for downward price movement
- **⚠️ NO SIGNAL**: Invalid chart or insufficient confidence

### Response Format
```
✅ Chart detected. Signal: 📈 CALL

📊 Analysis Details:
🔍 Keywords detected: 5
🎯 Confidence: 85.3%
⏰ Time: 14:30:25

💡 Detected elements: quotex, eur/usd, investment, price, call...

⚠️ Disclaimer: This is a demo signal. Always do your own analysis before trading!
```

## 🐛 Troubleshooting

### Common Issues

#### 1. "Tesseract not found" Error
**Solution:**
```bash
# Linux
sudo apt install tesseract-ocr

# macOS
brew install tesseract

# Windows: Download from official website and add to PATH
```

#### 2. "No module named 'telegram'" Error
**Solution:**
```bash
pip3 install python-telegram-bot==20.7
```

#### 3. Bot not responding
**Solutions:**
- Check bot token is correct
- Verify internet connection
- Check if bot is running: `ps aux | grep bot.py`
- Review logs for error messages

#### 4. "Invalid chart" for valid charts
**Solutions:**
- Ensure image is clear and readable
- Check if platform name is visible
- Try with different chart screenshot
- Verify Tesseract is working: `tesseract --version`

#### 5. High memory usage
**Solutions:**
- Restart bot periodically
- Clean temp_images directory
- Monitor with `top` or `htop`

### Debug Mode
Enable detailed logging by modifying `bot.py`:
```python
logging.basicConfig(level=logging.DEBUG)
```

## 🔒 Security Considerations

### Bot Token Security
- Never commit bot tokens to version control
- Use environment variables for production
- Regenerate tokens if compromised

### Image Processing
- Images are temporarily stored and deleted
- No permanent image storage
- OCR processing is local (no external APIs)

### User Privacy
- No user data is permanently stored
- Only processing happens locally
- Images are processed and immediately deleted

## 📈 Performance Optimization

### Image Processing
- Images are preprocessed for better OCR accuracy
- Temporary files are cleaned up automatically
- Async processing prevents blocking

### Memory Management
- Temp images are deleted after processing
- Limited image size handling
- Efficient keyword matching algorithms

### Scaling
- Single-threaded design suitable for personal use
- Can handle multiple users simultaneously
- Database can be added for persistence if needed

## 🔄 Updates and Maintenance

### Regular Updates
```bash
# Update Python packages
pip3 install --upgrade -r requirements.txt

# Update Tesseract (Linux)
sudo apt update && sudo apt upgrade tesseract-ocr
```

### Monitoring
- Check bot logs regularly
- Monitor memory and CPU usage
- Test with sample charts periodically

### Backup
- Backup bot configuration
- Save any custom keyword modifications
- Keep copy of working requirements.txt

## 🆘 Support

### Getting Help
1. Check this README thoroughly
2. Review error logs in console output
3. Test individual components (chart_checker.py)
4. Verify all dependencies are installed

### Contributing
- Fork the repository
- Make improvements
- Test thoroughly
- Submit pull requests

### Issues
- Clear description of problem
- Include error messages
- Specify operating system
- Provide steps to reproduce

## 📄 License

This project is for educational purposes. Please ensure compliance with:
- Telegram Bot API Terms of Service
- Your local trading regulations
- OCR library licenses (Tesseract)

## ⚠️ Disclaimer

**Important:** This bot provides demo signals for educational purposes only. 

- **Not Financial Advice**: Signals are for demonstration and learning
- **Risk Warning**: Trading involves significant financial risk
- **Do Your Research**: Always conduct your own analysis
- **Test Thoroughly**: Test the bot extensively before any real use
- **No Guarantees**: No warranty on signal accuracy or profitability

**Trade responsibly and never risk more than you can afford to lose.**

---

## 🎉 Ready to Start?

1. **Install**: Run `python3 setup_bot.py`
2. **Configure**: Update bot token in `bot.py`
3. **Start**: Run `python3 bot.py`
4. **Test**: Send a chart screenshot to your bot
5. **Trade**: Use signals responsibly!

Happy trading! 📈📉🤖