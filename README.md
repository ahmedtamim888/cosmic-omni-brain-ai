# 🤖 Telegram Binary Trading Bot

A professional Telegram bot that analyzes trading chart screenshots and provides binary trading signals. The bot validates chart images from popular trading platforms and responds with trading signals.

## 🌟 Features

- **Chart Validation**: Uses OCR to verify authentic trading charts from platforms like Quotex, TradingView, MetaTrader, Binomo, etc.
- **Smart Detection**: Identifies trading-specific keywords and interface elements
- **Signal Generation**: Provides CALL/PUT signals with duration and amount recommendations
- **Error Handling**: Rejects invalid images and provides helpful feedback
- **Logging**: Comprehensive logging for monitoring and debugging
- **Clean Architecture**: Modular design with separate chart checking logic

## 📦 Installation

### Prerequisites

1. **Python 3.8+** installed on your system
2. **Tesseract OCR** engine installed:

   **Ubuntu/Debian:**
   ```bash
   sudo apt update
   sudo apt install tesseract-ocr tesseract-ocr-eng
   ```

   **macOS:**
   ```bash
   brew install tesseract
   ```

   **Windows:**
   - Download and install from: https://github.com/UB-Mannheim/tesseract/wiki
   - Add tesseract to your system PATH

### Setup Instructions

1. **Clone or download the project files**
2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Tesseract installation:**
   ```bash
   tesseract --version
   ```

## 🚀 Usage

### Running the Bot

1. **Start the bot:**
   ```bash
   python bot.py
   ```

2. **The bot will start and display:**
   ```
   INFO - Starting Telegram Trading Bot...
   INFO - Application started
   ```

### Bot Commands

- **`/start`** - Welcome message and instructions
- **Send image** - Upload a trading chart screenshot for analysis

### Supported Trading Platforms

The bot recognizes charts from:
- Quotex
- TradingView  
- MetaTrader 4/5
- Binomo
- IQ Option
- Olymp Trade
- Pocket Option
- Expert Option

### Example Responses

**Valid Chart:**
```
✅ Chart detected. Signal: 📈 CALL/UP
⏰ Duration: 1 minute
💰 Recommended amount: 2-5% of balance
```

**Invalid Image:**
```
⚠️ This is not a valid chart. Please send a real chart screenshot

📊 Make sure your image contains a trading chart from platforms like:
• Quotex
• TradingView
• MetaTrader
• Binomo
• Other trading platforms
```

## 🔧 Configuration

### Bot Token
Your bot token is already configured in `bot.py`. If you need to change it:

```python
BOT_TOKEN = "YOUR_NEW_BOT_TOKEN_HERE"
```

### Chart Detection Settings
You can modify detection sensitivity in `chart_checker.py`:

```python
# Minimum keywords required for validation
self.min_keywords_required = 2  # Adjust as needed
```

### Adding New Keywords
To add more trading platform keywords, edit the `chart_keywords` list in `chart_checker.py`:

```python
self.chart_keywords = [
    # Add your new keywords here
    "your_platform_name",
    "specific_trading_term",
    # ... existing keywords
]
```

## 📁 Project Structure

```
trading-bot/
├── bot.py              # Main Telegram bot logic
├── chart_checker.py    # OCR-based chart validation
├── requirements.txt    # Python dependencies
└── README.md          # This documentation
```

## 🔍 How It Works

1. **Image Reception**: Bot receives image from user
2. **Temporary Storage**: Image saved temporarily with unique filename
3. **OCR Processing**: Text extracted using Tesseract OCR
4. **Keyword Analysis**: Extracted text analyzed for trading-specific terms
5. **Validation**: Multiple checks determine if image is a valid trading chart
6. **Response**: Bot sends appropriate signal or error message
7. **Cleanup**: Temporary image file deleted

## 🛠️ Troubleshooting

### Common Issues

**"No module named 'pytesseract'"**
```bash
pip install pytesseract
```

**"TesseractNotFoundError"**
- Ensure Tesseract is installed and in your system PATH
- On Windows, you may need to specify the path:
  ```python
  pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
  ```

**Bot not responding**
- Check your internet connection
- Verify bot token is correct
- Check bot logs for error messages

### Debugging

Enable debug logging by modifying the logging level in `bot.py`:

```python
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.DEBUG  # Change from INFO to DEBUG
)
```

## 📊 Chart Detection Keywords

The bot looks for these types of keywords:

- **Platform names**: quotex, tradingview, metatrader, binomo
- **Currency pairs**: usd, eur, gbp, jpy, btc, eth
- **Trading elements**: payout, investment, call, put, expiry
- **Time frames**: 1m, 5m, 15m, 30m, 1h
- **Market terms**: price, balance, profit, loss, trade

## ⚠️ Important Notes

- This bot provides **placeholder signals** for demonstration
- **Not financial advice** - for educational purposes only
- Always test with demo accounts first
- Consider implementing proper signal generation logic
- Monitor bot performance and adjust keywords as needed

## 🔒 Security

- Bot token is visible in code - consider using environment variables in production
- Temporary files are automatically cleaned up
- No user data is permanently stored

## 📞 Support

If you encounter issues:

1. Check the troubleshooting section
2. Review bot logs for error messages
3. Ensure all dependencies are properly installed
4. Verify Tesseract OCR is working correctly

---

**Happy Trading! 📈📉**