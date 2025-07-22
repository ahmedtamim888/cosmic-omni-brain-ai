# 🧠 COSMIC AI Binary Signal Bot

A professional Flask web application that analyzes candlestick chart screenshots from any broker (Quotex, Binomo, Pocket Option, etc.) and predicts the next 1-minute candle direction using advanced AI computer vision and pattern recognition.

## ✨ Features

- **🔍 Advanced Chart Analysis**: OpenCV-powered candlestick detection and pattern recognition
- **🧠 Multi-Stage AI Logic**: Analyzes momentum, trend, support/resistance, and market psychology
- **📊 Professional Signals**: Returns CALL/PUT/NO TRADE with confidence levels and detailed reasoning
- **📱 Telegram Integration**: Automatically sends signals to Telegram groups with professional formatting
- **🎨 Modern UI**: Clean, dark-themed, mobile-responsive interface with drag-and-drop functionality
- **⚡ Real-time Processing**: Fast analysis with visual feedback and loading animations

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Telegram Bot Token (create via @BotFather)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd cosmic-ai-binary-signal-bot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env file with your Telegram configuration
   ```

4. **Set up Telegram Bot**
   - Create a bot via [@BotFather](https://t.me/BotFather)
   - Get your bot token (already included in config.py)
   - Add the bot to your Telegram group
   - Get your group chat ID and update `TELEGRAM_CHAT_ID` in `.env`

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Access the application**
   - Open your browser to `http://localhost:5000`
   - Upload a candlestick chart screenshot
   - Get AI-powered trading signals!

## 🧠 AI Engine Logic

The COSMIC AI Engine uses advanced computer vision and pattern recognition:

### 1. **Candle Detection**
- OpenCV-based color detection for bullish/bearish candles
- Body and wick ratio analysis
- OHLC price estimation from visual data

### 2. **Pattern Recognition**
- Engulfing patterns
- Doji and hammer formations
- Breakout and trap zone detection
- Support/resistance level identification

### 3. **Market Psychology Analysis**
- Momentum shift detection
- Sentiment analysis (fear/greed index)
- Market conviction measurement
- Trend strength evaluation

### 4. **Strategy Generation**
- Dynamic strategy creation based on market story
- Confidence scoring (only signals above 85% threshold)
- Risk management with NO TRADE recommendations
- Detailed reasoning for each signal

## 📊 Signal Format

Signals are sent to Telegram in this format:

```
🧠 COSMIC AI SIGNAL

🕒 Time: 10:55 (UTC+6)
📈 Signal: CALL
📊 Reason: Breakout + Momentum Surge
🔒 Confidence: 92.4%

📋 Analysis Details:
🚀 Momentum: Bullish (87.3%)
📈 Trend: Uptrend
🎯 Patterns: Breakout, Engulfing
😊 Sentiment: Bullish

⚡ Powered by COSMIC AI Engine
```

## 🛠️ Configuration

### Main Configuration (`config.py`)

```python
CONFIDENCE_THRESHOLD = 85.0    # Minimum confidence for signals
MIN_CANDLES_REQUIRED = 6       # Minimum candles needed
MAX_CANDLES_ANALYZE = 8        # Maximum candles to analyze
TIMEZONE_OFFSET = 6            # Your timezone (UTC+6)
```

### Telegram Setup

1. **Get Bot Token**: Message [@BotFather](https://t.me/BotFather) and create a new bot
2. **Get Chat ID**: Add bot to your group and send a message, then visit:
   ```
   https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates
   ```
3. **Update Configuration**: Add your chat ID to `.env` file

## 📁 Project Structure

```
cosmic-ai-binary-signal-bot/
├── app.py                 # Main Flask application
├── config.py             # Configuration settings
├── telegram_bot.py       # Telegram integration
├── requirements.txt      # Dependencies
├── logic/
│   ├── __init__.py
│   └── ai_engine.py      # AI analysis engine
├── templates/
│   └── index.html        # Main UI template
├── static/
│   ├── style.css         # Styling
│   └── script.js         # Frontend JavaScript
├── uploads/              # Temporary file storage
└── README.md            # This file
```

## 🔧 API Endpoints

- `GET /` - Main application interface
- `POST /analyze` - Upload and analyze chart
- `GET /health` - Health check endpoint
- `GET /stats` - Application statistics

## 📱 Usage

1. **Upload Chart**: Drag and drop or browse for a candlestick chart screenshot
2. **Analyze**: Click "Analyze Chart" to process the image
3. **Review Signal**: Get CALL/PUT/NO TRADE recommendation with confidence
4. **Telegram Notification**: Automatic signal delivery to your group (if confidence ≥ 85%)

## 🎯 Supported Brokers

The AI engine works with screenshots from any binary options broker:
- Quotex
- Binomo
- Pocket Option
- IQ Option
- Olymp Trade
- And many others!

## 📈 Performance Tips

- **Image Quality**: Use high-resolution screenshots for better analysis
- **Chart Visibility**: Ensure candles are clearly visible with good contrast
- **Timeframe**: 1-minute charts work best for binary options analysis
- **Clean Charts**: Remove indicators and overlays for optimal detection

## 🔒 Security

- File size limited to 16MB
- Only image files accepted (JPG, PNG, WEBP)
- Input validation and sanitization
- Error handling and logging
- Secure file handling

## 🚀 Deployment

### Local Development
```bash
python app.py
```

### Production Deployment
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker Deployment
```bash
# Build image
docker build -t cosmic-ai-bot .

# Run container
docker run -p 5000:5000 -e TELEGRAM_CHAT_ID=your_chat_id cosmic-ai-bot
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This bot is for educational and research purposes. Always:
- Use proper risk management
- Test signals before using with real money
- Understand that past performance doesn't guarantee future results
- Trade responsibly and within your means

## 🆘 Support

If you encounter issues:
1. Check the logs for error messages
2. Verify Telegram bot configuration
3. Ensure all dependencies are installed
4. Test with `/health` endpoint

## 🌟 Features Roadmap

- [ ] Multi-timeframe analysis
- [ ] Historical performance tracking
- [ ] Advanced pattern recognition
- [ ] Machine learning improvements
- [ ] WebSocket real-time updates
- [ ] Multiple Telegram groups support
- [ ] Database integration for analytics

---

**🧠 Powered by COSMIC AI - Advanced Computer Vision for Binary Options Trading**