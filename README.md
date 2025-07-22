# 🧠 COSMIC AI Binary Signal Bot 🚀

A professional Flask web application that analyzes candlestick chart screenshots from any broker (Quotex, Binomo, Pocket Option, etc.) and predicts the next 1-minute candle (CALL / PUT / NO TRADE) using advanced multi-stage AI logic.

## ✨ Features

### 🔬 Advanced AI Analysis Engine
- **Candlestick Detection**: OpenCV-powered computer vision to detect and analyze candlestick patterns
- **Market Psychology Analysis**: Reads market sentiment and behavior patterns
- **Multi-Strategy System**: 5 different trading strategies with dynamic selection
- **Pattern Recognition**: Detects 8+ candlestick patterns including:
  - Engulfing patterns
  - Doji, Hammer, Shooting Star
  - Momentum shifts
  - Exhaustion patterns
  - Breakout/trap detection

### 🎯 Smart Strategy Engine
- **Breakout Continuation**: Follows momentum after breakouts
- **Reversal Play**: Identifies reversal opportunities at key levels
- **Momentum Shift**: Catches trend changes early
- **Trap Fade**: Fades false breakouts
- **Exhaustion Reversal**: Trades trend exhaustion

### 📱 Telegram Integration
- Automatic signal delivery to Telegram groups
- Beautiful formatted messages with emojis
- Confidence-based filtering (only sends high-confidence signals)
- Real-time signal updates

### 🌐 Modern Web Interface
- Dark-themed, mobile-responsive design
- Drag & drop file upload
- Real-time progress indicators
- Interactive results display
- System status monitoring

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd cosmic-ai-binary-bot
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure Telegram (Optional)**
```bash
# Set your Telegram chat ID
export TELEGRAM_CHAT_ID="your_chat_id_here"
```

4. **Run the application**
```bash
python app.py
```

5. **Open your browser**
Navigate to `http://localhost:5000`

## 📊 How It Works

### 1. Upload Chart Screenshot
- Drag & drop or browse to upload candlestick chart images
- Supports JPG, PNG, BMP formats
- Minimum size: 300x200 pixels

### 2. AI Analysis Process
```
Image Upload → Candle Detection → Pattern Analysis → Market Psychology → Strategy Selection → Signal Generation
```

### 3. Signal Output
- **CALL**: Bullish prediction for next 1M candle
- **PUT**: Bearish prediction for next 1M candle  
- **NO TRADE**: Low confidence, avoid trading

### 4. Confidence Threshold
- Only signals with >85% confidence are actionable
- Lower confidence signals are filtered out for safety

## ⚙️ Configuration

### Environment Variables
```bash
# Telegram Configuration
export TELEGRAM_CHAT_ID="your_chat_id_here"

# Optional Settings
export DEBUG="true"                    # Enable debug mode
export SECRET_KEY="your_secret_key"    # Flask secret key
```

### Config File (`config.py`)
```python
# AI Engine Settings
CONFIDENCE_THRESHOLD = 85.0    # Minimum confidence for signals
MIN_CANDLES_REQUIRED = 6       # Minimum candles needed
MAX_CANDLES_ANALYZED = 8       # Maximum candles to analyze

# Image Processing
MIN_IMAGE_WIDTH = 300          # Minimum image width
MIN_IMAGE_HEIGHT = 200         # Minimum image height
SUPPORTED_FORMATS = ['jpg', 'jpeg', 'png', 'bmp']
```

## 📱 Telegram Setup

### 1. Create Telegram Bot
1. Message [@BotFather](https://t.me/botfather) on Telegram
2. Use `/newbot` command
3. Get your bot token (already configured in the app)

### 2. Get Chat ID
1. Add your bot to a group or start a private chat
2. Send a message to the bot
3. Visit: `https://api.telegram.org/bot<BOT_TOKEN>/getUpdates`
4. Find your chat ID in the response
5. Set it as environment variable:
```bash
export TELEGRAM_CHAT_ID="your_chat_id"
```

### 3. Test Connection
Use the "Test Telegram Bot" button in the web interface to verify setup.

## 🔧 API Endpoints

### Main Routes
- `GET /` - Main web interface
- `POST /upload` - Chart analysis endpoint
- `POST /test-telegram` - Test Telegram bot
- `GET /health` - Health check
- `GET /api/status` - System status

### Upload API Example
```javascript
const formData = new FormData();
formData.append('chart_file', file);

fetch('/upload', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    console.log('Signal:', data.signal.signal);
    console.log('Confidence:', data.signal.confidence);
});
```

## 🏗️ Project Structure

```
cosmic-ai-binary-bot/
├── app.py                 # Main Flask application
├── config.py             # Configuration settings
├── telegram_bot.py       # Telegram integration
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── logic/
│   └── ai_engine.py     # AI analysis engine
├── templates/
│   └── index.html       # Web interface
├── static/
│   └── style.css        # Styling
└── uploads/             # Temporary file storage
```

## 🧠 AI Engine Architecture

### Core Components

1. **CandleDetector**: 
   - OpenCV-based image processing
   - Candlestick pattern extraction
   - OHLC data estimation

2. **MarketPerceptionEngine**:
   - Pattern recognition (8+ patterns)
   - Market psychology analysis
   - Support/resistance detection
   - Momentum analysis

3. **StrategyEngine**:
   - 5 trading strategies
   - Dynamic strategy selection
   - Confidence scoring
   - Risk assessment

### Analysis Flow
```python
# Example analysis flow
candles = detector.detect_candlesticks(image_path)
market_story = perception.analyze_market_story(candles)
strategy = strategy_engine.generate_strategy(market_story, candles)
signal = generate_final_signal(strategy)
```

## 📈 Trading Strategies

### 1. Breakout Continuation
- Detects price breakouts
- Confirms with momentum
- **Best for**: Trending markets

### 2. Reversal Play  
- Identifies reversal patterns
- Confirms at key levels
- **Best for**: Range-bound markets

### 3. Momentum Shift
- Catches trend changes
- Uses consecutive candle analysis
- **Best for**: Volatile markets

### 4. Trap Fade
- Fades false breakouts
- High-confidence strategy
- **Best for**: Choppy markets

### 5. Exhaustion Reversal
- Detects trend exhaustion
- Uses decreasing momentum
- **Best for**: Extended trends

## 🎨 UI Features

### Modern Design
- Dark theme optimized for trading
- Gradient backgrounds and smooth animations
- Mobile-responsive layout
- Accessibility features

### Interactive Elements
- Drag & drop file upload
- Real-time progress indicators
- Toast notifications
- Smooth scrolling and transitions

### Status Monitoring
- AI Engine status
- Telegram bot connectivity
- API version information
- System health checks

## ⚠️ Risk Disclaimer

**IMPORTANT**: This bot is for educational purposes only. Trading binary options involves significant risk and may not be suitable for all investors. Always:

- Use proper risk management
- Never risk more than you can afford to lose
- Test thoroughly before live trading
- Consider market conditions and volatility
- Seek professional financial advice

## 🔧 Development

### Running in Debug Mode
```bash
export DEBUG=true
python app.py
```

### Testing Individual Components
```python
# Test AI Engine
from logic.ai_engine import CosmicAIEngine
engine = CosmicAIEngine()
signal = engine.analyze_chart('path/to/chart.png')

# Test Telegram Bot
from telegram_bot import TelegramBot
bot = TelegramBot()
status = bot.validate_configuration()
```

### Adding New Strategies
1. Add strategy function to `StrategyEngine` class
2. Register in `self.strategies` dictionary
3. Implement confidence scoring logic
4. Test with various market conditions

## 📦 Dependencies

### Core Dependencies
- **Flask**: Web framework
- **OpenCV**: Computer vision
- **NumPy**: Numerical computing
- **Pillow**: Image processing
- **Requests**: HTTP client

### Full Requirements
See `requirements.txt` for complete dependency list with versions.

## 🚀 Deployment

### Local Development
```bash
python app.py
```

### Production Deployment
```bash
# Using Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Using Docker (create Dockerfile)
docker build -t cosmic-ai-bot .
docker run -p 5000:5000 cosmic-ai-bot
```

### Environment Variables for Production
```bash
export DEBUG=false
export TELEGRAM_CHAT_ID="production_chat_id"
export SECRET_KEY="production_secret_key"
```

## 📊 Performance Optimization

### Image Processing
- Optimized OpenCV operations
- Efficient contour detection
- Minimal memory usage

### Web Interface
- Lazy loading of components
- Compressed images and assets
- Efficient JavaScript execution

### AI Engine
- Vectorized NumPy operations
- Efficient pattern matching
- Parallel processing where possible

## 🔍 Troubleshooting

### Common Issues

1. **"No candlesticks detected"**
   - Ensure image is clear and well-lit
   - Check minimum image size requirements
   - Verify chart has visible candlesticks

2. **"Telegram not configured"**
   - Set TELEGRAM_CHAT_ID environment variable
   - Verify bot token is correct
   - Test with /test-telegram endpoint

3. **Low confidence signals**
   - Upload clearer chart images
   - Ensure sufficient candlesticks are visible
   - Check market conditions (avoid low volatility)

### Debug Mode
Enable debug mode for detailed error messages:
```bash
export DEBUG=true
python app.py
```

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review configuration settings
3. Test with sample chart images
4. Enable debug mode for detailed logs

## 🌟 Contributing

Contributions are welcome! Areas for improvement:
- Additional candlestick patterns
- New trading strategies  
- Enhanced image processing
- Mobile app development
- Performance optimizations

## 📄 License

This project is for educational purposes. Use at your own risk and always follow proper risk management when trading.

---

**🌌 Powered by Cosmic AI Engine | Built with ❤️ for traders**