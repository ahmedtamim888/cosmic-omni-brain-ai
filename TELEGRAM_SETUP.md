# 📱 COSMIC AI Telegram Bot Setup

## 🚀 Direct Image Analysis via Telegram

Send chart screenshots directly to your Telegram bot and get instant AI analysis!

## 🔧 Setup Steps

### 1. **Find Your Bot on Telegram**
Your bot is already created with token: `8155744845:AAGQ6s8RMSpCAjO16Mzb1xl8_otLnXh3OKQ`

**To find your bot:**
1. Open Telegram
2. Search for your bot using the token or username
3. Start a chat with the bot

### 2. **Start the Telegram Bot Server**
```bash
# Activate virtual environment
source venv/bin/activate

# Run the Telegram bot
python run_telegram_bot.py
```

### 3. **Test Your Bot**
1. Send `/start` to your bot in Telegram
2. Send a chart screenshot
3. Get instant AI analysis!

## 📸 How to Use

### **Commands Available:**
- `/start` - Welcome message and instructions
- `/help` - Show available commands
- `/status` - Check bot status
- `/stats` - View analysis statistics

### **Send Chart Screenshots:**
1. Take a screenshot of any candlestick chart from:
   - Quotex
   - Binomo  
   - Pocket Option
   - IQ Option
   - Any other broker

2. Send the image to your bot
3. Get instant analysis with:
   - CALL/PUT/NO TRADE signal
   - Confidence percentage
   - Detailed reasoning
   - Technical analysis breakdown

## 🎯 Bot Features

### **🧠 AI Analysis:**
- Advanced pattern recognition
- Market psychology analysis
- Support/resistance detection
- Momentum and trend analysis
- Multi-timeframe context

### **📊 Signal Types:**
- **CALL** 📈 - Bullish signal
- **PUT** 📉 - Bearish signal  
- **NO TRADE** ⏸️ - Uncertain conditions

### **🔒 Confidence Levels:**
- 🔥 95%+ - Extremely high confidence
- 🚀 90%+ - Very high confidence
- 🔒 85%+ - High confidence (threshold)
- ⚡ 70%+ - Medium confidence
- ⚠️ <70% - Low confidence (analysis only)

## 📱 Example Usage

### **1. Start the bot:**
```
/start
```

**Bot Response:**
```
🧠 COSMIC AI Binary Signal Bot

🚀 Ready to analyze your charts!

📸 How to use:
1. Send me a candlestick chart screenshot
2. I'll analyze it with advanced AI  
3. Get CALL/PUT/NO TRADE signals instantly

💫 Send me a chart screenshot to get started!
```

### **2. Send a chart image:**
Upload any chart screenshot

**Bot Response:**
```
🧠 COSMIC AI ANALYSIS COMPLETE

🕒 Time: 14:25 (UTC+6)
📈 Signal: CALL
🔒 Confidence: 87.3%
📊 Reason: Breakout + Momentum Surge

📋 Technical Analysis:
🚀 Momentum: Bullish (82.1%)
📈 Trend: Uptrend
🎯 Patterns: Breakout, Engulfing
📍 Position: Near Support

✅ HIGH CONFIDENCE SIGNAL!
💡 This signal meets the 85% threshold

⚡ Powered by COSMIC AI Engine
📸 Send another chart for more analysis!
```

## 🔧 Running Both Bots

You can run both the web interface and Telegram bot simultaneously:

### **Terminal 1 - Web Interface:**
```bash
source venv/bin/activate
python app.py
# Access at http://localhost:5000
```

### **Terminal 2 - Telegram Bot:**
```bash
source venv/bin/activate  
python run_telegram_bot.py
# Bot will handle direct messages
```

## 🌟 Advanced Features

### **Group Integration:**
- High-confidence signals (85%+) automatically sent to configured group
- Personal analysis always sent to individual chat
- Different formatting for groups vs personal messages

### **Error Handling:**
- Clear error messages for invalid images
- Helpful suggestions for better screenshots
- Automatic retry prompts

### **Performance:**
- 5-10 second analysis time
- Real-time progress updates
- Professional signal formatting

## 🆘 Troubleshooting

### **Bot Not Responding:**
1. Check if `run_telegram_bot.py` is running
2. Verify bot token in `config.py`
3. Check internet connection
4. View logs for error messages

### **Analysis Errors:**
- Use clear, high-quality screenshots
- Ensure chart shows candlesticks clearly
- Try JPG, PNG, or WEBP formats
- Keep file size under 20MB

### **Commands:**
```bash
# Check if bot is running
ps aux | grep telegram

# View bot logs
tail -f telegram_bot.log

# Stop bot
Ctrl+C in terminal running the bot
```

## 🎯 Success Tips

### **Best Screenshots:**
- Clear candlestick charts
- Good contrast and lighting
- Visible timeframes (1-5 minutes work best)
- Multiple candles visible (6-8 minimum)
- Clean chart without too many indicators

### **Optimal Usage:**
- Send screenshots during active market hours
- Use 1-minute timeframes for binary options
- Wait for complete candle formations
- Consider multiple timeframe confirmation

---

## 🚀 **Quick Start:**

1. **Start bot:** `python run_telegram_bot.py`
2. **Find bot in Telegram** (use the token to search)
3. **Send `/start`** to initialize
4. **Upload chart screenshot** 
5. **Get instant AI analysis!** 🧠✨

**Your COSMIC AI bot is ready to analyze charts directly via Telegram!**