# 🔥 Supreme Sniper Bot — Enhanced OTC Trading Bot

## 📋 Overview

The **Supreme Sniper Bot** is an advanced automated trading bot designed specifically for **OTC (Over-The-Counter)** markets on Quotex. It features **5 sophisticated sniper strategies** with real-time signal generation and professional Telegram notifications.

### ✨ Key Features

- 🎯 **5 Advanced OTC Strategies**: Three-candle traps, wick rejections, false breakouts, morning/evening stars, and inside bar patterns
- 📱 **Professional Telegram Signals**: Bullet-style messages with expiry times and confidence levels
- ⏱ **Smart Signal Spacing**: 15-second intervals between signals to prevent spam
- 🔄 **Multi-Timeframe Analysis**: Scans both 1M and 2M charts simultaneously
- 🛡 **Confluence-Based Filtering**: Requires multiple confirmations for higher accuracy
- 🌐 **Automated Browser Control**: Seamless integration with Quotex platform

---

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.8 or higher
- Chrome browser installed
- Quotex account with valid credentials
- Telegram bot token and chat ID

### 2. Installation

```bash
# Clone or download the project
git clone <repository-url>
cd supreme-sniper-bot

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Setup

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` with your credentials:
```bash
# Quotex Login Credentials
EMAIL=your_quotex_email@example.com
PASSWORD=your_quotex_password

# Telegram Bot Configuration
BOT_TOKEN=your_telegram_bot_token_here
CHAT_ID=your_telegram_chat_id_here
```

### 4. Running the Bot

```bash
# Method 1: Direct execution
python supreme_sniper_bot.py

# Method 2: Using the launcher (recommended)
python run_supreme_sniper.py
```

---

## 🎯 Trading Strategies Explained

### 1. 🟢🟢🟢 Three-Candle Trap
**Signal**: PUT after 3 consecutive green candles
- **Logic**: Market exhaustion after prolonged buying pressure
- **Entry**: When third green candle completes
- **Confidence**: High (reversal probability)

### 2. 🔴🔴🔴 Three-Red Candle Trap  
**Signal**: CALL after 3 consecutive red candles
- **Logic**: Oversold condition, bounce expected
- **Entry**: When third red candle completes
- **Confidence**: High (bounce probability)

### 3. 📏 Wick Rejection Pattern
**Signal**: Direction based on wick dominance
- **Logic**: Large wicks indicate rejection of price levels
- **Condition**: Body < 25% of total candle range
- **Entry**: Direction opposite to wick direction

### 4. 💥 False Breakout Strategy
**Signal**: Fade the false breakout
- **Logic**: Price breaks key levels but fails to sustain
- **Detection**: Break above/below previous high/low with immediate reversal
- **Entry**: Opposite direction of failed breakout

### 5. ⭐ Morning/Evening Star
**Signal**: Reversal after doji formation
- **Logic**: Classic reversal pattern with indecision candle
- **Pattern**: Strong candle → Doji → Opposite strong candle
- **Entry**: Direction of the third candle

### 6. 📦 Inside Bar Trap
**Signal**: Breakout direction after consolidation
- **Logic**: Compression leads to explosive moves
- **Pattern**: Current candle inside previous candle's range
- **Entry**: Direction of breakout candle

---

## 📱 Telegram Signal Format

```
🔥 𝗦𝗡𝗜𝗣𝗘𝗥 𝗦𝗜𝗚𝗡𝗔𝗟

• 📌 **Pair**: EUR/USD_otc
• ⏱ **Timeframe**: 1M
• 📉 **Direction**: CALL
• 🕓 **Time**: 14:23 UTC
• ⏳ **Expiry**: 1 Minute
• 🎯 **Strategy**: 3 Red Candle Trap + 1 extra confluence(s)
• ✅ **Confidence**: 90%
```

---

## 🔧 Configuration Options

### Bot Settings
```python
TIMEFRAMES = [1, 2]      # Scan 1M and 2M charts
SCAN_SLEEP = 1.5         # Seconds between asset scans
WINDOW = 50              # Number of candles to analyze
MIN_SCORE = 3            # Minimum confluences required
```

### Signal Filtering
- **Confluence Requirement**: Signals need ≥2 strategy confirmations
- **Time Spacing**: 15-second minimum between signals
- **Asset Filtering**: Only OTC pairs are analyzed

---

## 🧪 Testing

Test the strategies before going live:

```bash
# Run strategy tests with mock data
python test_supreme_sniper.py
```

This will validate:
- ✅ Strategy logic with different candle patterns
- ✅ Signal generation accuracy
- ✅ Telegram message formatting
- ✅ Confluence detection

---

## 🛡 Risk Management

### Built-in Safety Features
- **Signal Spacing**: Prevents over-trading
- **Confluence Filtering**: Reduces false signals
- **Multi-Strategy Validation**: Cross-verification of setups
- **Time-based Filtering**: Avoids market noise

### Recommended Practices
1. **Start Small**: Begin with minimum position sizes
2. **Monitor Performance**: Track win rates and adjust
3. **Set Stop Losses**: Use platform stop-loss features
4. **Time-Based Rules**: Avoid trading during major news
5. **Bankroll Management**: Never risk more than 2% per trade

---

## 📊 Performance Monitoring

### Log Analysis
The bot logs all activities:
```
2024-01-15 14:23:15 [INFO] Signal sent: EUR/USD_otc 1M CALL
2024-01-15 14:23:30 [INFO] Analyzing GBP/USD_otc 2M
2024-01-15 14:24:01 [INFO] No signal detected for USD/JPY_otc
```

### Key Metrics to Track
- Signal frequency per hour
- Strategy distribution
- Time between signals
- Error rates and exceptions

---

## 🚨 Troubleshooting

### Common Issues

**1. Login Failed**
```
ERROR: Login failed — check credentials/2FA
```
- Solution: Verify email/password, disable 2FA temporarily

**2. No Candle Data**
```
ERROR: Empty DataFrame returned
```
- Solution: Check if market is open, verify asset availability

**3. Telegram Errors**
```
ERROR: Telegram error: Unauthorized
```
- Solution: Verify BOT_TOKEN and CHAT_ID are correct

**4. Chrome Driver Issues**
```
ERROR: ChromeDriver not found
```
- Solution: The bot auto-installs ChromeDriver, ensure internet connection

### Debug Mode

Enable detailed logging:
```bash
export DEBUG=True
python supreme_sniper_bot.py
```

---

## 📈 Advanced Configuration

### Custom Timeframes
```python
TIMEFRAMES = [1, 2, 5]  # Add 5-minute analysis
```

### Strategy Weights
Modify strategy sensitivity in the `otc_sniper` method:
```python
# Require more confluences for higher accuracy
if len(signals) >= 3:  # Instead of 2
```

### Custom Assets
Filter specific OTC pairs:
```python
# In the main loop
preferred_assets = ["EUR/USD_otc", "GBP/USD_otc"]
assets = [a for a in all_assets if a in preferred_assets]
```

---

## 📞 Support & Updates

### Getting Help
1. Check the troubleshooting section above
2. Review log files for error details
3. Test with `test_supreme_sniper.py` first
4. Verify all environment variables are set

### Updates & Improvements
- Monitor the repository for strategy updates
- Test new versions in demo mode first
- Keep dependencies updated: `pip install -r requirements.txt --upgrade`

---

## ⚖️ Disclaimer

**IMPORTANT RISK WARNING**: 
- This bot is for educational and research purposes only
- Trading financial instruments carries significant risk
- Past performance does not guarantee future results
- Never invest more than you can afford to lose
- Always test strategies in demo accounts first
- The authors are not responsible for any financial losses

---

## 🎯 Success Tips

1. **Paper Trade First**: Test with demo account for 1-2 weeks
2. **Monitor Closely**: Watch first 24 hours of live trading
3. **Keep Records**: Track all signals and outcomes
4. **Stay Updated**: Markets evolve, strategies need adjustment
5. **Risk Management**: Never risk more than 1-2% per trade

---

**Happy Trading! 🚀**