#!/usr/bin/env python3
"""
📱 GHOST TRANSCENDENCE CORE - Telegram Bot Module
Separate module for Telegram bot to avoid async conflicts
"""

import os
import logging
import asyncio
from datetime import datetime
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import requests
import io

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GhostTelegramBot:
    """Ghost Transcendence Core Telegram Bot"""
    
    def __init__(self, token: str, api_url: str = "http://localhost:5000"):
        self.token = token
        self.api_url = api_url
        self.bot = None
        self.application = None
        
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        welcome_msg = """
🔥 WELCOME TO GHOST TRANSCENDENCE CORE ∞ vX

🎯 THE GOD-LEVEL AI TRADING BOT

✅ Send me a candlestick chart screenshot
✅ I'll analyze it with infinite intelligence
✅ Get precise CALL/PUT signals
✅ No fixed strategies - pure AI adaptation
✅ Works on any broker, any market condition

🧠 Just upload your chart and watch the magic happen!

📝 Commands:
/help - Show help information
/stats - View analysis statistics
/version - Show bot version
"""
        await update.message.reply_text(welcome_msg)

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        help_msg = """
🆘 GHOST TRANSCENDENCE CORE HELP

📸 Send Chart Image: Upload any candlestick chart
🎯 Get Signal: Receive CALL/PUT with confidence
📊 Supported: Any broker, timeframe, market
⚡ AI Features: Dynamic strategy creation, pattern recognition

🤖 How to use:
1. Send me a screenshot of your trading chart
2. Wait for AI analysis (30-60 seconds)
3. Receive trading signal with confidence level

📝 Commands:
/start - Welcome message
/help - This help message
/stats - Bot usage statistics
/version - Bot version information

🔥 Ghost Transcendence Features:
👻 Manipulation Resistant
🧠 Infinite Learning
⚡ Dynamic Strategies
🎯 No-Loss Logic
"""
        await update.message.reply_text(help_msg)

    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stats command"""
        try:
            # Try to get stats from the main API
            response = requests.get(f"{self.api_url}/api/stats", timeout=5)
            if response.status_code == 200:
                stats = response.json()
                total_analyses = stats.get('total_analyses', 0)
            else:
                total_analyses = "Unknown"
        except:
            total_analyses = "API Unavailable"
            
        stats_msg = f"""
📊 GHOST TRANSCENDENCE CORE STATS

🧠 Total Analyses: {total_analyses}
⚡ Version: ∞ vX
🎯 AI Personality: GHOST TRANSCENDENCE CORE
📈 Confidence Threshold: 75.0%
👻 Ghost Mode: ACTIVE
⚡ Manipulation Resistance: MAXIMUM

The AI is constantly learning and evolving!

🔥 Features Active:
✅ Dynamic Strategy Creation
✅ Pattern Recognition  
✅ Manipulation Detection
✅ Broker Adaptation
✅ Infinite Learning
"""
        await update.message.reply_text(stats_msg)

    async def version_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /version command"""
        version_msg = """
🔥 GHOST TRANSCENDENCE CORE ∞ vX

🧠 Ultimate AI Trading Bot
⚡ No-Loss Logic Builder
🎯 Dynamic Strategy Creation
📊 Works on Any Market Condition
👻 Manipulation Resistant
🌍 Universal Broker Support

🚀 Core Technologies:
• OpenCV + HSV Analysis
• Advanced Pattern Recognition
• Dynamic AI Strategy Creation
• Manipulation Detection
• Real-time Learning

Built with ❤️ for traders who refuse to accept losses.
"""
        await update.message.reply_text(version_msg)

    async def handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle photo messages (chart analysis)"""
        try:
            # Send processing message
            processing_msg = await update.message.reply_text(
                "🧠 GHOST TRANSCENDENCE CORE ACTIVATED\n"
                "⚡ Analyzing chart with infinite intelligence...\n"
                "👻 Manipulation resistance: MAXIMUM\n"
                "🎯 Creating dynamic strategy..."
            )
            
            # Get the highest resolution photo
            photo = update.message.photo[-1]
            
            # Download image
            photo_file = await photo.get_file()
            image_data = await photo_file.download_as_bytearray()
            
            # Prepare files for API request
            files = {'chart_image': ('chart.jpg', io.BytesIO(image_data), 'image/jpeg')}
            
            # Send to analysis API
            try:
                response = requests.post(
                    f"{self.api_url}/analyze",
                    files=files,
                    timeout=120  # 2 minutes timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Format response
                    if result.get('error'):
                        response_text = f"❌ Analysis failed: {result['error']}"
                    else:
                        signal_type = result.get('signal', 'NO SIGNAL')
                        confidence = result.get('confidence', 0)
                        timeframe = result.get('timeframe', '1M')
                        time_target = result.get('time_target', 'Next candle')
                        reasoning = result.get('reasoning', 'Advanced AI analysis completed.')
                        
                        # Create signal emoji
                        signal_emoji = "🚀" if signal_type == "CALL" else "📉" if signal_type == "PUT" else "⏸️"
                        
                        response_text = f"""
🔥 GHOST TRANSCENDENCE CORE ∞ vX

{signal_emoji} SIGNAL: **{signal_type}**
⏰ TIMEFRAME: {timeframe}
🎯 TARGET: {time_target}
📈 CONFIDENCE: {confidence:.1f}%

🧠 AI REASONING:
{reasoning}

⚡ This signal was generated using dynamic AI strategy - no fixed rules, pure intelligence adaptation.

👻 Ghost Features Active:
✅ Manipulation Resistance
✅ Broker Trap Detection  
✅ Fake Signal Immunity
✅ Adaptive Evolution
"""
                else:
                    response_text = f"❌ API Error: {response.status_code} - {response.text}"
                    
            except requests.exceptions.Timeout:
                response_text = "⏰ Analysis timeout. The chart might be too complex or servers are busy. Please try again."
            except requests.exceptions.ConnectionError:
                response_text = "🔌 Connection error. The analysis server might be down. Please try again later."
            except Exception as e:
                response_text = f"❌ Error during analysis: {str(e)}"
            
            # Delete processing message and send result
            try:
                await processing_msg.delete()
            except:
                pass  # Ignore if message already deleted
                
            await update.message.reply_text(response_text, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Chart analysis error: {str(e)}")
            try:
                await update.message.reply_text(
                    f"❌ Error processing chart: {str(e)}\n\n"
                    "Please try again with a clear chart screenshot."
                )
            except:
                pass

    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle text messages"""
        text = update.message.text.lower()
        
        # Check for keywords
        if any(word in text for word in ['chart', 'signal', 'analysis', 'help', 'trade']):
            await update.message.reply_text(
                "📸 Please send me a chart screenshot to analyze!\n\n"
                "🎯 I can analyze charts from any broker:\n"
                "• MetaTrader 4/5\n"
                "• TradingView\n"
                "• IQ Option\n"
                "• Binance\n"
                "• Any mobile trading app\n\n"
                "📝 Use /help for more information."
            )
        elif any(word in text for word in ['hello', 'hi', 'hey', 'start']):
            await update.message.reply_text(
                "👋 Hello! I'm the Ghost Transcendence Core AI!\n\n"
                "🔥 Send me a candlestick chart screenshot and I'll provide:\n"
                "• CALL/PUT signals\n"
                "• Confidence levels\n"
                "• Market analysis\n"
                "• Risk assessment\n\n"
                "📝 Use /start for full welcome message."
            )
        else:
            await update.message.reply_text(
                "🤖 I'm the Ghost Transcendence Core AI!\n\n"
                "📸 Send me a chart screenshot for analysis\n"
                "📝 Use /help to see all commands\n\n"
                "🎯 I specialize in:\n"
                "• Chart pattern recognition\n"
                "• Market manipulation detection\n"
                "• Dynamic strategy creation\n"
                "• High-confidence signal generation"
            )

    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE):
        """Handle errors"""
        logger.error(f"Exception while handling an update: {context.error}")
        
        # Try to send error message to user
        if update and hasattr(update, 'message') and update.message:
            try:
                await update.message.reply_text(
                    "❌ An error occurred while processing your request.\n"
                    "Please try again or contact support if the problem persists."
                )
            except:
                pass

    def setup_handlers(self):
        """Setup bot handlers"""
        # Command handlers
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("stats", self.stats_command))
        self.application.add_handler(CommandHandler("version", self.version_command))
        
        # Message handlers
        self.application.add_handler(MessageHandler(filters.PHOTO, self.handle_photo))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text))
        
        # Error handler
        self.application.add_error_handler(self.error_handler)

    def run(self):
        """Run the bot"""
        try:
            # Create application
            self.application = Application.builder().token(self.token).build()
            
            # Setup handlers
            self.setup_handlers()
            
            # Start the bot
            logger.info("📱 Ghost Transcendence Core Telegram Bot starting...")
            self.application.run_polling(
                drop_pending_updates=True,
                allowed_updates=Update.ALL_TYPES
            )
            
        except Exception as e:
            logger.error(f"Bot startup error: {str(e)}")

def main():
    """Main function"""
    # Get token from environment
    token = os.environ.get('TELEGRAM_BOT_TOKEN', '7604218758:AAHJj2zMDTfVwyJHpLClVCDzukNr2Psj-38')
    api_url = os.environ.get('API_URL', 'http://localhost:5000')
    
    if not token or token == 'your_bot_token_here':
        logger.error("❌ TELEGRAM_BOT_TOKEN not configured!")
        return
    
    # Create and run bot
    bot = GhostTelegramBot(token, api_url)
    bot.run()

if __name__ == '__main__':
    main()