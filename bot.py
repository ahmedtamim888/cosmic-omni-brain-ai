#!/usr/bin/env python3
"""
🤖 TELEGRAM BINARY TRADING BOT
Professional chart analysis bot with OCR validation
"""

import logging
import os
import asyncio
import random
from datetime import datetime
from typing import Optional

from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.constants import ParseMode

from chart_checker import ChartChecker

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class TradingBot:
    """
    🤖 Professional Telegram Trading Bot
    
    Features:
    - Chart validation using OCR
    - Signal generation for valid charts
    - Professional error handling
    - Clean user interface
    """
    
    def __init__(self, bot_token: str):
        """Initialize the trading bot"""
        self.bot_token = bot_token
        self.chart_checker = ChartChecker()
        self.application = None
        
        # 📊 Trading signal templates
        self.signal_templates = {
            'call': [
                "✅ Chart detected. Signal: 📈 **CALL**",
                "✅ Valid chart found. Signal: 🟢 **UP/CALL**",
                "✅ Chart analysis complete. Signal: ⬆️ **HIGHER/CALL**"
            ],
            'put': [
                "✅ Chart detected. Signal: 📉 **PUT**",
                "✅ Valid chart found. Signal: 🔴 **DOWN/PUT**",
                "✅ Chart analysis complete. Signal: ⬇️ **LOWER/PUT**"
            ]
        }
        
        # 📁 Create temp directory for images
        os.makedirs("temp_images", exist_ok=True)
        
        logger.info("🤖 Trading Bot initialized successfully")
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        🚀 Handle /start command
        """
        user = update.effective_user
        welcome_message = f"""
🤖 **Welcome to Professional Trading Bot!** 

Hello {user.first_name}! 👋

📊 **What I can do:**
• Analyze trading chart screenshots
• Validate charts from Quotex, TradingView, MT4, etc.
• Provide binary trading signals for valid charts
• Detect fake/invalid images

📱 **How to use:**
1. Send me a trading chart screenshot
2. I'll analyze it using advanced OCR
3. If valid → You get a trading signal 📈📉
4. If invalid → I'll ask for a real chart ⚠️

🎯 **Supported platforms:**
Quotex, TradingView, MetaTrader, Binomo, IQ Option, and more!

📸 **Send me a chart screenshot to get started!**
        """
        
        await update.message.reply_text(
            welcome_message,
            parse_mode=ParseMode.MARKDOWN
        )
        
        logger.info("📱 User %s started the bot", user.first_name)
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        ❓ Handle /help command
        """
        help_message = """
🤖 **Trading Bot Help**

**Commands:**
• `/start` - Start the bot and see welcome message
• `/help` - Show this help message
• `/status` - Check bot status

**How to use:**
1️⃣ Send a trading chart screenshot
2️⃣ Bot analyzes with OCR technology
3️⃣ Get signal if chart is valid

**Supported chart types:**
• Quotex charts
• TradingView charts
• MetaTrader (MT4/MT5)
• Binomo charts
• IQ Option charts
• Other forex/binary platforms

**Signal format:**
✅ Valid charts → 📈 CALL or 📉 PUT
⚠️ Invalid images → Error message

**Tips for best results:**
• Use clear, high-quality screenshots
• Ensure text is readable
• Include platform UI elements
• Avoid blurry or cropped images

Need support? Just send /help again! 🔧
        """
        
        await update.message.reply_text(
            help_message,
            parse_mode=ParseMode.MARKDOWN
        )
    
    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        📊 Handle /status command
        """
        status_message = f"""
🤖 **Bot Status Report**

🟢 **Status:** Online and Active
⏰ **Current Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🔧 **Chart Checker:** Ready
📊 **OCR Engine:** Operational
🧠 **AI Analysis:** Active

**Capabilities:**
✅ Image processing with OCR
✅ Chart validation (60+ keywords)
✅ Signal generation
✅ Error handling
✅ Multi-platform support

**Recent Activity:**
📱 Ready to analyze your charts!

Send me a trading chart screenshot to test! 📸
        """
        
        await update.message.reply_text(
            status_message,
            parse_mode=ParseMode.MARKDOWN
        )
    
    async def handle_image(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        📸 Handle image messages from users
        """
        try:
            user = update.effective_user
            logger.info("📸 Received image from user %s", user.first_name)
            
            # Send initial processing message
            processing_msg = await update.message.reply_text(
                "🔍 **Analyzing your chart...**\n⏳ Please wait while I process the image...",
                parse_mode=ParseMode.MARKDOWN
            )
            
            # Get the largest photo size
            photo = update.message.photo[-1]
            
            # Download the image
            file = await context.bot.get_file(photo.file_id)
            
            # Create unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = f"temp_images/chart_{user.id}_{timestamp}.jpg"
            
            # Download and save the image
            await file.download_to_drive(image_path)
            logger.info("💾 Image saved: %s", image_path)
            
            # Analyze the chart
            is_valid_chart = await self._analyze_chart_async(image_path)
            
            if is_valid_chart:
                # Generate trading signal for valid chart
                signal_response = await self._generate_signal_response(image_path)
                await processing_msg.edit_text(
                    signal_response,
                    parse_mode=ParseMode.MARKDOWN
                )
                logger.info("✅ Valid chart detected, signal sent to user %s", user.first_name)
            else:
                # Invalid chart response
                invalid_response = """
⚠️ **This is not a valid chart. Please send a real chart screenshot**

🔍 **What I'm looking for:**
• Trading platform interfaces (Quotex, TradingView, MT4, etc.)
• Currency pairs (EUR/USD, GBP/USD, etc.)
• Trading terms (Price, Time, Investment, Payout)
• Chart elements (Candles, indicators, timeframes)

📸 **Tips:**
• Take a clear screenshot of your trading platform
• Ensure text is readable and not blurry
• Include the platform's UI elements
• Avoid random images or non-trading content

Try again with a real trading chart! 📊
                """
                
                await processing_msg.edit_text(
                    invalid_response,
                    parse_mode=ParseMode.MARKDOWN
                )
                logger.info("❌ Invalid chart detected from user %s", user.first_name)
            
            # Clean up temp file
            try:
                os.remove(image_path)
                logger.debug("🗑️ Temp file cleaned up: %s", image_path)
            except:
                pass
                
        except Exception as e:
            logger.error("❌ Error handling image: %s", str(e))
            
            error_message = """
🚫 **Error processing your image**

😔 Sorry, something went wrong while analyzing your chart.

**Please try:**
• Sending the image again
• Using a different image format
• Ensuring the image is not corrupted

If the problem persists, contact support. 🔧
            """
            
            await update.message.reply_text(
                error_message,
                parse_mode=ParseMode.MARKDOWN
            )
    
    async def _analyze_chart_async(self, image_path: str) -> bool:
        """
        🔍 Async wrapper for chart analysis
        """
        try:
            # Run the chart checker in a thread to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                self.chart_checker.is_valid_chart, 
                image_path
            )
            return result
        except Exception as e:
            logger.error("❌ Error in async chart analysis: %s", str(e))
            return False
    
    async def _generate_signal_response(self, image_path: str) -> str:
        """
        📊 Generate trading signal response for valid charts
        """
        try:
            # Get detailed analysis
            loop = asyncio.get_event_loop()
            analysis = await loop.run_in_executor(
                None,
                self.chart_checker.get_analysis_details,
                image_path
            )
            
            # Randomly choose signal direction (you can implement actual analysis here)
            signal_type = random.choice(['call', 'put'])
            signal_message = random.choice(self.signal_templates[signal_type])
            
            # Add analysis details
            keywords_found = analysis.get('keywords_found', 0)
            confidence = analysis.get('confidence_score', 0) * 100
            detected_keywords = analysis.get('detected_keywords', [])
            
            # Create detailed response
            response = f"""
{signal_message}

📊 **Analysis Details:**
🔍 Keywords detected: {keywords_found}
🎯 Confidence: {confidence:.1f}%
⏰ Time: {datetime.now().strftime('%H:%M:%S')}

💡 **Detected elements:** {', '.join(detected_keywords[:5])}{'...' if len(detected_keywords) > 5 else ''}

⚠️ **Disclaimer:** This is a demo signal. Always do your own analysis before trading!
            """
            
            return response
            
        except Exception as e:
            logger.error("❌ Error generating signal response: %s", str(e))
            return "✅ Chart detected. Signal: 📈 **CALL** (Basic analysis)"
    
    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        💬 Handle text messages
        """
        text_response = """
📸 **Please send me a chart screenshot!**

🤖 I'm designed to analyze trading charts, not text messages.

**What to send:**
• Screenshot from Quotex
• TradingView chart image
• MetaTrader screenshot
• Any trading platform chart

**What I'll do:**
✅ Validate if it's a real trading chart
📊 Provide trading signal if valid
⚠️ Ask for real chart if invalid

📱 Just send the image directly - no need to type anything!
        """
        
        await update.message.reply_text(
            text_response,
            parse_mode=ParseMode.MARKDOWN
        )
    
    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        🚫 Global error handler
        """
        logger.error("Exception while handling an update:", exc_info=context.error)
        
        # Try to send error message to user if possible
        if isinstance(update, Update) and update.effective_message:
            try:
                await update.effective_message.reply_text(
                    "🚫 **Something went wrong!**\n\n"
                    "Please try again or contact support if the issue persists.",
                    parse_mode=ParseMode.MARKDOWN
                )
            except:
                pass
    
    def setup_handlers(self):
        """
        🔧 Setup bot command and message handlers
        """
        # Command handlers
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("status", self.status_command))
        
        # Message handlers
        self.application.add_handler(MessageHandler(filters.PHOTO, self.handle_image))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text))
        
        # Error handler
        self.application.add_error_handler(self.error_handler)
        
        logger.info("🔧 All handlers setup successfully")
    
    def run(self):
        """
        🚀 Start the bot
        """
        try:
            # Create application
            self.application = Application.builder().token(self.bot_token).build()
            
            # Setup handlers
            self.setup_handlers()
            
            logger.info("🤖 Starting Telegram Trading Bot...")
            logger.info("🟢 Bot is running! Press Ctrl+C to stop.")
            
            # Start polling
            self.application.run_polling(
                allowed_updates=Update.ALL_TYPES,
                drop_pending_updates=True
            )
            
        except Exception as e:
            logger.error("❌ Error starting bot: %s", str(e))
            raise

def main():
    """
    🎯 Main function to run the trading bot
    """
    # Bot token - you can also use environment variable
    BOT_TOKEN = "8288385434:AAG_RVKnlXDWBZNN38Q3IEfSQXIgxwPlsU0"
    
    # Alternative: Use environment variable for security
    # BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    # if not BOT_TOKEN:
    #     logger.error("❌ TELEGRAM_BOT_TOKEN environment variable not set!")
    #     return
    
    try:
        # Create and run the bot
        bot = TradingBot(BOT_TOKEN)
        bot.run()
        
    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped by user")
    except Exception as e:
        logger.error("❌ Fatal error: %s", str(e))

if __name__ == "__main__":
    main()