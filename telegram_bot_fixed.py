#!/usr/bin/env python3
"""
COSMIC AI Telegram Bot - Fixed Version
Robust Telegram bot with comprehensive error handling
"""

import os
import sys
import logging
import requests
import traceback
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from telegram import Update
    from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
    from config import Config
    from logic.ai_engine import CosmicAIEngine
    from telegram_bot import TelegramSignalBot
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("💡 Make sure all dependencies are installed:")
    print("   pip install python-telegram-bot")
    sys.exit(1)

# Enhanced logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('telegram_bot.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class RobustCosmicBot:
    def __init__(self):
        """Initialize the bot with error handling"""
        try:
            self.bot_token = Config.TELEGRAM_BOT_TOKEN
            logger.info(f"🤖 Bot token configured: {self.bot_token[:10]}...")
            
            # Initialize AI engine with error handling
            try:
                self.ai_engine = CosmicAIEngine()
                logger.info("✅ AI Engine initialized successfully")
            except Exception as e:
                logger.error(f"❌ Failed to initialize AI Engine: {e}")
                raise
            
            # Initialize signal bot
            try:
                self.signal_bot = TelegramSignalBot()
                logger.info("✅ Signal bot initialized successfully")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Signal bot: {e}")
                raise
                
            self.application = None
            logger.info("✅ RobustCosmicBot initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize bot: {e}")
            raise

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command"""
        try:
            user_name = update.effective_user.first_name or "User"
            logger.info(f"👤 /start command from {user_name}")
            
            welcome_message = f"""🧠 **COSMIC AI Binary Signal Bot**

👋 Hello {user_name}! I'm ready to analyze your charts!

📸 **How to use:**
1. Send me a candlestick chart screenshot
2. I'll analyze it with advanced AI
3. Get CALL/PUT/NO TRADE signals instantly

📊 **Supported formats:** JPG, PNG, WEBP
🎯 **Minimum confidence:** 85%
⏰ **Analysis time:** ~5-10 seconds

💫 **Ready!** Send me a chart screenshot to get started!

🔧 **Commands:**
/help - Show help
/status - Check bot status
            """
            
            await update.message.reply_text(
                welcome_message,
                parse_mode='Markdown'
            )
            
        except Exception as e:
            logger.error(f"❌ Error in start_command: {e}")
            await self._send_error_message(update, "Failed to process /start command")

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help command"""
        try:
            logger.info(f"👤 /help command from {update.effective_user.first_name}")
            
            help_text = """🧠 **COSMIC AI Help**

🔧 **Commands:**
/start - Welcome message
/help - This help message  
/status - Check bot status

📸 **How to use:**
• Send any candlestick chart screenshot
• Supports all brokers (Quotex, Binomo, IQ Option, etc.)
• Get instant AI analysis
• Only high-confidence signals (85%+)

⚡ **AI Features:**
• Pattern recognition (Engulfing, Doji, Hammer)
• Market psychology analysis
• Support/resistance detection
• Momentum and trend analysis

🎯 **Simply send a chart image to analyze!**
            """
            
            await update.message.reply_text(help_text, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"❌ Error in help_command: {e}")
            await self._send_error_message(update, "Failed to process /help command")

    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /status command"""
        try:
            logger.info(f"👤 /status command from {update.effective_user.first_name}")
            
            current_time = datetime.now().strftime('%H:%M:%S')
            status_message = f"""📊 **COSMIC AI STATUS**

✅ **System:** ONLINE
🧠 **AI Engine:** COSMIC AI v2.0  
🕒 **Time:** {current_time} (UTC+{Config.TIMEZONE_OFFSET})
🎯 **Confidence:** {Config.CONFIDENCE_THRESHOLD}%+
📈 **Status:** Ready for Analysis

🚀 Send me a chart screenshot!
            """
            
            await update.message.reply_text(status_message, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"❌ Error in status_command: {e}")
            await self._send_error_message(update, "Failed to process /status command")

    async def handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle photo uploads with comprehensive error handling"""
        user_name = update.effective_user.first_name or "User"
        logger.info(f"📸 Photo received from {user_name}")
        
        processing_msg = None
        try:
            # Send processing message
            processing_msg = await update.message.reply_text(
                "🧠 **COSMIC AI ANALYZING...**\n\n"
                "🔍 Processing your chart\n" 
                "⚡ Detecting patterns\n"
                "📊 Analyzing psychology\n\n"
                "_Please wait 5-10 seconds..._",
                parse_mode='Markdown'
            )
            
            # Get photo
            if not update.message.photo:
                raise ValueError("No photo found in message")
                
            photo = update.message.photo[-1]  # Get largest resolution
            logger.info(f"📱 Photo file_id: {photo.file_id}, size: {photo.file_size}")
            
            # Download photo
            try:
                file = await context.bot.get_file(photo.file_id)
                response = requests.get(file.file_path, timeout=30)
                response.raise_for_status()
                image_data = response.content
                logger.info(f"⬇️ Downloaded image: {len(image_data)} bytes")
                
            except Exception as e:
                logger.error(f"❌ Failed to download image: {e}")
                raise ValueError(f"Failed to download image: {e}")
            
            # Validate image
            if len(image_data) < 1000:  # Too small
                raise ValueError("Image too small or corrupted")
            
            if len(image_data) > 20 * 1024 * 1024:  # Too large
                raise ValueError("Image too large (max 20MB)")
            
            # Analyze with AI
            try:
                logger.info("🧠 Starting AI analysis...")
                analysis_result = self.ai_engine.analyze_chart(image_data)
                logger.info(f"✅ Analysis complete: {analysis_result['signal']} ({analysis_result['confidence']}%)")
                
            except Exception as e:
                logger.error(f"❌ AI analysis failed: {e}")
                raise ValueError(f"AI analysis failed: {e}")
            
            # Format response
            response_message = self._format_response(analysis_result, user_name)
            
            # Update message with results
            await processing_msg.edit_text(
                response_message,
                parse_mode='Markdown'
            )
            
            # Send to group if high confidence
            if (analysis_result['confidence'] >= Config.CONFIDENCE_THRESHOLD and 
                analysis_result['signal'] != 'NO TRADE'):
                try:
                    self.signal_bot.send_signal(analysis_result)
                    logger.info("📢 Signal sent to group")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to send to group: {e}")
            
        except Exception as e:
            logger.error(f"❌ Photo processing error: {e}")
            error_message = f"❌ **Analysis Failed**\n\n"
            
            if "download" in str(e).lower():
                error_message += "Failed to download image. Please try:\n• Using a smaller image\n• Checking your internet connection"
            elif "ai analysis" in str(e).lower():
                error_message += "AI analysis failed. Please try:\n• Using a clearer screenshot\n• Ensuring chart shows candlesticks\n• Using JPG/PNG format"
            else:
                error_message += f"Error: {str(e)}\n\nPlease try:\n• Using a clear chart screenshot\n• Checking image format (JPG/PNG/WEBP)\n• Ensuring good image quality"
            
            error_message += "\n\n🔄 **Try sending another chart!**"
            
            try:
                if processing_msg:
                    await processing_msg.edit_text(error_message, parse_mode='Markdown')
                else:
                    await update.message.reply_text(error_message, parse_mode='Markdown')
            except Exception as edit_error:
                logger.error(f"❌ Failed to send error message: {edit_error}")

    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle text messages"""
        try:
            await update.message.reply_text(
                "📸 **Please send a chart screenshot**\n\n"
                "I analyze candlestick chart images from:\n"
                "• Quotex\n"
                "• Binomo\n" 
                "• Pocket Option\n"
                "• IQ Option\n"
                "• Any other broker\n\n"
                "🎯 Use /help for more info",
                parse_mode='Markdown'
            )
        except Exception as e:
            logger.error(f"❌ Error handling text: {e}")

    def _format_response(self, result: Dict, user_name: str) -> str:
        """Format analysis response"""
        try:
            # Current time
            current_time = datetime.now(timezone.utc) + timedelta(hours=Config.TIMEZONE_OFFSET)
            time_str = current_time.strftime("%H:%M")
            
            # Emojis
            signal_emojis = {'CALL': '📈', 'PUT': '📉', 'NO TRADE': '⏸️'}
            confidence = result['confidence']
            
            if confidence >= 95:
                conf_emoji = '🔥'
            elif confidence >= 90:
                conf_emoji = '🚀'
            elif confidence >= 85:
                conf_emoji = '🔒'
            else:
                conf_emoji = '⚡'
            
            # Build message
            lines = [
                f"🧠 **COSMIC AI ANALYSIS** for {user_name}",
                "",
                f"🕒 **Time:** {time_str} (UTC+{Config.TIMEZONE_OFFSET})",
                f"{signal_emojis.get(result['signal'], '❓')} **Signal:** `{result['signal']}`",
                f"{conf_emoji} **Confidence:** `{confidence}%`",
                f"📊 **Reason:** {result['reason']}",
            ]
            
            # Add details if available
            if result.get('analysis_details'):
                details = result['analysis_details']
                lines.extend(["", "📋 **Technical Details:**"])
                
                if details.get('momentum'):
                    mom = details['momentum']
                    mom_emoji = '🚀' if mom['direction'] == 'bullish' else '🔻'
                    lines.append(f"{mom_emoji} **Momentum:** {mom['direction'].title()}")
                
                if details.get('patterns'):
                    patterns = ', '.join(details['patterns']).replace('_', ' ').title()
                    lines.append(f"🎯 **Patterns:** {patterns}")
            
            # Confidence message
            if confidence >= Config.CONFIDENCE_THRESHOLD:
                if result['signal'] != 'NO TRADE':
                    lines.extend(["", "✅ **HIGH CONFIDENCE SIGNAL!**"])
                else:
                    lines.extend(["", "⚠️ **NO TRADE RECOMMENDED**"])
            else:
                lines.extend(["", "📊 **ANALYSIS ONLY** (Below threshold)"])
            
            lines.extend(["", "⚡ *Send another chart for more analysis!*"])
            
            return "\n".join(lines)
            
        except Exception as e:
            logger.error(f"❌ Error formatting response: {e}")
            return f"✅ **Analysis Complete**\n\nSignal: {result.get('signal', 'ERROR')}\nConfidence: {result.get('confidence', 0)}%"

    async def _send_error_message(self, update: Update, message: str):
        """Send error message safely"""
        try:
            await update.message.reply_text(f"❌ {message}\n\n🔄 Please try again or use /help")
        except Exception as e:
            logger.error(f"❌ Failed to send error message: {e}")

    def run(self):
        """Run the bot with comprehensive error handling"""
        try:
            logger.info("🚀 Starting COSMIC AI Telegram Bot...")
            
            # Create application
            self.application = Application.builder().token(self.bot_token).build()
            
            # Add handlers
            self.application.add_handler(CommandHandler("start", self.start_command))
            self.application.add_handler(CommandHandler("help", self.help_command))
            self.application.add_handler(CommandHandler("status", self.status_command))
            self.application.add_handler(MessageHandler(filters.PHOTO, self.handle_photo))
            self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text))
            
            logger.info("✅ Handlers configured")
            
            # Start bot
            logger.info("🔄 Starting polling...")
            self.application.run_polling(drop_pending_updates=True)
            
        except KeyboardInterrupt:
            logger.info("🛑 Bot stopped by user (Ctrl+C)")
        except Exception as e:
            logger.error(f"❌ Bot crashed: {e}")
            logger.error(traceback.format_exc())
            raise

def main():
    """Main function with error handling"""
    try:
        print("🧠 COSMIC AI Telegram Bot - Fixed Version")
        print("=" * 50)
        print(f"🤖 Bot: @Itis_ghost_bot")
        print(f"🚀 Token: {Config.TELEGRAM_BOT_TOKEN[:15]}...")
        print("📱 Ready for chart analysis!")
        print("=" * 50)
        
        bot = RobustCosmicBot()
        bot.run()
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Check internet connection")
        print("2. Verify bot token in config.py")
        print("3. Ensure all dependencies installed")
        print("4. Check logs in telegram_bot.log")
        sys.exit(1)

if __name__ == "__main__":
    main()