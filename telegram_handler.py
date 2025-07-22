import asyncio
import logging
import io
import requests
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional

from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from config import Config
from logic.ai_engine import CosmicAIEngine
from telegram_bot import TelegramSignalBot

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

class CosmicTelegramBot:
    def __init__(self):
        self.bot_token = Config.TELEGRAM_BOT_TOKEN
        self.ai_engine = CosmicAIEngine()
        self.signal_bot = TelegramSignalBot()
        self.application = None
        
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command"""
        welcome_message = """🧠 **COSMIC AI Binary Signal Bot**

🚀 **Ready to analyze your charts!**

📸 **How to use:**
1. Send me a candlestick chart screenshot
2. I'll analyze it with advanced AI
3. Get CALL/PUT/NO TRADE signals instantly

📊 **Supported formats:** JPG, PNG, WEBP
🎯 **Minimum confidence:** 85%
⏰ **Analysis time:** ~5-10 seconds

💫 Send me a chart screenshot to get started!
        """
        
        await update.message.reply_text(
            welcome_message,
            parse_mode='Markdown',
            disable_web_page_preview=True
        )

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help command"""
        help_text = """🧠 **COSMIC AI Commands**

🔧 **Available Commands:**
/start - Start the bot and see welcome message
/help - Show this help message
/status - Check bot status

📸 **Using the Bot:**
• Send any candlestick chart screenshot
• Bot supports all brokers (Quotex, Binomo, etc.)
• Get instant AI-powered signals
• Only high-confidence signals (85%+) are sent

⚡ **Features:**
• Advanced pattern recognition
• Market psychology analysis
• Support/resistance detection
• Momentum and trend analysis
• Professional signal formatting

🎯 **Simply send a chart image to get started!**
        """
        
        await update.message.reply_text(
            help_text,
            parse_mode='Markdown',
            disable_web_page_preview=True
        )

    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /status command"""
        status_message = f"""📊 **COSMIC AI STATUS**

✅ **System Status:** ONLINE
🧠 **AI Engine:** COSMIC AI v2.0
🕒 **Current Time:** {datetime.now().strftime('%H:%M:%S')} (UTC+{Config.TIMEZONE_OFFSET})
🎯 **Confidence Threshold:** {Config.CONFIDENCE_THRESHOLD}%
📈 **Ready for Analysis:** YES

🚀 Send me a chart screenshot to analyze!
        """
        
        await update.message.reply_text(
            status_message,
            parse_mode='Markdown'
        )

    async def handle_image(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle image uploads and analyze them"""
        try:
            # Send processing message
            processing_msg = await update.message.reply_text(
                "🧠 **COSMIC AI ANALYZING...**\n\n"
                "🔍 Processing your chart\n"
                "⚡ Detecting patterns\n"
                "📊 Analyzing market psychology\n\n"
                "_Please wait 5-10 seconds..._",
                parse_mode='Markdown'
            )
            
            # Get the largest photo
            photo = update.message.photo[-1]
            
            # Download the photo
            file = await context.bot.get_file(photo.file_id)
            
            # Download image data
            response = requests.get(file.file_path)
            image_data = response.content
            
            logger.info(f"Processing image from user {update.effective_user.first_name}, size: {len(image_data)} bytes")
            
            # Analyze with AI engine
            analysis_result = self.ai_engine.analyze_chart(image_data)
            
            # Format the response
            response_message = self._format_analysis_response(analysis_result)
            
            # Edit the processing message with results
            await processing_msg.edit_text(
                response_message,
                parse_mode='Markdown',
                disable_web_page_preview=True
            )
            
            # If high confidence, also send to group (if different from current chat)
            if (analysis_result['confidence'] >= Config.CONFIDENCE_THRESHOLD and 
                analysis_result['signal'] != 'NO TRADE'):
                
                # Send to configured group if it's different from current chat
                if str(update.effective_chat.id) != Config.TELEGRAM_CHAT_ID:
                    self.signal_bot.send_signal(analysis_result)
            
            logger.info(f"Analysis completed: {analysis_result['signal']} ({analysis_result['confidence']}%)")
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            await update.message.reply_text(
                "❌ **Analysis Error**\n\n"
                "Sorry, I couldn't analyze your chart. Please try:\n"
                "• Using a clearer screenshot\n"
                "• Ensuring the chart shows candlesticks\n"
                "• Checking the image format (JPG/PNG/WEBP)\n\n"
                "🔄 Try sending another chart!",
                parse_mode='Markdown'
            )

    def _format_analysis_response(self, result: Dict) -> str:
        """Format analysis result for Telegram response"""
        
        # Get current time
        utc_time = datetime.now(timezone.utc)
        local_time = utc_time + timedelta(hours=Config.TIMEZONE_OFFSET)
        time_str = local_time.strftime("%H:%M")
        
        # Signal emojis
        signal_emojis = {
            'CALL': '📈',
            'PUT': '📉',
            'NO TRADE': '⏸️'
        }
        
        # Confidence emojis
        confidence = result['confidence']
        if confidence >= 95:
            confidence_emoji = '🔥'
        elif confidence >= 90:
            confidence_emoji = '🚀'
        elif confidence >= 85:
            confidence_emoji = '🔒'
        elif confidence >= 70:
            confidence_emoji = '⚡'
        else:
            confidence_emoji = '⚠️'
        
        # Build message
        message_lines = [
            "🧠 **COSMIC AI ANALYSIS COMPLETE**",
            "",
            f"🕒 **Time:** {time_str} (UTC+{Config.TIMEZONE_OFFSET})",
            f"{signal_emojis.get(result['signal'], '❓')} **Signal:** `{result['signal']}`",
            f"{confidence_emoji} **Confidence:** `{result['confidence']}%`",
            f"📊 **Reason:** {result['reason']}",
        ]
        
        # Add detailed analysis if available
        if 'analysis_details' in result and result['analysis_details']:
            details = result['analysis_details']
            message_lines.extend([
                "",
                "📋 **Technical Analysis:**"
            ])
            
            # Momentum
            if 'momentum' in details:
                momentum = details['momentum']
                momentum_emoji = '🚀' if momentum['direction'] == 'bullish' else '🔻' if momentum['direction'] == 'bearish' else '➡️'
                message_lines.append(f"{momentum_emoji} **Momentum:** {momentum['direction'].title()} ({momentum['strength']:.1f}%)")
            
            # Trend
            if 'trend' in details:
                trend = details['trend']
                trend_emoji = '📈' if trend['direction'] == 'uptrend' else '📉' if trend['direction'] == 'downtrend' else '↔️'
                message_lines.append(f"{trend_emoji} **Trend:** {trend['direction'].title()}")
            
            # Patterns
            if 'patterns' in details and details['patterns']:
                patterns_str = ', '.join(details['patterns']).replace('_', ' ').title()
                message_lines.append(f"🎯 **Patterns:** {patterns_str}")
            
            # Market position
            if 'support_resistance' in details:
                sr = details['support_resistance']
                position = sr['current_position'].replace('_', ' ').title()
                message_lines.append(f"📍 **Position:** {position}")
        
        # Add confidence interpretation
        if confidence >= Config.CONFIDENCE_THRESHOLD:
            if result['signal'] != 'NO TRADE':
                message_lines.extend([
                    "",
                    f"✅ **HIGH CONFIDENCE SIGNAL!**",
                    f"💡 This signal meets the {Config.CONFIDENCE_THRESHOLD}% threshold"
                ])
            else:
                message_lines.extend([
                    "",
                    f"⚠️ **NO TRADE RECOMMENDED**",
                    f"🛡️ Market conditions unclear"
                ])
        else:
            message_lines.extend([
                "",
                f"📊 **ANALYSIS ONLY**",
                f"⚠️ Below {Config.CONFIDENCE_THRESHOLD}% confidence threshold"
            ])
        
        # Footer
        message_lines.extend([
            "",
            "⚡ *Powered by COSMIC AI Engine*",
            "📸 Send another chart for more analysis!"
        ])
        
        return "\n".join(message_lines)

    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle text messages"""
        await update.message.reply_text(
            "📸 **Please send a chart screenshot**\n\n"
            "I can only analyze candlestick chart images.\n"
            "Send me a screenshot from any broker:\n"
            "• Quotex\n"
            "• Binomo\n"
            "• Pocket Option\n"
            "• IQ Option\n"
            "• And many more!\n\n"
            "🎯 Use /help for more information",
            parse_mode='Markdown'
        )

    def run_bot(self):
        """Run the Telegram bot using run_polling"""
        try:
            # Create application
            self.application = Application.builder().token(self.bot_token).build()
            
            # Add handlers
            self.application.add_handler(CommandHandler("start", self.start_command))
            self.application.add_handler(CommandHandler("help", self.help_command))
            self.application.add_handler(CommandHandler("status", self.status_command))
            self.application.add_handler(MessageHandler(filters.PHOTO, self.handle_image))
            self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text))
            
            logger.info("🚀 Starting COSMIC AI Telegram Bot...")
            
            # Run the bot
            self.application.run_polling(drop_pending_updates=True)
            
        except Exception as e:
            logger.error(f"❌ Failed to start bot: {str(e)}")
            raise

def main():
    """Main function to run the Telegram bot"""
    try:
        bot = CosmicTelegramBot()
        bot.run_bot()
        
    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped by user")
    except Exception as e:
        logger.error(f"❌ Bot crashed: {str(e)}")
        raise

if __name__ == "__main__":
    main()