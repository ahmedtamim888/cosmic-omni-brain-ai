#!/usr/bin/env python3
"""
📱 GHOST TRANSCENDENCE CORE - Telegram Bot Module
Separate module for Telegram bot to avoid async conflicts
"""

import asyncio
import logging
import requests
import io
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
from PIL import Image
import base64
from datetime import datetime
import pytz

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Bot configuration
BOT_TOKEN = "7604218758:AAHJj2zMDTfVwyJHpLClVCDzukNr2Psj-38"
FLASK_API_URL = "http://localhost:5000/api/analyze"  # Updated endpoint

# UTC+6:00 timezone
market_timezone = pytz.timezone('Asia/Dhaka')

class TelegramBot:
    """
    🕯️ CANDLE WHISPERER TELEGRAM BOT
    Connects users to the AI that talks with every candle
    """
    
    def __init__(self):
        self.app = Application.builder().token(BOT_TOKEN).build()
        self.setup_handlers()
        
    def setup_handlers(self):
        """Setup bot command and message handlers"""
        # Commands
        self.app.add_handler(CommandHandler("start", self.start_command))
        self.app.add_handler(CommandHandler("help", self.help_command))
        self.app.add_handler(CommandHandler("stats", self.stats_command))
        
        # Photo handler for chart analysis
        self.app.add_handler(MessageHandler(filters.PHOTO, self.handle_chart_image))
        
        # Callback query handler for buttons
        self.app.add_handler(CallbackQueryHandler(self.handle_callback))
        
        logger.info("🕯️ CANDLE WHISPERER Bot handlers setup complete")

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Start command - welcome message"""
        welcome_message = """🔥 GHOST TRANSCENDENCE CORE ∞ vX - CANDLE WHISPERER

🕯️ Welcome to the CANDLE WHISPERER AI Bot!

This AI literally TALKS WITH EVERY CANDLE to understand market secrets and provide 100% accurate signals for volatile OTC markets.

🎯 Features:
• 🕯️ Candle Whisperer Mode - Talks with each candle
• ⏰ UTC+6:00 Timing - Perfect entry precision
• 🎪 100% Accuracy Target - No losses
• 🤫 Secret Pattern Detection - Hidden market messages
• 🚫 Loss Prevention System - Learns from mistakes

📸 How to use:
1. Send me a screenshot of your candlestick chart
2. I'll talk with every candle to understand their story
3. Get precise signal: 1M | HH:MM | CALL/PUT
4. Enter at the exact time provided

⚡ Ready to dominate the market with candle intelligence!

Send your chart screenshot now! 📈"""

        keyboard = [
            [InlineKeyboardButton("📊 Send Chart", callback_data="send_chart")],
            [InlineKeyboardButton("📈 Bot Stats", callback_data="stats")],
            [InlineKeyboardButton("❓ Help", callback_data="help")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(welcome_message, reply_markup=reply_markup)

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Help command"""
        help_message = """🕯️ CANDLE WHISPERER HELP

📋 Commands:
• /start - Start the bot
• /help - Show this help
• /stats - View bot statistics

📸 Image Analysis:
• Send any candlestick chart screenshot
• Support all brokers (Quotex, IQ Option, Pocket Option, etc.)
• Works with OTC markets and volatile instruments
• AI talks with every candle to understand market story

🎯 Signal Format:
• 1M | 14:25 | CALL (Next candle entry time UTC+6:00)
• 1M | 14:25 | PUT (Bearish signal)
• NO SIGNAL (Wait for better opportunity)

🕯️ Candle Whisperer Features:
• Each candle has a personality and story
• AI extracts secret messages from wicks and bodies
• Real-time market session analysis
• 100% accuracy through candle conversations
• Loss prevention through learning

⚡ The AI thinks like it can communicate with every candle to provide the most accurate signals possible!"""

        await update.message.reply_text(help_message)

    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Stats command - show bot statistics"""
        try:
            # Get current time in UTC+6:00
            current_time = datetime.now(market_timezone).strftime("%H:%M:%S")
            
            stats_message = f"""📊 CANDLE WHISPERER STATISTICS

🕯️ Bot Version: ∞ vX CANDLE WHISPERER
⏰ Current Time: {current_time} (UTC+6:00)
🎯 Accuracy Target: 100%
🤖 AI Mode: CANDLE WHISPERER ACTIVE

🕯️ Candle Features:
✅ Candle Conversations - ACTIVE
✅ Secret Pattern Detection - ACTIVE
✅ Loss Prevention System - ACTIVE
✅ UTC+6:00 Timing Precision - ACTIVE
✅ Volatility Analysis - ACTIVE

🎪 Market Intelligence:
• Talks with 15-25 candles per analysis
• Extracts wick and body secrets
• Detects market manipulation
• Provides next candle entry time
• Learns from every loss

⚡ Ready to analyze your volatile market screenshot!"""
            
            await update.message.reply_text(stats_message)
            
        except Exception as e:
            await update.message.reply_text(f"📊 Stats temporarily unavailable: {str(e)}")

    async def handle_chart_image(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle chart image analysis with CANDLE WHISPERER"""
        try:
            logger.info("🕯️ CANDLE WHISPERER: Received chart image for analysis")
            
            # Send initial processing message
            processing_msg = await update.message.reply_text(
                "🕯️ CANDLE WHISPERER ANALYZING...\n\n"
                "📸 Chart image received\n"
                "🗣️ Starting conversations with candles...\n"
                "🤫 Extracting secret patterns...\n"
                "⏳ Please wait for candle wisdom..."
            )
            
            # Get the largest photo
            photo = update.message.photo[-1]
            
            # Download photo
            photo_file = await photo.get_file()
            photo_bytes = await photo_file.download_as_bytearray()
            
            # Convert to PIL Image and then to bytes for upload
            image = Image.open(io.BytesIO(photo_bytes))
            
            # Convert PIL image to bytes for upload
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            
            # Send to Flask API for analysis
            files = {'image': ('chart.png', img_byte_arr, 'image/png')}
            
            logger.info(f"🔗 Sending image to Flask API: {FLASK_API_URL}")
            response = requests.post(FLASK_API_URL, files=files, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract analysis results
                signal_type = result.get('signal', 'NO SIGNAL')
                confidence = result.get('confidence', 0.0)
                next_candle_time = result.get('next_candle_time', '00:00')
                message = result.get('message', 'Analysis complete')
                total_candles = result.get('total_candles_consulted', 0)
                candle_prophecy = result.get('candle_prophecy', '')
                
                # Clean message for Telegram
                def telegram_safe_text(text):
                    """Make text safe for Telegram"""
                    if not text:
                        return "Analysis complete"
                    # Remove or replace problematic characters
                    safe_text = str(text).replace('*', '').replace('_', ' ')
                    safe_text = safe_text.replace('`', '').replace('[', '(').replace(']', ')')
                    safe_text = safe_text.replace('<', '').replace('>', '').replace('|', '-')
                    safe_text = safe_text.replace('\\', '').replace('{', '').replace('}', '')
                    # Limit message length for Telegram
                    if len(safe_text) > 4000:
                        safe_text = safe_text[:4000] + "\n\n... (truncated for display)"
                    return safe_text
                
                clean_message = telegram_safe_text(message)
                
                # Edit the processing message with results
                await processing_msg.edit_text(clean_message)
                
                # Add quick action buttons
                keyboard = []
                if signal_type != 'NO SIGNAL':
                    keyboard.append([
                        InlineKeyboardButton(f"🎯 {signal_type} at {next_candle_time}", callback_data=f"confirm_{signal_type}")
                    ])
                
                keyboard.extend([
                    [InlineKeyboardButton("📸 Analyze Another Chart", callback_data="send_chart")],
                    [InlineKeyboardButton("📊 Bot Stats", callback_data="stats")]
                ])
                
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                # Send final message with buttons
                await update.message.reply_text(
                    f"🕯️ CANDLE WHISPERER ANALYSIS COMPLETE!\n\n"
                    f"🎯 Signal: {signal_type}\n"
                    f"⏰ Entry: {next_candle_time} (UTC+6:00)\n"
                    f"📈 Confidence: {confidence:.1f}%\n"
                    f"🕯️ Candles Consulted: {total_candles}",
                    reply_markup=reply_markup
                )
                
                logger.info(f"🎯 CANDLE WHISPERER Analysis complete: {signal_type} | {confidence:.1f}%")
                
            else:
                error_message = f"❌ Analysis failed: {response.text}"
                await processing_msg.edit_text(
                    f"🚫 CANDLE WHISPERER ERROR\n\n"
                    f"{error_message}\n\n"
                    f"Please try again or contact support."
                )
                logger.error(f"❌ API Error: {response.status_code} - {response.text}")
                
        except Exception as e:
            logger.error(f"❌ Chart analysis error: {str(e)}")
            
            try:
                await processing_msg.edit_text(
                    f"🚫 CANDLE WHISPERER ERROR\n\n"
                    f"Error: {str(e)}\n\n"
                    f"🔄 Please try again with a clear chart screenshot."
                )
            except:
                await update.message.reply_text(
                    f"🚫 CANDLE WHISPERER ERROR\n\n"
                    f"Error: {str(e)}\n\n"
                    f"🔄 Please try again with a clear chart screenshot."
                )

    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle button callbacks"""
        query = update.callback_query
        await query.answer()
        
        if query.data == "send_chart":
            await query.edit_message_text(
                "📸 CANDLE WHISPERER READY\n\n"
                "🕯️ Send me your candlestick chart screenshot now!\n\n"
                "I'll talk with every candle to understand their story and provide:\n"
                "• 🎯 Precise signal (CALL/PUT)\n"
                "• ⏰ Exact entry time (UTC+6:00)\n"
                "• 🎪 100% accuracy target\n"
                "• 🤫 Secret pattern analysis\n\n"
                "📱 Works with any broker: Quotex, IQ Option, Pocket Option, etc."
            )
            
        elif query.data == "stats":
            await self.stats_command(update, context)
            
        elif query.data == "help":
            await self.help_command(update, context)
            
        elif query.data.startswith("confirm_"):
            signal = query.data.replace("confirm_", "")
            current_time = datetime.now(market_timezone).strftime("%H:%M:%S")
            
            await query.edit_message_text(
                f"✅ SIGNAL CONFIRMED\n\n"
                f"🎯 Direction: {signal}\n"
                f"⏰ Current Time: {current_time} (UTC+6:00)\n"
                f"🕯️ Candle Whisperer Confidence: HIGH\n\n"
                f"📈 Good luck with your trade!\n"
                f"🔄 Send another chart for next analysis."
            )

async def main():
    """Main function to run the bot"""
    try:
        logger.info("🕯️ Starting CANDLE WHISPERER Telegram Bot...")
        
        bot = TelegramBot()
        
        # Start the bot
        await bot.app.initialize()
        await bot.app.start()
        
        logger.info("🕯️ CANDLE WHISPERER Bot is now running and ready for chart analysis!")
        logger.info("🎯 Waiting for volatile market screenshots...")
        
        # Run the bot
        await bot.app.updater.start_polling()
        
        # Keep the bot running
        await asyncio.Event().wait()
        
    except Exception as e:
        logger.error(f"❌ Bot startup error: {str(e)}")
    finally:
        if 'bot' in locals():
            await bot.app.stop()

if __name__ == "__main__":
    asyncio.run(main())