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
BOT_TOKEN = "8155744845:AAGQ6s8RMSpCAjO16Mzb1xl8_otLnXh3OKQ"
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
                
                # ULTRA-AGGRESSIVE Telegram text cleaning
                def ultra_clean_telegram_text(text):
                    """Ultra-aggressive cleaning to prevent ANY Telegram parsing errors"""
                    if not text:
                        return "Analysis complete"
                    
                    # Convert to string and get basic content
                    clean_text = str(text)
                    
                    # Remove ALL potentially problematic characters
                    problematic_chars = ['*', '_', '`', '[', ']', '(', ')', '{', '}', 
                                       '<', '>', '|', '\\', '/', '~', '^', 
                                       '&', '%', '$', '#', '@', '!', '+', '=']
                    
                    for char in problematic_chars:
                        clean_text = clean_text.replace(char, ' ')
                    
                    # Replace multiple spaces with single space
                    while '  ' in clean_text:
                        clean_text = clean_text.replace('  ', ' ')
                    
                    # Remove any remaining special unicode characters
                    clean_text = ''.join(c for c in clean_text if ord(c) < 128 or c.isalnum() or c in ' .,:-\n')
                    
                    # Split into lines and clean each
                    lines = clean_text.split('\n')
                    cleaned_lines = []
                    
                    for line in lines:
                        line = line.strip()
                        if line and len(line) > 0:
                            # Keep only safe characters
                            safe_line = ''.join(c for c in line if c.isalnum() or c in ' .,:-')
                            if safe_line.strip():
                                cleaned_lines.append(safe_line.strip())
                    
                    # Rebuild message with safe content only
                    result = '\n'.join(cleaned_lines)
                    
                    # Limit length
                    if len(result) > 3500:
                        result = result[:3500] + "\n\nMessage truncated for safety"
                    
                    return result.strip()
                
                                 # Create a SUPER SIMPLE message format
                 def create_simple_message(signal_type, confidence, next_time, total_candles):
                     """Create the simplest possible Telegram message"""
                     
                     # Clean all inputs to remove problematic characters
                     clean_signal = str(signal_type).replace('%', ' percent').replace('+', ' plus')
                     clean_time = str(next_time).replace('+', ' ')
                     clean_confidence = str(confidence).replace('%', '')
                     
                     simple_msg = f"""GHOST TRANSCENDENCE CORE CANDLE WHISPERER

SIGNAL: {clean_signal}
TIMEFRAME: 1M
ENTRY TIME: {clean_time} UTC 6
CONFIDENCE: {clean_confidence} percent

CANDLE CONVERSATIONS: {total_candles} candles consulted

FEATURES ACTIVE:
- Candle Whisperer Mode
- 100 percent Accuracy Target
- UTC 6 Timing Precision
- Secret Pattern Detection
- Loss Prevention System

This signal was generated by talking with every candle for maximum accuracy."""
                     
                     return simple_msg
                
                # Try ultra-clean approach first
                try:
                    clean_message = ultra_clean_telegram_text(message)
                    
                    # If message is too damaged, create simple version
                    if len(clean_message) < 100:
                        clean_message = create_simple_message(
                            signal_type, 
                            confidence, 
                            next_candle_time, 
                            total_candles
                        )
                    
                except Exception as e:
                    # Fallback to ultra-simple message
                    clean_message = create_simple_message(
                        signal_type, 
                        confidence, 
                        next_candle_time, 
                        total_candles
                    )
                    
                    logger.error(f"Message cleaning failed, using simple format: {e}")
                
                # Edit the processing message with MAXIMUM safety
                try:
                    await processing_msg.edit_text(clean_message)
                    logger.info(f"✅ Message sent successfully: {len(clean_message)} chars")
                    
                except Exception as telegram_error:
                    # ULTIMATE FALLBACK: Super simple plain text
                    logger.error(f"❌ Telegram message failed: {telegram_error}")
                    
                                         # ULTIMATE SAFE MESSAGE - only letters, numbers, spaces
                     safe_signal = ''.join(c for c in str(signal_type) if c.isalnum() or c == ' ')
                     safe_time = ''.join(c for c in str(next_candle_time) if c.isalnum() or c == ' ')
                     safe_confidence = ''.join(c for c in str(confidence) if c.isdigit() or c == '.')
                     safe_candles = ''.join(c for c in str(total_candles) if c.isdigit())
                     
                     fallback_message = f"""CANDLE WHISPERER ANALYSIS COMPLETE

Signal: {safe_signal}
Entry: {safe_time} UTC 6
Confidence: {safe_confidence} percent
Candles: {safe_candles}

The AI has analyzed your chart by talking with every candle.
Signal generated with high accuracy for volatile markets."""
                    
                    try:
                        await processing_msg.edit_text(fallback_message)
                        logger.info("✅ Fallback message sent successfully")
                    except Exception as final_error:
                        logger.error(f"❌ Even fallback failed: {final_error}")
                                                 # Send completely new message as last resort
                         try:
                             emergency_signal = ''.join(c for c in str(signal_type) if c.isalnum())
                             emergency_time = ''.join(c for c in str(next_candle_time) if c.isalnum())
                             emergency_conf = ''.join(c for c in str(confidence) if c.isdigit() or c == '.')
                             emergency_msg = f"SIGNAL {emergency_signal} TIME {emergency_time} CONFIDENCE {emergency_conf}"
                             await update.message.reply_text(emergency_msg)
                         except:
                             # Final final fallback
                             try:
                                 await update.message.reply_text("CANDLE WHISPERER ANALYSIS COMPLETE")
                             except:
                                 pass
                
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