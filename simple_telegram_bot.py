#!/usr/bin/env python3
"""
ULTRA-SIMPLE TELEGRAM BOT - NO PARSING ERRORS GUARANTEED
Uses only basic ASCII characters - completely Telegram-safe
"""

import asyncio
import logging
import requests
import io
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from PIL import Image
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Bot configuration
BOT_TOKEN = "8155744845:AAGQ6s8RMSpCAjO16Mzb1xl8_otLnXh3OKQ"
FLASK_API_URL = "http://localhost:5000/api/analyze"

class SimpleTelegramBot:
    """Ultra-simple Telegram bot with guaranteed no parsing errors"""
    
    def __init__(self):
        self.app = Application.builder().token(BOT_TOKEN).build()
        self.setup_handlers()
        
    def setup_handlers(self):
        self.app.add_handler(CommandHandler("start", self.start_command))
        self.app.add_handler(MessageHandler(filters.PHOTO, self.handle_photo))
        logger.info("Simple bot handlers setup complete")

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Ultra-simple start message"""
        message = """CANDLE WHISPERER BOT

Send me your chart screenshot and I will analyze it.

The AI talks with every candle to provide accurate signals.

Ready to analyze your volatile market charts."""
        
        await update.message.reply_text(message)

    async def handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle photos with ultra-simple responses"""
        try:
            # Send simple processing message
            processing_msg = await update.message.reply_text("Analyzing chart... Please wait...")
            
            # Get photo
            photo = update.message.photo[-1]
            photo_file = await photo.get_file()
            photo_bytes = await photo_file.download_as_bytearray()
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(photo_bytes))
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            
            # Send to API
            files = {'image': ('chart.png', img_byte_arr, 'image/png')}
            response = requests.post(FLASK_API_URL, files=files, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract basic info
                signal = result.get('signal', 'NO SIGNAL')
                confidence = result.get('confidence', 0)
                entry_time = result.get('next_candle_time', '00:00')
                candles = result.get('total_candles_consulted', 0)
                
                # Create ULTRA-SIMPLE message with only safe characters
                safe_message = self.create_safe_message(signal, confidence, entry_time, candles)
                
                # Send result
                await processing_msg.edit_text(safe_message)
                logger.info(f"Message sent successfully: {signal}")
                
            else:
                await processing_msg.edit_text(f"Analysis failed. Error code: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            try:
                await update.message.reply_text("Analysis failed. Please try again with a clear chart.")
            except:
                pass

    def create_safe_message(self, signal, confidence, entry_time, candles):
        """Create ultra-safe message with only basic ASCII"""
        
        # Clean all inputs - only alphanumeric and basic punctuation
        safe_signal = ''.join(c for c in str(signal) if c.isalnum() or c == ' ')
        safe_confidence = ''.join(c for c in str(confidence) if c.isdigit() or c == '.')
        safe_time = ''.join(c for c in str(entry_time) if c.isdigit() or c == ':')
        safe_candles = ''.join(c for c in str(candles) if c.isdigit())
        
        # Create message with only safe characters
        message = f"""CANDLE WHISPERER ANALYSIS COMPLETE

SIGNAL: {safe_signal}
TIMEFRAME: 1M
ENTRY: {safe_time} UTC
CONFIDENCE: {safe_confidence}

CANDLES ANALYZED: {safe_candles}

The AI analyzed your chart by talking with every candle.

Features:
- Candle Conversations
- Pattern Detection  
- Loss Prevention
- High Accuracy Target

Signal generated for volatile markets."""
        
        return message

async def main():
    """Main function"""
    try:
        logger.info("Starting SIMPLE Telegram Bot...")
        
        bot = SimpleTelegramBot()
        
        await bot.app.initialize()
        await bot.app.start()
        
        logger.info("Simple bot is running!")
        
        await bot.app.updater.start_polling()
        await asyncio.Event().wait()
        
    except Exception as e:
        logger.error(f"Bot error: {str(e)}")
    finally:
        if 'bot' in locals():
            await bot.app.stop()

if __name__ == "__main__":
    asyncio.run(main())