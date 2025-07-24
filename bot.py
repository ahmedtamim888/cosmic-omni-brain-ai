import os
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from chart_checker import ChartChecker
import asyncio

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class TradingBot:
    def __init__(self, token: str):
        """
        Initialize the Telegram Trading Bot
        
        Args:
            token (str): Telegram bot token
        """
        self.token = token
        self.chart_checker = ChartChecker()
        self.application = Application.builder().token(token).build()
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup command and message handlers"""
        # Command handlers
        self.application.add_handler(CommandHandler("start", self.start_command))
        
        # Message handlers
        self.application.add_handler(MessageHandler(filters.PHOTO, self.handle_image))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text))
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Handle /start command
        
        Args:
            update (Update): Telegram update object
            context (ContextTypes.DEFAULT_TYPE): Telegram context
        """
        welcome_message = (
            "🤖 **Welcome to Binary Trading Signal Bot!**\n\n"
            "📊 Send me a trading chart screenshot from platforms like:\n"
            "• Quotex\n"
            "• TradingView\n"
            "• MetaTrader 4/5\n"
            "• Binomo\n"
            "• And other trading platforms\n\n"
            "🔍 I'll analyze your chart and provide trading signals!\n\n"
            "📸 Just send me an image to get started."
        )
        
        await update.message.reply_text(welcome_message, parse_mode='Markdown')
        logger.info(f"User {update.effective_user.id} started the bot")
    
    async def handle_image(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Handle image messages and analyze trading charts
        
        Args:
            update (Update): Telegram update object
            context (ContextTypes.DEFAULT_TYPE): Telegram context
        """
        try:
            # Send processing message
            processing_msg = await update.message.reply_text("🔍 Analyzing your chart...")
            
            # Get the largest photo size
            photo = update.message.photo[-1]
            
            # Download the image
            file = await context.bot.get_file(photo.file_id)
            image_path = f"temp_chart_{update.effective_user.id}.jpg"
            await file.download_to_drive(image_path)
            
            logger.info(f"Downloaded image from user {update.effective_user.id}")
            
            # Check if it's a valid trading chart
            is_valid_chart = self.chart_checker.is_valid_chart(image_path)
            
            # Clean up temporary file
            if os.path.exists(image_path):
                os.remove(image_path)
            
            # Delete processing message
            await processing_msg.delete()
            
            if is_valid_chart:
                # Generate trading signal (placeholder logic)
                signal = self._generate_signal()
                await update.message.reply_text(signal, parse_mode='Markdown')
                logger.info(f"Valid chart detected for user {update.effective_user.id}")
            else:
                error_message = (
                    "⚠️ **This is not a valid chart. Please send a real chart screenshot**\n\n"
                    "📊 Make sure your image contains a trading chart from platforms like:\n"
                    "• Quotex\n"
                    "• TradingView\n"
                    "• MetaTrader\n"
                    "• Binomo\n"
                    "• Other trading platforms"
                )
                await update.message.reply_text(error_message, parse_mode='Markdown')
                logger.info(f"Invalid chart rejected for user {update.effective_user.id}")
                
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            await update.message.reply_text(
                "❌ Sorry, there was an error processing your image. Please try again."
            )
            
            # Clean up on error
            image_path = f"temp_chart_{update.effective_user.id}.jpg"
            if os.path.exists(image_path):
                os.remove(image_path)
    
    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Handle text messages
        
        Args:
            update (Update): Telegram update object
            context (ContextTypes.DEFAULT_TYPE): Telegram context
        """
        response = (
            "📸 Please send me a trading chart image!\n\n"
            "I can only analyze chart screenshots from trading platforms like "
            "Quotex, TradingView, MetaTrader, Binomo, etc."
        )
        await update.message.reply_text(response)
    
    def _generate_signal(self) -> str:
        """
        Generate a trading signal (placeholder implementation)
        
        Returns:
            str: Formatted trading signal message
        """
        import random
        
        # Placeholder signal generation (you can implement your own logic here)
        signals = [
            "✅ **Chart detected. Signal: 📈 CALL/UP**\n⏰ Duration: 1 minute\n💰 Recommended amount: 2-5% of balance",
            "✅ **Chart detected. Signal: 📉 PUT/DOWN**\n⏰ Duration: 1 minute\n💰 Recommended amount: 2-5% of balance",
            "✅ **Chart detected. Signal: 📈 CALL/UP**\n⏰ Duration: 5 minutes\n💰 Recommended amount: 2-5% of balance",
            "✅ **Chart detected. Signal: 📉 PUT/DOWN**\n⏰ Duration: 5 minutes\n💰 Recommended amount: 2-5% of balance"
        ]
        
        return random.choice(signals)
    
    def run(self):
        """Start the bot"""
        logger.info("Starting Telegram Trading Bot...")
        self.application.run_polling(allowed_updates=Update.ALL_TYPES)

def main():
    """Main function to run the bot"""
    # Bot token (replace with your actual token)
    BOT_TOKEN = "7340743983:AAEhbXHGdS29FRhdYLGMVNoIRpFOF26b2NU"
    
    if not BOT_TOKEN:
        logger.error("Bot token not provided!")
        return
    
    # Create and run the bot
    bot = TradingBot(BOT_TOKEN)
    bot.run()

if __name__ == "__main__":
    main()