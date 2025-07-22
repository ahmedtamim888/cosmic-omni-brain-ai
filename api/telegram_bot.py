import asyncio
import logging
from typing import Optional, Dict, Any
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters
import json
import base64
from io import BytesIO
from datetime import datetime

from strategies.cosmic_ai_strategy import CosmicAIStrategy
from config import Config

class TelegramTradingBot:
    """🌌 COSMIC AI Telegram Bot Interface"""
    
    def __init__(self, token: str):
        self.token = token
        self.cosmic_ai = CosmicAIStrategy()
        self.application = None
        self.active_users = {}
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self):
        """Initialize the Telegram bot"""
        self.application = Application.builder().token(self.token).build()
        
        # Command handlers
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("analyze", self.analyze_command))
        self.application.add_handler(CommandHandler("status", self.status_command))
        self.application.add_handler(CommandHandler("settings", self.settings_command))
        
        # Message handlers
        self.application.add_handler(MessageHandler(filters.PHOTO, self.handle_chart_image))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text_message))
        
        # Callback query handler for inline keyboards
        self.application.add_handler(CallbackQueryHandler(self.handle_callback_query))
        
        self.logger.info("Telegram bot initialized successfully")
    
    async def start_bot(self):
        """Start the Telegram bot"""
        await self.initialize()
        await self.application.run_polling()
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        user_id = update.effective_user.id
        
        welcome_message = """
🌌 **OMNI-BRAIN BINARY AI ACTIVATED**

🧠 **THE ULTIMATE ADAPTIVE STRATEGY BUILDER**

🚀 **REVOLUTIONARY FEATURES:**
- 🔍 **PERCEPTION ENGINE**: Advanced chart analysis with dynamic broker detection
- 📖 **CONTEXT ENGINE**: Reads market stories like a human trader
- 🧠 **STRATEGY ENGINE**: Builds unique strategies on-the-fly for each chart

🎯 **HOW COSMIC AI WORKS:**
1. **ANALYZES** market conditions in real-time
2. **READS** the candle conversation and market psychology  
3. **BUILDS** a custom strategy tree for current setup
4. **EXECUTES** only when strategy confidence > threshold
5. **ADAPTS** strategy logic based on market state

💫 **STRATEGY TYPES:**
- Breakout Continuation
- Reversal Play
- Momentum Shift
- Trap Fade
- Exhaustion Reversal

📍 **Predicts NEXT 1-minute candle direction with full reasoning**

**Send a chart image to activate COSMIC AI strategy building!**

Use /help for detailed commands.
        """
        
        keyboard = [
            [InlineKeyboardButton("📊 Analyze Chart", callback_data="analyze_chart")],
            [InlineKeyboardButton("⚙️ Settings", callback_data="settings"), 
             InlineKeyboardButton("📈 Status", callback_data="status")],
            [InlineKeyboardButton("❓ Help", callback_data="help")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(welcome_message, parse_mode='Markdown', reply_markup=reply_markup)
        
        # Initialize user session
        self.active_users[user_id] = {
            'started_at': datetime.now(),
            'analysis_count': 0,
            'last_analysis': None
        }
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        help_text = """
🔧 **COSMIC AI COMMANDS:**

📸 **Chart Analysis:**
- Send a chart image → Get instant AI analysis
- Supports all major binary brokers (Deriv, IQ Option, Quotex, etc.)
- Automatic broker and timeframe detection

💬 **Commands:**
- `/start` - Initialize COSMIC AI
- `/analyze` - Request chart analysis  
- `/status` - View analysis statistics
- `/settings` - Configure AI parameters
- `/help` - Show this help message

🎯 **How to Use:**
1. Take a screenshot of your trading chart
2. Send the image to this bot
3. COSMIC AI will analyze and provide:
   - Market story and psychology
   - Custom strategy tree
   - Direction prediction with confidence
   - Optimal expiry time
   - Risk assessment

🌟 **Features:**
- **Real-time Analysis**: Instant chart reading
- **Multi-Broker Support**: Works with any broker
- **Adaptive Strategies**: Custom strategy for each setup
- **Risk Management**: Built-in risk assessment
- **Learning AI**: Improves with each analysis

💡 **Tips:**
- Use clear, high-quality chart images
- Include multiple timeframes if possible
- Check confidence levels before trading
- Always manage your risk appropriately
        """
        
        await update.message.reply_text(help_text, parse_mode='Markdown')
    
    async def analyze_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /analyze command"""
        message = """
📊 **READY FOR COSMIC AI ANALYSIS**

Send me a chart image and I'll perform a complete analysis:

🔍 **What I'll analyze:**
- Chart patterns and trends
- Support/resistance levels
- Market psychology signals
- Volume and momentum
- Technical indicators

🧠 **What you'll get:**
- Complete market story
- Custom strategy tree
- Direction prediction
- Confidence score
- Optimal expiry time
- Risk assessment

📱 **Supported formats:**
- PNG, JPG, JPEG images
- Screenshots from any broker
- Mobile or desktop charts

**Just send the image now!**
        """
        
        await update.message.reply_text(message, parse_mode='Markdown')
    
    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command"""
        user_id = update.effective_user.id
        
        if user_id not in self.active_users:
            await update.message.reply_text("Please start the bot first with /start")
            return
        
        user_data = self.active_users[user_id]
        
        status_message = f"""
📊 **COSMIC AI STATUS**

👤 **User:** {update.effective_user.first_name}
🕐 **Session Started:** {user_data['started_at'].strftime('%Y-%m-%d %H:%M:%S')}
🔢 **Analyses Performed:** {user_data['analysis_count']}
📈 **Last Analysis:** {user_data['last_analysis'].strftime('%H:%M:%S') if user_data['last_analysis'] else 'None'}

🤖 **AI Status:**
- **Perception Engine:** ✅ Active
- **Context Engine:** ✅ Active  
- **Strategy Engine:** ✅ Active
- **Learning Module:** ✅ Active

🎯 **Current Settings:**
- **Confidence Threshold:** {self.cosmic_ai.confidence_threshold:.1%}
- **Strategy Types:** 5 Available
- **Pattern Memory:** {len(self.cosmic_ai.pattern_memory)} Patterns

**Ready for next analysis!**
        """
        
        keyboard = [
            [InlineKeyboardButton("📊 New Analysis", callback_data="analyze_chart")],
            [InlineKeyboardButton("⚙️ Settings", callback_data="settings")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(status_message, parse_mode='Markdown', reply_markup=reply_markup)
    
    async def settings_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /settings command"""
        settings_message = """
⚙️ **COSMIC AI SETTINGS**

🎯 **Current Configuration:**
- **Confidence Threshold:** 75%
- **Analysis Mode:** Full Spectrum
- **Strategy Adaptation:** Enabled
- **Learning Mode:** Active

🔧 **Adjustable Parameters:**
        """
        
        keyboard = [
            [InlineKeyboardButton("📊 Confidence: 75%", callback_data="confidence_75"),
             InlineKeyboardButton("📊 Confidence: 80%", callback_data="confidence_80")],
            [InlineKeyboardButton("🧠 Strategy Mode", callback_data="strategy_mode"),
             InlineKeyboardButton("📚 Learning", callback_data="learning_mode")],
            [InlineKeyboardButton("🔙 Back to Main", callback_data="main_menu")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(settings_message, parse_mode='Markdown', reply_markup=reply_markup)
    
    async def handle_chart_image(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle chart image uploads"""
        user_id = update.effective_user.id
        
        # Send initial processing message
        processing_msg = await update.message.reply_text(
            "🌌 **COSMIC AI ANALYZING...**\n\n"
            "🔍 Extracting chart elements...\n"
            "📖 Reading market story...\n"
            "🧠 Building strategy tree...",
            parse_mode='Markdown'
        )
        
        try:
            # Get the image
            photo = update.message.photo[-1]  # Get highest resolution
            file = await context.bot.get_file(photo.file_id)
            
            # Download image data
            image_data = BytesIO()
            await file.download_to_memory(image_data)
            image_bytes = image_data.getvalue()
            
            # Update progress
            await processing_msg.edit_text(
                "🌌 **COSMIC AI ANALYZING...**\n\n"
                "✅ Chart elements extracted\n"
                "📖 Reading market story...\n"
                "🧠 Building strategy tree...",
                parse_mode='Markdown'
            )
            
            # Perform COSMIC AI analysis
            analysis_result = await asyncio.get_event_loop().run_in_executor(
                None, self.cosmic_ai.execute_cosmic_analysis, image_bytes, None
            )
            
            # Update progress
            await processing_msg.edit_text(
                "🌌 **COSMIC AI ANALYZING...**\n\n"
                "✅ Chart elements extracted\n"
                "✅ Market story analyzed\n"
                "🧠 Finalizing strategy...",
                parse_mode='Markdown'
            )
            
            # Format and send results
            await self.send_analysis_results(update, analysis_result)
            
            # Update user statistics
            if user_id in self.active_users:
                self.active_users[user_id]['analysis_count'] += 1
                self.active_users[user_id]['last_analysis'] = datetime.now()
            
            # Delete processing message
            await processing_msg.delete()
            
        except Exception as e:
            self.logger.error(f"Error processing chart image: {e}")
            await processing_msg.edit_text(
                f"❌ **Analysis Failed**\n\n"
                f"Error: {str(e)}\n\n"
                f"Please try again with a clearer image.",
                parse_mode='Markdown'
            )
    
    async def send_analysis_results(self, update: Update, analysis_result: Dict):
        """Send formatted analysis results"""
        try:
            if 'error' in analysis_result:
                await update.message.reply_text(
                    f"❌ **Analysis Error**\n\n{analysis_result['error']}",
                    parse_mode='Markdown'
                )
                return
            
            # Extract analysis stages
            stages = analysis_result.get('analysis_stages', {})
            
            # Format perception stage
            perception_msg = self.format_perception_results(stages.get('perception', {}))
            
            # Format story stage  
            story_msg = self.format_story_results(stages.get('story', {}))
            
            # Format strategy stage
            strategy_msg = self.format_strategy_results(stages.get('strategy', {}))
            
            # Send perception results
            if perception_msg:
                await update.message.reply_text(perception_msg, parse_mode='Markdown')
            
            # Send story results
            if story_msg:
                await update.message.reply_text(story_msg, parse_mode='Markdown')
            
            # Send strategy results with action buttons
            if strategy_msg:
                keyboard = []
                if analysis_result.get('execute_trade', False):
                    signal = analysis_result.get('trading_signal', {})
                    direction = signal.get('direction', 'CALL')
                    keyboard.append([
                        InlineKeyboardButton(f"🚀 Execute {direction}", callback_data=f"execute_{direction}"),
                        InlineKeyboardButton("⏸️ Skip Trade", callback_data="skip_trade")
                    ])
                
                keyboard.append([
                    InlineKeyboardButton("📊 New Analysis", callback_data="analyze_chart"),
                    InlineKeyboardButton("📈 View Details", callback_data="view_details")
                ])
                
                reply_markup = InlineKeyboardMarkup(keyboard)
                await update.message.reply_text(strategy_msg, parse_mode='Markdown', reply_markup=reply_markup)
            
        except Exception as e:
            self.logger.error(f"Error sending analysis results: {e}")
            await update.message.reply_text(
                "❌ **Error formatting results**\n\nPlease try again.",
                parse_mode='Markdown'
            )
    
    def format_perception_results(self, perception_data: Dict) -> str:
        """Format perception engine results"""
        if not perception_data or 'error' in perception_data:
            return ""
        
        broker = perception_data.get('broker', 'Unknown')
        timeframe = perception_data.get('timeframe', 'Unknown')
        asset = perception_data.get('asset', 'Unknown')
        confidence = perception_data.get('confidence', 0)
        
        return f"""
🔍 **PERCEPTION ENGINE RESULTS**

📊 **Chart Details:**
- **Broker:** {broker.upper()}
- **Asset:** {asset}
- **Timeframe:** {timeframe}
- **Detection Confidence:** {confidence:.1%}

✅ **Chart elements successfully extracted**
        """
    
    def format_story_results(self, story_data: Dict) -> str:
        """Format story engine results"""
        if not story_data:
            return ""
        
        story = story_data.get('story', 'No story available')
        confidence = story_data.get('confidence', 0)
        narrative_bias = story_data.get('narrative_bias', 'neutral')
        
        # Format bias with emoji
        bias_emoji = "🟢" if narrative_bias == "bullish" else "🔴" if narrative_bias == "bearish" else "🟡"
        
        return f"""
📖 **MARKET STORY ANALYSIS**

{bias_emoji} **Narrative Bias:** {narrative_bias.upper()}
📊 **Story Confidence:** {confidence:.1%}

**Market Narrative:**
{story}

**Key Insights:**
- Market psychology detected
- Price action patterns identified
- Volume and momentum analyzed
        """
    
    def format_strategy_results(self, strategy_data: Dict) -> str:
        """Format strategy engine results"""
        if not strategy_data:
            return ""
        
        strategy_type = strategy_data.get('strategy_type', 'Unknown')
        confidence = strategy_data.get('overall_confidence', 0)
        direction = strategy_data.get('predicted_direction', 'CALL')
        reasoning = strategy_data.get('reasoning', 'No reasoning provided')
        expiry = strategy_data.get('expiry_recommendation', 300)
        risk_level = strategy_data.get('risk_level', 'medium')
        
        # Format direction with emoji
        direction_emoji = "📈" if direction == "call" else "📉"
        
        # Format confidence bar
        confidence_bars = "█" * int(confidence * 10)
        confidence_empty = "░" * (10 - int(confidence * 10))
        confidence_bar = confidence_bars + confidence_empty
        
        # Format risk level with emoji
        risk_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(risk_level, "🟡")
        
        return f"""
🧠 **STRATEGY ENGINE RESULTS**

🎯 **PREDICTION:** {direction_emoji} **{direction.upper()}**

📊 **Confidence:** {confidence:.1%}
{confidence_bar}

⏰ **Recommended Expiry:** {expiry // 60}m {expiry % 60}s
{risk_emoji} **Risk Level:** {risk_level.upper()}

🧭 **Strategy Type:** {strategy_type.replace('_', ' ').title()}

💭 **AI Reasoning:**
{reasoning}

🎲 **Strategy Nodes:**
        """
    
    async def handle_text_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle text messages"""
        text = update.message.text.lower()
        
        if any(word in text for word in ['analyze', 'chart', 'help', 'trade']):
            await update.message.reply_text(
                "📸 **Send me a chart image for analysis!**\n\n"
                "I can analyze charts from any broker and provide:\n"
                "- Market psychology reading\n"
                "- Custom strategy building\n" 
                "- Direction prediction\n"
                "- Risk assessment\n\n"
                "Just upload your chart screenshot!",
                parse_mode='Markdown'
            )
        else:
            await update.message.reply_text(
                "🤖 **COSMIC AI Ready**\n\n"
                "Send a chart image for analysis or use /help for commands.",
                parse_mode='Markdown'
            )
    
    async def handle_callback_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle inline keyboard callbacks"""
        query = update.callback_query
        await query.answer()
        
        data = query.data
        
        if data == "analyze_chart":
            await query.edit_message_text(
                "📊 **Ready for Analysis**\n\n"
                "Send me a chart image to activate COSMIC AI!",
                parse_mode='Markdown'
            )
        
        elif data == "status":
            await self.status_command(update, context)
        
        elif data == "settings":
            await self.settings_command(update, context)
        
        elif data == "help":
            await self.help_command(update, context)
        
        elif data.startswith("confidence_"):
            threshold = int(data.split("_")[1]) / 100
            self.cosmic_ai.confidence_threshold = threshold
            
            await query.edit_message_text(
                f"✅ **Confidence threshold updated to {threshold:.0%}**\n\n"
                f"The AI will now only execute trades with {threshold:.0%}+ confidence.",
                parse_mode='Markdown'
            )
        
        elif data.startswith("execute_"):
            direction = data.split("_")[1]
            await query.edit_message_text(
                f"🚀 **Execute {direction.upper()} Trade**\n\n"
                f"This would execute a {direction} trade based on COSMIC AI analysis.\n\n"
                f"⚠️ **Demo Mode**: Actual trading integration required.",
                parse_mode='Markdown'
            )
        
        elif data == "skip_trade":
            await query.edit_message_text(
                "⏸️ **Trade Skipped**\n\n"
                "COSMIC AI analysis completed but trade not executed.\n"
                "Send another chart for new analysis!",
                parse_mode='Markdown'
            )
    
    async def send_signal_alert(self, chat_id: str, signal_data: Dict):
        """Send trading signal alert to user"""
        direction = signal_data.get('direction', 'CALL')
        confidence = signal_data.get('confidence', 0)
        asset = signal_data.get('asset', 'Unknown')
        expiry = signal_data.get('expiry_time', 300)
        
        direction_emoji = "📈" if direction == "CALL" else "📉"
        
        alert_message = f"""
🚨 **COSMIC AI SIGNAL ALERT**

{direction_emoji} **{direction}** on **{asset}**

📊 **Confidence:** {confidence:.1%}
⏰ **Expiry:** {expiry // 60}m {expiry % 60}s
🕐 **Time:** {datetime.now().strftime('%H:%M:%S')}

🎯 **Action Required:** Review and execute if desired

**This is an automated signal from COSMIC AI**
        """
        
        keyboard = [
            [InlineKeyboardButton(f"🚀 Execute {direction}", callback_data=f"execute_{direction}")],
            [InlineKeyboardButton("⏸️ Skip", callback_data="skip_trade")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        try:
            await self.application.bot.send_message(
                chat_id=chat_id,
                text=alert_message,
                parse_mode='Markdown',
                reply_markup=reply_markup
            )
        except Exception as e:
            self.logger.error(f"Failed to send signal alert: {e}")

# Create global bot instance
telegram_bot = None

async def start_telegram_bot():
    """Start the Telegram bot"""
    global telegram_bot
    
    token = Config.TELEGRAM_BOT_TOKEN
    if not token:
        logging.error("Telegram bot token not configured")
        return
    
    telegram_bot = TelegramTradingBot(token)
    await telegram_bot.start_bot()

if __name__ == "__main__":
    asyncio.run(start_telegram_bot())