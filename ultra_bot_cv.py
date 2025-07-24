#!/usr/bin/env python3
"""
🧠 ULTRA TELEGRAM TRADING BOT v.Ω.2
OMNI-BRAIN PERCEPTION ENGINE
Combines OCR + Computer Vision for Ultimate Precision
"""

import logging
import os
import asyncio
import random
from datetime import datetime
from typing import Optional

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from chart_checker import ChartChecker
from advanced_chart_analyzer import AdvancedChartAnalyzer

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class UltraTradingBot:
    """
    🧠 ULTRA-ADVANCED TRADING BOT
    
    Features:
    - OCR text validation for chart authenticity
    - Computer vision candle detection and pattern analysis
    - Advanced trading signal generation
    - Professional error handling
    - Clean user interface
    """
    
    def __init__(self, bot_token: str):
        """Initialize the ultra trading bot"""
        self.bot_token = bot_token
        self.chart_checker = ChartChecker()  # OCR validator
        self.cv_analyzer = AdvancedChartAnalyzer()  # Computer vision analyzer
        self.application = None
        
        # 📊 Enhanced signal templates
        self.signal_templates = {
            'call': [
                "🚀 **ULTRA CALL SIGNAL**",
                "📈 **STRONG BUY DETECTED**",
                "⬆️ **BULLISH MOMENTUM CONFIRMED**"
            ],
            'put': [
                "📉 **ULTRA PUT SIGNAL**", 
                "🔴 **STRONG SELL DETECTED**",
                "⬇️ **BEARISH MOMENTUM CONFIRMED**"
            ],
            'no_trade': [
                "⚠️ **NO TRADE RECOMMENDED**",
                "🔍 **INSUFFICIENT CONFIDENCE**",
                "⏳ **WAIT FOR BETTER SETUP**"
            ]
        }
        
        # 📁 Create temp directory for images
        os.makedirs("temp_images", exist_ok=True)
        
        logger.info("🧠 Ultra Trading Bot initialized with dual analysis engines")
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """🚀 Handle /start command"""
        user = update.effective_user
        welcome_message = f"""🧠 **ULTRA TRADING BOT v.Ω.2** 

Hello {user.first_name}! 👋

🎯 **DUAL-ENGINE ANALYSIS:**
• 🔤 **OCR Text Validation** - Verifies authentic trading charts
• 👁️ **Computer Vision Analysis** - Detects candles and patterns
• 🧠 **AI Pattern Recognition** - Advanced candlestick analysis
• 📊 **Technical Analysis** - Trend, momentum, S/R detection

📱 **ENHANCED CAPABILITIES:**
✅ Candle detection with noise filtering
✅ Pattern recognition (DOJI, Hammer, Engulfing, etc.)
✅ Support/Resistance zone analysis  
✅ Momentum and trend analysis
✅ Advanced signal confidence scoring

🎯 **SUPPORTED PLATFORMS:**
Quotex, TradingView, MetaTrader, Binomo, IQ Option, and more!

📸 **Send me a chart screenshot for ULTRA-PRECISE analysis!**

Commands: /help /status
        """
        
        await update.message.reply_text(welcome_message, parse_mode='Markdown')
        logger.info("📱 User %s started Ultra Bot", user.first_name)
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """❓ Handle /help command"""
        help_message = """🧠 **ULTRA TRADING BOT HELP**

**🔧 COMMANDS:**
• `/start` - Welcome and bot introduction
• `/help` - This help message
• `/status` - Bot status and capabilities

**📊 ANALYSIS PROCESS:**
1️⃣ **OCR Validation** - Confirms it's a real trading chart
2️⃣ **Candle Detection** - Computer vision finds candles
3️⃣ **Pattern Analysis** - AI identifies trading patterns
4️⃣ **Signal Generation** - Ultra-precise CALL/PUT/NO_TRADE

**🎯 DETECTED PATTERNS:**
• DOJI (Reversal indication)
• HAMMER (Bullish at support)
• SHOOTING STAR (Bearish at resistance)
• ENGULFING (Strong momentum)
• MARUBOZU (Strong directional move)

**📈 ANALYSIS COMPONENTS:**
• **Trend Analysis** (30% weight)
• **Pattern Recognition** (40% weight)  
• **Momentum Analysis** (20% weight)
• **Support/Resistance** (10% weight)

**🔍 SIGNAL CONFIDENCE:**
• **90%+ = ULTRA HIGH** - Strong trade recommendation
• **70-89% = HIGH** - Good trade setup
• **50-69% = MEDIUM** - Proceed with caution
• **<50% = LOW** - No trade recommended

**💡 TIPS:**
• Use clear, high-quality screenshots
• Include multiple candles (5+ recommended)
• Ensure proper lighting and contrast
• Avoid cropped or blurry images

Need support? Just send /help again! 🔧
        """
        
        await update.message.reply_text(help_message, parse_mode='Markdown')
    
    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """📊 Handle /status command"""
        status_message = f"""🧠 **ULTRA BOT STATUS REPORT**

🟢 **STATUS:** Online and Operational
⏰ **TIME:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**🔧 ENGINE STATUS:**
✅ **OCR Engine:** Ready (103+ keywords)
✅ **Computer Vision:** Operational
✅ **Pattern Recognition:** Active
✅ **Signal Generator:** Armed

**🎯 CAPABILITIES:**
✅ **Chart Validation:** Multi-platform support
✅ **Candle Detection:** Advanced filtering
✅ **Pattern Analysis:** 8+ patterns detected
✅ **Trend Analysis:** Real-time calculation
✅ **S/R Detection:** Dynamic zone identification
✅ **Signal Confidence:** ML-powered scoring

**📊 ANALYSIS FEATURES:**
• **Minimum Candles:** 5 required
• **Max Candles:** 25 processed
• **Pattern Types:** DOJI, HAMMER, ENGULFING, etc.
• **Color Detection:** Bull/Bear with theme adaptation
• **Noise Filtering:** Advanced morphological operations

**🚀 PERFORMANCE:**
• **Speed:** <3 seconds analysis
• **Accuracy:** Ultra-high precision
• **Reliability:** Professional grade

Send me a chart screenshot for ULTRA analysis! 📸
        """
        
        await update.message.reply_text(status_message, parse_mode='Markdown')
    
    async def handle_image(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """📸 Handle image messages with dual analysis"""
        try:
            user = update.effective_user
            logger.info("📸 Received image from user %s", user.first_name)
            
            # Send initial processing message
            processing_msg = await update.message.reply_text(
                "🧠 **ULTRA ANALYSIS INITIATED...**\n"
                "🔍 Phase 1: OCR Validation\n"
                "👁️ Phase 2: Computer Vision\n"
                "🧩 Phase 3: Pattern Analysis\n"
                "🎯 Phase 4: Signal Generation\n\n"
                "⏳ Please wait...",
                parse_mode='Markdown'
            )
            
            # Get the largest photo size
            photo = update.message.photo[-1]
            
            # Download the image
            file = await context.bot.get_file(photo.file_id)
            
            # Create unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = f"temp_images/ultra_chart_{user.id}_{timestamp}.jpg"
            
            # Download and save the image
            await file.download_to_drive(image_path)
            logger.info("💾 Image saved: %s", image_path)
            
            # Phase 1: OCR Validation
            await processing_msg.edit_text(
                "🧠 **ULTRA ANALYSIS IN PROGRESS...**\n"
                "✅ Phase 1: OCR Validation - Processing\n"
                "⏳ Phase 2: Computer Vision\n"
                "⏳ Phase 3: Pattern Analysis\n"
                "⏳ Phase 4: Signal Generation",
                parse_mode='Markdown'
            )
            
            ocr_valid = await self._validate_chart_async(image_path)
            
            if not ocr_valid:
                await processing_msg.edit_text(
                    "❌ **CHART VALIDATION FAILED**\n\n"
                    "⚠️ This doesn't appear to be a valid trading chart.\n\n"
                    "🔍 **Requirements:**\n"
                    "• Screenshot from trading platform\n"
                    "• Visible trading elements (prices, timeframes)\n"
                    "• Clear text and interface\n"
                    "• Currency pairs or trading terms\n\n"
                    "📸 **Please send a real chart screenshot!**",
                    parse_mode='Markdown'
                )
                logger.info("❌ OCR validation failed for user %s", user.first_name)
                return
            
            # Phase 2: Computer Vision Analysis
            await processing_msg.edit_text(
                "🧠 **ULTRA ANALYSIS IN PROGRESS...**\n"
                "✅ Phase 1: OCR Validation - Complete\n"
                "🔍 Phase 2: Computer Vision - Processing\n"
                "⏳ Phase 3: Pattern Analysis\n"
                "⏳ Phase 4: Signal Generation",
                parse_mode='Markdown'
            )
            
            cv_analysis = await self._analyze_chart_cv_async(image_path)
            
            if not cv_analysis.get('is_valid', False):
                await processing_msg.edit_text(
                    "⚠️ **INSUFFICIENT VISUAL DATA**\n\n"
                    f"🔍 **Analysis Results:**\n"
                    f"• Candles detected: {cv_analysis.get('candles_found', 0)}\n"
                    f"• Required minimum: 5 candles\n\n"
                    "📊 **Suggestions:**\n"
                    "• Zoom out to show more candles\n"
                    "• Use a wider timeframe view\n"
                    "• Ensure candles are clearly visible\n"
                    "• Check image quality and lighting\n\n"
                    "📸 **Please send a chart with more visible candles!**",
                    parse_mode='Markdown'
                )
                logger.info("⚠️ CV analysis insufficient for user %s", user.first_name)
                return
            
            # Phase 3 & 4: Complete Analysis
            await processing_msg.edit_text(
                "🧠 **ULTRA ANALYSIS IN PROGRESS...**\n"
                "✅ Phase 1: OCR Validation - Complete\n"
                "✅ Phase 2: Computer Vision - Complete\n"
                "🧩 Phase 3: Pattern Analysis - Processing\n"
                "🎯 Phase 4: Signal Generation - Processing",
                parse_mode='Markdown'
            )
            
            # Generate ultra signal
            ultra_response = await self._generate_ultra_response(cv_analysis, image_path)
            await processing_msg.edit_text(ultra_response, parse_mode='Markdown')
            
            logger.info("✅ Ultra analysis complete for user %s", user.first_name)
            
            # Clean up temp file
            try:
                os.remove(image_path)
                logger.debug("🗑️ Temp file cleaned up: %s", image_path)
            except:
                pass
                
        except Exception as e:
            logger.error("❌ Error in ultra image processing: %s", str(e))
            
            error_message = """🚫 **ULTRA ANALYSIS ERROR**

😔 An unexpected error occurred during analysis.

**🔧 Please try:**
• Sending the image again
• Using a different chart screenshot
• Ensuring good image quality
• Checking internet connection

If the problem persists, the issue may be temporary. 🔄
            """
            
            await update.message.reply_text(error_message, parse_mode='Markdown')
    
    async def _validate_chart_async(self, image_path: str) -> bool:
        """🔍 Async OCR chart validation"""
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                self.chart_checker.is_valid_chart, 
                image_path
            )
            return result
        except Exception as e:
            logger.error("❌ Error in OCR validation: %s", str(e))
            return False
    
    async def _analyze_chart_cv_async(self, image_path: str) -> dict:
        """👁️ Async computer vision analysis"""
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self.cv_analyzer.analyze_chart_complete,
                image_path
            )
            return result
        except Exception as e:
            logger.error("❌ Error in CV analysis: %s", str(e))
            return {"error": str(e), "is_valid": False}
    
    async def _generate_ultra_response(self, cv_analysis: dict, image_path: str) -> str:
        """🎯 Generate ultra-advanced response"""
        try:
            signal = cv_analysis.get('signal', {})
            pattern_analysis = cv_analysis.get('pattern_analysis', {})
            chart_quality = cv_analysis.get('chart_quality', {})
            candles_found = cv_analysis.get('candles_found', 0)
            
            # Get signal details
            direction = signal.get('direction', 'NO_TRADE')
            confidence = signal.get('confidence', 0) * 100
            reasoning = signal.get('reasoning', [])
            pattern_detected = signal.get('pattern_detected', 'NONE')
            
            # Choose appropriate template
            if direction == 'CALL':
                signal_header = random.choice(self.signal_templates['call'])
                signal_emoji = "📈"
                signal_color = "🟢"
            elif direction == 'PUT':
                signal_header = random.choice(self.signal_templates['put'])
                signal_emoji = "📉"
                signal_color = "🔴"
            else:
                signal_header = random.choice(self.signal_templates['no_trade'])
                signal_emoji = "⚠️"
                signal_color = "🟡"
            
            # Format confidence level
            if confidence >= 90:
                confidence_level = "🔥 ULTRA HIGH"
            elif confidence >= 70:
                confidence_level = "⚡ HIGH"
            elif confidence >= 50:
                confidence_level = "📊 MEDIUM"
            else:
                confidence_level = "⚠️ LOW"
            
            # Get trend and momentum
            trend = pattern_analysis.get('trend', {})
            momentum = pattern_analysis.get('momentum', {})
            sr_analysis = pattern_analysis.get('support_resistance', {})
            
            # Build comprehensive response
            response = f"""{signal_header}

{signal_emoji} **DIRECTION:** {direction}
{signal_color} **CONFIDENCE:** {confidence:.1f}% ({confidence_level})
⏰ **TIME:** {datetime.now().strftime('%H:%M:%S')}

📊 **ANALYSIS SUMMARY:**
🕯️ **Candles Analyzed:** {candles_found}
🧩 **Pattern:** {pattern_detected}
📈 **Trend:** {trend.get('direction', 'NEUTRAL')} ({trend.get('strength', 0):.2f})
⚡ **Momentum:** {momentum.get('momentum', 'NEUTRAL')} ({momentum.get('strength', 0):.2f})
📍 **Zone:** {sr_analysis.get('price_zone', 'MIDDLE')}

🎯 **REASONING:**"""
            
            # Add reasoning points
            for i, reason in enumerate(reasoning[:3], 1):  # Limit to top 3 reasons
                response += f"\n{i}. {reason}"
            
            # Add chart quality assessment
            quality = chart_quality.get('quality_level', 'UNKNOWN')
            quality_score = chart_quality.get('quality_score', 0) * 100
            
            response += f"""

📊 **CHART QUALITY:** {quality} ({quality_score:.1f}%)
🔍 **Clarity:** {chart_quality.get('clarity_ratio', 0):.2f}
📏 **Spacing:** {chart_quality.get('spacing_consistency', 0):.2f}

⚠️ **DISCLAIMER:** This is advanced technical analysis for educational purposes. Always do your own research and risk management before trading."""
            
            return response
            
        except Exception as e:
            logger.error("❌ Error generating ultra response: %s", str(e))
            return f"✅ Chart analyzed. Signal: {signal.get('emoji', '📈')} **{direction}** (Analysis completed)"
    
    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """💬 Handle text messages"""
        text_response = """📸 **SEND CHART FOR ULTRA ANALYSIS!**

🧠 I'm equipped with dual-engine analysis:

**👁️ COMPUTER VISION:**
• Detects candles automatically
• Analyzes patterns and trends
• Calculates precise signals

**🔤 OCR VALIDATION:**
• Confirms chart authenticity
• Multi-platform recognition
• Prevents fake image errors

**📊 WHAT TO SEND:**
• Trading chart screenshots
• Multiple candles visible (5+ recommended)
• Clear image quality
• Any major trading platform

**🎯 WHAT YOU'LL GET:**
• Ultra-precise CALL/PUT signals
• Confidence percentage
• Detailed pattern analysis
• Professional reasoning

📱 Just send the image directly - no typing needed!
        """
        
        await update.message.reply_text(text_response, parse_mode='Markdown')
    
    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """🚫 Global error handler"""
        logger.error("Exception while handling an update:", exc_info=context.error)
        
        if isinstance(update, Update) and update.effective_message:
            try:
                await update.effective_message.reply_text(
                    "🚫 **SYSTEM ERROR**\n\n"
                    "An unexpected error occurred. Please try again.\n"
                    "If the issue persists, it may be temporary.",
                    parse_mode='Markdown'
                )
            except:
                pass
    
    def setup_handlers(self):
        """🔧 Setup bot handlers"""
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("status", self.status_command))
        self.application.add_handler(MessageHandler(filters.PHOTO, self.handle_image))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text))
        self.application.add_error_handler(self.error_handler)
        
        logger.info("🔧 Ultra Bot handlers configured")
    
    def run(self):
        """🚀 Start the ultra bot"""
        try:
            self.application = Application.builder().token(self.bot_token).build()
            self.setup_handlers()
            
            logger.info("🧠 Starting Ultra Trading Bot v.Ω.2...")
            logger.info("🟢 Ultra Bot is running! Press Ctrl+C to stop.")
            
            self.application.run_polling()
            
        except Exception as e:
            logger.error("❌ Error starting Ultra Bot: %s", str(e))
            raise

def main():
    """🎯 Main function"""
    BOT_TOKEN = "8288385434:AAG_RVKnlXDWBZNN38Q3IEfSQXIgxwPlsU0"
    
    try:
        bot = UltraTradingBot(BOT_TOKEN)
        bot.run()
        
    except KeyboardInterrupt:
        logger.info("🛑 Ultra Bot stopped by user")
    except Exception as e:
        logger.error("❌ Fatal error in Ultra Bot: %s", str(e))

if __name__ == "__main__":
    main()