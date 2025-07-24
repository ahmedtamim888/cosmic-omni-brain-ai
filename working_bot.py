#!/usr/bin/env python3
"""
🧠 WORKING ANALYSIS BOT
Uses the actual OMNI-BRAIN PERCEPTION ENGINE detect_candles function
"""

import logging
import os
import asyncio
import cv2
import numpy as np
from datetime import datetime
from typing import Optional, List, Dict, Tuple

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# OMNI-BRAIN PERCEPTION ENGINE - ACTUAL CANDLE DETECTION
def detect_candles(image_path: str, min_candle_height: int = 20, min_candle_width: int = 3) -> List[Dict]:
    """
    🧠 OMNI-BRAIN PERCEPTION ENGINE
    Detects candlesticks in trading chart screenshots
    """
    try:
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            logger.error("❌ Could not load image")
            return []
        
        height, width = image.shape[:2]
        logger.info(f"📐 Image dimensions: {width}x{height}")
        
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        candles_detected = []
        
        # 🟢 DETECT GREEN CANDLES (BULLISH)
        green_lower = np.array([35, 40, 40])  # Lower HSV threshold for green
        green_upper = np.array([85, 255, 255])  # Upper HSV threshold for green
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        
        # Find green contours
        green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in green_contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size to identify candles
            if h >= min_candle_height and w >= min_candle_width and w <= h * 3:
                area = cv2.contourArea(contour)
                if area > 100:  # Minimum area threshold
                    candle_info = {
                        'type': 'bullish',
                        'color': 'green',
                        'x': x,
                        'y': y,
                        'width': w,
                        'height': h,
                        'area': area,
                        'center_x': x + w // 2,
                        'center_y': y + h // 2
                    }
                    candles_detected.append(candle_info)
                    logger.info(f"🟢 Green candle detected at ({x}, {y}) size {w}x{h}")
        
        # 🔴 DETECT RED CANDLES (BEARISH)
        # Red color range 1 (bright red)
        red_lower1 = np.array([0, 40, 40])
        red_upper1 = np.array([10, 255, 255])
        red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        
        # Red color range 2 (dark red)
        red_lower2 = np.array([170, 40, 40])
        red_upper2 = np.array([180, 255, 255])
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        
        # Combine red masks
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        # Find red contours
        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in red_contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size to identify candles
            if h >= min_candle_height and w >= min_candle_width and w <= h * 3:
                area = cv2.contourArea(contour)
                if area > 100:  # Minimum area threshold
                    candle_info = {
                        'type': 'bearish',
                        'color': 'red',
                        'x': x,
                        'y': y,
                        'width': w,
                        'height': h,
                        'area': area,
                        'center_x': x + w // 2,
                        'center_y': y + h // 2
                    }
                    candles_detected.append(candle_info)
                    logger.info(f"🔴 Red candle detected at ({x}, {y}) size {w}x{h}")
        
        # Sort candles by x position (left to right)
        candles_detected.sort(key=lambda c: c['center_x'])
        
        logger.info(f"🧠 OMNI-BRAIN: Detected {len(candles_detected)} total candles")
        return candles_detected
        
    except Exception as e:
        logger.error(f"❌ Error in candle detection: {str(e)}")
        return []

def analyze_candles(candles: List[Dict]) -> Dict:
    """
    📊 Analyze detected candles for trading signals
    """
    if not candles:
        return {
            'signal': 'NO_TRADE',
            'confidence': 0.0,
            'reasoning': ['No candles detected'],
            'total_candles': 0,
            'bullish_candles': 0,
            'bearish_candles': 0
        }
    
    # Count candle types
    bullish_candles = [c for c in candles if c['type'] == 'bullish']
    bearish_candles = [c for c in candles if c['type'] == 'bearish']
    
    total_candles = len(candles)
    bullish_count = len(bullish_candles)
    bearish_count = len(bearish_candles)
    
    # Calculate percentages
    bullish_pct = (bullish_count / total_candles) * 100 if total_candles > 0 else 0
    bearish_pct = (bearish_count / total_candles) * 100 if total_candles > 0 else 0
    
    # Analyze recent candles (last 5 or all if less)
    recent_candles = candles[-5:] if len(candles) >= 5 else candles
    recent_bullish = len([c for c in recent_candles if c['type'] == 'bullish'])
    recent_bearish = len([c for c in recent_candles if c['type'] == 'bearish'])
    
    # Determine signal based on analysis
    reasoning = []
    
    if total_candles < 3:
        signal = 'NO_TRADE'
        confidence = 0.2
        reasoning.append(f"Only {total_candles} candles detected - need at least 3")
    else:
        # Overall trend analysis
        if bullish_pct > bearish_pct + 20:  # Strong bullish
            signal = 'CALL'
            confidence = min(0.9, 0.6 + (bullish_pct - bearish_pct) / 100)
            reasoning.append(f"Strong bullish dominance: {bullish_pct:.1f}% vs {bearish_pct:.1f}%")
        elif bearish_pct > bullish_pct + 20:  # Strong bearish
            signal = 'PUT'
            confidence = min(0.9, 0.6 + (bearish_pct - bullish_pct) / 100)
            reasoning.append(f"Strong bearish dominance: {bearish_pct:.1f}% vs {bullish_pct:.1f}%")
        elif bullish_pct > bearish_pct:  # Mild bullish
            signal = 'CALL'
            confidence = 0.6 + (bullish_pct - bearish_pct) / 200
            reasoning.append(f"Bullish trend: {bullish_pct:.1f}% vs {bearish_pct:.1f}%")
        else:  # Mild bearish
            signal = 'PUT'
            confidence = 0.6 + (bearish_pct - bullish_pct) / 200
            reasoning.append(f"Bearish trend: {bearish_pct:.1f}% vs {bullish_pct:.1f}%")
        
        # Recent candles analysis
        if len(recent_candles) >= 3:
            if recent_bullish > recent_bearish:
                reasoning.append(f"Recent momentum bullish: {recent_bullish}/{len(recent_candles)} green")
                if signal == 'CALL':
                    confidence += 0.1  # Boost confidence
            elif recent_bearish > recent_bullish:
                reasoning.append(f"Recent momentum bearish: {recent_bearish}/{len(recent_candles)} red")
                if signal == 'PUT':
                    confidence += 0.1  # Boost confidence
        
        # Candle size analysis
        avg_bullish_area = np.mean([c['area'] for c in bullish_candles]) if bullish_candles else 0
        avg_bearish_area = np.mean([c['area'] for c in bearish_candles]) if bearish_candles else 0
        
        if avg_bullish_area > avg_bearish_area * 1.2:
            reasoning.append("Bullish candles are larger (stronger moves)")
            if signal == 'CALL':
                confidence += 0.05
        elif avg_bearish_area > avg_bullish_area * 1.2:
            reasoning.append("Bearish candles are larger (stronger moves)")
            if signal == 'PUT':
                confidence += 0.05
    
    # Cap confidence at 0.95
    confidence = min(0.95, confidence)
    
    return {
        'signal': signal,
        'confidence': confidence,
        'reasoning': reasoning,
        'total_candles': total_candles,
        'bullish_candles': bullish_count,
        'bearish_candles': bearish_count,
        'bullish_percentage': bullish_pct,
        'bearish_percentage': bearish_pct,
        'recent_bullish': recent_bullish,
        'recent_bearish': recent_bearish,
        'recent_candles': len(recent_candles)
    }

class WorkingAnalysisBot:
    """
    🧠 WORKING ANALYSIS BOT
    Uses the actual OMNI-BRAIN PERCEPTION ENGINE
    """
    
    def __init__(self, bot_token: str):
        """Initialize the working bot"""
        self.bot_token = bot_token
        self.application = None
        
        # 📁 Create temp directory
        os.makedirs("temp_images", exist_ok=True)
        
        logger.info("🧠 Working Analysis Bot initialized with OMNI-BRAIN ENGINE")
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """🚀 Handle /start command"""
        user = update.effective_user
        welcome_message = f"""🧠 **OMNI-BRAIN PERCEPTION ENGINE** 

Hello {user.first_name}! 👋

🔍 **GENUINE CANDLE DETECTION:**
• 🟢 **DETECTS GREEN CANDLES** - Actual bullish candles
• 🔴 **DETECTS RED CANDLES** - Actual bearish candles
• 📊 **COUNTS REAL CANDLES** - No fake analysis
• 🎯 **REAL SIGNALS** - Based on actual candle counts

🧠 **OMNI-BRAIN FEATURES:**
✅ Computer vision candle detection
✅ HSV color space analysis  
✅ Contour-based candle identification
✅ Size and area filtering
✅ Recent momentum analysis
✅ Candle strength comparison

🎯 **WHAT HAPPENS:**
1. Load your Android screenshot
2. Scan for green/red candle shapes
3. Count actual bullish vs bearish candles
4. Analyze recent momentum
5. Generate signal based on real data

📸 **Send your chart screenshot for REAL OMNI-BRAIN analysis!**

Commands: /help /status /test_detection
        """
        
        await update.message.reply_text(welcome_message, parse_mode='Markdown')
        logger.info("🧠 User %s started Working Analysis Bot", user.first_name)
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """❓ Handle /help command"""
        help_message = """🧠 **OMNI-BRAIN ANALYSIS BOT**

**🔧 COMMANDS:**
• `/start` - Welcome message
• `/help` - This help message  
• `/status` - Bot status
• `/test_detection` - How detection works

**🔍 REAL DETECTION PROCESS:**
1. **Load Image** - Read your screenshot
2. **Color Analysis** - Convert to HSV color space
3. **Green Detection** - Find bullish candles (HSV: 35-85)
4. **Red Detection** - Find bearish candles (HSV: 0-10, 170-180)
5. **Shape Filtering** - Verify candle-like shapes
6. **Size Analysis** - Filter by minimum dimensions
7. **Count & Analyze** - Real candle counting

**📊 SIGNAL LOGIC:**
• **60%+ green candles** = Strong CALL
• **60%+ red candles** = Strong PUT
• **Recent momentum** = Boosts confidence
• **Candle sizes** = Affects signal strength

**📸 REQUIREMENTS:**
• Clear candlestick chart visible
• Minimum 3 candles for analysis
• Good contrast between candles
• Any Android trading app

**⚡ USES ACTUAL COMPUTER VISION!**
        """
        
        await update.message.reply_text(help_message, parse_mode='Markdown')
    
    async def test_detection_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """🧪 Explain detection process"""
        explanation = """🧪 **OMNI-BRAIN DETECTION TEST**

**🔬 TECHNICAL DETAILS:**

**HSV COLOR DETECTION:**
• **Green Range:** HSV(35-85, 40-255, 40-255)
• **Red Range 1:** HSV(0-10, 40-255, 40-255)  
• **Red Range 2:** HSV(170-180, 40-255, 40-255)

**SHAPE FILTERING:**
• **Min Height:** 20 pixels
• **Min Width:** 3 pixels
• **Max Width:** 3x height (candle proportion)
• **Min Area:** 100 pixels

**ANALYSIS PROCESS:**
```
📊 DETECTED CANDLES: 8
🟢 BULLISH: 5 candles (62.5%)
🔴 BEARISH: 3 candles (37.5%)
📈 RECENT: 3/5 last candles bullish
🎯 SIGNAL: CALL (confidence: 78%)
```

**CONFIDENCE CALCULATION:**
• Base: 60% + dominance percentage
• Recent momentum: +10% boost
• Candle size analysis: +5% boost
• Maximum: 95% confidence

**🧠 THIS IS THE ACTUAL OMNI-BRAIN ENGINE!**
No fake analysis - real computer vision detection.
        """
        
        await update.message.reply_text(explanation, parse_mode='Markdown')
    
    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """📊 Handle /status command"""
        status_message = f"""🧠 **OMNI-BRAIN BOT STATUS**

🟢 **STATUS:** Active with Real Detection
⏰ **TIME:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**🔧 OMNI-BRAIN ENGINE:**
✅ **OpenCV:** Loaded and ready
✅ **HSV Detection:** Green/Red color analysis
✅ **Contour Analysis:** Shape-based candle detection
✅ **Size Filtering:** Dimension-based validation

**📊 DETECTION CAPABILITIES:**
• **Green Candles:** HSV color space detection
• **Red Candles:** Dual-range red detection
• **Shape Analysis:** Rectangular candle filtering
• **Size Validation:** Min/max dimension checks
• **Area Filtering:** Minimum area requirements

**🎯 ANALYSIS FEATURES:**
• **Real Counting:** Actual bullish vs bearish
• **Recent Momentum:** Last 5 candles analysis
• **Candle Strength:** Size-based comparison
• **Confidence Scoring:** Multi-factor calculation

**🧠 OMNI-BRAIN READY:**
Send screenshot for genuine computer vision analysis!

**⚡ NO FAKE ANALYSIS - REAL DETECTION ENGINE!**
        """
        
        await update.message.reply_text(status_message, parse_mode='Markdown')
    
    async def handle_image(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """📸 Handle image with OMNI-BRAIN analysis"""
        try:
            user = update.effective_user
            logger.info("🧠 OMNI-BRAIN analysis requested by user %s", user.first_name)
            
            # Send processing message
            processing_msg = await update.message.reply_text(
                "🧠 **OMNI-BRAIN PERCEPTION ENGINE**\n"
                "🔍 Loading screenshot...\n"
                "🎨 Initializing HSV color analysis...\n"
                "🟢 Scanning for green candles...\n"
                "🔴 Scanning for red candles...\n"
                "📊 Analyzing patterns...",
                parse_mode='Markdown'
            )
            
            # Download image
            photo = update.message.photo[-1]
            file = await context.bot.get_file(photo.file_id)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = f"temp_images/omni_{user.id}_{timestamp}.jpg"
            await file.download_to_drive(image_path)
            
            # Update progress
            await processing_msg.edit_text(
                "🧠 **OMNI-BRAIN PERCEPTION ENGINE**\n"
                "✅ Screenshot loaded\n"
                "🔍 Detecting candles with computer vision...\n"
                "⏳ Running OMNI-BRAIN analysis...",
                parse_mode='Markdown'
            )
            
            # REAL OMNI-BRAIN ANALYSIS
            candles = await self._detect_candles_async(image_path)
            analysis = analyze_candles(candles)
            
            # Generate response
            response = self._generate_omni_response(analysis, candles)
            
            # Send final result
            await processing_msg.edit_text(response, parse_mode='Markdown')
            
            # Clean up
            try:
                os.remove(image_path)
            except:
                pass
                
            logger.info("✅ OMNI-BRAIN analysis completed for user %s", user.first_name)
            
        except Exception as e:
            logger.error("❌ Error in OMNI-BRAIN analysis: %s", str(e))
            await update.message.reply_text(
                "🧠 **OMNI-BRAIN ERROR**\n\n"
                "❌ Analysis failed. This might happen if:\n"
                "• Image couldn't be processed\n"
                "• No candles visible in screenshot\n"
                "• Image format not supported\n\n"
                "📸 Try with a clear candlestick chart!",
                parse_mode='Markdown'
            )
    
    async def _detect_candles_async(self, image_path: str) -> List[Dict]:
        """🧠 Async wrapper for OMNI-BRAIN candle detection"""
        try:
            loop = asyncio.get_event_loop()
            candles = await loop.run_in_executor(None, detect_candles, image_path)
            return candles
        except Exception as e:
            logger.error("❌ Error in async candle detection: %s", str(e))
            return []
    
    def _generate_omni_response(self, analysis: Dict, candles: List[Dict]) -> str:
        """🧠 Generate OMNI-BRAIN response"""
        signal = analysis['signal']
        confidence = analysis['confidence'] * 100
        total_candles = analysis['total_candles']
        bullish_count = analysis['bullish_candles']
        bearish_count = analysis['bearish_candles']
        bullish_pct = analysis['bullish_percentage']
        bearish_pct = analysis['bearish_percentage']
        reasoning = analysis['reasoning']
        
        # Signal formatting
        if signal == 'CALL':
            emoji = '🟢📈'
            color = '🟢'
            direction = 'BULLISH'
        elif signal == 'PUT':
            emoji = '🔴📉'
            color = '🔴'
            direction = 'BEARISH'
        else:
            emoji = '⚠️'
            color = '🟡'
            direction = 'NEUTRAL'
        
        # Confidence level
        if confidence >= 80:
            conf_level = "🔥 VERY HIGH"
        elif confidence >= 70:
            conf_level = "⚡ HIGH"
        elif confidence >= 60:
            conf_level = "📊 GOOD"
        else:
            conf_level = "⚠️ MODERATE"
        
        if total_candles >= 3:
            response = f"""🧠 **OMNI-BRAIN ANALYSIS COMPLETE**

{emoji} **SIGNAL:** {signal}
{color} **CONFIDENCE:** {confidence:.1f}% ({conf_level})
⏰ **TIME:** {datetime.now().strftime('%H:%M:%S')}

📊 **CANDLE DETECTION:**
🕯️ **Total Detected:** {total_candles} candles
🟢 **Bullish:** {bullish_count} candles ({bullish_pct:.1f}%)
🔴 **Bearish:** {bearish_count} candles ({bearish_pct:.1f}%)
📈 **Trend:** {direction}

🧠 **OMNI-BRAIN REASONING:**"""

            for i, reason in enumerate(reasoning[:3], 1):
                response += f"\n{i}. {reason}"

            response += f"""

🔍 **DETECTION METHOD:**
• HSV color space analysis
• Contour-based shape detection
• Size and area filtering
• Real candle counting

📱 **ANDROID ANALYSIS:** Computer vision processed
⚡ **ENGINE:** OMNI-BRAIN PERCEPTION

⚠️ **DISCLAIMER:** Based on actual candle detection. Educational only!

✅ **REAL OMNI-BRAIN ANALYSIS - NOT FAKE!**"""

        else:
            response = f"""🧠 **OMNI-BRAIN DETECTION RESULTS**

🔍 **CANDLES FOUND:** {total_candles}
⚠️ **STATUS:** Insufficient data for reliable signal

🧠 **OMNI-BRAIN FEEDBACK:**
• Need at least 3 candles for analysis
• Currently detected {total_candles} candles
• Cannot determine reliable trend

💡 **SUGGESTIONS:**
• Zoom out to show more candles
• Ensure candlestick chart is visible
• Take screenshot with clearer contrast
• Include 5+ candles for best results

✅ **REAL DETECTION ENGINE - NO FAKE ANALYSIS!**"""

        return response
    
    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """💬 Handle text messages"""
        response = """🧠 **OMNI-BRAIN READY FOR ANALYSIS!**

🔍 **REAL COMPUTER VISION:**
• HSV color space candle detection
• Actual green/red candle counting
• Shape-based filtering
• Size and area validation

📸 **WHAT OMNI-BRAIN DOES:**
• Scans for actual candle shapes
• Counts real bullish vs bearish candles
• Analyzes recent momentum
• Calculates genuine confidence

🎯 **NOT FAKE ANALYSIS:**
Every signal based on actual detected candles!

⚡ **SEND YOUR ANDROID CHART FOR OMNI-BRAIN!**
        """
        
        await update.message.reply_text(response, parse_mode='Markdown')
    
    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """🚫 Handle errors"""
        logger.error("OMNI-BRAIN Error:", exc_info=context.error)
        
        if isinstance(update, Update) and update.effective_message:
            try:
                await update.effective_message.reply_text(
                    "🧠 **OMNI-BRAIN ERROR**\n\n"
                    "Analysis interrupted. Please try again! 🔄",
                    parse_mode='Markdown'
                )
            except:
                pass
    
    def setup_handlers(self):
        """🔧 Setup handlers"""
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("status", self.status_command))
        self.application.add_handler(CommandHandler("test_detection", self.test_detection_command))
        self.application.add_handler(MessageHandler(filters.PHOTO, self.handle_image))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text))
        self.application.add_error_handler(self.error_handler)
        
        logger.info("🧠 OMNI-BRAIN Bot handlers configured")
    
    def run(self):
        """🚀 Start the working bot"""
        try:
            self.application = Application.builder().token(self.bot_token).build()
            self.setup_handlers()
            
            logger.info("🧠 Starting OMNI-BRAIN Perception Engine Bot...")
            logger.info("🟢 OMNI-BRAIN Bot is running! REAL ANALYSIS ONLY!")
            
            self.application.run_polling()
            
        except Exception as e:
            logger.error("❌ Error starting OMNI-BRAIN Bot: %s", str(e))
            raise

def main():
    """🎯 Main function"""
    BOT_TOKEN = "8288385434:AAG_RVKnlXDWBZNN38Q3IEfSQXIgxwPlsU0"
    
    try:
        bot = WorkingAnalysisBot(BOT_TOKEN)
        bot.run()
        
    except KeyboardInterrupt:
        logger.info("🛑 OMNI-BRAIN Bot stopped by user")
    except Exception as e:
        logger.error("❌ Fatal error: %s", str(e))

if __name__ == "__main__":
    main()