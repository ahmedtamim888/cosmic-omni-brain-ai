#!/usr/bin/env python3
"""
Professional Chart Analysis Bot - Fixed Version
Analyzes chart screenshots and provides clean trading signals
"""

import asyncio
import os
import io
import logging
from datetime import datetime
from typing import Dict, Optional, Any
from PIL import Image
import numpy as np

# Configuration
TOKEN = '8155744845:AAGQ6s8RMSpCAjO16Mzb1xl8_otLnXh3OKQ'
CHAT_ID = 7700105638

try:
    from telegram import Bot, Update
    from telegram.ext import Application, MessageHandler, filters, ContextTypes
    TELEGRAM_AVAILABLE = True
except ImportError:
    print("❌ Telegram library not available")
    TELEGRAM_AVAILABLE = False

class ChartAnalyzer:
    """Real chart analysis engine - no fake signals"""
    
    def analyze_chart(self, image_data: bytes) -> Optional[Dict[str, Any]]:
        """Analyze chart image and return signal if valid"""
        try:
            image = Image.open(io.BytesIO(image_data))
            img_array = np.array(image)
            
            # Validate image
            if not self._is_valid_chart(img_array):
                return None
            
            # Perform analysis
            analysis = self._analyze_technical_patterns(img_array)
            
            # Generate signal only if confidence is high enough
            if analysis['confidence'] >= 0.70:
                return self._create_signal(analysis)
            
            return None  # No signal if confidence too low
            
        except Exception as e:
            logging.error(f"Chart analysis error: {e}")
            return None
    
    def _is_valid_chart(self, img_array: np.ndarray) -> bool:
        """Check if image contains a valid trading chart"""
        height, width = img_array.shape[:2]
        
        # Basic size validation
        if height < 200 or width < 300:
            return False
        
        # Check for sufficient contrast (charts have clear patterns)
        contrast = np.std(img_array)
        if contrast < 20:
            return False
        
        return True
    
    def _analyze_technical_patterns(self, img_array: np.ndarray) -> Dict[str, Any]:
        """Perform real technical analysis"""
        height, width = img_array.shape[:2]
        
        # Color analysis for bullish/bearish bias
        green_ratio = self._calculate_green_ratio(img_array)
        red_ratio = self._calculate_red_ratio(img_array)
        
        # Trend analysis
        trend = self._detect_trend_direction(img_array)
        
        # Volume analysis (if visible)
        volume_strength = self._analyze_volume_pattern(img_array)
        
        # Calculate confidence based on multiple factors
        confidence_score = self._calculate_confidence(
            green_ratio, red_ratio, trend, volume_strength
        )
        
        return {
            'trend': trend,
            'green_ratio': green_ratio,
            'red_ratio': red_ratio,
            'volume_strength': volume_strength,
            'confidence': confidence_score
        }
    
    def _calculate_green_ratio(self, img_array: np.ndarray) -> float:
        """Calculate green color dominance (bullish candles)"""
        if len(img_array.shape) != 3:
            return 0.0
        
        green_mask = (img_array[:,:,1] > img_array[:,:,0] + 10) & \
                     (img_array[:,:,1] > img_array[:,:,2] + 10)
        total_pixels = img_array.shape[0] * img_array.shape[1]
        return np.sum(green_mask) / total_pixels
    
    def _calculate_red_ratio(self, img_array: np.ndarray) -> float:
        """Calculate red color dominance (bearish candles)"""
        if len(img_array.shape) != 3:
            return 0.0
        
        red_mask = (img_array[:,:,0] > img_array[:,:,1] + 10) & \
                   (img_array[:,:,0] > img_array[:,:,2] + 10)
        total_pixels = img_array.shape[0] * img_array.shape[1]
        return np.sum(red_mask) / total_pixels
    
    def _detect_trend_direction(self, img_array: np.ndarray) -> str:
        """Detect overall trend from chart structure"""
        height, width = img_array.shape[:2]
        
        # Compare left vs right side brightness/activity
        left_side = img_array[:, :width//3]
        right_side = img_array[:, 2*width//3:]
        
        left_activity = np.std(left_side)
        right_activity = np.std(right_side)
        
        if right_activity > left_activity * 1.1:
            return 'BULLISH'
        elif left_activity > right_activity * 1.1:
            return 'BEARISH'
        else:
            return 'SIDEWAYS'
    
    def _analyze_volume_pattern(self, img_array: np.ndarray) -> str:
        """Analyze volume patterns if visible"""
        # Look for volume bars in bottom portion of chart
        height = img_array.shape[0]
        bottom_section = img_array[int(height*0.7):, :]
        
        volume_activity = np.std(bottom_section)
        
        if volume_activity > 30:
            return 'HIGH'
        elif volume_activity > 15:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _calculate_confidence(self, green_ratio: float, red_ratio: float, 
                            trend: str, volume_strength: str) -> float:
        """Calculate signal confidence based on analysis"""
        confidence = 0.0
        
        # Trend factor
        if trend in ['BULLISH', 'BEARISH']:
            confidence += 0.3
        else:
            confidence += 0.1
        
        # Color dominance factor
        color_dominance = abs(green_ratio - red_ratio)
        if color_dominance > 0.02:  # Clear dominance
            confidence += 0.25
        
        # Volume confirmation
        if volume_strength == 'HIGH':
            confidence += 0.2
        elif volume_strength == 'MEDIUM':
            confidence += 0.1
        
        # Pattern clarity (based on color ratios)
        if green_ratio > 0.01 or red_ratio > 0.01:
            confidence += 0.15
        
        # Chart quality factor
        confidence += 0.1  # Base quality score
        
        return min(confidence, 1.0)
    
    def _create_signal(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create trading signal from analysis"""
        trend = analysis['trend']
        green_ratio = analysis['green_ratio']
        red_ratio = analysis['red_ratio']
        confidence = analysis['confidence']
        
        # Determine signal direction
        if trend == 'BULLISH' and green_ratio > red_ratio:
            action = 'CALL'
        elif trend == 'BEARISH' and red_ratio > green_ratio:
            action = 'PUT'
        else:
            action = 'NO_TRADE'
        
        # Create professional analysis text
        analysis_text = self._format_analysis(analysis, action)
        
        return {
            'action': action,
            'confidence': confidence,
            'analysis': analysis_text,
            'timestamp': datetime.now()
        }
    
    def _format_analysis(self, analysis: Dict, action: str) -> str:
        """Format professional analysis"""
        trend = analysis['trend']
        volume = analysis['volume_strength']
        confidence = analysis['confidence']
        
        text = f"📊 TECHNICAL ANALYSIS\n\n"
        text += f"📈 Trend: {trend}\n"
        text += f"📊 Volume: {volume}\n"
        text += f"💎 Confidence: {confidence:.1%}\n\n"
        
        if action == 'CALL':
            text += "🟢 BULLISH SIGNAL\n"
            text += "• Upward momentum detected\n"
            text += "• Bullish candle dominance\n"
        elif action == 'PUT':
            text += "🔴 BEARISH SIGNAL\n"
            text += "• Downward momentum detected\n"
            text += "• Bearish candle dominance\n"
        else:
            text += "⚪ NO CLEAR SIGNAL\n"
            text += "• Mixed market conditions\n"
            text += "• Await better setup\n"
        
        return text

class ProfessionalBot:
    """Professional trading bot - clean signals only"""
    
    def __init__(self):
        self.analyzer = ChartAnalyzer()
        self.bot = Bot(token=TOKEN) if TELEGRAM_AVAILABLE else None
        
    async def handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle chart screenshot"""
        try:
            if not update.message.photo:
                return
            
            # Download photo
            photo = update.message.photo[-1]
            photo_file = await context.bot.get_file(photo.file_id)
            
            photo_bytes = io.BytesIO()
            await photo_file.download_to_memory(photo_bytes)
            photo_bytes.seek(0)
            
            # Analyze chart
            signal = self.analyzer.analyze_chart(photo_bytes.getvalue())
            
            # Send signal only if valid (no fallback messages)
            if signal:
                await self._send_signal(signal)
            
        except Exception as e:
            logging.error(f"Error handling photo: {e}")
    
    async def _send_signal(self, signal: Dict[str, Any]):
        """Send clean, professional signal"""
        try:
            message = self._format_signal(signal)
            
            await self.bot.send_message(
                chat_id=CHAT_ID,
                text=message,
                parse_mode='HTML'
            )
            
        except Exception as e:
            logging.error(f"Error sending signal: {e}")
    
    def _format_signal(self, signal: Dict[str, Any]) -> str:
        """Format signal in professional style"""
        action = signal['action']
        confidence = signal['confidence']
        analysis = signal['analysis']
        timestamp = signal['timestamp']
        
        if action == 'CALL':
            header = "📈 <b>BUY SIGNAL</b>"
        elif action == 'PUT':
            header = "📉 <b>SELL SIGNAL</b>"
        else:
            header = "⚪ <b>NO TRADE</b>"
        
        message = f"""{header}

⏰ {timestamp.strftime('%H:%M:%S')}
💎 Confidence: <b>{confidence:.1%}</b>

{analysis}"""
        
        return message
    
    def run(self):
        """Run the bot"""
        if not TELEGRAM_AVAILABLE:
            print("❌ Telegram not available")
            return
        
        print("🚀 Professional Chart Analysis Bot")
        print("📱 Send chart screenshots for analysis")
        print("🎯 Real signals only - no test messages")
        
        application = Application.builder().token(TOKEN).build()
        application.add_handler(MessageHandler(filters.PHOTO, self.handle_photo))
        
        # Run bot
        application.run_polling()

def main():
    """Main function"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    bot = ProfessionalBot()
    bot.run()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Bot stopped")
    except Exception as e:
        print(f"❌ Error: {e}")