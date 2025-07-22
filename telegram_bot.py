import requests
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional
from config import Config
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TelegramSignalBot:
    def __init__(self):
        self.bot_token = Config.TELEGRAM_BOT_TOKEN
        self.chat_id = Config.TELEGRAM_CHAT_ID
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        
    def send_signal(self, analysis_result: Dict) -> bool:
        """Send trading signal to Telegram group"""
        try:
            if not self._validate_result(analysis_result):
                logger.error("Invalid analysis result format")
                return False
            
            message = self._format_signal_message(analysis_result)
            
            # Send message to Telegram
            success = self._send_telegram_message(message)
            
            if success:
                logger.info(f"Signal sent successfully: {analysis_result['signal']}")
            else:
                logger.error("Failed to send signal to Telegram")
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending Telegram signal: {str(e)}")
            return False
    
    def _validate_result(self, result: Dict) -> bool:
        """Validate analysis result structure"""
        required_fields = ['signal', 'reason', 'confidence', 'timestamp']
        return all(field in result for field in required_fields)
    
    def _format_signal_message(self, result: Dict) -> str:
        """Format the analysis result into a professional Telegram message"""
        
        # Get current time in specified timezone
        utc_time = datetime.now(timezone.utc)
        local_time = utc_time + timedelta(hours=Config.TIMEZONE_OFFSET)
        time_str = local_time.strftime("%H:%M")
        
        # Signal emoji mapping
        signal_emojis = {
            'CALL': '📈',
            'PUT': '📉',
            'NO TRADE': '⏸️'
        }
        
        # Confidence level emojis
        confidence = result['confidence']
        if confidence >= 90:
            confidence_emoji = '🔥'
        elif confidence >= 80:
            confidence_emoji = '🔒'
        elif confidence >= 70:
            confidence_emoji = '⚡'
        else:
            confidence_emoji = '⚠️'
        
        # Build the message
        message_lines = [
            "🧠 **COSMIC AI SIGNAL**",
            "",
            f"🕒 **Time:** {time_str} (UTC+{Config.TIMEZONE_OFFSET})",
            f"{signal_emojis.get(result['signal'], '❓')} **Signal:** `{result['signal']}`",
            f"📊 **Reason:** {result['reason']}",
            f"{confidence_emoji} **Confidence:** `{result['confidence']}%`",
        ]
        
        # Add analysis details if available
        if 'analysis_details' in result:
            details = result['analysis_details']
            message_lines.extend([
                "",
                "📋 **Analysis Details:**"
            ])
            
            # Momentum info
            if 'momentum' in details:
                momentum = details['momentum']
                momentum_emoji = '🚀' if momentum['direction'] == 'bullish' else '🔻' if momentum['direction'] == 'bearish' else '➡️'
                message_lines.append(f"{momentum_emoji} Momentum: {momentum['direction'].title()} ({momentum['strength']:.1f}%)")
            
            # Trend info
            if 'trend' in details:
                trend = details['trend']
                trend_emoji = '📈' if trend['direction'] == 'uptrend' else '📉' if trend['direction'] == 'downtrend' else '↔️'
                message_lines.append(f"{trend_emoji} Trend: {trend['direction'].title()}")
            
            # Patterns detected
            if 'patterns' in details and details['patterns']:
                patterns_str = ', '.join(details['patterns']).replace('_', ' ').title()
                message_lines.append(f"🎯 Patterns: {patterns_str}")
            
            # Psychology
            if 'psychology' in details:
                psychology = details['psychology']
                sentiment_emoji = '😊' if psychology['sentiment'] == 'bullish' else '😰' if psychology['sentiment'] == 'bearish' else '😐'
                message_lines.append(f"{sentiment_emoji} Sentiment: {psychology['sentiment'].title()}")
        
        # Add footer
        message_lines.extend([
            "",
            "⚡ *Powered by COSMIC AI Engine*",
            f"🤖 Analysis completed at {local_time.strftime('%Y-%m-%d %H:%M:%S')}"
        ])
        
        return "\n".join(message_lines)
    
    def _send_telegram_message(self, message: str) -> bool:
        """Send message to Telegram using Bot API"""
        try:
            url = f"{self.base_url}/sendMessage"
            
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'Markdown',
                'disable_web_page_preview': True
            }
            
            response = requests.post(url, data=payload, timeout=10)
            
            if response.status_code == 200:
                response_data = response.json()
                return response_data.get('ok', False)
            else:
                logger.error(f"Telegram API error: {response.status_code} - {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error sending to Telegram: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending to Telegram: {str(e)}")
            return False
    
    def test_connection(self) -> bool:
        """Test Telegram bot connection"""
        try:
            url = f"{self.base_url}/getMe"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('ok'):
                    bot_info = data.get('result', {})
                    logger.info(f"Telegram bot connected: {bot_info.get('first_name', 'Unknown')}")
                    return True
            
            logger.error(f"Telegram connection failed: {response.text}")
            return False
            
        except Exception as e:
            logger.error(f"Error testing Telegram connection: {str(e)}")
            return False
    
    def send_startup_message(self) -> bool:
        """Send a startup notification to the group"""
        try:
            utc_time = datetime.now(timezone.utc)
            local_time = utc_time + timedelta(hours=Config.TIMEZONE_OFFSET)
            
            message = f"""🧠 **COSMIC AI BOT ONLINE**

🚀 System Status: `ACTIVE`
🕒 Started at: {local_time.strftime('%H:%M')} (UTC+{Config.TIMEZONE_OFFSET})
📊 Confidence Threshold: `{Config.CONFIDENCE_THRESHOLD}%`
🎯 Ready to analyze binary signals!

⚡ *Upload a chart to get started*"""
            
            return self._send_telegram_message(message)
            
        except Exception as e:
            logger.error(f"Error sending startup message: {str(e)}")
            return False
    
    def send_error_notification(self, error_message: str) -> bool:
        """Send error notification to admin"""
        try:
            message = f"""⚠️ **COSMIC AI ERROR**

🔴 Error occurred: `{error_message}`
🕒 Time: {datetime.now().strftime('%H:%M:%S')}

🔧 Please check system logs for details."""
            
            return self._send_telegram_message(message)
            
        except Exception as e:
            logger.error(f"Error sending error notification: {str(e)}")
            return False

# Global instance
telegram_bot = TelegramSignalBot()