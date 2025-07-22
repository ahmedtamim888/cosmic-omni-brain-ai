import asyncio
import logging
from datetime import datetime
from typing import Optional
import requests
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TelegramBot:
    """Handles sending trading signals to Telegram group"""
    
    def __init__(self):
        self.bot_token = Config.TELEGRAM_BOT_TOKEN
        self.chat_id = Config.TELEGRAM_CHAT_ID
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        
    def send_signal(self, signal) -> bool:
        """Send trading signal to Telegram group"""
        try:
            if not self.chat_id:
                logger.warning("Telegram chat ID not configured")
                return False
                
            message = self._format_signal_message(signal)
            
            # Send message via Telegram API
            url = f"{self.base_url}/sendMessage"
            
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'HTML',
                'disable_web_page_preview': True
            }
            
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                logger.info(f"Signal sent successfully: {signal.signal}")
                return True
            else:
                logger.error(f"Failed to send signal: {response.status_code} - {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error sending signal: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending signal: {e}")
            return False
    
    def _format_signal_message(self, signal) -> str:
        """Format the signal into a beautiful Telegram message"""
        
        # Get current time in UTC+6
        current_time = signal.entry_time.replace(tzinfo=Config.TIMEZONE)
        time_str = current_time.strftime("%H:%M")
        
        # Determine signal emoji
        if signal.signal == "CALL":
            signal_emoji = "📈"
            signal_color = "🟢"
        elif signal.signal == "PUT":
            signal_emoji = "📉"
            signal_color = "🔴"
        else:
            signal_emoji = "⏸️"
            signal_color = "🟡"
        
        # Confidence level emoji
        if signal.confidence >= 90:
            confidence_emoji = "🔥"
        elif signal.confidence >= 85:
            confidence_emoji = "⚡"
        elif signal.confidence >= 80:
            confidence_emoji = "✅"
        else:
            confidence_emoji = "⚠️"
        
        # Format the message
        message = f"""🧠 <b>COSMIC AI SIGNAL</b> 🚀
        
🕒 <b>Time:</b> {time_str} (UTC+6)
{signal_emoji} <b>Signal:</b> {signal_color} <b>{signal.signal}</b>
📊 <b>Reason:</b> {signal.reasoning}
🎯 <b>Strategy:</b> {signal.strategy.replace('_', ' ').title()}
🧭 <b>Psychology:</b> {signal.market_psychology.replace('_', ' ').title()}
{confidence_emoji} <b>Confidence:</b> {signal.confidence:.1f}%
⏰ <b>Timeframe:</b> {signal.timeframe}

<i>🌌 Powered by Cosmic AI Engine</i>"""

        return message
    
    def send_test_message(self) -> bool:
        """Send a test message to verify bot configuration"""
        try:
            if not self.chat_id:
                logger.warning("Telegram chat ID not configured")
                return False
                
            test_message = """🧠 <b>COSMIC AI BOT TEST</b> 🚀

✅ Bot is working correctly!
🔗 Connection established
📡 Ready to send signals

<i>🌌 Cosmic AI Binary Signal Bot</i>"""
            
            url = f"{self.base_url}/sendMessage"
            
            payload = {
                'chat_id': self.chat_id,
                'text': test_message,
                'parse_mode': 'HTML'
            }
            
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                logger.info("Test message sent successfully")
                return True
            else:
                logger.error(f"Failed to send test message: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending test message: {e}")
            return False
    
    def send_analysis_summary(self, signal, additional_info: dict = None) -> bool:
        """Send detailed analysis summary (optional extended message)"""
        try:
            if not self.chat_id:
                return False
                
            # Extended analysis message
            message = f"""📊 <b>DETAILED ANALYSIS</b> 📈

🎯 <b>Primary Signal:</b> {signal.signal}
🔍 <b>Strategy Used:</b> {signal.strategy.replace('_', ' ').title()}
💭 <b>Market Psychology:</b> {signal.market_psychology.replace('_', ' ').title()}
📈 <b>Confidence Level:</b> {signal.confidence:.1f}%

<b>📋 Analysis Details:</b>
• {signal.reasoning}

⚡ <b>Entry Recommendation:</b> Next 1-minute candle
🕐 <b>Signal Time:</b> {signal.entry_time.strftime('%H:%M:%S')} UTC+6

<i>⚠️ Risk Management: Use proper position sizing</i>
<i>🌌 Cosmic AI Engine v1.0</i>"""

            if additional_info:
                patterns = additional_info.get('patterns', [])
                if patterns:
                    pattern_names = [p['name'] for p in patterns[:3]]  # Top 3 patterns
                    message += f"\n\n🔍 <b>Detected Patterns:</b>\n• " + "\n• ".join(pattern_names)
            
            url = f"{self.base_url}/sendMessage"
            
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            
            response = requests.post(url, json=payload, timeout=10)
            
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Error sending analysis summary: {e}")
            return False
    
    def get_bot_info(self) -> Optional[dict]:
        """Get bot information to verify token is valid"""
        try:
            url = f"{self.base_url}/getMe"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get bot info: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting bot info: {e}")
            return None
    
    def validate_configuration(self) -> dict:
        """Validate bot configuration and return status"""
        status = {
            'bot_token_valid': False,
            'chat_id_configured': False,
            'bot_info': None,
            'connection_working': False
        }
        
        # Check if bot token is configured
        if self.bot_token and self.bot_token != "YOUR_BOT_TOKEN_HERE":
            status['bot_token_valid'] = True
            
            # Try to get bot info
            bot_info = self.get_bot_info()
            if bot_info and bot_info.get('ok'):
                status['bot_info'] = bot_info.get('result', {})
                status['connection_working'] = True
        
        # Check if chat ID is configured
        if self.chat_id:
            status['chat_id_configured'] = True
        
        return status

# Convenience function for quick signal sending
def send_signal_to_telegram(signal, additional_info: dict = None) -> bool:
    """Quick function to send signal to Telegram"""
    bot = TelegramBot()
    
    # Send main signal
    main_sent = bot.send_signal(signal)
    
    # Optionally send detailed analysis (only for high confidence signals)
    if main_sent and signal.confidence >= 85 and additional_info:
        bot.send_analysis_summary(signal, additional_info)
    
    return main_sent

# Example usage and testing
if __name__ == "__main__":
    from datetime import datetime
    
    # Create a test signal as a simple object
    class TestSignal:
        def __init__(self):
            self.signal = "CALL"
            self.confidence = 92.4
            self.reasoning = "Breakout + Momentum Surge"
            self.strategy = "BREAKOUT_CONTINUATION"
            self.market_psychology = "STRONG_BULLISH_SENTIMENT"
            self.entry_time = datetime.now()
            self.timeframe = "1M"
    
    test_signal = TestSignal()
    
    # Test the bot
    bot = TelegramBot()
    
    # Validate configuration
    print("Validating Telegram bot configuration...")
    status = bot.validate_configuration()
    print(f"Configuration status: {status}")
    
    # Send test message if configured
    if status['bot_token_valid'] and status['chat_id_configured']:
        print("Sending test signal...")
        success = bot.send_signal(test_signal)
        print(f"Signal sent: {success}")
    else:
        print("Bot not properly configured. Please set TELEGRAM_CHAT_ID environment variable.")