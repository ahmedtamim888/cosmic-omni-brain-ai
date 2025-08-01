import asyncio
import requests
import json
from datetime import datetime
from typing import List, Dict
import time

class TelegramSignalSender:
    """
    📱 TELEGRAM SIGNAL SENDER
    Sends immortal trading signals to Telegram channel
    """
    
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        self.sent_signals = []
        
    def send_signal_batch(self, signals: List[Dict]) -> bool:
        """Send a batch of signals to Telegram"""
        try:
            # Format signals for Telegram
            message = self._format_signals_message(signals)
            
            # Send message
            success = self._send_message(message)
            
            if success:
                self.sent_signals.extend(signals)
                print(f"✅ Sent {len(signals)} signals to Telegram")
                return True
            else:
                print("❌ Failed to send signals to Telegram")
                return False
                
        except Exception as e:
            print(f"❌ Error sending signals: {e}")
            return False
    
    def send_individual_signal(self, signal: Dict) -> bool:
        """Send individual signal to Telegram"""
        try:
            message = self._format_individual_signal(signal)
            success = self._send_message(message)
            
            if success:
                self.sent_signals.append(signal)
                print(f"✅ Sent signal: {signal['time']} {signal['symbol']} {signal['signal']}")
                return True
            else:
                print(f"❌ Failed to send signal: {signal['symbol']}")
                return False
                
        except Exception as e:
            print(f"❌ Error sending individual signal: {e}")
            return False
    
    def _format_signals_message(self, signals: List[Dict]) -> str:
        """Format multiple signals for Telegram message"""
        
        header = "🧠 IMMORTAL QUOTEX OTC SIGNALS 🧠\n"
        header += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        header += "⚡ 999999 TRILLION YEARS OF EXPERIENCE ⚡\n"
        header += "🎯 UNBEATABLE ACCURACY - CANNOT BE DEFEATED 🎯\n"
        header += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        
        signal_lines = []
        for signal in signals:
            # Convert CALL/PUT to appropriate format
            action = "CALL" if signal['signal'] == "CALL" else "PUT"
            if action == "PUT":
                action = "PUT"
            
            signal_line = f"⏰ {signal['time']} {signal['symbol']} {action}"
            signal_lines.append(signal_line)
        
        footer = f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        footer += "📊 1-MINUTE EXPIRY | 3-5 SEC BEFORE CANDLE CLOSE\n"
        footer += "🔥 IMMORTAL PRECISION - DIVINE ACCURACY 🔥\n"
        footer += f"⚡ Generated: {datetime.now().strftime('%H:%M:%S')} ⚡"
        
        return header + "\n".join(signal_lines) + footer
    
    def _format_individual_signal(self, signal: Dict) -> str:
        """Format individual signal for Telegram"""
        
        action = "CALL" if signal['signal'] == "CALL" else "PUT"
        
        message = f"🎯 IMMORTAL SIGNAL ALERT 🎯\n"
        message += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        message += f"⏰ TIME: {signal['time']}\n"
        message += f"📈 PAIR: {signal['symbol']}\n"
        message += f"🚀 ACTION: {action}\n"
        message += f"⚡ CONFIDENCE: {signal['confidence']}%\n"
        message += f"🧠 PATTERN: {signal['pattern']}\n"
        message += f"💡 REASONING: {signal['reasoning']}\n"
        message += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        message += f"⏳ EXPIRY: 1 MINUTE\n"
        message += f"🎯 ENTRY: 3-5 seconds before candle close"
        
        return message
    
    def _send_message(self, message: str) -> bool:
        """Send message to Telegram channel"""
        try:
            url = f"{self.base_url}/sendMessage"
            
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'HTML',
                'disable_web_page_preview': True
            }
            
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                return True
            else:
                print(f"Telegram API Error: {response.status_code} - {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"Network error: {e}")
            return False
    
    def send_status_message(self, status: str) -> bool:
        """Send status update to Telegram"""
        try:
            message = f"🤖 IMMORTAL BOT STATUS 🤖\n"
            message += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            message += f"📊 {status}\n"
            message += f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            message += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            
            return self._send_message(message)
            
        except Exception as e:
            print(f"Error sending status: {e}")
            return False
    
    def send_performance_report(self, total_signals: int, accuracy: float) -> bool:
        """Send performance report"""
        try:
            message = f"📊 IMMORTAL PERFORMANCE REPORT 📊\n"
            message += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            message += f"🎯 Total Signals Sent: {total_signals}\n"
            message += f"⚡ Accuracy Rate: {accuracy:.1f}%\n"
            message += f"🏆 Status: UNDEFEATED\n"
            message += f"🧠 Experience: 999999 TRILLION YEARS\n"
            message += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            message += f"⏰ Report Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            return self._send_message(message)
            
        except Exception as e:
            print(f"Error sending performance report: {e}")
            return False
    
    def test_connection(self) -> bool:
        """Test Telegram bot connection"""
        try:
            url = f"{self.base_url}/getMe"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                bot_info = response.json()
                print(f"✅ Telegram bot connected: {bot_info['result']['first_name']}")
                return True
            else:
                print(f"❌ Telegram connection failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Telegram connection error: {e}")
            return False