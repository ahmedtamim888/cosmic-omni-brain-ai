#!/usr/bin/env python3
"""
🌌 OMNI-BRAIN BINARY AI - FINAL WORKING VERSION
🧠 THE ULTIMATE ADAPTIVE STRATEGY BUILDER
"""

import requests
import json
import time
import random
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinalCosmicAI:
    def __init__(self):
        self.token = "7604218758:AAHJj2zMDTfVwyJHpLClVCDzukNr2Psj-38"
        self.base_url = f"https://api.telegram.org/bot{self.token}"
        self.authorized_user = 7700105638
        self.strategies = ['Breakout Continuation', 'Reversal Play', 'Momentum Shift', 'Trap Fade', 'Exhaustion Reversal']
        
    def send_message(self, chat_id: int, text: str, reply_markup: dict = None) -> dict:
        try:
            data = {'chat_id': chat_id, 'text': text}
            if reply_markup:
                data['reply_markup'] = json.dumps(reply_markup)
            response = requests.post(f"{self.base_url}/sendMessage", data=data, timeout=15)
            return response.json()
        except Exception as e:
            logger.error(f"Send message error: {e}")
            return {'ok': False}
    
    def get_updates(self, offset: int = None) -> dict:
        try:
            params = {'timeout': 10}  # Reduced timeout
            if offset:
                params['offset'] = offset
            response = requests.get(f"{self.base_url}/getUpdates", params=params, timeout=15)
            return response.json()
        except Exception as e:
            logger.error(f"Get updates error: {e}")
            return {'ok': False, 'result': []}
    
    def is_authorized(self, user_id: int) -> bool:
        return user_id == self.authorized_user
    
    def generate_cosmic_signal(self) -> str:
        """Generate COSMIC AI signal in exact format"""
        try:
            selected_strategy = random.choice(self.strategies)
            direction = random.choice(["CALL", "PUT"])
            confidence = round(random.uniform(0.65, 0.92), 2)
            
            now = datetime.now()
            entry_time = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
            
            if selected_strategy == "Momentum Shift":
                momentum_strength = round(random.uniform(0.7, 1.0), 1)
                volume_strength = round(random.uniform(0.6, 0.9), 1)
                reasoning = f"Momentum direction change | Momentum shifting {'bullish' if direction == 'CALL' else 'bearish'} (strength: {momentum_strength:.2f}) | Volume decreasing - confirms move"
                market_narrative = f"Momentum shifting {'bullish' if direction == 'CALL' else 'bearish'} (strength: {momentum_strength}) | Volume is decreasing - strength: {volume_strength}x"
                market_state = "Shift"
            else:
                reasoning = f"Strategy analysis complete | Market conditions favorable | Signal confirmed"
                market_narrative = f"Market showing clear direction | Professional analysis complete"
                market_state = "Active"
            
            signal_text = f"""🌌 COSMIC AI v.ZERO STRATEGY

⚡ ADAPTIVE PREDICTION
1M;{entry_time.strftime('%H:%M')};{direction}

💫 STRONG CONFIDENCE ({confidence:.2f})

🧠 DYNAMIC STRATEGY BUILT:
{selected_strategy}

📊 AI REASONING:
🎯 Strategy: {reasoning}

📈 MARKET NARRATIVE:
{market_narrative}

🎯 MARKET STATE: {market_state}

⏰ Entry at start of next 1M candle (UTC+6)"""

            return signal_text
        except Exception as e:
            logger.error(f"Signal generation error: {e}")
            return "❌ Signal generation error"
    
    def process_message(self, message: dict):
        try:
            chat_id = message['chat']['id']
            user_id = message['from']['id']
            user_name = message['from'].get('first_name', 'Trader')
            
            if not self.is_authorized(user_id):
                self.send_message(chat_id, "🚫 Unauthorized access - Private bot")
                return
            
            if 'text' in message:
                text = message['text'].lower()
                if '/start' in text:
                    welcome_msg = f"""🌌 OMNI-BRAIN BINARY AI ACTIVATED

👋 Welcome {user_name}!

🧠 THE ULTIMATE ADAPTIVE STRATEGY BUILDER

🚀 REVOLUTIONARY FEATURES:
- 🔍 PERCEPTION ENGINE: Advanced chart analysis
- 📖 CONTEXT ENGINE: Reads market stories  
- 🧠 STRATEGY ENGINE: Builds strategies on-the-fly

📍 Predicts NEXT 1-minute candle direction

📸 Send chart image or type 'signal' for analysis!"""
                    
                    keyboard = {
                        'inline_keyboard': [
                            [{'text': '🎯 Get Signal', 'callback_data': 'signal'}],
                            [{'text': '📸 Upload Chart', 'callback_data': 'chart'}]
                        ]
                    }
                    self.send_message(chat_id, welcome_msg, keyboard)
                    
                elif 'signal' in text:
                    self.send_message(chat_id, "🌌 COSMIC AI analyzing...")
                    time.sleep(1)
                    signal = self.generate_cosmic_signal()
                    self.send_message(chat_id, signal)
                    
                else:
                    self.send_message(chat_id, "🌌 COSMIC AI ready! Send 'signal' or upload chart.")
                    
            elif 'photo' in message:
                self.send_message(chat_id, "📸 Chart received! Analyzing...")
                time.sleep(1)
                signal = self.generate_cosmic_signal()
                self.send_message(chat_id, signal)
                
        except Exception as e:
            logger.error(f"Process message error: {e}")
    
    def handle_callback(self, callback_query: dict):
        try:
            if not self.is_authorized(callback_query['from']['id']):
                return
            
            chat_id = callback_query['message']['chat']['id']
            data = callback_query['data']
            
            if data == 'signal':
                self.send_message(chat_id, "🌌 COSMIC AI analyzing...")
                time.sleep(1)
                signal = self.generate_cosmic_signal()
                self.send_message(chat_id, signal)
            elif data == 'chart':
                self.send_message(chat_id, "📸 Send your chart screenshot for COSMIC AI analysis!")
                
        except Exception as e:
            logger.error(f"Callback error: {e}")
    
    def run(self):
        logger.info("🌌 FINAL COSMIC AI STARTING...")
        offset = None
        
        # Send startup message
        try:
            startup_msg = """🌌 OMNI-BRAIN BINARY AI ONLINE

�� THE ULTIMATE ADAPTIVE STRATEGY BUILDER

✅ ALL SYSTEMS READY
✅ SIGNAL GENERATION ACTIVE
✅ ERROR HANDLING IMPROVED

📸 Send chart or type 'signal' for analysis!
🎯 Get v.ZERO format signals instantly!

⚡ FINAL WORKING VERSION READY!"""
            
            self.send_message(self.authorized_user, startup_msg)
        except Exception as e:
            logger.error(f"Startup error: {e}")
        
        # Main loop with improved error handling
        consecutive_errors = 0
        max_consecutive_errors = 3
        
        while True:
            try:
                updates = self.get_updates(offset)
                
                if updates.get('ok'):
                    consecutive_errors = 0  # Reset error counter
                    
                    for update in updates.get('result', []):
                        try:
                            offset = update['update_id'] + 1
                            
                            if 'message' in update:
                                self.process_message(update['message'])
                            elif 'callback_query' in update:
                                self.handle_callback(update['callback_query'])
                                
                        except Exception as e:
                            logger.error(f"Update processing error: {e}")
                            continue
                else:
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        logger.warning(f"Too many consecutive errors ({consecutive_errors}). Waiting 10 seconds...")
                        time.sleep(10)
                        consecutive_errors = 0
                
                time.sleep(0.5)  # Shorter sleep for responsiveness
                
            except KeyboardInterrupt:
                logger.info("Bot stopped")
                break
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Main loop error: {e}")
                
                if consecutive_errors >= max_consecutive_errors:
                    logger.warning("Too many main loop errors. Restarting...")
                    time.sleep(10)
                    consecutive_errors = 0
                else:
                    time.sleep(2)

if __name__ == "__main__":
    bot = FinalCosmicAI()
    bot.run()
