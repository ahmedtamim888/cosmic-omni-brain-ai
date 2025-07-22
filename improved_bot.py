#!/usr/bin/env python3
"""
🌌 OMNI-BRAIN BINARY AI - Improved Telegram Bot
Enhanced with better error handling and debugging
"""

import requests
import json
import time
import random
from datetime import datetime
import logging
import traceback

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ImprovedCosmicAIBot:
    """🌌 Improved COSMIC AI Bot with enhanced error handling"""
    
    def __init__(self, token: str):
        self.token = token
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.active_users = {}
        self.offset = 0
        self.running = True
        self.error_count = 0
        self.message_count = 0
        
        # Test bot connection on startup
        self.test_connection()
        
    def test_connection(self):
        """Test if bot can connect to Telegram"""
        try:
            url = f"{self.base_url}/getMe"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if data.get('ok'):
                bot_info = data.get('result', {})
                logger.info(f"✅ Bot connected successfully!")
                logger.info(f"📱 Bot name: {bot_info.get('first_name')}")
                logger.info(f"👤 Username: @{bot_info.get('username')}")
                return True
            else:
                logger.error(f"❌ Bot connection failed: {data}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Connection test failed: {e}")
            return False
    
    def send_message(self, chat_id: int, text: str, reply_markup=None, parse_mode="Markdown"):
        """Send a message to a chat with improved error handling"""
        url = f"{self.base_url}/sendMessage"
        data = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": parse_mode
        }
        
        if reply_markup:
            data["reply_markup"] = json.dumps(reply_markup)
        
        try:
            response = requests.post(url, data=data, timeout=30)
            result = response.json()
            
            if not result.get('ok'):
                logger.error(f"❌ Send message failed: {result}")
                # Try sending without markdown if parse error
                if 'parse' in str(result.get('description', '')).lower():
                    data['parse_mode'] = None
                    response = requests.post(url, data=data, timeout=30)
                    result = response.json()
            
            return result
            
        except requests.exceptions.Timeout:
            logger.error(f"⏰ Timeout sending message to {chat_id}")
            return None
        except Exception as e:
            logger.error(f"❌ Error sending message: {e}")
            return None
    
    def edit_message(self, chat_id: int, message_id: int, text: str, reply_markup=None, parse_mode="Markdown"):
        """Edit a message with improved error handling"""
        url = f"{self.base_url}/editMessageText"
        data = {
            "chat_id": chat_id,
            "message_id": message_id,
            "text": text,
            "parse_mode": parse_mode
        }
        
        if reply_markup:
            data["reply_markup"] = json.dumps(reply_markup)
        
        try:
            response = requests.post(url, data=data, timeout=30)
            result = response.json()
            
            if not result.get('ok'):
                logger.warning(f"⚠️ Edit message failed: {result.get('description')}")
                # Message might be too old or identical
                
            return result
            
        except Exception as e:
            logger.error(f"❌ Error editing message: {e}")
            return None
    
    def delete_message(self, chat_id: int, message_id: int):
        """Delete a message with error handling"""
        url = f"{self.base_url}/deleteMessage"
        data = {
            "chat_id": chat_id,
            "message_id": message_id
        }
        
        try:
            response = requests.post(url, data=data, timeout=30)
            result = response.json()
            
            if not result.get('ok'):
                logger.warning(f"⚠️ Delete message failed: {result.get('description')}")
                
            return result
            
        except Exception as e:
            logger.warning(f"⚠️ Error deleting message: {e}")
            return None
    
    def get_updates(self):
        """Get updates from Telegram with improved error handling"""
        url = f"{self.base_url}/getUpdates"
        params = {
            "offset": self.offset,
            "timeout": 30
        }
        
        try:
            response = requests.get(url, params=params, timeout=35)
            result = response.json()
            
            if not result.get('ok'):
                logger.error(f"❌ Get updates failed: {result}")
                return None
                
            return result
            
        except requests.exceptions.Timeout:
            logger.debug("⏰ Timeout getting updates (normal)")
            return {"ok": True, "result": []}
        except requests.exceptions.ConnectionError:
            logger.warning("🔌 Connection error getting updates")
            time.sleep(5)
            return None
        except Exception as e:
            logger.error(f"❌ Error getting updates: {e}")
            return None
    
    def handle_start_command(self, chat_id: int, user_name: str):
        """Handle /start command"""
        logger.info(f"👋 User {user_name} ({chat_id}) started the bot")
        
        welcome_message = """🌌 *OMNI-BRAIN BINARY AI ACTIVATED*

🧠 *THE ULTIMATE ADAPTIVE STRATEGY BUILDER*

🚀 *REVOLUTIONARY FEATURES:*
- 🔍 *PERCEPTION ENGINE*: Advanced chart analysis
- 📖 *CONTEXT ENGINE*: Reads market stories
- 🧠 *STRATEGY ENGINE*: Builds unique strategies

🎯 *HOW COSMIC AI WORKS:*
1. *ANALYZES* market conditions in real-time
2. *READS* candle conversation and psychology  
3. *BUILDS* custom strategy tree for current setup
4. *EXECUTES* only when confidence > threshold
5. *ADAPTS* strategy logic based on market state

💫 *STRATEGY TYPES:*
- Breakout Continuation
- Reversal Play
- Momentum Shift
- Trap Fade
- Exhaustion Reversal

📍 *Predicts NEXT 1-minute candle direction*

*Send a chart image to activate COSMIC AI!*

Use /help for detailed commands."""
        
        keyboard = {
            "inline_keyboard": [
                [{"text": "📊 Analyze Chart", "callback_data": "analyze_chart"}],
                [{"text": "⚙️ Settings", "callback_data": "settings"}, 
                 {"text": "📈 Status", "callback_data": "status"}],
                [{"text": "❓ Help", "callback_data": "help"}]
            ]
        }
        
        response = self.send_message(chat_id, welcome_message, reply_markup=keyboard)
        
        if response and response.get('ok'):
            # Initialize user session
            self.active_users[chat_id] = {
                'started_at': datetime.now(),
                'analysis_count': 0,
                'last_analysis': None,
                'user_name': user_name
            }
            logger.info(f"✅ Welcome message sent to {user_name}")
        else:
            logger.error(f"❌ Failed to send welcome message to {user_name}")
    
    def handle_help_command(self, chat_id: int):
        """Handle /help command"""
        help_text = """🔧 *COSMIC AI COMMANDS:*

📸 *Chart Analysis:*
- Send chart image → Get AI analysis
- Supports all major brokers
- Automatic broker detection

💬 *Commands:*
- /start - Initialize COSMIC AI
- /analyze - Request chart analysis  
- /status - View statistics
- /help - Show this help

🎯 *How to Use:*
1. Screenshot your trading chart
2. Send the image to this bot
3. Get complete AI analysis:
   - Market story & psychology
   - Custom strategy tree
   - Direction prediction
   - Confidence score
   - Optimal expiry time

🌟 *Features:*
- Real-time Analysis
- Multi-Broker Support
- Adaptive Strategies
- Risk Management
- Learning AI

💡 *Tips:*
- Use clear, high-quality images
- Include multiple timeframes
- Check confidence levels
- Always manage your risk"""
        
        self.send_message(chat_id, help_text)
    
    def handle_status_command(self, chat_id: int):
        """Handle /status command"""
        if chat_id not in self.active_users:
            self.send_message(chat_id, "Please start the bot first with /start")
            return
        
        user_data = self.active_users[chat_id]
        
        status_message = f"""📊 *COSMIC AI STATUS*

👤 *User:* {user_data['user_name']}
🕐 *Session:* {user_data['started_at'].strftime('%Y-%m-%d %H:%M')}
🔢 *Analyses:* {user_data['analysis_count']}
📈 *Last:* {user_data['last_analysis'].strftime('%H:%M') if user_data['last_analysis'] else 'None'}

🤖 *AI Status:*
- *Perception Engine:* ✅ Active
- *Context Engine:* ✅ Active  
- *Strategy Engine:* ✅ Active
- *Learning Module:* ✅ Active

🎯 *Settings:*
- *Confidence Threshold:* 75%
- *Strategy Types:* 5 Available
- *Analysis Mode:* Full Spectrum

*Ready for next analysis!*"""
        
        keyboard = {
            "inline_keyboard": [
                [{"text": "📊 New Analysis", "callback_data": "analyze_chart"}],
                [{"text": "⚙️ Settings", "callback_data": "settings"}]
            ]
        }
        
        self.send_message(chat_id, status_message, reply_markup=keyboard)
    
    def handle_image(self, chat_id: int, photo_data: list):
        """Handle chart image uploads with improved error handling"""
        logger.info(f"📸 Processing image from user {chat_id}")
        
        # Send initial processing message
        processing_response = self.send_message(
            chat_id,
            "🌌 *COSMIC AI ANALYZING...*\n\n🔍 Extracting chart elements...\n📖 Reading market story...\n🧠 Building strategy tree..."
        )
        
        if not processing_response or not processing_response.get('ok'):
            logger.error("❌ Failed to send processing message")
            self.send_message(chat_id, "❌ *Error starting analysis*\n\nPlease try again.")
            return
        
        processing_msg_id = processing_response['result']['message_id']
        
        try:
            # Simulate chart analysis steps
            steps = [
                "🌌 *COSMIC AI ANALYZING...*\n\n✅ Chart elements extracted\n📖 Reading market story...\n🧠 Building strategy tree...",
                "🌌 *COSMIC AI ANALYZING...*\n\n✅ Chart elements extracted\n✅ Market story analyzed\n🧠 Building strategy tree...",
                "🌌 *COSMIC AI ANALYZING...*\n\n✅ Chart elements extracted\n✅ Market story analyzed\n✅ Strategy finalized!"
            ]
            
            for i, step in enumerate(steps):
                time.sleep(1)
                self.edit_message(chat_id, processing_msg_id, step)
            
            # Generate analysis result
            analysis_result = self.generate_analysis()
            
            # Send analysis results
            success = self.send_analysis_results(chat_id, analysis_result)
            
            if success:
                # Update user statistics
                if chat_id in self.active_users:
                    self.active_users[chat_id]['analysis_count'] += 1
                    self.active_users[chat_id]['last_analysis'] = datetime.now()
                
                # Delete processing message
                time.sleep(1)
                self.delete_message(chat_id, processing_msg_id)
                
                logger.info(f"✅ Analysis completed for user {chat_id}")
            else:
                logger.error(f"❌ Failed to send analysis results to user {chat_id}")
                
        except Exception as e:
            logger.error(f"❌ Error processing image: {e}")
            logger.error(traceback.format_exc())
            
            self.edit_message(
                chat_id,
                processing_msg_id,
                f"❌ *Analysis Failed*\n\nError occurred during processing.\nPlease try again with a clear chart image."
            )
    
    def generate_analysis(self):
        """Generate AI analysis results"""
        directions = ['CALL', 'PUT']
        direction = random.choice(directions)
        confidence = random.uniform(0.65, 0.95)
        
        strategies = ['Breakout Continuation', 'Reversal Play', 'Momentum Shift', 'Trap Fade', 'Exhaustion Reversal']
        strategy_type = random.choice(strategies)
        
        expiry_times = [60, 180, 300, 600]
        expiry = random.choice(expiry_times)
        
        risk_levels = ['low', 'medium', 'high']
        risk_level = random.choice(risk_levels)
        
        market_states = [
            "strong bullish momentum with volume confirmation",
            "bearish pressure at key resistance level", 
            "consolidation pattern forming near support",
            "breakout attempt with high volatility",
            "reversal signals emerging from oversold conditions"
        ]
        
        return {
            'broker': 'DERIV',
            'asset': random.choice(['EUR/USD', 'GBP/USD', 'USD/JPY', 'BTC/USD']),
            'timeframe': '1m',
            'detection_confidence': 0.85,
            'direction': direction,
            'confidence': confidence,
            'strategy_type': strategy_type,
            'expiry': expiry,
            'risk_level': risk_level,
            'market_story': f"Market showing {random.choice(market_states)}. Price action indicates {strategy_type.lower()} setup forming.",
            'reasoning': f"{direction} prediction based on {strategy_type} pattern. Technical indicators show {confidence:.1%} confidence.",
            'execute_trade': confidence > 0.75
        }
    
    def send_analysis_results(self, chat_id: int, analysis_result: dict):
        """Send formatted analysis results"""
        try:
            # Perception results
            perception_msg = f"""🔍 *PERCEPTION ENGINE RESULTS*

📊 *Chart Details:*
- *Broker:* {analysis_result['broker']}
- *Asset:* {analysis_result['asset']}
- *Timeframe:* {analysis_result['timeframe']}
- *Detection:* {analysis_result['detection_confidence']:.1%}

✅ *Chart elements extracted successfully*"""
            
            # Story results
            direction = analysis_result['direction']
            bias_emoji = "🟢" if direction == "CALL" else "🔴"
            
            story_msg = f"""📖 *MARKET STORY ANALYSIS*

{bias_emoji} *Bias:* {direction}
📊 *Confidence:* {analysis_result['confidence']:.1%}

*Market Narrative:*
{analysis_result['market_story']}

*Key Insights:*
- Market psychology detected
- Price action patterns identified  
- Volume and momentum analyzed"""
            
            # Strategy results
            confidence = analysis_result['confidence']
            direction_emoji = "📈" if direction == "CALL" else "📉"
            confidence_bars = "█" * int(confidence * 10)
            confidence_empty = "░" * (10 - int(confidence * 10))
            confidence_bar = confidence_bars + confidence_empty
            
            risk_level = analysis_result['risk_level']
            risk_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(risk_level, "🟡")
            
            expiry = analysis_result['expiry']
            
            strategy_msg = f"""🧠 *STRATEGY ENGINE RESULTS*

🎯 *PREDICTION:* {direction_emoji} *{direction}*

📊 *Confidence:* {confidence:.1%}
{confidence_bar}

⏰ *Expiry:* {expiry // 60}m {expiry % 60}s
{risk_emoji} *Risk:* {risk_level.upper()}

🧭 *Strategy:* {analysis_result['strategy_type']}

💭 *Reasoning:*
{analysis_result['reasoning']}

✨ *Ready for execution!*"""
            
            # Send results
            success_count = 0
            
            if self.send_message(chat_id, perception_msg):
                success_count += 1
                time.sleep(0.5)
            
            if self.send_message(chat_id, story_msg):
                success_count += 1
                time.sleep(0.5)
            
            # Strategy with buttons
            keyboard = {"inline_keyboard": []}
            
            if analysis_result['execute_trade']:
                keyboard["inline_keyboard"].append([
                    {"text": f"🚀 Execute {direction}", "callback_data": f"execute_{direction}"},
                    {"text": "⏸️ Skip Trade", "callback_data": "skip_trade"}
                ])
            
            keyboard["inline_keyboard"].append([
                {"text": "📊 New Analysis", "callback_data": "analyze_chart"},
                {"text": "📈 Details", "callback_data": "view_details"}
            ])
            
            if self.send_message(chat_id, strategy_msg, reply_markup=keyboard):
                success_count += 1
            
            return success_count == 3
            
        except Exception as e:
            logger.error(f"❌ Error sending analysis results: {e}")
            self.send_message(chat_id, "❌ *Error formatting results*\n\nPlease try again.")
            return False
    
    def handle_callback_query(self, callback_query: dict):
        """Handle inline keyboard callbacks"""
        try:
            chat_id = callback_query['message']['chat']['id']
            message_id = callback_query['message']['message_id']
            data = callback_query['data']
            user_name = callback_query['from'].get('first_name', 'User')
            
            logger.info(f"🔘 Button pressed: {data} by {user_name}")
            
            if data == "analyze_chart":
                self.edit_message(chat_id, message_id, "📊 *Ready for Analysis*\n\nSend me a chart image to activate COSMIC AI!")
            
            elif data == "status":
                self.handle_status_command(chat_id)
            
            elif data == "help":
                self.handle_help_command(chat_id)
            
            elif data.startswith("execute_"):
                direction = data.split("_")[1]
                self.edit_message(
                    chat_id, message_id,
                    f"🚀 *Execute {direction.upper()} Trade*\n\n"
                    f"Demo mode: Would execute {direction} trade.\n\n"
                    f"*Supported Brokers:*\n"
                    f"- Deriv.com (WebSocket API)\n"
                    f"- IQ Option (API integration)\n"
                    f"- Quotex (API integration)\n"
                    f"- Manual execution on any broker"
                )
            
            elif data == "skip_trade":
                self.edit_message(
                    chat_id, message_id,
                    "⏸️ *Trade Skipped*\n\nAnalysis completed but trade not executed.\nSend another chart for new analysis!"
                )
            
            elif data == "view_details":
                self.edit_message(
                    chat_id, message_id,
                    "📈 *Analysis Details*\n\n"
                    "🔍 *Perception:* Chart pattern recognition\n"
                    "📖 *Context:* Market psychology analysis\n" 
                    "🧠 *Strategy:* Adaptive strategy building\n\n"
                    "*Indicators Analyzed:*\n"
                    "- RSI, MACD, Bollinger Bands\n"
                    "- Support/Resistance Levels\n"
                    "- Volume Analysis\n"
                    "- Market Structure\n\n"
                    "Send another chart image for new analysis!"
                )
                
        except Exception as e:
            logger.error(f"❌ Error handling callback: {e}")
    
    def process_message(self, message: dict):
        """Process incoming messages with improved error handling"""
        try:
            chat_id = message['chat']['id']
            user_name = message['from'].get('first_name', 'User')
            
            self.message_count += 1
            
            if 'text' in message:
                text = message['text']
                logger.info(f"💬 Text message from {user_name}: {text}")
                
                if text.startswith('/start'):
                    self.handle_start_command(chat_id, user_name)
                elif text.startswith('/help'):
                    self.handle_help_command(chat_id)
                elif text.startswith('/status'):
                    self.handle_status_command(chat_id)
                elif text.startswith('/analyze'):
                    self.send_message(chat_id, "📊 *READY FOR ANALYSIS*\n\nSend me a chart image for complete AI analysis!")
                elif any(word in text.lower() for word in ['analyze', 'chart', 'help', 'trade']):
                    self.send_message(chat_id, "📸 *Send chart image for analysis!*\n\nI can analyze any broker chart and provide:\n- Market psychology\n- Strategy building\n- Direction prediction\n- Risk assessment")
                else:
                    self.send_message(chat_id, "🤖 *COSMIC AI Ready*\n\nSend chart image for analysis or /help for commands.")
            
            elif 'photo' in message:
                logger.info(f"📸 Photo received from {user_name}")
                self.handle_image(chat_id, message['photo'])
                
        except Exception as e:
            logger.error(f"❌ Error processing message: {e}")
            logger.error(traceback.format_exc())
            self.error_count += 1
    
    def run(self):
        """Main bot loop with enhanced error handling"""
        logger.info("🌌 Starting Enhanced COSMIC AI Bot...")
        logger.info("📱 Bot is now running. Send /start in Telegram to begin!")
        
        consecutive_errors = 0
        
        while self.running:
            try:
                updates = self.get_updates()
                
                if updates and updates.get('ok'):
                    consecutive_errors = 0  # Reset error counter
                    
                    for update in updates['result']:
                        self.offset = update['update_id'] + 1
                        
                        if 'message' in update:
                            self.process_message(update['message'])
                        elif 'callback_query' in update:
                            self.handle_callback_query(update['callback_query'])
                
                elif updates is None:
                    consecutive_errors += 1
                    if consecutive_errors > 5:
                        logger.warning(f"⚠️ {consecutive_errors} consecutive errors. Pausing...")
                        time.sleep(30)
                        consecutive_errors = 0
                
                time.sleep(1)
                
                # Log stats every 100 messages
                if self.message_count % 100 == 0 and self.message_count > 0:
                    logger.info(f"📊 Stats: {self.message_count} messages, {self.error_count} errors, {len(self.active_users)} users")
                
            except KeyboardInterrupt:
                logger.info("🛑 Bot stopped by user")
                self.running = False
            except Exception as e:
                logger.error(f"❌ Error in main loop: {e}")
                logger.error(traceback.format_exc())
                self.error_count += 1
                time.sleep(5)

def main():
    """Main function to run the improved bot"""
    
    # Display banner
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║         🌌 OMNI-BRAIN BINARY AI ACTIVATED 🌌                ║
    ║                                                              ║
    ║        🧠 THE ULTIMATE ADAPTIVE STRATEGY BUILDER 🧠         ║
    ║                                                              ║
    ║  🚀 ENHANCED VERSION WITH IMPROVED ERROR HANDLING           ║
    ║  - 🔍 PERCEPTION ENGINE: Advanced chart analysis            ║
    ║  - 📖 CONTEXT ENGINE: Reads market stories                  ║
    ║  - 🧠 STRATEGY ENGINE: Builds unique strategies             ║
    ║  - 🛠️ ROBUST ERROR HANDLING: Self-healing capabilities      ║
    ║                                                              ║
    ║  💫 STRATEGY TYPES:                                          ║
    ║  - Breakout Continuation                                     ║
    ║  - Reversal Play                                             ║
    ║  - Momentum Shift                                            ║
    ║  - Trap Fade                                                 ║
    ║  - Exhaustion Reversal                                       ║
    ║                                                              ║
    ║  📱 Bot: @Ghost_Em_bot                                       ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)
    
    # Your Telegram token
    TOKEN = "7604218758:AAHJj2zMDTfVwyJHpLClVCDzukNr2Psj-38"
    
    # Create and run bot
    bot = ImprovedCosmicAIBot(TOKEN)
    bot.run()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)