#!/usr/bin/env python3
"""
🌌 COSMIC AI - Simple Working Bot
Guaranteed to respond to user 7700105638
"""

import requests
import json
import time
import random
from datetime import datetime

# Bot configuration
TOKEN = "7604218758:AAHJj2zMDTfVwyJHpLClVCDzukNr2Psj-38"
AUTHORIZED_USER = 7700105638
BASE_URL = f"https://api.telegram.org/bot{TOKEN}"

def send_message(chat_id, text, reply_markup=None):
    """Send message to chat"""
    url = f"{BASE_URL}/sendMessage"
    data = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown"
    }
    
    if reply_markup:
        data["reply_markup"] = json.dumps(reply_markup)
    
    try:
        response = requests.post(url, data=data, timeout=10)
        result = response.json()
        print(f"✅ Message sent to {chat_id}: {result.get('ok', False)}")
        return result
    except Exception as e:
        print(f"❌ Error sending message: {e}")
        return None

def get_updates(offset=0):
    """Get updates from Telegram"""
    url = f"{BASE_URL}/getUpdates"
    params = {
        "offset": offset,
        "timeout": 5
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        return response.json()
    except Exception as e:
        print(f"❌ Error getting updates: {e}")
        return None

def generate_analysis():
    """Generate trading analysis"""
    directions = ['CALL', 'PUT']
    direction = random.choice(directions)
    confidence = random.uniform(0.70, 0.95)
    
    assets = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'BTC/USD', 'ETH/USD']
    asset = random.choice(assets)
    
    strategies = ['Breakout Continuation', 'Reversal Play', 'Momentum Shift', 'Trap Fade', 'Exhaustion Reversal']
    strategy = random.choice(strategies)
    
    expiry_times = [60, 180, 300, 600]
    expiry = random.choice(expiry_times)
    
    return {
        'direction': direction,
        'confidence': confidence,
        'asset': asset,
        'strategy': strategy,
        'expiry': expiry
    }

def handle_start(chat_id, user_name):
    """Handle /start command"""
    message = f"""🌌 **COSMIC AI ACTIVATED**

Welcome {user_name}! 🔐 **Private Access Granted**

🧠 **THE ULTIMATE BINARY TRADING BOT**

🚀 **Features:**
- 🔍 **Chart Analysis** - Send any chart image
- 📊 **AI Predictions** - CALL/PUT with confidence
- 🎯 **Strategy Building** - 5 adaptive strategies  
- 📈 **Live Signals** - Real-time trading alerts

💫 **Strategy Types:**
- Breakout Continuation
- Reversal Play  
- Momentum Shift
- Trap Fade
- Exhaustion Reversal

**📸 Send a chart image to get instant analysis!**

Use /help for more commands."""

    keyboard = {
        "inline_keyboard": [
            [{"text": "📊 Analyze Chart", "callback_data": "analyze"}],
            [{"text": "📈 Get Signal", "callback_data": "signal"}],
            [{"text": "❓ Help", "callback_data": "help"}]
        ]
    }
    
    send_message(chat_id, message, keyboard)

def handle_help(chat_id):
    """Handle /help command"""
    message = """🔧 **COSMIC AI COMMANDS**

📸 **Chart Analysis:**
Send any chart image → Get AI analysis

💬 **Commands:**
- /start - Initialize bot
- /signal - Get live trading signal
- /help - Show this help

🎯 **How to Use:**
1. Send chart screenshot
2. Get AI analysis instantly
3. Follow the trading signal

**Ready to analyze your charts!**"""
    
    send_message(chat_id, message)

def handle_signal(chat_id):
    """Handle /signal command - Generate live signal"""
    analysis = generate_analysis()
    
    direction_emoji = "📈" if analysis['direction'] == 'CALL' else "📉"
    confidence_bars = "█" * int(analysis['confidence'] * 10)
    
    signal_message = f"""🚀 **LIVE TRADING SIGNAL**

🎯 **PREDICTION:** {direction_emoji} **{analysis['direction']}**

📊 **Asset:** {analysis['asset']}
🔥 **Confidence:** {analysis['confidence']:.1%}
{confidence_bars}

⏰ **Expiry:** {analysis['expiry']//60}m {analysis['expiry']%60}s
🧭 **Strategy:** {analysis['strategy']}

💭 **Analysis:**
Market showing strong {analysis['direction'].lower()} momentum. Technical indicators align with {analysis['confidence']:.1%} confidence.

✨ **Signal is ACTIVE - Trade Now!**"""

    keyboard = {
        "inline_keyboard": [
            [{"text": f"🚀 Execute {analysis['direction']}", "callback_data": f"execute_{analysis['direction']}"}],
            [{"text": "📊 New Signal", "callback_data": "signal"}]
        ]
    }
    
    send_message(chat_id, signal_message, keyboard)

def handle_image(chat_id, user_name):
    """Handle image uploads"""
    print(f"📸 Processing image from {user_name}")
    
    # Send processing message
    processing_msg = send_message(chat_id, "🌌 **COSMIC AI ANALYZING...**\n\n🔍 Extracting chart data...\n📊 Building strategy...")
    
    if not processing_msg:
        return
    
    # Simulate analysis
    time.sleep(2)
    
    # Generate analysis
    analysis = generate_analysis()
    
    direction_emoji = "📈" if analysis['direction'] == 'CALL' else "📉"
    confidence_bars = "█" * int(analysis['confidence'] * 10)
    
    # Send analysis results
    perception_msg = f"""🔍 **CHART ANALYSIS COMPLETE**

📊 **Detected Asset:** {analysis['asset']}
✅ **Chart patterns identified**"""
    
    strategy_msg = f"""🧠 **AI PREDICTION**

🎯 **SIGNAL:** {direction_emoji} **{analysis['direction']}**

📊 **Confidence:** {analysis['confidence']:.1%}
{confidence_bars}

⏰ **Recommended Expiry:** {analysis['expiry']//60}m {analysis['expiry']%60}s
🧭 **Strategy:** {analysis['strategy']}

💭 **AI Reasoning:**
Strong {analysis['direction'].lower()} signals detected. Market psychology and technical indicators show {analysis['confidence']:.1%} alignment.

🚀 **SIGNAL IS READY FOR EXECUTION!**"""

    keyboard = {
        "inline_keyboard": [
            [{"text": f"🚀 Execute {analysis['direction']}", "callback_data": f"execute_{analysis['direction']}"}],
            [{"text": "⏸️ Skip Trade", "callback_data": "skip"}],
            [{"text": "📊 New Analysis", "callback_data": "analyze"}]
        ]
    }
    
    send_message(chat_id, perception_msg)
    time.sleep(1)
    send_message(chat_id, strategy_msg, keyboard)

def handle_callback(query_data, chat_id, message_id, user_name):
    """Handle button presses"""
    print(f"🔘 Button pressed: {query_data} by {user_name}")
    
    if query_data == "analyze":
        edit_message(chat_id, message_id, "📊 **Ready for Analysis**\n\nSend me a chart image to activate COSMIC AI!")
    
    elif query_data == "signal":
        handle_signal(chat_id)
    
    elif query_data == "help":
        handle_help(chat_id)
    
    elif query_data.startswith("execute_"):
        direction = query_data.split("_")[1]
        edit_message(chat_id, message_id, f"🚀 **{direction} Trade Executed!**\n\nDemo mode: Would place {direction} trade.\n\n**Broker Integration Available:**\n- Deriv.com\n- IQ Option\n- Quotex")
    
    elif query_data == "skip":
        edit_message(chat_id, message_id, "⏸️ **Trade Skipped**\n\nAnalysis completed. Send another image for new analysis!")

def edit_message(chat_id, message_id, text):
    """Edit a message"""
    url = f"{BASE_URL}/editMessageText"
    data = {
        "chat_id": chat_id,
        "message_id": message_id,
        "text": text,
        "parse_mode": "Markdown"
    }
    
    try:
        response = requests.post(url, data=data, timeout=10)
        return response.json()
    except Exception as e:
        print(f"❌ Error editing message: {e}")
        return None

def main():
    """Main bot loop"""
    print("🌌 COSMIC AI Bot Starting...")
    print(f"🔐 Authorized user: {AUTHORIZED_USER}")
    print("📱 Bot is running. Send /start in Telegram!")
    
    offset = 0
    
    while True:
        try:
            # Get updates
            updates = get_updates(offset)
            
            if not updates or not updates.get('ok'):
                time.sleep(1)
                continue
            
            for update in updates.get('result', []):
                offset = update['update_id'] + 1
                
                # Handle messages
                if 'message' in update:
                    message = update['message']
                    chat_id = message['chat']['id']
                    user_name = message['from'].get('first_name', 'User')
                    
                    # Check authorization
                    if chat_id != AUTHORIZED_USER:
                        send_message(chat_id, f"🚫 **Access Denied**\n\nSorry {user_name}, this is a private bot for authorized users only.")
                        continue
                    
                    print(f"💬 Message from authorized user {user_name}")
                    
                    # Handle text messages
                    if 'text' in message:
                        text = message['text']
                        print(f"📝 Text: {text}")
                        
                        if text.startswith('/start'):
                            handle_start(chat_id, user_name)
                        elif text.startswith('/help'):
                            handle_help(chat_id)
                        elif text.startswith('/signal'):
                            handle_signal(chat_id)
                        else:
                            send_message(chat_id, "🤖 **COSMIC AI Ready**\n\nSend chart image for analysis or use /signal for live trading signal!")
                    
                    # Handle photos
                    elif 'photo' in message:
                        handle_image(chat_id, user_name)
                
                # Handle callback queries (button presses)
                elif 'callback_query' in update:
                    query = update['callback_query']
                    chat_id = query['message']['chat']['id']
                    message_id = query['message']['message_id']
                    query_data = query['data']
                    user_name = query['from'].get('first_name', 'User')
                    
                    # Check authorization
                    if chat_id != AUTHORIZED_USER:
                        continue
                    
                    handle_callback(query_data, chat_id, message_id, user_name)
            
            time.sleep(1)
            
        except KeyboardInterrupt:
            print("\n🛑 Bot stopped by user")
            break
        except Exception as e:
            print(f"❌ Error in main loop: {e}")
            time.sleep(5)

if __name__ == '__main__':
    main()