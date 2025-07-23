#!/usr/bin/env python3
"""
Send a test trading signal to Telegram
"""

import asyncio
import os
from datetime import datetime

# Set environment variables
os.environ['TELEGRAM_BOT_TOKEN'] = '8155744845:AAGQ6s8RMSpCAjO16Mzb1xl8_otLnXh3OKQ'
os.environ['TELEGRAM_CHAT_IDS'] = '7700105638'

async def send_test_signal():
    print("🚀 Sending Test Trading Signal...")
    print("=" * 50)
    
    try:
        from communication.telegram_bot import TelegramBot
        
        # Create bot instance
        bot = TelegramBot('8155744845:AAGQ6s8RMSpCAjO16Mzb1xl8_otLnXh3OKQ')
        bot.chat_ids = [7700105638]
        
        # Test connection first
        print("🔄 Testing connection...")
        connection_ok = await bot.test_connection()
        
        if not connection_ok:
            print("❌ Connection test failed!")
            return False
        
        print("✅ Connection successful!")
        
        # Create a realistic test trading signal
        test_signal = {
            'action': 'CALL',
            'confidence': 0.96,
            'strategy': 'GOD_MODE_AI',
            'reason': '''🔥 GOD MODE ACTIVATED! 🔥

📊 CONFLUENCE ANALYSIS:
• Bullish engulfing pattern confirmed
• Strong support level bounce at 1.0850
• RSI oversold reversal (32 → 58)
• Volume spike +150% above average
• Smart money accumulation detected

🧠 MARKET PSYCHOLOGY:
• Bears exhausted after 3-candle decline
• Institutional buying pressure visible
• Retail panic selling absorbed
• Momentum shifting bullish

⚡ QUANTUM COHERENCE: 0.987
🌌 CONSCIOUSNESS SCORE: 0.943
🔗 CONFLUENCES: 5/5 ALIGNED

🎯 ENTRY: Next candle open
⏰ EXPIRY: 1 minute
💎 CONFIDENCE: 96%''',
            'timestamp': datetime.now(),
            'next_candle_prediction': {
                'direction': 'UP',
                'strength': 'STRONG'
            },
            'volume_condition': 'rising',
            'trend_alignment': 'bullish_trend',
            'god_mode_metrics': {
                'confluences_count': 5,
                'quantum_coherence': 0.987,
                'consciousness_score': 0.943
            }
        }
        
        print("📤 Sending God Mode trading signal...")
        await bot.send_signal(test_signal)
        
        print("✅ Test signal sent successfully!")
        print("📱 Check your Telegram chat for the signal!")
        
        # Send a welcome message
        welcome_signal = {
            'action': 'NO_TRADE',
            'confidence': 1.0,
            'strategy': 'WELCOME_MESSAGE',
            'reason': '''🎉 WELCOME TO ULTRA-ACCURATE TRADING BOT! 🎉

🚀 Your Telegram bot is now FULLY OPERATIONAL!

⚡ FEATURES ACTIVATED:
• God Mode AI Engine ✅
• Market Psychology Analysis ✅  
• Pattern Recognition Engine ✅
• Support/Resistance Detection ✅
• Memory Engine (Anti-Trap) ✅
• Real-time Signal Delivery ✅

🎯 SIGNAL TYPES:
📈 CALL - Buy signal (95%+ confidence)
📉 PUT - Sell signal (95%+ confidence)  
⏸️ NO_TRADE - Wait for better setup
⚡ GOD_MODE - Ultimate 97%+ signals

🔥 Ready for God-Tier Trading!
No lag, no doubt, no mercy! 💎''',
            'timestamp': datetime.now()
        }
        
        print("📤 Sending welcome message...")
        await bot.send_signal(welcome_signal)
        
        return True
        
    except Exception as e:
        print(f"❌ Error sending test signal: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(send_test_signal())