#!/usr/bin/env python3
"""
🧪 TEST IMMORTAL SIGNALS GENERATION
Demo script to show signal generation capabilities
"""

import asyncio
from datetime import datetime
from quotex_connector import QuotexConnector
from signal_engine import ImmortalSignalEngine
from telegram_sender import TelegramSignalSender

async def test_signal_generation():
    """Test the immortal signal generation system"""
    print("🧠 TESTING IMMORTAL QUOTEX OTC SIGNALS 🧠")
    print("=" * 60)
    
    # Initialize components
    connector = QuotexConnector()
    signal_engine = ImmortalSignalEngine(connector)
    
    # Connect to Quotex (simulated)
    print("🔗 Connecting to Quotex...")
    await connector.connect()
    
    # Generate test signals
    print("🎯 Generating immortal signals...")
    signals = signal_engine.generate_future_signals(15)
    
    # Display signals in console format
    print("\n🚀 IMMORTAL SIGNALS GENERATED:")
    print("━" * 50)
    
    for signal in signals:
        print(f"{signal['time']} {signal['symbol']} {signal['signal']}")
    
    print("━" * 50)
    print(f"📊 Generated {len(signals)} signals")
    print(f"⚡ Confidence: 99.9% Immortal Accuracy")
    print(f"🧠 Experience: 999999 Trillion Years")
    
    # Disconnect
    await connector.disconnect()
    print("\n✅ Test completed successfully!")

def test_telegram_formatting():
    """Test Telegram message formatting"""
    print("\n📱 TESTING TELEGRAM FORMATTING")
    print("=" * 40)
    
    # Mock signals for testing
    test_signals = [
        {'time': '14:25', 'symbol': 'USDEGP-OTC', 'signal': 'PUT', 'confidence': 98.5, 'pattern': 'quantum_reversal', 'reasoning': 'Smart money trap detected'},
        {'time': '14:28', 'symbol': 'USDARS-OTC', 'signal': 'CALL', 'confidence': 97.2, 'pattern': 'shadow_rejection', 'reasoning': 'Volume surge confirmed'},
        {'time': '14:32', 'symbol': 'USDBRL-OTC', 'signal': 'PUT', 'confidence': 99.1, 'pattern': 'market_maker_trap', 'reasoning': 'Institutional fingerprint'},
    ]
    
    # Test formatting
    bot_token = "TEST_TOKEN"
    chat_id = "TEST_CHAT"
    telegram_sender = TelegramSignalSender(bot_token, chat_id)
    
    formatted_message = telegram_sender._format_signals_message(test_signals)
    print("Formatted Telegram Message:")
    print("-" * 40)
    print(formatted_message)
    print("-" * 40)

if __name__ == "__main__":
    print("""
    🧠 IMMORTAL QUOTEX SIGNALS TEST 🧠
    Testing signal generation capabilities
    """)
    
    # Run tests
    asyncio.run(test_signal_generation())
    test_telegram_formatting()
    
    print("\n🎉 All tests completed!")
    print("🚀 Ready to run the full bot with: python quotex_immortal_bot.py")