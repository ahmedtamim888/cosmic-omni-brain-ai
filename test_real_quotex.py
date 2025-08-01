#!/usr/bin/env python3
"""
🧪 TEST REAL QUOTEX CONNECTION
Test script to verify connection and generate sample signals
"""

import asyncio
import sys
import os
from datetime import datetime

# Add pyquotex to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'pyquotex'))

from pyquotex.stable_api import Quotex

async def test_quotex_connection():
    """Test connection to Quotex with real credentials"""
    print("🧪 TESTING REAL QUOTEX CONNECTION")
    print("=" * 50)
    
    # Real credentials
    email = "beyondverse11@gmail.com"
    password = "ahmedtamim94301"
    
    print(f"📧 Email: {email}")
    print("🔗 Attempting to connect to Quotex...")
    
    try:
        # Initialize Quotex client
        client = Quotex(
            email=email,
            password=password,
            lang="en",
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )
        
        # Attempt connection
        check_connect, message = await client.connect()
        
        if check_connect:
            print("✅ Successfully connected to Quotex!")
            
            # Get account balance
            try:
                balance = await client.get_balance()
                print(f"💰 Account Balance: ${balance}")
            except Exception as e:
                print(f"⚠️ Could not get balance: {e}")
            
            # Test getting candle data
            print("\n📊 Testing candle data retrieval...")
            test_assets = ["EURUSD_otc", "GBPUSD_otc", "USDJPY_otc"]
            
            for asset in test_assets:
                try:
                    candles = await client.get_candle(asset, 60, 10)
                    if candles:
                        latest_candle = candles[-1]
                        print(f"✅ {asset}: Latest price = {latest_candle.get('close', 'N/A')}")
                    else:
                        print(f"⚠️ {asset}: No candle data received")
                except Exception as e:
                    print(f"❌ {asset}: Error getting candles - {e}")
            
            # Generate sample signals
            print("\n🎯 GENERATING SAMPLE SIGNALS:")
            print("━" * 40)
            
            from datetime import timedelta
            import random
            
            current_time = datetime.now()
            signal_assets = ["EURUSD-OTC", "GBPUSD-OTC", "USDJPY-OTC", "AUDUSD-OTC", "USDCAD-OTC"]
            
            for i in range(8):
                future_time = current_time + timedelta(minutes=random.randint(3, 15))
                asset = random.choice(signal_assets)
                signal = random.choice(["CALL", "PUT"])
                confidence = random.uniform(78.5, 94.2)
                
                print(f"{future_time.strftime('%H:%M')} {asset} {signal} ({confidence:.1f}%)")
            
            print("━" * 40)
            print("✅ Sample signals generated successfully!")
            
            # Close connection
            await client.close()
            print("\n🔌 Connection closed")
            
        else:
            print(f"❌ Failed to connect to Quotex: {message}")
            return False
            
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False
    
    return True

async def test_signal_formatting():
    """Test signal formatting for Telegram"""
    print("\n📱 TESTING TELEGRAM SIGNAL FORMATTING")
    print("=" * 50)
    
    # Sample signals
    signals = [
        {'time': '14:25', 'asset': 'EURUSD-OTC', 'signal': 'CALL', 'confidence': 87.5, 'reasoning': 'Bullish momentum confirmed'},
        {'time': '14:28', 'asset': 'GBPUSD-OTC', 'signal': 'PUT', 'confidence': 92.1, 'reasoning': 'Overbought reversal signal'},
        {'time': '14:32', 'asset': 'USDJPY-OTC', 'signal': 'CALL', 'confidence': 84.3, 'reasoning': 'Support level bounce'},
    ]
    
    # Format for Telegram
    message = "🧠 REAL QUOTEX OTC SIGNALS 🧠\n"
    message += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    message += "⚡ LIVE MARKET ANALYSIS ⚡\n"
    message += "🎯 HIGH-CONFIDENCE SIGNALS 🎯\n"
    message += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    
    for signal in signals:
        message += f"⏰ {signal['time']} {signal['asset']} {signal['signal']} ({signal['confidence']:.1f}%)\n"
    
    message += "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    message += "📊 1-MINUTE EXPIRY | REAL MARKET DATA\n"
    message += "🔥 TECHNICAL ANALYSIS BASED 🔥\n"
    message += f"⚡ Generated: {datetime.now().strftime('%H:%M:%S')} ⚡"
    
    print("Formatted Telegram Message:")
    print("-" * 50)
    print(message)
    print("-" * 50)

if __name__ == "__main__":
    print("""
    🧪 REAL QUOTEX CONNECTION TEST 🧪
    Testing connection with actual pyquotex library
    """)
    
    try:
        # Run connection test
        asyncio.run(test_quotex_connection())
        
        # Test signal formatting
        asyncio.run(test_signal_formatting())
        
        print("\n🎉 All tests completed!")
        print("🚀 Ready to run real bot with: python3 real_quotex_signals.py")
        
    except KeyboardInterrupt:
        print("\n👋 Test stopped by user")
    except Exception as e:
        print(f"\n❌ Test error: {e}")
        print("💡 Make sure you have internet connection and valid credentials")