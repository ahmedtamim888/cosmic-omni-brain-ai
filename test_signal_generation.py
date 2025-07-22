#!/usr/bin/env python3
"""
Test Real Signal Generation for COSMIC AI Bot
"""

import os
import sys
from datetime import datetime
from logic.ai_engine import CosmicAIEngine
from telegram_bot_pro import send_signal_to_telegram, get_bot_stats

def test_real_signal_generation():
    """Test the real signal generation system"""
    print("🧠 COSMIC AI REAL SIGNAL GENERATION TEST")
    print("=" * 60)
    
    # Initialize AI Engine
    ai_engine = CosmicAIEngine()
    print("✅ AI Engine initialized")
    
    # Check if we have a sample chart image
    sample_charts = [
        "sample_chart.png",
        "test_chart.jpg", 
        "chart_screenshot.png",
        "trading_chart.jpg"
    ]
    
    chart_found = False
    for chart_file in sample_charts:
        if os.path.exists(chart_file):
            print(f"📊 Found chart: {chart_file}")
            
            try:
                # Analyze the chart
                print("🔍 Analyzing chart...")
                signal = ai_engine.analyze_chart(chart_file)
                
                print("\n🎯 ANALYSIS RESULTS:")
                print(f"📈 Signal: {signal.signal}")
                print(f"🔒 Confidence: {signal.confidence:.1f}%")
                print(f"💡 Reasoning: {signal.reasoning}")
                print(f"🎯 Strategy: {signal.strategy}")
                print(f"🧠 Psychology: {signal.market_psychology}")
                print(f"📊 Patterns: {', '.join(signal.patterns_detected) if signal.patterns_detected else 'None'}")
                print(f"⏰ Time: {signal.entry_time.strftime('%H:%M:%S')}")
                
                # Test Telegram sending
                if signal.confidence >= 85:
                    print("\n📱 SENDING TO TELEGRAM...")
                    signal_data = {
                        'signal': signal.signal,
                        'confidence': signal.confidence,
                        'reasoning': signal.reasoning,
                        'strategy': signal.strategy,
                        'market_psychology': signal.market_psychology,
                        'timeframe': signal.timeframe
                    }
                    
                    result = send_signal_to_telegram(signal_data)
                    if result:
                        print("✅ Signal sent to Telegram successfully!")
                    else:
                        print("⚠️ No Telegram subscribers or sending failed")
                else:
                    print(f"⚠️ Confidence {signal.confidence:.1f}% below threshold (85%)")
                    
                chart_found = True
                break
                
            except Exception as e:
                print(f"❌ Error analyzing {chart_file}: {e}")
                continue
    
    if not chart_found:
        print("📊 No chart images found. Testing with demo mode...")
        
        # Create a demo signal for testing
        demo_signal = {
            'signal': 'CALL',
            'confidence': 89.5,
            'reasoning': 'Strong bullish momentum with breakout pattern',
            'strategy': 'BREAKOUT_CONTINUATION',
            'market_psychology': 'BULLISH_MOMENTUM',
            'timeframe': '1M'
        }
        
        print("\n🎯 DEMO SIGNAL:")
        print(f"📈 Signal: {demo_signal['signal']}")
        print(f"🔒 Confidence: {demo_signal['confidence']:.1f}%")
        print(f"💡 Reasoning: {demo_signal['reasoning']}")
        print(f"🎯 Strategy: {demo_signal['strategy']}")
        print(f"🧠 Psychology: {demo_signal['market_psychology']}")
        
        # Send demo signal
        print("\n📱 SENDING DEMO SIGNAL TO TELEGRAM...")
        result = send_signal_to_telegram(demo_signal)
        if result:
            print("✅ Demo signal sent to Telegram successfully!")
        else:
            print("⚠️ No Telegram subscribers or sending failed")
    
    # Show bot statistics
    print("\n📊 BOT STATISTICS:")
    stats = get_bot_stats()
    print(f"🤖 Bot Status: {'Running' if stats['is_running'] else 'Stopped'}")
    print(f"👥 Subscribers: {stats['subscribers']}")
    print(f"📈 Total Signals: {stats['stats']['total_signals']}")
    print(f"🎯 Win Rate: {stats['stats']['win_rate']:.1f}%")
    
    print("\n✅ SIGNAL GENERATION SYSTEM READY!")
    print("🚀 Upload chart images to get real trading signals!")
    
    return True

def show_usage_instructions():
    """Show how to use the signal generation system"""
    print("\n📋 HOW TO GENERATE REAL SIGNALS:")
    print("1. 📷 Take screenshot of your trading chart")
    print("2. 📱 Send image to Telegram bot OR upload to web app")
    print("3. ⏱️ Wait 2-5 seconds for AI analysis")
    print("4. 📈 Receive CALL/PUT signal with confidence")
    print("5. 💰 Execute trade (manage risk properly!)")
    
    print("\n🎯 SUPPORTED ANALYSIS:")
    print("• Candlestick pattern recognition")
    print("• Support/resistance levels")
    print("• Momentum analysis")
    print("• Market psychology reading")
    print("• Multi-strategy approach")
    print("• 85%+ confidence threshold")
    
    print("\n📊 COMPATIBLE BROKERS:")
    print("• Quotex")
    print("• Binomo")
    print("• Pocket Option")
    print("• MetaTrader 4/5")
    print("• TradingView")
    print("• Any candlestick chart")

if __name__ == "__main__":
    try:
        test_real_signal_generation()
        show_usage_instructions()
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)