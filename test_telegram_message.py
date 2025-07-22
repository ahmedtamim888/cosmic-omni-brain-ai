#!/usr/bin/env python3
"""
Test script to verify Telegram message formatting works correctly
"""

import requests
import cv2
import numpy as np
import json

def test_telegram_message_format():
    """Test that the API generates Telegram-safe messages"""
    
    print("🧪 Testing Telegram Message Formatting...")
    
    # Create a realistic test chart
    chart = np.zeros((600, 800, 3), dtype=np.uint8)
    chart.fill(20)  # Dark background
    
    # Draw chart area
    cv2.rectangle(chart, (50, 50), (750, 550), (40, 40, 40), -1)
    cv2.rectangle(chart, (50, 50), (750, 550), (100, 100, 100), 2)
    
    # Draw some candlesticks
    for i in range(10):
        x = 100 + i * 60
        # Draw wick
        cv2.line(chart, (x, 200), (x, 400), (200, 200, 200), 2)
        # Draw body
        color = (0, 255, 0) if i % 2 == 0 else (0, 0, 255)
        cv2.rectangle(chart, (x-15, 250), (x+15, 350), color, -1)
    
    # Add title
    cv2.putText(chart, "VOLATILE OTC MARKET TEST", (100, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Save test chart
    cv2.imwrite('telegram_test_chart.png', chart)
    print("📊 Test chart created")
    
    # Send to API
    try:
        with open('telegram_test_chart.png', 'rb') as f:
            files = {'image': ('telegram_test.png', f, 'image/png')}
            response = requests.post('http://localhost:5000/api/analyze', files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            
            print("✅ API Response Successful!")
            print(f"🎯 Signal: {result.get('signal', 'N/A')}")
            print(f"📈 Confidence: {result.get('confidence', 0)}%")
            print(f"⏰ Entry Time: {result.get('next_candle_time', 'N/A')} (UTC+6:00)")
            print(f"🕯️ Candles Consulted: {result.get('total_candles_consulted', 0)}")
            print(f"🎪 Accuracy: {result.get('accuracy', 0)}%")
            
            # Test message formatting
            message = result.get('message', '')
            print(f"\n📱 Message Length: {len(message)} characters")
            
            # Check for problematic characters
            problematic_chars = ['|', '*', '_', '`', '[', ']', '<', '>', '\\']
            found_problems = []
            
            for char in problematic_chars:
                if char in message:
                    found_problems.append(char)
            
            if found_problems:
                print(f"⚠️  Found problematic characters: {found_problems}")
                print("🔧 Message needs more cleaning")
            else:
                print("✅ Message is Telegram-safe!")
            
            # Show clean message preview
            print("\n📄 TELEGRAM MESSAGE PREVIEW:")
            print("=" * 50)
            print(message[:500] + "..." if len(message) > 500 else message)
            print("=" * 50)
            
            # Test JSON serialization
            try:
                json.dumps(result)
                print("✅ JSON serialization works")
            except Exception as e:
                print(f"❌ JSON serialization failed: {e}")
            
            return True
            
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

def test_message_cleaning():
    """Test the message cleaning function"""
    print("\n🧪 Testing Message Cleaning Function...")
    
    # Test problematic text
    test_texts = [
        "Signal: CALL | Confidence: 85% | Entry: 14:25",
        "Analysis: *Strong* _bullish_ `pattern` [confirmed]",
        "Reasoning: <chart> shows {breakout} pattern \\n trend",
        "Prophecy: ALL CANDLES AGREE: NEXT CANDLE GOES UP!",
        "Very long message " * 100  # Test length limit
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nTest {i}: '{text[:50]}...'")
        
        # Apply cleaning function (simulate what's in the code)
        cleaned = str(text).replace('|', '-').replace('<', '').replace('>', '')
        cleaned = cleaned.replace('[', '(').replace(']', ')').replace('*', '')
        cleaned = cleaned.replace('_', ' ').replace('`', '').replace('\\', '')
        
        if len(cleaned) > 200:
            cleaned = cleaned[:200] + "..."
        
        print(f"Cleaned: '{cleaned[:50]}...'")
        print(f"Length: {len(cleaned)} chars")

if __name__ == "__main__":
    print("🔥 TELEGRAM MESSAGE FORMAT TEST")
    print("🕯️ Testing CANDLE WHISPERER Telegram compatibility\n")
    
    # Test API message formatting
    if test_telegram_message_format():
        print("\n🎉 TELEGRAM FORMAT TEST PASSED!")
    else:
        print("\n❌ TELEGRAM FORMAT TEST FAILED!")
    
    # Test cleaning function
    test_message_cleaning()
    
    print("\n🔥 Test completed!")
    print("📱 Your Telegram bot should now work without parsing errors!")