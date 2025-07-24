#!/usr/bin/env python3
"""
Test script to verify chart validation rejects random/fake images
"""

import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

def create_test_images():
    """Create various test images to verify validation"""
    
    # Create test directory
    test_dir = "test_images"
    os.makedirs(test_dir, exist_ok=True)
    
    # 1. Create a VALID trading chart (fake but chart-like)
    print("🎯 Creating valid chart image...")
    chart_img = Image.new('RGB', (800, 400), color='black')
    draw = ImageDraw.Draw(chart_img)
    
    # Draw grid lines (typical of charts)
    for i in range(0, 800, 50):
        draw.line([(i, 0), (i, 400)], fill='gray', width=1)
    for i in range(0, 400, 25):
        draw.line([(0, i), (800, i)], fill='gray', width=1)
    
    # Draw some candlestick-like rectangles
    for x in range(50, 750, 40):
        # Random candle heights
        high = np.random.randint(100, 200)
        low = np.random.randint(250, 350)
        open_price = np.random.randint(high + 10, low - 10)
        close = np.random.randint(high + 10, low - 10)
        
        # Draw wick
        draw.line([(x + 15, high), (x + 15, low)], fill='white', width=2)
        
        # Draw body
        color = 'green' if close > open_price else 'red'
        draw.rectangle([x + 10, min(open_price, close), x + 20, max(open_price, close)], 
                      fill=color, outline='white')
    
    # Add price labels on right
    for i, price in enumerate(range(100, 401, 50)):
        draw.text((770, price - 10), f"{1.2345 + i*0.01:.4f}", fill='white')
    
    chart_img.save(f"{test_dir}/valid_chart.png")
    
    # 2. Create INVALID images
    
    # Random photo-like image
    print("📸 Creating random photo...")
    random_img = Image.new('RGB', (640, 480), color='blue')
    draw = ImageDraw.Draw(random_img)
    # Add random shapes (like a photo)
    for _ in range(50):
        x1, y1 = np.random.randint(0, 600), np.random.randint(0, 440)
        x2, y2 = x1 + np.random.randint(10, 40), y1 + np.random.randint(10, 40)
        color = tuple(np.random.randint(0, 255, 3))
        draw.ellipse([x1, y1, x2, y2], fill=color)
    random_img.save(f"{test_dir}/random_photo.png")
    
    # Text screenshot
    print("📝 Creating text screenshot...")
    text_img = Image.new('RGB', (600, 300), color='white')
    draw = ImageDraw.Draw(text_img)
    draw.text((50, 50), "This is just a text screenshot", fill='black')
    draw.text((50, 100), "Not a trading chart", fill='black')
    draw.text((50, 150), "Should be rejected", fill='black')
    text_img.save(f"{test_dir}/text_screenshot.png")
    
    # Solid color image
    print("🎨 Creating solid color image...")
    solid_img = Image.new('RGB', (500, 300), color='red')
    solid_img.save(f"{test_dir}/solid_color.png")
    
    # Very small image
    print("🔍 Creating tiny image...")
    tiny_img = Image.new('RGB', (100, 50), color='green')
    tiny_img.save(f"{test_dir}/tiny_image.png")
    
    print(f"✅ Test images created in {test_dir}/")
    return test_dir

def test_validation():
    """Test the validation on various images"""
    
    # Create test images
    test_dir = create_test_images()
    
    # Import the validator
    try:
        from logic.chart_validator import ChartValidator
        validator = ChartValidator()
        print("✅ Chart validator imported successfully")
    except ImportError as e:
        print(f"❌ Cannot import validator: {e}")
        return
    
    # Test each image
    test_files = [
        ("valid_chart.png", True, "Should be accepted"),
        ("random_photo.png", False, "Should reject random photo"),
        ("text_screenshot.png", False, "Should reject text"),
        ("solid_color.png", False, "Should reject solid color"),
        ("tiny_image.png", False, "Should reject tiny image")
    ]
    
    print("\n" + "="*60)
    print("🔍 TESTING CHART VALIDATION")
    print("="*60)
    
    for filename, expected_valid, description in test_files:
        file_path = os.path.join(test_dir, filename)
        
        print(f"\n📋 Testing: {filename}")
        print(f"💭 Expected: {description}")
        
        result = validator.validate_chart_image(file_path)
        
        is_valid = result['is_valid']
        reason = result['reason']
        confidence = result['confidence']
        
        if is_valid == expected_valid:
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
        
        print(f"🎯 Result: {status}")
        print(f"📊 Valid: {is_valid}")
        print(f"🔒 Confidence: {confidence:.2f}")
        print(f"💡 Reason: {reason}")
    
    print("\n" + "="*60)

def test_ai_engine():
    """Test the AI engine with validation"""
    print("\n🧠 TESTING AI ENGINE WITH VALIDATION")
    print("="*60)
    
    try:
        from logic.ai_engine import CosmicAIEngine
        ai_engine = CosmicAIEngine()
        print("✅ AI Engine loaded with validation")
        
        test_dir = "test_images"
        
        for filename in ["valid_chart.png", "random_photo.png", "text_screenshot.png"]:
            file_path = os.path.join(test_dir, filename)
            if os.path.exists(file_path):
                print(f"\n📊 Testing {filename} with AI Engine...")
                
                signal = ai_engine.analyze_chart(file_path)
                
                print(f"🎯 Signal: {signal.signal}")
                print(f"🔒 Confidence: {signal.confidence}")
                print(f"💡 Reasoning: {signal.reasoning}")
                print(f"🎯 Strategy: {signal.strategy}")
                
                if filename != "valid_chart.png" and signal.strategy == "VALIDATION_FAILED":
                    print("✅ CORRECTLY REJECTED non-chart image!")
                elif filename == "valid_chart.png" and signal.strategy != "VALIDATION_FAILED":
                    print("✅ CORRECTLY ACCEPTED chart image!")
                else:
                    print("⚠️  Unexpected result")
        
    except ImportError as e:
        print(f"❌ Cannot import AI engine: {e}")

def main():
    """Main test function"""
    print("🔍 CHART VALIDATION TEST SUITE")
    print("Testing if bot properly rejects random images")
    print("="*60)
    
    # Test validation
    test_validation()
    
    # Test AI engine integration
    test_ai_engine()
    
    print("\n🎉 VALIDATION TESTS COMPLETE!")
    print("Your bot should now reject random images and only analyze real charts.")

if __name__ == "__main__":
    main()