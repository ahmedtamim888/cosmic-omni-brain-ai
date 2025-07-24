#!/usr/bin/env python3
"""
📊 CHART CHECKER MODULE
Professional trading chart validation using OCR and image analysis
"""

import logging
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
import re
from typing import List, Tuple, Dict
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChartChecker:
    """
    🔍 Professional Trading Chart Validator
    
    Uses OCR and image analysis to determine if an image contains
    a valid trading chart from platforms like Quotex, TradingView, MT4, etc.
    """
    
    def __init__(self):
        """Initialize the chart checker with trading platform keywords"""
        
        # 📋 Trading Platform Keywords
        self.trading_keywords = [
            # Platform Names
            "quotex", "tradingview", "metatrader", "mt4", "mt5", "binomo", 
            "iqoption", "olymptrade", "pocket option", "deriv", "binary.com",
            "etoro", "plus500", "xm", "fxpro", "pepperstone",
            
            # Currency Pairs
            "eur/usd", "gbp/usd", "usd/jpy", "aud/usd", "usd/cad", "nzd/usd",
            "eur/gbp", "eur/jpy", "gbp/jpy", "aud/jpy", "chf/jpy",
            "eurusd", "gbpusd", "usdjpy", "audusd", "usdcad", "nzdusd",
            
            # Trading Terms
            "payout", "investment", "price", "time", "expiry", "expiration",
            "call", "put", "higher", "lower", "up", "down", "buy", "sell",
            "trade", "trading", "chart", "candle", "candlestick",
            
            # Mobile-specific terms (common on Android apps)
            "balance", "profit", "loss", "amount", "win", "rate", "signal",
            
            # Currency Indicators
            "usd", "eur", "gbp", "jpy", "aud", "cad", "chf", "nzd",
            "btc", "eth", "ltc", "bitcoin", "ethereum", "crypto",
            
            # Market Terms
            "otc", "forex", "binary", "options", "market", "volume",
            "high", "low", "open", "close", "bid", "ask", "spread",
            
            # Time Indicators
            "1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w",
            "minute", "minutes", "hour", "hours", "second", "seconds",
            
            # Technical Indicators
            "rsi", "macd", "ema", "sma", "bollinger", "stochastic",
            "support", "resistance", "trend", "pattern"
        ]
        
        # 🎯 Minimum confidence thresholds (RELAXED FOR MOBILE)
        self.min_keywords_found = 1  # Minimum keywords to consider valid (reduced for mobile)
        self.min_confidence_score = 0.15  # Minimum confidence score (reduced for mobile)
        
        logger.info("📊 Chart Checker initialized with %d keywords", len(self.trading_keywords))
    
    def is_valid_chart(self, image_path: str) -> bool:
        """
        🔍 Main function to check if image is a valid trading chart
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            bool: True if valid trading chart, False otherwise
        """
        try:
            logger.info("🔍 Analyzing image: %s", image_path)
            
            # Check if file exists
            if not os.path.exists(image_path):
                logger.error("❌ Image file not found: %s", image_path)
                return False
            
            # Extract text from image
            extracted_text = self._extract_text_from_image(image_path)
            
            if not extracted_text:
                logger.warning("⚠️ No text extracted from image")
                return False
            
            # Analyze extracted text
            analysis_result = self._analyze_extracted_text(extracted_text)
            
            # Log results
            logger.info("📊 Analysis Result:")
            logger.info("   Keywords found: %d", analysis_result['keywords_found'])
            logger.info("   Confidence score: %.2f", analysis_result['confidence_score'])
            logger.info("   Detected keywords: %s", analysis_result['detected_keywords'])
            
            # Determine if chart is valid
            is_valid = (
                analysis_result['keywords_found'] >= self.min_keywords_found and
                analysis_result['confidence_score'] >= self.min_confidence_score
            )
            
            result_emoji = "✅" if is_valid else "❌"
            logger.info("%s Chart validation result: %s", result_emoji, is_valid)
            
            return is_valid
            
        except Exception as e:
            logger.error("❌ Error during chart validation: %s", str(e))
            return False
    
    def _extract_text_from_image(self, image_path: str) -> str:
        """
        🔤 Extract text from image using OCR with preprocessing
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            str: Extracted text
        """
        try:
            # Load and preprocess image
            processed_image = self._preprocess_image(image_path)
            
            # Configure OCR settings for better accuracy
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789./:()[]{}+-=$%'
            
            # Extract text using pytesseract
            extracted_text = pytesseract.image_to_string(processed_image, config=custom_config)
            
            # Clean and normalize text
            cleaned_text = self._clean_extracted_text(extracted_text)
            
            logger.info("🔤 Extracted text length: %d characters", len(cleaned_text))
            logger.debug("📝 Raw extracted text: %s", cleaned_text[:200] + "..." if len(cleaned_text) > 200 else cleaned_text)
            
            return cleaned_text
            
        except Exception as e:
            logger.error("❌ Error extracting text from image: %s", str(e))
            return ""
    
    def _preprocess_image(self, image_path: str) -> Image.Image:
        """
        🖼️ Preprocess image for better OCR accuracy
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            PIL.Image: Preprocessed image
        """
        try:
            # Load image
            image = Image.open(image_path)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.5)
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(2.0)
            
            # Convert to grayscale for better OCR
            image = image.convert('L')
            
            # Apply slight blur to reduce noise
            image = image.filter(ImageFilter.MedianFilter(size=3))
            
            # Convert to OpenCV format for advanced processing
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_GRAY2BGR)
            
            # Apply adaptive thresholding
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            # Convert back to PIL Image
            processed_image = Image.fromarray(thresh)
            
            logger.debug("🖼️ Image preprocessed successfully")
            return processed_image
            
        except Exception as e:
            logger.error("❌ Error preprocessing image: %s", str(e))
            # Return original image if preprocessing fails
            return Image.open(image_path)
    
    def _clean_extracted_text(self, text: str) -> str:
        """
        🧹 Clean and normalize extracted text
        
        Args:
            text (str): Raw extracted text
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
        
        # Convert to lowercase for case-insensitive matching
        cleaned_text = text.lower()
        
        # Remove extra whitespace and newlines
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        
        # Remove special characters that might interfere with matching
        cleaned_text = re.sub(r'[^\w\s/.:()[\]{}+-=$%]', ' ', cleaned_text)
        
        # Strip leading/trailing whitespace
        cleaned_text = cleaned_text.strip()
        
        return cleaned_text
    
    def _analyze_extracted_text(self, text: str) -> Dict:
        """
        📊 Analyze extracted text for trading-related keywords
        
        Args:
            text (str): Cleaned extracted text
            
        Returns:
            dict: Analysis results with keywords found and confidence score
        """
        detected_keywords = []
        keyword_positions = []
        
        # Search for trading keywords in the text
        for keyword in self.trading_keywords:
            if keyword in text:
                detected_keywords.append(keyword)
                # Find all positions of this keyword
                start_pos = 0
                while True:
                    pos = text.find(keyword, start_pos)
                    if pos == -1:
                        break
                    keyword_positions.append((keyword, pos))
                    start_pos = pos + 1
        
        # Calculate confidence score based on various factors
        confidence_score = self._calculate_confidence_score(
            text, detected_keywords, keyword_positions
        )
        
        return {
            'keywords_found': len(detected_keywords),
            'detected_keywords': detected_keywords,
            'confidence_score': confidence_score,
            'keyword_positions': keyword_positions,
            'text_length': len(text)
        }
    
    def _calculate_confidence_score(self, text: str, keywords: List[str], positions: List[Tuple]) -> float:
        """
        🎯 Calculate confidence score for chart validity
        
        Args:
            text (str): Extracted text
            keywords (List[str]): Found keywords
            positions (List[Tuple]): Keyword positions
            
        Returns:
            float: Confidence score between 0 and 1
        """
        if not keywords or not text:
            return 0.0
        
        # Base score from keyword count
        keyword_score = min(len(keywords) / 10.0, 1.0)  # Max 1.0 for 10+ keywords
        
        # Bonus for currency pairs (strong indicator)
        currency_pairs = ["eur/usd", "gbp/usd", "usd/jpy", "aud/usd", "eurusd", "gbpusd", "usdjpy"]
        currency_bonus = 0.2 if any(pair in keywords for pair in currency_pairs) else 0.0
        
        # Bonus for platform names (strong indicator)
        platforms = ["quotex", "tradingview", "metatrader", "mt4", "mt5", "binomo"]
        platform_bonus = 0.3 if any(platform in keywords for platform in platforms) else 0.0
        
        # Bonus for trading terms
        trading_terms = ["payout", "investment", "call", "put", "trade", "trading"]
        trading_bonus = 0.2 if any(term in keywords for term in trading_terms) else 0.0
        
        # Penalty for very short text (likely false positive)
        length_penalty = 0.1 if len(text) < 50 else 0.0
        
        # Calculate final confidence score
        confidence = keyword_score + currency_bonus + platform_bonus + trading_bonus - length_penalty
        
        # Ensure score is between 0 and 1
        confidence = max(0.0, min(1.0, confidence))
        
        return confidence
    
    def get_analysis_details(self, image_path: str) -> Dict:
        """
        📊 Get detailed analysis of the image
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            dict: Detailed analysis results
        """
        try:
            extracted_text = self._extract_text_from_image(image_path)
            analysis = self._analyze_extracted_text(extracted_text)
            is_valid = self.is_valid_chart(image_path)
            
            return {
                'is_valid_chart': is_valid,
                'extracted_text': extracted_text,
                'keywords_found': analysis['keywords_found'],
                'detected_keywords': analysis['detected_keywords'],
                'confidence_score': analysis['confidence_score'],
                'analysis_details': analysis
            }
            
        except Exception as e:
            logger.error("❌ Error getting analysis details: %s", str(e))
            return {
                'is_valid_chart': False,
                'error': str(e)
            }

# 🧪 Testing function
def test_chart_checker():
    """Test function for the chart checker"""
    checker = ChartChecker()
    
    # Test with a sample image (you would replace this with actual test images)
    test_images = [
        "test_chart.png",
        "test_random.png"
    ]
    
    for image_path in test_images:
        if os.path.exists(image_path):
            print(f"\n🔍 Testing: {image_path}")
            result = checker.is_valid_chart(image_path)
            details = checker.get_analysis_details(image_path)
            print(f"Result: {'✅ Valid Chart' if result else '❌ Invalid Chart'}")
            print(f"Keywords: {details.get('detected_keywords', [])}")
            print(f"Confidence: {details.get('confidence_score', 0):.2f}")

if __name__ == "__main__":
    test_chart_checker()