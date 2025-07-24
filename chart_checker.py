import pytesseract
from PIL import Image
import logging
import os
import re

# Configure logging
logger = logging.getLogger(__name__)

class ChartChecker:
    """
    A class to validate trading chart screenshots using OCR text extraction
    """
    
    def __init__(self):
        """Initialize the ChartChecker with trading platform keywords"""
        # Keywords that indicate a valid trading chart
        self.chart_keywords = [
            # Trading platforms
            "quotex", "binomo", "tradingview", "metatrader", "mt4", "mt5", 
            "iq option", "iqoption", "olymp trade", "olymptrade", "pocket option",
            "pocketoption", "expert option", "expertoption",
            
            # Currency pairs and symbols
            "usd", "eur", "gbp", "jpy", "aud", "cad", "chf", "nzd",
            "eurusd", "gbpusd", "usdjpy", "audusd", "usdcad", "usdchf",
            "btc", "eth", "bitcoin", "ethereum", "crypto",
            
            # Trading interface elements
            "payout", "investment", "amount", "profit", "loss",
            "call", "put", "higher", "lower", "up", "down",
            "expiry", "expiration", "duration", "timer",
            "balance", "price", "rate", "percentage", "%",
            
            # Chart indicators and elements
            "chart", "candlestick", "line chart", "volume",
            "support", "resistance", "trend", "analysis",
            "buy", "sell", "trade", "trading", "market",
            
            # Time-related keywords
            "1m", "5m", "15m", "30m", "1h", "4h", "1d",
            "minute", "minutes", "hour", "hours", "second", "seconds",
            
            # Platform-specific terms
            "otc", "turbo", "digital", "binary", "option", "options",
            "asset", "assets", "index", "indices", "commodity", "forex",
            
            # Interface elements
            "open", "close", "high", "low", "bid", "ask", "spread"
        ]
        
        # Minimum number of keywords required for validation
        self.min_keywords_required = 2
        
        # Common false positive words to avoid
        self.false_positive_words = [
            "screenshot", "photo", "image", "picture", "camera",
            "gallery", "album", "download", "upload", "share"
        ]
    
    def is_valid_chart(self, image_path: str) -> bool:
        """
        Check if the image contains a valid trading chart
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            bool: True if valid trading chart, False otherwise
        """
        try:
            # Check if image file exists
            if not os.path.exists(image_path):
                logger.error(f"Image file not found: {image_path}")
                return False
            
            # Extract text from image using OCR
            extracted_text = self._extract_text_from_image(image_path)
            
            if not extracted_text:
                logger.warning("No text extracted from image")
                return False
            
            # Analyze the extracted text
            is_valid = self._analyze_extracted_text(extracted_text)
            
            logger.info(f"Chart validation result: {is_valid} for image: {image_path}")
            logger.debug(f"Extracted text sample: {extracted_text[:200]}...")
            
            return is_valid
            
        except Exception as e:
            logger.error(f"Error validating chart: {e}")
            return False
    
    def _extract_text_from_image(self, image_path: str) -> str:
        """
        Extract text from image using pytesseract OCR
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            str: Extracted text from the image
        """
        try:
            # Open and process the image
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Apply basic preprocessing for better OCR results
                img = self._preprocess_image(img)
                
                # Extract text using pytesseract
                extracted_text = pytesseract.image_to_string(img, config='--psm 6')
                
                return extracted_text.strip()
                
        except Exception as e:
            logger.error(f"Error extracting text from image: {e}")
            return ""
    
    def _preprocess_image(self, img: Image.Image) -> Image.Image:
        """
        Apply basic preprocessing to improve OCR accuracy
        
        Args:
            img (Image.Image): PIL Image object
            
        Returns:
            Image.Image: Preprocessed image
        """
        try:
            # Resize image if it's too small (improves OCR accuracy)
            width, height = img.size
            if width < 800 or height < 600:
                scale_factor = max(800/width, 600/height)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            return img
            
        except Exception as e:
            logger.warning(f"Error preprocessing image: {e}")
            return img
    
    def _analyze_extracted_text(self, text: str) -> bool:
        """
        Analyze extracted text to determine if it's from a trading chart
        
        Args:
            text (str): Extracted text from the image
            
        Returns:
            bool: True if text indicates a trading chart, False otherwise
        """
        if not text:
            return False
        
        # Convert text to lowercase for case-insensitive matching
        text_lower = text.lower()
        
        # Remove extra whitespace and normalize
        text_normalized = re.sub(r'\s+', ' ', text_lower).strip()
        
        # Count matching keywords
        keyword_matches = []
        
        for keyword in self.chart_keywords:
            if keyword in text_normalized:
                keyword_matches.append(keyword)
        
        # Remove duplicates
        keyword_matches = list(set(keyword_matches))
        
        # Check for false positives
        false_positive_count = 0
        for false_word in self.false_positive_words:
            if false_word in text_normalized:
                false_positive_count += 1
        
        # Log analysis details
        logger.debug(f"Found {len(keyword_matches)} chart keywords: {keyword_matches[:5]}...")
        logger.debug(f"Found {false_positive_count} false positive indicators")
        
        # Validation logic
        # Must have minimum required keywords and low false positive count
        has_enough_keywords = len(keyword_matches) >= self.min_keywords_required
        has_low_false_positives = false_positive_count <= 1
        
        # Additional checks for better accuracy
        has_currency_pair = any(
            keyword in text_normalized 
            for keyword in ["usd", "eur", "gbp", "jpy", "btc", "eth"]
        )
        
        has_trading_platform = any(
            keyword in text_normalized 
            for keyword in ["quotex", "tradingview", "metatrader", "binomo", "iq option"]
        )
        
        has_trading_elements = any(
            keyword in text_normalized 
            for keyword in ["payout", "investment", "call", "put", "expiry", "balance"]
        )
        
        # Final validation decision
        is_valid = (
            has_enough_keywords and 
            has_low_false_positives and 
            (has_currency_pair or has_trading_platform or has_trading_elements)
        )
        
        return is_valid
    
    def get_detected_keywords(self, image_path: str) -> list:
        """
        Get list of detected trading keywords from an image (for debugging)
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            list: List of detected keywords
        """
        try:
            text = self._extract_text_from_image(image_path)
            text_lower = text.lower()
            
            detected = []
            for keyword in self.chart_keywords:
                if keyword in text_lower:
                    detected.append(keyword)
            
            return list(set(detected))
            
        except Exception as e:
            logger.error(f"Error getting detected keywords: {e}")
            return []