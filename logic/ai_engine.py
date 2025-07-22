import cv2
import numpy as np
from PIL import Image
import math
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from config import Config

@dataclass
class Candle:
    """Represents a single candlestick with all its properties"""
    open_price: float
    close_price: float
    high_price: float
    low_price: float
    body_top: float
    body_bottom: float
    upper_wick: float
    lower_wick: float
    body_size: float
    total_size: float
    is_bullish: bool
    color: str
    position: int
    timestamp: Optional[datetime] = None

@dataclass
class MarketSignal:
    """Represents the final trading signal"""
    signal: str  # CALL, PUT, NO_TRADE
    confidence: float
    reasoning: str
    strategy: str
    market_psychology: str
    entry_time: datetime
    timeframe: str = "1M"
    patterns_detected: List[str] = None
    
    def __post_init__(self):
        if self.patterns_detected is None:
            self.patterns_detected = []

class CandleDetector:
    """Detects and extracts candlestick data from chart images"""
    
    def __init__(self):
        self.min_candle_height = 10
        self.min_candle_width = 3
        
    def preprocess_image(self, image_path: str) -> tuple:
        """Preprocess the chart image for candlestick detection"""
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not load image")
            
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        # Use Canny edge detection for better line detection
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        
        # Apply morphological operations to connect nearby edges
        kernel = np.ones((3,3), np.uint8)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        return img, gray, closed
    
    def detect_candlesticks(self, image_path: str) -> List[Candle]:
        """Detect candlesticks from chart image using advanced computer vision"""
        img, gray, edges = self.preprocess_image(image_path)
        height, width = img.shape[:2]
        
        # Detect horizontal and vertical lines (price levels and time separators)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        
        horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
        
        # Combine to get grid structure
        grid = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
        
        # Find potential candlestick regions by analyzing vertical patterns
        candles = self._extract_candlestick_data(img, gray, grid, width, height)
        
        # Sort by x-coordinate (time order) and return recent candles
        candles = sorted(candles, key=lambda c: c.position)
        recent_candles = candles[-Config.MAX_CANDLES_ANALYZED:] if len(candles) > Config.MAX_CANDLES_ANALYZED else candles
        
        # Normalize price data to relative values
        if recent_candles:
            self._normalize_price_data(recent_candles)
            
        return recent_candles
    
    def _extract_candlestick_data(self, img: np.ndarray, gray: np.ndarray, grid: np.ndarray, width: int, height: int) -> List[Candle]:
        """Extract candlestick data using robust color analysis"""
        candles = []
        
        # More flexible approach - scan multiple segment sizes
        for segment_count in [15, 20, 25, 30]:  # Try different granularities
            segment_width = width // segment_count
            temp_candles = []
            
            # Start from right side (most recent candles)
            start_segment = max(0, segment_count - 12)  # Last 12 segments
            
            for i in range(start_segment, segment_count):
                x_start = i * segment_width
                x_end = min((i + 1) * segment_width, width)
                
                if x_end - x_start < 3:  # Skip too narrow segments
                    continue
                    
                # Extract and analyze candle slice
                candle_slice = img[:, x_start:x_end]
                
                # Only use real analysis - no fallbacks
                candle = self._analyze_price_action_slice(candle_slice, None, x_start, i)
                
                if candle:
                    temp_candles.append(candle)
            
            # Use the segmentation that found the most candles
            if len(temp_candles) > len(candles):
                candles = temp_candles
                
        return candles[-8:] if len(candles) > 8 else candles  # Return last 8 candles
    
    def _analyze_price_action_slice(self, color_slice: np.ndarray, gray_slice: np.ndarray, x_pos: int, position: int) -> Optional[Candle]:
        """Analyze a vertical slice to extract REAL OHLC data from broker charts"""
        if color_slice.size == 0:
            return None
            
        height, width = color_slice.shape[:2]
        
        # Multiple detection methods for different broker chart types
        candle = None
        
        # Method 1: HSV Color Detection (for colored candles)
        candle = self._detect_colored_candle(color_slice, height, position)
        if candle:
            return candle
            
        # Method 2: Grayscale Analysis (for black/white candles)
        candle = self._detect_grayscale_candle(color_slice, height, position)
        if candle:
            return candle
            
        # Method 3: Edge Detection (for outlined candles)
        candle = self._detect_edge_candle(color_slice, height, position)
        if candle:
            return candle
            
        return None  # No valid candle detected
    
    def _detect_colored_candle(self, color_slice: np.ndarray, height: int, position: int) -> Optional[Candle]:
        """Detect colored candles (green/red) from broker charts"""
        hsv_slice = cv2.cvtColor(color_slice, cv2.COLOR_BGR2HSV)
        
        # Expanded color ranges for different broker themes
        green_ranges = [
            ([35, 30, 30], [85, 255, 255]),   # Standard green
            ([40, 40, 40], [80, 255, 255]),   # Bright green
            ([30, 20, 20], [90, 255, 255]),   # Wide green range
        ]
        
        red_ranges = [
            ([0, 30, 30], [20, 255, 255]),    # Red range 1
            ([160, 30, 30], [180, 255, 255]), # Red range 2  
            ([170, 40, 40], [180, 255, 255]), # Bright red
        ]
        
        # Combine all green masks
        green_mask = np.zeros(hsv_slice.shape[:2], dtype=np.uint8)
        for lower, upper in green_ranges:
            mask = cv2.inRange(hsv_slice, np.array(lower), np.array(upper))
            green_mask = cv2.bitwise_or(green_mask, mask)
        
        # Combine all red masks
        red_mask = np.zeros(hsv_slice.shape[:2], dtype=np.uint8)
        for lower, upper in red_ranges:
            mask = cv2.inRange(hsv_slice, np.array(lower), np.array(upper))
            red_mask = cv2.bitwise_or(red_mask, mask)
        
        green_pixels = cv2.countNonZero(green_mask)
        red_pixels = cv2.countNonZero(red_mask)
        
        # Need sufficient color pixels to be valid
        if green_pixels < 20 and red_pixels < 20:
            return None
            
        is_bullish = green_pixels > red_pixels
        dominant_mask = green_mask if is_bullish else red_mask
        
        return self._extract_ohlc_from_mask(dominant_mask, height, is_bullish, position)
    
    def _detect_grayscale_candle(self, color_slice: np.ndarray, height: int, position: int) -> Optional[Candle]:
        """Detect black/white candles from broker charts"""
        gray_slice = cv2.cvtColor(color_slice, cv2.COLOR_BGR2GRAY)
        
        # White candles (bullish)
        white_mask = cv2.threshold(gray_slice, 180, 255, cv2.THRESH_BINARY)[1]
        # Black/dark candles (bearish)
        black_mask = cv2.threshold(gray_slice, 80, 255, cv2.THRESH_BINARY_INV)[1]
        
        white_pixels = cv2.countNonZero(white_mask)
        black_pixels = cv2.countNonZero(black_mask)
        
        if white_pixels < 20 and black_pixels < 20:
            return None
            
        is_bullish = white_pixels > black_pixels
        dominant_mask = white_mask if is_bullish else black_mask
        
        return self._extract_ohlc_from_mask(dominant_mask, height, is_bullish, position)
    
    def _detect_edge_candle(self, color_slice: np.ndarray, height: int, position: int) -> Optional[Candle]:
        """Detect outlined candles using edge detection"""
        gray_slice = cv2.cvtColor(color_slice, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray_slice, 30, 100)
        
        # Dilate edges to make them thicker for analysis
        kernel = np.ones((2,2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        edge_pixels = cv2.countNonZero(edges)
        if edge_pixels < 10:
            return None
            
        # Analyze the filled vs empty areas to determine candle type
        avg_gray = np.mean(gray_slice)
        is_bullish = avg_gray > 128  # Brighter = bullish
        
        return self._extract_ohlc_from_mask(edges, height, is_bullish, position)
    
    def _extract_ohlc_from_mask(self, mask: np.ndarray, height: int, is_bullish: bool, position: int) -> Optional[Candle]:
        """Extract OHLC values from a binary mask"""
        # Find vertical profile of the mask
        vertical_profile = np.sum(mask, axis=1)
        non_zero_indices = np.where(vertical_profile > 0)[0]
        
        if len(non_zero_indices) < 3:  # Need at least some vertical extent
            return None
            
        # Extract key price levels
        highest_point = non_zero_indices[0]     # High
        lowest_point = non_zero_indices[-1]    # Low
        
        # Find body region (thickest part)
        max_thickness = np.max(vertical_profile)
        thick_threshold = max_thickness * 0.7
        body_indices = np.where(vertical_profile >= thick_threshold)[0]
        
        if len(body_indices) == 0:
            body_top_idx = highest_point
            body_bottom_idx = lowest_point
        else:
            body_top_idx = body_indices[0]
            body_bottom_idx = body_indices[-1]
        
        # Convert to price values (invert Y-axis)
        high_price = 100 - (highest_point / height) * 100
        low_price = 100 - (lowest_point / height) * 100
        
        if is_bullish:
            open_price = 100 - (body_bottom_idx / height) * 100
            close_price = 100 - (body_top_idx / height) * 100
        else:
            open_price = 100 - (body_top_idx / height) * 100
            close_price = 100 - (body_bottom_idx / height) * 100
        
        # Validate price relationships
        if high_price < low_price or abs(high_price - low_price) < 0.5:
            return None
            
        # Calculate candle properties
        body_top = max(open_price, close_price)
        body_bottom = min(open_price, close_price)
        body_size = abs(close_price - open_price)
        upper_wick = high_price - body_top
        lower_wick = body_bottom - low_price
        total_size = high_price - low_price
        
        # Quality checks
        if total_size < 1.0 or body_size < 0:
            return None
            
        return Candle(
            open_price=open_price,
            close_price=close_price,
            high_price=high_price,
            low_price=low_price,
            body_top=body_top,
            body_bottom=body_bottom,
            upper_wick=upper_wick,
            lower_wick=lower_wick,
            body_size=body_size,
            total_size=total_size,
            is_bullish=is_bullish,
            color="GREEN" if is_bullish else "RED",
            position=position,
            timestamp=datetime.now()
        )
    
    def _normalize_price_data(self, candles: List[Candle]) -> None:
        """Normalize price data to create consistent relative values"""
        if not candles:
            return
            
        # Find overall high and low for normalization
        all_highs = [c.high_price for c in candles]
        all_lows = [c.low_price for c in candles]
        
        global_high = max(all_highs)
        global_low = min(all_lows)
        price_range = global_high - global_low
        
        if price_range == 0:
            return
            
        # Normalize all prices to 0-100 scale
        for candle in candles:
            # Normalize each price point
            candle.high_price = ((candle.high_price - global_low) / price_range) * 100
            candle.low_price = ((candle.low_price - global_low) / price_range) * 100
            candle.open_price = ((candle.open_price - global_low) / price_range) * 100
            candle.close_price = ((candle.close_price - global_low) / price_range) * 100
            
            # Recalculate derived values
            candle.body_top = max(candle.open_price, candle.close_price)
            candle.body_bottom = min(candle.open_price, candle.close_price)
            candle.body_size = abs(candle.close_price - candle.open_price)
            candle.upper_wick = candle.high_price - candle.body_top
            candle.lower_wick = candle.body_bottom - candle.low_price
            candle.total_size = candle.high_price - candle.low_price
    
    def _analyze_simple_candle_slice_REMOVED(self, color_slice: np.ndarray, x_pos: int, position: int) -> Optional[Candle]:
        """Simple candle analysis using basic color detection"""
        if color_slice.size == 0:
            return None
            
        height, width = color_slice.shape[:2]
        
        # Convert to different color spaces for better detection
        hsv_slice = cv2.cvtColor(color_slice, cv2.COLOR_BGR2HSV)
        
        # Broader color detection for green/red
        # Green detection (multiple ranges)
        green_ranges = [
            ([30, 30, 30], [90, 255, 255]),  # Broader green range
            ([40, 20, 20], [80, 255, 255]),  # Alternative green
        ]
        
        green_mask = np.zeros(hsv_slice.shape[:2], dtype=np.uint8)
        for lower, upper in green_ranges:
            mask = cv2.inRange(hsv_slice, np.array(lower), np.array(upper))
            green_mask = cv2.bitwise_or(green_mask, mask)
        
        # Red detection (multiple ranges)
        red_ranges = [
            ([0, 30, 30], [30, 255, 255]),    # Red range 1
            ([150, 30, 30], [180, 255, 255]), # Red range 2
        ]
        
        red_mask = np.zeros(hsv_slice.shape[:2], dtype=np.uint8)
        for lower, upper in red_ranges:
            mask = cv2.inRange(hsv_slice, np.array(lower), np.array(upper))
            red_mask = cv2.bitwise_or(red_mask, mask)
        
        # Count pixels
        green_pixels = cv2.countNonZero(green_mask)
        red_pixels = cv2.countNonZero(red_mask)
        
        # Also try brightness-based detection for black/white candles
        gray_slice = cv2.cvtColor(color_slice, cv2.COLOR_BGR2GRAY)
        white_mask = cv2.threshold(gray_slice, 200, 255, cv2.THRESH_BINARY)[1]
        black_mask = cv2.threshold(gray_slice, 50, 255, cv2.THRESH_BINARY_INV)[1]
        
        white_pixels = cv2.countNonZero(white_mask)
        black_pixels = cv2.countNonZero(black_mask)
        
        # Determine candle type
        total_colored = green_pixels + red_pixels + white_pixels + black_pixels
        if total_colored < 10:  # Not enough signal
            return None
            
        is_bullish = (green_pixels + white_pixels) > (red_pixels + black_pixels)
        
        # Create synthetic OHLC based on position and trend
        base_price = 50 + (position * 0.5)  # Trending base
        volatility = 8 + (position % 5)
        
        if is_bullish:
            open_price = base_price + (position % 3) - 1
            close_price = open_price + volatility * 0.6
            high_price = close_price + volatility * 0.3
            low_price = open_price - volatility * 0.2
        else:
            open_price = base_price + volatility * 0.4
            close_price = base_price - (position % 3)
            high_price = open_price + volatility * 0.2
            low_price = close_price - volatility * 0.3
            
        return self._create_candle_from_prices(open_price, high_price, low_price, close_price, is_bullish, position)
    
    def _analyze_fallback_slice_REMOVED(self, color_slice: np.ndarray, x_pos: int, position: int) -> Optional[Candle]:
        """Fallback analysis - always generates a candle based on position"""
        if color_slice.size == 0:
            return None
            
        height, width = color_slice.shape[:2]
        
        # Analyze overall characteristics
        gray = cv2.cvtColor(color_slice, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        
        # Determine trend based on position (simulate market movement)
        trend_factor = (position % 8) / 8.0  # Cycling trend
        is_bullish = (position % 3) != 0  # 2/3 chance bullish (realistic market)
        
        # Generate realistic OHLC based on position in sequence
        base_price = 45 + (position * 1.2) + np.sin(position * 0.5) * 5  # Trending with noise
        volatility = 6 + (position % 4)
        
        # Use deterministic values based on position instead of random
        variation = (position * 0.7) % 1.0  # 0-1 based on position
        
        if is_bullish:
            open_price = base_price + (variation - 0.5) * 2
            close_price = open_price + volatility * (0.3 + variation * 0.5)
            high_price = max(open_price, close_price) + volatility * (0.1 + variation * 0.3)
            low_price = min(open_price, close_price) - volatility * (0.1 + variation * 0.2)
        else:
            open_price = base_price + (variation - 0.5) * 2
            close_price = open_price - volatility * (0.3 + variation * 0.5)
            high_price = max(open_price, close_price) + volatility * (0.1 + variation * 0.2)
            low_price = min(open_price, close_price) - volatility * (0.1 + variation * 0.3)
            
        return self._create_candle_from_prices(open_price, high_price, low_price, close_price, is_bullish, position)
    
    def _create_candle_from_prices(self, open_price: float, high_price: float, low_price: float, close_price: float, is_bullish: bool, position: int) -> Candle:
        """Create a Candle object from OHLC prices"""
        body_top = max(open_price, close_price)
        body_bottom = min(open_price, close_price)
        body_size = abs(close_price - open_price)
        upper_wick = high_price - body_top
        lower_wick = body_bottom - low_price
        total_size = high_price - low_price
        
        return Candle(
            open_price=open_price,
            close_price=close_price,
            high_price=high_price,
            low_price=low_price,
            body_top=body_top,
            body_bottom=body_bottom,
            upper_wick=upper_wick,
            lower_wick=lower_wick,
            body_size=body_size,
            total_size=total_size,
            is_bullish=is_bullish,
            color="GREEN" if is_bullish else "RED",
            position=position,
            timestamp=datetime.now()
        )
    
    def _analyze_candle_region(self, region: np.ndarray, x: int, y: int, w: int, h: int, position: int) -> Optional[Candle]:
        """Analyze individual candle region to extract OHLC data"""
        if region.size == 0:
            return None
            
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        
        # Detect if bullish (green) or bearish (red)
        green_mask = cv2.inRange(hsv, (40, 50, 50), (80, 255, 255))
        red_mask = cv2.inRange(hsv, (0, 50, 50), (20, 255, 255))
        
        green_pixels = cv2.countNonZero(green_mask)
        red_pixels = cv2.countNonZero(red_mask)
        
        is_bullish = green_pixels > red_pixels
        color = "GREEN" if is_bullish else "RED"
        
        # Estimate OHLC based on candle structure
        # This is a simplified approach - in real implementation, 
        # you'd need more sophisticated price extraction
        
        # Calculate relative prices based on candle position and size
        high_price = y  # Top of candle
        low_price = y + h  # Bottom of candle
        
        if is_bullish:
            open_price = low_price + (h * 0.2)  # Bottom part of body
            close_price = high_price + (h * 0.8)  # Top part of body
        else:
            open_price = high_price + (h * 0.8)  # Top part of body
            close_price = low_price + (h * 0.2)  # Bottom part of body
            
        # Calculate body and wick sizes
        body_top = max(open_price, close_price)
        body_bottom = min(open_price, close_price)
        body_size = abs(close_price - open_price)
        upper_wick = high_price - body_top
        lower_wick = body_bottom - low_price
        total_size = high_price - low_price
        
        return Candle(
            open_price=open_price,
            close_price=close_price,
            high_price=high_price,
            low_price=low_price,
            body_top=body_top,
            body_bottom=body_bottom,
            upper_wick=upper_wick,
            lower_wick=lower_wick,
            body_size=body_size,
            total_size=total_size,
            is_bullish=is_bullish,
            color=color,
            position=position
        )

class MarketPerceptionEngine:
    """Analyzes market psychology and patterns from candlestick sequences"""
    
    def __init__(self):
        self.patterns = {
            'ENGULFING': self._detect_engulfing,
            'DOJI': self._detect_doji,
            'HAMMER': self._detect_hammer,
            'SHOOTING_STAR': self._detect_shooting_star,
            'MOMENTUM_SHIFT': self._detect_momentum_shift,
            'EXHAUSTION': self._detect_exhaustion,
            'BREAKOUT': self._detect_breakout,
            'TRAP': self._detect_trap
        }
    
    def analyze_market_story(self, candles: List[Candle]) -> Dict:
        """Analyze the market story from candlestick sequence"""
        if len(candles) < Config.MIN_CANDLES_REQUIRED:
            return {'error': 'Insufficient candles for analysis'}
        
        story = {
            'patterns': [],
            'momentum': self._analyze_momentum(candles),
            'volatility': self._analyze_volatility(candles),
            'support_resistance': self._analyze_support_resistance(candles),
            'market_psychology': self._analyze_psychology(candles),
            'trend_direction': self._analyze_trend(candles),
            'volume_analysis': self._analyze_volume_proxy(candles)
        }
        
        # Detect patterns
        for pattern_name, detector in self.patterns.items():
            pattern_result = detector(candles)
            if pattern_result['detected']:
                story['patterns'].append({
                    'name': pattern_name,
                    'strength': pattern_result['strength'],
                    'location': pattern_result['location'],
                    'implications': pattern_result['implications']
                })
        
        return story
    
    def _analyze_momentum(self, candles: List[Candle]) -> Dict:
        """Analyze momentum characteristics"""
        recent_candles = candles[-4:]  # Last 4 candles
        
        bullish_count = sum(1 for c in recent_candles if c.is_bullish)
        bearish_count = len(recent_candles) - bullish_count
        
        avg_body_size = np.mean([c.body_size for c in recent_candles])
        momentum_direction = "BULLISH" if bullish_count > bearish_count else "BEARISH"
        momentum_strength = abs(bullish_count - bearish_count) / len(recent_candles)
        
        return {
            'direction': momentum_direction,
            'strength': momentum_strength,
            'avg_body_size': avg_body_size,
            'bullish_candles': bullish_count,
            'bearish_candles': bearish_count
        }
    
    def _analyze_volatility(self, candles: List[Candle]) -> Dict:
        """Analyze market volatility"""
        wick_ratios = [(c.upper_wick + c.lower_wick) / c.total_size for c in candles if c.total_size > 0]
        body_ratios = [c.body_size / c.total_size for c in candles if c.total_size > 0]
        
        avg_wick_ratio = np.mean(wick_ratios) if wick_ratios else 0
        avg_body_ratio = np.mean(body_ratios) if body_ratios else 0
        
        volatility_level = "HIGH" if avg_wick_ratio > 0.6 else "MEDIUM" if avg_wick_ratio > 0.3 else "LOW"
        
        return {
            'level': volatility_level,
            'wick_ratio': avg_wick_ratio,
            'body_ratio': avg_body_ratio,
            'indecision_level': 1 - avg_body_ratio
        }
    
    def _analyze_support_resistance(self, candles: List[Candle]) -> Dict:
        """Identify potential support and resistance levels"""
        highs = [c.high_price for c in candles]
        lows = [c.low_price for c in candles]
        
        resistance_level = max(highs)
        support_level = min(lows)
        current_price = candles[-1].close_price
        
        distance_to_resistance = abs(current_price - resistance_level)
        distance_to_support = abs(current_price - support_level)
        
        near_resistance = distance_to_resistance < (resistance_level - support_level) * 0.1
        near_support = distance_to_support < (resistance_level - support_level) * 0.1
        
        return {
            'resistance_level': resistance_level,
            'support_level': support_level,
            'current_price': current_price,
            'near_resistance': near_resistance,
            'near_support': near_support,
            'range_position': (current_price - support_level) / (resistance_level - support_level) if resistance_level != support_level else 0.5
        }
    
    def _analyze_psychology(self, candles: List[Candle]) -> str:
        """Analyze market psychology"""
        recent = candles[-3:]
        
        if all(c.is_bullish for c in recent):
            return "STRONG_BULLISH_SENTIMENT"
        elif all(not c.is_bullish for c in recent):
            return "STRONG_BEARISH_SENTIMENT"
        elif len(recent) >= 2 and recent[-1].is_bullish != recent[-2].is_bullish:
            return "INDECISION_REVERSAL_POTENTIAL"
        else:
            return "MIXED_SENTIMENT"
    
    def _analyze_trend(self, candles: List[Candle]) -> str:
        """Analyze overall trend direction"""
        closes = [c.close_price for c in candles]
        
        if len(closes) < 3:
            return "UNKNOWN"
        
        # Simple trend analysis using closing prices
        upward_moves = sum(1 for i in range(1, len(closes)) if closes[i] > closes[i-1])
        downward_moves = len(closes) - 1 - upward_moves
        
        if upward_moves > downward_moves * 1.5:
            return "UPTREND"
        elif downward_moves > upward_moves * 1.5:
            return "DOWNTREND"
        else:
            return "SIDEWAYS"
    
    def _analyze_volume_proxy(self, candles: List[Candle]) -> Dict:
        """Analyze volume proxy using body sizes"""
        body_sizes = [c.body_size for c in candles]
        avg_body_size = np.mean(body_sizes)
        recent_body_size = candles[-1].body_size
        
        volume_trend = "INCREASING" if recent_body_size > avg_body_size * 1.2 else "DECREASING" if recent_body_size < avg_body_size * 0.8 else "STABLE"
        
        return {
            'trend': volume_trend,
            'relative_strength': recent_body_size / avg_body_size if avg_body_size > 0 else 1,
            'avg_body_size': avg_body_size
        }
    
    # Pattern detection methods
    def _detect_engulfing(self, candles: List[Candle]) -> Dict:
        """Detect engulfing patterns"""
        if len(candles) < 2:
            return {'detected': False, 'strength': 0, 'location': -1, 'implications': ''}
        
        last_two = candles[-2:]
        prev_candle, curr_candle = last_two
        
        # Bullish engulfing
        bullish_engulfing = (not prev_candle.is_bullish and curr_candle.is_bullish and 
                           curr_candle.body_size > prev_candle.body_size * 1.2)
        
        # Bearish engulfing
        bearish_engulfing = (prev_candle.is_bullish and not curr_candle.is_bullish and 
                           curr_candle.body_size > prev_candle.body_size * 1.2)
        
        if bullish_engulfing or bearish_engulfing:
            pattern_type = "BULLISH_ENGULFING" if bullish_engulfing else "BEARISH_ENGULFING"
            strength = min(curr_candle.body_size / prev_candle.body_size, 2.0) / 2.0
            implications = f"Strong {pattern_type.split('_')[0].lower()} reversal signal"
            
            return {
                'detected': True,
                'strength': strength,
                'location': len(candles) - 1,
                'implications': implications,
                'type': pattern_type
            }
        
        return {'detected': False, 'strength': 0, 'location': -1, 'implications': ''}
    
    def _detect_doji(self, candles: List[Candle]) -> Dict:
        """Detect doji patterns"""
        last_candle = candles[-1]
        
        # Doji: body size is very small compared to total size
        if last_candle.total_size > 0:
            body_ratio = last_candle.body_size / last_candle.total_size
            if body_ratio < 0.1:  # Very small body
                return {
                    'detected': True,
                    'strength': 1 - body_ratio,
                    'location': len(candles) - 1,
                    'implications': 'Market indecision, potential reversal',
                    'type': 'DOJI'
                }
        
        return {'detected': False, 'strength': 0, 'location': -1, 'implications': ''}
    
    def _detect_hammer(self, candles: List[Candle]) -> Dict:
        """Detect hammer patterns"""
        last_candle = candles[-1]
        
        # Hammer: small body, long lower wick, small upper wick
        if (last_candle.total_size > 0 and 
            last_candle.lower_wick > last_candle.body_size * 2 and
            last_candle.upper_wick < last_candle.body_size * 0.5):
            
            strength = last_candle.lower_wick / last_candle.total_size
            return {
                'detected': True,
                'strength': strength,
                'location': len(candles) - 1,
                'implications': 'Bullish reversal signal at support',
                'type': 'HAMMER'
            }
        
        return {'detected': False, 'strength': 0, 'location': -1, 'implications': ''}
    
    def _detect_shooting_star(self, candles: List[Candle]) -> Dict:
        """Detect shooting star patterns"""
        last_candle = candles[-1]
        
        # Shooting star: small body, long upper wick, small lower wick
        if (last_candle.total_size > 0 and 
            last_candle.upper_wick > last_candle.body_size * 2 and
            last_candle.lower_wick < last_candle.body_size * 0.5):
            
            strength = last_candle.upper_wick / last_candle.total_size
            return {
                'detected': True,
                'strength': strength,
                'location': len(candles) - 1,
                'implications': 'Bearish reversal signal at resistance',
                'type': 'SHOOTING_STAR'
            }
        
        return {'detected': False, 'strength': 0, 'location': -1, 'implications': ''}
    
    def _detect_momentum_shift(self, candles: List[Candle]) -> Dict:
        """Detect momentum shift patterns"""
        if len(candles) < 3:
            return {'detected': False, 'strength': 0, 'location': -1, 'implications': ''}
        
        last_three = candles[-3:]
        
        # Check for momentum shift (change in candle types)
        color_changes = sum(1 for i in range(1, len(last_three)) 
                          if last_three[i].is_bullish != last_three[i-1].is_bullish)
        
        if color_changes >= 2:  # At least 2 color changes in 3 candles
            return {
                'detected': True,
                'strength': color_changes / 2.0,
                'location': len(candles) - 1,
                'implications': 'Momentum shift detected, trend change possible',
                'type': 'MOMENTUM_SHIFT'
            }
        
        return {'detected': False, 'strength': 0, 'location': -1, 'implications': ''}
    
    def _detect_exhaustion(self, candles: List[Candle]) -> Dict:
        """Detect exhaustion patterns"""
        if len(candles) < 4:
            return {'detected': False, 'strength': 0, 'location': -1, 'implications': ''}
        
        last_four = candles[-4:]
        
        # Check for decreasing body sizes (exhaustion)
        body_sizes = [c.body_size for c in last_four]
        is_decreasing = all(body_sizes[i] >= body_sizes[i+1] for i in range(len(body_sizes)-1))
        
        if is_decreasing and body_sizes[0] > body_sizes[-1] * 2:
            strength = (body_sizes[0] - body_sizes[-1]) / body_sizes[0]
            return {
                'detected': True,
                'strength': strength,
                'location': len(candles) - 1,
                'implications': 'Trend exhaustion, reversal likely',
                'type': 'EXHAUSTION'
            }
        
        return {'detected': False, 'strength': 0, 'location': -1, 'implications': ''}
    
    def _detect_breakout(self, candles: List[Candle]) -> Dict:
        """Detect breakout patterns"""
        if len(candles) < 5:
            return {'detected': False, 'strength': 0, 'location': -1, 'implications': ''}
        
        # Simple breakout detection based on recent high/low breaks
        recent_highs = [c.high_price for c in candles[:-1]]
        recent_lows = [c.low_price for c in candles[:-1]]
        
        last_candle = candles[-1]
        
        broke_resistance = last_candle.close_price > max(recent_highs)
        broke_support = last_candle.close_price < min(recent_lows)
        
        if broke_resistance or broke_support:
            breakout_type = "UPWARD" if broke_resistance else "DOWNWARD"
            return {
                'detected': True,
                'strength': 0.8,
                'location': len(candles) - 1,
                'implications': f'{breakout_type} breakout detected',
                'type': f'{breakout_type}_BREAKOUT'
            }
        
        return {'detected': False, 'strength': 0, 'location': -1, 'implications': ''}
    
    def _detect_trap(self, candles: List[Candle]) -> Dict:
        """Detect trap patterns (fake breakouts)"""
        if len(candles) < 6:
            return {'detected': False, 'strength': 0, 'location': -1, 'implications': ''}
        
        # Look for pattern: breakout followed by immediate reversal
        last_three = candles[-3:]
        
        # Simple trap detection: strong move followed by reversal
        if (len(last_three) >= 3 and 
            last_three[0].body_size > np.mean([c.body_size for c in candles[:-3]]) * 1.5 and
            last_three[0].is_bullish != last_three[-1].is_bullish and
            last_three[-1].body_size > last_three[0].body_size * 0.8):
            
            return {
                'detected': True,
                'strength': 0.7,
                'location': len(candles) - 1,
                'implications': 'Potential trap pattern, reversal signal',
                'type': 'TRAP_REVERSAL'
            }
        
        return {'detected': False, 'strength': 0, 'location': -1, 'implications': ''}

class StrategyEngine:
    """Generates trading strategies based on market analysis"""
    
    def __init__(self):
        self.strategies = {
            'BREAKOUT_CONTINUATION': self._breakout_continuation_strategy,
            'REVERSAL_PLAY': self._reversal_play_strategy,
            'MOMENTUM_SHIFT': self._momentum_shift_strategy,
            'TRAP_FADE': self._trap_fade_strategy,
            'EXHAUSTION_REVERSAL': self._exhaustion_reversal_strategy
        }
    
    def generate_strategy(self, market_story: Dict, candles: List[Candle]) -> Dict:
        """Generate the best strategy based on market analysis"""
        strategy_scores = {}
        
        # Evaluate each strategy
        for strategy_name, strategy_func in self.strategies.items():
            score = strategy_func(market_story, candles)
            strategy_scores[strategy_name] = score
        
        # Find the best strategy
        best_strategy = max(strategy_scores.items(), key=lambda x: x[1]['confidence'])
        
        return {
            'selected_strategy': best_strategy[0],
            'details': best_strategy[1],
            'all_scores': strategy_scores
        }
    
    def _breakout_continuation_strategy(self, market_story: Dict, candles: List[Candle]) -> Dict:
        """Enhanced Breakout Continuation Strategy - High Win Rate Focus"""
        confidence = 0
        signal = "NO_TRADE"
        reasoning = ""
        
        if len(candles) < 6:
            return {'signal': signal, 'confidence': confidence, 'reasoning': 'Insufficient data'}
        
        # Analyze last 6-8 candles for breakout setup
        recent_candles = candles[-6:]
        last_candle = candles[-1]
        prev_candle = candles[-2]
        
        # Detect strong breakout conditions
        sr_analysis = market_story['support_resistance']
        momentum = market_story['momentum']
        volatility = market_story['volatility']
        
        # High probability breakout signals
        breakout_detected = False
        breakout_type = ""
        
        # 1. Resistance breakout with volume surge
        if (sr_analysis['near_resistance'] and 
            last_candle.is_bullish and 
            last_candle.body_size > np.mean([c.body_size for c in recent_candles]) * 1.5):
            
            # Confirm with momentum
            if momentum['direction'] == 'BULLISH' and momentum['strength'] > 0.7:
                confidence = 88 + (momentum['strength'] * 7)
                signal = "CALL"
                reasoning = "Strong resistance breakout with volume surge and bullish momentum"
                breakout_detected = True
                breakout_type = "RESISTANCE_BREAK"
        
        # 2. Support breakdown with volume
        elif (sr_analysis['near_support'] and 
              not last_candle.is_bullish and 
              last_candle.body_size > np.mean([c.body_size for c in recent_candles]) * 1.5):
            
            if momentum['direction'] == 'BEARISH' and momentum['strength'] > 0.7:
                confidence = 88 + (momentum['strength'] * 7)
                signal = "PUT"
                reasoning = "Strong support breakdown with volume surge and bearish momentum"
                breakout_detected = True
                breakout_type = "SUPPORT_BREAK"
        
        # 3. Consolidation breakout pattern
        elif self._detect_consolidation_breakout(recent_candles):
            consolidation_result = self._analyze_consolidation_breakout(recent_candles, momentum)
            if consolidation_result['valid']:
                confidence = consolidation_result['confidence']
                signal = consolidation_result['signal']
                reasoning = consolidation_result['reasoning']
                breakout_detected = True
                breakout_type = "CONSOLIDATION_BREAK"
        
        # Apply additional filters for high win rate
        if breakout_detected and confidence > 85:
            # Trend alignment filter
            trend = market_story['trend_direction']
            if ((signal == "CALL" and trend in ["UPTREND", "SIDEWAYS"]) or
                (signal == "PUT" and trend in ["DOWNTREND", "SIDEWAYS"])):
                confidence += 3  # Bonus for trend alignment
            elif ((signal == "CALL" and trend == "DOWNTREND") or
                  (signal == "PUT" and trend == "UPTREND")):
                confidence -= 8  # Penalty for counter-trend
            
            # Time-of-trend filter (avoid late entries)
            if self._is_trend_mature(candles):
                confidence -= 5
        
        return {
            'signal': signal,
            'confidence': min(confidence, 97),
            'reasoning': reasoning
        }
    
    def _reversal_play_strategy(self, market_story: Dict, candles: List[Candle]) -> Dict:
        """Enhanced Reversal Play Strategy - High Probability Reversals"""
        confidence = 0
        signal = "NO_TRADE"
        reasoning = ""
        
        if len(candles) < 6:
            return {'signal': signal, 'confidence': confidence, 'reasoning': 'Insufficient data'}
        
        recent_candles = candles[-6:]
        last_candle = candles[-1]
        second_last = candles[-2]
        third_last = candles[-3]
        
        sr_analysis = market_story['support_resistance']
        trend = market_story['trend_direction']
        momentum = market_story['momentum']
        
        # 1. Double Bottom Reversal at Support
        if (sr_analysis['near_support'] and trend == 'DOWNTREND'):
            double_bottom = self._detect_double_bottom(recent_candles, sr_analysis)
            if double_bottom['detected']:
                confidence = 89 + double_bottom['strength']
                signal = "CALL"
                reasoning = "Double bottom reversal at key support with bullish confirmation"
        
        # 2. Double Top Reversal at Resistance  
        elif (sr_analysis['near_resistance'] and trend == 'UPTREND'):
            double_top = self._detect_double_top(recent_candles, sr_analysis)
            if double_top['detected']:
                confidence = 89 + double_top['strength']
                signal = "PUT"
                reasoning = "Double top reversal at key resistance with bearish confirmation"
        
        # 3. Hammer/Doji at Support (Strong Reversal)
        elif (sr_analysis['near_support'] and self._is_reversal_candle(last_candle, "BULLISH")):
            if (trend == 'DOWNTREND' and 
                last_candle.lower_wick > last_candle.body_size * 2 and
                last_candle.upper_wick < last_candle.body_size * 0.5):
                confidence = 87
                signal = "CALL"
                reasoning = "Strong hammer reversal at support after downtrend"
        
        # 4. Shooting Star at Resistance (Strong Reversal)
        elif (sr_analysis['near_resistance'] and self._is_reversal_candle(last_candle, "BEARISH")):
            if (trend == 'UPTREND' and 
                last_candle.upper_wick > last_candle.body_size * 2 and
                last_candle.lower_wick < last_candle.body_size * 0.5):
                confidence = 87
                signal = "PUT"
                reasoning = "Strong shooting star reversal at resistance after uptrend"
        
        # 5. Engulfing Pattern Reversal
        elif self._detect_engulfing_reversal(second_last, last_candle, sr_analysis):
            engulfing_result = self._analyze_engulfing_reversal(second_last, last_candle, trend)
            confidence = engulfing_result['confidence']
            signal = engulfing_result['signal']
            reasoning = engulfing_result['reasoning']
        
        # 6. Three-Candle Reversal Pattern
        elif self._detect_three_candle_reversal(third_last, second_last, last_candle, trend):
            three_candle_result = self._analyze_three_candle_reversal(recent_candles[-3:], trend)
            confidence = three_candle_result['confidence']
            signal = three_candle_result['signal']
            reasoning = three_candle_result['reasoning']
        
        # Apply momentum confirmation filter
        if confidence > 85:
            # Momentum divergence increases reversal probability
            if ((signal == "CALL" and momentum['direction'] == 'BEARISH') or
                (signal == "PUT" and momentum['direction'] == 'BULLISH')):
                confidence += 4  # Momentum divergence bonus
            
            # RSI-like analysis using price action
            recent_highs = [c.high_price for c in recent_candles]
            recent_lows = [c.low_price for c in recent_candles]
            price_momentum = (recent_highs[-1] - recent_lows[-1]) / (max(recent_highs) - min(recent_lows))
            
            if signal == "CALL" and price_momentum < 0.3:  # Oversold condition
                confidence += 3
            elif signal == "PUT" and price_momentum > 0.7:  # Overbought condition
                confidence += 3
        
        return {
            'signal': signal,
            'confidence': min(confidence, 96),
            'reasoning': reasoning
        }
    
    def _momentum_shift_strategy(self, market_story: Dict, candles: List[Candle]) -> Dict:
        """Momentum shift strategy"""
        confidence = 0
        signal = "NO_TRADE"
        reasoning = ""
        
        momentum = market_story['momentum']
        momentum_patterns = [p for p in market_story['patterns'] if 'MOMENTUM_SHIFT' in p['name']]
        
        if momentum_patterns or momentum['strength'] > 0.6:
            last_candle = candles[-1]
            
            # Strong bullish momentum shift
            if momentum['direction'] == 'BULLISH' and momentum['strength'] > 0.6:
                confidence = 70 + (momentum['strength'] * 20)
                signal = "CALL"
                reasoning = f"Strong bullish momentum shift detected ({momentum['bullish_candles']}/{len(candles[-4:])} recent bullish candles)"
            
            # Strong bearish momentum shift
            elif momentum['direction'] == 'BEARISH' and momentum['strength'] > 0.6:
                confidence = 70 + (momentum['strength'] * 20)
                signal = "PUT"
                reasoning = f"Strong bearish momentum shift detected ({momentum['bearish_candles']}/{len(candles[-4:])} recent bearish candles)"
        
        return {
            'signal': signal,
            'confidence': min(confidence, 95),
            'reasoning': reasoning
        }
    
    def _trap_fade_strategy(self, market_story: Dict, candles: List[Candle]) -> Dict:
        """Trap fade strategy (fade false breakouts)"""
        confidence = 0
        signal = "NO_TRADE"
        reasoning = ""
        
        trap_patterns = [p for p in market_story['patterns'] if 'TRAP' in p['name']]
        
        if trap_patterns:
            trap = trap_patterns[0]
            volatility = market_story['volatility']
            
            # High volatility traps are more reliable
            if volatility['level'] == 'HIGH':
                last_candle = candles[-1]
                
                # Fade the trap direction
                if last_candle.is_bullish:
                    confidence = 85 + (trap['strength'] * 10)
                    signal = "PUT"
                    reasoning = "Fading bullish trap pattern in high volatility"
                else:
                    confidence = 85 + (trap['strength'] * 10)
                    signal = "CALL"
                    reasoning = "Fading bearish trap pattern in high volatility"
        
        return {
            'signal': signal,
            'confidence': min(confidence, 95),
            'reasoning': reasoning
        }
    
    def _exhaustion_reversal_strategy(self, market_story: Dict, candles: List[Candle]) -> Dict:
        """Exhaustion reversal strategy"""
        confidence = 0
        signal = "NO_TRADE"
        reasoning = ""
        
        exhaustion_patterns = [p for p in market_story['patterns'] if 'EXHAUSTION' in p['name']]
        trend = market_story['trend_direction']
        
        if exhaustion_patterns:
            exhaustion = exhaustion_patterns[0]
            
            # Exhaustion in uptrend = bearish reversal
            if trend == 'UPTREND':
                confidence = 78 + (exhaustion['strength'] * 12)
                signal = "PUT"
                reasoning = "Trend exhaustion in uptrend, bearish reversal expected"
            
            # Exhaustion in downtrend = bullish reversal
            elif trend == 'DOWNTREND':
                confidence = 78 + (exhaustion['strength'] * 12)
                signal = "CALL"
                reasoning = "Trend exhaustion in downtrend, bullish reversal expected"
        
        return {
            'signal': signal,
            'confidence': min(confidence, 95),
            'reasoning': reasoning
        }
    
    # Helper methods for enhanced strategies
    def _detect_consolidation_breakout(self, candles: List[Candle]) -> bool:
        """Detect if price is breaking out of consolidation"""
        if len(candles) < 5:
            return False
            
        # Calculate price range for consolidation detection
        highs = [c.high_price for c in candles[:-1]]  # Exclude last candle
        lows = [c.low_price for c in candles[:-1]]
        
        price_range = max(highs) - min(lows)
        last_candle = candles[-1]
        
        # Breakout if last candle moves significantly beyond recent range
        breakout_threshold = price_range * 0.3
        
        return (last_candle.high_price > max(highs) + breakout_threshold or
                last_candle.low_price < min(lows) - breakout_threshold)
    
    def _analyze_consolidation_breakout(self, candles: List[Candle], momentum: Dict) -> Dict:
        """Analyze consolidation breakout for signal generation"""
        last_candle = candles[-1]
        prev_candles = candles[:-1]
        
        highs = [c.high_price for c in prev_candles]
        lows = [c.low_price for c in prev_candles]
        
        # Determine breakout direction
        if last_candle.high_price > max(highs):
            signal = "CALL"
            confidence = 86 + (momentum['strength'] * 8)
            reasoning = "Bullish consolidation breakout with strong momentum"
        elif last_candle.low_price < min(lows):
            signal = "PUT"
            confidence = 86 + (momentum['strength'] * 8)
            reasoning = "Bearish consolidation breakdown with strong momentum"
        else:
            return {'valid': False}
        
        return {
            'valid': True,
            'signal': signal,
            'confidence': min(confidence, 94),
            'reasoning': reasoning
        }
    
    def _is_trend_mature(self, candles: List[Candle]) -> bool:
        """Check if current trend is mature (avoid late entries)"""
        if len(candles) < 8:
            return False
            
        recent_8 = candles[-8:]
        same_direction_count = 0
        
        for i in range(1, len(recent_8)):
            prev_close = recent_8[i-1].close_price
            curr_close = recent_8[i].close_price
            
            if curr_close > prev_close:  # Upward move
                same_direction_count += 1
            elif curr_close < prev_close:  # Downward move
                same_direction_count -= 1
        
        # Trend is mature if 6+ candles in same direction
        return abs(same_direction_count) >= 6
    
    def _detect_double_bottom(self, candles: List[Candle], sr_analysis: Dict) -> Dict:
        """Detect double bottom reversal pattern"""
        if len(candles) < 5:
            return {'detected': False}
            
        lows = [c.low_price for c in candles]
        
        # Find two lowest points
        sorted_lows = sorted(enumerate(lows), key=lambda x: x[1])
        lowest_indices = [sorted_lows[0][0], sorted_lows[1][0]]
        
        # Check if they're reasonably spaced and at similar levels
        if (abs(lowest_indices[0] - lowest_indices[1]) >= 2 and
            abs(sorted_lows[0][1] - sorted_lows[1][1]) < 2.0):
            
            return {'detected': True, 'strength': 6}
        
        return {'detected': False}
    
    def _detect_double_top(self, candles: List[Candle], sr_analysis: Dict) -> Dict:
        """Detect double top reversal pattern"""
        if len(candles) < 5:
            return {'detected': False}
            
        highs = [c.high_price for c in candles]
        
        # Find two highest points
        sorted_highs = sorted(enumerate(highs), key=lambda x: x[1], reverse=True)
        highest_indices = [sorted_highs[0][0], sorted_highs[1][0]]
        
        # Check if they're reasonably spaced and at similar levels
        if (abs(highest_indices[0] - highest_indices[1]) >= 2 and
            abs(sorted_highs[0][1] - sorted_highs[1][1]) < 2.0):
            
            return {'detected': True, 'strength': 6}
        
        return {'detected': False}
    
    def _is_reversal_candle(self, candle: Candle, direction: str) -> bool:
        """Check if candle shows reversal characteristics"""
        if direction == "BULLISH":
            # Hammer-like: long lower wick, small upper wick
            return (candle.lower_wick > candle.body_size * 1.5 and
                    candle.upper_wick < candle.body_size * 0.5)
        else:  # BEARISH
            # Shooting star-like: long upper wick, small lower wick
            return (candle.upper_wick > candle.body_size * 1.5 and
                    candle.lower_wick < candle.body_size * 0.5)
    
    def _detect_engulfing_reversal(self, prev_candle: Candle, curr_candle: Candle, sr_analysis: Dict) -> bool:
        """Detect engulfing reversal pattern"""
        # Bullish engulfing: bearish candle followed by larger bullish candle
        bullish_engulfing = (not prev_candle.is_bullish and curr_candle.is_bullish and
                           curr_candle.body_size > prev_candle.body_size * 1.2 and
                           curr_candle.close_price > prev_candle.open_price and
                           curr_candle.open_price < prev_candle.close_price)
        
        # Bearish engulfing: bullish candle followed by larger bearish candle
        bearish_engulfing = (prev_candle.is_bullish and not curr_candle.is_bullish and
                           curr_candle.body_size > prev_candle.body_size * 1.2 and
                           curr_candle.close_price < prev_candle.open_price and
                           curr_candle.open_price > prev_candle.close_price)
        
        return bullish_engulfing or bearish_engulfing
    
    def _analyze_engulfing_reversal(self, prev_candle: Candle, curr_candle: Candle, trend: str) -> Dict:
        """Analyze engulfing pattern for signal generation"""
        if (not prev_candle.is_bullish and curr_candle.is_bullish and
            curr_candle.body_size > prev_candle.body_size * 1.2):
            return {
                'signal': 'CALL',
                'confidence': 88,
                'reasoning': 'Strong bullish engulfing reversal pattern'
            }
        elif (prev_candle.is_bullish and not curr_candle.is_bullish and
              curr_candle.body_size > prev_candle.body_size * 1.2):
            return {
                'signal': 'PUT',
                'confidence': 88,
                'reasoning': 'Strong bearish engulfing reversal pattern'
            }
        
        return {'signal': 'NO_TRADE', 'confidence': 0, 'reasoning': 'No valid engulfing pattern'}
    
    def _detect_three_candle_reversal(self, first: Candle, second: Candle, third: Candle, trend: str) -> bool:
        """Detect three-candle reversal patterns"""
        # Morning star pattern (bullish reversal)
        morning_star = (not first.is_bullish and 
                       second.body_size < first.body_size * 0.5 and
                       third.is_bullish and third.body_size > first.body_size * 0.8 and
                       trend == 'DOWNTREND')
        
        # Evening star pattern (bearish reversal)
        evening_star = (first.is_bullish and 
                       second.body_size < first.body_size * 0.5 and
                       not third.is_bullish and third.body_size > first.body_size * 0.8 and
                       trend == 'UPTREND')
        
        return morning_star or evening_star
    
    def _analyze_three_candle_reversal(self, candles: List[Candle], trend: str) -> Dict:
        """Analyze three-candle reversal for signal generation"""
        first, second, third = candles
        
        # Morning star
        if (not first.is_bullish and second.body_size < first.body_size * 0.5 and
            third.is_bullish and trend == 'DOWNTREND'):
            return {
                'signal': 'CALL',
                'confidence': 86,
                'reasoning': 'Three-candle morning star reversal pattern'
            }
        # Evening star
        elif (first.is_bullish and second.body_size < first.body_size * 0.5 and
              not third.is_bullish and trend == 'UPTREND'):
            return {
                'signal': 'PUT',
                'confidence': 86,
                'reasoning': 'Three-candle evening star reversal pattern'
            }
        
        return {'signal': 'NO_TRADE', 'confidence': 0, 'reasoning': 'No valid three-candle pattern'}

class CosmicAIEngine:
    """Main AI engine that orchestrates the analysis and signal generation"""
    
    def __init__(self):
        self.candle_detector = CandleDetector()
        self.perception_engine = MarketPerceptionEngine()
        self.strategy_engine = StrategyEngine()
    
    def analyze_chart(self, image_path: str) -> MarketSignal:
        """Analyze chart image and generate trading signal"""
        try:
            # Step 1: Detect candlesticks
            candles = self.candle_detector.detect_candlesticks(image_path)
            
            if len(candles) < Config.MIN_CANDLES_REQUIRED:
                return MarketSignal(
                    signal="NO_TRADE",
                    confidence=0,
                    reasoning="Insufficient candlesticks detected for analysis",
                    strategy="NONE",
                    market_psychology="UNKNOWN",
                    entry_time=datetime.now()
                )
            
            # Step 2: Analyze market story
            market_story = self.perception_engine.analyze_market_story(candles)
            
            if 'error' in market_story:
                return MarketSignal(
                    signal="NO_TRADE",
                    confidence=0,
                    reasoning=market_story['error'],
                    strategy="NONE",
                    market_psychology="UNKNOWN",
                    entry_time=datetime.now()
                )
            
            # Step 3: Generate strategy
            strategy_result = self.strategy_engine.generate_strategy(market_story, candles)
            
            best_strategy = strategy_result['selected_strategy']
            strategy_details = strategy_result['details']
            
            # Step 4: Apply confidence threshold
            if strategy_details['confidence'] < Config.CONFIDENCE_THRESHOLD:
                return MarketSignal(
                    signal="NO_TRADE",
                    confidence=strategy_details['confidence'],
                    reasoning=f"Strategy confidence ({strategy_details['confidence']:.1f}%) below threshold ({Config.CONFIDENCE_THRESHOLD}%)",
                    strategy=best_strategy,
                    market_psychology=market_story['market_psychology'],
                    entry_time=datetime.now()
                )
            
            # Step 5: Generate final signal
            patterns_found = [p['name'] for p in market_story.get('patterns', [])]
            
            return MarketSignal(
                signal=strategy_details['signal'],
                confidence=strategy_details['confidence'],
                reasoning=strategy_details['reasoning'],
                strategy=best_strategy,
                market_psychology=market_story['market_psychology'],
                entry_time=datetime.now(),
                timeframe=Config.SIGNAL_TIMEFRAME,
                patterns_detected=patterns_found
            )
            
        except Exception as e:
            return MarketSignal(
                signal="NO_TRADE",
                confidence=0,
                reasoning=f"Analysis error: {str(e)}",
                strategy="ERROR",
                market_psychology="UNKNOWN",
                entry_time=datetime.now()
            )