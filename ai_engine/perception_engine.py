#!/usr/bin/env python3
"""
🧠 PERCEPTION ENGINE - AI Eye for Chart Reading
Reads candle bodies, wicks, trends using OpenCV + HSV filters
Adapts to any broker UI, light/dark themes
"""

import cv2
import numpy as np
from PIL import Image
import io
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

class PerceptionEngine:
    """
    Advanced image processing engine that reads candlestick charts
    """
    
    def __init__(self):
        self.version = "∞ vX"
        self.detection_confidence = 0.85
        self.candle_patterns = []
        
    async def process_image(self, image_data: bytes) -> Optional[Dict]:
        """
        Main image processing pipeline
        """
        try:
            # Convert bytes to OpenCV image
            image = self._bytes_to_cv2(image_data)
            if image is None:
                return None
            
            # Enhance image quality
            enhanced_image = self._enhance_image(image)
            
            # Detect chart area
            chart_area = self._detect_chart_area(enhanced_image)
            
            # Extract candlestick data
            candles = await self._extract_candles(chart_area)
            
            # Analyze chart properties
            chart_properties = self._analyze_chart_properties(chart_area)
            
            # Detect patterns and formations
            patterns = self._detect_patterns(candles)
            
            result = {
                "candles": candles,
                "properties": chart_properties,
                "patterns": patterns,
                "image_quality": self._assess_image_quality(image),
                "broker_type": self._detect_broker_type(image),
                "timeframe": self._extract_timeframe(image)
            }
            
            logger.info(f"🧠 Perception Engine processed {len(candles)} candles")
            return result
            
        except Exception as e:
            logger.error(f"Perception Engine error: {str(e)}")
            return None
    
    def _bytes_to_cv2(self, image_data: bytes) -> Optional[np.ndarray]:
        """Convert bytes to OpenCV image"""
        try:
            image = Image.open(io.BytesIO(image_data))
            image_rgb = image.convert('RGB')
            return cv2.cvtColor(np.array(image_rgb), cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.error(f"Image conversion error: {str(e)}")
            return None
    
    def _enhance_image(self, image: np.ndarray) -> np.ndarray:
        """Enhance image quality for better analysis"""
        try:
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            # Noise reduction
            enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
            
            return enhanced
        except:
            return image
    
    def _detect_chart_area(self, image: np.ndarray) -> np.ndarray:
        """Detect the main chart area and crop it"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Find contours
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Find the largest rectangular contour (likely the chart)
            chart_contour = None
            max_area = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > max_area and area > 10000:  # Minimum area threshold
                    # Check if contour is roughly rectangular
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    if len(approx) >= 4:
                        max_area = area
                        chart_contour = contour
            
            if chart_contour is not None:
                x, y, w, h = cv2.boundingRect(chart_contour)
                return image[y:y+h, x:x+w]
            
            return image
        except:
            return image
    
    async def _extract_candles(self, chart_area: np.ndarray) -> List[Dict]:
        """Extract individual candlestick data"""
        try:
            candles = []
            height, width = chart_area.shape[:2]
            
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(chart_area, cv2.COLOR_BGR2HSV)
            
            # Define color ranges for bullish (green) and bearish (red) candles
            green_ranges = [
                (np.array([35, 40, 40]), np.array([85, 255, 255])),  # Green range 1
                (np.array([40, 50, 50]), np.array([80, 255, 255]))   # Green range 2
            ]
            
            red_ranges = [
                (np.array([0, 40, 40]), np.array([15, 255, 255])),   # Red range 1
                (np.array([160, 40, 40]), np.array([180, 255, 255])) # Red range 2
            ]
            
            # Detect vertical lines (candle wicks and bodies)
            gray = cv2.cvtColor(chart_area, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Find vertical lines
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=10, maxLineGap=5)
            
            if lines is not None:
                # Group lines into candle formations
                candle_positions = self._group_lines_to_candles(lines, width)
                
                for pos in candle_positions:
                    candle_data = self._analyze_candle_at_position(chart_area, hsv, pos, green_ranges, red_ranges)
                    if candle_data:
                        candles.append(candle_data)
            
            # Sort candles by x position (time order)
            candles.sort(key=lambda x: x.get('x_position', 0))
            
            # If we couldn't detect enough candles, use alternative method
            if len(candles) < 5:
                candles = await self._fallback_candle_detection(chart_area)
            
            return candles[-20:]  # Return last 20 candles for analysis
            
        except Exception as e:
            logger.error(f"Candle extraction error: {str(e)}")
            return []
    
    def _group_lines_to_candles(self, lines: np.ndarray, width: int) -> List[int]:
        """Group detected lines into candle x-positions"""
        x_positions = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Check if line is roughly vertical
            if abs(x2 - x1) < 5:  # Vertical tolerance
                x_pos = (x1 + x2) // 2
                x_positions.append(x_pos)
        
        # Remove duplicates and sort
        x_positions = sorted(list(set(x_positions)))
        
        # Group nearby positions
        grouped_positions = []
        current_group = [x_positions[0]] if x_positions else []
        
        for i in range(1, len(x_positions)):
            if x_positions[i] - x_positions[i-1] < 10:  # Group threshold
                current_group.append(x_positions[i])
            else:
                if current_group:
                    grouped_positions.append(sum(current_group) // len(current_group))
                current_group = [x_positions[i]]
        
        if current_group:
            grouped_positions.append(sum(current_group) // len(current_group))
        
        return grouped_positions
    
    def _analyze_candle_at_position(self, chart_area: np.ndarray, hsv: np.ndarray, x_pos: int, green_ranges: List, red_ranges: List) -> Optional[Dict]:
        """Analyze individual candle at given x position"""
        try:
            height, width = chart_area.shape[:2]
            candle_width = max(3, width // 100)  # Adaptive candle width
            
            # Extract vertical slice
            x_start = max(0, x_pos - candle_width//2)
            x_end = min(width, x_pos + candle_width//2)
            
            candle_slice = chart_area[:, x_start:x_end]
            hsv_slice = hsv[:, x_start:x_end]
            
            # Detect candle type (bullish/bearish)
            candle_type = self._detect_candle_type(hsv_slice, green_ranges, red_ranges)
            
            # Find OHLC values
            ohlc = self._extract_ohlc_from_slice(candle_slice, hsv_slice)
            
            if ohlc:
                return {
                    'x_position': x_pos,
                    'type': candle_type,
                    'open': ohlc['open'],
                    'high': ohlc['high'],
                    'low': ohlc['low'],
                    'close': ohlc['close'],
                    'body_size': abs(ohlc['close'] - ohlc['open']),
                    'upper_wick': ohlc['high'] - max(ohlc['open'], ohlc['close']),
                    'lower_wick': min(ohlc['open'], ohlc['close']) - ohlc['low'],
                    'volume_indicator': self._estimate_volume(candle_slice)
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Candle analysis error: {str(e)}")
            return None
    
    def _detect_candle_type(self, hsv_slice: np.ndarray, green_ranges: List, red_ranges: List) -> str:
        """Detect if candle is bullish (green) or bearish (red)"""
        try:
            green_pixels = 0
            red_pixels = 0
            
            for green_range in green_ranges:
                mask = cv2.inRange(hsv_slice, green_range[0], green_range[1])
                green_pixels += np.sum(mask > 0)
            
            for red_range in red_ranges:
                mask = cv2.inRange(hsv_slice, red_range[0], red_range[1])
                red_pixels += np.sum(mask > 0)
            
            if green_pixels > red_pixels:
                return "bullish"
            elif red_pixels > green_pixels:
                return "bearish"
            else:
                return "neutral"
                
        except:
            return "unknown"
    
    def _extract_ohlc_from_slice(self, candle_slice: np.ndarray, hsv_slice: np.ndarray) -> Optional[Dict]:
        """Extract OHLC values from candle slice"""
        try:
            height, width = candle_slice.shape[:2]
            
            # Find the darkest/brightest regions (candle body and wicks)
            gray_slice = cv2.cvtColor(candle_slice, cv2.COLOR_BGR2GRAY)
            
            # Find contours in the slice
            edges = cv2.Canny(gray_slice, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find the main candle contour
                main_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(main_contour)
                
                # Estimate OHLC based on contour bounds
                high = height - y  # Inverted Y coordinate
                low = height - (y + h)
                
                # Estimate open/close based on candle type and body position
                body_top = height - y
                body_bottom = height - (y + h)
                
                # Simple estimation - can be improved with more sophisticated analysis
                if body_top > body_bottom:
                    close = body_top
                    open_val = body_bottom
                else:
                    close = body_bottom
                    open_val = body_top
                
                return {
                    'open': max(0, open_val),
                    'high': max(0, high),
                    'low': max(0, low),
                    'close': max(0, close)
                }
            
            return None
            
        except:
            return None
    
    def _estimate_volume(self, candle_slice: np.ndarray) -> float:
        """Estimate volume indicator from candle thickness/intensity"""
        try:
            # Simple volume estimation based on candle body thickness
            gray = cv2.cvtColor(candle_slice, cv2.COLOR_BGR2GRAY)
            non_zero_pixels = np.count_nonzero(gray < 200)  # Count dark pixels
            total_pixels = gray.shape[0] * gray.shape[1]
            
            return non_zero_pixels / total_pixels if total_pixels > 0 else 0.0
        except:
            return 0.0
    
    async def _fallback_candle_detection(self, chart_area: np.ndarray) -> List[Dict]:
        """Fallback method for candle detection using grid analysis"""
        try:
            candles = []
            height, width = chart_area.shape[:2]
            
            # Divide chart into vertical sections
            num_sections = min(20, width // 10)
            section_width = width // num_sections
            
            for i in range(num_sections):
                x_start = i * section_width
                x_end = min(width, (i + 1) * section_width)
                
                section = chart_area[:, x_start:x_end]
                
                # Analyze this section for price data
                gray_section = cv2.cvtColor(section, cv2.COLOR_BGR2GRAY)
                
                # Find price levels by analyzing horizontal distribution
                intensity_profile = np.mean(gray_section, axis=1)
                
                # Detect significant changes in intensity (price levels)
                diff = np.diff(intensity_profile)
                peaks = np.where(np.abs(diff) > np.std(diff))[0]
                
                if len(peaks) >= 2:
                    high_y = min(peaks)
                    low_y = max(peaks)
                    
                    # Estimate open/close
                    mid_point = (high_y + low_y) // 2
                    close_y = peaks[0] if peaks[0] < mid_point else peaks[-1]
                    open_y = peaks[-1] if close_y == peaks[0] else peaks[0]
                    
                    candle = {
                        'x_position': (x_start + x_end) // 2,
                        'type': 'bullish' if close_y < open_y else 'bearish',
                        'open': height - open_y,
                        'high': height - high_y,
                        'low': height - low_y,
                        'close': height - close_y,
                        'body_size': abs(close_y - open_y),
                        'upper_wick': high_y - min(open_y, close_y),
                        'lower_wick': max(open_y, close_y) - low_y,
                        'volume_indicator': 0.5
                    }
                    
                    candles.append(candle)
            
            return candles
            
        except Exception as e:
            logger.error(f"Fallback detection error: {str(e)}")
            return []
    
    def _analyze_chart_properties(self, chart_area: np.ndarray) -> Dict:
        """Analyze overall chart properties"""
        try:
            height, width = chart_area.shape[:2]
            
            # Detect trend direction
            gray = cv2.cvtColor(chart_area, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Use Hough transform to detect trend lines
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=width//4, maxLineGap=10)
            
            trend_direction = "sideways"
            if lines is not None:
                angles = []
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    if abs(x2 - x1) > 10:  # Ignore vertical lines
                        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                        angles.append(angle)
                
                if angles:
                    avg_angle = np.mean(angles)
                    if avg_angle > 5:
                        trend_direction = "downtrend"
                    elif avg_angle < -5:
                        trend_direction = "uptrend"
            
            return {
                'width': width,
                'height': height,
                'trend_direction': trend_direction,
                'chart_quality': self._assess_image_quality(chart_area),
                'complexity': len(lines) if lines is not None else 0
            }
            
        except:
            return {'trend_direction': 'unknown', 'chart_quality': 0.5}
    
    def _detect_patterns(self, candles: List[Dict]) -> List[Dict]:
        """Detect candlestick patterns and formations"""
        patterns = []
        
        if len(candles) < 3:
            return patterns
        
        try:
            # Detect common patterns
            patterns.extend(self._detect_doji_patterns(candles))
            patterns.extend(self._detect_hammer_patterns(candles))
            patterns.extend(self._detect_engulfing_patterns(candles))
            patterns.extend(self._detect_momentum_patterns(candles))
            
        except Exception as e:
            logger.error(f"Pattern detection error: {str(e)}")
        
        return patterns
    
    def _detect_doji_patterns(self, candles: List[Dict]) -> List[Dict]:
        """Detect Doji patterns"""
        patterns = []
        
        for i, candle in enumerate(candles):
            body_size = candle.get('body_size', 0)
            total_range = candle.get('high', 0) - candle.get('low', 0)
            
            if total_range > 0 and body_size / total_range < 0.1:
                patterns.append({
                    'type': 'doji',
                    'position': i,
                    'strength': 1.0 - (body_size / total_range),
                    'signal': 'reversal_potential'
                })
        
        return patterns
    
    def _detect_hammer_patterns(self, candles: List[Dict]) -> List[Dict]:
        """Detect Hammer and Hanging Man patterns"""
        patterns = []
        
        for i, candle in enumerate(candles):
            body_size = candle.get('body_size', 0)
            lower_wick = candle.get('lower_wick', 0)
            upper_wick = candle.get('upper_wick', 0)
            
            # Hammer: small body, long lower wick, small upper wick
            if lower_wick > body_size * 2 and upper_wick < body_size * 0.5:
                pattern_type = 'hammer' if i > 0 and candles[i-1].get('type') == 'bearish' else 'hanging_man'
                patterns.append({
                    'type': pattern_type,
                    'position': i,
                    'strength': min(1.0, lower_wick / (body_size + 0.1)),
                    'signal': 'bullish_reversal' if pattern_type == 'hammer' else 'bearish_reversal'
                })
        
        return patterns
    
    def _detect_engulfing_patterns(self, candles: List[Dict]) -> List[Dict]:
        """Detect Bullish/Bearish Engulfing patterns"""
        patterns = []
        
        for i in range(1, len(candles)):
            prev_candle = candles[i-1]
            curr_candle = candles[i]
            
            prev_body_size = prev_candle.get('body_size', 0)
            curr_body_size = curr_candle.get('body_size', 0)
            
            # Engulfing: current candle body engulfs previous candle body
            if curr_body_size > prev_body_size * 1.5:
                if prev_candle.get('type') == 'bearish' and curr_candle.get('type') == 'bullish':
                    patterns.append({
                        'type': 'bullish_engulfing',
                        'position': i,
                        'strength': curr_body_size / (prev_body_size + 0.1),
                        'signal': 'bullish_reversal'
                    })
                elif prev_candle.get('type') == 'bullish' and curr_candle.get('type') == 'bearish':
                    patterns.append({
                        'type': 'bearish_engulfing',
                        'position': i,
                        'strength': curr_body_size / (prev_body_size + 0.1),
                        'signal': 'bearish_reversal'
                    })
        
        return patterns
    
    def _detect_momentum_patterns(self, candles: List[Dict]) -> List[Dict]:
        """Detect momentum and continuation patterns"""
        patterns = []
        
        if len(candles) < 5:
            return patterns
        
        # Detect consecutive candles in same direction
        bullish_streak = 0
        bearish_streak = 0
        
        for candle in candles[-5:]:
            if candle.get('type') == 'bullish':
                bullish_streak += 1
                bearish_streak = 0
            elif candle.get('type') == 'bearish':
                bearish_streak += 1
                bullish_streak = 0
            else:
                bullish_streak = bearish_streak = 0
        
        if bullish_streak >= 3:
            patterns.append({
                'type': 'bullish_momentum',
                'position': len(candles) - 1,
                'strength': min(1.0, bullish_streak / 5.0),
                'signal': 'continuation_bullish'
            })
        elif bearish_streak >= 3:
            patterns.append({
                'type': 'bearish_momentum',
                'position': len(candles) - 1,
                'strength': min(1.0, bearish_streak / 5.0),
                'signal': 'continuation_bearish'
            })
        
        return patterns
    
    def _assess_image_quality(self, image: np.ndarray) -> float:
        """Assess the quality of the input image"""
        try:
            # Calculate image sharpness using Laplacian variance
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Normalize to 0-1 range
            quality = min(1.0, laplacian_var / 1000.0)
            
            return quality
        except:
            return 0.5
    
    def _detect_broker_type(self, image: np.ndarray) -> str:
        """Attempt to detect broker type from UI elements"""
        # This is a simplified version - can be expanded with specific broker detection
        height, width = image.shape[:2]
        
        if width > height * 2:
            return "desktop_platform"
        else:
            return "mobile_platform"
    
    def _extract_timeframe(self, image: np.ndarray) -> str:
        """Extract timeframe information from chart"""
        # Simplified timeframe detection - can be improved with OCR
        return "1M"  # Default assumption