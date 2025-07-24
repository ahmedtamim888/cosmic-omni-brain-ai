"""
Chart Validation Module - Ensures only genuine trading charts are analyzed
Rejects random images, photos, text, etc.
"""

import cv2
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class ChartValidator:
    """Validates that an image is actually a trading chart before analysis"""
    
    def __init__(self):
        self.min_chart_width = 300
        self.min_chart_height = 200
        
    def validate_chart_image(self, image_path: str) -> dict:
        """
        Validate if image is a genuine trading chart
        Returns: {'is_valid': bool, 'reason': str, 'confidence': float}
        """
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                return {'is_valid': False, 'reason': 'Cannot load image', 'confidence': 0.0}
            
            height, width = img.shape[:2]
            
            # Basic size check
            if width < self.min_chart_width or height < self.min_chart_height:
                return {
                    'is_valid': False, 
                    'reason': f'Image too small for chart analysis. Minimum: {self.min_chart_width}x{self.min_chart_height}',
                    'confidence': 0.0
                }
            
            # Run multiple validation checks
            checks = [
                self._check_chart_structure(img),
                self._check_candlestick_patterns(img),
                self._check_price_axis(img),
                self._check_time_axis(img),
                self._check_chart_colors(img),
                self._reject_common_non_charts(img)
            ]
            
            # Smart validation logic - must have key trading chart characteristics
            valid_checks = [c for c in checks if c['is_valid']]
            failed_checks = [c for c in checks if not c['is_valid']]
            
            # Key requirements: Must have either chart structure OR candlestick patterns
            has_structure = any('grid structure' in c['reason'].lower() for c in valid_checks)
            has_candles = any('candlestick patterns' in c['reason'].lower() for c in valid_checks)
            has_colors = any('chart colors' in c['reason'].lower() for c in valid_checks)
            passes_non_chart_test = any('rejection tests' in c['reason'].lower() for c in valid_checks)
            
            # If it's clearly not a chart (fails non-chart rejection), reject immediately
            if not passes_non_chart_test:
                return {
                    'is_valid': False,
                    'reason': "This appears to be a photo or non-trading content, not a chart",
                    'confidence': 0.0
                }
            
            # Must have at least 2 of: structure, candles, or colors
            key_features = sum([has_structure, has_candles, has_colors])
            if key_features < 2:
                reasons = [c['reason'] for c in failed_checks]
                return {
                    'is_valid': False,
                    'reason': f"Missing key chart features. Issues: {'; '.join(reasons[:2])}",
                    'confidence': 0.0
                }
            
            # Calculate confidence from valid checks
            if valid_checks:
                avg_confidence = np.mean([c['confidence'] for c in valid_checks])
                
                # Lower threshold if we have the key features
                threshold = 0.5 if key_features >= 2 else 0.7
                
                if avg_confidence >= threshold:
                    return {
                        'is_valid': True,
                        'reason': 'Valid trading chart detected',
                        'confidence': avg_confidence
                    }
                else:
                    return {
                        'is_valid': False,
                        'reason': 'Chart characteristics too weak to be reliable',
                        'confidence': avg_confidence
                    }
            else:
                return {
                    'is_valid': False,
                    'reason': 'No chart characteristics detected',
                    'confidence': 0.0
                }
            
            return {
                'is_valid': True,
                'reason': 'Valid trading chart detected',
                'confidence': avg_confidence
            }
            
        except Exception as e:
            return {
                'is_valid': False,
                'reason': f'Validation error: {str(e)}',
                'confidence': 0.0
            }
    
    def _check_chart_structure(self, img: np.ndarray) -> dict:
        """Check for basic chart structure (grid lines, axes)"""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect horizontal and vertical lines (typical in charts)
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            
            horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
            vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)
            
            h_line_count = cv2.countNonZero(horizontal_lines)
            v_line_count = cv2.countNonZero(vertical_lines)
            
            # Charts typically have grid structure
            if h_line_count > 100 and v_line_count > 50:
                confidence = min(0.9, (h_line_count + v_line_count) / 2000)
                return {'is_valid': True, 'reason': 'Chart grid structure detected', 'confidence': confidence}
            else:
                return {'is_valid': False, 'reason': 'No chart grid structure found', 'confidence': 0.1}
                
        except:
            return {'is_valid': False, 'reason': 'Structure analysis failed', 'confidence': 0.0}
    
    def _check_candlestick_patterns(self, img: np.ndarray) -> dict:
        """Check for candlestick-like vertical patterns"""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            
            # Look for vertical patterns (candles)
            vertical_segments = 0
            segment_width = width // 20
            
            for i in range(5, 15):  # Check middle sections
                x_start = i * segment_width
                x_end = (i + 1) * segment_width
                
                if x_end > width:
                    break
                    
                segment = gray[:, x_start:x_end]
                
                # Look for vertical variation (typical in candlesticks)
                vertical_std = np.std(np.mean(segment, axis=1))
                
                if vertical_std > 10:  # Sufficient vertical variation
                    vertical_segments += 1
            
            if vertical_segments >= 3:  # Found multiple candlestick-like patterns
                confidence = min(0.8, vertical_segments / 10)
                return {'is_valid': True, 'reason': 'Candlestick patterns detected', 'confidence': confidence}
            else:
                return {'is_valid': False, 'reason': 'No candlestick patterns found', 'confidence': 0.2}
                
        except:
            return {'is_valid': False, 'reason': 'Candlestick detection failed', 'confidence': 0.0}
    
    def _check_price_axis(self, img: np.ndarray) -> dict:
        """Check for price axis (numbers on right side)"""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            
            # Check right side for price labels (common in trading charts)
            right_section = gray[:, int(width * 0.85):]
            
            # Look for text-like patterns (price labels)
            # Numbers typically have specific patterns
            edges = cv2.Canny(right_section, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter for text-like contours
            text_contours = 0
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                # Text-like aspect ratio and size
                if 5 < w < 50 and 8 < h < 25 and 0.3 < w/h < 3:
                    text_contours += 1
            
            if text_contours >= 3:  # Multiple price labels found
                confidence = min(0.7, text_contours / 10)
                return {'is_valid': True, 'reason': 'Price axis detected', 'confidence': confidence}
            else:
                return {'is_valid': False, 'reason': 'No price axis found', 'confidence': 0.3}
                
        except:
            return {'is_valid': False, 'reason': 'Price axis check failed', 'confidence': 0.0}
    
    def _check_time_axis(self, img: np.ndarray) -> dict:
        """Check for time axis (bottom of chart)"""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            
            # Check bottom section for time labels
            bottom_section = gray[int(height * 0.85):, :]
            
            # Look for horizontal text patterns
            edges = cv2.Canny(bottom_section, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            time_labels = 0
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                # Time label characteristics
                if 10 < w < 80 and 5 < h < 20:
                    time_labels += 1
            
            if time_labels >= 2:  # Time axis found
                confidence = min(0.6, time_labels / 8)
                return {'is_valid': True, 'reason': 'Time axis detected', 'confidence': confidence}
            else:
                return {'is_valid': False, 'reason': 'No time axis found', 'confidence': 0.3}
                
        except:
            return {'is_valid': False, 'reason': 'Time axis check failed', 'confidence': 0.0}
    
    def _check_chart_colors(self, img: np.ndarray) -> dict:
        """Check for typical chart colors (green/red candles)"""
        try:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Define trading chart color ranges
            green_ranges = [
                ([35, 30, 30], [85, 255, 255]),   # Green candles
                ([40, 40, 40], [80, 255, 255]),   # Bright green
            ]
            
            red_ranges = [
                ([0, 30, 30], [20, 255, 255]),    # Red candles
                ([160, 30, 30], [180, 255, 255]), # Red range 2
            ]
            
            green_pixels = 0
            red_pixels = 0
            
            for lower, upper in green_ranges:
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                green_pixels += cv2.countNonZero(mask)
            
            for lower, upper in red_ranges:
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                red_pixels += cv2.countNonZero(mask)
            
            total_chart_pixels = green_pixels + red_pixels
            total_pixels = img.shape[0] * img.shape[1]
            color_ratio = total_chart_pixels / total_pixels
            
            if color_ratio > 0.05:  # At least 5% chart colors
                confidence = min(0.8, color_ratio * 5)
                return {'is_valid': True, 'reason': 'Chart colors detected', 'confidence': confidence}
            else:
                return {'is_valid': False, 'reason': 'No chart colors found', 'confidence': 0.2}
                
        except:
            return {'is_valid': False, 'reason': 'Color analysis failed', 'confidence': 0.0}
    
    def _reject_common_non_charts(self, img: np.ndarray) -> dict:
        """Reject common non-chart images"""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Check for photos (high texture/complexity)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Photos typically have much higher texture than charts
            if laplacian_var > 2000:  # Very high texture = likely a photo
                return {'is_valid': False, 'reason': 'Appears to be a photo, not a chart', 'confidence': 0.0}
            
            # Check for uniform colors (screenshots of text, solid colors)
            unique_colors = len(np.unique(gray))
            total_pixels = gray.shape[0] * gray.shape[1]
            color_diversity = unique_colors / total_pixels
            
            if color_diversity < 0.01:  # Very low color diversity
                return {'is_valid': False, 'reason': 'Too uniform to be a chart', 'confidence': 0.0}
            
            # Check aspect ratio (charts are typically landscape)
            height, width = gray.shape
            aspect_ratio = width / height
            
            if aspect_ratio < 0.8 or aspect_ratio > 4:  # Too tall or too wide
                return {'is_valid': False, 'reason': 'Unusual aspect ratio for trading chart', 'confidence': 0.2}
            
            return {'is_valid': True, 'reason': 'Passed non-chart rejection tests', 'confidence': 0.7}
            
        except:
            return {'is_valid': False, 'reason': 'Non-chart rejection failed', 'confidence': 0.0}