"""
Simple but effective chart validator
Focuses on rejecting obvious non-charts while being permissive to real charts
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class SimpleChartValidator:
    """Simple validator that rejects obvious non-charts"""
    
    def validate_chart_image(self, image_path: str) -> dict:
        """
        Simple validation that rejects obvious non-charts
        Returns: {'is_valid': bool, 'reason': str}
        """
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                return {'is_valid': False, 'reason': 'Cannot load image'}
            
            height, width = img.shape[:2]
            
            # Basic size check
            if width < 300 or height < 200:
                return {'is_valid': False, 'reason': 'Image too small for chart analysis'}
            
            # Check if it's obviously not a chart
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 1. Check for photos (very high texture/complexity)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var > 8000:  # Very high texture = likely a photo
                return {'is_valid': False, 'reason': 'Appears to be a photo, not a trading chart'}
            
            # 2. Check for solid colors or very uniform images (be very permissive)
            unique_colors = len(np.unique(gray))
            total_pixels = gray.shape[0] * gray.shape[1]
            color_diversity = unique_colors / total_pixels
            
            if color_diversity < 0.0001:  # Only reject truly solid colors
                return {'is_valid': False, 'reason': 'Image appears to be solid color'}
            
            # 3. Check aspect ratio (charts are typically landscape)
            aspect_ratio = width / height
            if aspect_ratio < 0.5 or aspect_ratio > 5:  # Too tall or too wide
                return {'is_valid': False, 'reason': 'Unusual aspect ratio for trading chart'}
            
            # 4. Check for some trading-like colors (green/red)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Look for any green or red pixels (common in trading charts)
            green_mask = cv2.inRange(hsv, np.array([35, 30, 30]), np.array([85, 255, 255]))
            red_mask1 = cv2.inRange(hsv, np.array([0, 30, 30]), np.array([20, 255, 255]))
            red_mask2 = cv2.inRange(hsv, np.array([160, 30, 30]), np.array([180, 255, 255]))
            
            green_pixels = cv2.countNonZero(green_mask)
            red_pixels = cv2.countNonZero(red_mask1) + cv2.countNonZero(red_mask2)
            
            total_colored_pixels = green_pixels + red_pixels
            
            # If no trading colors AND very high texture, probably not a chart
            if total_colored_pixels < (total_pixels * 0.005) and laplacian_var > 4000:
                return {'is_valid': False, 'reason': 'No trading chart characteristics detected'}
            
            # If we get here, it passes basic checks - allow it
            return {'is_valid': True, 'reason': 'Image appears to be a potential trading chart'}
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return {'is_valid': False, 'reason': f'Validation error: {str(e)}'}