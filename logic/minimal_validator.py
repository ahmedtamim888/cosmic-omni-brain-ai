"""
Minimal Chart Validator - EXTREMELY PERMISSIVE
Only rejects the most obvious non-chart content
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class MinimalChartValidator:
    """Minimal validator - accepts almost everything, rejects only obvious non-charts"""
    
    def validate_chart_image(self, image_path: str) -> dict:
        """
        Minimal validation - accept unless EXTREMELY obviously not a chart
        Returns: {'is_valid': bool, 'reason': str}
        """
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                return {'is_valid': False, 'reason': 'Cannot load image file'}
            
            height, width = img.shape[:2]
            
            # Only check basic size - very minimal
            if width < 100 or height < 100:
                return {'is_valid': False, 'reason': 'Image too small'}
            
            # Convert for analysis
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # ONLY reject EXTREMELY obvious cases
            rejection_count = 0
            
            # 1. Extremely uniform (like solid color)
            unique_colors = len(np.unique(gray))
            if unique_colors < 2:
                rejection_count += 1
            
            # 2. Extremely high skin tone concentration (obvious selfie/person photo)
            skin_lower = np.array([0, 50, 100])
            skin_upper = np.array([20, 255, 255])
            skin_mask = cv2.inRange(hsv, skin_lower, skin_upper)
            skin_ratio = cv2.countNonZero(skin_mask) / (width * height)
            
            if skin_ratio > 0.3:  # 30% skin tones - obvious person photo
                rejection_count += 1
            
            # 3. Extremely high texture complexity (ultra-detailed photo)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var > 20000:  # Extremely high threshold
                rejection_count += 1
            
            # Only reject if multiple EXTREME indicators
            if rejection_count >= 2:
                return {
                    'is_valid': False,
                    'reason': 'Extremely obvious non-chart content detected'
                }
            
            # Accept everything else
            return {
                'is_valid': True,
                'reason': 'Image accepted for analysis'
            }
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            # If validation fails, always accept
            return {
                'is_valid': True,
                'reason': 'Validation error - accepting image'
            }