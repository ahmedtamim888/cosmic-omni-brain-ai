"""
Balanced Chart Validator
- ACCEPTS: Real trading charts (even if imperfect)
- REJECTS: Obvious non-charts (photos, random images, etc.)
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class BalancedChartValidator:
    """Balanced validator that accepts charts while rejecting obvious non-charts"""
    
    def validate_chart_image(self, image_path: str) -> dict:
        """
        Balanced validation - accept unless obviously not a chart
        Returns: {'is_valid': bool, 'reason': str}
        """
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                return {'is_valid': False, 'reason': 'Cannot load image file'}
            
            height, width = img.shape[:2]
            
            # Basic size check - very lenient
            if width < 200 or height < 150:
                return {'is_valid': False, 'reason': 'Image too small for analysis'}
            
            # Convert for analysis
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Calculate metrics
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # REJECTION CRITERIA - Only reject obvious non-charts
            rejection_reasons = []
            
            # 1. Extremely high texture (complex photos)
            if laplacian_var > 8000:
                rejection_reasons.append("extremely high texture complexity (likely a detailed photo)")
            
            # 2. Check for obvious skin tones (people photos)
            skin_lower = np.array([0, 40, 80])
            skin_upper = np.array([25, 255, 255])
            skin_mask = cv2.inRange(hsv, skin_lower, skin_upper)
            skin_ratio = cv2.countNonZero(skin_mask) / (width * height)
            
            if skin_ratio > 0.15:  # 15% skin tones
                rejection_reasons.append("contains significant skin tones (likely a photo of people)")
            
            # 3. Very uniform/solid colors
            unique_colors = len(np.unique(gray))
            if unique_colors < 3:
                rejection_reasons.append("too uniform (appears to be solid color)")
            
            # 4. Check for extremely unnatural color distributions
            # Most trading charts have some dark areas (background) and some bright areas (text/lines)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            
            # If >80% of pixels are in a very narrow range, it's likely not a chart
            total_pixels = width * height
            max_concentration = 0
            for i in range(0, 256-10, 10):  # Check 10-pixel windows
                concentration = np.sum(hist[i:i+10]) / total_pixels
                max_concentration = max(max_concentration, concentration)
            
            if max_concentration > 0.8:
                rejection_reasons.append("color distribution too concentrated (not typical of charts)")
            
            # DECISION: Only reject if we have strong evidence it's NOT a chart
            if len(rejection_reasons) >= 2:  # Need multiple red flags
                return {
                    'is_valid': False,
                    'reason': f"Not a trading chart: {rejection_reasons[0]}"
                }
            
            # If we get here, accept it as potentially valid
            # This gives the benefit of doubt to real charts
            return {
                'is_valid': True,
                'reason': 'Image appears to be acceptable for chart analysis'
            }
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            # If validation fails, be permissive and allow analysis
            return {
                'is_valid': True,
                'reason': 'Validation inconclusive - allowing analysis'
            }