"""
Smart chart validator that specifically targets photo rejection
while being permissive to potential trading charts
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class SmartChartValidator:
    """Smart validator that rejects photos but accepts potential charts"""
    
    def validate_chart_image(self, image_path: str) -> dict:
        """
        Validate image by rejecting obvious photos
        Returns: {'is_valid': bool, 'reason': str}
        """
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                return {'is_valid': False, 'reason': 'Cannot load image'}
            
            height, width = img.shape[:2]
            
            # Basic size check
            if width < 200 or height < 150:
                return {'is_valid': False, 'reason': 'Image too small for chart analysis'}
            
            # Convert to different color spaces for analysis
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Calculate texture measures
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Calculate color statistics
            blue_channel = img[:, :, 0]
            green_channel = img[:, :, 1]
            red_channel = img[:, :, 2]
            
            # Photos typically have a wider range of colors and more natural color distribution
            color_std = np.std([np.std(blue_channel), np.std(green_channel), np.std(red_channel)])
            
            # Check for skin tones (common in photos with people)
            skin_lower = np.array([0, 20, 70])
            skin_upper = np.array([20, 255, 255])
            skin_mask = cv2.inRange(hsv, skin_lower, skin_upper)
            skin_pixels = cv2.countNonZero(skin_mask)
            skin_ratio = skin_pixels / (width * height)
            
            # Check for natural/organic edges (typical in photos)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = cv2.countNonZero(edges) / (width * height)
            
            # Rejection criteria (identify photos)
            rejection_score = 0
            rejection_reasons = []
            
            # Very high texture with natural color distribution = likely photo
            if laplacian_var > 3000 and color_std > 40:
                rejection_score += 3
                rejection_reasons.append("High texture complexity typical of photos")
            
            # Significant skin tones = likely photo with people
            if skin_ratio > 0.05:
                rejection_score += 2
                rejection_reasons.append("Contains skin tones typical of photos")
            
            # Very high edge density with high texture = complex photo
            if edge_density > 0.15 and laplacian_var > 2500:
                rejection_score += 2
                rejection_reasons.append("Complex edge patterns typical of photos")
            
            # Ultra-high texture = definitely a photo
            if laplacian_var > 6000:
                rejection_score += 3
                rejection_reasons.append("Extremely high texture indicates photo")
            
            # Check for truly solid/empty images
            unique_colors = len(np.unique(gray))
            if unique_colors < 5:
                rejection_score += 3
                rejection_reasons.append("Image is too uniform/empty")
            
            # Check for low-texture random images (but allow charts which can also be low texture)
            mean_intensity = np.mean(gray)
            std_dev = np.std(gray)
            
            # Random simple images often have poor contrast and few colors
            if unique_colors < 100 and std_dev < 35 and laplacian_var < 1000:
                # But make sure it's not a dark trading chart (which could have similar properties)
                # Trading charts usually have some structure even if low contrast
                edges = cv2.Canny(gray, 30, 100)
                edge_density = cv2.countNonZero(edges) / (width * height)
                
                # If very few edges AND low texture AND few colors = probably random simple image
                if edge_density < 0.02:
                    rejection_score += 2
                    rejection_reasons.append("Low complexity suggests simple/random content")
            
            # Make decision
            if rejection_score >= 3:
                return {
                    'is_valid': False, 
                    'reason': f"Likely a photo: {'; '.join(rejection_reasons[:2])}"
                }
            
            # If we get here, allow it (be permissive for potential charts)
            return {'is_valid': True, 'reason': 'Image passes basic validation for chart analysis'}
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            # If validation fails, be permissive and allow it
            return {'is_valid': True, 'reason': 'Validation inconclusive, allowing analysis'}