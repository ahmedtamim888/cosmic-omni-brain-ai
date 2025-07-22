#!/usr/bin/env python3
"""
📊 CHART ANALYZER - Advanced Chart Analysis Utilities
Additional chart analysis functions and helpers
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

class ChartAnalyzer:
    """
    Advanced chart analysis utilities
    """
    
    def __init__(self):
        self.version = "∞ vX"
        
    def analyze_chart_quality(self, image: np.ndarray) -> Dict:
        """Analyze overall chart quality and characteristics"""
        try:
            quality_metrics = {
                'sharpness': self._calculate_sharpness(image),
                'contrast': self._calculate_contrast(image),
                'noise_level': self._calculate_noise_level(image),
                'resolution_quality': self._assess_resolution(image),
                'color_richness': self._assess_color_richness(image),
                'overall_score': 0.0
            }
            
            # Calculate overall quality score
            scores = [
                quality_metrics['sharpness'] * 0.3,
                quality_metrics['contrast'] * 0.25,
                (1.0 - quality_metrics['noise_level']) * 0.2,
                quality_metrics['resolution_quality'] * 0.15,
                quality_metrics['color_richness'] * 0.1
            ]
            
            quality_metrics['overall_score'] = sum(scores)
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Chart quality analysis error: {str(e)}")
            return {'overall_score': 0.5}
    
    def _calculate_sharpness(self, image: np.ndarray) -> float:
        """Calculate image sharpness using Laplacian variance"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            # Normalize to 0-1 range (1000 is a reasonable max for sharp images)
            return min(1.0, laplacian_var / 1000.0)
        except:
            return 0.5
    
    def _calculate_contrast(self, image: np.ndarray) -> float:
        """Calculate image contrast"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            contrast = gray.std() / 255.0
            return min(1.0, contrast * 4)  # Scale to reasonable range
        except:
            return 0.5
    
    def _calculate_noise_level(self, image: np.ndarray) -> float:
        """Calculate noise level in image"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Use median filter to estimate noise
            median_filtered = cv2.medianBlur(gray, 5)
            noise = np.abs(gray.astype(np.float32) - median_filtered.astype(np.float32))
            noise_level = np.mean(noise) / 255.0
            return min(1.0, noise_level * 10)  # Scale to 0-1
        except:
            return 0.3
    
    def _assess_resolution(self, image: np.ndarray) -> float:
        """Assess resolution quality"""
        try:
            height, width = image.shape[:2]
            total_pixels = height * width
            
            # Quality based on resolution
            if total_pixels > 1920 * 1080:
                return 1.0
            elif total_pixels > 1280 * 720:
                return 0.8
            elif total_pixels > 800 * 600:
                return 0.6
            elif total_pixels > 640 * 480:
                return 0.4
            else:
                return 0.2
        except:
            return 0.5
    
    def _assess_color_richness(self, image: np.ndarray) -> float:
        """Assess color richness and diversity"""
        try:
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Calculate color diversity
            unique_colors = len(np.unique(hsv.reshape(-1, 3), axis=0))
            total_possible = image.shape[0] * image.shape[1]
            
            color_richness = min(1.0, unique_colors / (total_possible * 0.1))
            return color_richness
        except:
            return 0.5
    
    def detect_ui_elements(self, image: np.ndarray) -> Dict:
        """Detect broker UI elements"""
        try:
            ui_elements = {
                'has_grid': False,
                'has_indicators': False,
                'has_volume_bars': False,
                'has_price_labels': False,
                'ui_complexity': 0.0
            }
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect grid lines
            ui_elements['has_grid'] = self._detect_grid_lines(gray)
            
            # Detect indicators (colored overlays)
            ui_elements['has_indicators'] = self._detect_indicators(image)
            
            # Detect volume bars
            ui_elements['has_volume_bars'] = self._detect_volume_bars(gray)
            
            # Detect price labels
            ui_elements['has_price_labels'] = self._detect_price_labels(gray)
            
            # Calculate UI complexity
            complexity_score = sum([
                ui_elements['has_grid'],
                ui_elements['has_indicators'],
                ui_elements['has_volume_bars'],
                ui_elements['has_price_labels']
            ]) / 4.0
            
            ui_elements['ui_complexity'] = complexity_score
            
            return ui_elements
            
        except Exception as e:
            logger.error(f"UI element detection error: {str(e)}")
            return {'ui_complexity': 0.5}
    
    def _detect_grid_lines(self, gray_image: np.ndarray) -> bool:
        """Detect grid lines in chart"""
        try:
            # Use HoughLines to detect straight lines
            edges = cv2.Canny(gray_image, 50, 150)
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is not None and len(lines) > 20:
                return True
            return False
        except:
            return False
    
    def _detect_indicators(self, image: np.ndarray) -> bool:
        """Detect technical indicators (colored overlays)"""
        try:
            # Look for colored lines that might be indicators
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Define color ranges for common indicator colors
            color_ranges = [
                (np.array([100, 50, 50]), np.array([130, 255, 255])),  # Blue
                (np.array([160, 50, 50]), np.array([180, 255, 255])),  # Red
                (np.array([40, 50, 50]), np.array([80, 255, 255])),    # Green
                (np.array([20, 50, 50]), np.array([40, 255, 255]))     # Yellow/Orange
            ]
            
            indicator_pixels = 0
            total_pixels = image.shape[0] * image.shape[1]
            
            for lower, upper in color_ranges:
                mask = cv2.inRange(hsv, lower, upper)
                indicator_pixels += np.sum(mask > 0)
            
            # If more than 1% of pixels are indicator colors, assume indicators present
            return (indicator_pixels / total_pixels) > 0.01
            
        except:
            return False
    
    def _detect_volume_bars(self, gray_image: np.ndarray) -> bool:
        """Detect volume bars at bottom of chart"""
        try:
            height, width = gray_image.shape
            bottom_section = gray_image[int(height * 0.8):, :]
            
            # Look for vertical rectangular patterns (volume bars)
            edges = cv2.Canny(bottom_section, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            vertical_rectangles = 0
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if h > w and h > 10:  # More tall than wide
                    vertical_rectangles += 1
            
            return vertical_rectangles > 5
            
        except:
            return False
    
    def _detect_price_labels(self, gray_image: np.ndarray) -> bool:
        """Detect price labels on the right side"""
        try:
            height, width = gray_image.shape
            right_section = gray_image[:, int(width * 0.85):]
            
            # Look for text-like patterns
            # Text typically has specific frequency characteristics
            edges = cv2.Canny(right_section, 30, 100)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            text_like_contours = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if 50 < area < 500:  # Typical text size
                    text_like_contours += 1
            
            return text_like_contours > 3
            
        except:
            return False
    
    def analyze_timeframe_indicators(self, image: np.ndarray) -> Dict:
        """Analyze indicators of timeframe from chart appearance"""
        try:
            timeframe_analysis = {
                'estimated_timeframe': '1M',
                'candle_density': 0.0,
                'chart_span_estimate': 'unknown'
            }
            
            # Analyze candle density
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Count vertical lines (potential candles)
            height, width = edges.shape
            vertical_lines = 0
            
            for x in range(0, width, 10):  # Sample every 10 pixels
                column = edges[:, x]
                if np.sum(column > 0) > height * 0.1:  # If more than 10% pixels are edges
                    vertical_lines += 1
            
            candle_density = vertical_lines / (width / 10)
            timeframe_analysis['candle_density'] = candle_density
            
            # Estimate timeframe based on density
            if candle_density > 0.8:
                timeframe_analysis['estimated_timeframe'] = '1M'
                timeframe_analysis['chart_span_estimate'] = 'short_term'
            elif candle_density > 0.5:
                timeframe_analysis['estimated_timeframe'] = '5M'
                timeframe_analysis['chart_span_estimate'] = 'medium_term'
            elif candle_density > 0.3:
                timeframe_analysis['estimated_timeframe'] = '15M'
                timeframe_analysis['chart_span_estimate'] = 'medium_term'
            else:
                timeframe_analysis['estimated_timeframe'] = '1H+'
                timeframe_analysis['chart_span_estimate'] = 'long_term'
            
            return timeframe_analysis
            
        except Exception as e:
            logger.error(f"Timeframe analysis error: {str(e)}")
            return {'estimated_timeframe': '1M', 'candle_density': 0.5}
    
    def detect_broker_platform(self, image: np.ndarray) -> Dict:
        """Detect broker platform characteristics"""
        try:
            platform_analysis = {
                'platform_type': 'unknown',
                'mobile_indicators': [],
                'desktop_indicators': [],
                'confidence': 0.0
            }
            
            height, width = image.shape[:2]
            aspect_ratio = width / height
            
            # Mobile indicators
            if aspect_ratio < 1.0:  # Portrait orientation
                platform_analysis['mobile_indicators'].append('portrait_orientation')
            
            if width < 800:  # Low resolution width
                platform_analysis['mobile_indicators'].append('low_resolution')
            
            # Desktop indicators
            if aspect_ratio > 1.5:  # Wide aspect ratio
                platform_analysis['desktop_indicators'].append('wide_aspect_ratio')
            
            if width > 1200:  # High resolution
                platform_analysis['desktop_indicators'].append('high_resolution')
            
            # UI element analysis for platform detection
            ui_elements = self.detect_ui_elements(image)
            if ui_elements['ui_complexity'] > 0.6:
                platform_analysis['desktop_indicators'].append('complex_ui')
            else:
                platform_analysis['mobile_indicators'].append('simple_ui')
            
            # Determine platform type
            mobile_score = len(platform_analysis['mobile_indicators'])
            desktop_score = len(platform_analysis['desktop_indicators'])
            
            if mobile_score > desktop_score:
                platform_analysis['platform_type'] = 'mobile'
                platform_analysis['confidence'] = mobile_score / (mobile_score + desktop_score)
            elif desktop_score > mobile_score:
                platform_analysis['platform_type'] = 'desktop'
                platform_analysis['confidence'] = desktop_score / (mobile_score + desktop_score)
            else:
                platform_analysis['platform_type'] = 'unknown'
                platform_analysis['confidence'] = 0.5
            
            return platform_analysis
            
        except Exception as e:
            logger.error(f"Platform detection error: {str(e)}")
            return {'platform_type': 'unknown', 'confidence': 0.0}
    
    def analyze_market_sentiment_visual(self, image: np.ndarray) -> Dict:
        """Analyze visual market sentiment from colors and patterns"""
        try:
            sentiment_analysis = {
                'color_sentiment': 'neutral',
                'pattern_sentiment': 'neutral',
                'overall_sentiment': 'neutral',
                'confidence': 0.0
            }
            
            # Color-based sentiment analysis
            sentiment_analysis['color_sentiment'] = self._analyze_color_sentiment(image)
            
            # Pattern-based sentiment analysis
            sentiment_analysis['pattern_sentiment'] = self._analyze_pattern_sentiment(image)
            
            # Combine sentiments
            color_sent = sentiment_analysis['color_sentiment']
            pattern_sent = sentiment_analysis['pattern_sentiment']
            
            if color_sent == pattern_sent:
                sentiment_analysis['overall_sentiment'] = color_sent
                sentiment_analysis['confidence'] = 0.8
            elif color_sent == 'neutral' or pattern_sent == 'neutral':
                non_neutral = color_sent if color_sent != 'neutral' else pattern_sent
                sentiment_analysis['overall_sentiment'] = non_neutral
                sentiment_analysis['confidence'] = 0.6
            else:
                sentiment_analysis['overall_sentiment'] = 'neutral'
                sentiment_analysis['confidence'] = 0.4
            
            return sentiment_analysis
            
        except Exception as e:
            logger.error(f"Visual sentiment analysis error: {str(e)}")
            return {'overall_sentiment': 'neutral', 'confidence': 0.0}
    
    def _analyze_color_sentiment(self, image: np.ndarray) -> str:
        """Analyze sentiment based on dominant colors"""
        try:
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Define color ranges
            green_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
            red_mask = cv2.inRange(hsv, np.array([0, 40, 40]), np.array([15, 255, 255]))
            red_mask2 = cv2.inRange(hsv, np.array([160, 40, 40]), np.array([180, 255, 255]))
            red_mask = cv2.bitwise_or(red_mask, red_mask2)
            
            green_pixels = np.sum(green_mask > 0)
            red_pixels = np.sum(red_mask > 0)
            
            if green_pixels > red_pixels * 1.5:
                return 'bullish'
            elif red_pixels > green_pixels * 1.5:
                return 'bearish'
            else:
                return 'neutral'
                
        except:
            return 'neutral'
    
    def _analyze_pattern_sentiment(self, image: np.ndarray) -> str:
        """Analyze sentiment based on visual patterns"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            
            # Analyze general trend by looking at line slopes
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=width//4, maxLineGap=10)
            
            if lines is None:
                return 'neutral'
            
            upward_lines = 0
            downward_lines = 0
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(x2 - x1) > 10:  # Ignore vertical lines
                    slope = (y2 - y1) / (x2 - x1)
                    if slope < -0.1:  # Upward trend (y decreases as x increases in image coordinates)
                        upward_lines += 1
                    elif slope > 0.1:  # Downward trend
                        downward_lines += 1
            
            if upward_lines > downward_lines * 1.2:
                return 'bullish'
            elif downward_lines > upward_lines * 1.2:
                return 'bearish'
            else:
                return 'neutral'
                
        except:
            return 'neutral'