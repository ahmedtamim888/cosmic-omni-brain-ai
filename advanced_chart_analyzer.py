#!/usr/bin/env python3
"""
🧠 ADVANCED CHART ANALYZER v.Ω.2
OMNI-BRAIN PERCEPTION ENGINE with Computer Vision
Ultra-precise candle detection and pattern analysis
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from PIL import Image
import os # Added missing import for os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedChartAnalyzer:
    """
    🔮 ULTRA-ADVANCED CHART ANALYZER
    
    Combines OCR text validation with computer vision candle detection
    for maximum precision in trading signal generation.
    """
    
    def __init__(self):
        """Initialize the advanced analyzer"""
        self.min_candles_required = 3  # Reduced for mobile compatibility
        self.pattern_confidence_threshold = 0.75
        
        logger.info("🧠 Advanced Chart Analyzer initialized")
    
    def analyze_chart_complete(self, image_path: str) -> Dict:
        """
        🎯 COMPLETE CHART ANALYSIS
        
        Combines visual candle detection with pattern analysis
        for ultra-precise trading signals.
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return {"error": "Could not load image", "is_valid": False}
            
            logger.info("🔍 Starting complete chart analysis...")
            
            # Phase 1: Detect candles using computer vision
            candles = self.detect_candles(image)
            
            if len(candles) < self.min_candles_required:
                logger.warning(f"⚠️ Only {len(candles)} candles detected, minimum {self.min_candles_required} required")
                return {
                    "is_valid": False,
                    "reason": f"Insufficient candles detected ({len(candles)}/{self.min_candles_required})",
                    "candles_found": len(candles),
                    "visual_analysis": False
                }
            
            # Phase 2: Analyze candle patterns
            pattern_analysis = self.analyze_patterns(candles, image)
            
            # Phase 3: Generate trading signal
            signal = self.generate_advanced_signal(candles, pattern_analysis, image)
            
            return {
                "is_valid": True,
                "candles_found": len(candles),
                "visual_analysis": True,
                "pattern_analysis": pattern_analysis,
                "signal": signal,
                "candle_details": candles[-6:] if len(candles) >= 6 else candles,  # Last 6 candles
                "chart_quality": self.assess_chart_quality(candles, image)
            }
            
        except Exception as e:
            logger.error(f"❌ Error in complete chart analysis: {str(e)}")
            return {"error": str(e), "is_valid": False}

    def detect_candles(self, image):
        """
        🔍 OMNI-BRAIN PERCEPTION ENGINE v.Ω.2
        ULTRA-PRECISE candle detection with noise filtering
        """
        
        # Phase 1: Intelligent Chart Area Detection
        chart_area = self.detect_chart_area(image)
        
        # Phase 2: Dynamic Broker Theme Detection
        broker_theme = self.detect_broker_theme(chart_area)
        
        # Phase 3: Precise Candle Detection with Noise Filtering
        candles = self.extract_candles_precise(chart_area, broker_theme)
        
        # Phase 4: Validate and Clean Candles
        validated_candles = self.validate_candle_detection(chart_area, candles)
        
        # Phase 5: Enhanced Analysis
        enhanced_candles = self.analyze_candle_details(chart_area, validated_candles, broker_theme)
        
        logger.info(f"🕯️ Detected {len(enhanced_candles)} valid candles")
        return enhanced_candles

    def detect_chart_area(self, image):
        """📊 INTELLIGENT CHART AREA DETECTION"""
        
        height, width = image.shape[:2]
        
        # Focus on the main chart area (exclude UI elements)
        # Typically the chart is in the center, excluding top/bottom UI
        margin_top = int(height * 0.15)    # Skip top 15% (UI)
        margin_bottom = int(height * 0.25) # Skip bottom 25% (trading buttons)
        margin_left = int(width * 0.05)    # Skip left 5%
        margin_right = int(width * 0.05)   # Skip right 5%
        
        # Extract chart area
        chart_area = image[margin_top:height-margin_bottom, margin_left:width-margin_right]
        
        return chart_area

    def detect_broker_theme(self, image):
        """🎨 ENHANCED BROKER THEME DETECTION"""
        
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Sample center areas to detect theme
        height, width = image.shape[:2]
        center_sample = image[height//3:2*height//3, width//4:3*width//4]
        
        # Analyze background color
        avg_bg = np.mean(center_sample, axis=(0, 1))
        brightness = np.mean(avg_bg)
        
        # More precise color ranges for IQCent/Quotex style
        if brightness < 80:  # Dark theme (like your screenshot)
            theme = "DARK_THEME"
            # Green (bullish) - more precise for dark themes
            bull_range = [(50, 120, 80), (80, 255, 255)]
            # Red (bearish) - more precise for dark themes  
            bear_range = [(0, 120, 80), (15, 255, 255)]
        else:
            theme = "LIGHT_THEME"
            bull_range = [(35, 100, 100), (80, 255, 200)]
            bear_range = [(0, 100, 100), (20, 255, 200)]
        
        return {
            "theme_type": theme,
            "background_brightness": brightness,
            "bull_hsv_range": bull_range,
            "bear_hsv_range": bear_range,
            "avg_background": avg_bg
        }

    def extract_candles_precise(self, image, broker_theme):
        """🎯 ULTRA-PRECISE CANDLE EXTRACTION"""
        
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Get theme-specific color ranges
        bull_range = broker_theme["bull_hsv_range"]
        bear_range = broker_theme["bear_hsv_range"]
        
        # Create masks with noise reduction
        bull_mask = cv2.inRange(hsv, np.array(bull_range[0]), np.array(bull_range[1]))
        bear_mask = cv2.inRange(hsv, np.array(bear_range[0]), np.array(bear_range[1]))
        
        # Morphological operations to reduce noise
        kernel = np.ones((3,3), np.uint8)
        bull_mask = cv2.morphologyEx(bull_mask, cv2.MORPH_CLOSE, kernel)
        bear_mask = cv2.morphologyEx(bear_mask, cv2.MORPH_CLOSE, kernel)
        
        # Remove small noise
        bull_mask = cv2.morphologyEx(bull_mask, cv2.MORPH_OPEN, kernel)
        bear_mask = cv2.morphologyEx(bear_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        bull_contours, _ = cv2.findContours(bull_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bear_contours, _ = cv2.findContours(bear_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candles = []
        img_height, img_width = image.shape[:2]
        
        # Process bullish candles with strict filtering
        for contour in bull_contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            
            # Strict candle validation
            if (area > 200 and                    # Minimum size
                w > 8 and h > 15 and              # Minimum dimensions
                h > w and                         # Height > width (candle shape)
                area < img_width * img_height * 0.1 and  # Not too large
                w < img_width * 0.15):            # Reasonable width
                
                candles.append({
                    'x': x, 'y': y, 'width': w, 'height': h,
                    'area': area, 'type': 'bullish',
                    'contour': contour,
                    'aspect_ratio': h/w if w > 0 else 0
                })
        
        # Process bearish candles with strict filtering
        for contour in bear_contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            
            # Strict candle validation
            if (area > 200 and                    # Minimum size
                w > 8 and h > 15 and              # Minimum dimensions
                h > w and                         # Height > width (candle shape)
                area < img_width * img_height * 0.1 and  # Not too large
                w < img_width * 0.15):            # Reasonable width
                
                candles.append({
                    'x': x, 'y': y, 'width': w, 'height': h,
                    'area': area, 'type': 'bearish',
                    'contour': contour,
                    'aspect_ratio': h/w if w > 0 else 0
                })
        
        return candles

    def validate_candle_detection(self, image, candles):
        """✅ VALIDATE AND CLEAN CANDLE DETECTION"""
        
        if not candles:
            return []
        
        # Sort by x-coordinate
        candles.sort(key=lambda c: c['x'])
        
        # Remove overlapping candles (keep the larger one)
        cleaned_candles = []
        for candle in candles:
            is_duplicate = False
            for existing in cleaned_candles:
                # Check for significant overlap
                x_overlap = (candle['x'] < existing['x'] + existing['width'] and 
                            candle['x'] + candle['width'] > existing['x'])
                y_overlap = (candle['y'] < existing['y'] + existing['height'] and 
                            candle['y'] + candle['height'] > existing['y'])
                
                if x_overlap and y_overlap:
                    # Keep the larger candle
                    if candle['area'] > existing['area']:
                        cleaned_candles.remove(existing)
                    else:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                cleaned_candles.append(candle)
        
        # Filter by reasonable spacing (candles shouldn't be too close)
        img_width = image.shape[1]
        min_spacing = img_width // 30  # Minimum spacing between candles
        
        spaced_candles = []
        for candle in cleaned_candles:
            is_too_close = False
            for existing in spaced_candles:
                distance = abs(candle['x'] - existing['x'])
                if distance < min_spacing:
                    is_too_close = True
                    break
            
            if not is_too_close:
                spaced_candles.append(candle)
        
        # Limit to reasonable number of candles (max 25 for a typical chart)
        if len(spaced_candles) > 25:
            # Keep the most recent ones (rightmost)
            spaced_candles = spaced_candles[-25:]
        
        return spaced_candles

    def analyze_candle_details(self, image, candles, broker_theme):
        """🧠 ENHANCED CANDLE ANALYSIS"""
        
        enhanced_candles = []
        
        for candle in candles:
            enhanced_candle = candle.copy()
            
            # Calculate candle center
            center_x = candle['x'] + candle['width'] // 2
            center_y = candle['y'] + candle['height'] // 2
            enhanced_candle['center'] = (center_x, center_y)
            
            # Body and wick analysis
            body_analysis = self.analyze_body_and_wicks(image, candle)
            enhanced_candle.update(body_analysis)
            
            # Volume approximation
            enhanced_candle['volume_estimate'] = candle['area'] / 1000
            
            # Position analysis
            position_analysis = self.analyze_candle_position(image, candle)
            enhanced_candle.update(position_analysis)
            
            # Color analysis
            color_analysis = self.analyze_color_intensity(image, candle, broker_theme)
            enhanced_candle.update(color_analysis)
            
            enhanced_candles.append(enhanced_candle)
        
        return enhanced_candles

    def analyze_body_and_wicks(self, image, candle):
        """🕯️ PRECISE BODY AND WICK ANALYSIS"""
        
        x, y, w, h = candle['x'], candle['y'], candle['width'], candle['height']
        
        # Estimate body vs total ratio
        body_ratio = min(w / h if h > 0 else 0, 1.0)
        
        # Wick detection based on aspect ratio
        aspect_ratio = candle['aspect_ratio']
        
        # More precise wick detection
        has_upper_wick = aspect_ratio > 6  # Very tall candles likely have wicks
        has_lower_wick = aspect_ratio > 6
        
        # Simple classification
        if body_ratio < 0.2 and aspect_ratio > 8:
            candle_structure = 'DOJI'
        elif aspect_ratio > 10 and candle['type'] == 'bearish':
            candle_structure = 'SHOOTING_STAR'
        elif aspect_ratio > 10 and candle['type'] == 'bullish':
            candle_structure = 'HAMMER'
        elif body_ratio > 0.6:
            candle_structure = 'MARUBOZU'
        else:
            candle_structure = 'STANDARD'
        
        return {
            'body_ratio': body_ratio,
            'has_upper_wick': has_upper_wick,
            'has_lower_wick': has_lower_wick,
            'candle_structure': candle_structure,
            'body_size': int(h * body_ratio),
            'total_range': h
        }

    def analyze_candle_position(self, image, candle):
        """📍 CANDLE POSITION ANALYSIS"""
        
        img_height, img_width = image.shape[:2]
        
        rel_x = candle['x'] / img_width
        rel_y = candle['y'] / img_height
        
        # Vertical zones
        if rel_y < 0.25:
            vertical_zone = 'RESISTANCE_ZONE'
        elif rel_y > 0.75:
            vertical_zone = 'SUPPORT_ZONE'
        else:
            vertical_zone = 'MIDDLE_ZONE'
        
        # Time position
        if rel_x > 0.8:
            time_position = 'RECENT'
        elif rel_x > 0.6:
            time_position = 'NEAR_RECENT'
        else:
            time_position = 'HISTORICAL'
        
        return {
            'relative_x': rel_x,
            'relative_y': rel_y,
            'vertical_zone': vertical_zone,
            'time_position': time_position
        }

    def analyze_color_intensity(self, image, candle, broker_theme):
        """🎨 COLOR INTENSITY ANALYSIS"""
        
        x, y, w, h = candle['x'], candle['y'], candle['width'], candle['height']
        
        # Safe region extraction
        y_end = min(y + h, image.shape[0])
        x_end = min(x + w, image.shape[1])
        candle_region = image[y:y_end, x:x_end]
        
        if candle_region.size > 0:
            avg_color = np.mean(candle_region, axis=(0, 1))
            hsv_region = cv2.cvtColor(candle_region, cv2.COLOR_BGR2HSV)
            saturation = np.mean(hsv_region[:, :, 1])
            
            if saturation > 150:
                color_strength = 'STRONG'
            elif saturation > 100:
                color_strength = 'MEDIUM'
            else:
                color_strength = 'WEAK'
        else:
            avg_color = [0, 0, 0]
            saturation = 0
            color_strength = 'WEAK'
        
        return {
            'avg_color_bgr': avg_color.tolist() if hasattr(avg_color, 'tolist') else [0, 0, 0],
            'color_saturation': float(saturation),
            'color_strength': color_strength,
            'brightness': float(np.mean(avg_color))
        }

    def analyze_patterns(self, candles: List[Dict], image) -> Dict:
        """
        🧩 ADVANCED PATTERN ANALYSIS
        
        Analyzes candle patterns for trading signals
        """
        if len(candles) < 3:
            return {"pattern": "INSUFFICIENT_DATA", "confidence": 0.0}
        
        # Get last 6 candles for pattern analysis
        recent_candles = candles[-6:] if len(candles) >= 6 else candles
        
        # Trend analysis
        trend = self.analyze_trend(recent_candles)
        
        # Support/Resistance analysis
        sr_analysis = self.analyze_support_resistance(recent_candles, image)
        
        # Pattern detection
        pattern = self.detect_candlestick_patterns(recent_candles)
        
        # Momentum analysis
        momentum = self.analyze_momentum(recent_candles)
        
        return {
            "trend": trend,
            "support_resistance": sr_analysis,
            "pattern": pattern,
            "momentum": momentum,
            "recent_candles_count": len(recent_candles)
        }

    def analyze_trend(self, candles: List[Dict]) -> Dict:
        """📈 TREND ANALYSIS"""
        if len(candles) < 3:
            return {"direction": "NEUTRAL", "strength": 0.0}
        
        # Simple trend based on candle positions
        positions = [candle['center'][1] for candle in candles]  # Y positions
        
        # Calculate trend
        if len(positions) >= 3:
            trend_slope = (positions[-1] - positions[0]) / len(positions)
            
            if trend_slope < -5:  # Moving up (Y decreases = price increases)
                direction = "BULLISH"
                strength = min(abs(trend_slope) / 20, 1.0)
            elif trend_slope > 5:   # Moving down (Y increases = price decreases)
                direction = "BEARISH"
                strength = min(abs(trend_slope) / 20, 1.0)
            else:
                direction = "NEUTRAL"
                strength = 0.0
        else:
            direction = "NEUTRAL"
            strength = 0.0
        
        return {
            "direction": direction,
            "strength": strength,
            "slope": trend_slope if 'trend_slope' in locals() else 0
        }

    def analyze_support_resistance(self, candles: List[Dict], image) -> Dict:
        """📊 SUPPORT/RESISTANCE ANALYSIS"""
        if len(candles) < 3:
            return {"near_support": False, "near_resistance": False}
        
        img_height = image.shape[0]
        
        # Get Y positions (price levels)
        y_positions = [candle['center'][1] for candle in candles]
        
        # Recent candle position
        recent_y = y_positions[-1]
        
        # Simple S/R detection based on chart zones
        relative_position = recent_y / img_height
        
        near_resistance = relative_position < 0.3  # Top 30% of chart
        near_support = relative_position > 0.7     # Bottom 30% of chart
        
        return {
            "near_support": near_support,
            "near_resistance": near_resistance,
            "relative_position": relative_position,
            "price_zone": "RESISTANCE" if near_resistance else "SUPPORT" if near_support else "MIDDLE"
        }

    def detect_candlestick_patterns(self, candles: List[Dict]) -> Dict:
        """🕯️ CANDLESTICK PATTERN DETECTION"""
        if len(candles) < 2:
            return {"pattern_name": "NONE", "confidence": 0.0}
        
        last_candle = candles[-1]
        prev_candle = candles[-2] if len(candles) >= 2 else None
        
        # Pattern detection
        if last_candle['candle_structure'] == 'DOJI':
            return {"pattern_name": "DOJI", "confidence": 0.8, "signal": "REVERSAL_POSSIBLE"}
        
        elif last_candle['candle_structure'] == 'HAMMER' and last_candle['vertical_zone'] == 'SUPPORT_ZONE':
            return {"pattern_name": "HAMMER_AT_SUPPORT", "confidence": 0.9, "signal": "BULLISH"}
        
        elif last_candle['candle_structure'] == 'SHOOTING_STAR' and last_candle['vertical_zone'] == 'RESISTANCE_ZONE':
            return {"pattern_name": "SHOOTING_STAR_AT_RESISTANCE", "confidence": 0.9, "signal": "BEARISH"}
        
        # Engulfing patterns
        elif (prev_candle and 
              last_candle['type'] == 'bullish' and prev_candle['type'] == 'bearish' and
              last_candle['area'] > prev_candle['area'] * 1.2):
            return {"pattern_name": "BULLISH_ENGULFING", "confidence": 0.85, "signal": "BULLISH"}
        
        elif (prev_candle and 
              last_candle['type'] == 'bearish' and prev_candle['type'] == 'bullish' and
              last_candle['area'] > prev_candle['area'] * 1.2):
            return {"pattern_name": "BEARISH_ENGULFING", "confidence": 0.85, "signal": "BEARISH"}
        
        else:
            return {"pattern_name": "STANDARD", "confidence": 0.5, "signal": "NEUTRAL"}

    def analyze_momentum(self, candles: List[Dict]) -> Dict:
        """⚡ MOMENTUM ANALYSIS"""
        if len(candles) < 3:
            return {"momentum": "NEUTRAL", "strength": 0.0}
        
        # Analyze recent 3 candles
        recent_3 = candles[-3:]
        
        bullish_count = sum(1 for c in recent_3 if c['type'] == 'bullish')
        bearish_count = sum(1 for c in recent_3 if c['type'] == 'bearish')
        
        # Volume-weighted momentum
        total_bull_volume = sum(c['volume_estimate'] for c in recent_3 if c['type'] == 'bullish')
        total_bear_volume = sum(c['volume_estimate'] for c in recent_3 if c['type'] == 'bearish')
        
        if bullish_count >= 2 and total_bull_volume > total_bear_volume:
            momentum = "BULLISH"
            strength = min((bullish_count / 3) * (total_bull_volume / (total_bull_volume + total_bear_volume + 1)), 1.0)
        elif bearish_count >= 2 and total_bear_volume > total_bull_volume:
            momentum = "BEARISH"
            strength = min((bearish_count / 3) * (total_bear_volume / (total_bull_volume + total_bear_volume + 1)), 1.0)
        else:
            momentum = "NEUTRAL"
            strength = 0.0
        
        return {
            "momentum": momentum,
            "strength": strength,
            "bullish_candles": bullish_count,
            "bearish_candles": bearish_count
        }

    def generate_advanced_signal(self, candles: List[Dict], pattern_analysis: Dict, image) -> Dict:
        """
        🎯 ULTRA-ADVANCED SIGNAL GENERATION
        
        Combines all analysis for final trading signal
        """
        
        # Get components
        trend = pattern_analysis.get("trend", {})
        sr_analysis = pattern_analysis.get("support_resistance", {})
        pattern = pattern_analysis.get("pattern", {})
        momentum = pattern_analysis.get("momentum", {})
        
        # Calculate signal strength
        signal_strength = 0.0
        signal_direction = "NEUTRAL"
        reasoning = []
        
        # Trend component (30% weight)
        if trend.get("direction") == "BULLISH":
            signal_strength += trend.get("strength", 0) * 0.3
            reasoning.append(f"Bullish trend (strength: {trend.get('strength', 0):.2f})")
        elif trend.get("direction") == "BEARISH":
            signal_strength -= trend.get("strength", 0) * 0.3
            reasoning.append(f"Bearish trend (strength: {trend.get('strength', 0):.2f})")
        
        # Pattern component (40% weight)
        pattern_signal = pattern.get("signal", "NEUTRAL")
        pattern_confidence = pattern.get("confidence", 0)
        
        if pattern_signal == "BULLISH":
            signal_strength += pattern_confidence * 0.4
            reasoning.append(f"{pattern.get('pattern_name', 'Unknown')} pattern (bullish)")
        elif pattern_signal == "BEARISH":
            signal_strength -= pattern_confidence * 0.4
            reasoning.append(f"{pattern.get('pattern_name', 'Unknown')} pattern (bearish)")
        
        # Momentum component (20% weight)
        if momentum.get("momentum") == "BULLISH":
            signal_strength += momentum.get("strength", 0) * 0.2
            reasoning.append(f"Bullish momentum ({momentum.get('bullish_candles', 0)}/3 candles)")
        elif momentum.get("momentum") == "BEARISH":
            signal_strength -= momentum.get("strength", 0) * 0.2
            reasoning.append(f"Bearish momentum ({momentum.get('bearish_candles', 0)}/3 candles)")
        
        # Support/Resistance component (10% weight)
        if sr_analysis.get("near_support") and signal_strength > 0:
            signal_strength += 0.1
            reasoning.append("Near support level (bullish)")
        elif sr_analysis.get("near_resistance") and signal_strength < 0:
            signal_strength -= 0.1
            reasoning.append("Near resistance level (bearish)")
        
        # Determine final signal
        confidence = abs(signal_strength)
        
        if signal_strength > 0.6:
            signal_direction = "CALL"
            signal_emoji = "📈"
        elif signal_strength < -0.6:
            signal_direction = "PUT"
            signal_emoji = "📉"
        else:
            signal_direction = "NO_TRADE"
            signal_emoji = "⚠️"
            reasoning.append("Insufficient confidence for trade")
        
        return {
            "direction": signal_direction,
            "confidence": min(confidence, 1.0),
            "emoji": signal_emoji,
            "reasoning": reasoning,
            "raw_strength": signal_strength,
            "candles_analyzed": len(candles),
            "pattern_detected": pattern.get("pattern_name", "NONE")
        }

    def assess_chart_quality(self, candles: List[Dict], image) -> Dict:
        """📊 ASSESS CHART QUALITY"""
        
        quality_score = 0.0
        
        # Number of candles (more is better)
        candle_score = min(len(candles) / 15, 1.0)  # Max score at 15+ candles
        quality_score += candle_score * 0.4
        
        # Candle clarity (based on color strength)
        clarity_scores = [c.get('color_strength', 'WEAK') for c in candles]
        strong_candles = clarity_scores.count('STRONG')
        clarity_score = strong_candles / len(candles) if candles else 0
        quality_score += clarity_score * 0.3
        
        # Candle spacing consistency
        if len(candles) >= 3:
            spacings = []
            for i in range(1, len(candles)):
                spacing = candles[i]['x'] - candles[i-1]['x']
                spacings.append(spacing)
            
            avg_spacing = np.mean(spacings)
            spacing_variance = np.var(spacings) if len(spacings) > 1 else 0
            spacing_score = max(0, 1 - (spacing_variance / (avg_spacing ** 2)) if avg_spacing > 0 else 0)
            quality_score += spacing_score * 0.3
        
        # Overall quality assessment
        if quality_score > 0.8:
            quality_level = "EXCELLENT"
        elif quality_score > 0.6:
            quality_level = "GOOD"
        elif quality_score > 0.4:
            quality_level = "FAIR"
        else:
            quality_level = "POOR"
        
        return {
            "quality_score": quality_score,
            "quality_level": quality_level,
            "candle_count": len(candles),
            "clarity_ratio": clarity_score,
            "spacing_consistency": spacing_score if 'spacing_score' in locals() else 0.0
        }

# Test function
def test_advanced_analyzer():
    """🧪 Test the advanced analyzer"""
    analyzer = AdvancedChartAnalyzer()
    
    # Test with a sample image
    test_image = "test_chart.png"
    if os.path.exists(test_image):
        result = analyzer.analyze_chart_complete(test_image)
        print("🔍 Advanced Analysis Result:")
        print(f"Valid: {result.get('is_valid', False)}")
        print(f"Candles Found: {result.get('candles_found', 0)}")
        if result.get('signal'):
            signal = result['signal']
            print(f"Signal: {signal['emoji']} {signal['direction']} (Confidence: {signal['confidence']:.2f})")
    else:
        print("❌ Test image not found")

if __name__ == "__main__":
    test_advanced_analyzer()