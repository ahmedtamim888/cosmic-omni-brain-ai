import cv2
import numpy as np
from PIL import Image
import io
import base64
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from config import Config

class CandleData:
    def __init__(self, open_price, high_price, low_price, close_price, body_size, wick_ratio, color):
        self.open = open_price
        self.high = high_price
        self.low = low_price
        self.close = close_price
        self.body_size = body_size
        self.wick_ratio = wick_ratio
        self.color = color  # 'green' for bullish, 'red' for bearish
        self.is_bullish = color == 'green'
        self.is_bearish = color == 'red'

class CosmicAIEngine:
    def __init__(self):
        self.confidence_threshold = Config.CONFIDENCE_THRESHOLD
        self.min_candles = Config.MIN_CANDLES_REQUIRED
        self.max_candles = Config.MAX_CANDLES_ANALYZE
        
    def analyze_chart(self, image_data: bytes) -> Dict:
        """Main analysis function that processes chart image and returns signal"""
        try:
            # Convert image data to OpenCV format
            image = self._bytes_to_cv2(image_data)
            
            # Detect and extract candles
            candles = self._detect_candles(image)
            
            if len(candles) < self.min_candles:
                return {
                    'signal': 'NO TRADE',
                    'reason': f'Insufficient candles detected ({len(candles)} < {self.min_candles})',
                    'confidence': 0.0,
                    'timestamp': datetime.now().isoformat()
                }
            
            # Analyze market perception and context
            market_story = self._analyze_market_perception(candles)
            
            # Generate strategy based on market story
            strategy_result = self._generate_strategy(market_story, candles)
            
            return strategy_result
            
        except Exception as e:
            return {
                'signal': 'NO TRADE',
                'reason': f'Analysis error: {str(e)}',
                'confidence': 0.0,
                'timestamp': datetime.now().isoformat()
            }
    
    def _bytes_to_cv2(self, image_data: bytes) -> np.ndarray:
        """Convert bytes to OpenCV image format"""
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image
    
    def _detect_candles(self, image: np.ndarray) -> List[CandleData]:
        """Detect and extract candlestick data from chart image"""
        candles = []
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        height, width = image.shape[:2]
        
        # Define color ranges for green and red candles
        green_lower = np.array([40, 40, 40])
        green_upper = np.array([80, 255, 255])
        red_lower = np.array([0, 40, 40])
        red_upper = np.array([20, 255, 255])
        
        # Create masks for green and red areas
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        red_mask = cv2.inRange(hsv, red_lower, red_upper)
        
        # Find contours for candle bodies
        green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process green (bullish) candles
        for contour in green_contours:
            candle = self._extract_candle_data(contour, image, 'green')
            if candle:
                candles.append(candle)
        
        # Process red (bearish) candles
        for contour in red_contours:
            candle = self._extract_candle_data(contour, image, 'red')
            if candle:
                candles.append(candle)
        
        # Sort candles by x-coordinate (time order)
        candles.sort(key=lambda c: getattr(c, 'x_position', 0))
        
        # Return last N candles for analysis
        return candles[-self.max_candles:] if len(candles) > self.max_candles else candles
    
    def _extract_candle_data(self, contour, image: np.ndarray, color: str) -> Optional[CandleData]:
        """Extract candle data from contour"""
        try:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter out noise (too small rectangles)
            if w < 3 or h < 5:
                return None
            
            # Calculate candle properties
            body_size = h / image.shape[0]  # Relative to image height
            wick_ratio = self._calculate_wick_ratio(x, y, w, h, image)
            
            # Simulate OHLC data based on visual analysis
            if color == 'green':  # Bullish candle
                close_price = y  # Top of body
                open_price = y + h  # Bottom of body
            else:  # Bearish candle
                open_price = y  # Top of body
                close_price = y + h  # Bottom of body
            
            # Estimate high/low including wicks
            high_price = y - (h * wick_ratio / 2)
            low_price = y + h + (h * wick_ratio / 2)
            
            candle = CandleData(
                open_price=open_price,
                high_price=high_price,
                low_price=low_price,
                close_price=close_price,
                body_size=body_size,
                wick_ratio=wick_ratio,
                color=color
            )
            
            # Store x position for sorting
            candle.x_position = x
            
            return candle
            
        except Exception:
            return None
    
    def _calculate_wick_ratio(self, x: int, y: int, w: int, h: int, image: np.ndarray) -> float:
        """Calculate the wick to body ratio"""
        # Simple estimation based on surrounding pixels
        try:
            # Check pixels above and below the body for wicks
            above_region = image[max(0, y-10):y, x:x+w]
            below_region = image[y+h:min(image.shape[0], y+h+10), x:x+w]
            
            # Count non-background pixels as wick indicators
            above_pixels = np.sum(above_region > 50) if above_region.size > 0 else 0
            below_pixels = np.sum(below_region > 50) if below_region.size > 0 else 0
            
            wick_length = (above_pixels + below_pixels) / max(1, w * h)
            return min(wick_length, 2.0)  # Cap at reasonable ratio
            
        except Exception:
            return 0.1  # Default small wick ratio
    
    def _analyze_market_perception(self, candles: List[CandleData]) -> Dict:
        """Analyze market psychology and story from candle patterns"""
        if len(candles) < 3:
            return {'story': 'insufficient_data', 'patterns': []}
        
        patterns = []
        market_story = {
            'momentum': self._analyze_momentum(candles),
            'trend': self._analyze_trend(candles),
            'support_resistance': self._analyze_support_resistance(candles),
            'patterns': patterns,
            'psychology': self._analyze_market_psychology(candles)
        }
        
        # Detect specific patterns
        if self._is_engulfing_pattern(candles[-2:]):
            patterns.append('engulfing')
        
        if self._is_doji_pattern(candles[-1]):
            patterns.append('doji')
        
        if self._is_hammer_pattern(candles[-1]):
            patterns.append('hammer')
        
        if self._is_breakout_setup(candles):
            patterns.append('breakout')
        
        if self._is_trap_zone(candles):
            patterns.append('trap_zone')
        
        market_story['patterns'] = patterns
        return market_story
    
    def _analyze_momentum(self, candles: List[CandleData]) -> Dict:
        """Analyze momentum shift and strength"""
        if len(candles) < 3:
            return {'direction': 'neutral', 'strength': 0.0}
        
        recent_candles = candles[-3:]
        bullish_count = sum(1 for c in recent_candles if c.is_bullish)
        bearish_count = sum(1 for c in recent_candles if c.is_bearish)
        
        # Calculate body size momentum
        avg_body_size = np.mean([c.body_size for c in recent_candles])
        
        if bullish_count > bearish_count:
            direction = 'bullish'
            strength = (bullish_count / len(recent_candles)) * avg_body_size * 100
        elif bearish_count > bullish_count:
            direction = 'bearish'
            strength = (bearish_count / len(recent_candles)) * avg_body_size * 100
        else:
            direction = 'neutral'
            strength = 0.0
        
        return {
            'direction': direction,
            'strength': min(strength, 100.0),
            'consistency': abs(bullish_count - bearish_count) / len(recent_candles)
        }
    
    def _analyze_trend(self, candles: List[CandleData]) -> Dict:
        """Analyze overall trend direction"""
        if len(candles) < 4:
            return {'direction': 'sideways', 'strength': 0.0}
        
        # Compare recent highs and lows
        first_half = candles[:len(candles)//2]
        second_half = candles[len(candles)//2:]
        
        first_avg = np.mean([(c.high + c.low) / 2 for c in first_half])
        second_avg = np.mean([(c.high + c.low) / 2 for c in second_half])
        
        price_change = (second_avg - first_avg) / first_avg if first_avg != 0 else 0
        
        if price_change > 0.005:  # 0.5% threshold
            return {'direction': 'uptrend', 'strength': min(abs(price_change) * 100, 100)}
        elif price_change < -0.005:
            return {'direction': 'downtrend', 'strength': min(abs(price_change) * 100, 100)}
        else:
            return {'direction': 'sideways', 'strength': 0}
    
    def _analyze_support_resistance(self, candles: List[CandleData]) -> Dict:
        """Identify support and resistance levels"""
        if len(candles) < 4:
            return {'levels': [], 'current_position': 'middle'}
        
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        
        # Find resistance (highest highs)
        resistance = max(highs)
        resistance_touches = sum(1 for h in highs if abs(h - resistance) / resistance < 0.01)
        
        # Find support (lowest lows)
        support = min(lows)
        support_touches = sum(1 for l in lows if abs(l - support) / support < 0.01)
        
        # Current price position
        current_price = candles[-1].close
        price_range = resistance - support
        
        if price_range == 0:
            position = 'middle'
        else:
            position_pct = (current_price - support) / price_range
            if position_pct > 0.8:
                position = 'near_resistance'
            elif position_pct < 0.2:
                position = 'near_support'
            else:
                position = 'middle'
        
        return {
            'resistance': resistance,
            'support': support,
            'resistance_strength': resistance_touches,
            'support_strength': support_touches,
            'current_position': position
        }
    
    def _analyze_market_psychology(self, candles: List[CandleData]) -> Dict:
        """Analyze market psychology and sentiment"""
        if len(candles) < 3:
            return {'sentiment': 'neutral', 'fear_greed': 50}
        
        recent = candles[-3:]
        
        # Analyze body sizes (conviction)
        avg_body = np.mean([c.body_size for c in recent])
        conviction = 'high' if avg_body > 0.02 else 'medium' if avg_body > 0.01 else 'low'
        
        # Analyze wick ratios (indecision)
        avg_wick = np.mean([c.wick_ratio for c in recent])
        indecision = 'high' if avg_wick > 1.0 else 'medium' if avg_wick > 0.5 else 'low'
        
        # Sentiment based on recent candles
        bullish_momentum = sum(1 for c in recent if c.is_bullish and c.body_size > 0.015)
        bearish_momentum = sum(1 for c in recent if c.is_bearish and c.body_size > 0.015)
        
        if bullish_momentum > bearish_momentum:
            sentiment = 'bullish'
            fear_greed = min(70 + (bullish_momentum * 10), 100)
        elif bearish_momentum > bullish_momentum:
            sentiment = 'bearish'
            fear_greed = max(30 - (bearish_momentum * 10), 0)
        else:
            sentiment = 'neutral'
            fear_greed = 50
        
        return {
            'sentiment': sentiment,
            'conviction': conviction,
            'indecision': indecision,
            'fear_greed': fear_greed
        }
    
    def _is_engulfing_pattern(self, candles: List[CandleData]) -> bool:
        """Detect engulfing pattern"""
        if len(candles) < 2:
            return False
        
        prev, curr = candles[-2], candles[-1]
        
        # Bullish engulfing
        if prev.is_bearish and curr.is_bullish:
            return curr.body_size > prev.body_size * 1.2
        
        # Bearish engulfing
        if prev.is_bullish and curr.is_bearish:
            return curr.body_size > prev.body_size * 1.2
        
        return False
    
    def _is_doji_pattern(self, candle: CandleData) -> bool:
        """Detect doji pattern (indecision)"""
        return candle.body_size < 0.003 and candle.wick_ratio > 0.8
    
    def _is_hammer_pattern(self, candle: CandleData) -> bool:
        """Detect hammer/hanging man pattern"""
        return candle.wick_ratio > 1.5 and candle.body_size < 0.01
    
    def _is_breakout_setup(self, candles: List[CandleData]) -> bool:
        """Detect potential breakout setup"""
        if len(candles) < 5:
            return False
        
        # Look for consolidation followed by momentum
        middle_candles = candles[-4:-1]
        latest = candles[-1]
        
        # Check if middle candles are consolidating
        avg_body = np.mean([c.body_size for c in middle_candles])
        
        # Check if latest candle shows strong momentum
        return avg_body < 0.008 and latest.body_size > avg_body * 2
    
    def _is_trap_zone(self, candles: List[CandleData]) -> bool:
        """Detect potential trap zones (fakeouts)"""
        if len(candles) < 4:
            return False
        
        # Look for quick reversal patterns
        recent = candles[-3:]
        
        # Check for alternating colors with increasing body sizes
        if len(set(c.color for c in recent)) > 1:
            body_sizes = [c.body_size for c in recent]
            return body_sizes[-1] > body_sizes[0] * 1.5
        
        return False
    
    def _generate_strategy(self, market_story: Dict, candles: List[CandleData]) -> Dict:
        """Generate trading strategy based on market analysis"""
        confidence = 0.0
        signal = 'NO TRADE'
        reasons = []
        
        momentum = market_story['momentum']
        trend = market_story['trend']
        sr_analysis = market_story['support_resistance']
        patterns = market_story['patterns']
        psychology = market_story['psychology']
        
        # Base confidence on pattern strength
        base_confidence = 30.0
        
        # Momentum analysis
        if momentum['strength'] > 70:
            if momentum['direction'] == 'bullish':
                signal = 'CALL'
                confidence += 25
                reasons.append(f"Strong bullish momentum ({momentum['strength']:.1f}%)")
            elif momentum['direction'] == 'bearish':
                signal = 'PUT'
                confidence += 25
                reasons.append(f"Strong bearish momentum ({momentum['strength']:.1f}%)")
        
        # Trend confirmation
        if trend['strength'] > 60:
            if trend['direction'] == 'uptrend' and signal == 'CALL':
                confidence += 15
                reasons.append("Trend confirmation (uptrend)")
            elif trend['direction'] == 'downtrend' and signal == 'PUT':
                confidence += 15
                reasons.append("Trend confirmation (downtrend)")
        
        # Support/Resistance analysis
        if sr_analysis['current_position'] == 'near_support' and signal == 'CALL':
            confidence += 10
            reasons.append("Near support level")
        elif sr_analysis['current_position'] == 'near_resistance' and signal == 'PUT':
            confidence += 10
            reasons.append("Near resistance level")
        
        # Pattern bonuses
        if 'engulfing' in patterns:
            confidence += 20
            reasons.append("Engulfing pattern detected")
        
        if 'breakout' in patterns:
            confidence += 15
            reasons.append("Breakout setup identified")
        
        if 'hammer' in patterns:
            confidence += 10
            reasons.append("Reversal signal (hammer pattern)")
        
        # Psychology confirmation
        if psychology['conviction'] == 'high':
            confidence += 10
            reasons.append("High market conviction")
        
        # Trap zone penalty
        if 'trap_zone' in patterns:
            confidence -= 20
            reasons.append("Potential trap zone detected")
        
        # Doji penalty (indecision)
        if 'doji' in patterns:
            confidence -= 15
            reasons.append("Market indecision (doji)")
        
        # Final confidence calculation
        total_confidence = min(base_confidence + confidence, 100.0)
        
        # Decision logic
        if total_confidence < self.confidence_threshold:
            signal = 'NO TRADE'
            final_reason = f"Insufficient confidence ({total_confidence:.1f}% < {self.confidence_threshold}%)"
        else:
            if signal == 'NO TRADE':
                # If we have confidence but no clear signal, default to momentum
                if momentum['direction'] == 'bullish':
                    signal = 'CALL'
                elif momentum['direction'] == 'bearish':
                    signal = 'PUT'
            
            final_reason = ' + '.join(reasons) if reasons else "Technical analysis convergence"
        
        return {
            'signal': signal,
            'reason': final_reason,
            'confidence': round(total_confidence, 1),
            'timestamp': datetime.now().isoformat(),
            'analysis_details': {
                'momentum': momentum,
                'trend': trend,
                'support_resistance': sr_analysis,
                'patterns': patterns,
                'psychology': psychology
            }
        }