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

class CandleDetector:
    """Detects and extracts candlestick data from chart images"""
    
    def __init__(self):
        self.min_candle_height = 10
        self.min_candle_width = 3
        
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess the chart image for candlestick detection"""
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not load image")
            
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        return img, gray, thresh
    
    def detect_candlesticks(self, image_path: str) -> List[Candle]:
        """Detect candlesticks from chart image"""
        img, gray, thresh = self.preprocess_image(image_path)
        height, width = img.shape[:2]
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and analyze potential candlestick regions
        candles = []
        
        # Sort contours by x-coordinate (left to right)
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
        
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size constraints
            if w >= self.min_candle_width and h >= self.min_candle_height:
                # Extract candle region
                candle_region = img[y:y+h, x:x+w]
                candle = self._analyze_candle_region(candle_region, x, y, w, h, i)
                if candle:
                    candles.append(candle)
        
        # Return last 8 candles (most recent)
        return candles[-Config.MAX_CANDLES_ANALYZED:] if len(candles) > Config.MAX_CANDLES_ANALYZED else candles
    
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
        """Breakout continuation strategy"""
        confidence = 0
        signal = "NO_TRADE"
        reasoning = ""
        
        # Check for breakout patterns
        breakout_patterns = [p for p in market_story['patterns'] if 'BREAKOUT' in p['name']]
        
        if breakout_patterns:
            breakout = breakout_patterns[0]
            momentum = market_story['momentum']
            
            # Upward breakout with bullish momentum
            if 'UPWARD' in breakout.get('type', '') and momentum['direction'] == 'BULLISH':
                confidence = 75 + (breakout['strength'] * 15) + (momentum['strength'] * 10)
                signal = "CALL"
                reasoning = "Upward breakout with strong bullish momentum continuation"
            
            # Downward breakout with bearish momentum
            elif 'DOWNWARD' in breakout.get('type', '') and momentum['direction'] == 'BEARISH':
                confidence = 75 + (breakout['strength'] * 15) + (momentum['strength'] * 10)
                signal = "PUT"
                reasoning = "Downward breakout with strong bearish momentum continuation"
        
        return {
            'signal': signal,
            'confidence': min(confidence, 95),
            'reasoning': reasoning
        }
    
    def _reversal_play_strategy(self, market_story: Dict, candles: List[Candle]) -> Dict:
        """Reversal play strategy"""
        confidence = 0
        signal = "NO_TRADE"
        reasoning = ""
        
        # Look for reversal patterns
        reversal_patterns = [p for p in market_story['patterns'] 
                           if any(x in p['name'] for x in ['HAMMER', 'SHOOTING_STAR', 'ENGULFING', 'DOJI'])]
        
        if reversal_patterns:
            pattern = reversal_patterns[0]
            sr_analysis = market_story['support_resistance']
            
            # Bullish reversal at support
            if (pattern['name'] in ['HAMMER', 'BULLISH_ENGULFING'] and 
                sr_analysis['near_support']):
                confidence = 80 + (pattern['strength'] * 15)
                signal = "CALL"
                reasoning = f"Bullish reversal pattern ({pattern['name']}) at support level"
            
            # Bearish reversal at resistance
            elif (pattern['name'] in ['SHOOTING_STAR', 'BEARISH_ENGULFING'] and 
                  sr_analysis['near_resistance']):
                confidence = 80 + (pattern['strength'] * 15)
                signal = "PUT"
                reasoning = f"Bearish reversal pattern ({pattern['name']}) at resistance level"
            
            # Doji indecision reversal
            elif pattern['name'] == 'DOJI':
                trend = market_story['trend_direction']
                if trend == 'UPTREND':
                    confidence = 70 + (pattern['strength'] * 10)
                    signal = "PUT"
                    reasoning = "Doji indecision after uptrend, bearish reversal expected"
                elif trend == 'DOWNTREND':
                    confidence = 70 + (pattern['strength'] * 10)
                    signal = "CALL"
                    reasoning = "Doji indecision after downtrend, bullish reversal expected"
        
        return {
            'signal': signal,
            'confidence': min(confidence, 95),
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
            return MarketSignal(
                signal=strategy_details['signal'],
                confidence=strategy_details['confidence'],
                reasoning=strategy_details['reasoning'],
                strategy=best_strategy,
                market_psychology=market_story['market_psychology'],
                entry_time=datetime.now(),
                timeframe=Config.SIGNAL_TIMEFRAME
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