#!/usr/bin/env python3
"""
🧠 CONTEXT ENGINE - Candle Memory & Market Understanding
Reads last 5-10 candles like a story
Understands momentum, reversals, trap candles, structure breaks, fakeouts
"""

import logging
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)

class ContextEngine:
    """
    Advanced context analysis engine that understands market dynamics
    """
    
    def __init__(self):
        self.version = "∞ vX"
        self.market_memory = []
        self.volatility_threshold = 0.3
        self.trend_memory_length = 10
        
    async def analyze_context(self, chart_data: Dict) -> Dict:
        """
        Main context analysis - understands the market story
        """
        try:
            candles = chart_data.get('candles', [])
            patterns = chart_data.get('patterns', [])
            properties = chart_data.get('properties', {})
            
            if len(candles) < 3:
                return {"error": "Insufficient candle data for context analysis"}
            
            # Analyze market context
            context = {
                "market_phase": await self._identify_market_phase(candles),
                "momentum": self._analyze_momentum(candles),
                "volatility": self._calculate_volatility(candles),
                "structure": await self._analyze_market_structure(candles),
                "support_resistance": self._identify_support_resistance(candles),
                "sentiment": self._analyze_market_sentiment(candles, patterns),
                "risk_factors": await self._identify_risk_factors(candles, patterns),
                "opportunity_score": self._calculate_opportunity_score(candles, patterns),
                "manipulation_signs": self._detect_manipulation(candles),
                "next_candle_prediction": await self._predict_next_candle_context(candles)
            }
            
            # Store in memory for learning
            self.market_memory.append({
                'timestamp': datetime.now(),
                'context': context,
                'candles_snapshot': candles[-5:]  # Keep last 5 candles
            })
            
            # Limit memory size
            if len(self.market_memory) > 100:
                self.market_memory = self.market_memory[-100:]
            
            logger.info(f"🧠 Context Analysis: Phase={context['market_phase']}, Opportunity={context['opportunity_score']:.2f}")
            return context
            
        except Exception as e:
            logger.error(f"Context Engine error: {str(e)}")
            return {"error": str(e)}
    
    async def _identify_market_phase(self, candles: List[Dict]) -> str:
        """Identify current market phase"""
        try:
            if len(candles) < 5:
                return "insufficient_data"
            
            recent_candles = candles[-10:]
            
            # Calculate price movement metrics
            prices = [c.get('close', 0) for c in recent_candles]
            highs = [c.get('high', 0) for c in recent_candles]
            lows = [c.get('low', 0) for c in recent_candles]
            
            # Trend analysis
            price_change = (prices[-1] - prices[0]) / (prices[0] + 0.001) * 100
            
            # Volatility analysis
            ranges = [(h - l) for h, l in zip(highs, lows)]
            avg_range = np.mean(ranges)
            current_range = ranges[-1]
            
            # Higher high, higher low pattern
            hh_hl = self._check_higher_highs_lows(recent_candles)
            ll_lh = self._check_lower_highs_lows(recent_candles)
            
            # Determine phase
            if price_change > 2 and hh_hl:
                return "strong_uptrend"
            elif price_change > 0.5:
                return "weak_uptrend"
            elif price_change < -2 and ll_lh:
                return "strong_downtrend"
            elif price_change < -0.5:
                return "weak_downtrend"
            elif current_range > avg_range * 1.5:
                return "high_volatility_range"
            elif current_range < avg_range * 0.7:
                return "low_volatility_range"
            else:
                return "sideways_consolidation"
                
        except Exception as e:
            logger.error(f"Market phase identification error: {str(e)}")
            return "unknown"
    
    def _check_higher_highs_lows(self, candles: List[Dict]) -> bool:
        """Check for higher highs and higher lows pattern"""
        if len(candles) < 3:
            return False
        
        highs = [c.get('high', 0) for c in candles[-3:]]
        lows = [c.get('low', 0) for c in candles[-3:]]
        
        return highs[-1] > highs[-2] and lows[-1] > lows[-2]
    
    def _check_lower_highs_lows(self, candles: List[Dict]) -> bool:
        """Check for lower highs and lower lows pattern"""
        if len(candles) < 3:
            return False
        
        highs = [c.get('high', 0) for c in candles[-3:]]
        lows = [c.get('low', 0) for c in candles[-3:]]
        
        return highs[-1] < highs[-2] and lows[-1] < lows[-2]
    
    def _analyze_momentum(self, candles: List[Dict]) -> Dict:
        """Analyze price momentum"""
        try:
            if len(candles) < 5:
                return {"strength": 0, "direction": "neutral"}
            
            recent_candles = candles[-5:]
            
            # Calculate momentum metrics
            bullish_candles = sum(1 for c in recent_candles if c.get('type') == 'bullish')
            bearish_candles = sum(1 for c in recent_candles if c.get('type') == 'bearish')
            
            # Body size momentum
            body_sizes = [c.get('body_size', 0) for c in recent_candles]
            avg_body_size = np.mean(body_sizes)
            last_body_size = body_sizes[-1]
            
            # Calculate momentum strength
            directional_strength = abs(bullish_candles - bearish_candles) / len(recent_candles)
            size_momentum = last_body_size / (avg_body_size + 0.1)
            
            overall_strength = (directional_strength + min(2.0, size_momentum)) / 2
            
            # Determine direction
            if bullish_candles > bearish_candles:
                direction = "bullish"
            elif bearish_candles > bullish_candles:
                direction = "bearish"
            else:
                direction = "neutral"
            
            return {
                "strength": min(1.0, overall_strength),
                "direction": direction,
                "bullish_ratio": bullish_candles / len(recent_candles),
                "bearish_ratio": bearish_candles / len(recent_candles),
                "size_momentum": size_momentum
            }
            
        except Exception as e:
            logger.error(f"Momentum analysis error: {str(e)}")
            return {"strength": 0, "direction": "neutral"}
    
    def _calculate_volatility(self, candles: List[Dict]) -> Dict:
        """Calculate market volatility metrics"""
        try:
            if len(candles) < 5:
                return {"level": "unknown", "score": 0}
            
            recent_candles = candles[-10:]
            
            # Calculate price ranges
            ranges = []
            for candle in recent_candles:
                high = candle.get('high', 0)
                low = candle.get('low', 0)
                if high > 0 and low > 0:
                    ranges.append((high - low) / ((high + low) / 2))
            
            if not ranges:
                return {"level": "unknown", "score": 0}
            
            avg_range = np.mean(ranges)
            std_range = np.std(ranges)
            current_range = ranges[-1] if ranges else 0
            
            # Volatility classification
            volatility_score = avg_range
            
            if volatility_score > 0.05:
                level = "very_high"
            elif volatility_score > 0.03:
                level = "high"
            elif volatility_score > 0.02:
                level = "medium"
            elif volatility_score > 0.01:
                level = "low"
            else:
                level = "very_low"
            
            return {
                "level": level,
                "score": min(1.0, volatility_score * 20),  # Normalize to 0-1
                "current_vs_average": current_range / (avg_range + 0.001),
                "consistency": 1 - (std_range / (avg_range + 0.001))
            }
            
        except Exception as e:
            logger.error(f"Volatility calculation error: {str(e)}")
            return {"level": "unknown", "score": 0}
    
    async def _analyze_market_structure(self, candles: List[Dict]) -> Dict:
        """Analyze market structure and key levels"""
        try:
            if len(candles) < 7:
                return {"type": "insufficient_data"}
            
            recent_candles = candles[-15:]
            
            # Find swing highs and lows
            swing_highs = self._find_swing_points(recent_candles, 'high')
            swing_lows = self._find_swing_points(recent_candles, 'low')
            
            # Analyze structure breaks
            structure_breaks = self._detect_structure_breaks(swing_highs, swing_lows)
            
            # Market structure type
            if len(structure_breaks) >= 2:
                if structure_breaks[-1]['type'] == 'bullish':
                    structure_type = "bullish_structure"
                else:
                    structure_type = "bearish_structure"
            elif len(swing_highs) >= 2 and len(swing_lows) >= 2:
                if swing_highs[-1]['price'] > swing_highs[-2]['price']:
                    structure_type = "ascending_structure"
                elif swing_lows[-1]['price'] < swing_lows[-2]['price']:
                    structure_type = "descending_structure"
                else:
                    structure_type = "ranging_structure"
            else:
                structure_type = "unclear_structure"
            
            return {
                "type": structure_type,
                "swing_highs": swing_highs[-3:],  # Last 3 swing highs
                "swing_lows": swing_lows[-3:],   # Last 3 swing lows
                "structure_breaks": structure_breaks[-2:],  # Last 2 breaks
                "key_levels": self._identify_key_levels(swing_highs, swing_lows)
            }
            
        except Exception as e:
            logger.error(f"Market structure analysis error: {str(e)}")
            return {"type": "error"}
    
    def _find_swing_points(self, candles: List[Dict], point_type: str) -> List[Dict]:
        """Find swing highs or lows"""
        swing_points = []
        key = 'high' if point_type == 'high' else 'low'
        
        for i in range(2, len(candles) - 2):
            current_value = candles[i].get(key, 0)
            
            if point_type == 'high':
                # Check if current is higher than surrounding candles
                if (current_value > candles[i-1].get(key, 0) and 
                    current_value > candles[i-2].get(key, 0) and
                    current_value > candles[i+1].get(key, 0) and 
                    current_value > candles[i+2].get(key, 0)):
                    
                    swing_points.append({
                        'index': i,
                        'price': current_value,
                        'type': 'swing_high'
                    })
            else:
                # Check if current is lower than surrounding candles
                if (current_value < candles[i-1].get(key, 0) and 
                    current_value < candles[i-2].get(key, 0) and
                    current_value < candles[i+1].get(key, 0) and 
                    current_value < candles[i+2].get(key, 0)):
                    
                    swing_points.append({
                        'index': i,
                        'price': current_value,
                        'type': 'swing_low'
                    })
        
        return swing_points
    
    def _detect_structure_breaks(self, swing_highs: List[Dict], swing_lows: List[Dict]) -> List[Dict]:
        """Detect market structure breaks"""
        breaks = []
        
        # Bullish structure break: price breaks above previous swing high
        if len(swing_highs) >= 2:
            for i in range(1, len(swing_highs)):
                if swing_highs[i]['price'] > swing_highs[i-1]['price']:
                    breaks.append({
                        'type': 'bullish',
                        'level': swing_highs[i-1]['price'],
                        'break_price': swing_highs[i]['price'],
                        'strength': (swing_highs[i]['price'] - swing_highs[i-1]['price']) / swing_highs[i-1]['price']
                    })
        
        # Bearish structure break: price breaks below previous swing low
        if len(swing_lows) >= 2:
            for i in range(1, len(swing_lows)):
                if swing_lows[i]['price'] < swing_lows[i-1]['price']:
                    breaks.append({
                        'type': 'bearish',
                        'level': swing_lows[i-1]['price'],
                        'break_price': swing_lows[i]['price'],
                        'strength': (swing_lows[i-1]['price'] - swing_lows[i]['price']) / swing_lows[i-1]['price']
                    })
        
        return breaks
    
    def _identify_key_levels(self, swing_highs: List[Dict], swing_lows: List[Dict]) -> List[Dict]:
        """Identify key support and resistance levels"""
        key_levels = []
        
        # Add swing highs as resistance
        for swing in swing_highs[-3:]:
            key_levels.append({
                'price': swing['price'],
                'type': 'resistance',
                'strength': 0.8  # Default strength
            })
        
        # Add swing lows as support
        for swing in swing_lows[-3:]:
            key_levels.append({
                'price': swing['price'],
                'type': 'support',
                'strength': 0.8  # Default strength
            })
        
        return key_levels
    
    def _identify_support_resistance(self, candles: List[Dict]) -> Dict:
        """Identify current support and resistance levels"""
        try:
            if len(candles) < 10:
                return {"support": [], "resistance": []}
            
            prices = []
            for candle in candles:
                prices.extend([candle.get('high', 0), candle.get('low', 0), candle.get('close', 0)])
            
            prices = [p for p in prices if p > 0]
            if not prices:
                return {"support": [], "resistance": []}
            
            # Find price clusters (potential S/R levels)
            price_clusters = self._find_price_clusters(prices)
            
            current_price = candles[-1].get('close', 0)
            
            support_levels = [cluster for cluster in price_clusters if cluster < current_price]
            resistance_levels = [cluster for cluster in price_clusters if cluster > current_price]
            
            return {
                "support": sorted(support_levels, reverse=True)[:3],  # Top 3 support levels
                "resistance": sorted(resistance_levels)[:3],          # Top 3 resistance levels
                "current_price": current_price
            }
            
        except Exception as e:
            logger.error(f"Support/Resistance identification error: {str(e)}")
            return {"support": [], "resistance": []}
    
    def _find_price_clusters(self, prices: List[float]) -> List[float]:
        """Find price clusters using simple grouping"""
        try:
            if not prices:
                return []
            
            sorted_prices = sorted(prices)
            clusters = []
            
            # Simple clustering - group prices within 0.1% of each other
            current_cluster = [sorted_prices[0]]
            
            for price in sorted_prices[1:]:
                if abs(price - current_cluster[-1]) / current_cluster[-1] < 0.001:  # 0.1% threshold
                    current_cluster.append(price)
                else:
                    if len(current_cluster) >= 3:  # Minimum cluster size
                        clusters.append(np.mean(current_cluster))
                    current_cluster = [price]
            
            # Add the last cluster
            if len(current_cluster) >= 3:
                clusters.append(np.mean(current_cluster))
            
            return clusters
            
        except:
            return []
    
    def _analyze_market_sentiment(self, candles: List[Dict], patterns: List[Dict]) -> Dict:
        """Analyze market sentiment from candles and patterns"""
        try:
            if len(candles) < 5:
                return {"sentiment": "neutral", "confidence": 0}
            
            recent_candles = candles[-5:]
            
            # Candle sentiment analysis
            bullish_count = sum(1 for c in recent_candles if c.get('type') == 'bullish')
            bearish_count = sum(1 for c in recent_candles if c.get('type') == 'bearish')
            
            # Pattern sentiment
            bullish_patterns = sum(1 for p in patterns if 'bullish' in p.get('signal', ''))
            bearish_patterns = sum(1 for p in patterns if 'bearish' in p.get('signal', ''))
            
            # Combined sentiment score
            candle_score = (bullish_count - bearish_count) / len(recent_candles)
            pattern_score = (bullish_patterns - bearish_patterns) / max(1, len(patterns))
            
            combined_score = (candle_score + pattern_score) / 2
            
            if combined_score > 0.3:
                sentiment = "bullish"
                confidence = min(1.0, abs(combined_score))
            elif combined_score < -0.3:
                sentiment = "bearish"
                confidence = min(1.0, abs(combined_score))
            else:
                sentiment = "neutral"
                confidence = 1.0 - abs(combined_score)
            
            return {
                "sentiment": sentiment,
                "confidence": confidence,
                "candle_bias": candle_score,
                "pattern_bias": pattern_score
            }
            
        except Exception as e:
            logger.error(f"Sentiment analysis error: {str(e)}")
            return {"sentiment": "neutral", "confidence": 0}
    
    async def _identify_risk_factors(self, candles: List[Dict], patterns: List[Dict]) -> List[Dict]:
        """Identify potential risk factors"""
        risk_factors = []
        
        try:
            # Check for reversal patterns
            reversal_patterns = [p for p in patterns if 'reversal' in p.get('signal', '')]
            if reversal_patterns:
                risk_factors.append({
                    'type': 'reversal_pattern_detected',
                    'severity': 'medium',
                    'description': f"Detected {len(reversal_patterns)} reversal pattern(s)"
                })
            
            # Check for low volume/weak candles
            if len(candles) >= 3:
                recent_volumes = [c.get('volume_indicator', 0.5) for c in candles[-3:]]
                if np.mean(recent_volumes) < 0.3:
                    risk_factors.append({
                        'type': 'low_volume',
                        'severity': 'low',
                        'description': "Recent candles show low volume/weak momentum"
                    })
            
            # Check for extreme volatility
            volatility = self._calculate_volatility(candles)
            if volatility.get('level') == 'very_high':
                risk_factors.append({
                    'type': 'high_volatility',
                    'severity': 'high',
                    'description': "Market showing extreme volatility"
                })
            
            # Check for conflicting signals
            if len(patterns) >= 3:
                bullish_signals = sum(1 for p in patterns if 'bullish' in p.get('signal', ''))
                bearish_signals = sum(1 for p in patterns if 'bearish' in p.get('signal', ''))
                
                if bullish_signals > 0 and bearish_signals > 0:
                    risk_factors.append({
                        'type': 'conflicting_signals',
                        'severity': 'medium',
                        'description': "Mixed signals detected in pattern analysis"
                    })
            
        except Exception as e:
            logger.error(f"Risk factor identification error: {str(e)}")
        
        return risk_factors
    
    def _calculate_opportunity_score(self, candles: List[Dict], patterns: List[Dict]) -> float:
        """Calculate overall opportunity score"""
        try:
            if len(candles) < 3:
                return 0.0
            
            # Momentum score
            momentum = self._analyze_momentum(candles)
            momentum_score = momentum.get('strength', 0)
            
            # Pattern score
            strong_patterns = [p for p in patterns if p.get('strength', 0) > 0.7]
            pattern_score = min(1.0, len(strong_patterns) * 0.3)
            
            # Volatility score (moderate volatility is good)
            volatility = self._calculate_volatility(candles)
            vol_score = volatility.get('score', 0)
            if vol_score > 0.7:  # Too high volatility reduces opportunity
                vol_score = 1.0 - vol_score
            
            # Structure score
            structure_score = 0.5  # Default neutral
            
            # Combined opportunity score
            opportunity_score = (momentum_score * 0.4 + pattern_score * 0.3 + 
                               vol_score * 0.2 + structure_score * 0.1)
            
            return min(1.0, opportunity_score)
            
        except Exception as e:
            logger.error(f"Opportunity score calculation error: {str(e)}")
            return 0.0
    
    def _detect_manipulation(self, candles: List[Dict]) -> List[Dict]:
        """Detect potential market manipulation signs"""
        manipulation_signs = []
        
        try:
            if len(candles) < 5:
                return manipulation_signs
            
            recent_candles = candles[-5:]
            
            # Check for sudden large moves
            body_sizes = [c.get('body_size', 0) for c in recent_candles]
            avg_body_size = np.mean(body_sizes)
            
            for i, candle in enumerate(recent_candles):
                if candle.get('body_size', 0) > avg_body_size * 3:
                    manipulation_signs.append({
                        'type': 'unusual_large_move',
                        'position': i,
                        'severity': 'medium',
                        'description': 'Unusually large candle detected'
                    })
            
            # Check for wick manipulation
            for i, candle in enumerate(recent_candles):
                body_size = candle.get('body_size', 0)
                upper_wick = candle.get('upper_wick', 0)
                lower_wick = candle.get('lower_wick', 0)
                
                if upper_wick > body_size * 4 or lower_wick > body_size * 4:
                    manipulation_signs.append({
                        'type': 'wick_manipulation',
                        'position': i,
                        'severity': 'high',
                        'description': 'Potential wick manipulation detected'
                    })
            
            # Check for gap movements (significant price jumps)
            for i in range(1, len(recent_candles)):
                prev_close = recent_candles[i-1].get('close', 0)
                curr_open = recent_candles[i].get('open', 0)
                
                if prev_close > 0:
                    gap_percentage = abs(curr_open - prev_close) / prev_close
                    if gap_percentage > 0.02:  # 2% gap
                        manipulation_signs.append({
                            'type': 'price_gap',
                            'position': i,
                            'severity': 'medium',
                            'description': f'Significant price gap detected ({gap_percentage:.2%})'
                        })
            
        except Exception as e:
            logger.error(f"Manipulation detection error: {str(e)}")
        
        return manipulation_signs
    
    async def _predict_next_candle_context(self, candles: List[Dict]) -> Dict:
        """Predict context for next candle formation"""
        try:
            if len(candles) < 5:
                return {"prediction": "insufficient_data"}
            
            # Analyze recent momentum
            momentum = self._analyze_momentum(candles)
            
            # Analyze current structure
            structure = await self._analyze_market_structure(candles)
            
            # Predict next candle characteristics
            prediction = {
                "expected_direction": momentum.get('direction', 'neutral'),
                "confidence": momentum.get('strength', 0) * 0.8,
                "volatility_expectation": self._predict_volatility(candles),
                "key_levels_to_watch": structure.get('key_levels', []),
                "potential_reversal_zone": self._identify_reversal_zones(candles)
            }
            
            return prediction
            
        except Exception as e:
            logger.error(f"Next candle prediction error: {str(e)}")
            return {"prediction": "error"}
    
    def _predict_volatility(self, candles: List[Dict]) -> Dict:
        """Predict next candle volatility"""
        try:
            recent_volatility = self._calculate_volatility(candles)
            
            # Simple volatility prediction based on recent trend
            current_level = recent_volatility.get('level', 'medium')
            current_score = recent_volatility.get('score', 0.5)
            
            if current_score > 0.7:
                prediction = "decreasing"
            elif current_score < 0.3:
                prediction = "increasing"
            else:
                prediction = "stable"
            
            return {
                "prediction": prediction,
                "expected_level": current_level,
                "confidence": 0.6
            }
            
        except:
            return {"prediction": "unknown", "confidence": 0}
    
    def _identify_reversal_zones(self, candles: List[Dict]) -> List[Dict]:
        """Identify potential reversal zones"""
        try:
            if len(candles) < 10:
                return []
            
            reversal_zones = []
            
            # Find recent swing points
            recent_candles = candles[-10:]
            highs = [c.get('high', 0) for c in recent_candles]
            lows = [c.get('low', 0) for c in recent_candles]
            
            # Identify resistance zones (multiple touches of similar highs)
            high_clusters = self._find_price_clusters(highs)
            for cluster in high_clusters:
                reversal_zones.append({
                    'level': cluster,
                    'type': 'resistance_zone',
                    'strength': 0.7
                })
            
            # Identify support zones (multiple touches of similar lows)
            low_clusters = self._find_price_clusters(lows)
            for cluster in low_clusters:
                reversal_zones.append({
                    'level': cluster,
                    'type': 'support_zone',
                    'strength': 0.7
                })
            
            return reversal_zones[-4:]  # Return top 4 zones
            
        except:
            return []