"""
Market Psychology Engine
Analyzes market psychology from candle stories and human behavior patterns
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import asyncio
from datetime import datetime
import logging
from sklearn.preprocessing import StandardScaler
from scipy import stats

class MarketPsychologyEngine:
    """
    Advanced market psychology analysis engine
    Detects human behavior patterns in market data
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.psychology_model = None
        self.fear_greed_history = []
        self.sentiment_indicators = {}
        self.is_initialized = False
        
    async def initialize_psychology_model(self, historical_data: pd.DataFrame):
        """Initialize psychology model with historical data"""
        try:
            self.logger.info("🧠 Initializing Market Psychology Engine...")
            
            # Build sentiment indicators from historical data
            await self._build_sentiment_indicators(historical_data)
            
            # Initialize fear/greed index
            await self._initialize_fear_greed_index(historical_data)
            
            # Initialize behavioral pattern recognition
            await self._initialize_behavioral_patterns(historical_data)
            
            self.is_initialized = True
            self.logger.info("✅ Market Psychology Engine initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing psychology model: {e}")
    
    async def analyze_psychology(self, candles: List[Dict]) -> Dict[str, Any]:
        """
        Main psychology analysis function
        """
        try:
            if not self.is_initialized:
                return {}
            
            # Convert to DataFrame
            df = pd.DataFrame(candles)
            
            # Core psychology analysis
            psychology_analysis = {
                'fear_greed_index': await self._calculate_fear_greed_index(df),
                'smart_money_flow': await self._detect_smart_money_flow(df),
                'retail_exhaustion': await self._detect_retail_exhaustion(df),
                'institutional_accumulation': await self._detect_institutional_accumulation(df),
                'momentum_psychology': await self._analyze_momentum_psychology(df),
                'volatility_psychology': await self._analyze_volatility_psychology(df),
                'volume_psychology': await self._analyze_volume_psychology(df),
                'crowd_behavior': await self._analyze_crowd_behavior(df),
                'panic_greed_signals': await self._detect_panic_greed_signals(df),
                'manipulation_signals': await self._detect_manipulation_signals(df)
            }
            
            return psychology_analysis
            
        except Exception as e:
            self.logger.error(f"Error in psychology analysis: {e}")
            return {}
    
    async def _calculate_fear_greed_index(self, df: pd.DataFrame) -> float:
        """Calculate Fear & Greed Index"""
        try:
            # Prepare features
            df = self._prepare_candle_features(df)
            
            # Component 1: Price momentum (30%)
            price_momentum = self._calculate_price_momentum(df)
            
            # Component 2: Volatility (25%)
            volatility_fear = self._calculate_volatility_fear(df)
            
            # Component 3: Volume behavior (20%)
            volume_sentiment = self._calculate_volume_sentiment(df)
            
            # Component 4: Market breadth (15%)
            market_breadth = self._calculate_market_breadth(df)
            
            # Component 5: Safe haven demand (10%)
            safe_haven = self._calculate_safe_haven_demand(df)
            
            # Weighted Fear & Greed Index
            fear_greed_index = (
                price_momentum * 0.30 +
                volatility_fear * 0.25 +
                volume_sentiment * 0.20 +
                market_breadth * 0.15 +
                safe_haven * 0.10
            )
            
            # Normalize to 0-1 scale
            fear_greed_index = max(0, min(1, fear_greed_index))
            
            # Store in history
            self.fear_greed_history.append({
                'timestamp': datetime.now(),
                'value': fear_greed_index
            })
            
            # Keep only last 100 readings
            if len(self.fear_greed_history) > 100:
                self.fear_greed_history = self.fear_greed_history[-100:]
            
            return fear_greed_index
            
        except Exception as e:
            self.logger.error(f"Error calculating fear/greed index: {e}")
            return 0.5
    
    async def _detect_smart_money_flow(self, df: pd.DataFrame) -> bool:
        """Detect smart money flow patterns"""
        try:
            df = self._prepare_candle_features(df)
            
            # Smart money indicators
            indicators = []
            
            # 1. Volume-price divergence
            if len(df) >= 5:
                price_change = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5]
                volume_change = (df.get('volume', df['synthetic_volume']).iloc[-1] - 
                               df.get('volume', df['synthetic_volume']).iloc[-5]) / df.get('volume', df['synthetic_volume']).iloc[-5]
                
                # Smart money: price up, volume down (accumulation quietly)
                # or price down, volume up (distribution)
                if (price_change > 0.01 and volume_change < -0.1) or (price_change < -0.01 and volume_change > 0.2):
                    indicators.append(True)
                else:
                    indicators.append(False)
            
            # 2. Wick patterns suggesting smart rejection
            latest = df.iloc[-1]
            if 'upper_wick' in df.columns and 'lower_wick' in df.columns:
                total_range = latest['high'] - latest['low']
                upper_wick_ratio = latest['upper_wick'] / (total_range + 1e-10)
                lower_wick_ratio = latest['lower_wick'] / (total_range + 1e-10)
                
                # Smart rejection patterns
                smart_upper_rejection = upper_wick_ratio > 0.6
                smart_lower_rejection = lower_wick_ratio > 0.6
                
                indicators.append(smart_upper_rejection or smart_lower_rejection)
            
            # 3. Off-hours or low liquidity accumulation
            current_hour = datetime.now().hour
            off_hours = current_hour < 8 or current_hour > 18
            if off_hours and len(df) >= 3:
                recent_volume = df.get('volume', df['synthetic_volume']).iloc[-3:].mean()
                historical_volume = df.get('volume', df['synthetic_volume']).iloc[:-3].mean()
                quiet_accumulation = recent_volume > historical_volume * 1.2
                indicators.append(quiet_accumulation)
            
            # Smart money detected if majority of indicators are positive
            return sum(indicators) >= len(indicators) / 2
            
        except Exception as e:
            self.logger.error(f"Error detecting smart money flow: {e}")
            return False
    
    async def _detect_retail_exhaustion(self, df: pd.DataFrame) -> bool:
        """Detect retail trader exhaustion patterns"""
        try:
            df = self._prepare_candle_features(df)
            
            if len(df) < 5:
                return False
            
            # Retail exhaustion indicators
            indicators = []
            
            # 1. High volatility with decreasing volume
            volatility = df['high'] - df['low']
            volume = df.get('volume', df['synthetic_volume'])
            
            recent_volatility = volatility.iloc[-3:].mean()
            historical_volatility = volatility.iloc[:-3].mean()
            recent_volume = volume.iloc[-3:].mean()
            historical_volume = volume.iloc[:-3].mean()
            
            high_vol_low_volume = (recent_volatility > historical_volatility * 1.2 and 
                                 recent_volume < historical_volume * 0.8)
            indicators.append(high_vol_low_volume)
            
            # 2. Multiple failed breakout attempts
            highs = df['high'].rolling(window=3).max()
            recent_highs = highs.iloc[-5:]
            failed_breakouts = sum(1 for i in range(1, len(recent_highs)) 
                                 if recent_highs.iloc[i] <= recent_highs.iloc[i-1])
            indicators.append(failed_breakouts >= 3)
            
            # 3. Decreasing body sizes with same direction (exhaustion)
            body_sizes = abs(df['close'] - df['open'])
            if len(body_sizes) >= 4:
                recent_bodies = body_sizes.iloc[-4:]
                decreasing_bodies = all(recent_bodies.iloc[i] >= recent_bodies.iloc[i+1] 
                                      for i in range(len(recent_bodies)-1))
                
                # Same direction candles
                recent_directions = (df['close'] > df['open']).iloc[-4:]
                same_direction = len(set(recent_directions)) == 1
                
                indicators.append(decreasing_bodies and same_direction)
            
            # 4. Extreme sentiment readings
            if hasattr(self, 'fear_greed_history') and self.fear_greed_history:
                latest_fg = self.fear_greed_history[-1]['value']
                extreme_sentiment = latest_fg < 0.15 or latest_fg > 0.85
                indicators.append(extreme_sentiment)
            
            return sum(indicators) >= 2
            
        except Exception as e:
            self.logger.error(f"Error detecting retail exhaustion: {e}")
            return False
    
    async def _detect_institutional_accumulation(self, df: pd.DataFrame) -> bool:
        """Detect institutional accumulation patterns"""
        try:
            df = self._prepare_candle_features(df)
            
            if len(df) < 10:
                return False
            
            # Institutional accumulation indicators
            indicators = []
            
            # 1. Consistent volume at support levels
            lows = df['low'].rolling(window=5).min()
            support_level = lows.iloc[-10:-5].min()
            
            near_support = abs(df['low'].iloc[-1] - support_level) < (support_level * 0.005)
            high_volume_support = (df.get('volume', df['synthetic_volume']).iloc[-1] > 
                                 df.get('volume', df['synthetic_volume']).iloc[-10:-1].mean() * 1.5)
            
            indicators.append(near_support and high_volume_support)
            
            # 2. Gradual price increase with steady volume
            price_trend = np.polyfit(range(len(df)), df['close'], 1)[0]
            volume_stability = (df.get('volume', df['synthetic_volume']).std() / 
                              df.get('volume', df['synthetic_volume']).mean()) < 0.3
            
            gradual_accumulation = price_trend > 0 and volume_stability
            indicators.append(gradual_accumulation)
            
            # 3. Large body candles with minimal wicks (institutional size)
            latest = df.iloc[-1]
            body_size = abs(latest['close'] - latest['open'])
            total_range = latest['high'] - latest['low']
            
            if total_range > 0:
                body_ratio = body_size / total_range
                large_body = body_ratio > 0.7
                
                if 'upper_wick' in df.columns and 'lower_wick' in df.columns:
                    minimal_wicks = (latest['upper_wick'] + latest['lower_wick']) / total_range < 0.3
                    indicators.append(large_body and minimal_wicks)
            
            # 4. Time-based accumulation (during optimal hours)
            current_hour = datetime.now().hour
            optimal_hours = 9 <= current_hour <= 16  # Market hours
            if optimal_hours:
                indicators.append(True)
            
            return sum(indicators) >= 2
            
        except Exception as e:
            self.logger.error(f"Error detecting institutional accumulation: {e}")
            return False
    
    async def _analyze_momentum_psychology(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze momentum-based psychology"""
        try:
            df = self._prepare_candle_features(df)
            
            if len(df) < 5:
                return {}
            
            # Momentum indicators
            closes = df['close'].values
            
            # Short-term momentum
            short_momentum = (closes[-1] - closes[-3]) / closes[-3] if closes[-3] != 0 else 0
            
            # Medium-term momentum
            medium_momentum = (closes[-1] - closes[-5]) / closes[-5] if len(closes) >= 5 and closes[-5] != 0 else 0
            
            # Momentum acceleration
            if len(closes) >= 6:
                recent_momentum = (closes[-1] - closes[-3]) / closes[-3] if closes[-3] != 0 else 0
                previous_momentum = (closes[-3] - closes[-6]) / closes[-6] if closes[-6] != 0 else 0
                momentum_acceleration = recent_momentum - previous_momentum
            else:
                momentum_acceleration = 0
            
            # Psychology interpretation
            bullish_momentum = short_momentum > 0.01 and medium_momentum > 0.01
            bearish_momentum = short_momentum < -0.01 and medium_momentum < -0.01
            momentum_shift = abs(momentum_acceleration) > 0.005
            
            # FOMO detection (Fear of Missing Out)
            if len(df) >= 4:
                recent_ranges = (df['high'] - df['low']).iloc[-4:]
                expanding_ranges = all(recent_ranges.iloc[i] < recent_ranges.iloc[i+1] 
                                     for i in range(len(recent_ranges)-1))
                
                # Same direction with expanding ranges = FOMO
                recent_directions = (df['close'] > df['open']).iloc[-4:]
                same_direction = len(set(recent_directions)) == 1
                
                fomo_detected = expanding_ranges and same_direction
            else:
                fomo_detected = False
            
            return {
                'short_momentum': short_momentum,
                'medium_momentum': medium_momentum,
                'momentum_acceleration': momentum_acceleration,
                'bullish_momentum': bullish_momentum,
                'bearish_momentum': bearish_momentum,
                'momentum_shift_detected': momentum_shift,
                'fomo_detected': fomo_detected,
                'momentum_strength': abs(short_momentum) + abs(medium_momentum)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing momentum psychology: {e}")
            return {}
    
    async def _analyze_volatility_psychology(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volatility-based psychology"""
        try:
            df = self._prepare_candle_features(df)
            
            # Volatility measures
            ranges = df['high'] - df['low']
            
            # Current vs historical volatility
            current_volatility = ranges.iloc[-1]
            avg_volatility = ranges.mean()
            volatility_ratio = current_volatility / (avg_volatility + 1e-10)
            
            # Volatility trend
            if len(ranges) >= 5:
                volatility_trend = np.polyfit(range(len(ranges)), ranges, 1)[0]
            else:
                volatility_trend = 0
            
            # Fear/complacency signals
            high_volatility = volatility_ratio > 2.0  # Fear
            low_volatility = volatility_ratio < 0.5   # Complacency
            
            # Volatility clustering
            high_vol_periods = ranges > avg_volatility * 1.5
            volatility_clustering = sum(high_vol_periods.iloc[-5:]) >= 3 if len(high_vol_periods) >= 5 else False
            
            # Volatility breakout
            compression_threshold = avg_volatility * 0.6
            recent_compression = all(ranges.iloc[-3:] < compression_threshold) if len(ranges) >= 3 else False
            volatility_breakout = recent_compression and current_volatility > avg_volatility * 1.3
            
            return {
                'volatility_ratio': volatility_ratio,
                'volatility_trend': volatility_trend,
                'high_volatility_fear': high_volatility,
                'low_volatility_complacency': low_volatility,
                'volatility_clustering': volatility_clustering,
                'volatility_breakout': volatility_breakout,
                'fear_level': min(1.0, volatility_ratio / 3.0)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing volatility psychology: {e}")
            return {}
    
    async def _analyze_volume_psychology(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume-based psychology"""
        try:
            df = self._prepare_candle_features(df)
            
            volume = df.get('volume', df['synthetic_volume'])
            
            # Volume trend
            if len(volume) >= 5:
                volume_trend = np.polyfit(range(len(volume)), volume, 1)[0]
            else:
                volume_trend = 0
            
            # Volume spikes and dry-ups
            avg_volume = volume.mean()
            current_volume = volume.iloc[-1]
            
            volume_spike = current_volume > avg_volume * 2.0
            volume_dry_up = current_volume < avg_volume * 0.5
            
            # Panic volume (high volume + high volatility)
            current_range = df['high'].iloc[-1] - df['low'].iloc[-1]
            avg_range = (df['high'] - df['low']).mean()
            
            panic_volume = volume_spike and current_range > avg_range * 1.5
            
            # Distribution/accumulation
            # Accumulation: price stable/up, volume increasing
            # Distribution: price stable/down, volume increasing
            
            if len(df) >= 5:
                price_change = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5]
                volume_change = (volume.iloc[-1] - volume.iloc[-5]) / volume.iloc[-5]
                
                accumulation = price_change >= 0 and volume_change > 0.2
                distribution = price_change <= 0 and volume_change > 0.2
            else:
                accumulation = False
                distribution = False
            
            return {
                'volume_trend': volume_trend,
                'volume_spike': volume_spike,
                'volume_dry_up': volume_dry_up,
                'panic_volume': panic_volume,
                'accumulation_detected': accumulation,
                'distribution_detected': distribution,
                'volume_participation': current_volume / (avg_volume + 1e-10)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing volume psychology: {e}")
            return {}
    
    async def _analyze_crowd_behavior(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze crowd psychology and herding behavior"""
        try:
            df = self._prepare_candle_features(df)
            
            # Herding indicators
            directions = df['close'] > df['open']
            
            # Consecutive same-direction candles (herding)
            if len(directions) >= 5:
                recent_directions = directions.iloc[-5:]
                herding_bullish = all(recent_directions)
                herding_bearish = not any(recent_directions)
                herding_detected = herding_bullish or herding_bearish
            else:
                herding_detected = False
                herding_bullish = False
                herding_bearish = False
            
            # Crowd exhaustion (decreasing participation in trend)
            if len(df) >= 5:
                body_sizes = abs(df['close'] - df['open'])
                recent_bodies = body_sizes.iloc[-3:]
                
                decreasing_participation = all(recent_bodies.iloc[i] >= recent_bodies.iloc[i+1] 
                                             for i in range(len(recent_bodies)-1))
            else:
                decreasing_participation = False
            
            # Contrarian signals
            extreme_herding = len(set(directions.iloc[-7:])) == 1 if len(directions) >= 7 else False
            contrarian_setup = extreme_herding and decreasing_participation
            
            return {
                'herding_detected': herding_detected,
                'herding_bullish': herding_bullish,
                'herding_bearish': herding_bearish,
                'decreasing_participation': decreasing_participation,
                'extreme_herding': extreme_herding,
                'contrarian_setup': contrarian_setup,
                'crowd_sentiment': 'bullish' if herding_bullish else 'bearish' if herding_bearish else 'neutral'
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing crowd behavior: {e}")
            return {}
    
    async def _detect_panic_greed_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect panic and greed signals"""
        try:
            df = self._prepare_candle_features(df)
            
            # Panic signals
            panic_indicators = []
            
            # 1. Large red candle with high volume
            latest = df.iloc[-1]
            is_red = latest['close'] < latest['open']
            large_body = abs(latest['close'] - latest['open']) / (latest['high'] - latest['low'] + 1e-10) > 0.7
            
            volume = df.get('volume', df['synthetic_volume'])
            high_volume = volume.iloc[-1] > volume.mean() * 2.0
            
            panic_candle = is_red and large_body and high_volume
            panic_indicators.append(panic_candle)
            
            # 2. Gap down with continuation
            if len(df) >= 2:
                prev_candle = df.iloc[-2]
                gap_down = latest['open'] < prev_candle['low']
                continuation_down = latest['close'] < latest['open']
                
                panic_gap = gap_down and continuation_down
                panic_indicators.append(panic_gap)
            
            # Greed signals
            greed_indicators = []
            
            # 1. Large green candle with expanding volume
            is_green = latest['close'] > latest['open']
            greed_candle = is_green and large_body and high_volume
            greed_indicators.append(greed_candle)
            
            # 2. Gap up with continuation
            if len(df) >= 2:
                gap_up = latest['open'] > prev_candle['high']
                continuation_up = latest['close'] > latest['open']
                
                greed_gap = gap_up and continuation_up
                greed_indicators.append(greed_gap)
            
            # 3. Parabolic move (multiple large candles same direction)
            if len(df) >= 4:
                recent_candles = df.iloc[-4:]
                all_green = all(candle['close'] > candle['open'] for _, candle in recent_candles.iterrows())
                expanding_sizes = True
                
                if all_green:
                    body_sizes = [abs(candle['close'] - candle['open']) for _, candle in recent_candles.iterrows()]
                    expanding_sizes = all(body_sizes[i] <= body_sizes[i+1] for i in range(len(body_sizes)-1))
                
                parabolic_greed = all_green and expanding_sizes
                greed_indicators.append(parabolic_greed)
            
            panic_detected = sum(panic_indicators) >= 1
            greed_detected = sum(greed_indicators) >= 1
            
            return {
                'panic_detected': panic_detected,
                'greed_detected': greed_detected,
                'panic_strength': sum(panic_indicators),
                'greed_strength': sum(greed_indicators),
                'extreme_emotion': panic_detected or greed_detected
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting panic/greed signals: {e}")
            return {}
    
    async def _detect_manipulation_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect potential market manipulation signals"""
        try:
            df = self._prepare_candle_features(df)
            
            manipulation_signals = []
            
            # 1. Stop hunting - quick spike and reversal
            if len(df) >= 3:
                recent = df.iloc[-3:]
                
                # Look for spike pattern
                middle_candle = recent.iloc[1]
                prev_candle = recent.iloc[0]
                latest_candle = recent.iloc[2]
                
                # Upward spike and reversal
                upward_spike = (middle_candle['high'] > prev_candle['high'] * 1.005 and
                              latest_candle['close'] < middle_candle['open'])
                
                # Downward spike and reversal
                downward_spike = (middle_candle['low'] < prev_candle['low'] * 0.995 and
                                latest_candle['close'] > middle_candle['open'])
                
                stop_hunting = upward_spike or downward_spike
                manipulation_signals.append(stop_hunting)
            
            # 2. Artificial volume spike without follow-through
            volume = df.get('volume', df['synthetic_volume'])
            if len(volume) >= 3:
                volume_spike = volume.iloc[-2] > volume.iloc[:-2].mean() * 3.0
                no_follow_through = volume.iloc[-1] < volume.iloc[:-2].mean()
                
                artificial_spike = volume_spike and no_follow_through
                manipulation_signals.append(artificial_spike)
            
            # 3. Painted tape (small volume, large price moves)
            if len(df) >= 2:
                latest = df.iloc[-1]
                price_move = abs(latest['close'] - latest['open']) / latest['open']
                low_volume = volume.iloc[-1] < volume.mean() * 0.5
                
                painted_tape = price_move > 0.01 and low_volume
                manipulation_signals.append(painted_tape)
            
            manipulation_detected = sum(manipulation_signals) >= 1
            
            return {
                'manipulation_detected': manipulation_detected,
                'manipulation_strength': sum(manipulation_signals),
                'stop_hunting_suspected': len(manipulation_signals) > 0 and manipulation_signals[0] if len(manipulation_signals) > 0 else False,
                'artificial_volume_suspected': len(manipulation_signals) > 1 and manipulation_signals[1] if len(manipulation_signals) > 1 else False
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting manipulation signals: {e}")
            return {}
    
    def _prepare_candle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare standard candle features"""
        # Add synthetic volume if not present
        if 'volume' not in df.columns:
            df['synthetic_volume'] = (df['high'] - df['low']) * abs(df['close'] - df['open']) * 1000
        
        # Add wick calculations if not present
        if 'upper_wick' not in df.columns:
            df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        if 'lower_wick' not in df.columns:
            df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        
        return df
    
    # Helper methods for Fear & Greed Index calculation
    def _calculate_price_momentum(self, df: pd.DataFrame) -> float:
        """Calculate price momentum component"""
        if len(df) < 5:
            return 0.5
        
        # 1-day vs 5-day momentum
        recent_change = (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]
        longer_change = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5]
        
        momentum_score = (recent_change + longer_change) / 2
        
        # Normalize to 0-1 scale (0 = extreme fear, 1 = extreme greed)
        normalized = (momentum_score + 0.05) / 0.1  # Assuming +/-5% is extreme
        return max(0, min(1, normalized))
    
    def _calculate_volatility_fear(self, df: pd.DataFrame) -> float:
        """Calculate volatility component (high volatility = fear)"""
        ranges = df['high'] - df['low']
        current_volatility = ranges.iloc[-1]
        avg_volatility = ranges.mean()
        
        volatility_ratio = current_volatility / (avg_volatility + 1e-10)
        
        # High volatility = fear (low score), low volatility = greed (high score)
        fear_score = 1 / (1 + volatility_ratio)  # Inverse relationship
        return max(0, min(1, fear_score))
    
    def _calculate_volume_sentiment(self, df: pd.DataFrame) -> float:
        """Calculate volume sentiment component"""
        volume = df.get('volume', df.get('synthetic_volume', pd.Series([1000] * len(df))))
        
        if len(volume) < 3:
            return 0.5
        
        current_volume = volume.iloc[-1]
        avg_volume = volume.mean()
        
        # High volume with up move = greed, high volume with down move = fear
        price_change = (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]
        volume_ratio = current_volume / (avg_volume + 1e-10)
        
        if price_change > 0:
            sentiment = 0.5 + (volume_ratio - 1) * 0.1  # Greed with high volume
        else:
            sentiment = 0.5 - (volume_ratio - 1) * 0.1  # Fear with high volume
        
        return max(0, min(1, sentiment))
    
    def _calculate_market_breadth(self, df: pd.DataFrame) -> float:
        """Calculate market breadth component"""
        # For single instrument, use trend consistency
        if len(df) < 5:
            return 0.5
        
        # Count bullish vs bearish candles in recent period
        recent_candles = df.iloc[-5:]
        bullish_count = sum(candle['close'] > candle['open'] for _, candle in recent_candles.iterrows())
        
        breadth_score = bullish_count / len(recent_candles)
        return breadth_score
    
    def _calculate_safe_haven_demand(self, df: pd.DataFrame) -> float:
        """Calculate safe haven demand component"""
        # For forex/crypto, use volatility as proxy
        # High volatility = high safe haven demand = fear
        ranges = df['high'] - df['low']
        volatility = ranges.std() / ranges.mean() if ranges.mean() > 0 else 0
        
        # Normalize volatility to 0-1 scale
        safe_haven_score = 1 - min(1, volatility * 10)  # High volatility = low score (fear)
        return max(0, safe_haven_score)
    
    # Placeholder initialization methods
    async def _build_sentiment_indicators(self, data: pd.DataFrame):
        """Build sentiment indicators from historical data"""
        pass
    
    async def _initialize_fear_greed_index(self, data: pd.DataFrame):
        """Initialize fear/greed index with historical baseline"""
        pass
    
    async def _initialize_behavioral_patterns(self, data: pd.DataFrame):
        """Initialize behavioral pattern recognition"""
        pass