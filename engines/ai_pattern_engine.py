"""
AI Pattern Engine - Advanced Market Psychology Analysis
Reads candle stories and detects institutional trading patterns
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import asyncio
from datetime import datetime
import logging
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import tensorflow as tf
from tensorflow import keras

class AIPatternEngine:
    """
    Advanced AI pattern recognition engine for market psychology
    Analyzes last 6-10 candles as a cohesive story
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.pattern_classifier = None
        self.scaler = StandardScaler()
        self.neural_network = None
        self.pattern_memory = []
        self.is_trained = False
        
        # Pattern definitions
        self.secret_patterns = {
            'shadow_trap': self._detect_shadow_trap,
            'double_pressure_reversal': self._detect_double_pressure_reversal,
            'fake_break_continuation': self._detect_fake_break_continuation,
            'volume_silent_reversal': self._detect_volume_silent_reversal,
            'body_stretch_signal': self._detect_body_stretch_signal,
            'time_memory_trap': self._detect_time_memory_trap,
            'volume_spike_sr': self._detect_volume_spike_sr,
            'volume_drop_breakout': self._detect_volume_drop_breakout
        }
    
    async def train_patterns(self, historical_data: pd.DataFrame):
        """Train AI on historical patterns"""
        try:
            self.logger.info("🧠 Training AI Pattern Recognition...")
            
            # Extract features for training
            features = await self._extract_training_features(historical_data)
            
            if len(features) < 100:
                self.logger.warning("Insufficient training data")
                return
            
            # Train traditional ML classifier
            await self._train_pattern_classifier(features)
            
            # Train neural network for complex patterns
            await self._train_neural_network(features)
            
            self.is_trained = True
            self.logger.info("✅ AI Pattern training completed")
            
        except Exception as e:
            self.logger.error(f"Error training patterns: {e}")
    
    async def detect_patterns(self, candles: List[Dict]) -> Dict[str, Any]:
        """
        Main pattern detection - analyze candle story
        """
        try:
            if len(candles) < 6:
                return {'patterns_detected': []}
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame(candles)
            
            # Extract story features
            story_features = await self._extract_story_features(df)
            
            # Detect secret trading patterns
            secret_patterns = await self._detect_secret_patterns(df)
            
            # AI-based pattern recognition
            ai_patterns = await self._ai_pattern_detection(story_features)
            
            # Market psychology analysis
            psychology = await self._analyze_market_psychology(df, story_features)
            
            # Institutional behavior detection
            institutional_signals = await self._detect_institutional_behavior(df)
            
            # Momentum and trap detection
            momentum_analysis = await self._analyze_momentum_patterns(df)
            
            return {
                'patterns_detected': secret_patterns + ai_patterns,
                'story_features': story_features,
                'market_psychology': psychology,
                'institutional_signals': institutional_signals,
                'momentum_analysis': momentum_analysis,
                'pattern_confidence': self._calculate_pattern_confidence(secret_patterns, ai_patterns),
                'story_coherence': await self._calculate_story_coherence(df)
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting patterns: {e}")
            return {'patterns_detected': []}
    
    async def _extract_story_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract features that tell the market story"""
        try:
            # Prepare basic features
            df['body_size'] = abs(df['close'] - df['open'])
            df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
            df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
            df['total_range'] = df['high'] - df['low']
            df['is_bullish'] = df['close'] > df['open']
            
            # Story progression features
            story_features = {
                'candle_count': len(df),
                'dominant_direction': self._get_dominant_direction(df),
                'volatility_progression': self._get_volatility_progression(df),
                'volume_story': self._get_volume_story(df),
                'pressure_evolution': self._get_pressure_evolution(df),
                'wick_story': self._get_wick_story(df),
                'body_evolution': self._get_body_evolution(df),
                'color_sequence': self._get_color_sequence(df)
            }
            
            return story_features
            
        except Exception as e:
            self.logger.error(f"Error extracting story features: {e}")
            return {}
    
    async def _detect_secret_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect all secret trading patterns"""
        detected_patterns = []
        
        for pattern_name, detector_func in self.secret_patterns.items():
            try:
                result = await detector_func(df)
                if result['detected']:
                    detected_patterns.append({
                        'name': pattern_name,
                        'confidence': result['confidence'],
                        'signal': result['signal'],
                        'description': result['description']
                    })
            except Exception as e:
                self.logger.error(f"Error detecting {pattern_name}: {e}")
        
        return detected_patterns
    
    async def _detect_shadow_trap(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Shadow Trap: Volume rises, candle weak = exhaustion coming"""
        try:
            if len(df) < 3:
                return {'detected': False}
            
            # Calculate synthetic volume if not available
            if 'volume' not in df.columns:
                df['volume'] = df['total_range'] * abs(df['close'] - df['open']) * 1000
            
            recent = df.iloc[-3:]
            
            # Volume rising
            volume_rising = recent['volume'].iloc[-1] > recent['volume'].iloc[0]
            volume_trend = np.polyfit(range(len(recent)), recent['volume'], 1)[0] > 0
            
            # Candle weak (small body)
            latest_candle = df.iloc[-1]
            body_ratio = abs(latest_candle['close'] - latest_candle['open']) / (latest_candle['high'] - latest_candle['low'] + 1e-10)
            candle_weak = body_ratio < 0.3
            
            # Additional confirmation
            volume_spike = recent['volume'].iloc[-1] > recent['volume'].mean() * 1.5
            
            detected = volume_rising and volume_trend and candle_weak and volume_spike
            
            return {
                'detected': detected,
                'confidence': 0.85 if detected else 0,
                'signal': 'reversal_expected',
                'description': 'Volume trap detected - exhaustion signal'
            }
            
        except Exception as e:
            return {'detected': False}
    
    async def _detect_double_pressure_reversal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Strong red + high vol → Doji → green small + high vol = instant up"""
        try:
            if len(df) < 3:
                return {'detected': False}
            
            candles = df.iloc[-3:]
            
            # Pattern sequence check
            c1, c2, c3 = candles.iloc[0], candles.iloc[1], candles.iloc[2]
            
            # First candle: Strong red with high volume
            c1_strong_red = c1['close'] < c1['open'] and abs(c1['close'] - c1['open']) / (c1['high'] - c1['low'] + 1e-10) > 0.6
            
            # Second candle: Doji
            c2_doji = abs(c2['close'] - c2['open']) / (c2['high'] - c2['low'] + 1e-10) < 0.2
            
            # Third candle: Green small with high volume
            c3_green_small = c3['close'] > c3['open'] and abs(c3['close'] - c3['open']) / (c3['high'] - c3['low'] + 1e-10) < 0.4
            
            # Volume confirmation (synthetic if needed)
            if 'volume' not in df.columns:
                df['volume'] = df['total_range'] * abs(df['close'] - df['open']) * 1000
            
            volumes = candles['volume']
            high_volume_pattern = volumes.iloc[0] > volumes.mean() * 1.2 and volumes.iloc[2] > volumes.mean() * 1.2
            
            detected = c1_strong_red and c2_doji and c3_green_small and high_volume_pattern
            
            return {
                'detected': detected,
                'confidence': 0.92 if detected else 0,
                'signal': 'strong_bullish',
                'description': 'Double pressure reversal - instant up signal'
            }
            
        except Exception as e:
            return {'detected': False}
    
    async def _detect_fake_break_continuation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Price breaks resistance, closes under, next 2 = sharp drop"""
        try:
            if len(df) < 5:
                return {'detected': False}
            
            # Identify potential resistance level
            highs = df['high'].rolling(window=3).max()
            resistance_level = highs.iloc[-5:-2].max()
            
            # Check for fake breakout pattern
            break_candle = df.iloc[-3]
            close_candle = df.iloc[-2]
            continuation_candle = df.iloc[-1]
            
            # Break above resistance
            fake_break = break_candle['high'] > resistance_level and break_candle['close'] < resistance_level
            
            # Continuation down
            continued_drop = close_candle['close'] < break_candle['close'] and continuation_candle['close'] < close_candle['close']
            
            # Volume confirmation
            if 'volume' not in df.columns:
                df['volume'] = df['total_range'] * abs(df['close'] - df['open']) * 1000
            
            volume_decline = df['volume'].iloc[-2:].mean() < df['volume'].iloc[-5:-3].mean()
            
            detected = fake_break and continued_drop and volume_decline
            
            return {
                'detected': detected,
                'confidence': 0.88 if detected else 0,
                'signal': 'strong_bearish',
                'description': 'Fake break continuation - sharp drop expected'
            }
            
        except Exception as e:
            return {'detected': False}
    
    async def _detect_volume_silent_reversal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Big reversal candle with tiny volume = false move, continuation expected"""
        try:
            if len(df) < 3:
                return {'detected': False}
            
            latest = df.iloc[-1]
            previous = df.iloc[-2]
            
            # Check for reversal candle
            direction_change = (latest['close'] > latest['open']) != (previous['close'] > previous['open'])
            big_body = abs(latest['close'] - latest['open']) / (latest['high'] - latest['low'] + 1e-10) > 0.6
            
            # Volume analysis
            if 'volume' not in df.columns:
                df['volume'] = df['total_range'] * abs(df['close'] - df['open']) * 1000
            
            avg_volume = df['volume'].iloc[-5:-1].mean()
            tiny_volume = latest['volume'] < avg_volume * 0.5
            
            detected = direction_change and big_body and tiny_volume
            
            return {
                'detected': detected,
                'confidence': 0.86 if detected else 0,
                'signal': 'continuation_expected',
                'description': 'Volume silent reversal - false move detected'
            }
            
        except Exception as e:
            return {'detected': False}
    
    async def _detect_body_stretch_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Body expands candle by candle with no wick → breakout setup"""
        try:
            if len(df) < 4:
                return {'detected': False}
            
            recent_candles = df.iloc[-4:]
            
            # Check for expanding bodies
            body_sizes = abs(recent_candles['close'] - recent_candles['open'])
            expanding_bodies = all(body_sizes.iloc[i] < body_sizes.iloc[i+1] for i in range(len(body_sizes)-1))
            
            # Check for minimal wicks
            upper_wicks = recent_candles['high'] - recent_candles[['open', 'close']].max(axis=1)
            lower_wicks = recent_candles[['open', 'close']].min(axis=1) - recent_candles['low']
            
            minimal_wicks = (upper_wicks < body_sizes * 0.2).all() and (lower_wicks < body_sizes * 0.2).all()
            
            # Same direction
            directions = recent_candles['close'] > recent_candles['open']
            same_direction = directions.all() or (~directions).all()
            
            detected = expanding_bodies and minimal_wicks and same_direction
            
            return {
                'detected': detected,
                'confidence': 0.90 if detected else 0,
                'signal': 'breakout_setup',
                'description': 'Body stretch signal - breakout imminent'
            }
            
        except Exception as e:
            return {'detected': False}
    
    async def _detect_time_memory_trap(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Previous fakeouts become trap zones"""
        try:
            if len(df) < 10:
                return {'detected': False}
            
            # This requires memory of previous fakeouts
            # For now, implement basic version
            current_price = df['close'].iloc[-1]
            
            # Look for previous rejection levels
            highs = df['high'].values
            lows = df['low'].values
            
            # Identify levels that were touched multiple times
            rejection_levels = []
            for i in range(len(df) - 5):
                level = highs[i]
                touches = sum(1 for h in highs[i+1:i+5] if abs(h - level) < (level * 0.001))
                if touches >= 2:
                    rejection_levels.append(level)
            
            # Check if current price is near a rejection level
            near_trap_zone = any(abs(current_price - level) < (current_price * 0.002) for level in rejection_levels)
            
            return {
                'detected': near_trap_zone,
                'confidence': 0.75 if near_trap_zone else 0,
                'signal': 'avoid_zone',
                'description': 'Time memory trap - avoid previously faked zones'
            }
            
        except Exception as e:
            return {'detected': False}
    
    async def _detect_volume_spike_sr(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Volume spike at S/R = Respect zone (STOP)"""
        try:
            if len(df) < 5:
                return {'detected': False}
            
            # Identify S/R levels
            recent = df.iloc[-5:]
            resistance = recent['high'].max()
            support = recent['low'].min()
            
            latest = df.iloc[-1]
            
            # Check if price is at S/R level
            at_resistance = abs(latest['high'] - resistance) < (resistance * 0.001)
            at_support = abs(latest['low'] - support) < (support * 0.001)
            
            # Volume analysis
            if 'volume' not in df.columns:
                df['volume'] = df['total_range'] * abs(df['close'] - df['open']) * 1000
            
            volume_spike = latest['volume'] > df['volume'].iloc[-5:-1].mean() * 2
            
            detected = (at_resistance or at_support) and volume_spike
            
            return {
                'detected': detected,
                'confidence': 0.80 if detected else 0,
                'signal': 'no_trade',
                'description': 'Volume spike at S/R - respect zone detected'
            }
            
        except Exception as e:
            return {'detected': False}
    
    async def _detect_volume_drop_breakout(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Volume drop at breakout = No trust (NO TRADE)"""
        try:
            if len(df) < 5:
                return {'detected': False}
            
            # Identify breakout
            recent_high = df['high'].iloc[-5:-1].max()
            recent_low = df['low'].iloc[-5:-1].min()
            
            latest = df.iloc[-1]
            
            # Breakout detection
            upward_breakout = latest['close'] > recent_high
            downward_breakout = latest['close'] < recent_low
            
            # Volume analysis
            if 'volume' not in df.columns:
                df['volume'] = df['total_range'] * abs(df['close'] - df['open']) * 1000
            
            volume_drop = latest['volume'] < df['volume'].iloc[-5:-1].mean() * 0.7
            
            detected = (upward_breakout or downward_breakout) and volume_drop
            
            return {
                'detected': detected,
                'confidence': 0.85 if detected else 0,
                'signal': 'no_trade',
                'description': 'Volume drop at breakout - no trust signal'
            }
            
        except Exception as e:
            return {'detected': False}
    
    async def _ai_pattern_detection(self, story_features: Dict) -> List[Dict]:
        """AI-based pattern detection using trained models"""
        if not self.is_trained:
            return []
        
        try:
            # Prepare features for AI
            feature_vector = self._prepare_feature_vector(story_features)
            
            # Traditional ML prediction
            ml_prediction = self.pattern_classifier.predict_proba([feature_vector])[0]
            
            # Neural network prediction
            nn_prediction = self.neural_network.predict(np.array([feature_vector]))[0]
            
            # Combine predictions
            patterns = []
            
            if ml_prediction.max() > 0.7:
                patterns.append({
                    'name': 'ml_pattern',
                    'confidence': ml_prediction.max(),
                    'signal': self._interpret_ml_prediction(ml_prediction),
                    'description': 'Machine Learning detected pattern'
                })
            
            if nn_prediction.max() > 0.8:
                patterns.append({
                    'name': 'neural_pattern',
                    'confidence': nn_prediction.max(),
                    'signal': self._interpret_nn_prediction(nn_prediction),
                    'description': 'Neural Network detected pattern'
                })
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error in AI pattern detection: {e}")
            return []
    
    async def _analyze_market_psychology(self, df: pd.DataFrame, story_features: Dict) -> Dict[str, Any]:
        """Analyze market psychology from candle story"""
        try:
            psychology = {
                'fear_greed_index': self._calculate_fear_greed(df),
                'smart_money_flow': self._detect_smart_money(df),
                'retail_exhaustion': self._detect_retail_exhaustion(df),
                'institutional_accumulation': self._detect_institutional_accumulation(df),
                'momentum_psychology': self._analyze_momentum_psychology(df),
                'volatility_psychology': self._analyze_volatility_psychology(df)
            }
            
            return psychology
            
        except Exception as e:
            self.logger.error(f"Error analyzing market psychology: {e}")
            return {}
    
    async def _detect_institutional_behavior(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect institutional trading behavior"""
        try:
            signals = {
                'large_block_trades': self._detect_large_blocks(df),
                'algorithmic_patterns': self._detect_algo_patterns(df),
                'liquidity_hunting': self._detect_liquidity_hunting(df),
                'stop_hunting': self._detect_stop_hunting(df),
                'accumulation_distribution': self._detect_accumulation_distribution(df)
            }
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error detecting institutional behavior: {e}")
            return {}
    
    # Helper methods for story analysis
    def _get_dominant_direction(self, df: pd.DataFrame) -> str:
        """Get dominant direction of the story"""
        bullish_count = (df['close'] > df['open']).sum()
        bearish_count = len(df) - bullish_count
        
        if bullish_count > bearish_count * 1.5:
            return 'bullish'
        elif bearish_count > bullish_count * 1.5:
            return 'bearish'
        else:
            return 'neutral'
    
    def _get_volatility_progression(self, df: pd.DataFrame) -> str:
        """Analyze volatility progression"""
        ranges = df['high'] - df['low']
        if len(ranges) < 3:
            return 'stable'
        
        trend = np.polyfit(range(len(ranges)), ranges, 1)[0]
        
        if trend > 0.0001:
            return 'expanding'
        elif trend < -0.0001:
            return 'contracting'
        else:
            return 'stable'
    
    def _get_volume_story(self, df: pd.DataFrame) -> str:
        """Analyze volume story progression"""
        if 'volume' not in df.columns:
            return 'synthetic'
        
        volumes = df['volume']
        if len(volumes) < 3:
            return 'stable'
        
        trend = np.polyfit(range(len(volumes)), volumes, 1)[0]
        
        if trend > 0:
            return 'increasing'
        elif trend < 0:
            return 'decreasing'
        else:
            return 'stable'
    
    def _get_pressure_evolution(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze pressure evolution"""
        buying_pressure = []
        selling_pressure = []
        
        for _, candle in df.iterrows():
            if candle['close'] > candle['open']:
                buying_pressure.append(abs(candle['close'] - candle['open']) / (candle['high'] - candle['low'] + 1e-10))
                selling_pressure.append((candle['high'] - candle['close']) / (candle['high'] - candle['low'] + 1e-10))
            else:
                selling_pressure.append(abs(candle['close'] - candle['open']) / (candle['high'] - candle['low'] + 1e-10))
                buying_pressure.append((candle['close'] - candle['low']) / (candle['high'] - candle['low'] + 1e-10))
        
        return {
            'avg_buying_pressure': np.mean(buying_pressure),
            'avg_selling_pressure': np.mean(selling_pressure),
            'pressure_trend': np.polyfit(range(len(buying_pressure)), buying_pressure, 1)[0]
        }
    
    def _get_wick_story(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze wick story"""
        upper_wicks = df['high'] - df[['open', 'close']].max(axis=1)
        lower_wicks = df[['open', 'close']].min(axis=1) - df['low']
        
        return {
            'upper_wick_trend': np.polyfit(range(len(upper_wicks)), upper_wicks, 1)[0],
            'lower_wick_trend': np.polyfit(range(len(lower_wicks)), lower_wicks, 1)[0],
            'rejection_direction': 'upper' if upper_wicks.mean() > lower_wicks.mean() else 'lower'
        }
    
    def _get_body_evolution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze body size evolution"""
        body_sizes = abs(df['close'] - df['open'])
        
        return {
            'body_trend': np.polyfit(range(len(body_sizes)), body_sizes, 1)[0],
            'avg_body_size': body_sizes.mean(),
            'body_volatility': body_sizes.std()
        }
    
    def _get_color_sequence(self, df: pd.DataFrame) -> List[str]:
        """Get color sequence"""
        return ['green' if close > open else 'red' for open, close in zip(df['open'], df['close'])]
    
    # Additional helper methods would be implemented here...
    # (Training methods, feature preparation, psychology analysis, etc.)
    
    async def _extract_training_features(self, data: pd.DataFrame) -> List[Dict]:
        """Extract features for training"""
        # Implementation for training feature extraction
        return []
    
    async def _train_pattern_classifier(self, features: List[Dict]):
        """Train the pattern classifier"""
        # Implementation for training ML classifier
        pass
    
    async def _train_neural_network(self, features: List[Dict]):
        """Train neural network"""
        # Implementation for training neural network
        pass
    
    def _calculate_pattern_confidence(self, secret_patterns: List, ai_patterns: List) -> float:
        """Calculate overall pattern confidence"""
        if not secret_patterns and not ai_patterns:
            return 0.0
        
        all_patterns = secret_patterns + ai_patterns
        confidences = [p['confidence'] for p in all_patterns]
        
        return np.mean(confidences) if confidences else 0.0
    
    async def _calculate_story_coherence(self, df: pd.DataFrame) -> float:
        """Calculate how coherent the candle story is"""
        # Implementation for story coherence calculation
        return 0.8  # Placeholder
    
    # Placeholder implementations for psychology analysis methods
    def _calculate_fear_greed(self, df: pd.DataFrame) -> float:
        return 0.5  # Placeholder
    
    def _detect_smart_money(self, df: pd.DataFrame) -> bool:
        return False  # Placeholder
    
    def _detect_retail_exhaustion(self, df: pd.DataFrame) -> bool:
        return False  # Placeholder
    
    def _detect_institutional_accumulation(self, df: pd.DataFrame) -> bool:
        return False  # Placeholder
    
    def _analyze_momentum_psychology(self, df: pd.DataFrame) -> Dict:
        return {}  # Placeholder
    
    def _analyze_volatility_psychology(self, df: pd.DataFrame) -> Dict:
        return {}  # Placeholder
    
    def _detect_large_blocks(self, df: pd.DataFrame) -> bool:
        return False  # Placeholder
    
    def _detect_algo_patterns(self, df: pd.DataFrame) -> bool:
        return False  # Placeholder
    
    def _detect_liquidity_hunting(self, df: pd.DataFrame) -> bool:
        return False  # Placeholder
    
    def _detect_stop_hunting(self, df: pd.DataFrame) -> bool:
        return False  # Placeholder
    
    def _detect_accumulation_distribution(self, df: pd.DataFrame) -> Dict:
        return {}  # Placeholder
    
    def _prepare_feature_vector(self, story_features: Dict) -> List[float]:
        return [0.5] * 10  # Placeholder
    
    def _interpret_ml_prediction(self, prediction: np.ndarray) -> str:
        return 'bullish'  # Placeholder
    
    def _interpret_nn_prediction(self, prediction: np.ndarray) -> str:
        return 'bearish'  # Placeholder