"""
Advanced Candle Analyzer Engine
OpenCV-style pattern recognition for candle analysis
"""

import numpy as np
import pandas as pd
import cv2
from typing import Dict, List, Tuple, Optional, Any
import asyncio
from datetime import datetime
import logging

class CandleAnalyzer:
    """
    Advanced candle analysis using computer vision techniques
    Analyzes body, wicks, size, color, spacing, and volume patterns
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    async def analyze_candles(self, candles: List[Dict]) -> Dict[str, Any]:
        """
        Comprehensive candle analysis using OpenCV-style techniques
        """
        try:
            if len(candles) < 3:
                return {}
            
            # Convert to structured format
            candle_data = self._prepare_candle_data(candles)
            
            # Core candle feature extraction
            features = {
                'body_analysis': await self._analyze_bodies(candle_data),
                'wick_analysis': await self._analyze_wicks(candle_data),
                'size_analysis': await self._analyze_sizes(candle_data),
                'color_pattern': await self._analyze_color_patterns(candle_data),
                'spacing_analysis': await self._analyze_spacing(candle_data),
                'volume_analysis': await self._analyze_volume_patterns(candle_data),
                'pressure_detection': await self._detect_pressure(candle_data),
                'momentum_signals': await self._detect_momentum_shifts(candle_data)
            }
            
            # Advanced pattern detection
            features.update({
                'engulfing_patterns': await self._detect_engulfing(candle_data),
                'doji_patterns': await self._detect_doji_patterns(candle_data),
                'hammer_patterns': await self._detect_hammer_patterns(candle_data),
                'shooting_star_patterns': await self._detect_shooting_star(candle_data),
                'spinning_top_patterns': await self._detect_spinning_tops(candle_data)
            })
            
            # Synthetic volume if not available
            if 'volume' not in candle_data.columns:
                features['synthetic_volume'] = await self._calculate_synthetic_volume(candle_data)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error in candle analysis: {e}")
            return {}
    
    def _prepare_candle_data(self, candles: List[Dict]) -> pd.DataFrame:
        """Convert candle list to structured DataFrame"""
        df = pd.DataFrame(candles)
        
        # Ensure required columns
        required_cols = ['open', 'high', 'low', 'close', 'timestamp']
        for col in required_cols:
            if col not in df.columns:
                if col == 'timestamp':
                    df[col] = pd.date_range(start=datetime.now(), periods=len(df), freq='1min')
                else:
                    df[col] = 1.0  # Default values
        
        # Calculate derived features
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        df['total_range'] = df['high'] - df['low']
        df['is_bullish'] = df['close'] > df['open']
        df['body_ratio'] = df['body_size'] / (df['total_range'] + 1e-10)
        
        return df
    
    async def _analyze_bodies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze candle body characteristics"""
        try:
            latest_candle = df.iloc[-1]
            
            # Body size analysis
            avg_body_size = df['body_size'].rolling(window=5).mean().iloc[-1]
            body_size_ratio = latest_candle['body_size'] / (avg_body_size + 1e-10)
            
            # Body position analysis
            body_position = (latest_candle['close'] + latest_candle['open']) / 2
            range_position = (body_position - latest_candle['low']) / (latest_candle['total_range'] + 1e-10)
            
            # Body strength analysis
            body_strength = latest_candle['body_ratio']
            
            return {
                'size_ratio': body_size_ratio,
                'position_in_range': range_position,
                'strength': body_strength,
                'is_strong_body': body_strength > 0.7,
                'is_weak_body': body_strength < 0.3,
                'size_trend': 'expanding' if body_size_ratio > 1.2 else 'contracting' if body_size_ratio < 0.8 else 'stable'
            }
            
        except Exception as e:
            self.logger.error(f"Error in body analysis: {e}")
            return {}
    
    async def _analyze_wicks(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Advanced wick analysis for market psychology"""
        try:
            latest = df.iloc[-1]
            
            # Wick ratios
            upper_wick_ratio = latest['upper_wick'] / (latest['total_range'] + 1e-10)
            lower_wick_ratio = latest['lower_wick'] / (latest['total_range'] + 1e-10)
            
            # Wick significance
            avg_upper_wick = df['upper_wick'].rolling(window=5).mean().iloc[-1]
            avg_lower_wick = df['lower_wick'].rolling(window=5).mean().iloc[-1]
            
            upper_wick_significance = latest['upper_wick'] / (avg_upper_wick + 1e-10)
            lower_wick_significance = latest['lower_wick'] / (avg_lower_wick + 1e-10)
            
            # Smart rejection analysis
            smart_rejection_upper = upper_wick_ratio > 0.5 and upper_wick_significance > 2.0
            smart_rejection_lower = lower_wick_ratio > 0.5 and lower_wick_significance > 2.0
            
            return {
                'upper_wick_ratio': upper_wick_ratio,
                'lower_wick_ratio': lower_wick_ratio,
                'upper_wick_significance': upper_wick_significance,
                'lower_wick_significance': lower_wick_significance,
                'smart_rejection_upper': smart_rejection_upper,
                'smart_rejection_lower': smart_rejection_lower,
                'wick_balance': upper_wick_ratio / (lower_wick_ratio + 1e-10),
                'rejection_pressure': 'upper' if smart_rejection_upper else 'lower' if smart_rejection_lower else 'none'
            }
            
        except Exception as e:
            self.logger.error(f"Error in wick analysis: {e}")
            return {}
    
    async def _analyze_sizes(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze candle size patterns and trends"""
        try:
            # Size progression analysis
            sizes = df['total_range'].values
            size_trend = np.polyfit(range(len(sizes)), sizes, 1)[0]
            
            # Size volatility
            size_volatility = np.std(sizes) / (np.mean(sizes) + 1e-10)
            
            # Size momentum
            recent_avg = np.mean(sizes[-3:])
            historical_avg = np.mean(sizes[:-3]) if len(sizes) > 3 else recent_avg
            size_momentum = recent_avg / (historical_avg + 1e-10)
            
            return {
                'size_trend': 'expanding' if size_trend > 0 else 'contracting',
                'size_volatility': size_volatility,
                'size_momentum': size_momentum,
                'is_size_breakout': size_momentum > 1.5,
                'is_size_compression': size_momentum < 0.7
            }
            
        except Exception as e:
            self.logger.error(f"Error in size analysis: {e}")
            return {}
    
    async def _analyze_color_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze color sequences and patterns"""
        try:
            colors = df['is_bullish'].values
            
            # Color streaks
            current_streak = 1
            for i in range(len(colors) - 2, -1, -1):
                if colors[i] == colors[-1]:
                    current_streak += 1
                else:
                    break
            
            # Color alternation patterns
            alternations = sum(1 for i in range(1, len(colors)) if colors[i] != colors[i-1])
            alternation_ratio = alternations / (len(colors) - 1) if len(colors) > 1 else 0
            
            # Recent color distribution
            recent_colors = colors[-5:] if len(colors) >= 5 else colors
            bullish_ratio = sum(recent_colors) / len(recent_colors)
            
            return {
                'current_color': 'bullish' if colors[-1] else 'bearish',
                'streak_length': current_streak,
                'alternation_ratio': alternation_ratio,
                'recent_bullish_ratio': bullish_ratio,
                'color_pattern': self._classify_color_pattern(colors[-5:] if len(colors) >= 5 else colors)
            }
            
        except Exception as e:
            self.logger.error(f"Error in color pattern analysis: {e}")
            return {}
    
    async def _analyze_spacing(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze spacing and gaps between candles"""
        try:
            # Calculate gaps
            gaps = []
            for i in range(1, len(df)):
                if df.iloc[i]['open'] > df.iloc[i-1]['high']:
                    gaps.append(df.iloc[i]['open'] - df.iloc[i-1]['high'])
                elif df.iloc[i]['open'] < df.iloc[i-1]['low']:
                    gaps.append(df.iloc[i-1]['low'] - df.iloc[i]['open'])
                else:
                    gaps.append(0)
            
            if not gaps:
                return {}
            
            # Spacing analysis
            avg_gap = np.mean([abs(g) for g in gaps])
            latest_gap = gaps[-1] if gaps else 0
            
            return {
                'has_recent_gap': abs(latest_gap) > avg_gap * 2,
                'gap_direction': 'up' if latest_gap > 0 else 'down' if latest_gap < 0 else 'none',
                'gap_significance': abs(latest_gap) / (avg_gap + 1e-10),
                'avg_gap_size': avg_gap
            }
            
        except Exception as e:
            self.logger.error(f"Error in spacing analysis: {e}")
            return {}
    
    async def _analyze_volume_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume patterns (real or synthetic)"""
        try:
            if 'volume' not in df.columns:
                # Create synthetic volume based on range and body
                df['volume'] = df['total_range'] * df['body_ratio'] * 1000
            
            volumes = df['volume'].values
            
            # Volume trend
            volume_trend = np.polyfit(range(len(volumes)), volumes, 1)[0]
            
            # Volume momentum
            recent_vol = np.mean(volumes[-3:])
            historical_vol = np.mean(volumes[:-3]) if len(volumes) > 3 else recent_vol
            volume_momentum = recent_vol / (historical_vol + 1e-10)
            
            # Volume-price correlation
            price_changes = np.diff(df['close'].values)
            volume_changes = np.diff(volumes)
            
            if len(price_changes) > 1 and len(volume_changes) > 1:
                vol_price_corr = np.corrcoef(price_changes[:-1], volume_changes[1:])[0, 1]
            else:
                vol_price_corr = 0
            
            return {
                'volume_trend': 'rising' if volume_trend > 0 else 'falling',
                'volume_momentum': volume_momentum,
                'volume_spike': volume_momentum > 2.0,
                'volume_dry_up': volume_momentum < 0.5,
                'volume_price_correlation': vol_price_corr
            }
            
        except Exception as e:
            self.logger.error(f"Error in volume analysis: {e}")
            return {}
    
    async def _detect_pressure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect buying/selling pressure"""
        try:
            latest = df.iloc[-1]
            
            # Pressure indicators
            upper_pressure = latest['upper_wick'] / (latest['total_range'] + 1e-10)
            lower_pressure = latest['lower_wick'] / (latest['total_range'] + 1e-10)
            
            # Body pressure
            if latest['is_bullish']:
                buying_pressure = latest['body_ratio']
                selling_pressure = upper_pressure
            else:
                selling_pressure = latest['body_ratio']
                buying_pressure = lower_pressure
            
            # Overall pressure assessment
            net_pressure = buying_pressure - selling_pressure
            
            return {
                'buying_pressure': buying_pressure,
                'selling_pressure': selling_pressure,
                'net_pressure': net_pressure,
                'pressure_direction': 'bullish' if net_pressure > 0.2 else 'bearish' if net_pressure < -0.2 else 'neutral'
            }
            
        except Exception as e:
            self.logger.error(f"Error in pressure detection: {e}")
            return {}
    
    async def _detect_momentum_shifts(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect momentum shifts in candle patterns"""
        try:
            if len(df) < 3:
                return {}
            
            # Calculate momentum indicators
            closes = df['close'].values
            momentum = []
            
            for i in range(2, len(closes)):
                short_momentum = closes[i] - closes[i-1]
                long_momentum = closes[i] - closes[i-2]
                momentum.append({'short': short_momentum, 'long': long_momentum})
            
            if not momentum:
                return {}
            
            latest_momentum = momentum[-1]
            
            # Momentum shift detection
            momentum_shift = False
            if len(momentum) >= 2:
                prev_momentum = momentum[-2]
                if (latest_momentum['short'] * prev_momentum['short'] < 0 or
                    latest_momentum['long'] * prev_momentum['long'] < 0):
                    momentum_shift = True
            
            return {
                'short_momentum': latest_momentum['short'],
                'long_momentum': latest_momentum['long'],
                'momentum_shift_detected': momentum_shift,
                'momentum_direction': 'bullish' if latest_momentum['short'] > 0 else 'bearish'
            }
            
        except Exception as e:
            self.logger.error(f"Error in momentum shift detection: {e}")
            return {}
    
    async def _detect_engulfing(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect engulfing patterns"""
        try:
            if len(df) < 2:
                return {'engulfing_detected': False}
            
            current = df.iloc[-1]
            previous = df.iloc[-2]
            
            # Bullish engulfing
            bullish_engulfing = (
                not previous['is_bullish'] and current['is_bullish'] and
                current['open'] < previous['close'] and
                current['close'] > previous['open']
            )
            
            # Bearish engulfing
            bearish_engulfing = (
                previous['is_bullish'] and not current['is_bullish'] and
                current['open'] > previous['close'] and
                current['close'] < previous['open']
            )
            
            return {
                'engulfing_detected': bullish_engulfing or bearish_engulfing,
                'engulfing_type': 'bullish' if bullish_engulfing else 'bearish' if bearish_engulfing else 'none',
                'engulfing_strength': current['body_size'] / (previous['body_size'] + 1e-10)
            }
            
        except Exception as e:
            self.logger.error(f"Error in engulfing detection: {e}")
            return {'engulfing_detected': False}
    
    async def _detect_doji_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect doji patterns"""
        try:
            latest = df.iloc[-1]
            
            # Doji criteria
            is_doji = latest['body_ratio'] < 0.1
            
            # Doji types
            doji_type = 'none'
            if is_doji:
                upper_wick_ratio = latest['upper_wick'] / (latest['total_range'] + 1e-10)
                lower_wick_ratio = latest['lower_wick'] / (latest['total_range'] + 1e-10)
                
                if upper_wick_ratio > 0.4 and lower_wick_ratio < 0.2:
                    doji_type = 'dragonfly'
                elif lower_wick_ratio > 0.4 and upper_wick_ratio < 0.2:
                    doji_type = 'gravestone'
                elif upper_wick_ratio > 0.3 and lower_wick_ratio > 0.3:
                    doji_type = 'long_legged'
                else:
                    doji_type = 'standard'
            
            return {
                'is_doji': is_doji,
                'doji_type': doji_type,
                'doji_significance': 1 - latest['body_ratio'] if is_doji else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error in doji detection: {e}")
            return {'is_doji': False}
    
    async def _detect_hammer_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect hammer patterns"""
        try:
            latest = df.iloc[-1]
            
            # Hammer criteria
            lower_wick_ratio = latest['lower_wick'] / (latest['total_range'] + 1e-10)
            upper_wick_ratio = latest['upper_wick'] / (latest['total_range'] + 1e-10)
            body_ratio = latest['body_ratio']
            
            is_hammer = (
                lower_wick_ratio > 0.5 and
                upper_wick_ratio < 0.2 and
                body_ratio > 0.1 and body_ratio < 0.4
            )
            
            return {
                'is_hammer': is_hammer,
                'hammer_strength': lower_wick_ratio if is_hammer else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error in hammer detection: {e}")
            return {'is_hammer': False}
    
    async def _detect_shooting_star(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect shooting star patterns"""
        try:
            latest = df.iloc[-1]
            
            # Shooting star criteria
            upper_wick_ratio = latest['upper_wick'] / (latest['total_range'] + 1e-10)
            lower_wick_ratio = latest['lower_wick'] / (latest['total_range'] + 1e-10)
            body_ratio = latest['body_ratio']
            
            is_shooting_star = (
                upper_wick_ratio > 0.5 and
                lower_wick_ratio < 0.2 and
                body_ratio > 0.1 and body_ratio < 0.4
            )
            
            return {
                'is_shooting_star': is_shooting_star,
                'shooting_star_strength': upper_wick_ratio if is_shooting_star else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error in shooting star detection: {e}")
            return {'is_shooting_star': False}
    
    async def _detect_spinning_tops(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect spinning top patterns"""
        try:
            latest = df.iloc[-1]
            
            # Spinning top criteria
            upper_wick_ratio = latest['upper_wick'] / (latest['total_range'] + 1e-10)
            lower_wick_ratio = latest['lower_wick'] / (latest['total_range'] + 1e-10)
            body_ratio = latest['body_ratio']
            
            is_spinning_top = (
                upper_wick_ratio > 0.25 and lower_wick_ratio > 0.25 and
                body_ratio < 0.3 and body_ratio > 0.05
            )
            
            return {
                'is_spinning_top': is_spinning_top,
                'spinning_top_balance': min(upper_wick_ratio, lower_wick_ratio) if is_spinning_top else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error in spinning top detection: {e}")
            return {'is_spinning_top': False}
    
    async def _calculate_synthetic_volume(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate synthetic volume when real volume is not available"""
        try:
            # Synthetic volume based on price action
            synthetic_vol = df['total_range'] * (1 + df['body_ratio']) * 1000
            
            # Add volatility component
            price_volatility = df['close'].pct_change().rolling(window=3).std().fillna(0)
            synthetic_vol *= (1 + price_volatility * 10)
            
            return {
                'synthetic_volume': synthetic_vol.tolist(),
                'volume_trend': 'rising' if synthetic_vol.iloc[-1] > synthetic_vol.iloc[-3:-1].mean() else 'falling'
            }
            
        except Exception as e:
            self.logger.error(f"Error in synthetic volume calculation: {e}")
            return {}
    
    def _classify_color_pattern(self, colors: np.ndarray) -> str:
        """Classify color patterns"""
        if len(colors) < 3:
            return 'insufficient_data'
        
        # Pattern classification
        if all(colors):
            return 'all_bullish'
        elif not any(colors):
            return 'all_bearish'
        elif len(set(colors)) == 2:
            alternations = sum(1 for i in range(1, len(colors)) if colors[i] != colors[i-1])
            if alternations >= len(colors) - 2:
                return 'alternating'
            else:
                return 'mixed'
        else:
            return 'random'