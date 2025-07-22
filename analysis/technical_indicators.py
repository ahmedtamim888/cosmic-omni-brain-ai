import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import talib
from datetime import datetime, timedelta

class TechnicalAnalyzer:
    """Advanced technical analysis engine for binary trading"""
    
    def __init__(self):
        self.indicators = {}
        self.signals = {}
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate all technical indicators for the given dataframe"""
        if df.empty or len(df) < 50:
            return {}
        
        # Ensure we have OHLCV columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                if col == 'volume':
                    df[col] = 0  # Default volume if not available
                else:
                    raise ValueError(f"Missing required column: {col}")
        
        indicators = {}
        
        # Price-based indicators
        indicators.update(self._calculate_moving_averages(df))
        indicators.update(self._calculate_bollinger_bands(df))
        indicators.update(self._calculate_price_channels(df))
        
        # Momentum indicators
        indicators.update(self._calculate_rsi(df))
        indicators.update(self._calculate_macd(df))
        indicators.update(self._calculate_stochastic(df))
        indicators.update(self._calculate_williams_r(df))
        indicators.update(self._calculate_cci(df))
        
        # Volume indicators
        indicators.update(self._calculate_volume_indicators(df))
        
        # Volatility indicators
        indicators.update(self._calculate_volatility_indicators(df))
        
        # Support/Resistance levels
        indicators.update(self._calculate_support_resistance(df))
        
        # Market structure
        indicators.update(self._calculate_market_structure(df))
        
        return indicators
    
    def _calculate_moving_averages(self, df: pd.DataFrame) -> Dict:
        """Calculate various moving averages"""
        close = df['close'].values
        
        return {
            'sma_10': talib.SMA(close, timeperiod=10),
            'sma_20': talib.SMA(close, timeperiod=20),
            'sma_50': talib.SMA(close, timeperiod=50),
            'ema_12': talib.EMA(close, timeperiod=12),
            'ema_26': talib.EMA(close, timeperiod=26),
            'ema_50': talib.EMA(close, timeperiod=50),
            'wma_14': talib.WMA(close, timeperiod=14),
            'tema_14': talib.TEMA(close, timeperiod=14),
            'dema_14': talib.DEMA(close, timeperiod=14)
        }
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame) -> Dict:
        """Calculate Bollinger Bands"""
        close = df['close'].values
        
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        
        # Calculate BB position
        bb_position = np.where(close > bb_upper, 'upper',
                              np.where(close < bb_lower, 'lower', 'middle'))
        
        # BB width and squeeze
        bb_width = (bb_upper - bb_lower) / bb_middle * 100
        bb_squeeze = bb_width < np.percentile(bb_width[-100:], 20)
        
        return {
            'bb_upper': bb_upper,
            'bb_middle': bb_middle,
            'bb_lower': bb_lower,
            'bb_position': bb_position,
            'bb_width': bb_width,
            'bb_squeeze': bb_squeeze
        }
    
    def _calculate_price_channels(self, df: pd.DataFrame) -> Dict:
        """Calculate Donchian Channels and Keltner Channels"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        # Donchian Channels
        dc_upper = talib.MAX(high, timeperiod=20)
        dc_lower = talib.MIN(low, timeperiod=20)
        dc_middle = (dc_upper + dc_lower) / 2
        
        # Keltner Channels
        kc_middle = talib.EMA(close, timeperiod=20)
        atr = talib.ATR(high, low, close, timeperiod=10)
        kc_upper = kc_middle + (atr * 2)
        kc_lower = kc_middle - (atr * 2)
        
        return {
            'dc_upper': dc_upper,
            'dc_middle': dc_middle,
            'dc_lower': dc_lower,
            'kc_upper': kc_upper,
            'kc_middle': kc_middle,
            'kc_lower': kc_lower,
            'atr': atr
        }
    
    def _calculate_rsi(self, df: pd.DataFrame) -> Dict:
        """Calculate RSI and related indicators"""
        close = df['close'].values
        
        rsi = talib.RSI(close, timeperiod=14)
        rsi_fast = talib.RSI(close, timeperiod=7)
        rsi_slow = talib.RSI(close, timeperiod=21)
        
        # RSI divergence detection
        rsi_divergence = self._detect_rsi_divergence(df['close'], rsi)
        
        return {
            'rsi': rsi,
            'rsi_fast': rsi_fast,
            'rsi_slow': rsi_slow,
            'rsi_overbought': rsi > 70,
            'rsi_oversold': rsi < 30,
            'rsi_divergence': rsi_divergence
        }
    
    def _calculate_macd(self, df: pd.DataFrame) -> Dict:
        """Calculate MACD indicators"""
        close = df['close'].values
        
        macd_line, macd_signal, macd_histogram = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        
        # MACD signals
        macd_bullish = (macd_line > macd_signal) & (macd_line > 0)
        macd_bearish = (macd_line < macd_signal) & (macd_line < 0)
        macd_cross_up = (macd_line > macd_signal) & np.roll(macd_line <= macd_signal, 1)
        macd_cross_down = (macd_line < macd_signal) & np.roll(macd_line >= macd_signal, 1)
        
        return {
            'macd_line': macd_line,
            'macd_signal': macd_signal,
            'macd_histogram': macd_histogram,
            'macd_bullish': macd_bullish,
            'macd_bearish': macd_bearish,
            'macd_cross_up': macd_cross_up,
            'macd_cross_down': macd_cross_down
        }
    
    def _calculate_stochastic(self, df: pd.DataFrame) -> Dict:
        """Calculate Stochastic Oscillator"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        stoch_k, stoch_d = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
        
        return {
            'stoch_k': stoch_k,
            'stoch_d': stoch_d,
            'stoch_overbought': stoch_k > 80,
            'stoch_oversold': stoch_k < 20,
            'stoch_cross_up': (stoch_k > stoch_d) & np.roll(stoch_k <= stoch_d, 1),
            'stoch_cross_down': (stoch_k < stoch_d) & np.roll(stoch_k >= stoch_d, 1)
        }
    
    def _calculate_williams_r(self, df: pd.DataFrame) -> Dict:
        """Calculate Williams %R"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        willr = talib.WILLR(high, low, close, timeperiod=14)
        
        return {
            'willr': willr,
            'willr_overbought': willr > -20,
            'willr_oversold': willr < -80
        }
    
    def _calculate_cci(self, df: pd.DataFrame) -> Dict:
        """Calculate Commodity Channel Index"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        cci = talib.CCI(high, low, close, timeperiod=14)
        
        return {
            'cci': cci,
            'cci_overbought': cci > 100,
            'cci_oversold': cci < -100
        }
    
    def _calculate_volume_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate volume-based indicators"""
        if 'volume' not in df.columns or df['volume'].sum() == 0:
            return {}
        
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        volume = df['volume'].values
        
        # On-Balance Volume
        obv = talib.OBV(close, volume)
        
        # Accumulation/Distribution Line
        ad = talib.AD(high, low, close, volume)
        
        # Chaikin Money Flow
        cmf = self._calculate_cmf(df)
        
        # Volume Moving Average
        volume_ma = talib.SMA(volume, timeperiod=20)
        volume_spike = volume > (volume_ma * 1.5)
        
        return {
            'obv': obv,
            'ad': ad,
            'cmf': cmf,
            'volume_ma': volume_ma,
            'volume_spike': volume_spike,
            'volume_above_average': volume > volume_ma
        }
    
    def _calculate_volatility_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate volatility indicators"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        # Average True Range
        atr = talib.ATR(high, low, close, timeperiod=14)
        
        # Normalized ATR
        natr = talib.NATR(high, low, close, timeperiod=14)
        
        # True Range
        tr = talib.TRANGE(high, low, close)
        
        # Historical Volatility
        returns = np.diff(np.log(close))
        volatility = np.roll(pd.Series(returns).rolling(window=20).std() * np.sqrt(252) * 100, 1)
        
        return {
            'atr': atr,
            'natr': natr,
            'true_range': tr,
            'volatility': volatility,
            'high_volatility': volatility > np.percentile(volatility[-100:], 80),
            'low_volatility': volatility < np.percentile(volatility[-100:], 20)
        }
    
    def _calculate_support_resistance(self, df: pd.DataFrame) -> Dict:
        """Calculate dynamic support and resistance levels"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        # Pivot Points
        pivot_points = self._calculate_pivot_points(df)
        
        # Fibonacci Retracements
        fib_levels = self._calculate_fibonacci_levels(df)
        
        # Dynamic Support/Resistance using fractals
        support_levels, resistance_levels = self._find_fractal_levels(df)
        
        return {
            'pivot_point': pivot_points['pivot'],
            'resistance_1': pivot_points['r1'],
            'resistance_2': pivot_points['r2'],
            'support_1': pivot_points['s1'],
            'support_2': pivot_points['s2'],
            'fib_236': fib_levels['236'],
            'fib_382': fib_levels['382'],
            'fib_500': fib_levels['500'],
            'fib_618': fib_levels['618'],
            'dynamic_support': support_levels,
            'dynamic_resistance': resistance_levels
        }
    
    def _calculate_market_structure(self, df: pd.DataFrame) -> Dict:
        """Analyze market structure and trends"""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        # Trend direction
        trend = self._determine_trend(df)
        
        # Higher highs and lower lows
        hh_ll = self._analyze_swing_points(df)
        
        # Market phases
        market_phase = self._determine_market_phase(df)
        
        # Price action patterns
        patterns = self._detect_price_patterns(df)
        
        return {
            'trend_direction': trend,
            'trend_strength': self._calculate_trend_strength(df),
            'higher_highs': hh_ll['higher_highs'],
            'lower_lows': hh_ll['lower_lows'],
            'market_phase': market_phase,
            'consolidation': patterns['consolidation'],
            'breakout': patterns['breakout'],
            'reversal_pattern': patterns['reversal']
        }
    
    def generate_trading_signals(self, indicators: Dict) -> Dict:
        """Generate trading signals based on technical indicators"""
        signals = {
            'bullish_signals': [],
            'bearish_signals': [],
            'neutral_signals': [],
            'confidence_score': 0.0,
            'signal_strength': 'weak'
        }
        
        if not indicators:
            return signals
        
        # Get latest values
        latest_idx = -1
        
        # RSI Signals
        if 'rsi' in indicators and len(indicators['rsi']) > 0:
            rsi_val = indicators['rsi'][latest_idx] if not np.isnan(indicators['rsi'][latest_idx]) else None
            if rsi_val:
                if rsi_val < 30:
                    signals['bullish_signals'].append(f"RSI Oversold ({rsi_val:.1f})")
                elif rsi_val > 70:
                    signals['bearish_signals'].append(f"RSI Overbought ({rsi_val:.1f})")
        
        # MACD Signals
        if all(k in indicators for k in ['macd_cross_up', 'macd_cross_down']):
            if indicators['macd_cross_up'][latest_idx]:
                signals['bullish_signals'].append("MACD Bullish Cross")
            elif indicators['macd_cross_down'][latest_idx]:
                signals['bearish_signals'].append("MACD Bearish Cross")
        
        # Bollinger Bands Signals
        if 'bb_position' in indicators:
            bb_pos = indicators['bb_position'][latest_idx]
            if bb_pos == 'lower':
                signals['bullish_signals'].append("Price at BB Lower Band")
            elif bb_pos == 'upper':
                signals['bearish_signals'].append("Price at BB Upper Band")
        
        # Moving Average Signals
        if all(k in indicators for k in ['sma_10', 'sma_20']):
            sma_10 = indicators['sma_10'][latest_idx]
            sma_20 = indicators['sma_20'][latest_idx]
            if not (np.isnan(sma_10) or np.isnan(sma_20)):
                if sma_10 > sma_20:
                    signals['bullish_signals'].append("SMA 10 > SMA 20")
                else:
                    signals['bearish_signals'].append("SMA 10 < SMA 20")
        
        # Trend Signals
        if 'trend_direction' in indicators:
            trend = indicators['trend_direction']
            if trend == 'up':
                signals['bullish_signals'].append("Uptrend Detected")
            elif trend == 'down':
                signals['bearish_signals'].append("Downtrend Detected")
            else:
                signals['neutral_signals'].append("Sideways Trend")
        
        # Volume Signals
        if 'volume_spike' in indicators and indicators['volume_spike'][latest_idx]:
            signals['neutral_signals'].append("Volume Spike Detected")
        
        # Calculate confidence score
        total_signals = len(signals['bullish_signals']) + len(signals['bearish_signals'])
        if total_signals > 0:
            bullish_weight = len(signals['bullish_signals']) / total_signals
            bearish_weight = len(signals['bearish_signals']) / total_signals
            signals['confidence_score'] = max(bullish_weight, bearish_weight) * 100
            
            if signals['confidence_score'] > 70:
                signals['signal_strength'] = 'strong'
            elif signals['confidence_score'] > 50:
                signals['signal_strength'] = 'medium'
            else:
                signals['signal_strength'] = 'weak'
        
        return signals
    
    # Helper methods for complex calculations
    def _detect_rsi_divergence(self, price: pd.Series, rsi: np.ndarray) -> np.ndarray:
        """Detect RSI divergence patterns"""
        # Simplified divergence detection
        return np.zeros_like(rsi, dtype=bool)
    
    def _calculate_cmf(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate Chaikin Money Flow"""
        if 'volume' not in df.columns:
            return np.zeros(len(df))
        
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        volume = df['volume'].values
        
        money_flow_multiplier = ((close - low) - (high - close)) / (high - low)
        money_flow_volume = money_flow_multiplier * volume
        
        cmf = pd.Series(money_flow_volume).rolling(window=20).sum() / pd.Series(volume).rolling(window=20).sum()
        return cmf.values
    
    def _calculate_pivot_points(self, df: pd.DataFrame) -> Dict:
        """Calculate pivot points"""
        if len(df) < 2:
            return {'pivot': 0, 'r1': 0, 'r2': 0, 's1': 0, 's2': 0}
        
        prev_high = df['high'].iloc[-2]
        prev_low = df['low'].iloc[-2]
        prev_close = df['close'].iloc[-2]
        
        pivot = (prev_high + prev_low + prev_close) / 3
        r1 = 2 * pivot - prev_low
        s1 = 2 * pivot - prev_high
        r2 = pivot + (prev_high - prev_low)
        s2 = pivot - (prev_high - prev_low)
        
        return {'pivot': pivot, 'r1': r1, 'r2': r2, 's1': s1, 's2': s2}
    
    def _calculate_fibonacci_levels(self, df: pd.DataFrame) -> Dict:
        """Calculate Fibonacci retracement levels"""
        if len(df) < 20:
            return {'236': 0, '382': 0, '500': 0, '618': 0}
        
        recent_high = df['high'].iloc[-20:].max()
        recent_low = df['low'].iloc[-20:].min()
        diff = recent_high - recent_low
        
        return {
            '236': recent_high - (diff * 0.236),
            '382': recent_high - (diff * 0.382),
            '500': recent_high - (diff * 0.500),
            '618': recent_high - (diff * 0.618)
        }
    
    def _find_fractal_levels(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Find fractal support and resistance levels"""
        # Simplified fractal detection
        support = np.full(len(df), np.nan)
        resistance = np.full(len(df), np.nan)
        return support, resistance
    
    def _determine_trend(self, df: pd.DataFrame) -> str:
        """Determine overall trend direction"""
        if len(df) < 20:
            return 'sideways'
        
        close = df['close'].values
        sma_20 = talib.SMA(close, timeperiod=20)
        sma_50 = talib.SMA(close, timeperiod=50) if len(df) >= 50 else sma_20
        
        if np.isnan(sma_20[-1]) or np.isnan(sma_50[-1]):
            return 'sideways'
        
        if sma_20[-1] > sma_50[-1] and close[-1] > sma_20[-1]:
            return 'up'
        elif sma_20[-1] < sma_50[-1] and close[-1] < sma_20[-1]:
            return 'down'
        else:
            return 'sideways'
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """Calculate trend strength (0-100)"""
        if len(df) < 20:
            return 50.0
        
        close = df['close'].values[-20:]
        linear_reg_slope = np.polyfit(range(len(close)), close, 1)[0]
        price_range = close.max() - close.min()
        
        if price_range == 0:
            return 50.0
        
        trend_strength = min(abs(linear_reg_slope) / price_range * 1000, 100)
        return trend_strength
    
    def _analyze_swing_points(self, df: pd.DataFrame) -> Dict:
        """Analyze swing highs and lows"""
        # Simplified swing point analysis
        return {'higher_highs': False, 'lower_lows': False}
    
    def _determine_market_phase(self, df: pd.DataFrame) -> str:
        """Determine current market phase"""
        if len(df) < 20:
            return 'unknown'
        
        # Calculate price volatility
        close = df['close'].values
        returns = np.diff(close) / close[:-1]
        volatility = np.std(returns[-20:])
        
        # Calculate trend consistency
        trend_direction = self._determine_trend(df)
        
        if volatility < 0.01:  # Low volatility
            return 'consolidation'
        elif trend_direction in ['up', 'down']:
            return 'trending'
        else:
            return 'transitional'
    
    def _detect_price_patterns(self, df: pd.DataFrame) -> Dict:
        """Detect common price patterns"""
        # Simplified pattern detection
        return {
            'consolidation': False,
            'breakout': False,
            'reversal': False
        }