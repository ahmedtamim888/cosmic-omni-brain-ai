"""
Support/Resistance Detection Engine
Advanced S/R level detection using price clustering and pivot analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import asyncio
from datetime import datetime
import logging
from sklearn.cluster import DBSCAN
from scipy.signal import argrelextrema

class SupportResistanceDetector:
    """
    Advanced Support/Resistance level detection and analysis
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.sr_levels = []
        self.level_history = []
        self.rejection_memory = []
        
    async def detect_levels(self, candles: List[Dict]) -> List[Dict]:
        """
        Main S/R level detection function
        """
        try:
            if len(candles) < 10:
                return []
            
            df = pd.DataFrame(candles)
            
            # Multiple detection methods
            pivot_levels = await self._detect_pivot_levels(df)
            cluster_levels = await self._detect_cluster_levels(df)
            psychological_levels = await self._detect_psychological_levels(df)
            volume_levels = await self._detect_volume_levels(df)
            
            # Combine and validate levels
            all_levels = pivot_levels + cluster_levels + psychological_levels + volume_levels
            validated_levels = await self._validate_levels(df, all_levels)
            
            # Update level history
            self._update_level_history(validated_levels)
            
            return validated_levels
            
        except Exception as e:
            self.logger.error(f"Error detecting S/R levels: {e}")
            return []
    
    async def analyze_rejections(self, candles: List[Dict], sr_levels: List[Dict]) -> Dict[str, Any]:
        """
        Analyze rejections from S/R levels
        """
        try:
            if not candles or not sr_levels:
                return {}
            
            df = pd.DataFrame(candles)
            latest_candle = df.iloc[-1]
            
            # Check for rejections
            rejection_analysis = {
                'rejection_detected': False,
                'rejection_type': 'none',
                'rejection_level': None,
                'rejection_strength': 0,
                'multiple_touches': False,
                'volume_confirmation': False,
                'wick_rejection': False
            }
            
            for level in sr_levels:
                rejection_result = await self._analyze_level_rejection(df, latest_candle, level)
                
                if rejection_result['rejected']:
                    rejection_analysis.update({
                        'rejection_detected': True,
                        'rejection_type': level['type'],
                        'rejection_level': level['price'],
                        'rejection_strength': rejection_result['strength'],
                        'multiple_touches': rejection_result['multiple_touches'],
                        'volume_confirmation': rejection_result['volume_confirmation'],
                        'wick_rejection': rejection_result['wick_rejection']
                    })
                    
                    # Store rejection in memory
                    self.rejection_memory.append({
                        'timestamp': datetime.now(),
                        'level': level['price'],
                        'type': level['type'],
                        'strength': rejection_result['strength']
                    })
                    
                    break  # Take first/strongest rejection
            
            # Clean old rejection memory
            self._clean_rejection_memory()
            
            return rejection_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing rejections: {e}")
            return {}
    
    async def _detect_pivot_levels(self, df: pd.DataFrame) -> List[Dict]:
        """Detect pivot-based S/R levels"""
        try:
            levels = []
            
            # Find pivot highs and lows
            window = min(5, len(df) // 4)  # Adaptive window
            if window < 2:
                return levels
            
            # Pivot highs (resistance)
            high_pivots = argrelextrema(df['high'].values, np.greater, order=window)[0]
            
            for idx in high_pivots:
                if idx >= window and idx < len(df) - window:  # Valid pivot
                    pivot_high = df['high'].iloc[idx]
                    
                    # Check significance
                    surrounding_highs = df['high'].iloc[max(0, idx-window):idx+window+1]
                    if pivot_high == surrounding_highs.max():
                        
                        # Calculate strength based on touches
                        touches = self._count_touches(df, pivot_high, 'resistance')
                        
                        levels.append({
                            'price': pivot_high,
                            'type': 'resistance',
                            'method': 'pivot',
                            'strength': min(1.0, touches / 3.0),
                            'touches': touches,
                            'age': len(df) - idx,
                            'fresh': touches <= 2
                        })
            
            # Pivot lows (support)
            low_pivots = argrelextrema(df['low'].values, np.less, order=window)[0]
            
            for idx in low_pivots:
                if idx >= window and idx < len(df) - window:  # Valid pivot
                    pivot_low = df['low'].iloc[idx]
                    
                    # Check significance
                    surrounding_lows = df['low'].iloc[max(0, idx-window):idx+window+1]
                    if pivot_low == surrounding_lows.min():
                        
                        # Calculate strength based on touches
                        touches = self._count_touches(df, pivot_low, 'support')
                        
                        levels.append({
                            'price': pivot_low,
                            'type': 'support',
                            'method': 'pivot',
                            'strength': min(1.0, touches / 3.0),
                            'touches': touches,
                            'age': len(df) - idx,
                            'fresh': touches <= 2
                        })
            
            return levels
            
        except Exception as e:
            self.logger.error(f"Error detecting pivot levels: {e}")
            return []
    
    async def _detect_cluster_levels(self, df: pd.DataFrame) -> List[Dict]:
        """Detect S/R levels using price clustering"""
        try:
            levels = []
            
            if len(df) < 20:
                return levels
            
            # Combine significant prices (highs, lows, closes)
            significant_prices = []
            significant_prices.extend(df['high'].tolist())
            significant_prices.extend(df['low'].tolist())
            significant_prices.extend(df['close'].tolist())
            
            # Convert to numpy array for clustering
            prices_array = np.array(significant_prices).reshape(-1, 1)
            
            # Use DBSCAN for clustering
            # eps is the maximum distance between points in the same cluster
            price_range = df['high'].max() - df['low'].min()
            eps = price_range * 0.002  # 0.2% of total range
            
            clustering = DBSCAN(eps=eps, min_samples=3).fit(prices_array)
            
            # Extract cluster centers
            unique_labels = set(clustering.labels_)
            
            for label in unique_labels:
                if label == -1:  # Noise points
                    continue
                
                cluster_points = prices_array[clustering.labels_ == label]
                cluster_center = np.mean(cluster_points)
                cluster_size = len(cluster_points)
                
                # Determine if support or resistance
                current_price = df['close'].iloc[-1]
                level_type = 'support' if cluster_center < current_price else 'resistance'
                
                # Calculate strength based on cluster size and recency
                base_strength = min(1.0, cluster_size / 10.0)
                
                # Check for recent touches
                recent_touches = self._count_recent_touches(df, cluster_center, level_type)
                
                levels.append({
                    'price': float(cluster_center),
                    'type': level_type,
                    'method': 'cluster',
                    'strength': base_strength,
                    'touches': cluster_size,
                    'recent_touches': recent_touches,
                    'age': 0,  # Cluster age is dynamic
                    'fresh': recent_touches <= 1
                })
            
            return levels
            
        except Exception as e:
            self.logger.error(f"Error detecting cluster levels: {e}")
            return []
    
    async def _detect_psychological_levels(self, df: pd.DataFrame) -> List[Dict]:
        """Detect psychological S/R levels (round numbers)"""
        try:
            levels = []
            
            current_price = df['close'].iloc[-1]
            price_range = df['high'].max() - df['low'].min()
            
            # Determine significant digits based on price
            if current_price >= 1000:
                step = 100  # Round hundreds
            elif current_price >= 100:
                step = 50   # Round fifties
            elif current_price >= 10:
                step = 5    # Round fives
            elif current_price >= 1:
                step = 0.5  # Round half units
            else:
                step = 0.1  # Round tenths
            
            # Find psychological levels within range
            min_price = df['low'].min()
            max_price = df['high'].max()
            
            # Generate round number levels
            start_level = int(min_price / step) * step
            current_level = start_level
            
            while current_level <= max_price:
                if min_price <= current_level <= max_price:
                    # Check if this level has been tested
                    touches = self._count_touches(df, current_level, 'both')
                    
                    if touches > 0:  # Only include tested levels
                        level_type = 'support' if current_level < current_price else 'resistance'
                        
                        levels.append({
                            'price': current_level,
                            'type': level_type,
                            'method': 'psychological',
                            'strength': min(1.0, touches / 2.0),
                            'touches': touches,
                            'age': 0,  # Psychological levels are timeless
                            'fresh': touches <= 1
                        })
                
                current_level += step
            
            return levels
            
        except Exception as e:
            self.logger.error(f"Error detecting psychological levels: {e}")
            return []
    
    async def _detect_volume_levels(self, df: pd.DataFrame) -> List[Dict]:
        """Detect S/R levels based on volume profile"""
        try:
            levels = []
            
            # Create synthetic volume if not available
            if 'volume' not in df.columns:
                df['volume'] = (df['high'] - df['low']) * abs(df['close'] - df['open']) * 1000
            
            # Create price bins for volume profile
            min_price = df['low'].min()
            max_price = df['high'].max()
            price_range = max_price - min_price
            
            num_bins = min(20, len(df))  # Adaptive number of bins
            bin_size = price_range / num_bins
            
            # Calculate volume at each price level
            volume_profile = {}
            
            for _, candle in df.iterrows():
                # Distribute volume across the candle's range
                candle_range = candle['high'] - candle['low']
                if candle_range > 0:
                    volume_per_tick = candle['volume'] / candle_range
                    
                    # Find bins this candle touches
                    start_bin = int((candle['low'] - min_price) / bin_size)
                    end_bin = int((candle['high'] - min_price) / bin_size)
                    
                    for bin_idx in range(start_bin, end_bin + 1):
                        bin_price = min_price + (bin_idx * bin_size)
                        
                        if bin_price not in volume_profile:
                            volume_profile[bin_price] = 0
                        volume_profile[bin_price] += volume_per_tick
            
            # Find high volume areas (potential S/R levels)
            if volume_profile:
                avg_volume = np.mean(list(volume_profile.values()))
                high_volume_threshold = avg_volume * 1.5
                
                current_price = df['close'].iloc[-1]
                
                for price, volume in volume_profile.items():
                    if volume >= high_volume_threshold:
                        level_type = 'support' if price < current_price else 'resistance'
                        
                        # Check actual touches at this level
                        touches = self._count_touches(df, price, level_type)
                        
                        if touches > 0:
                            strength = min(1.0, (volume / avg_volume) / 3.0)
                            
                            levels.append({
                                'price': price,
                                'type': level_type,
                                'method': 'volume',
                                'strength': strength,
                                'touches': touches,
                                'volume': volume,
                                'age': 0,
                                'fresh': touches <= 2
                            })
            
            return levels
            
        except Exception as e:
            self.logger.error(f"Error detecting volume levels: {e}")
            return []
    
    async def _validate_levels(self, df: pd.DataFrame, levels: List[Dict]) -> List[Dict]:
        """Validate and filter S/R levels"""
        try:
            if not levels:
                return []
            
            current_price = df['close'].iloc[-1]
            price_range = df['high'].max() - df['low'].min()
            
            validated_levels = []
            
            # Sort levels by strength (descending)
            levels.sort(key=lambda x: x['strength'], reverse=True)
            
            for level in levels:
                level_price = level['price']
                
                # Filter criteria
                valid = True
                
                # 1. Must be within reasonable range
                distance_from_current = abs(level_price - current_price) / current_price
                if distance_from_current > 0.1:  # More than 10% away
                    valid = False
                
                # 2. Must have minimum strength
                if level['strength'] < 0.2:
                    valid = False
                
                # 3. Check for duplicate levels (merge close levels)
                for existing_level in validated_levels:
                    if abs(level_price - existing_level['price']) < (price_range * 0.001):
                        # Merge levels - keep stronger one
                        if level['strength'] > existing_level['strength']:
                            validated_levels.remove(existing_level)
                        else:
                            valid = False
                        break
                
                if valid:
                    validated_levels.append(level)
            
            # Limit to top levels
            validated_levels = validated_levels[:10]  # Keep top 10 levels
            
            return validated_levels
            
        except Exception as e:
            self.logger.error(f"Error validating levels: {e}")
            return levels  # Return original if validation fails
    
    async def _analyze_level_rejection(self, df: pd.DataFrame, latest_candle: pd.Series, 
                                     level: Dict) -> Dict[str, Any]:
        """Analyze if current candle shows rejection from a level"""
        try:
            level_price = level['price']
            level_type = level['type']
            
            # Tolerance for level testing
            tolerance = abs(latest_candle['close']) * 0.001  # 0.1% tolerance
            
            rejection_result = {
                'rejected': False,
                'strength': 0,
                'multiple_touches': False,
                'volume_confirmation': False,
                'wick_rejection': False
            }
            
            # Check if price tested the level
            if level_type == 'resistance':
                tested_level = latest_candle['high'] >= (level_price - tolerance)
                rejection = tested_level and latest_candle['close'] < (level_price - tolerance)
            else:  # support
                tested_level = latest_candle['low'] <= (level_price + tolerance)
                rejection = tested_level and latest_candle['close'] > (level_price + tolerance)
            
            if not rejection:
                return rejection_result
            
            # Calculate rejection strength
            if level_type == 'resistance':
                # Wick rejection strength
                upper_wick = latest_candle['high'] - max(latest_candle['open'], latest_candle['close'])
                total_range = latest_candle['high'] - latest_candle['low']
                wick_ratio = upper_wick / (total_range + 1e-10)
                
                # Distance from level
                distance_ratio = (level_price - latest_candle['close']) / (level_price + 1e-10)
                
                rejection_result['wick_rejection'] = wick_ratio > 0.5
                rejection_result['strength'] = min(1.0, (wick_ratio + distance_ratio))
                
            else:  # support
                # Wick rejection strength
                lower_wick = min(latest_candle['open'], latest_candle['close']) - latest_candle['low']
                total_range = latest_candle['high'] - latest_candle['low']
                wick_ratio = lower_wick / (total_range + 1e-10)
                
                # Distance from level
                distance_ratio = (latest_candle['close'] - level_price) / (level_price + 1e-10)
                
                rejection_result['wick_rejection'] = wick_ratio > 0.5
                rejection_result['strength'] = min(1.0, (wick_ratio + distance_ratio))
            
            # Check for multiple touches
            recent_touches = self._count_recent_touches(df, level_price, level_type, periods=5)
            rejection_result['multiple_touches'] = recent_touches >= 2
            
            # Volume confirmation
            if 'volume' in df.columns or 'synthetic_volume' in df.columns:
                volume_col = 'volume' if 'volume' in df.columns else 'synthetic_volume'
                current_volume = df[volume_col].iloc[-1] if volume_col in df.columns else 1000
                avg_volume = df[volume_col].iloc[-10:].mean() if volume_col in df.columns else 1000
                
                rejection_result['volume_confirmation'] = current_volume > avg_volume * 1.2
            
            # Final rejection decision
            rejection_result['rejected'] = True
            
            # Boost strength for fresh levels
            if level.get('fresh', False):
                rejection_result['strength'] *= 1.2
            
            # Boost strength for multiple confirmations
            if rejection_result['multiple_touches']:
                rejection_result['strength'] *= 1.1
            
            if rejection_result['volume_confirmation']:
                rejection_result['strength'] *= 1.1
            
            rejection_result['strength'] = min(1.0, rejection_result['strength'])
            
            return rejection_result
            
        except Exception as e:
            self.logger.error(f"Error analyzing level rejection: {e}")
            return {'rejected': False, 'strength': 0}
    
    def _count_touches(self, df: pd.DataFrame, level_price: float, level_type: str) -> int:
        """Count how many times a level has been touched"""
        try:
            tolerance = abs(level_price) * 0.002  # 0.2% tolerance
            touches = 0
            
            for _, candle in df.iterrows():
                if level_type == 'resistance' or level_type == 'both':
                    if abs(candle['high'] - level_price) <= tolerance:
                        touches += 1
                
                if level_type == 'support' or level_type == 'both':
                    if abs(candle['low'] - level_price) <= tolerance:
                        touches += 1
            
            return touches
            
        except Exception as e:
            return 0
    
    def _count_recent_touches(self, df: pd.DataFrame, level_price: float, 
                            level_type: str, periods: int = 10) -> int:
        """Count recent touches to a level"""
        try:
            if len(df) < periods:
                recent_df = df
            else:
                recent_df = df.iloc[-periods:]
            
            return self._count_touches(recent_df, level_price, level_type)
            
        except Exception as e:
            return 0
    
    def _update_level_history(self, levels: List[Dict]):
        """Update level history for tracking"""
        try:
            current_time = datetime.now()
            
            for level in levels:
                level['timestamp'] = current_time
            
            self.level_history.extend(levels)
            
            # Keep only recent history (last 1000 levels)
            if len(self.level_history) > 1000:
                self.level_history = self.level_history[-1000:]
                
        except Exception as e:
            self.logger.error(f"Error updating level history: {e}")
    
    def _clean_rejection_memory(self):
        """Clean old rejections from memory"""
        try:
            current_time = datetime.now()
            
            # Keep only rejections from last 24 hours
            self.rejection_memory = [
                rejection for rejection in self.rejection_memory
                if (current_time - rejection['timestamp']).total_seconds() < 86400
            ]
            
        except Exception as e:
            self.logger.error(f"Error cleaning rejection memory: {e}")
    
    def get_level_statistics(self) -> Dict[str, Any]:
        """Get statistics about detected levels"""
        try:
            if not self.level_history:
                return {}
            
            recent_levels = [level for level in self.level_history 
                           if (datetime.now() - level['timestamp']).total_seconds() < 3600]
            
            stats = {
                'total_levels_detected': len(self.level_history),
                'recent_levels': len(recent_levels),
                'recent_rejections': len(self.rejection_memory),
                'average_level_strength': np.mean([level['strength'] for level in recent_levels]) if recent_levels else 0,
                'fresh_levels_ratio': sum(1 for level in recent_levels if level.get('fresh', False)) / len(recent_levels) if recent_levels else 0
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error calculating level statistics: {e}")
            return {}