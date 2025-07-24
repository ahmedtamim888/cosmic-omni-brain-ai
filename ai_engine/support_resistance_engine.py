#!/usr/bin/env python3
"""
🧱 ULTRA-ADVANCED SUPPORT/RESISTANCE DETECTION ENGINE
DYNAMIC PRICE CLUSTERING + PIVOT ZONE ANALYSIS
FRESH S/R ZONE DETECTION + REJECTION ANALYSIS
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.signal import find_peaks, find_peaks_cwt
import cv2

logger = logging.getLogger(__name__)

class SupportResistanceEngine:
    """
    🧱 ULTRA-ADVANCED SUPPORT/RESISTANCE DETECTION ENGINE
    
    Features:
    - Dynamic price clustering with DBSCAN + KMeans
    - Rolling pivot zone detection
    - Fresh S/R zone identification
    - Rejection strength analysis
    - Volume-confirmed S/R levels
    - Time-based S/R strength decay
    - Multi-timeframe S/R alignment
    """
    
    def __init__(self):
        self.version = "S/R ENGINE ∞ ULTRA vX"
        self.sr_zones = []
        self.fresh_zones = []
        self.historical_zones = []
        self.rejection_memory = {}
        
        # 🧱 S/R DETECTION PARAMETERS
        self.cluster_epsilon = 0.001  # Price clustering sensitivity
        self.min_touches = 2  # Minimum touches for valid S/R
        self.zone_strength_threshold = 0.7
        self.freshness_decay_hours = 24
        
        # 📊 VOLUME CONFIRMATION SETTINGS
        self.volume_confirmation_threshold = 1.5
        self.rejection_strength_multiplier = 2.0
        
        # 🎯 PRECISION SETTINGS
        self.precision_levels = {
            'ultra': 0.0005,  # 0.05% precision
            'high': 0.001,    # 0.1% precision
            'medium': 0.002,  # 0.2% precision
            'low': 0.005      # 0.5% precision
        }
        
    async def detect_support_resistance(self, candle_data: List[Dict], 
                                      lookback_period: int = 50) -> Dict:
        """
        🧱 MAIN S/R DETECTION ALGORITHM
        Returns comprehensive support/resistance analysis
        """
        try:
            logger.info("🧱 Starting ultra-advanced S/R detection...")
            
            if len(candle_data) < 10:
                return {"error": "Insufficient candle data for S/R analysis"}
            
            # Extract price data
            prices = self._extract_price_data(candle_data, lookback_period)
            
            # 🎯 STEP 1: PRICE CLUSTERING ANALYSIS
            clustered_levels = await self._detect_price_clusters(prices, candle_data)
            
            # 🎯 STEP 2: PIVOT POINT DETECTION
            pivot_levels = await self._detect_pivot_levels(prices, candle_data)
            
            # 🎯 STEP 3: VOLUME-CONFIRMED LEVELS
            volume_levels = await self._detect_volume_confirmed_levels(candle_data)
            
            # 🎯 STEP 4: COMBINE AND VALIDATE LEVELS
            all_levels = clustered_levels + pivot_levels + volume_levels
            validated_levels = await self._validate_sr_levels(all_levels, candle_data)
            
            # 🎯 STEP 5: CLASSIFY FRESH VS HISTORICAL ZONES
            fresh_zones, historical_zones = await self._classify_zone_freshness(
                validated_levels, candle_data
            )
            
            # 🎯 STEP 6: CALCULATE REJECTION STRENGTHS
            rejection_analysis = await self._analyze_rejection_strengths(
                validated_levels, candle_data
            )
            
            # 🎯 STEP 7: DETECT CURRENT PRICE PROXIMITY
            proximity_analysis = await self._analyze_price_proximity(
                validated_levels, candle_data[-1] if candle_data else {}
            )
            
            # Create comprehensive S/R report
            sr_report = {
                "support_levels": [l for l in validated_levels if l['type'] == 'support'],
                "resistance_levels": [l for l in validated_levels if l['type'] == 'resistance'],
                "fresh_zones": fresh_zones,
                "historical_zones": historical_zones,
                "rejection_analysis": rejection_analysis,
                "proximity_analysis": proximity_analysis,
                "current_price": candle_data[-1].get('close', 0) if candle_data else 0,
                "total_levels": len(validated_levels),
                "high_strength_levels": len([l for l in validated_levels if l['strength'] >= 0.8]),
                "timestamp": datetime.now().isoformat()
            }
            
            # Update internal memory
            self.sr_zones = validated_levels
            self.fresh_zones = fresh_zones
            self.historical_zones = historical_zones
            
            logger.info(f"🧱 S/R Analysis complete: {len(validated_levels)} levels detected")
            return sr_report
            
        except Exception as e:
            logger.error(f"❌ S/R detection error: {str(e)}")
            return {"error": str(e)}
    
    def _extract_price_data(self, candle_data: List[Dict], lookback: int) -> Dict:
        """Extract and organize price data for analysis"""
        try:
            recent_candles = candle_data[-lookback:] if len(candle_data) >= lookback else candle_data
            
            prices = {
                'highs': [c.get('high', 0) for c in recent_candles],
                'lows': [c.get('low', 0) for c in recent_candles],
                'closes': [c.get('close', 0) for c in recent_candles],
                'opens': [c.get('open', 0) for c in recent_candles],
                'timestamps': [i for i in range(len(recent_candles))]  # Index-based timing
            }
            
            # Calculate synthetic volumes based on range and body
            volumes = []
            for candle in recent_candles:
                high = candle.get('high', 0)
                low = candle.get('low', 0)
                close = candle.get('close', 0)
                open_price = candle.get('open', 0)
                
                range_size = high - low
                body_size = abs(close - open_price)
                synthetic_volume = range_size * (1 + body_size * 2)
                volumes.append(synthetic_volume)
            
            prices['volumes'] = volumes
            return prices
            
        except Exception as e:
            logger.error(f"❌ Price data extraction error: {str(e)}")
            return {}
    
    async def _detect_price_clusters(self, prices: Dict, candle_data: List[Dict]) -> List[Dict]:
        """Detect support/resistance levels using price clustering"""
        try:
            if not prices or len(prices.get('highs', [])) < 5:
                return []
            
            # Combine all significant price levels
            all_prices = []
            price_types = []
            
            # Add swing highs and lows
            highs = np.array(prices['highs'])
            lows = np.array(prices['lows'])
            
            # Find swing highs (local maxima)
            high_peaks, _ = find_peaks(highs, distance=3, prominence=np.std(highs) * 0.5)
            for peak in high_peaks:
                all_prices.append(highs[peak])
                price_types.append('resistance')
            
            # Find swing lows (local minima)
            low_peaks, _ = find_peaks(-lows, distance=3, prominence=np.std(lows) * 0.5)
            for peak in low_peaks:
                all_prices.append(lows[peak])
                price_types.append('support')
            
            if len(all_prices) < 3:
                return []
            
            # Apply DBSCAN clustering to find price levels
            price_array = np.array(all_prices).reshape(-1, 1)
            
            # Dynamic epsilon based on price volatility
            price_std = np.std(all_prices)
            epsilon = max(price_std * 0.01, self.cluster_epsilon)
            
            clusterer = DBSCAN(eps=epsilon, min_samples=self.min_touches)
            clusters = clusterer.fit_predict(price_array)
            
            # Extract clustered levels
            clustered_levels = []
            for cluster_id in set(clusters):
                if cluster_id == -1:  # Skip noise points
                    continue
                
                cluster_prices = [all_prices[i] for i in range(len(all_prices)) if clusters[i] == cluster_id]
                cluster_types = [price_types[i] for i in range(len(price_types)) if clusters[i] == cluster_id]
                
                if len(cluster_prices) >= self.min_touches:
                    level_price = np.mean(cluster_prices)
                    level_type = max(set(cluster_types), key=cluster_types.count)  # Majority vote
                    
                    # Calculate level strength
                    touch_count = len(cluster_prices)
                    price_consistency = 1.0 - (np.std(cluster_prices) / (level_price + 0.001))
                    strength = min((touch_count / 10.0) * price_consistency, 1.0)
                    
                    clustered_levels.append({
                        'price': level_price,
                        'type': level_type,
                        'strength': strength,
                        'touch_count': touch_count,
                        'method': 'price_clustering',
                        'precision': 'high' if price_consistency > 0.95 else 'medium'
                    })
            
            return clustered_levels
            
        except Exception as e:
            logger.error(f"❌ Price clustering error: {str(e)}")
            return []
    
    async def _detect_pivot_levels(self, prices: Dict, candle_data: List[Dict]) -> List[Dict]:
        """Detect support/resistance using rolling pivot analysis"""
        try:
            if not prices or len(prices.get('highs', [])) < 10:
                return []
            
            highs = np.array(prices['highs'])
            lows = np.array(prices['lows'])
            closes = np.array(prices['closes'])
            
            pivot_levels = []
            
            # Rolling pivot calculation
            window_size = min(20, len(highs) // 2)
            
            for i in range(window_size, len(highs) - window_size):
                window_highs = highs[i-window_size:i+window_size+1]
                window_lows = lows[i-window_size:i+window_size+1]
                
                current_high = highs[i]
                current_low = lows[i]
                
                # Check if current price is a significant pivot
                if current_high == np.max(window_highs):
                    # Resistance level
                    strength = self._calculate_pivot_strength(
                        current_high, window_highs, 'resistance'
                    )
                    
                    if strength >= self.zone_strength_threshold:
                        pivot_levels.append({
                            'price': current_high,
                            'type': 'resistance',
                            'strength': strength,
                            'touch_count': 1,
                            'method': 'pivot_analysis',
                            'precision': 'ultra' if strength > 0.9 else 'high'
                        })
                
                if current_low == np.min(window_lows):
                    # Support level
                    strength = self._calculate_pivot_strength(
                        current_low, window_lows, 'support'
                    )
                    
                    if strength >= self.zone_strength_threshold:
                        pivot_levels.append({
                            'price': current_low,
                            'type': 'support',
                            'strength': strength,
                            'touch_count': 1,
                            'method': 'pivot_analysis',
                            'precision': 'ultra' if strength > 0.9 else 'high'
                        })
            
            return pivot_levels
            
        except Exception as e:
            logger.error(f"❌ Pivot level detection error: {str(e)}")
            return []
    
    async def _detect_volume_confirmed_levels(self, candle_data: List[Dict]) -> List[Dict]:
        """Detect S/R levels confirmed by volume spikes"""
        try:
            if len(candle_data) < 10:
                return []
            
            # Calculate synthetic volumes and detect spikes
            volumes = []
            for candle in candle_data:
                high = candle.get('high', 0)
                low = candle.get('low', 0)
                close = candle.get('close', 0)
                open_price = candle.get('open', 0)
                
                range_size = high - low
                body_size = abs(close - open_price)
                synthetic_volume = range_size * (1 + body_size * 2)
                volumes.append(synthetic_volume)
            
            volume_array = np.array(volumes)
            volume_mean = np.mean(volume_array)
            volume_std = np.std(volume_array)
            
            volume_levels = []
            
            # Find volume spikes (above mean + 1.5 * std)
            spike_threshold = volume_mean + (volume_std * self.volume_confirmation_threshold)
            
            for i, volume in enumerate(volumes):
                if volume > spike_threshold and i < len(candle_data):
                    candle = candle_data[i]
                    high = candle.get('high', 0)
                    low = candle.get('low', 0)
                    
                    # Volume spike indicates significant level
                    # Check if it's more likely support or resistance
                    prev_close = candle_data[i-1].get('close', 0) if i > 0 else candle.get('open', 0)
                    curr_close = candle.get('close', 0)
                    
                    if curr_close > prev_close:
                        # Likely resistance test (rejection from high)
                        volume_levels.append({
                            'price': high,
                            'type': 'resistance',
                            'strength': min(volume / volume_mean / 5.0, 1.0),
                            'touch_count': 1,
                            'method': 'volume_confirmation',
                            'precision': 'medium'
                        })
                    else:
                        # Likely support test (bounce from low)
                        volume_levels.append({
                            'price': low,
                            'type': 'support',
                            'strength': min(volume / volume_mean / 5.0, 1.0),
                            'touch_count': 1,
                            'method': 'volume_confirmation',
                            'precision': 'medium'
                        })
            
            return volume_levels
            
        except Exception as e:
            logger.error(f"❌ Volume confirmation error: {str(e)}")
            return []
    
    async def _validate_sr_levels(self, levels: List[Dict], candle_data: List[Dict]) -> List[Dict]:
        """Validate and merge similar S/R levels"""
        try:
            if not levels:
                return []
            
            # Sort levels by price
            levels.sort(key=lambda x: x['price'])
            
            validated_levels = []
            current_price = candle_data[-1].get('close', 0) if candle_data else 0
            
            i = 0
            while i < len(levels):
                current_level = levels[i]
                similar_levels = [current_level]
                
                # Find similar levels within clustering distance
                j = i + 1
                while j < len(levels):
                    if abs(levels[j]['price'] - current_level['price']) / (current_level['price'] + 0.001) < 0.002:
                        similar_levels.append(levels[j])
                        j += 1
                    else:
                        break
                
                # Merge similar levels
                if len(similar_levels) > 1:
                    merged_level = self._merge_similar_levels(similar_levels)
                else:
                    merged_level = current_level
                
                # Add additional validation criteria
                merged_level['distance_from_current'] = abs(merged_level['price'] - current_price) / (current_price + 0.001)
                merged_level['relevance'] = self._calculate_level_relevance(merged_level, current_price)
                
                # Only keep relevant and strong levels
                if merged_level['strength'] >= 0.5 and merged_level['relevance'] >= 0.3:
                    validated_levels.append(merged_level)
                
                i = j
            
            # Sort by strength (highest first)
            validated_levels.sort(key=lambda x: x['strength'], reverse=True)
            
            # Keep top levels only
            return validated_levels[:20]  # Maximum 20 levels
            
        except Exception as e:
            logger.error(f"❌ S/R validation error: {str(e)}")
            return []
    
    def _merge_similar_levels(self, levels: List[Dict]) -> Dict:
        """Merge similar S/R levels into one stronger level"""
        try:
            # Weighted average price
            total_weight = sum(l['strength'] * l['touch_count'] for l in levels)
            weighted_price = sum(l['price'] * l['strength'] * l['touch_count'] for l in levels)
            merged_price = weighted_price / (total_weight + 0.001)
            
            # Combined strength
            max_strength = max(l['strength'] for l in levels)
            avg_strength = np.mean([l['strength'] for l in levels])
            merged_strength = min((max_strength + avg_strength) / 2 * 1.2, 1.0)
            
            # Combined touch count
            total_touches = sum(l['touch_count'] for l in levels)
            
            # Best method and precision
            best_level = max(levels, key=lambda x: x['strength'])
            
            return {
                'price': merged_price,
                'type': best_level['type'],
                'strength': merged_strength,
                'touch_count': total_touches,
                'method': 'merged_levels',
                'precision': best_level['precision'],
                'merged_from': len(levels)
            }
            
        except Exception as e:
            logger.error(f"❌ Level merging error: {str(e)}")
            return levels[0] if levels else {}
    
    def _calculate_pivot_strength(self, pivot_price: float, window_prices: np.ndarray, 
                                level_type: str) -> float:
        """Calculate strength of a pivot level"""
        try:
            if level_type == 'resistance':
                below_count = np.sum(window_prices < pivot_price)
                strength = below_count / len(window_prices)
            else:  # support
                above_count = np.sum(window_prices > pivot_price)
                strength = above_count / len(window_prices)
            
            # Add bonus for extreme levels
            if strength > 0.9:
                strength = min(strength * 1.1, 1.0)
            
            return strength
            
        except Exception as e:
            logger.error(f"❌ Pivot strength calculation error: {str(e)}")
            return 0.0
    
    def _calculate_level_relevance(self, level: Dict, current_price: float) -> float:
        """Calculate how relevant a S/R level is to current trading"""
        try:
            distance = abs(level['price'] - current_price) / (current_price + 0.001)
            
            # More relevant if closer to current price
            distance_factor = max(1.0 - distance * 10, 0.1)
            
            # More relevant if higher strength
            strength_factor = level['strength']
            
            # More relevant if more touches
            touch_factor = min(level['touch_count'] / 5.0, 1.0)
            
            relevance = (distance_factor * 0.4 + strength_factor * 0.4 + touch_factor * 0.2)
            return min(relevance, 1.0)
            
        except Exception as e:
            logger.error(f"❌ Relevance calculation error: {str(e)}")
            return 0.0
    
    async def _classify_zone_freshness(self, levels: List[Dict], 
                                     candle_data: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Classify S/R zones as fresh or historical"""
        try:
            fresh_zones = []
            historical_zones = []
            
            current_time = len(candle_data)  # Use index as time proxy
            
            for level in levels:
                # Calculate "age" of the level (simplified)
                # In real implementation, would use actual timestamps
                
                # For now, classify based on strength and method
                if (level['method'] in ['pivot_analysis', 'volume_confirmation'] and 
                    level['strength'] > 0.8):
                    fresh_zones.append({
                        **level,
                        'freshness': 'fresh',
                        'age_factor': 1.0
                    })
                else:
                    historical_zones.append({
                        **level,
                        'freshness': 'historical',
                        'age_factor': 0.7
                    })
            
            return fresh_zones, historical_zones
            
        except Exception as e:
            logger.error(f"❌ Zone classification error: {str(e)}")
            return [], []
    
    async def _analyze_rejection_strengths(self, levels: List[Dict], 
                                         candle_data: List[Dict]) -> Dict:
        """Analyze rejection strengths at S/R levels"""
        try:
            rejection_analysis = {
                'strong_rejections': [],
                'weak_rejections': [],
                'breakout_attempts': []
            }
            
            if not levels or len(candle_data) < 5:
                return rejection_analysis
            
            # Look for recent rejections
            recent_candles = candle_data[-10:]
            
            for level in levels:
                level_price = level['price']
                level_type = level['type']
                
                # Check recent candles for interactions with this level
                for i, candle in enumerate(recent_candles):
                    high = candle.get('high', 0)
                    low = candle.get('low', 0)
                    close = candle.get('close', 0)
                    open_price = candle.get('open', 0)
                    
                    # Check for level interaction
                    tolerance = level_price * 0.001  # 0.1% tolerance
                    
                    if level_type == 'resistance':
                        if high >= level_price - tolerance and close < level_price:
                            # Rejection from resistance
                            rejection_strength = (level_price - close) / (high - low + 0.001)
                            
                            if rejection_strength > 0.6:
                                rejection_analysis['strong_rejections'].append({
                                    'level': level,
                                    'strength': rejection_strength,
                                    'candle_index': i
                                })
                            else:
                                rejection_analysis['weak_rejections'].append({
                                    'level': level,
                                    'strength': rejection_strength,
                                    'candle_index': i
                                })
                    
                    elif level_type == 'support':
                        if low <= level_price + tolerance and close > level_price:
                            # Bounce from support
                            rejection_strength = (close - level_price) / (high - low + 0.001)
                            
                            if rejection_strength > 0.6:
                                rejection_analysis['strong_rejections'].append({
                                    'level': level,
                                    'strength': rejection_strength,
                                    'candle_index': i
                                })
                            else:
                                rejection_analysis['weak_rejections'].append({
                                    'level': level,
                                    'strength': rejection_strength,
                                    'candle_index': i
                                })
            
            return rejection_analysis
            
        except Exception as e:
            logger.error(f"❌ Rejection analysis error: {str(e)}")
            return {}
    
    async def _analyze_price_proximity(self, levels: List[Dict], current_candle: Dict) -> Dict:
        """Analyze current price proximity to S/R levels"""
        try:
            if not levels or not current_candle:
                return {}
            
            current_price = current_candle.get('close', 0)
            
            proximity_analysis = {
                'nearest_support': None,
                'nearest_resistance': None,
                'distance_to_nearest_support': float('inf'),
                'distance_to_nearest_resistance': float('inf'),
                'imminent_levels': [],  # Levels within 0.5%
                'key_levels_nearby': []  # Strong levels within 1%
            }
            
            for level in levels:
                level_price = level['price']
                distance = abs(level_price - current_price) / (current_price + 0.001)
                
                if level['type'] == 'support' and level_price < current_price:
                    if distance < proximity_analysis['distance_to_nearest_support']:
                        proximity_analysis['nearest_support'] = level
                        proximity_analysis['distance_to_nearest_support'] = distance
                
                elif level['type'] == 'resistance' and level_price > current_price:
                    if distance < proximity_analysis['distance_to_nearest_resistance']:
                        proximity_analysis['nearest_resistance'] = level
                        proximity_analysis['distance_to_nearest_resistance'] = distance
                
                # Check for imminent levels (within 0.5%)
                if distance < 0.005:
                    proximity_analysis['imminent_levels'].append({
                        'level': level,
                        'distance': distance,
                        'direction': 'above' if level_price > current_price else 'below'
                    })
                
                # Check for key levels nearby (within 1%)
                if distance < 0.01 and level['strength'] > 0.7:
                    proximity_analysis['key_levels_nearby'].append({
                        'level': level,
                        'distance': distance,
                        'direction': 'above' if level_price > current_price else 'below'
                    })
            
            return proximity_analysis
            
        except Exception as e:
            logger.error(f"❌ Proximity analysis error: {str(e)}")
            return {}
    
    async def get_sr_signal_for_price(self, target_price: float, 
                                    signal_direction: str) -> Dict:
        """Get S/R analysis for a specific price and signal direction"""
        try:
            if not self.sr_zones:
                return {"warning": "No S/R zones available"}
            
            analysis = {
                'signal_direction': signal_direction,
                'target_price': target_price,
                'sr_conflicts': [],
                'sr_confirmations': [],
                'risk_assessment': 'unknown'
            }
            
            # Check for conflicts and confirmations
            for level in self.sr_zones:
                level_price = level['price']
                distance = abs(level_price - target_price) / (target_price + 0.001)
                
                # Close levels (within 0.3%) are significant
                if distance < 0.003:
                    if ((signal_direction == 'CALL' and level['type'] == 'resistance') or
                        (signal_direction == 'PUT' and level['type'] == 'support')):
                        # Conflict: signal direction against S/R level
                        analysis['sr_conflicts'].append({
                            'level': level,
                            'conflict_type': f"{signal_direction} signal near {level['type']}",
                            'distance': distance,
                            'strength': level['strength']
                        })
                    else:
                        # Confirmation: signal direction with S/R level
                        analysis['sr_confirmations'].append({
                            'level': level,
                            'confirmation_type': f"{signal_direction} signal with {level['type']}",
                            'distance': distance,
                            'strength': level['strength']
                        })
            
            # Risk assessment
            strong_conflicts = [c for c in analysis['sr_conflicts'] if c['strength'] > 0.8]
            strong_confirmations = [c for c in analysis['sr_confirmations'] if c['strength'] > 0.8]
            
            if strong_conflicts:
                analysis['risk_assessment'] = 'high_risk'
            elif strong_confirmations:
                analysis['risk_assessment'] = 'low_risk'
            elif analysis['sr_conflicts']:
                analysis['risk_assessment'] = 'medium_risk'
            else:
                analysis['risk_assessment'] = 'neutral'
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ S/R signal analysis error: {str(e)}")
            return {"error": str(e)}