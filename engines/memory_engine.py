"""
Memory Engine - Advanced Exhaustion Memory System
Remembers previous fakeouts, breakouts, reversals, and exhaustion patterns
Avoids trading zones faked-out twice - Advanced memory-based pattern recognition
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import asyncio
from datetime import datetime, timedelta
import logging
import json
import pickle
from collections import defaultdict

class MemoryEngine:
    """
    Advanced memory system for trading pattern recognition
    Tracks and learns from market deceptions and exhaustion patterns
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Memory storage
        self.fakeout_memory = []
        self.breakout_memory = []
        self.exhaustion_memory = []
        self.reversal_memory = []
        self.trap_zones = []
        
        # Memory configuration
        self.memory_retention_hours = 168  # 1 week retention
        self.fakeout_threshold = 2  # Don't trade zones faked twice
        self.exhaustion_cooldown = 24   # Hours before re-entering exhausted zones
        
        # Pattern learning
        self.pattern_success_rates = defaultdict(lambda: {'attempts': 0, 'successes': 0})
        self.zone_reliability = defaultdict(lambda: {'tests': 0, 'holds': 0})
        
        # Memory statistics
        self.memory_stats = {
            'total_fakeouts_remembered': 0,
            'zones_avoided': 0,
            'memory_saves': 0,
            'pattern_accuracy_improvement': 0.0
        }
    
    async def load_pattern_memory(self, historical_data: pd.DataFrame):
        """Initialize memory engine with historical pattern data"""
        try:
            self.logger.info("🧠 Loading pattern memory from historical data...")
            
            # Analyze historical data for patterns
            await self._analyze_historical_fakeouts(historical_data)
            await self._analyze_historical_exhaustion(historical_data)
            await self._analyze_historical_breakouts(historical_data)
            
            self.logger.info("✅ Pattern memory loaded")
            
        except Exception as e:
            self.logger.error(f"Error loading pattern memory: {e}")
    
    async def update_memory(self, market_data: Dict, signal: Optional[Dict]):
        """Update memory with new market data and signal results"""
        try:
            if not market_data:
                return
            
            candles = market_data.get('candles', [])
            if len(candles) < 2:
                return
            
            current_candle = candles[-1]
            previous_candle = candles[-2]
            
            # Update different memory types
            await self._update_fakeout_memory(candles)
            await self._update_breakout_memory(candles)
            await self._update_exhaustion_memory(candles)
            await self._update_reversal_memory(candles)
            
            # Update trap zones
            await self._update_trap_zones(candles)
            
            # Learn from signal results if provided
            if signal:
                await self._learn_from_signal(signal, current_candle)
            
            # Clean old memories
            await self._clean_old_memories()
            
        except Exception as e:
            self.logger.error(f"Error updating memory: {e}")
    
    async def check_trap_zones(self, current_candle: Dict) -> Dict[str, Any]:
        """Check if current price is in a known trap zone"""
        try:
            current_price = current_candle.get('close', 0)
            
            memory_check = {
                'in_trap_zone': False,
                'trap_zone_type': 'none',
                'fakeout_count': 0,
                'last_fakeout_time': None,
                'exhaustion_detected': False,
                'exhaustion_zones': [],
                'fakeout_zones': [],
                'avoid_zone': False,
                'memory_recommendation': 'proceed'
            }
            
            # Check fakeout zones
            fakeout_zones = await self._check_fakeout_zones(current_price)
            memory_check['fakeout_zones'] = fakeout_zones
            
            # Check exhaustion zones
            exhaustion_zones = await self._check_exhaustion_zones(current_price)
            memory_check['exhaustion_zones'] = exhaustion_zones
            
            # Determine if in trap zone
            if fakeout_zones:
                most_dangerous_zone = max(fakeout_zones, key=lambda x: x['count'])
                if most_dangerous_zone['count'] >= self.fakeout_threshold:
                    memory_check.update({
                        'in_trap_zone': True,
                        'trap_zone_type': 'fakeout',
                        'fakeout_count': most_dangerous_zone['count'],
                        'last_fakeout_time': most_dangerous_zone['last_time'],
                        'avoid_zone': True,
                        'memory_recommendation': 'avoid'
                    })
                    
                    self.memory_stats['zones_avoided'] += 1
            
            # Check exhaustion zones
            if exhaustion_zones:
                recent_exhaustion = any(
                    (datetime.now() - zone['time']).total_seconds() < (self.exhaustion_cooldown * 3600)
                    for zone in exhaustion_zones
                )
                
                if recent_exhaustion:
                    memory_check.update({
                        'exhaustion_detected': True,
                        'avoid_zone': True,
                        'memory_recommendation': 'wait_for_cooldown'
                    })
            
            return memory_check
            
        except Exception as e:
            self.logger.error(f"Error checking trap zones: {e}")
            return {'in_trap_zone': False, 'avoid_zone': False}
    
    async def _update_fakeout_memory(self, candles: List[Dict]):
        """Update memory of fakeout patterns"""
        try:
            if len(candles) < 5:
                return
            
            # Look for fakeout patterns in recent candles
            recent_candles = candles[-5:]
            
            # Detect potential fakeouts
            for i in range(1, len(recent_candles) - 1):
                prev_candle = recent_candles[i-1]
                current_candle = recent_candles[i]
                next_candle = recent_candles[i+1]
                
                # Upward fakeout detection
                upward_fakeout = (
                    current_candle['high'] > prev_candle['high'] * 1.002 and  # Break above
                    current_candle['close'] < prev_candle['high'] and          # Close below
                    next_candle['close'] < current_candle['low']               # Follow-through down
                )
                
                # Downward fakeout detection
                downward_fakeout = (
                    current_candle['low'] < prev_candle['low'] * 0.998 and    # Break below
                    current_candle['close'] > prev_candle['low'] and          # Close above
                    next_candle['close'] > current_candle['high']             # Follow-through up
                )
                
                if upward_fakeout:
                    fakeout_level = prev_candle['high']
                    await self._store_fakeout(fakeout_level, 'resistance', current_candle)
                
                elif downward_fakeout:
                    fakeout_level = prev_candle['low']
                    await self._store_fakeout(fakeout_level, 'support', current_candle)
            
        except Exception as e:
            self.logger.error(f"Error updating fakeout memory: {e}")
    
    async def _update_breakout_memory(self, candles: List[Dict]):
        """Update memory of successful breakout patterns"""
        try:
            if len(candles) < 5:
                return
            
            recent_candles = candles[-5:]
            
            # Detect successful breakouts
            for i in range(2, len(recent_candles)):
                base_candle = recent_candles[i-2]
                break_candle = recent_candles[i-1]
                confirm_candle = recent_candles[i]
                
                # Upward breakout
                upward_breakout = (
                    break_candle['close'] > base_candle['high'] * 1.002 and
                    confirm_candle['close'] > break_candle['close'] and
                    confirm_candle['low'] > base_candle['high'] * 0.999  # Held above
                )
                
                # Downward breakout
                downward_breakout = (
                    break_candle['close'] < base_candle['low'] * 0.998 and
                    confirm_candle['close'] < break_candle['close'] and
                    confirm_candle['high'] < base_candle['low'] * 1.001  # Held below
                )
                
                if upward_breakout:
                    breakout_level = base_candle['high']
                    await self._store_breakout(breakout_level, 'upward', break_candle, True)
                
                elif downward_breakout:
                    breakout_level = base_candle['low']
                    await self._store_breakout(breakout_level, 'downward', break_candle, True)
            
        except Exception as e:
            self.logger.error(f"Error updating breakout memory: {e}")
    
    async def _update_exhaustion_memory(self, candles: List[Dict]):
        """Update memory of exhaustion patterns"""
        try:
            if len(candles) < 6:
                return
            
            recent_candles = candles[-6:]
            
            # Detect exhaustion patterns
            # 1. Multiple attempts without follow-through
            highs = [c['high'] for c in recent_candles]
            lows = [c['low'] for c in recent_candles]
            
            # Check for multiple touches of same level without breakout
            for i in range(len(recent_candles) - 3):
                level = highs[i]
                touches = sum(1 for h in highs[i+1:] if abs(h - level) < level * 0.001)
                
                if touches >= 2:
                    # Check if level held (no significant breakout)
                    max_break = max(highs[i+1:])
                    if max_break < level * 1.005:  # Less than 0.5% breakout
                        await self._store_exhaustion(level, 'resistance', recent_candles[-1])
            
            for i in range(len(recent_candles) - 3):
                level = lows[i]
                touches = sum(1 for l in lows[i+1:] if abs(l - level) < level * 0.001)
                
                if touches >= 2:
                    # Check if level held (no significant breakdown)
                    min_break = min(lows[i+1:])
                    if min_break > level * 0.995:  # Less than 0.5% breakdown
                        await self._store_exhaustion(level, 'support', recent_candles[-1])
            
            # 2. Volume exhaustion patterns
            if all('volume' in c for c in recent_candles):
                volumes = [c['volume'] for c in recent_candles]
                
                # Declining volume with attempts to move
                if len(volumes) >= 4:
                    volume_decline = all(volumes[i] >= volumes[i+1] for i in range(len(volumes)-1))
                    price_attempts = (
                        max(highs[-3:]) > max(highs[-6:-3]) or
                        min(lows[-3:]) < min(lows[-6:-3])
                    )
                    
                    if volume_decline and price_attempts:
                        current_level = recent_candles[-1]['close']
                        await self._store_exhaustion(current_level, 'volume_exhaustion', recent_candles[-1])
            
        except Exception as e:
            self.logger.error(f"Error updating exhaustion memory: {e}")
    
    async def _update_reversal_memory(self, candles: List[Dict]):
        """Update memory of reversal patterns"""
        try:
            if len(candles) < 4:
                return
            
            recent_candles = candles[-4:]
            
            # Detect reversal patterns
            for i in range(1, len(recent_candles) - 1):
                prev_candle = recent_candles[i-1]
                reversal_candle = recent_candles[i]
                confirm_candle = recent_candles[i+1]
                
                # Strong reversal criteria
                strong_reversal = abs(reversal_candle['close'] - reversal_candle['open']) > abs(prev_candle['close'] - prev_candle['open']) * 1.5
                
                # Direction change
                prev_bullish = prev_candle['close'] > prev_candle['open']
                reversal_bullish = reversal_candle['close'] > reversal_candle['open']
                confirm_bullish = confirm_candle['close'] > confirm_candle['open']
                
                # Confirmed reversal
                confirmed_reversal = (
                    strong_reversal and 
                    prev_bullish != reversal_bullish and 
                    reversal_bullish == confirm_bullish
                )
                
                if confirmed_reversal:
                    reversal_level = reversal_candle['close']
                    reversal_type = 'bullish' if reversal_bullish else 'bearish'
                    await self._store_reversal(reversal_level, reversal_type, reversal_candle)
            
        except Exception as e:
            self.logger.error(f"Error updating reversal memory: {e}")
    
    async def _update_trap_zones(self, candles: List[Dict]):
        """Update trap zones based on all memory types"""
        try:
            current_time = datetime.now()
            
            # Add fakeout zones to trap zones
            for fakeout in self.fakeout_memory:
                if (current_time - fakeout['time']).total_seconds() < (self.memory_retention_hours * 3600):
                    existing_zone = next(
                        (zone for zone in self.trap_zones 
                         if abs(zone['level'] - fakeout['level']) < fakeout['level'] * 0.001),
                        None
                    )
                    
                    if existing_zone:
                        existing_zone['danger_count'] += 1
                        existing_zone['last_updated'] = current_time
                    else:
                        self.trap_zones.append({
                            'level': fakeout['level'],
                            'type': 'fakeout_trap',
                            'danger_count': 1,
                            'created': fakeout['time'],
                            'last_updated': current_time
                        })
            
            # Clean old trap zones
            self.trap_zones = [
                zone for zone in self.trap_zones
                if (current_time - zone['created']).total_seconds() < (self.memory_retention_hours * 3600)
            ]
            
        except Exception as e:
            self.logger.error(f"Error updating trap zones: {e}")
    
    async def _store_fakeout(self, level: float, fakeout_type: str, candle: Dict):
        """Store a fakeout in memory"""
        try:
            fakeout_record = {
                'level': level,
                'type': fakeout_type,
                'time': datetime.now(),
                'candle_data': {
                    'open': candle['open'],
                    'high': candle['high'],
                    'low': candle['low'],
                    'close': candle['close']
                }
            }
            
            self.fakeout_memory.append(fakeout_record)
            self.memory_stats['total_fakeouts_remembered'] += 1
            
        except Exception as e:
            self.logger.error(f"Error storing fakeout: {e}")
    
    async def _store_breakout(self, level: float, direction: str, candle: Dict, success: bool):
        """Store a breakout in memory"""
        try:
            breakout_record = {
                'level': level,
                'direction': direction,
                'success': success,
                'time': datetime.now(),
                'candle_data': {
                    'open': candle['open'],
                    'high': candle['high'],
                    'low': candle['low'],
                    'close': candle['close']
                }
            }
            
            self.breakout_memory.append(breakout_record)
            
        except Exception as e:
            self.logger.error(f"Error storing breakout: {e}")
    
    async def _store_exhaustion(self, level: float, exhaustion_type: str, candle: Dict):
        """Store an exhaustion pattern in memory"""
        try:
            exhaustion_record = {
                'level': level,
                'type': exhaustion_type,
                'time': datetime.now(),
                'candle_data': {
                    'open': candle['open'],
                    'high': candle['high'],
                    'low': candle['low'],
                    'close': candle['close']
                }
            }
            
            self.exhaustion_memory.append(exhaustion_record)
            
        except Exception as e:
            self.logger.error(f"Error storing exhaustion: {e}")
    
    async def _store_reversal(self, level: float, reversal_type: str, candle: Dict):
        """Store a reversal pattern in memory"""
        try:
            reversal_record = {
                'level': level,
                'type': reversal_type,
                'time': datetime.now(),
                'candle_data': {
                    'open': candle['open'],
                    'high': candle['high'],
                    'low': candle['low'],
                    'close': candle['close']
                }
            }
            
            self.reversal_memory.append(reversal_record)
            
        except Exception as e:
            self.logger.error(f"Error storing reversal: {e}")
    
    async def _check_fakeout_zones(self, current_price: float) -> List[Dict]:
        """Check if current price is near known fakeout zones"""
        try:
            fakeout_zones = []
            
            # Group fakeouts by level
            level_groups = defaultdict(list)
            for fakeout in self.fakeout_memory:
                # Check if recent enough
                if (datetime.now() - fakeout['time']).total_seconds() < (self.memory_retention_hours * 3600):
                    level_groups[round(fakeout['level'], 5)].append(fakeout)
            
            # Check if current price is near any fakeout levels
            for level, fakeouts in level_groups.items():
                distance = abs(current_price - level) / current_price
                if distance < 0.002:  # Within 0.2%
                    fakeout_zones.append({
                        'level': level,
                        'count': len(fakeouts),
                        'last_time': max(f['time'] for f in fakeouts),
                        'types': [f['type'] for f in fakeouts]
                    })
            
            return fakeout_zones
            
        except Exception as e:
            self.logger.error(f"Error checking fakeout zones: {e}")
            return []
    
    async def _check_exhaustion_zones(self, current_price: float) -> List[Dict]:
        """Check if current price is near known exhaustion zones"""
        try:
            exhaustion_zones = []
            
            for exhaustion in self.exhaustion_memory:
                # Check if recent enough
                if (datetime.now() - exhaustion['time']).total_seconds() < (self.memory_retention_hours * 3600):
                    distance = abs(current_price - exhaustion['level']) / current_price
                    if distance < 0.002:  # Within 0.2%
                        exhaustion_zones.append({
                            'level': exhaustion['level'],
                            'type': exhaustion['type'],
                            'time': exhaustion['time']
                        })
            
            return exhaustion_zones
            
        except Exception as e:
            self.logger.error(f"Error checking exhaustion zones: {e}")
            return []
    
    async def _learn_from_signal(self, signal: Dict, candle: Dict):
        """Learn from signal results to improve future performance"""
        try:
            if not signal or signal.get('action') == 'NO_TRADE':
                return
            
            signal_type = signal.get('strategy', 'unknown')
            confidence = signal.get('confidence', 0)
            
            # Store pattern attempt
            pattern_key = f"{signal_type}_{signal.get('action', 'unknown')}"
            self.pattern_success_rates[pattern_key]['attempts'] += 1
            
            # This would be enhanced with actual trade result tracking
            # For now, we use confidence as a proxy for success probability
            if confidence > 0.8:
                self.pattern_success_rates[pattern_key]['successes'] += 1
            
        except Exception as e:
            self.logger.error(f"Error learning from signal: {e}")
    
    async def _clean_old_memories(self):
        """Clean old memories beyond retention period"""
        try:
            current_time = datetime.now()
            retention_seconds = self.memory_retention_hours * 3600
            
            # Clean fakeout memory
            self.fakeout_memory = [
                f for f in self.fakeout_memory
                if (current_time - f['time']).total_seconds() < retention_seconds
            ]
            
            # Clean breakout memory
            self.breakout_memory = [
                b for b in self.breakout_memory
                if (current_time - b['time']).total_seconds() < retention_seconds
            ]
            
            # Clean exhaustion memory
            self.exhaustion_memory = [
                e for e in self.exhaustion_memory
                if (current_time - e['time']).total_seconds() < retention_seconds
            ]
            
            # Clean reversal memory
            self.reversal_memory = [
                r for r in self.reversal_memory
                if (current_time - r['time']).total_seconds() < retention_seconds
            ]
            
        except Exception as e:
            self.logger.error(f"Error cleaning old memories: {e}")
    
    async def _analyze_historical_fakeouts(self, data: pd.DataFrame):
        """Analyze historical data for fakeout patterns"""
        try:
            if len(data) < 10:
                return
            
            # Look for historical fakeout patterns
            for i in range(5, len(data) - 2):
                current_row = data.iloc[i]
                prev_row = data.iloc[i-1]
                next_row = data.iloc[i+1]
                
                # Simple fakeout detection for historical data
                upward_fakeout = (
                    current_row['high'] > prev_row['high'] * 1.002 and
                    current_row['close'] < prev_row['high'] and
                    next_row['close'] < current_row['low']
                )
                
                downward_fakeout = (
                    current_row['low'] < prev_row['low'] * 0.998 and
                    current_row['close'] > prev_row['low'] and
                    next_row['close'] > current_row['high']
                )
                
                if upward_fakeout or downward_fakeout:
                    # Store in historical memory (with older timestamp)
                    fake_time = datetime.now() - timedelta(hours=len(data) - i)
                    level = prev_row['high'] if upward_fakeout else prev_row['low']
                    fakeout_type = 'resistance' if upward_fakeout else 'support'
                    
                    self.fakeout_memory.append({
                        'level': level,
                        'type': fakeout_type,
                        'time': fake_time,
                        'candle_data': {
                            'open': current_row['open'],
                            'high': current_row['high'],
                            'low': current_row['low'],
                            'close': current_row['close']
                        }
                    })
            
        except Exception as e:
            self.logger.error(f"Error analyzing historical fakeouts: {e}")
    
    async def _analyze_historical_exhaustion(self, data: pd.DataFrame):
        """Analyze historical data for exhaustion patterns"""
        try:
            if len(data) < 20:
                return
            
            # Look for historical exhaustion patterns
            window_size = 10
            for i in range(window_size, len(data) - window_size):
                window = data.iloc[i-window_size:i+window_size]
                
                # Check for repeated tests of levels
                highs = window['high'].values
                lows = window['low'].values
                
                # Find levels with multiple touches
                for j in range(window_size):
                    level = highs[j]
                    touches = sum(1 for h in highs[j+1:j+5] if abs(h - level) < level * 0.001)
                    
                    if touches >= 2:
                        # Check if exhaustion occurred (no follow-through)
                        max_break = max(highs[j+1:j+5]) if j+5 < len(highs) else level
                        if max_break < level * 1.003:  # Minimal breakout
                            exhaustion_time = datetime.now() - timedelta(hours=len(data) - i)
                            
                            self.exhaustion_memory.append({
                                'level': level,
                                'type': 'resistance_exhaustion',
                                'time': exhaustion_time,
                                'candle_data': {
                                    'open': data.iloc[i]['open'],
                                    'high': data.iloc[i]['high'],
                                    'low': data.iloc[i]['low'],
                                    'close': data.iloc[i]['close']
                                }
                            })
            
        except Exception as e:
            self.logger.error(f"Error analyzing historical exhaustion: {e}")
    
    async def _analyze_historical_breakouts(self, data: pd.DataFrame):
        """Analyze historical data for breakout patterns"""
        try:
            if len(data) < 10:
                return
            
            # Look for historical breakout patterns
            for i in range(5, len(data) - 3):
                base_row = data.iloc[i-2]
                break_row = data.iloc[i]
                confirm_row = data.iloc[i+2]
                
                # Successful upward breakout
                upward_success = (
                    break_row['close'] > base_row['high'] * 1.002 and
                    confirm_row['close'] > break_row['close'] and
                    confirm_row['low'] > base_row['high'] * 0.999
                )
                
                # Successful downward breakout
                downward_success = (
                    break_row['close'] < base_row['low'] * 0.998 and
                    confirm_row['close'] < break_row['close'] and
                    confirm_row['high'] < base_row['low'] * 1.001
                )
                
                if upward_success or downward_success:
                    breakout_time = datetime.now() - timedelta(hours=len(data) - i)
                    level = base_row['high'] if upward_success else base_row['low']
                    direction = 'upward' if upward_success else 'downward'
                    
                    self.breakout_memory.append({
                        'level': level,
                        'direction': direction,
                        'success': True,
                        'time': breakout_time,
                        'candle_data': {
                            'open': break_row['open'],
                            'high': break_row['high'],
                            'low': break_row['low'],
                            'close': break_row['close']
                        }
                    })
            
        except Exception as e:
            self.logger.error(f"Error analyzing historical breakouts: {e}")
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        try:
            current_time = datetime.now()
            
            # Recent memory counts
            recent_fakeouts = len([
                f for f in self.fakeout_memory
                if (current_time - f['time']).total_seconds() < 86400  # Last 24 hours
            ])
            
            recent_exhaustions = len([
                e for e in self.exhaustion_memory
                if (current_time - e['time']).total_seconds() < 86400
            ])
            
            # Pattern success rates
            pattern_stats = {}
            for pattern, stats in self.pattern_success_rates.items():
                if stats['attempts'] > 0:
                    success_rate = stats['successes'] / stats['attempts']
                    pattern_stats[pattern] = {
                        'attempts': stats['attempts'],
                        'success_rate': success_rate
                    }
            
            # Zone reliability
            zone_stats = {}
            for zone, stats in self.zone_reliability.items():
                if stats['tests'] > 0:
                    hold_rate = stats['holds'] / stats['tests']
                    zone_stats[zone] = {
                        'tests': stats['tests'],
                        'hold_rate': hold_rate
                    }
            
            return {
                'total_fakeouts_remembered': len(self.fakeout_memory),
                'total_breakouts_remembered': len(self.breakout_memory),
                'total_exhaustions_remembered': len(self.exhaustion_memory),
                'total_reversals_remembered': len(self.reversal_memory),
                'active_trap_zones': len(self.trap_zones),
                'recent_fakeouts_24h': recent_fakeouts,
                'recent_exhaustions_24h': recent_exhaustions,
                'memory_retention_hours': self.memory_retention_hours,
                'zones_avoided_total': self.memory_stats['zones_avoided'],
                'memory_saves': self.memory_stats['memory_saves'],
                'pattern_success_rates': pattern_stats,
                'zone_reliability': zone_stats
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating memory statistics: {e}")
            return {}
    
    def save_memory(self, filepath: str = "trading_memory.pkl"):
        """Save memory to disk for persistence"""
        try:
            memory_data = {
                'fakeout_memory': self.fakeout_memory,
                'breakout_memory': self.breakout_memory,
                'exhaustion_memory': self.exhaustion_memory,
                'reversal_memory': self.reversal_memory,
                'trap_zones': self.trap_zones,
                'pattern_success_rates': dict(self.pattern_success_rates),
                'zone_reliability': dict(self.zone_reliability),
                'memory_stats': self.memory_stats
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(memory_data, f)
            
            self.logger.info(f"Memory saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving memory: {e}")
    
    def load_memory(self, filepath: str = "trading_memory.pkl"):
        """Load memory from disk"""
        try:
            with open(filepath, 'rb') as f:
                memory_data = pickle.load(f)
            
            self.fakeout_memory = memory_data.get('fakeout_memory', [])
            self.breakout_memory = memory_data.get('breakout_memory', [])
            self.exhaustion_memory = memory_data.get('exhaustion_memory', [])
            self.reversal_memory = memory_data.get('reversal_memory', [])
            self.trap_zones = memory_data.get('trap_zones', [])
            self.pattern_success_rates = defaultdict(lambda: {'attempts': 0, 'successes': 0}, 
                                                   memory_data.get('pattern_success_rates', {}))
            self.zone_reliability = defaultdict(lambda: {'tests': 0, 'holds': 0},
                                              memory_data.get('zone_reliability', {}))
            self.memory_stats = memory_data.get('memory_stats', self.memory_stats)
            
            self.logger.info(f"Memory loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error loading memory: {e}")