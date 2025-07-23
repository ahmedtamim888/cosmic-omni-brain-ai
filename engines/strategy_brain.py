"""
Strategy Brain - Dynamic Strategy Selection Engine
Auto-adapts to market conditions and creates strategy trees
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import asyncio
from datetime import datetime
import logging
from enum import Enum

class MarketRegime(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    CONSOLIDATING = "consolidating"
    BREAKOUT = "breakout"

class StrategyBrain:
    """
    Dynamic strategy selection and adaptation engine
    Creates custom strategies for each market situation
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.current_regime = MarketRegime.RANGING
        self.strategy_history = []
        self.performance_metrics = {}
        
        # Strategy templates
        self.strategy_templates = {
            MarketRegime.TRENDING_UP: self._create_trend_following_strategy,
            MarketRegime.TRENDING_DOWN: self._create_trend_following_strategy,
            MarketRegime.RANGING: self._create_range_trading_strategy,
            MarketRegime.VOLATILE: self._create_volatility_strategy,
            MarketRegime.CONSOLIDATING: self._create_consolidation_strategy,
            MarketRegime.BREAKOUT: self._create_breakout_strategy
        }
    
    async def select_strategy(self, candle_features: Dict, pattern_signals: Dict,
                            psychology_score: Dict, sr_signals: Dict) -> Dict[str, Any]:
        """
        Main strategy selection function
        Analyzes market conditions and selects optimal strategy
        """
        try:
            # Detect current market regime
            market_regime = await self._detect_market_regime(candle_features, pattern_signals, psychology_score)
            
            # Update current regime
            self.current_regime = market_regime
            
            # Get strategy template for current regime
            strategy_creator = self.strategy_templates.get(market_regime, self._create_default_strategy)
            
            # Create custom strategy
            strategy = await strategy_creator(candle_features, pattern_signals, psychology_score, sr_signals)
            
            # Enhance strategy with confluence analysis
            enhanced_strategy = await self._enhance_strategy_with_confluence(
                strategy, candle_features, pattern_signals, psychology_score, sr_signals
            )
            
            # Add performance tracking
            enhanced_strategy['regime'] = market_regime.value
            enhanced_strategy['timestamp'] = datetime.now()
            
            # Store strategy
            self.strategy_history.append(enhanced_strategy)
            
            return enhanced_strategy
            
        except Exception as e:
            self.logger.error(f"Error selecting strategy: {e}")
            return await self._create_default_strategy()
    
    async def _detect_market_regime(self, candle_features: Dict, pattern_signals: Dict,
                                  psychology_score: Dict) -> MarketRegime:
        """Detect current market regime"""
        try:
            # Extract key indicators
            momentum = candle_features.get('momentum_signals', {})
            volatility = candle_features.get('size_analysis', {})
            volume = candle_features.get('volume_analysis', {})
            
            # Trend detection
            momentum_direction = momentum.get('momentum_direction', 'neutral')
            momentum_strength = momentum.get('short_momentum', 0) + momentum.get('long_momentum', 0)
            
            # Volatility analysis
            size_breakout = volatility.get('is_size_breakout', False)
            size_compression = volatility.get('is_size_compression', False)
            
            # Volume analysis
            volume_spike = volume.get('volume_spike', False)
            volume_trend = volume.get('volume_trend', 'stable')
            
            # Pattern analysis
            patterns = pattern_signals.get('patterns_detected', [])
            breakout_patterns = [p for p in patterns if 'breakout' in p.get('name', '').lower()]
            reversal_patterns = [p for p in patterns if 'reversal' in p.get('name', '').lower()]
            
            # Regime decision logic
            if size_breakout and volume_spike and breakout_patterns:
                return MarketRegime.BREAKOUT
            
            elif abs(momentum_strength) > 0.02:  # Strong momentum
                if momentum_direction == 'bullish':
                    return MarketRegime.TRENDING_UP
                elif momentum_direction == 'bearish':
                    return MarketRegime.TRENDING_DOWN
            
            elif size_compression and volume.get('volume_dry_up', False):
                return MarketRegime.CONSOLIDATING
            
            elif volatility.get('size_volatility', 0) > 1.5:  # High volatility
                return MarketRegime.VOLATILE
            
            else:
                return MarketRegime.RANGING  # Default to ranging
                
        except Exception as e:
            self.logger.error(f"Error detecting market regime: {e}")
            return MarketRegime.RANGING
    
    async def _create_trend_following_strategy(self, candle_features: Dict, pattern_signals: Dict,
                                             psychology_score: Dict, sr_signals: Dict) -> Dict[str, Any]:
        """Create trend-following strategy"""
        try:
            momentum = candle_features.get('momentum_signals', {})
            volume = candle_features.get('volume_analysis', {})
            
            # Determine trend direction
            trend_direction = momentum.get('momentum_direction', 'neutral')
            momentum_strength = abs(momentum.get('short_momentum', 0))
            
            # Strategy conditions
            if trend_direction == 'bullish':
                action = 'CALL'
                entry_conditions = [
                    'bullish_momentum_confirmed',
                    'volume_supporting_trend',
                    'no_resistance_rejection'
                ]
                
                # Specific entry logic
                volume_confirmation = volume.get('volume_trend') == 'rising'
                no_resistance = not sr_signals.get('rejection_detected', False) or sr_signals.get('rejection_type') != 'resistance'
                
                entry_score = 0.6  # Base score for trend
                if volume_confirmation:
                    entry_score += 0.2
                if no_resistance:
                    entry_score += 0.2
                
            elif trend_direction == 'bearish':
                action = 'PUT'
                entry_conditions = [
                    'bearish_momentum_confirmed',
                    'volume_supporting_trend',
                    'no_support_rejection'
                ]
                
                # Specific entry logic
                volume_confirmation = volume.get('volume_trend') == 'rising'
                no_support = not sr_signals.get('rejection_detected', False) or sr_signals.get('rejection_type') != 'support'
                
                entry_score = 0.6  # Base score for trend
                if volume_confirmation:
                    entry_score += 0.2
                if no_support:
                    entry_score += 0.2
                    
            else:
                action = 'NO_TRADE'
                entry_conditions = ['trend_unclear']
                entry_score = 0.0
            
            # Next candle prediction
            next_candle_prediction = await self._predict_trend_continuation(candle_features, trend_direction)
            
            return {
                'name': 'TREND_FOLLOWING',
                'action': action,
                'entry_conditions': entry_conditions,
                'entry_score': entry_score,
                'trend_state': f"trending_{trend_direction}",
                'reason': f"Trend following strategy - {trend_direction} momentum detected",
                'confidence_boost': 0.1 if momentum_strength > 0.015 else 0,
                'next_candle_prediction': next_candle_prediction
            }
            
        except Exception as e:
            self.logger.error(f"Error creating trend following strategy: {e}")
            return await self._create_default_strategy()
    
    async def _create_range_trading_strategy(self, candle_features: Dict, pattern_signals: Dict,
                                           psychology_score: Dict, sr_signals: Dict) -> Dict[str, Any]:
        """Create range trading strategy"""
        try:
            # Range trading focuses on S/R bounces
            rejection_detected = sr_signals.get('rejection_detected', False)
            rejection_type = sr_signals.get('rejection_type', 'none')
            rejection_strength = sr_signals.get('rejection_strength', 0)
            
            # Volume confirmation
            volume = candle_features.get('volume_analysis', {})
            volume_confirmation = volume.get('volume_spike', False)
            
            if rejection_detected and rejection_strength > 0.6:
                if rejection_type == 'resistance':
                    action = 'PUT'
                    entry_conditions = [
                        'resistance_rejection_confirmed',
                        'range_trading_setup',
                        'volume_confirmation'
                    ]
                    reason = f"Range trading - resistance rejection at {sr_signals.get('rejection_level', 'unknown')}"
                    
                elif rejection_type == 'support':
                    action = 'CALL'
                    entry_conditions = [
                        'support_bounce_confirmed',
                        'range_trading_setup',
                        'volume_confirmation'
                    ]
                    reason = f"Range trading - support bounce at {sr_signals.get('rejection_level', 'unknown')}"
                    
                else:
                    action = 'NO_TRADE'
                    entry_conditions = ['unclear_rejection']
                    reason = "Range trading - unclear rejection signal"
                
                # Calculate entry score
                entry_score = 0.5 + (rejection_strength * 0.3)
                if volume_confirmation:
                    entry_score += 0.2
                    
            else:
                action = 'NO_TRADE'
                entry_conditions = ['no_clear_range_setup']
                entry_score = 0.0
                reason = "Range trading - no clear range setup detected"
            
            # Next candle prediction for range
            next_candle_prediction = await self._predict_range_movement(candle_features, sr_signals)
            
            return {
                'name': 'RANGE_TRADING',
                'action': action,
                'entry_conditions': entry_conditions,
                'entry_score': entry_score,
                'trend_state': 'ranging',
                'reason': reason,
                'confidence_boost': 0.05 if rejection_strength > 0.8 else 0,
                'next_candle_prediction': next_candle_prediction
            }
            
        except Exception as e:
            self.logger.error(f"Error creating range trading strategy: {e}")
            return await self._create_default_strategy()
    
    async def _create_volatility_strategy(self, candle_features: Dict, pattern_signals: Dict,
                                        psychology_score: Dict, sr_signals: Dict) -> Dict[str, Any]:
        """Create volatility-based strategy"""
        try:
            # Volatility strategy focuses on expansion/contraction cycles
            size_analysis = candle_features.get('size_analysis', {})
            volume = candle_features.get('volume_analysis', {})
            
            size_breakout = size_analysis.get('is_size_breakout', False)
            size_compression = size_analysis.get('is_size_compression', False)
            volume_spike = volume.get('volume_spike', False)
            
            # Look for volatility expansion after compression
            if size_compression:
                # Wait for breakout
                action = 'NO_TRADE'
                entry_conditions = ['volatility_compression_detected', 'waiting_for_breakout']
                reason = "Volatility strategy - compression phase, waiting for expansion"
                entry_score = 0.3
                
            elif size_breakout and volume_spike:
                # Determine direction from momentum
                momentum = candle_features.get('momentum_signals', {})
                direction = momentum.get('momentum_direction', 'neutral')
                
                if direction == 'bullish':
                    action = 'CALL'
                    entry_conditions = ['volatility_expansion', 'bullish_breakout', 'volume_confirmation']
                    reason = "Volatility strategy - bullish expansion breakout"
                elif direction == 'bearish':
                    action = 'PUT'
                    entry_conditions = ['volatility_expansion', 'bearish_breakout', 'volume_confirmation']
                    reason = "Volatility strategy - bearish expansion breakout"
                else:
                    action = 'NO_TRADE'
                    entry_conditions = ['volatility_expansion', 'unclear_direction']
                    reason = "Volatility strategy - expansion without clear direction"
                
                entry_score = 0.7 if direction != 'neutral' else 0.3
                
            else:
                action = 'NO_TRADE'
                entry_conditions = ['no_volatility_setup']
                entry_score = 0.0
                reason = "Volatility strategy - no clear volatility setup"
            
            # Next candle prediction
            next_candle_prediction = await self._predict_volatility_movement(candle_features)
            
            return {
                'name': 'VOLATILITY_EXPANSION',
                'action': action,
                'entry_conditions': entry_conditions,
                'entry_score': entry_score,
                'trend_state': 'volatile',
                'reason': reason,
                'confidence_boost': 0.15 if size_breakout and volume_spike else 0,
                'next_candle_prediction': next_candle_prediction
            }
            
        except Exception as e:
            self.logger.error(f"Error creating volatility strategy: {e}")
            return await self._create_default_strategy()
    
    async def _create_consolidation_strategy(self, candle_features: Dict, pattern_signals: Dict,
                                           psychology_score: Dict, sr_signals: Dict) -> Dict[str, Any]:
        """Create consolidation strategy"""
        try:
            # Consolidation strategy waits for breakout
            size_analysis = candle_features.get('size_analysis', {})
            volume = candle_features.get('volume_analysis', {})
            
            size_compression = size_analysis.get('is_size_compression', False)
            volume_dry_up = volume.get('volume_dry_up', False)
            
            # During consolidation, we mainly wait
            if size_compression and volume_dry_up:
                action = 'NO_TRADE'
                entry_conditions = ['consolidation_confirmed', 'waiting_for_breakout']
                reason = "Consolidation strategy - market consolidating, waiting for direction"
                entry_score = 0.1
                
            else:
                # Look for early breakout signs
                momentum = candle_features.get('momentum_signals', {})
                momentum_shift = momentum.get('momentum_shift_detected', False)
                
                if momentum_shift:
                    direction = momentum.get('momentum_direction', 'neutral')
                    
                    if direction == 'bullish':
                        action = 'CALL'
                        entry_conditions = ['early_breakout_signal', 'bullish_momentum_shift']
                        reason = "Consolidation strategy - early bullish breakout signal"
                        entry_score = 0.6
                    elif direction == 'bearish':
                        action = 'PUT'
                        entry_conditions = ['early_breakout_signal', 'bearish_momentum_shift']
                        reason = "Consolidation strategy - early bearish breakout signal"
                        entry_score = 0.6
                    else:
                        action = 'NO_TRADE'
                        entry_conditions = ['momentum_shift', 'unclear_direction']
                        reason = "Consolidation strategy - momentum shift but unclear direction"
                        entry_score = 0.2
                else:
                    action = 'NO_TRADE'
                    entry_conditions = ['still_consolidating']
                    reason = "Consolidation strategy - still consolidating"
                    entry_score = 0.1
            
            # Next candle prediction
            next_candle_prediction = await self._predict_consolidation_movement(candle_features)
            
            return {
                'name': 'CONSOLIDATION_BREAKOUT',
                'action': action,
                'entry_conditions': entry_conditions,
                'entry_score': entry_score,
                'trend_state': 'consolidating',
                'reason': reason,
                'confidence_boost': 0.05 if action != 'NO_TRADE' else 0,
                'next_candle_prediction': next_candle_prediction
            }
            
        except Exception as e:
            self.logger.error(f"Error creating consolidation strategy: {e}")
            return await self._create_default_strategy()
    
    async def _create_breakout_strategy(self, candle_features: Dict, pattern_signals: Dict,
                                      psychology_score: Dict, sr_signals: Dict) -> Dict[str, Any]:
        """Create breakout strategy"""
        try:
            # Breakout strategy capitalizes on momentum continuation
            size_analysis = candle_features.get('size_analysis', {})
            volume = candle_features.get('volume_analysis', {})
            momentum = candle_features.get('momentum_signals', {})
            
            size_breakout = size_analysis.get('is_size_breakout', False)
            volume_spike = volume.get('volume_spike', False)
            momentum_direction = momentum.get('momentum_direction', 'neutral')
            
            # Check for breakout patterns
            patterns = pattern_signals.get('patterns_detected', [])
            breakout_patterns = [p for p in patterns if 'breakout' in p.get('name', '').lower()]
            
            if size_breakout and volume_spike and breakout_patterns:
                # Strong breakout confirmed
                if momentum_direction == 'bullish':
                    action = 'CALL'
                    entry_conditions = ['confirmed_bullish_breakout', 'volume_confirmation', 'pattern_confirmation']
                    reason = "Breakout strategy - confirmed bullish breakout with volume"
                    entry_score = 0.8
                    
                elif momentum_direction == 'bearish':
                    action = 'PUT'
                    entry_conditions = ['confirmed_bearish_breakout', 'volume_confirmation', 'pattern_confirmation']
                    reason = "Breakout strategy - confirmed bearish breakout with volume"
                    entry_score = 0.8
                    
                else:
                    action = 'NO_TRADE'
                    entry_conditions = ['breakout_detected', 'unclear_direction']
                    reason = "Breakout strategy - breakout detected but direction unclear"
                    entry_score = 0.3
                    
            elif size_breakout or volume_spike:
                # Partial breakout signals
                action = 'NO_TRADE'
                entry_conditions = ['partial_breakout_signals']
                reason = "Breakout strategy - partial breakout signals, waiting for confirmation"
                entry_score = 0.4
                
            else:
                action = 'NO_TRADE'
                entry_conditions = ['no_breakout_detected']
                reason = "Breakout strategy - no breakout detected"
                entry_score = 0.0
            
            # Next candle prediction
            next_candle_prediction = await self._predict_breakout_continuation(candle_features, momentum_direction)
            
            return {
                'name': 'BREAKOUT_MOMENTUM',
                'action': action,
                'entry_conditions': entry_conditions,
                'entry_score': entry_score,
                'trend_state': 'breakout',
                'reason': reason,
                'confidence_boost': 0.2 if entry_score > 0.7 else 0,
                'next_candle_prediction': next_candle_prediction
            }
            
        except Exception as e:
            self.logger.error(f"Error creating breakout strategy: {e}")
            return await self._create_default_strategy()
    
    async def _create_default_strategy(self) -> Dict[str, Any]:
        """Create default no-trade strategy"""
        return {
            'name': 'DEFAULT_NO_TRADE',
            'action': 'NO_TRADE',
            'entry_conditions': ['insufficient_signals'],
            'entry_score': 0.0,
            'trend_state': 'unclear',
            'reason': 'Default strategy - insufficient clear signals for trading',
            'confidence_boost': 0,
            'next_candle_prediction': {'direction': 'UNKNOWN', 'strength': 'WEAK'}
        }
    
    async def _enhance_strategy_with_confluence(self, strategy: Dict, candle_features: Dict,
                                              pattern_signals: Dict, psychology_score: Dict,
                                              sr_signals: Dict) -> Dict[str, Any]:
        """Enhance strategy with confluence analysis"""
        try:
            confluence_factors = []
            
            # Pattern confluence
            patterns = pattern_signals.get('patterns_detected', [])
            if len(patterns) >= 2:
                confluence_factors.append('multiple_patterns')
                strategy['confidence_boost'] += 0.05
            
            # Psychology confluence
            market_psychology = psychology_score.get('market_psychology', {})
            if market_psychology.get('extreme_emotion', False):
                confluence_factors.append('extreme_psychology')
                strategy['confidence_boost'] += 0.03
            
            # Volume confluence
            volume_analysis = candle_features.get('volume_analysis', {})
            if volume_analysis.get('volume_spike', False) and strategy['action'] != 'NO_TRADE':
                confluence_factors.append('volume_confirmation')
                strategy['confidence_boost'] += 0.05
            
            # S/R confluence
            if sr_signals.get('rejection_detected', False) and strategy['action'] != 'NO_TRADE':
                confluence_factors.append('sr_confluence')
                strategy['confidence_boost'] += 0.04
            
            # Time confluence (session analysis)
            current_hour = datetime.now().hour
            if 9 <= current_hour <= 16:  # Active trading hours
                confluence_factors.append('optimal_time')
                strategy['confidence_boost'] += 0.02
            
            strategy['confluence_factors'] = confluence_factors
            strategy['confluence_count'] = len(confluence_factors)
            
            # Cap confidence boost
            strategy['confidence_boost'] = min(0.3, strategy['confidence_boost'])
            
            return strategy
            
        except Exception as e:
            self.logger.error(f"Error enhancing strategy with confluence: {e}")
            return strategy
    
    # Prediction methods for next candle
    async def _predict_trend_continuation(self, candle_features: Dict, direction: str) -> Dict[str, Any]:
        """Predict next candle for trend continuation"""
        momentum = candle_features.get('momentum_signals', {})
        momentum_strength = abs(momentum.get('short_momentum', 0))
        
        if direction == 'bullish':
            predicted_direction = 'UP'
        elif direction == 'bearish':
            predicted_direction = 'DOWN'
        else:
            predicted_direction = 'SIDEWAYS'
        
        strength = 'STRONG' if momentum_strength > 0.02 else 'MODERATE' if momentum_strength > 0.01 else 'WEAK'
        
        return {
            'direction': predicted_direction,
            'strength': strength,
            'confidence': min(1.0, momentum_strength * 50)
        }
    
    async def _predict_range_movement(self, candle_features: Dict, sr_signals: Dict) -> Dict[str, Any]:
        """Predict next candle for range movement"""
        rejection_type = sr_signals.get('rejection_type', 'none')
        rejection_strength = sr_signals.get('rejection_strength', 0)
        
        if rejection_type == 'resistance':
            predicted_direction = 'DOWN'
        elif rejection_type == 'support':
            predicted_direction = 'UP'
        else:
            predicted_direction = 'SIDEWAYS'
        
        strength = 'STRONG' if rejection_strength > 0.8 else 'MODERATE' if rejection_strength > 0.5 else 'WEAK'
        
        return {
            'direction': predicted_direction,
            'strength': strength,
            'confidence': rejection_strength
        }
    
    async def _predict_volatility_movement(self, candle_features: Dict) -> Dict[str, Any]:
        """Predict next candle for volatility movement"""
        size_analysis = candle_features.get('size_analysis', {})
        momentum = candle_features.get('momentum_signals', {})
        
        size_breakout = size_analysis.get('is_size_breakout', False)
        momentum_direction = momentum.get('momentum_direction', 'neutral')
        
        if size_breakout:
            if momentum_direction == 'bullish':
                predicted_direction = 'UP'
            elif momentum_direction == 'bearish':
                predicted_direction = 'DOWN'
            else:
                predicted_direction = 'LARGE_MOVE'
            strength = 'STRONG'
        else:
            predicted_direction = 'SIDEWAYS'
            strength = 'WEAK'
        
        return {
            'direction': predicted_direction,
            'strength': strength,
            'confidence': 0.8 if size_breakout else 0.3
        }
    
    async def _predict_consolidation_movement(self, candle_features: Dict) -> Dict[str, Any]:
        """Predict next candle for consolidation"""
        size_analysis = candle_features.get('size_analysis', {})
        
        size_compression = size_analysis.get('is_size_compression', False)
        
        if size_compression:
            predicted_direction = 'SMALL_MOVE'
            strength = 'WEAK'
            confidence = 0.7
        else:
            predicted_direction = 'SIDEWAYS'
            strength = 'WEAK'
            confidence = 0.5
        
        return {
            'direction': predicted_direction,
            'strength': strength,
            'confidence': confidence
        }
    
    async def _predict_breakout_continuation(self, candle_features: Dict, direction: str) -> Dict[str, Any]:
        """Predict next candle for breakout continuation"""
        volume = candle_features.get('volume_analysis', {})
        volume_spike = volume.get('volume_spike', False)
        
        if direction == 'bullish':
            predicted_direction = 'UP'
        elif direction == 'bearish':
            predicted_direction = 'DOWN'
        else:
            predicted_direction = 'LARGE_MOVE'
        
        strength = 'STRONG' if volume_spike else 'MODERATE'
        confidence = 0.85 if volume_spike else 0.6
        
        return {
            'direction': predicted_direction,
            'strength': strength,
            'confidence': confidence
        }
    
    def get_strategy_statistics(self) -> Dict[str, Any]:
        """Get strategy performance statistics"""
        try:
            if not self.strategy_history:
                return {}
            
            recent_strategies = self.strategy_history[-20:]  # Last 20 strategies
            
            strategy_types = {}
            actions = {'CALL': 0, 'PUT': 0, 'NO_TRADE': 0}
            avg_entry_score = 0
            avg_confidence_boost = 0
            
            for strategy in recent_strategies:
                strategy_name = strategy.get('name', 'UNKNOWN')
                strategy_types[strategy_name] = strategy_types.get(strategy_name, 0) + 1
                
                action = strategy.get('action', 'NO_TRADE')
                actions[action] = actions.get(action, 0) + 1
                
                avg_entry_score += strategy.get('entry_score', 0)
                avg_confidence_boost += strategy.get('confidence_boost', 0)
            
            avg_entry_score /= len(recent_strategies)
            avg_confidence_boost /= len(recent_strategies)
            
            return {
                'current_regime': self.current_regime.value,
                'strategy_distribution': strategy_types,
                'action_distribution': actions,
                'average_entry_score': avg_entry_score,
                'average_confidence_boost': avg_confidence_boost,
                'total_strategies_used': len(recent_strategies)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating strategy statistics: {e}")
            return {}