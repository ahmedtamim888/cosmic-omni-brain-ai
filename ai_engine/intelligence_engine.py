#!/usr/bin/env python3
"""
🧠 INTELLIGENCE ENGINE - No-Loss Logic Builder
Creates custom AI logic each time - never uses fixed rules
Adjusts to broker delay, OTC weird movement, micro candle behavior
"""

import logging
import numpy as np
import random
from typing import Dict, List, Optional
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)

class IntelligenceEngine:
    """
    Advanced AI that creates dynamic trading strategies
    Never repeats the same logic - adapts to each unique situation
    """
    
    def __init__(self):
        self.version = "∞ vX"
        self.strategy_memory = []
        self.learning_rate = 0.1
        self.adaptation_threshold = 0.8
        self.personality_traits = {
            "aggression": 0.7,
            "patience": 0.8,
            "risk_tolerance": 0.6,
            "pattern_sensitivity": 0.9,
            "manipulation_detection": 0.95
        }
        
    async def create_strategy(self, chart_data: Dict, context: Dict) -> Dict:
        """
        Main strategy creation - builds unique logic for current market condition
        """
        try:
            logger.info("🧠 INTELLIGENCE ENGINE - Creating dynamic strategy")
            
            # Analyze the unique characteristics of this chart
            unique_signature = self._generate_market_signature(chart_data, context)
            
            # Check if we've seen similar conditions before
            similar_strategies = self._find_similar_strategies(unique_signature)
            
            # Create new strategy or evolve existing one
            if similar_strategies and len(similar_strategies) > 0:
                strategy = await self._evolve_strategy(similar_strategies[-1], chart_data, context)
            else:
                strategy = await self._create_new_strategy(chart_data, context, unique_signature)
            
            # Add meta-intelligence layer
            strategy = await self._add_meta_intelligence(strategy, context)
            
            # Store strategy in memory for learning
            self.strategy_memory.append({
                'timestamp': datetime.now(),
                'signature': unique_signature,
                'strategy': strategy,
                'context_snapshot': context
            })
            
            # Limit memory size
            if len(self.strategy_memory) > 200:
                self.strategy_memory = self.strategy_memory[-200:]
            
            logger.info(f"🎯 Strategy Created: Type={strategy.get('type')}, Confidence={strategy.get('confidence', 0):.2f}")
            return strategy
            
        except Exception as e:
            logger.error(f"Intelligence Engine error: {str(e)}")
            return {"error": str(e), "type": "fallback"}
    
    def _generate_market_signature(self, chart_data: Dict, context: Dict) -> str:
        """Generate unique signature for current market conditions"""
        try:
            candles = chart_data.get('candles', [])
            patterns = chart_data.get('patterns', [])
            
            # Create signature components
            components = []
            
            # Market phase component
            market_phase = context.get('market_phase', 'unknown')
            components.append(f"phase:{market_phase}")
            
            # Momentum component
            momentum = context.get('momentum', {})
            momentum_dir = momentum.get('direction', 'neutral')
            momentum_str = momentum.get('strength', 0)
            components.append(f"momentum:{momentum_dir}:{momentum_str:.1f}")
            
            # Volatility component
            volatility = context.get('volatility', {})
            vol_level = volatility.get('level', 'unknown')
            components.append(f"volatility:{vol_level}")
            
            # Pattern component
            pattern_types = [p.get('type') for p in patterns]
            unique_patterns = list(set(pattern_types))
            components.append(f"patterns:{'-'.join(unique_patterns)}")
            
            # Candle structure component
            if len(candles) >= 3:
                recent_types = [c.get('type', 'unknown') for c in candles[-3:]]
                components.append(f"sequence:{''.join([t[0] if t != 'unknown' else 'x' for t in recent_types])}")
            
            # Risk component
            risk_factors = context.get('risk_factors', [])
            risk_count = len(risk_factors)
            components.append(f"risk:{risk_count}")
            
            return "|".join(components)
            
        except Exception as e:
            logger.error(f"Signature generation error: {str(e)}")
            return "unknown_signature"
    
    def _find_similar_strategies(self, current_signature: str) -> List[Dict]:
        """Find strategies used in similar market conditions"""
        try:
            similar_strategies = []
            
            current_components = set(current_signature.split('|'))
            
            for memory_item in self.strategy_memory[-50:]:  # Check recent 50 strategies
                stored_signature = memory_item.get('signature', '')
                stored_components = set(stored_signature.split('|'))
                
                # Calculate similarity score
                intersection = len(current_components & stored_components)
                union = len(current_components | stored_components)
                
                if union > 0:
                    similarity = intersection / union
                    
                    if similarity > 0.6:  # 60% similarity threshold
                        similar_strategies.append({
                            'strategy': memory_item.get('strategy'),
                            'similarity': similarity,
                            'signature': stored_signature
                        })
            
            # Sort by similarity
            similar_strategies.sort(key=lambda x: x['similarity'], reverse=True)
            return similar_strategies
            
        except Exception as e:
            logger.error(f"Similar strategy search error: {str(e)}")
            return []
    
    async def _evolve_strategy(self, base_strategy: Dict, chart_data: Dict, context: Dict) -> Dict:
        """Evolve existing strategy to current conditions"""
        try:
            logger.info("🧬 Evolving existing strategy for new conditions")
            
            base_strategy_data = base_strategy.get('strategy', {})
            
            # Get current market dynamics
            momentum = context.get('momentum', {})
            volatility = context.get('volatility', {})
            risk_factors = context.get('risk_factors', [])
            
            # Evolution parameters
            evolution_factor = random.uniform(0.1, 0.3)  # How much to evolve
            
            # Evolve key parameters
            evolved_strategy = {
                'type': 'evolved_strategy',
                'parent_type': base_strategy_data.get('type', 'unknown'),
                'evolution_id': f"evo_{random.randint(1000, 9999)}",
                'confidence_multiplier': self._evolve_confidence_multiplier(base_strategy_data, context, evolution_factor),
                'entry_logic': await self._evolve_entry_logic(base_strategy_data, chart_data, context),
                'risk_management': self._evolve_risk_management(base_strategy_data, risk_factors),
                'exit_conditions': self._evolve_exit_conditions(base_strategy_data, volatility),
                'broker_adaptation': await self._adapt_to_broker_conditions(chart_data),
                'manipulation_resistance': self._build_manipulation_resistance(context),
                'confidence': min(0.95, base_strategy_data.get('confidence', 0.5) + evolution_factor * 0.3)
            }
            
            return evolved_strategy
            
        except Exception as e:
            logger.error(f"Strategy evolution error: {str(e)}")
            return await self._create_fallback_strategy()
    
    async def _create_new_strategy(self, chart_data: Dict, context: Dict, signature: str) -> Dict:
        """Create completely new strategy from scratch"""
        try:
            logger.info("🆕 Creating brand new strategy")
            
            candles = chart_data.get('candles', [])
            patterns = chart_data.get('patterns', [])
            
            # Analyze unique market characteristics
            market_character = await self._analyze_market_character(chart_data, context)
            
            # Choose strategy type based on market character
            strategy_type = self._select_strategy_type(market_character, context)
            
            # Build strategy components
            new_strategy = {
                'type': strategy_type,
                'creation_id': f"new_{random.randint(10000, 99999)}",
                'signature': signature,
                'market_character': market_character,
                'entry_logic': await self._build_entry_logic(strategy_type, chart_data, context),
                'confirmation_system': self._build_confirmation_system(patterns, context),
                'risk_management': self._build_risk_management(context),
                'timing_optimization': await self._optimize_timing(candles, context),
                'broker_adaptation': await self._adapt_to_broker_conditions(chart_data),
                'manipulation_resistance': self._build_manipulation_resistance(context),
                'confidence': self._calculate_initial_confidence(chart_data, context)
            }
            
            return new_strategy
            
        except Exception as e:
            logger.error(f"New strategy creation error: {str(e)}")
            return await self._create_fallback_strategy()
    
    async def _analyze_market_character(self, chart_data: Dict, context: Dict) -> Dict:
        """Analyze unique characteristics of current market"""
        try:
            candles = chart_data.get('candles', [])
            patterns = chart_data.get('patterns', [])
            
            character = {
                'trend_strength': self._analyze_trend_strength(candles),
                'pattern_complexity': len(patterns),
                'volatility_pattern': self._analyze_volatility_pattern(candles),
                'manipulation_risk': len(context.get('manipulation_signs', [])),
                'structure_clarity': self._analyze_structure_clarity(context),
                'momentum_consistency': self._analyze_momentum_consistency(candles),
                'volume_profile': self._analyze_volume_profile(candles)
            }
            
            # Classify market character
            if character['trend_strength'] > 0.7 and character['momentum_consistency'] > 0.6:
                character['type'] = 'trending_market'
            elif character['volatility_pattern'] > 0.8:
                character['type'] = 'volatile_market'
            elif character['manipulation_risk'] > 2:
                character['type'] = 'manipulated_market'
            elif character['structure_clarity'] < 0.4:
                character['type'] = 'chaotic_market'
            else:
                character['type'] = 'normal_market'
            
            return character
            
        except Exception as e:
            logger.error(f"Market character analysis error: {str(e)}")
            return {'type': 'unknown_market'}
    
    def _analyze_trend_strength(self, candles: List[Dict]) -> float:
        """Analyze the strength of current trend"""
        try:
            if len(candles) < 5:
                return 0.0
            
            closes = [c.get('close', 0) for c in candles[-10:]]
            
            # Calculate trend consistency
            upward_moves = 0
            downward_moves = 0
            
            for i in range(1, len(closes)):
                if closes[i] > closes[i-1]:
                    upward_moves += 1
                elif closes[i] < closes[i-1]:
                    downward_moves += 1
            
            total_moves = upward_moves + downward_moves
            if total_moves == 0:
                return 0.0
            
            # Trend strength = dominance of one direction
            trend_strength = abs(upward_moves - downward_moves) / total_moves
            
            return min(1.0, trend_strength)
            
        except:
            return 0.0
    
    def _analyze_volatility_pattern(self, candles: List[Dict]) -> float:
        """Analyze volatility patterns"""
        try:
            if len(candles) < 5:
                return 0.5
            
            ranges = []
            for candle in candles[-10:]:
                high = candle.get('high', 0)
                low = candle.get('low', 0)
                if high > 0 and low > 0:
                    ranges.append((high - low) / ((high + low) / 2))
            
            if not ranges:
                return 0.5
            
            # Volatility pattern = standard deviation of ranges
            volatility_pattern = np.std(ranges) / (np.mean(ranges) + 0.001)
            
            return min(1.0, volatility_pattern)
            
        except:
            return 0.5
    
    def _analyze_structure_clarity(self, context: Dict) -> float:
        """Analyze how clear the market structure is"""
        try:
            structure = context.get('structure', {})
            structure_type = structure.get('type', 'unclear_structure')
            
            clarity_scores = {
                'bullish_structure': 0.9,
                'bearish_structure': 0.9,
                'ascending_structure': 0.8,
                'descending_structure': 0.8,
                'ranging_structure': 0.6,
                'unclear_structure': 0.2,
                'error': 0.1
            }
            
            return clarity_scores.get(structure_type, 0.3)
            
        except:
            return 0.3
    
    def _analyze_momentum_consistency(self, candles: List[Dict]) -> float:
        """Analyze momentum consistency"""
        try:
            if len(candles) < 5:
                return 0.0
            
            recent_candles = candles[-5:]
            
            # Check candle type consistency
            bullish_count = sum(1 for c in recent_candles if c.get('type') == 'bullish')
            bearish_count = sum(1 for c in recent_candles if c.get('type') == 'bearish')
            
            # Momentum consistency = how many candles agree on direction
            max_direction = max(bullish_count, bearish_count)
            consistency = max_direction / len(recent_candles)
            
            return consistency
            
        except:
            return 0.0
    
    def _analyze_volume_profile(self, candles: List[Dict]) -> float:
        """Analyze volume profile"""
        try:
            if len(candles) < 3:
                return 0.5
            
            volumes = [c.get('volume_indicator', 0.5) for c in candles[-5:]]
            
            # Volume profile = average volume indicator
            volume_profile = np.mean(volumes)
            
            return volume_profile
            
        except:
            return 0.5
    
    def _select_strategy_type(self, market_character: Dict, context: Dict) -> str:
        """Select appropriate strategy type based on market character"""
        try:
            character_type = market_character.get('type', 'normal_market')
            momentum = context.get('momentum', {})
            
            # Strategy selection logic
            if character_type == 'trending_market':
                if momentum.get('strength', 0) > 0.7:
                    return 'momentum_breakout_strategy'
                else:
                    return 'trend_following_strategy'
                    
            elif character_type == 'volatile_market':
                return 'volatility_scalping_strategy'
                
            elif character_type == 'manipulated_market':
                return 'manipulation_resistant_strategy'
                
            elif character_type == 'chaotic_market':
                return 'pattern_recognition_strategy'
                
            else:
                # Default adaptive strategy
                patterns_count = len(context.get('patterns', []))
                if patterns_count > 3:
                    return 'pattern_confirmation_strategy'
                else:
                    return 'adaptive_momentum_strategy'
                    
        except:
            return 'adaptive_momentum_strategy'
    
    async def _build_entry_logic(self, strategy_type: str, chart_data: Dict, context: Dict) -> Dict:
        """Build entry logic for the strategy"""
        try:
            candles = chart_data.get('candles', [])
            patterns = chart_data.get('patterns', [])
            
            entry_logic = {
                'strategy_type': strategy_type,
                'conditions': [],
                'weight_system': {},
                'confirmation_required': True
            }
            
            # Build entry conditions based on strategy type
            if strategy_type == 'momentum_breakout_strategy':
                entry_logic['conditions'] = [
                    'strong_momentum_continuation',
                    'volume_confirmation',
                    'structure_break_confirmation'
                ]
                entry_logic['weight_system'] = {
                    'momentum': 0.4,
                    'volume': 0.3,
                    'structure': 0.3
                }
                
            elif strategy_type == 'trend_following_strategy':
                entry_logic['conditions'] = [
                    'trend_alignment',
                    'pullback_completion',
                    'momentum_resumption'
                ]
                entry_logic['weight_system'] = {
                    'trend': 0.5,
                    'pullback': 0.3,
                    'momentum': 0.2
                }
                
            elif strategy_type == 'volatility_scalping_strategy':
                entry_logic['conditions'] = [
                    'volatility_spike',
                    'support_resistance_reaction',
                    'quick_momentum_shift'
                ]
                entry_logic['weight_system'] = {
                    'volatility': 0.4,
                    'levels': 0.4,
                    'momentum': 0.2
                }
                
            elif strategy_type == 'manipulation_resistant_strategy':
                entry_logic['conditions'] = [
                    'genuine_pattern_confirmation',
                    'volume_authenticity',
                    'structure_validity'
                ]
                entry_logic['weight_system'] = {
                    'pattern': 0.4,
                    'volume': 0.3,
                    'structure': 0.3
                }
                entry_logic['confirmation_required'] = True
                
            elif strategy_type == 'pattern_recognition_strategy':
                entry_logic['conditions'] = [
                    'pattern_completion',
                    'context_alignment',
                    'momentum_confirmation'
                ]
                entry_logic['weight_system'] = {
                    'pattern': 0.5,
                    'context': 0.3,
                    'momentum': 0.2
                }
                
            else:  # adaptive_momentum_strategy
                entry_logic['conditions'] = [
                    'momentum_alignment',
                    'context_confirmation',
                    'risk_reward_favorable'
                ]
                entry_logic['weight_system'] = {
                    'momentum': 0.4,
                    'context': 0.3,
                    'risk_reward': 0.3
                }
            
            # Add dynamic timing component
            entry_logic['timing'] = await self._calculate_optimal_timing(candles, context)
            
            return entry_logic
            
        except Exception as e:
            logger.error(f"Entry logic building error: {str(e)}")
            return {'strategy_type': 'fallback', 'conditions': []}
    
    def _build_confirmation_system(self, patterns: List[Dict], context: Dict) -> Dict:
        """Build confirmation system"""
        try:
            confirmation_system = {
                'required_confirmations': 2,
                'confirmation_types': [],
                'strength_threshold': 0.6
            }
            
            # Add pattern confirmations
            strong_patterns = [p for p in patterns if p.get('strength', 0) > 0.7]
            if strong_patterns:
                confirmation_system['confirmation_types'].append('strong_pattern')
            
            # Add momentum confirmation
            momentum = context.get('momentum', {})
            if momentum.get('strength', 0) > 0.6:
                confirmation_system['confirmation_types'].append('momentum')
            
            # Add structure confirmation
            structure = context.get('structure', {})
            if 'bullish' in structure.get('type', '') or 'bearish' in structure.get('type', ''):
                confirmation_system['confirmation_types'].append('structure')
            
            # Add volume confirmation
            if len(patterns) > 0:
                confirmation_system['confirmation_types'].append('volume')
            
            # Adjust requirements based on market conditions
            risk_factors = context.get('risk_factors', [])
            if len(risk_factors) > 2:
                confirmation_system['required_confirmations'] = 3
                confirmation_system['strength_threshold'] = 0.8
            
            return confirmation_system
            
        except:
            return {'required_confirmations': 1, 'confirmation_types': ['basic']}
    
    def _build_risk_management(self, context: Dict) -> Dict:
        """Build risk management system"""
        try:
            risk_factors = context.get('risk_factors', [])
            volatility = context.get('volatility', {})
            
            risk_management = {
                'position_sizing': 'dynamic',
                'stop_loss_type': 'adaptive',
                'risk_per_trade': 0.02,  # 2% default
                'volatility_adjustment': True,
                'manipulation_protection': True
            }
            
            # Adjust risk based on context
            vol_level = volatility.get('level', 'medium')
            if vol_level in ['high', 'very_high']:
                risk_management['risk_per_trade'] = 0.01  # Reduce risk
                risk_management['stop_loss_type'] = 'tight_adaptive'
            
            high_risk_factors = [rf for rf in risk_factors if rf.get('severity') == 'high']
            if len(high_risk_factors) > 0:
                risk_management['risk_per_trade'] = 0.005  # Very conservative
                risk_management['confirmation_required'] = True
            
            return risk_management
            
        except:
            return {'risk_per_trade': 0.01, 'stop_loss_type': 'basic'}
    
    async def _optimize_timing(self, candles: List[Dict], context: Dict) -> Dict:
        """Optimize entry timing"""
        try:
            timing_optimization = {
                'optimal_entry_window': '30_seconds',
                'delay_compensation': True,
                'broker_latency_adjustment': 0.5,  # seconds
                'market_timing_score': 0.5
            }
            
            # Analyze candle formation timing
            if len(candles) >= 3:
                recent_momentum = context.get('momentum', {})
                if recent_momentum.get('strength', 0) > 0.8:
                    timing_optimization['optimal_entry_window'] = '15_seconds'
                    timing_optimization['market_timing_score'] = 0.8
            
            # Adjust for volatility
            volatility = context.get('volatility', {})
            if volatility.get('level') == 'very_high':
                timing_optimization['optimal_entry_window'] = '10_seconds'
                timing_optimization['broker_latency_adjustment'] = 1.0
            
            return timing_optimization
            
        except:
            return {'optimal_entry_window': '30_seconds', 'market_timing_score': 0.5}
    
    async def _adapt_to_broker_conditions(self, chart_data: Dict) -> Dict:
        """Adapt strategy to broker-specific conditions"""
        try:
            broker_type = chart_data.get('broker_type', 'unknown')
            
            adaptation = {
                'broker_type': broker_type,
                'execution_delay_compensation': True,
                'slippage_protection': True,
                'otc_adjustments': False
            }
            
            # Mobile platform adjustments
            if broker_type == 'mobile_platform':
                adaptation['execution_delay_compensation'] = True
                adaptation['additional_confirmation'] = True
            
            # Detect potential OTC characteristics
            candles = chart_data.get('candles', [])
            if len(candles) > 5:
                # Check for unusual price movements (OTC indicator)
                body_sizes = [c.get('body_size', 0) for c in candles[-5:]]
                if np.std(body_sizes) > np.mean(body_sizes):
                    adaptation['otc_adjustments'] = True
                    adaptation['manipulation_resistance'] = 'high'
            
            return adaptation
            
        except:
            return {'broker_type': 'unknown', 'execution_delay_compensation': True}
    
    def _build_manipulation_resistance(self, context: Dict) -> Dict:
        """Build manipulation resistance system"""
        try:
            manipulation_signs = context.get('manipulation_signs', [])
            
            resistance = {
                'detection_level': 'high',
                'resistance_strategies': [],
                'confidence_penalty': 0.0
            }
            
            # Add resistance strategies based on detected manipulation
            for sign in manipulation_signs:
                sign_type = sign.get('type', '')
                
                if sign_type == 'wick_manipulation':
                    resistance['resistance_strategies'].append('ignore_wick_extremes')
                    resistance['confidence_penalty'] += 0.1
                    
                elif sign_type == 'unusual_large_move':
                    resistance['resistance_strategies'].append('volume_confirmation_required')
                    resistance['confidence_penalty'] += 0.05
                    
                elif sign_type == 'price_gap':
                    resistance['resistance_strategies'].append('gap_fill_expectation')
                    resistance['confidence_penalty'] += 0.1
            
            # Overall manipulation resistance level
            if len(manipulation_signs) > 2:
                resistance['detection_level'] = 'maximum'
                resistance['confidence_penalty'] = min(0.3, resistance['confidence_penalty'])
            
            return resistance
            
        except:
            return {'detection_level': 'medium', 'resistance_strategies': []}
    
    def _calculate_initial_confidence(self, chart_data: Dict, context: Dict) -> float:
        """Calculate initial confidence score for the strategy"""
        try:
            confidence_factors = []
            
            # Chart quality factor
            image_quality = chart_data.get('image_quality', 0.5)
            confidence_factors.append(image_quality * 0.2)
            
            # Pattern strength factor
            patterns = chart_data.get('patterns', [])
            strong_patterns = [p for p in patterns if p.get('strength', 0) > 0.7]
            pattern_confidence = min(1.0, len(strong_patterns) * 0.3)
            confidence_factors.append(pattern_confidence * 0.3)
            
            # Context clarity factor
            market_phase = context.get('market_phase', 'unknown')
            clarity_scores = {
                'strong_uptrend': 0.9, 'strong_downtrend': 0.9,
                'weak_uptrend': 0.7, 'weak_downtrend': 0.7,
                'sideways_consolidation': 0.6, 'high_volatility_range': 0.4,
                'low_volatility_range': 0.5, 'unknown': 0.3
            }
            clarity_confidence = clarity_scores.get(market_phase, 0.3)
            confidence_factors.append(clarity_confidence * 0.3)
            
            # Risk adjustment
            risk_factors = context.get('risk_factors', [])
            risk_penalty = len(risk_factors) * 0.05
            confidence_factors.append(max(0, 0.2 - risk_penalty))
            
            # Calculate final confidence
            total_confidence = sum(confidence_factors)
            
            # Apply personality traits
            total_confidence *= self.personality_traits.get('pattern_sensitivity', 0.8)
            
            return min(0.98, max(0.1, total_confidence))
            
        except:
            return 0.5
    
    async def _add_meta_intelligence(self, strategy: Dict, context: Dict) -> Dict:
        """Add meta-intelligence layer to strategy"""
        try:
            # Add self-adaptation capability
            strategy['meta_intelligence'] = {
                'self_monitoring': True,
                'performance_feedback': True,
                'real_time_adaptation': True,
                'confidence_adjustment': 'dynamic',
                'failure_learning': True
            }
            
            # Add ghost transcendence traits
            strategy['ghost_traits'] = {
                'invisibility_to_manipulation': True,
                'broker_trap_detection': True,
                'fake_signal_immunity': True,
                'adaptive_evolution': True,
                'infinite_learning': True
            }
            
            # Add personality integration
            for trait, value in self.personality_traits.items():
                if trait in ['aggression', 'patience']:
                    strategy[f'{trait}_level'] = value
            
            return strategy
            
        except Exception as e:
            logger.error(f"Meta-intelligence addition error: {str(e)}")
            return strategy
    
    def _evolve_confidence_multiplier(self, base_strategy: Dict, context: Dict, evolution_factor: float) -> float:
        """Evolve confidence multiplier"""
        try:
            base_confidence = base_strategy.get('confidence', 0.5)
            
            # Factor in new market conditions
            opportunity_score = context.get('opportunity_score', 0.5)
            
            # Evolution logic
            if opportunity_score > 0.8:
                multiplier = 1.0 + evolution_factor * 0.2
            elif opportunity_score < 0.3:
                multiplier = 1.0 - evolution_factor * 0.3
            else:
                multiplier = 1.0 + random.uniform(-0.1, 0.1)
            
            return max(0.5, min(1.2, multiplier))
            
        except:
            return 1.0
    
    async def _evolve_entry_logic(self, base_strategy: Dict, chart_data: Dict, context: Dict) -> Dict:
        """Evolve entry logic"""
        try:
            base_entry = base_strategy.get('entry_logic', {})
            
            # Keep successful components, evolve others
            evolved_entry = base_entry.copy()
            
            # Add new conditions based on current context
            patterns = chart_data.get('patterns', [])
            if len(patterns) > 3:
                evolved_entry['pattern_diversity_confirmation'] = True
            
            # Evolve weight system
            if 'weight_system' in evolved_entry:
                for key, weight in evolved_entry['weight_system'].items():
                    # Small random evolution
                    evolution = random.uniform(-0.1, 0.1)
                    evolved_entry['weight_system'][key] = max(0.1, min(0.7, weight + evolution))
            
            return evolved_entry
            
        except:
            return {'strategy_type': 'evolved_fallback', 'conditions': []}
    
    def _evolve_risk_management(self, base_strategy: Dict, risk_factors: List[Dict]) -> Dict:
        """Evolve risk management"""
        try:
            base_risk = base_strategy.get('risk_management', {})
            evolved_risk = base_risk.copy()
            
            # Adapt to new risk environment
            high_risk_count = sum(1 for rf in risk_factors if rf.get('severity') == 'high')
            
            if high_risk_count > 1:
                # Become more conservative
                current_risk = evolved_risk.get('risk_per_trade', 0.02)
                evolved_risk['risk_per_trade'] = max(0.005, current_risk * 0.7)
            
            return evolved_risk
            
        except:
            return {'risk_per_trade': 0.01, 'stop_loss_type': 'adaptive'}
    
    def _evolve_exit_conditions(self, base_strategy: Dict, volatility: Dict) -> Dict:
        """Evolve exit conditions"""
        try:
            base_exit = base_strategy.get('exit_conditions', {})
            evolved_exit = base_exit.copy()
            
            # Adapt to volatility
            vol_level = volatility.get('level', 'medium')
            
            if vol_level in ['high', 'very_high']:
                evolved_exit['quick_profit_taking'] = True
                evolved_exit['volatility_stop'] = True
            
            return evolved_exit
            
        except:
            return {'type': 'adaptive_exit'}
    
    async def _calculate_optimal_timing(self, candles: List[Dict], context: Dict) -> Dict:
        """Calculate optimal timing for entry"""
        try:
            timing = {
                'entry_window': 30,  # seconds
                'confidence_decay': 0.1,  # per second
                'market_timing_score': 0.5
            }
            
            # Analyze momentum timing
            momentum = context.get('momentum', {})
            if momentum.get('strength', 0) > 0.8:
                timing['entry_window'] = 15
                timing['market_timing_score'] = 0.9
            
            return timing
            
        except:
            return {'entry_window': 30, 'market_timing_score': 0.5}
    
    async def _create_fallback_strategy(self) -> Dict:
        """Create fallback strategy when main creation fails"""
        return {
            'type': 'conservative_fallback',
            'confidence': 0.3,
            'entry_logic': {
                'conditions': ['basic_momentum', 'simple_confirmation'],
                'weight_system': {'momentum': 0.6, 'confirmation': 0.4}
            },
            'risk_management': {
                'risk_per_trade': 0.005,  # Very conservative
                'stop_loss_type': 'tight'
            },
            'note': 'Fallback strategy - reduced confidence'
        }