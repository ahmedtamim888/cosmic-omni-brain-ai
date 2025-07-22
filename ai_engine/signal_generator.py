#!/usr/bin/env python3
"""
🎯 SIGNAL GENERATOR - Final Decision Maker
Makes final CALL/PUT decisions based on strategy and context
Always optimized for profit, never random guesses
"""

import logging
import numpy as np
import random
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import asyncio

logger = logging.getLogger(__name__)

class SignalGenerator:
    """
    Final decision engine that generates precise trading signals
    """
    
    def __init__(self):
        self.version = "∞ vX"
        self.signal_history = []
        self.success_rate = 0.0
        self.confidence_calibration = 1.0
        self.minimum_confidence = 0.75  # Only signal when very confident
        
    async def generate_signal(self, strategy: Dict, context: Dict) -> Dict:
        """
        Generate final trading signal based on strategy and context
        """
        try:
            logger.info("🎯 SIGNAL GENERATOR - Generating final signal")
            
            # Evaluate all signal components
            signal_components = await self._evaluate_signal_components(strategy, context)
            
            # Calculate overall signal strength
            signal_strength = self._calculate_signal_strength(signal_components, strategy)
            
            # Determine signal direction
            signal_direction = self._determine_signal_direction(signal_components, context)
            
            # Calculate final confidence
            final_confidence = self._calculate_final_confidence(signal_strength, strategy, context)
            
            # Generate time prediction
            time_prediction = self._generate_time_prediction(context)
            
            # Make final decision
            if final_confidence >= self.minimum_confidence:
                signal_type = "CALL" if signal_direction > 0 else "PUT"
            else:
                signal_type = "NO SIGNAL"
                final_confidence = 0.0
            
            # Generate reasoning
            reasoning = self._generate_reasoning(signal_components, strategy, context, signal_type)
            
            # Build final signal
            final_signal = {
                "signal": signal_type,
                "confidence": final_confidence * 100,  # Convert to percentage
                "timeframe": "1M",
                "time_target": time_prediction,
                "reasoning": reasoning,
                "strategy_type": strategy.get('type', 'unknown'),
                "market_conditions": self._summarize_market_conditions(context),
                "risk_assessment": self._assess_signal_risk(signal_components, context),
                "entry_timing": self._calculate_entry_timing(strategy, context),
                "ghost_factor": self._calculate_ghost_factor(strategy, context)
            }
            
            # Store signal for learning
            self._store_signal_for_learning(final_signal, signal_components, strategy, context)
            
            logger.info(f"🎯 Final Signal: {signal_type} | Confidence: {final_confidence:.1%}")
            return final_signal
            
        except Exception as e:
            logger.error(f"Signal generation error: {str(e)}")
            return {
                "signal": "NO SIGNAL",
                "confidence": 0.0,
                "error": str(e),
                "reasoning": "Signal generation failed due to technical error"
            }
    
    async def _evaluate_signal_components(self, strategy: Dict, context: Dict) -> Dict:
        """Evaluate all components that contribute to signal decision"""
        try:
            components = {}
            
            # Strategy-based components
            components['strategy_confidence'] = strategy.get('confidence', 0.5)
            components['strategy_type_score'] = self._score_strategy_type(strategy.get('type', 'unknown'))
            
            # Market context components
            momentum = context.get('momentum', {})
            components['momentum_strength'] = momentum.get('strength', 0)
            components['momentum_direction'] = 1 if momentum.get('direction') == 'bullish' else -1 if momentum.get('direction') == 'bearish' else 0
            
            # Pattern components
            patterns = context.get('patterns', [])
            components['pattern_strength'] = self._evaluate_pattern_strength(patterns)
            components['pattern_direction'] = self._evaluate_pattern_direction(patterns)
            
            # Market structure components
            structure = context.get('structure', {})
            components['structure_clarity'] = self._evaluate_structure_clarity(structure)
            components['structure_direction'] = self._evaluate_structure_direction(structure)
            
            # Risk components
            risk_factors = context.get('risk_factors', [])
            components['risk_penalty'] = len(risk_factors) * 0.1
            components['manipulation_risk'] = len(context.get('manipulation_signs', [])) * 0.05
            
            # Volatility components
            volatility = context.get('volatility', {})
            components['volatility_favorability'] = self._evaluate_volatility_favorability(volatility)
            
            # Opportunity components
            components['opportunity_score'] = context.get('opportunity_score', 0.5)
            
            # Meta-intelligence components
            components['ghost_transcendence'] = self._evaluate_ghost_transcendence(strategy)
            
            return components
            
        except Exception as e:
            logger.error(f"Signal component evaluation error: {str(e)}")
            return {}
    
    def _score_strategy_type(self, strategy_type: str) -> float:
        """Score strategy type effectiveness"""
        strategy_scores = {
            'momentum_breakout_strategy': 0.9,
            'trend_following_strategy': 0.8,
            'volatility_scalping_strategy': 0.7,
            'manipulation_resistant_strategy': 0.95,
            'pattern_recognition_strategy': 0.85,
            'pattern_confirmation_strategy': 0.8,
            'adaptive_momentum_strategy': 0.75,
            'evolved_strategy': 0.85,
            'conservative_fallback': 0.4,
            'unknown': 0.3
        }
        
        return strategy_scores.get(strategy_type, 0.5)
    
    def _evaluate_pattern_strength(self, patterns: List[Dict]) -> float:
        """Evaluate overall pattern strength"""
        if not patterns:
            return 0.0
        
        strong_patterns = [p for p in patterns if p.get('strength', 0) > 0.7]
        total_strength = sum(p.get('strength', 0) for p in patterns)
        
        return min(1.0, total_strength / len(patterns))
    
    def _evaluate_pattern_direction(self, patterns: List[Dict]) -> float:
        """Evaluate pattern directional bias"""
        if not patterns:
            return 0.0
        
        bullish_weight = 0
        bearish_weight = 0
        
        for pattern in patterns:
            signal = pattern.get('signal', '')
            strength = pattern.get('strength', 0)
            
            if 'bullish' in signal:
                bullish_weight += strength
            elif 'bearish' in signal:
                bearish_weight += strength
        
        if bullish_weight + bearish_weight == 0:
            return 0.0
        
        return (bullish_weight - bearish_weight) / (bullish_weight + bearish_weight)
    
    def _evaluate_structure_clarity(self, structure: Dict) -> float:
        """Evaluate market structure clarity"""
        structure_type = structure.get('type', 'unclear_structure')
        
        clarity_scores = {
            'bullish_structure': 0.95,
            'bearish_structure': 0.95,
            'ascending_structure': 0.8,
            'descending_structure': 0.8,
            'ranging_structure': 0.6,
            'unclear_structure': 0.2,
            'insufficient_data': 0.1,
            'error': 0.0
        }
        
        return clarity_scores.get(structure_type, 0.3)
    
    def _evaluate_structure_direction(self, structure: Dict) -> float:
        """Evaluate structure directional bias"""
        structure_type = structure.get('type', 'unclear_structure')
        
        if 'bullish' in structure_type or 'ascending' in structure_type:
            return 1.0
        elif 'bearish' in structure_type or 'descending' in structure_type:
            return -1.0
        else:
            return 0.0
    
    def _evaluate_volatility_favorability(self, volatility: Dict) -> float:
        """Evaluate if volatility is favorable for trading"""
        vol_level = volatility.get('level', 'medium')
        
        favorability_scores = {
            'very_low': 0.3,    # Too quiet
            'low': 0.6,         # Good for precision
            'medium': 0.9,      # Ideal
            'high': 0.7,        # Good but risky
            'very_high': 0.4,   # Too chaotic
            'unknown': 0.5
        }
        
        return favorability_scores.get(vol_level, 0.5)
    
    def _evaluate_ghost_transcendence(self, strategy: Dict) -> float:
        """Evaluate ghost transcendence capabilities"""
        ghost_traits = strategy.get('ghost_traits', {})
        meta_intelligence = strategy.get('meta_intelligence', {})
        
        transcendence_score = 0.0
        
        # Ghost traits scoring
        if ghost_traits.get('invisibility_to_manipulation'):
            transcendence_score += 0.2
        if ghost_traits.get('broker_trap_detection'):
            transcendence_score += 0.2
        if ghost_traits.get('fake_signal_immunity'):
            transcendence_score += 0.2
        if ghost_traits.get('adaptive_evolution'):
            transcendence_score += 0.2
        if ghost_traits.get('infinite_learning'):
            transcendence_score += 0.2
        
        return transcendence_score
    
    def _calculate_signal_strength(self, components: Dict, strategy: Dict) -> float:
        """Calculate overall signal strength"""
        try:
            # Base strength from strategy confidence
            base_strength = components.get('strategy_confidence', 0.5)
            
            # Strategy type multiplier
            strategy_multiplier = components.get('strategy_type_score', 0.5)
            
            # Momentum contribution
            momentum_contribution = components.get('momentum_strength', 0) * 0.3
            
            # Pattern contribution
            pattern_contribution = components.get('pattern_strength', 0) * 0.25
            
            # Structure contribution
            structure_contribution = components.get('structure_clarity', 0) * 0.2
            
            # Opportunity contribution
            opportunity_contribution = components.get('opportunity_score', 0.5) * 0.15
            
            # Ghost transcendence boost
            ghost_boost = components.get('ghost_transcendence', 0) * 0.1
            
            # Volatility adjustment
            volatility_adjustment = components.get('volatility_favorability', 0.5)
            
            # Calculate combined strength
            combined_strength = (
                base_strength * strategy_multiplier +
                momentum_contribution +
                pattern_contribution +
                structure_contribution +
                opportunity_contribution +
                ghost_boost
            ) * volatility_adjustment
            
            # Apply risk penalties
            risk_penalty = components.get('risk_penalty', 0)
            manipulation_penalty = components.get('manipulation_risk', 0)
            
            final_strength = max(0.0, combined_strength - risk_penalty - manipulation_penalty)
            
            return min(1.0, final_strength)
            
        except Exception as e:
            logger.error(f"Signal strength calculation error: {str(e)}")
            return 0.0
    
    def _determine_signal_direction(self, components: Dict, context: Dict) -> float:
        """Determine signal direction (-1 for PUT, +1 for CALL)"""
        try:
            directional_signals = []
            
            # Momentum direction
            momentum_dir = components.get('momentum_direction', 0)
            momentum_strength = components.get('momentum_strength', 0)
            directional_signals.append(momentum_dir * momentum_strength * 0.4)
            
            # Pattern direction
            pattern_dir = components.get('pattern_direction', 0)
            pattern_strength = components.get('pattern_strength', 0)
            directional_signals.append(pattern_dir * pattern_strength * 0.3)
            
            # Structure direction
            structure_dir = components.get('structure_direction', 0)
            structure_clarity = components.get('structure_clarity', 0)
            directional_signals.append(structure_dir * structure_clarity * 0.2)
            
            # Market sentiment
            sentiment = context.get('sentiment', {})
            sentiment_dir = 1 if sentiment.get('sentiment') == 'bullish' else -1 if sentiment.get('sentiment') == 'bearish' else 0
            sentiment_confidence = sentiment.get('confidence', 0)
            directional_signals.append(sentiment_dir * sentiment_confidence * 0.1)
            
            # Combine all directional signals
            final_direction = sum(directional_signals)
            
            return max(-1.0, min(1.0, final_direction))
            
        except Exception as e:
            logger.error(f"Signal direction determination error: {str(e)}")
            return 0.0
    
    def _calculate_final_confidence(self, signal_strength: float, strategy: Dict, context: Dict) -> float:
        """Calculate final confidence score"""
        try:
            # Start with signal strength
            confidence = signal_strength
            
            # Apply strategy-specific adjustments
            strategy_type = strategy.get('type', 'unknown')
            
            if strategy_type == 'manipulation_resistant_strategy':
                # High confidence in manipulation resistance
                confidence *= 1.1
            elif strategy_type == 'conservative_fallback':
                # Lower confidence in fallback
                confidence *= 0.7
            elif strategy_type == 'evolved_strategy':
                # Higher confidence in evolved strategies
                confidence *= 1.05
            
            # Apply market condition adjustments
            market_phase = context.get('market_phase', 'unknown')
            
            if market_phase in ['strong_uptrend', 'strong_downtrend']:
                confidence *= 1.1  # Higher confidence in strong trends
            elif market_phase == 'sideways_consolidation':
                confidence *= 0.9  # Lower confidence in sideways markets
            elif market_phase == 'unknown':
                confidence *= 0.8  # Lower confidence when phase is unclear
            
            # Apply risk adjustments
            risk_factors = context.get('risk_factors', [])
            high_risk_count = sum(1 for rf in risk_factors if rf.get('severity') == 'high')
            confidence *= max(0.5, 1.0 - high_risk_count * 0.15)
            
            # Apply manipulation adjustments
            manipulation_signs = context.get('manipulation_signs', [])
            if len(manipulation_signs) > 2:
                confidence *= 0.8  # Reduce confidence when manipulation detected
            
            # Apply self-calibration
            confidence *= self.confidence_calibration
            
            # Ensure confidence is within bounds
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logger.error(f"Final confidence calculation error: {str(e)}")
            return 0.0
    
    def _generate_time_prediction(self, context: Dict) -> str:
        """Generate time prediction for the signal"""
        try:
            current_time = datetime.now()
            
            # Analyze momentum for timing
            momentum = context.get('momentum', {})
            momentum_strength = momentum.get('strength', 0)
            
            # Analyze volatility for timing
            volatility = context.get('volatility', {})
            vol_level = volatility.get('level', 'medium')
            
            # Calculate optimal entry time
            if momentum_strength > 0.8:
                # Strong momentum - enter quickly
                target_time = current_time + timedelta(seconds=random.randint(15, 30))
                time_desc = "Next candle (Strong momentum)"
            elif vol_level in ['high', 'very_high']:
                # High volatility - be more precise with timing
                target_time = current_time + timedelta(seconds=random.randint(20, 40))
                time_desc = "Next candle (High volatility)"
            else:
                # Normal conditions
                target_time = current_time + timedelta(seconds=random.randint(30, 60))
                time_desc = "Next 1-2 candles"
            
            return f"{target_time.strftime('%H:%M')} | {time_desc}"
            
        except:
            current_time = datetime.now()
            return f"{current_time.strftime('%H:%M')} | Next candle"
    
    def _generate_reasoning(self, components: Dict, strategy: Dict, context: Dict, signal_type: str) -> str:
        """Generate human-readable reasoning for the signal"""
        try:
            if signal_type == "NO SIGNAL":
                return self._generate_no_signal_reasoning(components, context)
            
            reasoning_parts = []
            
            # Strategy reasoning
            strategy_type = strategy.get('type', 'unknown')
            reasoning_parts.append(f"🧠 Strategy: {strategy_type.replace('_', ' ').title()}")
            
            # Market phase reasoning
            market_phase = context.get('market_phase', 'unknown')
            reasoning_parts.append(f"📊 Market Phase: {market_phase.replace('_', ' ').title()}")
            
            # Momentum reasoning
            momentum = context.get('momentum', {})
            momentum_dir = momentum.get('direction', 'neutral')
            momentum_str = momentum.get('strength', 0)
            reasoning_parts.append(f"⚡ Momentum: {momentum_dir.title()} ({momentum_str:.1%} strength)")
            
            # Pattern reasoning
            patterns = context.get('patterns', [])
            if patterns:
                strong_patterns = [p for p in patterns if p.get('strength', 0) > 0.7]
                if strong_patterns:
                    pattern_names = [p.get('type', 'unknown').replace('_', ' ').title() for p in strong_patterns[:2]]
                    reasoning_parts.append(f"📈 Patterns: {', '.join(pattern_names)}")
            
            # Structure reasoning
            structure = context.get('structure', {})
            structure_type = structure.get('type', 'unclear')
            if structure_type != 'unclear':
                reasoning_parts.append(f"🏗️ Structure: {structure_type.replace('_', ' ').title()}")
            
            # Risk reasoning
            risk_factors = context.get('risk_factors', [])
            if risk_factors:
                reasoning_parts.append(f"⚠️ Risk Factors: {len(risk_factors)} identified")
            
            # Ghost transcendence reasoning
            ghost_traits = strategy.get('ghost_traits', {})
            if ghost_traits.get('invisibility_to_manipulation'):
                reasoning_parts.append("👻 Manipulation resistance active")
            
            # Confidence reasoning
            confidence = components.get('strategy_confidence', 0)
            reasoning_parts.append(f"🎯 AI Confidence: {confidence:.1%}")
            
            return "\n".join(reasoning_parts)
            
        except Exception as e:
            logger.error(f"Reasoning generation error: {str(e)}")
            return f"Advanced AI analysis completed. Signal: {signal_type}"
    
    def _generate_no_signal_reasoning(self, components: Dict, context: Dict) -> str:
        """Generate reasoning for NO SIGNAL decision"""
        try:
            reasons = []
            
            # Check low confidence
            confidence = components.get('strategy_confidence', 0)
            if confidence < self.minimum_confidence:
                reasons.append(f"Confidence below threshold ({confidence:.1%} < {self.minimum_confidence:.1%})")
            
            # Check high risk
            risk_factors = context.get('risk_factors', [])
            if len(risk_factors) > 2:
                reasons.append(f"Multiple risk factors detected ({len(risk_factors)})")
            
            # Check manipulation
            manipulation_signs = context.get('manipulation_signs', [])
            if len(manipulation_signs) > 1:
                reasons.append("Market manipulation signs detected")
            
            # Check unclear market
            market_phase = context.get('market_phase', 'unknown')
            if market_phase == 'unknown':
                reasons.append("Market phase unclear")
            
            # Check conflicting signals
            momentum_dir = components.get('momentum_direction', 0)
            pattern_dir = components.get('pattern_direction', 0)
            if abs(momentum_dir - pattern_dir) > 1.0:
                reasons.append("Conflicting momentum and pattern signals")
            
            if reasons:
                return "🚫 NO SIGNAL: " + " | ".join(reasons[:3])
            else:
                return "🚫 NO SIGNAL: Market conditions not optimal for high-confidence trade"
                
        except:
            return "🚫 NO SIGNAL: Analysis inconclusive"
    
    def _summarize_market_conditions(self, context: Dict) -> Dict:
        """Summarize current market conditions"""
        try:
            return {
                "phase": context.get('market_phase', 'unknown'),
                "momentum": context.get('momentum', {}).get('direction', 'neutral'),
                "volatility": context.get('volatility', {}).get('level', 'unknown'),
                "sentiment": context.get('sentiment', {}).get('sentiment', 'neutral'),
                "structure": context.get('structure', {}).get('type', 'unclear'),
                "risk_level": "high" if len(context.get('risk_factors', [])) > 2 else "medium" if len(context.get('risk_factors', [])) > 0 else "low"
            }
        except:
            return {"phase": "unknown", "risk_level": "unknown"}
    
    def _assess_signal_risk(self, components: Dict, context: Dict) -> Dict:
        """Assess risk level of the signal"""
        try:
            risk_factors = context.get('risk_factors', [])
            manipulation_signs = context.get('manipulation_signs', [])
            
            # Calculate risk score
            risk_score = 0.0
            risk_score += len(risk_factors) * 0.1
            risk_score += len(manipulation_signs) * 0.05
            risk_score += components.get('risk_penalty', 0)
            
            # Determine risk level
            if risk_score > 0.3:
                risk_level = "high"
            elif risk_score > 0.15:
                risk_level = "medium"
            else:
                risk_level = "low"
            
            return {
                "level": risk_level,
                "score": min(1.0, risk_score),
                "factors": len(risk_factors),
                "manipulation_risk": len(manipulation_signs) > 1
            }
            
        except:
            return {"level": "unknown", "score": 0.5}
    
    def _calculate_entry_timing(self, strategy: Dict, context: Dict) -> Dict:
        """Calculate optimal entry timing"""
        try:
            timing_optimization = strategy.get('timing_optimization', {})
            
            entry_window = timing_optimization.get('optimal_entry_window', '30_seconds')
            market_timing_score = timing_optimization.get('market_timing_score', 0.5)
            
            # Convert entry window to seconds
            if 'seconds' in entry_window:
                window_seconds = int(entry_window.split('_')[0])
            else:
                window_seconds = 30
            
            return {
                "optimal_window_seconds": window_seconds,
                "timing_score": market_timing_score,
                "delay_compensation": timing_optimization.get('delay_compensation', True),
                "broker_adjustment": timing_optimization.get('broker_latency_adjustment', 0.5)
            }
            
        except:
            return {"optimal_window_seconds": 30, "timing_score": 0.5}
    
    def _calculate_ghost_factor(self, strategy: Dict, context: Dict) -> float:
        """Calculate the ghost transcendence factor"""
        try:
            ghost_traits = strategy.get('ghost_traits', {})
            meta_intelligence = strategy.get('meta_intelligence', {})
            
            ghost_factor = 0.0
            
            # Ghost traits contribution
            trait_count = sum(1 for trait, active in ghost_traits.items() if active)
            ghost_factor += trait_count * 0.15
            
            # Meta-intelligence contribution
            meta_count = sum(1 for feature, active in meta_intelligence.items() if active)
            ghost_factor += meta_count * 0.1
            
            # Strategy evolution bonus
            if strategy.get('type') == 'evolved_strategy':
                ghost_factor += 0.1
            
            # Manipulation resistance bonus
            manipulation_resistance = strategy.get('manipulation_resistance', {})
            if manipulation_resistance.get('detection_level') == 'maximum':
                ghost_factor += 0.2
            
            return min(1.0, ghost_factor)
            
        except:
            return 0.5
    
    def _store_signal_for_learning(self, signal: Dict, components: Dict, strategy: Dict, context: Dict):
        """Store signal for learning and adaptation"""
        try:
            signal_record = {
                'timestamp': datetime.now(),
                'signal': signal,
                'components': components,
                'strategy_snapshot': strategy,
                'context_snapshot': context
            }
            
            self.signal_history.append(signal_record)
            
            # Limit history size
            if len(self.signal_history) > 1000:
                self.signal_history = self.signal_history[-1000:]
            
            # Update success rate and calibration (simplified)
            self._update_confidence_calibration()
            
        except Exception as e:
            logger.error(f"Signal storage error: {str(e)}")
    
    def _update_confidence_calibration(self):
        """Update confidence calibration based on historical performance"""
        try:
            if len(self.signal_history) < 10:
                return
            
            # Simple calibration update (in real system, this would be based on actual trade results)
            recent_signals = self.signal_history[-50:]
            high_confidence_signals = [s for s in recent_signals if s['signal'].get('confidence', 0) > 80]
            
            if len(high_confidence_signals) > 30:
                # Too many high confidence signals, be more conservative
                self.confidence_calibration *= 0.98
            elif len(high_confidence_signals) < 5:
                # Too few high confidence signals, be more aggressive
                self.confidence_calibration *= 1.02
            
            # Keep calibration within reasonable bounds
            self.confidence_calibration = max(0.5, min(1.5, self.confidence_calibration))
            
        except:
            pass