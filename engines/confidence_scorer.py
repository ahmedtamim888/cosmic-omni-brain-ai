"""
Confidence Scorer - ML-based Confidence Calculation Engine
Uses machine learning to score signal confidence with 95%+ threshold
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import asyncio
from datetime import datetime
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

class ConfidenceScorer:
    """
    Advanced ML-based confidence scoring system
    Combines multiple models for ultra-accurate confidence estimation
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # ML Models
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.nn_model = MLPRegressor(hidden_layer_sizes=(50, 30), random_state=42, max_iter=1000)
        
        # Scalers
        self.feature_scaler = StandardScaler()
        
        # Model weights for ensemble
        self.model_weights = {
            'random_forest': 0.4,
            'gradient_boost': 0.4,
            'neural_network': 0.2
        }
        
        # Training data storage
        self.training_features = []
        self.training_scores = []
        
        # Confidence thresholds
        self.min_confidence_threshold = 0.95  # 95% minimum
        self.god_mode_threshold = 0.97       # 97% for God Mode
        
        # Model performance tracking
        self.model_performance = {
            'predictions_made': 0,
            'high_confidence_signals': 0,
            'model_accuracy': 0.0
        }
        
        self.is_trained = False
    
    async def calculate_confidence(self, candle_features: Dict, pattern_signals: Dict,
                                 psychology_score: Dict, sr_signals: Dict,
                                 strategy: Dict) -> float:
        """
        Main confidence calculation function
        Returns confidence score between 0 and 1
        """
        try:
            # Extract comprehensive features
            feature_vector = await self._extract_confidence_features(
                candle_features, pattern_signals, psychology_score, sr_signals, strategy
            )
            
            if not feature_vector:
                return 0.0
            
            # Base confidence from individual components
            base_confidence = await self._calculate_base_confidence(
                candle_features, pattern_signals, psychology_score, sr_signals, strategy
            )
            
            # ML-enhanced confidence (if models are trained)
            if self.is_trained:
                ml_confidence = await self._calculate_ml_confidence(feature_vector)
            else:
                ml_confidence = base_confidence
            
            # Confluence bonus
            confluence_bonus = await self._calculate_confluence_bonus(
                candle_features, pattern_signals, psychology_score, sr_signals
            )
            
            # Risk penalty
            risk_penalty = await self._calculate_risk_penalty(
                candle_features, pattern_signals, psychology_score, sr_signals
            )
            
            # Final confidence calculation
            final_confidence = min(1.0, max(0.0, 
                (ml_confidence * 0.7) + 
                (base_confidence * 0.3) + 
                confluence_bonus - 
                risk_penalty
            ))
            
            # Update performance tracking
            self.model_performance['predictions_made'] += 1
            if final_confidence >= self.min_confidence_threshold:
                self.model_performance['high_confidence_signals'] += 1
            
            # Store for potential training
            await self._store_training_data(feature_vector, final_confidence)
            
            return final_confidence
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence: {e}")
            return 0.0
    
    async def _extract_confidence_features(self, candle_features: Dict, pattern_signals: Dict,
                                         psychology_score: Dict, sr_signals: Dict,
                                         strategy: Dict) -> List[float]:
        """Extract comprehensive feature vector for ML confidence scoring"""
        try:
            features = []
            
            # Candle-based features
            candle_features_vector = await self._extract_candle_features(candle_features)
            features.extend(candle_features_vector)
            
            # Pattern-based features
            pattern_features_vector = await self._extract_pattern_features(pattern_signals)
            features.extend(pattern_features_vector)
            
            # Psychology-based features
            psychology_features_vector = await self._extract_psychology_features(psychology_score)
            features.extend(psychology_features_vector)
            
            # S/R-based features
            sr_features_vector = await self._extract_sr_features(sr_signals)
            features.extend(sr_features_vector)
            
            # Strategy-based features
            strategy_features_vector = await self._extract_strategy_features(strategy)
            features.extend(strategy_features_vector)
            
            # Market context features
            context_features_vector = await self._extract_context_features()
            features.extend(context_features_vector)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting confidence features: {e}")
            return []
    
    async def _extract_candle_features(self, candle_features: Dict) -> List[float]:
        """Extract candle-based confidence features"""
        features = []
        
        # Body analysis features
        body_analysis = candle_features.get('body_analysis', {})
        features.extend([
            body_analysis.get('strength', 0),
            body_analysis.get('size_ratio', 1),
            body_analysis.get('position_in_range', 0.5),
            1.0 if body_analysis.get('is_strong_body', False) else 0.0,
            1.0 if body_analysis.get('is_weak_body', False) else 0.0
        ])
        
        # Wick analysis features
        wick_analysis = candle_features.get('wick_analysis', {})
        features.extend([
            wick_analysis.get('upper_wick_ratio', 0),
            wick_analysis.get('lower_wick_ratio', 0),
            wick_analysis.get('upper_wick_significance', 1),
            wick_analysis.get('lower_wick_significance', 1),
            1.0 if wick_analysis.get('smart_rejection_upper', False) else 0.0,
            1.0 if wick_analysis.get('smart_rejection_lower', False) else 0.0
        ])
        
        # Size analysis features
        size_analysis = candle_features.get('size_analysis', {})
        features.extend([
            size_analysis.get('size_momentum', 1),
            size_analysis.get('size_volatility', 0),
            1.0 if size_analysis.get('is_size_breakout', False) else 0.0,
            1.0 if size_analysis.get('is_size_compression', False) else 0.0
        ])
        
        # Volume analysis features
        volume_analysis = candle_features.get('volume_analysis', {})
        features.extend([
            volume_analysis.get('volume_momentum', 1),
            1.0 if volume_analysis.get('volume_spike', False) else 0.0,
            1.0 if volume_analysis.get('volume_dry_up', False) else 0.0,
            volume_analysis.get('volume_price_correlation', 0)
        ])
        
        # Momentum features
        momentum_signals = candle_features.get('momentum_signals', {})
        features.extend([
            momentum_signals.get('short_momentum', 0),
            momentum_signals.get('long_momentum', 0),
            1.0 if momentum_signals.get('momentum_shift_detected', False) else 0.0,
            momentum_signals.get('momentum_strength', 0)
        ])
        
        return features
    
    async def _extract_pattern_features(self, pattern_signals: Dict) -> List[float]:
        """Extract pattern-based confidence features"""
        features = []
        
        # Pattern count and quality
        patterns = pattern_signals.get('patterns_detected', [])
        features.extend([
            len(patterns),
            np.mean([p.get('confidence', 0) for p in patterns]) if patterns else 0,
            pattern_signals.get('pattern_confidence', 0),
            pattern_signals.get('story_coherence', 0)
        ])
        
        # Specific pattern types
        pattern_types = {
            'reversal': 0,
            'continuation': 0,
            'breakout': 0,
            'exhaustion': 0,
            'trap': 0
        }
        
        for pattern in patterns:
            pattern_name = pattern.get('name', '').lower()
            for pattern_type in pattern_types:
                if pattern_type in pattern_name:
                    pattern_types[pattern_type] += 1
        
        features.extend(list(pattern_types.values()))
        
        # Market psychology patterns
        market_psychology = pattern_signals.get('market_psychology', {})
        features.extend([
            market_psychology.get('fear_greed_index', 0.5),
            1.0 if market_psychology.get('smart_money_flow', False) else 0.0,
            1.0 if market_psychology.get('retail_exhaustion', False) else 0.0,
            1.0 if market_psychology.get('institutional_accumulation', False) else 0.0
        ])
        
        return features
    
    async def _extract_psychology_features(self, psychology_score: Dict) -> List[float]:
        """Extract psychology-based confidence features"""
        features = []
        
        # Core psychology metrics
        features.extend([
            psychology_score.get('fear_greed_index', 0.5),
            1.0 if psychology_score.get('smart_money_flow', False) else 0.0,
            1.0 if psychology_score.get('retail_exhaustion', False) else 0.0,
            1.0 if psychology_score.get('institutional_accumulation', False) else 0.0
        ])
        
        # Momentum psychology
        momentum_psychology = psychology_score.get('momentum_psychology', {})
        features.extend([
            momentum_psychology.get('momentum_strength', 0),
            1.0 if momentum_psychology.get('fomo_detected', False) else 0.0,
            1.0 if momentum_psychology.get('momentum_shift_detected', False) else 0.0
        ])
        
        # Volatility psychology
        volatility_psychology = psychology_score.get('volatility_psychology', {})
        features.extend([
            volatility_psychology.get('volatility_ratio', 1),
            volatility_psychology.get('fear_level', 0),
            1.0 if volatility_psychology.get('volatility_breakout', False) else 0.0
        ])
        
        # Crowd behavior
        crowd_behavior = psychology_score.get('crowd_behavior', {})
        features.extend([
            1.0 if crowd_behavior.get('herding_detected', False) else 0.0,
            1.0 if crowd_behavior.get('contrarian_setup', False) else 0.0,
            1.0 if crowd_behavior.get('extreme_herding', False) else 0.0
        ])
        
        # Panic/greed signals
        panic_greed = psychology_score.get('panic_greed_signals', {})
        features.extend([
            1.0 if panic_greed.get('panic_detected', False) else 0.0,
            1.0 if panic_greed.get('greed_detected', False) else 0.0,
            panic_greed.get('panic_strength', 0),
            panic_greed.get('greed_strength', 0)
        ])
        
        return features
    
    async def _extract_sr_features(self, sr_signals: Dict) -> List[float]:
        """Extract S/R-based confidence features"""
        features = []
        
        # Basic S/R features
        features.extend([
            1.0 if sr_signals.get('rejection_detected', False) else 0.0,
            sr_signals.get('rejection_strength', 0),
            1.0 if sr_signals.get('multiple_touches', False) else 0.0,
            1.0 if sr_signals.get('volume_confirmation', False) else 0.0,
            1.0 if sr_signals.get('wick_rejection', False) else 0.0
        ])
        
        # Rejection type encoding
        rejection_type = sr_signals.get('rejection_type', 'none')
        features.extend([
            1.0 if rejection_type == 'support' else 0.0,
            1.0 if rejection_type == 'resistance' else 0.0
        ])
        
        return features
    
    async def _extract_strategy_features(self, strategy: Dict) -> List[float]:
        """Extract strategy-based confidence features"""
        features = []
        
        # Basic strategy features
        features.extend([
            strategy.get('entry_score', 0),
            strategy.get('confidence_boost', 0),
            len(strategy.get('entry_conditions', [])),
            len(strategy.get('confluence_factors', []))
        ])
        
        # Strategy type encoding
        strategy_name = strategy.get('name', 'UNKNOWN')
        strategy_types = [
            'TREND_FOLLOWING', 'RANGE_TRADING', 'VOLATILITY_EXPANSION',
            'CONSOLIDATION_BREAKOUT', 'BREAKOUT_MOMENTUM', 'DEFAULT_NO_TRADE'
        ]
        
        for strategy_type in strategy_types:
            features.append(1.0 if strategy_name == strategy_type else 0.0)
        
        # Action encoding
        action = strategy.get('action', 'NO_TRADE')
        features.extend([
            1.0 if action == 'CALL' else 0.0,
            1.0 if action == 'PUT' else 0.0,
            1.0 if action == 'NO_TRADE' else 0.0
        ])
        
        return features
    
    async def _extract_context_features(self) -> List[float]:
        """Extract market context features"""
        features = []
        
        # Time-based features
        now = datetime.now()
        features.extend([
            now.hour / 24.0,  # Hour of day (normalized)
            now.weekday() / 6.0,  # Day of week (normalized)
        ])
        
        # Session features
        london_session = 1.0 if 8 <= now.hour <= 16 else 0.0
        ny_session = 1.0 if 13 <= now.hour <= 21 else 0.0
        overlap_session = 1.0 if 13 <= now.hour <= 16 else 0.0
        
        features.extend([london_session, ny_session, overlap_session])
        
        return features
    
    async def _calculate_base_confidence(self, candle_features: Dict, pattern_signals: Dict,
                                       psychology_score: Dict, sr_signals: Dict,
                                       strategy: Dict) -> float:
        """Calculate base confidence from component analysis"""
        try:
            confidence_components = []
            
            # Candle confidence
            candle_conf = await self._calculate_candle_confidence(candle_features)
            confidence_components.append(candle_conf * 0.25)
            
            # Pattern confidence
            pattern_conf = pattern_signals.get('pattern_confidence', 0)
            confidence_components.append(pattern_conf * 0.25)
            
            # Psychology confidence
            psychology_conf = await self._calculate_psychology_confidence(psychology_score)
            confidence_components.append(psychology_conf * 0.20)
            
            # S/R confidence
            sr_conf = sr_signals.get('rejection_strength', 0) if sr_signals.get('rejection_detected', False) else 0.5
            confidence_components.append(sr_conf * 0.15)
            
            # Strategy confidence
            strategy_conf = strategy.get('entry_score', 0)
            confidence_components.append(strategy_conf * 0.15)
            
            base_confidence = sum(confidence_components)
            return min(1.0, max(0.0, base_confidence))
            
        except Exception as e:
            self.logger.error(f"Error calculating base confidence: {e}")
            return 0.5
    
    async def _calculate_candle_confidence(self, candle_features: Dict) -> float:
        """Calculate confidence from candle features"""
        try:
            confidence_factors = []
            
            # Body strength
            body_analysis = candle_features.get('body_analysis', {})
            body_strength = body_analysis.get('strength', 0)
            confidence_factors.append(body_strength)
            
            # Wick rejection quality
            wick_analysis = candle_features.get('wick_analysis', {})
            rejection_quality = 0
            if wick_analysis.get('smart_rejection_upper', False) or wick_analysis.get('smart_rejection_lower', False):
                rejection_quality = max(
                    wick_analysis.get('upper_wick_significance', 0),
                    wick_analysis.get('lower_wick_significance', 0)
                )
            confidence_factors.append(rejection_quality)
            
            # Volume confirmation
            volume_analysis = candle_features.get('volume_analysis', {})
            volume_conf = 0.8 if volume_analysis.get('volume_spike', False) else 0.6
            confidence_factors.append(volume_conf)
            
            # Momentum strength
            momentum = candle_features.get('momentum_signals', {})
            momentum_strength = momentum.get('momentum_strength', 0)
            confidence_factors.append(min(1.0, momentum_strength * 10))  # Scale momentum
            
            return np.mean(confidence_factors)
            
        except Exception as e:
            return 0.5
    
    async def _calculate_psychology_confidence(self, psychology_score: Dict) -> float:
        """Calculate confidence from psychology analysis"""
        try:
            confidence_factors = []
            
            # Fear/greed extremes increase confidence for contrarian plays
            fear_greed = psychology_score.get('fear_greed_index', 0.5)
            if fear_greed < 0.2 or fear_greed > 0.8:
                confidence_factors.append(0.8)  # High confidence for extremes
            else:
                confidence_factors.append(0.5)  # Neutral confidence
            
            # Smart money alignment
            if psychology_score.get('smart_money_flow', False):
                confidence_factors.append(0.9)
            
            # Retail exhaustion
            if psychology_score.get('retail_exhaustion', False):
                confidence_factors.append(0.8)
            
            # Institutional accumulation
            if psychology_score.get('institutional_accumulation', False):
                confidence_factors.append(0.85)
            
            # Panic/greed signals
            panic_greed = psychology_score.get('panic_greed_signals', {})
            if panic_greed.get('extreme_emotion', False):
                confidence_factors.append(0.8)
            
            return np.mean(confidence_factors) if confidence_factors else 0.5
            
        except Exception as e:
            return 0.5
    
    async def _calculate_ml_confidence(self, feature_vector: List[float]) -> float:
        """Calculate ML-enhanced confidence"""
        try:
            # Prepare features
            features_array = np.array(feature_vector).reshape(1, -1)
            
            # Scale features
            if hasattr(self.feature_scaler, 'scale_'):
                features_scaled = self.feature_scaler.transform(features_array)
            else:
                features_scaled = features_array
            
            # Get predictions from all models
            predictions = {}
            
            try:
                predictions['rf'] = self.rf_model.predict(features_scaled)[0]
            except:
                predictions['rf'] = 0.5
            
            try:
                predictions['gb'] = self.gb_model.predict(features_scaled)[0]
            except:
                predictions['gb'] = 0.5
            
            try:
                predictions['nn'] = self.nn_model.predict(features_scaled)[0]
            except:
                predictions['nn'] = 0.5
            
            # Ensemble prediction
            ensemble_confidence = (
                predictions['rf'] * self.model_weights['random_forest'] +
                predictions['gb'] * self.model_weights['gradient_boost'] +
                predictions['nn'] * self.model_weights['neural_network']
            )
            
            return min(1.0, max(0.0, ensemble_confidence))
            
        except Exception as e:
            self.logger.error(f"Error in ML confidence calculation: {e}")
            return 0.5
    
    async def _calculate_confluence_bonus(self, candle_features: Dict, pattern_signals: Dict,
                                        psychology_score: Dict, sr_signals: Dict) -> float:
        """Calculate bonus for multiple confluences"""
        try:
            confluences = 0
            
            # Strong candle signals
            body_analysis = candle_features.get('body_analysis', {})
            if body_analysis.get('is_strong_body', False):
                confluences += 1
            
            # Smart rejection
            wick_analysis = candle_features.get('wick_analysis', {})
            if wick_analysis.get('smart_rejection_upper', False) or wick_analysis.get('smart_rejection_lower', False):
                confluences += 1
            
            # Volume confirmation
            volume_analysis = candle_features.get('volume_analysis', {})
            if volume_analysis.get('volume_spike', False):
                confluences += 1
            
            # Multiple patterns
            patterns = pattern_signals.get('patterns_detected', [])
            if len(patterns) >= 2:
                confluences += 1
            
            # S/R rejection
            if sr_signals.get('rejection_detected', False):
                confluences += 1
            
            # Psychology extremes
            if psychology_score.get('fear_greed_index', 0.5) < 0.2 or psychology_score.get('fear_greed_index', 0.5) > 0.8:
                confluences += 1
            
            # Confluence bonus calculation
            if confluences >= 4:
                return 0.15  # High confluence bonus
            elif confluences >= 3:
                return 0.10  # Medium confluence bonus
            elif confluences >= 2:
                return 0.05  # Low confluence bonus
            else:
                return 0.0   # No bonus
                
        except Exception as e:
            return 0.0
    
    async def _calculate_risk_penalty(self, candle_features: Dict, pattern_signals: Dict,
                                    psychology_score: Dict, sr_signals: Dict) -> float:
        """Calculate penalty for risk factors"""
        try:
            risk_penalty = 0.0
            
            # High volatility without volume
            size_analysis = candle_features.get('size_analysis', {})
            volume_analysis = candle_features.get('volume_analysis', {})
            
            if size_analysis.get('size_volatility', 0) > 2.0 and volume_analysis.get('volume_dry_up', False):
                risk_penalty += 0.10
            
            # Manipulation signals
            manipulation_signals = psychology_score.get('manipulation_signals', {})
            if manipulation_signals.get('manipulation_detected', False):
                risk_penalty += 0.15
            
            # Conflicting signals
            patterns = pattern_signals.get('patterns_detected', [])
            conflicting_patterns = 0
            pattern_directions = []
            
            for pattern in patterns:
                signal = pattern.get('signal', 'neutral')
                if signal in ['bullish', 'bearish']:
                    pattern_directions.append(signal)
            
            if len(set(pattern_directions)) > 1:  # Conflicting directions
                risk_penalty += 0.08
            
            # News/event risk (simplified - could be enhanced with news API)
            current_hour = datetime.now().hour
            if 8 <= current_hour <= 10 or 14 <= current_hour <= 16:  # News times
                risk_penalty += 0.03
            
            return min(0.3, risk_penalty)  # Cap penalty at 30%
            
        except Exception as e:
            return 0.0
    
    async def _store_training_data(self, feature_vector: List[float], confidence: float):
        """Store data for potential model retraining"""
        try:
            self.training_features.append(feature_vector)
            self.training_scores.append(confidence)
            
            # Keep only recent data
            if len(self.training_features) > 1000:
                self.training_features = self.training_features[-1000:]
                self.training_scores = self.training_scores[-1000:]
                
        except Exception as e:
            self.logger.error(f"Error storing training data: {e}")
    
    async def retrain_models(self):
        """Retrain ML models with new data"""
        try:
            if len(self.training_features) < 50:
                self.logger.warning("Insufficient training data for retraining")
                return
            
            self.logger.info("🔄 Retraining confidence models...")
            
            X = np.array(self.training_features)
            y = np.array(self.training_scores)
            
            # Scale features
            X_scaled = self.feature_scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            
            # Train models
            self.rf_model.fit(X_train, y_train)
            self.gb_model.fit(X_train, y_train)
            self.nn_model.fit(X_train, y_train)
            
            # Calculate accuracy
            rf_score = self.rf_model.score(X_test, y_test)
            gb_score = self.gb_model.score(X_test, y_test)
            nn_score = self.nn_model.score(X_test, y_test)
            
            avg_accuracy = (rf_score + gb_score + nn_score) / 3
            self.model_performance['model_accuracy'] = avg_accuracy
            
            self.is_trained = True
            
            self.logger.info(f"✅ Models retrained. Average accuracy: {avg_accuracy:.3f}")
            
        except Exception as e:
            self.logger.error(f"Error retraining models: {e}")
    
    def get_confidence_statistics(self) -> Dict[str, Any]:
        """Get confidence scoring statistics"""
        try:
            high_conf_ratio = (self.model_performance['high_confidence_signals'] / 
                             max(1, self.model_performance['predictions_made']))
            
            return {
                'predictions_made': self.model_performance['predictions_made'],
                'high_confidence_signals': self.model_performance['high_confidence_signals'],
                'high_confidence_ratio': high_conf_ratio,
                'model_accuracy': self.model_performance['model_accuracy'],
                'is_trained': self.is_trained,
                'training_samples': len(self.training_features),
                'min_threshold': self.min_confidence_threshold,
                'god_mode_threshold': self.god_mode_threshold
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence statistics: {e}")
            return {}
    
    def save_models(self, filepath: str = "confidence_models.pkl"):
        """Save trained models to disk"""
        try:
            model_data = {
                'rf_model': self.rf_model,
                'gb_model': self.gb_model,
                'nn_model': self.nn_model,
                'feature_scaler': self.feature_scaler,
                'model_weights': self.model_weights,
                'is_trained': self.is_trained
            }
            
            joblib.dump(model_data, filepath)
            self.logger.info(f"Models saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving models: {e}")
    
    def load_models(self, filepath: str = "confidence_models.pkl"):
        """Load trained models from disk"""
        try:
            model_data = joblib.load(filepath)
            
            self.rf_model = model_data['rf_model']
            self.gb_model = model_data['gb_model']
            self.nn_model = model_data['nn_model']
            self.feature_scaler = model_data['feature_scaler']
            self.model_weights = model_data['model_weights']
            self.is_trained = model_data['is_trained']
            
            self.logger.info(f"Models loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")