import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import json
import logging
from dataclasses import dataclass
from enum import Enum
import cv2
import base64
from io import BytesIO
from PIL import Image

from analysis.technical_indicators import TechnicalAnalyzer
from models.trade import TradeDirection, SignalAlert

class StrategyType(Enum):
    BREAKOUT_CONTINUATION = "breakout_continuation"
    REVERSAL_PLAY = "reversal_play"
    MOMENTUM_SHIFT = "momentum_shift"
    TRAP_FADE = "trap_fade"
    EXHAUSTION_REVERSAL = "exhaustion_reversal"

class MarketState(Enum):
    TRENDING = "trending"
    CONSOLIDATING = "consolidating"
    VOLATILE = "volatile"
    QUIET = "quiet"
    BREAKOUT = "breakout"

@dataclass
class StrategyNode:
    """Individual node in the strategy tree"""
    condition: str
    weight: float
    signal_type: str  # 'bullish', 'bearish', 'neutral'
    confidence: float
    reasoning: str

@dataclass
class StrategyTree:
    """Complete strategy tree for a specific market setup"""
    strategy_type: StrategyType
    nodes: List[StrategyNode]
    overall_confidence: float
    predicted_direction: TradeDirection
    reasoning: str
    expiry_recommendation: int  # in seconds
    risk_level: str

class CosmicAIStrategy:
    """🌌 OMNI-BRAIN BINARY AI - Ultimate Adaptive Strategy Builder"""
    
    def __init__(self):
        self.technical_analyzer = TechnicalAnalyzer()
        self.strategy_patterns = self._load_strategy_patterns()
        self.market_psychology_rules = self._load_psychology_rules()
        self.confidence_threshold = 0.75
        
        # AI Learning components
        self.pattern_memory = {}
        self.success_patterns = {}
        self.failure_patterns = {}
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def analyze_chart_image(self, image_data: bytes) -> Dict[str, Any]:
        """🔍 PERCEPTION ENGINE: Advanced chart analysis with dynamic broker detection"""
        try:
            # Decode image
            image = Image.open(BytesIO(image_data))
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Extract chart elements
            chart_analysis = self._extract_chart_elements(cv_image)
            
            # Detect broker and timeframe
            broker_info = self._detect_broker_from_image(cv_image)
            
            # Convert image patterns to market data
            market_data = self._image_to_market_data(cv_image, chart_analysis)
            
            return {
                'broker': broker_info['broker'],
                'timeframe': broker_info['timeframe'],
                'asset': broker_info['asset'],
                'market_data': market_data,
                'chart_patterns': chart_analysis['patterns'],
                'support_resistance': chart_analysis['levels'],
                'confidence': chart_analysis['confidence']
            }
            
        except Exception as e:
            self.logger.error(f"Chart analysis failed: {e}")
            return {'error': str(e)}
    
    def read_market_story(self, market_data: Dict) -> Dict[str, Any]:
        """📖 CONTEXT ENGINE: Reads market stories like a human trader"""
        
        if not market_data or 'df' not in market_data:
            return {'story': 'Insufficient data', 'confidence': 0.0}
        
        df = market_data['df']
        indicators = self.technical_analyzer.calculate_all_indicators(df)
        
        # Analyze market narrative
        story_elements = {
            'market_state': self._determine_market_state(df, indicators),
            'price_action_story': self._read_price_action(df),
            'volume_story': self._read_volume_narrative(df, indicators),
            'momentum_story': self._read_momentum_narrative(indicators),
            'structure_story': self._read_market_structure(df, indicators),
            'psychology_signals': self._read_market_psychology(df, indicators)
        }
        
        # Synthesize the complete market story
        complete_story = self._synthesize_market_story(story_elements)
        
        return {
            'story': complete_story,
            'elements': story_elements,
            'confidence': self._calculate_story_confidence(story_elements),
            'key_levels': self._identify_key_levels(df, indicators),
            'narrative_bias': self._determine_narrative_bias(story_elements)
        }
    
    def build_strategy_tree(self, market_story: Dict, chart_analysis: Dict) -> StrategyTree:
        """🧠 STRATEGY ENGINE: Builds unique strategies on-the-fly for each chart"""
        
        # Determine optimal strategy type
        strategy_type = self._select_strategy_type(market_story, chart_analysis)
        
        # Build strategy nodes based on market conditions
        nodes = self._build_strategy_nodes(strategy_type, market_story, chart_analysis)
        
        # Calculate overall confidence
        overall_confidence = self._calculate_strategy_confidence(nodes, market_story)
        
        # Predict direction and expiry
        direction, reasoning = self._predict_direction(nodes, market_story)
        expiry_time = self._recommend_expiry(strategy_type, market_story)
        risk_level = self._assess_risk_level(market_story, overall_confidence)
        
        return StrategyTree(
            strategy_type=strategy_type,
            nodes=nodes,
            overall_confidence=overall_confidence,
            predicted_direction=direction,
            reasoning=reasoning,
            expiry_recommendation=expiry_time,
            risk_level=risk_level
        )
    
    def execute_cosmic_analysis(self, image_data: Optional[bytes] = None, 
                              market_data: Optional[Dict] = None) -> Dict[str, Any]:
        """🚀 Main execution pipeline for COSMIC AI"""
        
        analysis_result = {
            'timestamp': datetime.now().isoformat(),
            'cosmic_ai_version': '1.0',
            'analysis_stages': {}
        }
        
        try:
            # Stage 1: Chart Perception
            if image_data:
                chart_analysis = self.analyze_chart_image(image_data)
                analysis_result['analysis_stages']['perception'] = chart_analysis
                
                if 'error' in chart_analysis:
                    return analysis_result
                
                # Use extracted market data if available
                if 'market_data' in chart_analysis:
                    market_data = chart_analysis['market_data']
            
            # Stage 2: Story Reading
            if market_data:
                market_story = self.read_market_story(market_data)
                analysis_result['analysis_stages']['story'] = market_story
                
                # Stage 3: Strategy Building
                chart_context = chart_analysis if image_data else {}
                strategy_tree = self.build_strategy_tree(market_story, chart_context)
                analysis_result['analysis_stages']['strategy'] = self._serialize_strategy_tree(strategy_tree)
                
                # Stage 4: Signal Generation
                if strategy_tree.overall_confidence >= self.confidence_threshold:
                    signal = self._generate_trading_signal(strategy_tree, market_data)
                    analysis_result['trading_signal'] = signal
                    analysis_result['execute_trade'] = True
                else:
                    analysis_result['execute_trade'] = False
                    analysis_result['reason'] = f"Confidence {strategy_tree.overall_confidence:.1%} below threshold {self.confidence_threshold:.1%}"
                
                # Stage 5: Learning and Adaptation
                self._update_pattern_memory(market_story, strategy_tree)
                
            else:
                analysis_result['error'] = 'No market data available for analysis'
                
        except Exception as e:
            self.logger.error(f"Cosmic AI analysis failed: {e}")
            analysis_result['error'] = str(e)
        
        return analysis_result
    
    def _extract_chart_elements(self, cv_image: np.ndarray) -> Dict:
        """Extract patterns and levels from chart image"""
        # Simplified chart pattern recognition
        # In a full implementation, this would use computer vision
        # to detect candlestick patterns, trend lines, etc.
        
        height, width = cv_image.shape[:2]
        
        # Detect horizontal levels (support/resistance)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=width//4, maxLineGap=10)
        
        levels = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(y1 - y2) < 10:  # Horizontal line
                    levels.append(y1)
        
        return {
            'patterns': ['ascending_triangle', 'support_test'],  # Placeholder
            'levels': levels,
            'confidence': 0.85
        }
    
    def _detect_broker_from_image(self, cv_image: np.ndarray) -> Dict:
        """Detect broker platform from chart image"""
        # Simplified broker detection
        # In reality, this would analyze UI elements, fonts, colors
        
        return {
            'broker': 'deriv',
            'timeframe': '1m',
            'asset': 'frxEURUSD'
        }
    
    def _image_to_market_data(self, cv_image: np.ndarray, chart_analysis: Dict) -> Dict:
        """Convert chart image to market data structure"""
        # Simplified conversion - would use advanced CV techniques
        # to extract OHLC data from candlestick chart
        
        # Generate sample data for demonstration
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1T')
        
        # Generate realistic price data
        base_price = 1.1000
        price_data = []
        current_price = base_price
        
        for i in range(100):
            # Random walk with some trend
            change = np.random.normal(0, 0.0001)
            current_price += change
            
            # Generate OHLC
            open_price = current_price
            high_price = open_price + abs(np.random.normal(0, 0.0002))
            low_price = open_price - abs(np.random.normal(0, 0.0002))
            close_price = open_price + np.random.normal(0, 0.0001)
            volume = np.random.randint(1000, 10000)
            
            price_data.append({
                'timestamp': dates[i],
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
            
            current_price = close_price
        
        df = pd.DataFrame(price_data)
        df.set_index('timestamp', inplace=True)
        
        return {'df': df}
    
    def _determine_market_state(self, df: pd.DataFrame, indicators: Dict) -> MarketState:
        """Determine current market state"""
        if 'trend_direction' in indicators and 'volatility' in indicators:
            trend = indicators['trend_direction']
            
            if len(indicators['volatility']) > 0:
                volatility = indicators['volatility'][-1] if not np.isnan(indicators['volatility'][-1]) else 0
                
                if volatility > 30:  # High volatility
                    return MarketState.VOLATILE
                elif trend in ['up', 'down']:
                    return MarketState.TRENDING
                else:
                    return MarketState.CONSOLIDATING
        
        return MarketState.QUIET
    
    def _read_price_action(self, df: pd.DataFrame) -> str:
        """Read the price action story"""
        if len(df) < 20:
            return "Insufficient data for price action analysis"
        
        recent_candles = df.tail(10)
        
        # Analyze recent price movement
        price_change = recent_candles['close'].iloc[-1] - recent_candles['close'].iloc[0]
        volatility = recent_candles['high'].max() - recent_candles['low'].min()
        
        if price_change > 0:
            bias = "bullish"
        elif price_change < 0:
            bias = "bearish"
        else:
            bias = "neutral"
        
        story = f"Recent price action shows {bias} momentum with "
        
        if volatility > recent_candles['close'].mean() * 0.01:
            story += "high volatility indicating strong market sentiment."
        else:
            story += "low volatility suggesting consolidation phase."
        
        return story
    
    def _read_volume_narrative(self, df: pd.DataFrame, indicators: Dict) -> str:
        """Read volume story if available"""
        if 'volume' not in df.columns or df['volume'].sum() == 0:
            return "Volume data not available - focusing on price action."
        
        if 'volume_spike' in indicators and indicators['volume_spike'][-1]:
            return "High volume spike detected - confirms current price movement."
        elif 'volume_above_average' in indicators and indicators['volume_above_average'][-1]:
            return "Above average volume supports the current trend."
        else:
            return "Low volume suggests weak conviction in current move."
    
    def _read_momentum_narrative(self, indicators: Dict) -> str:
        """Read momentum story from indicators"""
        momentum_story = []
        
        if 'rsi' in indicators and len(indicators['rsi']) > 0:
            rsi_val = indicators['rsi'][-1]
            if not np.isnan(rsi_val):
                if rsi_val > 70:
                    momentum_story.append("overbought conditions")
                elif rsi_val < 30:
                    momentum_story.append("oversold conditions")
                else:
                    momentum_story.append("neutral momentum")
        
        if 'macd_bullish' in indicators and indicators['macd_bullish'][-1]:
            momentum_story.append("MACD showing bullish divergence")
        elif 'macd_bearish' in indicators and indicators['macd_bearish'][-1]:
            momentum_story.append("MACD showing bearish divergence")
        
        if momentum_story:
            return "Momentum indicators reveal: " + ", ".join(momentum_story) + "."
        else:
            return "Momentum indicators are inconclusive."
    
    def _read_market_structure(self, df: pd.DataFrame, indicators: Dict) -> str:
        """Read market structure narrative"""
        if 'trend_direction' in indicators:
            trend = indicators['trend_direction']
            
            if trend == 'up':
                return "Market structure shows clear uptrend with higher highs and higher lows."
            elif trend == 'down':
                return "Market structure shows clear downtrend with lower highs and lower lows."
            else:
                return "Market structure is sideways with no clear directional bias."
        
        return "Market structure analysis inconclusive."
    
    def _read_market_psychology(self, df: pd.DataFrame, indicators: Dict) -> List[str]:
        """Read market psychology signals"""
        psychology_signals = []
        
        # Fear and greed indicators
        if 'rsi' in indicators and len(indicators['rsi']) > 0:
            rsi_val = indicators['rsi'][-1]
            if not np.isnan(rsi_val):
                if rsi_val > 80:
                    psychology_signals.append("Extreme greed detected")
                elif rsi_val < 20:
                    psychology_signals.append("Extreme fear detected")
        
        # Bollinger Band psychology
        if 'bb_position' in indicators:
            bb_pos = indicators['bb_position'][-1]
            if bb_pos == 'upper':
                psychology_signals.append("Price at upper band - potential exhaustion")
            elif bb_pos == 'lower':
                psychology_signals.append("Price at lower band - potential reversal")
        
        return psychology_signals
    
    def _synthesize_market_story(self, story_elements: Dict) -> str:
        """Synthesize complete market narrative"""
        story_parts = []
        
        story_parts.append(f"Market State: {story_elements['market_state'].value}")
        story_parts.append(story_elements['price_action_story'])
        story_parts.append(story_elements['volume_story'])
        story_parts.append(story_elements['momentum_story'])
        story_parts.append(story_elements['structure_story'])
        
        if story_elements['psychology_signals']:
            story_parts.append("Psychology: " + "; ".join(story_elements['psychology_signals']))
        
        return "\n".join(story_parts)
    
    def _calculate_story_confidence(self, story_elements: Dict) -> float:
        """Calculate confidence in market story"""
        confidence_factors = []
        
        # Market state confidence
        if story_elements['market_state'] != MarketState.QUIET:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.3)
        
        # Psychology signals confidence
        if story_elements['psychology_signals']:
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.5)
        
        return np.mean(confidence_factors) if confidence_factors else 0.5
    
    def _identify_key_levels(self, df: pd.DataFrame, indicators: Dict) -> Dict:
        """Identify key support and resistance levels"""
        levels = {}
        
        if 'pivot_point' in indicators:
            levels['pivot'] = indicators['pivot_point']
        
        if 'support_1' in indicators:
            levels['support'] = indicators['support_1']
        
        if 'resistance_1' in indicators:
            levels['resistance'] = indicators['resistance_1']
        
        # Recent high/low levels
        recent_data = df.tail(20)
        levels['recent_high'] = recent_data['high'].max()
        levels['recent_low'] = recent_data['low'].min()
        
        return levels
    
    def _determine_narrative_bias(self, story_elements: Dict) -> str:
        """Determine overall narrative bias"""
        bullish_signals = 0
        bearish_signals = 0
        
        # Count signals from different elements
        if story_elements['market_state'] == MarketState.TRENDING:
            # Would need to check if trending up or down
            pass
        
        psychology_signals = story_elements.get('psychology_signals', [])
        for signal in psychology_signals:
            if 'fear' in signal.lower():
                bullish_signals += 1
            elif 'greed' in signal.lower():
                bearish_signals += 1
        
        if bullish_signals > bearish_signals:
            return 'bullish'
        elif bearish_signals > bullish_signals:
            return 'bearish'
        else:
            return 'neutral'
    
    def _select_strategy_type(self, market_story: Dict, chart_analysis: Dict) -> StrategyType:
        """Select optimal strategy type based on market conditions"""
        market_state = market_story['elements']['market_state']
        narrative_bias = market_story['narrative_bias']
        
        if market_state == MarketState.BREAKOUT:
            return StrategyType.BREAKOUT_CONTINUATION
        elif market_state == MarketState.VOLATILE:
            return StrategyType.MOMENTUM_SHIFT
        elif market_state == MarketState.CONSOLIDATING:
            if narrative_bias != 'neutral':
                return StrategyType.REVERSAL_PLAY
            else:
                return StrategyType.TRAP_FADE
        else:
            return StrategyType.EXHAUSTION_REVERSAL
    
    def _build_strategy_nodes(self, strategy_type: StrategyType, 
                            market_story: Dict, chart_analysis: Dict) -> List[StrategyNode]:
        """Build strategy nodes for the given strategy type"""
        nodes = []
        
        # Base nodes for all strategies
        nodes.append(StrategyNode(
            condition="market_state",
            weight=0.3,
            signal_type=market_story['narrative_bias'],
            confidence=market_story['confidence'],
            reasoning=f"Market state: {market_story['elements']['market_state'].value}"
        ))
        
        # Strategy-specific nodes
        if strategy_type == StrategyType.BREAKOUT_CONTINUATION:
            nodes.extend(self._build_breakout_nodes(market_story, chart_analysis))
        elif strategy_type == StrategyType.REVERSAL_PLAY:
            nodes.extend(self._build_reversal_nodes(market_story, chart_analysis))
        elif strategy_type == StrategyType.MOMENTUM_SHIFT:
            nodes.extend(self._build_momentum_nodes(market_story, chart_analysis))
        elif strategy_type == StrategyType.TRAP_FADE:
            nodes.extend(self._build_trap_fade_nodes(market_story, chart_analysis))
        elif strategy_type == StrategyType.EXHAUSTION_REVERSAL:
            nodes.extend(self._build_exhaustion_nodes(market_story, chart_analysis))
        
        return nodes
    
    def _build_breakout_nodes(self, market_story: Dict, chart_analysis: Dict) -> List[StrategyNode]:
        """Build nodes for breakout continuation strategy"""
        return [
            StrategyNode(
                condition="volume_confirmation",
                weight=0.4,
                signal_type="bullish",
                confidence=0.8,
                reasoning="Volume spike confirms breakout momentum"
            ),
            StrategyNode(
                condition="resistance_break",
                weight=0.3,
                signal_type="bullish", 
                confidence=0.75,
                reasoning="Price broke above key resistance level"
            )
        ]
    
    def _build_reversal_nodes(self, market_story: Dict, chart_analysis: Dict) -> List[StrategyNode]:
        """Build nodes for reversal play strategy"""
        return [
            StrategyNode(
                condition="oversold_rsi",
                weight=0.35,
                signal_type="bullish",
                confidence=0.7,
                reasoning="RSI showing oversold conditions"
            ),
            StrategyNode(
                condition="support_hold",
                weight=0.25,
                signal_type="bullish",
                confidence=0.65,
                reasoning="Price holding at support level"
            )
        ]
    
    def _build_momentum_nodes(self, market_story: Dict, chart_analysis: Dict) -> List[StrategyNode]:
        """Build nodes for momentum shift strategy"""
        return [
            StrategyNode(
                condition="macd_cross",
                weight=0.4,
                signal_type="bullish",
                confidence=0.8,
                reasoning="MACD showing bullish crossover"
            )
        ]
    
    def _build_trap_fade_nodes(self, market_story: Dict, chart_analysis: Dict) -> List[StrategyNode]:
        """Build nodes for trap fade strategy"""
        return [
            StrategyNode(
                condition="false_breakout",
                weight=0.5,
                signal_type="bearish",
                confidence=0.7,
                reasoning="False breakout detected - fade the move"
            )
        ]
    
    def _build_exhaustion_nodes(self, market_story: Dict, chart_analysis: Dict) -> List[StrategyNode]:
        """Build nodes for exhaustion reversal strategy"""
        return [
            StrategyNode(
                condition="extreme_rsi",
                weight=0.4,
                signal_type="bearish",
                confidence=0.75,
                reasoning="Extreme RSI levels indicate exhaustion"
            )
        ]
    
    def _calculate_strategy_confidence(self, nodes: List[StrategyNode], market_story: Dict) -> float:
        """Calculate overall strategy confidence"""
        total_weight = sum(node.weight for node in nodes)
        weighted_confidence = sum(node.confidence * node.weight for node in nodes)
        
        base_confidence = weighted_confidence / total_weight if total_weight > 0 else 0.5
        
        # Adjust based on market story confidence
        story_confidence = market_story.get('confidence', 0.5)
        
        return (base_confidence + story_confidence) / 2
    
    def _predict_direction(self, nodes: List[StrategyNode], market_story: Dict) -> Tuple[TradeDirection, str]:
        """Predict trade direction based on strategy nodes"""
        bullish_weight = sum(node.weight for node in nodes if node.signal_type == 'bullish')
        bearish_weight = sum(node.weight for node in nodes if node.signal_type == 'bearish')
        
        reasoning_parts = [node.reasoning for node in nodes if node.signal_type != 'neutral']
        
        if bullish_weight > bearish_weight:
            direction = TradeDirection.CALL
            reasoning = "CALL prediction based on: " + "; ".join(reasoning_parts)
        else:
            direction = TradeDirection.PUT
            reasoning = "PUT prediction based on: " + "; ".join(reasoning_parts)
        
        return direction, reasoning
    
    def _recommend_expiry(self, strategy_type: StrategyType, market_story: Dict) -> int:
        """Recommend expiry time based on strategy type and market conditions"""
        market_state = market_story['elements']['market_state']
        
        if strategy_type == StrategyType.BREAKOUT_CONTINUATION:
            return 300  # 5 minutes for breakout momentum
        elif strategy_type == StrategyType.MOMENTUM_SHIFT:
            return 180  # 3 minutes for momentum plays
        elif strategy_type == StrategyType.REVERSAL_PLAY:
            return 600  # 10 minutes for reversals to develop
        elif market_state == MarketState.VOLATILE:
            return 60   # 1 minute for volatile markets
        else:
            return 300  # Default 5 minutes
    
    def _assess_risk_level(self, market_story: Dict, confidence: float) -> str:
        """Assess risk level of the trade"""
        if confidence > 0.8:
            return 'low'
        elif confidence > 0.6:
            return 'medium'
        else:
            return 'high'
    
    def _serialize_strategy_tree(self, strategy_tree: StrategyTree) -> Dict:
        """Serialize strategy tree for JSON output"""
        return {
            'strategy_type': strategy_tree.strategy_type.value,
            'overall_confidence': strategy_tree.overall_confidence,
            'predicted_direction': strategy_tree.predicted_direction.value,
            'reasoning': strategy_tree.reasoning,
            'expiry_recommendation': strategy_tree.expiry_recommendation,
            'risk_level': strategy_tree.risk_level,
            'nodes': [
                {
                    'condition': node.condition,
                    'weight': node.weight,
                    'signal_type': node.signal_type,
                    'confidence': node.confidence,
                    'reasoning': node.reasoning
                }
                for node in strategy_tree.nodes
            ]
        }
    
    def _generate_trading_signal(self, strategy_tree: StrategyTree, market_data: Dict) -> Dict:
        """Generate trading signal from strategy tree"""
        return {
            'direction': strategy_tree.predicted_direction.value,
            'confidence': strategy_tree.overall_confidence,
            'strategy_type': strategy_tree.strategy_type.value,
            'expiry_time': strategy_tree.expiry_recommendation,
            'risk_level': strategy_tree.risk_level,
            'reasoning': strategy_tree.reasoning,
            'recommended_amount': 10.0,  # Would be calculated based on risk management
            'timestamp': datetime.now().isoformat()
        }
    
    def _update_pattern_memory(self, market_story: Dict, strategy_tree: StrategyTree):
        """Update AI pattern memory for learning"""
        pattern_key = f"{strategy_tree.strategy_type.value}_{market_story['elements']['market_state'].value}"
        
        if pattern_key not in self.pattern_memory:
            self.pattern_memory[pattern_key] = []
        
        self.pattern_memory[pattern_key].append({
            'confidence': strategy_tree.overall_confidence,
            'direction': strategy_tree.predicted_direction.value,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only recent patterns (last 1000)
        if len(self.pattern_memory[pattern_key]) > 1000:
            self.pattern_memory[pattern_key] = self.pattern_memory[pattern_key][-1000:]
    
    def _load_strategy_patterns(self) -> Dict:
        """Load predefined strategy patterns"""
        return {
            'breakout_patterns': [
                'ascending_triangle',
                'bull_flag',
                'resistance_break'
            ],
            'reversal_patterns': [
                'double_bottom',
                'hammer',
                'doji'
            ],
            'momentum_patterns': [
                'macd_cross',
                'rsi_momentum',
                'moving_average_cross'
            ]
        }
    
    def _load_psychology_rules(self) -> Dict:
        """Load market psychology rules"""
        return {
            'fear_indicators': [
                'rsi_below_20',
                'volume_spike_down',
                'gap_down'
            ],
            'greed_indicators': [
                'rsi_above_80',
                'volume_spike_up',
                'gap_up'
            ],
            'uncertainty_indicators': [
                'doji_candles',
                'low_volume',
                'narrow_range'
            ]
        }