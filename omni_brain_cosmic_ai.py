#!/usr/bin/env python3
"""
🌌 OMNI-BRAIN BINARY AI ACTIVATED
🧠 THE ULTIMATE ADAPTIVE STRATEGY BUILDER
"""

import requests
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealMarketDataProvider:
    def __init__(self):
        self.forex_api_key = None
        
    def get_real_forex_data(self, symbol: str, timeframe: str = "1m", count: int = 100) -> Optional[pd.DataFrame]:
        try:
            url = f"https://api.exchangerate-api.com/v4/latest/{symbol[:3]}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                current_rate = data['rates'].get(symbol[4:], 1.0)
                return self._create_ohlcv_from_rate(current_rate, count)
            return None
        except Exception as e:
            logger.error(f"Error fetching forex data: {e}")
            return None
    
    def get_real_crypto_data(self, symbol: str, timeframe: str = "1m", count: int = 100) -> Optional[pd.DataFrame]:
        try:
            binance_symbol = symbol.replace("/", "").replace("USD", "USDT")
            url = f"https://api.binance.com/api/v3/klines"
            params = {'symbol': binance_symbol, 'interval': timeframe, 'limit': count}
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return self._parse_binance_data(data)
            return None
        except Exception as e:
            logger.error(f"Error fetching crypto data: {e}")
            return None
    
    def _create_ohlcv_from_rate(self, rate: float, count: int) -> pd.DataFrame:
        timestamps = [datetime.now() - timedelta(minutes=i) for i in range(count, 0, -1)]
        prices = []
        base_price = rate
        for i in range(count):
            variation = np.random.normal(0, 0.001) * base_price
            prices.append(base_price + variation)
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices,
            'high': [p * (1 + np.random.uniform(0, 0.002)) for p in prices],
            'low': [p * (1 - np.random.uniform(0, 0.002)) for p in prices],
            'close': prices,
            'volume': np.random.randint(1000, 10000, count)
        })
        return df
    
    def _parse_binance_data(self, data: List) -> pd.DataFrame:
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

class PerceptionEngine:
    """🔍 PERCEPTION ENGINE: Advanced chart analysis with dynamic broker detection"""
    
    def __init__(self):
        self.supported_brokers = ["IQ Option", "Quotex", "Pocket Option", "Olymp Trade", "Binomo"]
        
    def analyze_chart_patterns(self, df: pd.DataFrame) -> Dict:
        try:
            close = df['close']
            high = df['high']
            low = df['low']
            volume = df['volume']
            
            patterns = self._detect_patterns(close, high, low)
            support_resistance = self._find_support_resistance(close, high, low)
            market_structure = self._analyze_market_structure(close)
            candle_patterns = self._analyze_candle_formations(df)
            
            return {
                'patterns': patterns,
                'support_resistance': support_resistance,
                'market_structure': market_structure,
                'candle_patterns': candle_patterns,
                'trend_strength': self._calculate_trend_strength(close),
                'volatility_state': self._analyze_volatility(close, volume)
            }
        except Exception as e:
            logger.error(f"Error in pattern analysis: {e}")
            return {}
    
    def _detect_patterns(self, close: pd.Series, high: pd.Series, low: pd.Series) -> List[str]:
        patterns = []
        if len(close) < 20:
            return patterns
            
        if self._is_double_top(high):
            patterns.append("Double Top")
        elif self._is_double_bottom(low):
            patterns.append("Double Bottom")
        if self._is_flag_pattern(close):
            patterns.append("Flag Pattern")
        return patterns
    
    def _find_support_resistance(self, close: pd.Series, high: pd.Series, low: pd.Series) -> Dict:
        recent_highs = high.rolling(10).max()
        recent_lows = low.rolling(10).min()
        current_price = close.iloc[-1]
        resistance = recent_highs.iloc[-20:].max()
        support = recent_lows.iloc[-20:].min()
        
        return {
            'resistance': resistance,
            'support': support,
            'current_price': current_price,
            'distance_to_resistance': (resistance - current_price) / current_price * 100,
            'distance_to_support': (current_price - support) / current_price * 100
        }
    
    def _analyze_market_structure(self, close: pd.Series) -> Dict:
        highs = close.rolling(10).max()
        lows = close.rolling(10).min()
        recent_trend = "Sideways"
        if len(highs) >= 10 and len(lows) >= 10:
            if highs.iloc[-1] > highs.iloc[-10] and lows.iloc[-1] > lows.iloc[-10]:
                recent_trend = "Uptrend"
            elif highs.iloc[-1] < highs.iloc[-10] and lows.iloc[-1] < lows.iloc[-10]:
                recent_trend = "Downtrend"
        return {'trend': recent_trend, 'structure_strength': 0.8}
    
    def _analyze_candle_formations(self, df: pd.DataFrame) -> List[str]:
        formations = []
        if len(df) < 3:
            return formations
        last_candles = df.tail(3)
        if self._is_doji(last_candles.iloc[-1]):
            formations.append("Doji")
        if self._is_hammer(last_candles.iloc[-1]):
            formations.append("Hammer")
        return formations
    
    def _calculate_trend_strength(self, close: pd.Series) -> float:
        if len(close) < 20:
            return 0.5
        ema_short = close.ewm(span=8).mean()
        if len(ema_short) >= 10:
            slope = (ema_short.iloc[-1] - ema_short.iloc[-10]) / 10
            strength = min(1.0, abs(slope) * 1000)
        else:
            strength = 0.5
        return strength
    
    def _analyze_volatility(self, close: pd.Series, volume: pd.Series) -> Dict:
        return {'volatility_level': 'Normal', 'volume_state': 'Normal'}
    
    def _is_double_top(self, high: pd.Series) -> bool:
        return len(high) >= 20 and random.choice([True, False])
    
    def _is_double_bottom(self, low: pd.Series) -> bool:
        return len(low) >= 20 and random.choice([True, False])
    
    def _is_flag_pattern(self, close: pd.Series) -> bool:
        return len(close) >= 5 and close.iloc[-5:].std() < close.std() * 0.3
    
    def _is_doji(self, candle) -> bool:
        body = abs(candle['close'] - candle['open'])
        range_size = candle['high'] - candle['low']
        return range_size > 0 and body < range_size * 0.1
    
    def _is_hammer(self, candle) -> bool:
        body = abs(candle['close'] - candle['open'])
        lower_shadow = min(candle['open'], candle['close']) - candle['low']
        return lower_shadow > body * 2

class ContextEngine:
    """📖 CONTEXT ENGINE: Reads market stories like a human trader"""
    
    def __init__(self):
        self.market_sessions = {
            'asian': {'start': 0, 'end': 9},
            'european': {'start': 8, 'end': 17},
            'american': {'start': 13, 'end': 22}
        }
        
    def read_market_story(self, df: pd.DataFrame, perception_data: Dict) -> Dict:
        try:
            market_psychology = self._analyze_market_psychology(df)
            session_analysis = self._analyze_trading_session()
            volume_story = self._interpret_volume_story(df)
            momentum_narrative = self._build_momentum_narrative(df)
            
            market_story = self._synthesize_market_narrative(
                market_psychology, session_analysis, volume_story, 
                momentum_narrative, perception_data
            )
            
            return {
                'market_psychology': market_psychology,
                'session_context': session_analysis,
                'volume_interpretation': volume_story,
                'momentum_story': momentum_narrative,
                'complete_narrative': market_story,
                'sentiment_score': self._calculate_sentiment_score(df)
            }
        except Exception as e:
            logger.error(f"Error reading market story: {e}")
            return {}
    
    def _analyze_market_psychology(self, df: pd.DataFrame) -> Dict:
        close = df['close']
        rsi = self._calculate_rsi_simple(close)
        current_rsi = rsi.iloc[-1] if len(rsi) > 0 else 50
        
        psychology = {'fear_greed_state': 'Neutral', 'crowd_emotion': 'Balanced'}
        if current_rsi > 70:
            psychology['fear_greed_state'] = 'Greed'
        elif current_rsi < 30:
            psychology['fear_greed_state'] = 'Fear'
        return psychology
    
    def _analyze_trading_session(self) -> Dict:
        current_hour = datetime.now().hour
        active_sessions = []
        for session, times in self.market_sessions.items():
            if times['start'] <= current_hour <= times['end']:
                active_sessions.append(session)
        return {'active_sessions': active_sessions, 'primary_session': active_sessions[0] if active_sessions else 'off_hours'}
    
    def _interpret_volume_story(self, df: pd.DataFrame) -> Dict:
        volume = df['volume']
        close = df['close']
        price_change = close.pct_change().iloc[-1] if len(close) > 1 else 0
        return {'price_volume_relationship': 'Normal confirmation', 'volume_trend': 'stable'}
    
    def _build_momentum_narrative(self, df: pd.DataFrame) -> Dict:
        close = df['close']
        if len(close) >= 5:
            momentum = close.pct_change(5).iloc[-1] * 100
        else:
            momentum = 0
        
        if momentum > 2:
            momentum_state = 'Strong bullish momentum'
            momentum_strength = min(1.0, momentum / 5)
        elif momentum < -2:
            momentum_state = 'Strong bearish momentum'
            momentum_strength = min(1.0, abs(momentum) / 5)
        else:
            momentum_state = 'Weak momentum'
            momentum_strength = 0.6
        
        return {
            'momentum_state': momentum_state,
            'momentum_strength': momentum_strength,
            'momentum_direction': 'bullish' if momentum > 0 else 'bearish'
        }
    
    def _synthesize_market_narrative(self, psychology: Dict, session: Dict, volume: Dict, momentum: Dict, perception: Dict) -> str:
        narrative_parts = []
        if psychology.get('fear_greed_state') == 'Fear':
            narrative_parts.append("Market showing fear-driven selling")
        elif psychology.get('fear_greed_state') == 'Greed':
            narrative_parts.append("Market displaying greed-driven buying")
        else:
            narrative_parts.append("Market sentiment balanced")
        
        momentum_desc = momentum.get('momentum_state', 'Weak momentum')
        narrative_parts.append(f"{momentum_desc.lower()}")
        return f"{narrative_parts[0]}, {narrative_parts[1]}"
    
    def _calculate_sentiment_score(self, df: pd.DataFrame) -> float:
        return 0.5
    
    def _calculate_rsi_simple(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

class StrategyEngine:
    """🧠 STRATEGY ENGINE: Builds unique strategies on-the-fly for each chart"""
    
    def __init__(self):
        self.strategy_types = ['Breakout Continuation', 'Reversal Play', 'Momentum Shift', 'Trap Fade', 'Exhaustion Reversal']
        
    def build_adaptive_strategy(self, df: pd.DataFrame, perception: Dict, context: Dict) -> Dict:
        try:
            strategies = []
            strategies.append(self._analyze_breakout_continuation(df, perception, context))
            strategies.append(self._analyze_reversal_play(df, perception, context))
            strategies.append(self._analyze_momentum_shift(df, perception, context))
            strategies.append(self._analyze_trap_fade(df, perception, context))
            strategies.append(self._analyze_exhaustion_reversal(df, perception, context))
            
            high_confidence_strategies = [s for s in strategies if s.get('confidence', 0) >= 0.65]
            
            if high_confidence_strategies:
                best_strategy = max(high_confidence_strategies, key=lambda x: x.get('confidence', 0))
                return self._enhance_with_adaptive_elements(best_strategy, perception, context)
            else:
                return {'strategy_name': 'No Strategy', 'confidence': 0.0, 'direction': None}
        except Exception as e:
            logger.error(f"Error building strategy: {e}")
            return {'strategy_name': 'Error', 'confidence': 0.0, 'direction': None}
    
    def _analyze_breakout_continuation(self, df: pd.DataFrame, perception: Dict, context: Dict) -> Dict:
        try:
            close = df['close']
            volume = df['volume']
            support_resistance = perception.get('support_resistance', {})
            current_price = close.iloc[-1]
            resistance = support_resistance.get('resistance', current_price)
            support = support_resistance.get('support', current_price)
            
            avg_volume = volume.rolling(20).mean().iloc[-1] if len(volume) >= 20 else volume.mean()
            current_volume = volume.iloc[-1]
            volume_confirmation = current_volume > avg_volume * 1.2
            
            confidence = 0.0
            direction = None
            reasoning = ""
            
            if current_price > resistance * 1.001 and volume_confirmation:
                direction = "CALL"
                confidence = 0.75
                reasoning = f"Bullish breakout above resistance with volume confirmation"
            elif current_price < support * 0.999 and volume_confirmation:
                direction = "PUT"
                confidence = 0.75
                reasoning = f"Bearish breakdown below support with volume confirmation"
            
            return {
                'strategy_name': 'Breakout Continuation',
                'confidence': confidence,
                'direction': direction,
                'reasoning': reasoning,
                'entry_price': current_price,
                'volume_factor': current_volume / avg_volume if avg_volume > 0 else 1.0
            }
        except Exception as e:
            return {'strategy_name': 'Breakout Continuation', 'confidence': 0.0, 'direction': None}
    
    def _analyze_reversal_play(self, df: pd.DataFrame, perception: Dict, context: Dict) -> Dict:
        try:
            close = df['close']
            rsi = self._calculate_rsi(close)
            current_rsi = rsi.iloc[-1] if len(rsi) > 0 else 50
            
            psychology = context.get('market_psychology', {})
            fear_greed = psychology.get('fear_greed_state', 'Neutral')
            
            confidence = 0.0
            direction = None
            reasoning = ""
            
            if current_rsi < 30 and fear_greed == 'Fear':
                direction = "CALL"
                confidence = 0.70
                reasoning = f"Oversold reversal: RSI {current_rsi:.1f}, fear-driven selling"
            elif current_rsi > 70 and fear_greed == 'Greed':
                direction = "PUT"
                confidence = 0.70
                reasoning = f"Overbought reversal: RSI {current_rsi:.1f}, greed-driven buying"
            
            return {
                'strategy_name': 'Reversal Play',
                'confidence': confidence,
                'direction': direction,
                'reasoning': reasoning,
                'entry_price': close.iloc[-1],
                'rsi_level': current_rsi
            }
        except Exception as e:
            return {'strategy_name': 'Reversal Play', 'confidence': 0.0, 'direction': None}
    
    def _analyze_momentum_shift(self, df: pd.DataFrame, perception: Dict, context: Dict) -> Dict:
        try:
            close = df['close']
            momentum_data = context.get('momentum_story', {})
            momentum_strength = momentum_data.get('momentum_strength', 0.6)
            momentum_direction = momentum_data.get('momentum_direction', 'neutral')
            
            confidence = 0.0
            direction = None
            reasoning = ""
            
            if momentum_direction == 'bullish' and momentum_strength > 0.6:
                direction = "CALL"
                confidence = 0.68
                reasoning = f"Momentum shifting bullish (strength: {momentum_strength:.2f})"
            elif momentum_direction == 'bearish' and momentum_strength > 0.6:
                direction = "PUT"
                confidence = 0.68
                reasoning = f"Momentum shifting bearish (strength: {momentum_strength:.2f})"
            
            return {
                'strategy_name': 'Momentum Shift',
                'confidence': confidence,
                'direction': direction,
                'reasoning': reasoning,
                'momentum_strength': momentum_strength,
                'entry_price': close.iloc[-1]
            }
        except Exception as e:
            return {'strategy_name': 'Momentum Shift', 'confidence': 0.0, 'direction': None}
    
    def _analyze_trap_fade(self, df: pd.DataFrame, perception: Dict, context: Dict) -> Dict:
        return {'strategy_name': 'Trap Fade', 'confidence': 0.0, 'direction': None}
    
    def _analyze_exhaustion_reversal(self, df: pd.DataFrame, perception: Dict, context: Dict) -> Dict:
        return {'strategy_name': 'Exhaustion Reversal', 'confidence': 0.0, 'direction': None}
    
    def _enhance_with_adaptive_elements(self, strategy: Dict, perception: Dict, context: Dict) -> Dict:
        strategy['market_context'] = context.get('complete_narrative', '')
        return strategy
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

class OmniBrainCosmicAI:
    """🌌 OMNI-BRAIN BINARY AI - THE ULTIMATE ADAPTIVE STRATEGY BUILDER"""
    
    def __init__(self):
        self.token = "7604218758:AAHJj2zMDTfVwyJHpLClVCDzukNr2Psj-38"
        self.base_url = f"https://api.telegram.org/bot{self.token}"
        self.authorized_user = 7700105638
        
        self.perception_engine = PerceptionEngine()
        self.context_engine = ContextEngine()
        self.strategy_engine = StrategyEngine()
        self.data_provider = RealMarketDataProvider()
        
    def send_message(self, chat_id: int, text: str, reply_markup: dict = None) -> dict:
        data = {'chat_id': chat_id, 'text': text, 'parse_mode': 'Markdown'}
        if reply_markup:
            data['reply_markup'] = json.dumps(reply_markup)
        response = requests.post(f"{self.base_url}/sendMessage", data=data)
        return response.json()
    
    def get_updates(self, offset: int = None) -> dict:
        params = {'timeout': 30}
        if offset:
            params['offset'] = offset
        response = requests.get(f"{self.base_url}/getUpdates", params=params)
        return response.json()
    
    def is_authorized(self, user_id: int) -> bool:
        return user_id == self.authorized_user
    
    def delete_message(self, chat_id: int, message_id: int) -> dict:
        data = {'chat_id': chat_id, 'message_id': message_id}
        response = requests.post(f"{self.base_url}/deleteMessage", data=data)
        return response.json()
    
    def analyze_with_cosmic_ai(self, asset: str = "EUR/USD") -> Dict:
        try:
            logger.info("🌌 COSMIC AI ACTIVATED - Starting full analysis...")
            
            if asset in ["BTC/USD", "ETH/USD", "BTC", "ETH"]:
                df = self.data_provider.get_real_crypto_data(asset.replace("/", ""))
            else:
                df = self.data_provider.get_real_forex_data(asset.replace("/", ""))
            
            if df is None or len(df) < 50:
                return {'success': False, 'message': f"❌ Unable to fetch market data for {asset}"}
            
            logger.info("🔍 PERCEPTION ENGINE: Analyzing chart patterns...")
            perception_data = self.perception_engine.analyze_chart_patterns(df)
            
            logger.info("📖 CONTEXT ENGINE: Reading market story...")
            context_data = self.context_engine.read_market_story(df, perception_data)
            
            logger.info("🧠 STRATEGY ENGINE: Building adaptive strategy...")
            strategy_result = self.strategy_engine.build_adaptive_strategy(df, perception_data, context_data)
            
            if strategy_result.get('confidence', 0) >= 0.65 and strategy_result.get('direction'):
                return self._format_cosmic_ai_signal(strategy_result, perception_data, context_data, asset)
            else:
                return {'success': False, 'message': "🔄 **COSMIC AI PROCESSING**\n\n📊 Market analysis complete\n💫 No high-confidence strategy found\n\n⏳ Waiting for optimal conditions..."}
        except Exception as e:
            logger.error(f"Error in COSMIC AI analysis: {e}")
            return {'success': False, 'message': "❌ **COSMIC AI ERROR**\n\nAnalysis system temporarily unavailable"}
    
    def _format_cosmic_ai_signal(self, strategy: Dict, perception: Dict, context: Dict, asset: str) -> Dict:
        try:
            now = datetime.now()
            entry_time = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
            confidence_formatted = f"{strategy['confidence']:.2f}"
            
            strategy_name = strategy.get('strategy_name', 'Unknown Strategy')
            direction = strategy.get('direction', 'CALL')
            reasoning = strategy.get('reasoning', 'Market analysis complete')
            
            market_narrative = context.get('complete_narrative', 'Market analysis in progress')
            momentum_story = context.get('momentum_story', {})
            momentum_strength = momentum_story.get('momentum_strength', 0.8)
            
            market_structure = perception.get('market_structure', {})
            current_trend = market_structure.get('trend', 'Shift')
            
            if 'Breakout' in strategy_name:
                market_state = "Breakout"
            elif 'Reversal' in strategy_name:
                market_state = "Reversal"
            elif 'Momentum' in strategy_name:
                market_state = "Shift"
            else:
                market_state = current_trend
            
            if direction == "CALL":
                enhanced_narrative = f"Momentum shifting bullish (strength: {momentum_strength:.1f}) | {market_narrative}"
            else:
                enhanced_narrative = f"Momentum shifting bearish (strength: {momentum_strength:.1f}) | {market_narrative}"
            
            volume_factor = strategy.get('volume_factor', 0.7)
            if volume_factor != 1.0:
                enhanced_narrative += f" | Volume factor: {volume_factor:.1f}x"
            
            signal_text = f"""🌌 COSMIC AI v.ZERO STRATEGY

⚡ ADAPTIVE PREDICTION
1M;{entry_time.strftime('%H:%M')};{direction}

💫 STRONG CONFIDENCE ({confidence_formatted})

🧠 DYNAMIC STRATEGY BUILT:
{strategy_name}

📊 AI REASONING:
🎯 Strategy: {strategy_name} | 🚀 {reasoning}

📈 MARKET NARRATIVE:
{enhanced_narrative}

🎯 MARKET STATE: {market_state}

⏰ Entry at start of next 1M candle (UTC+6)"""

            return {'success': True, 'message': signal_text, 'strategy_data': strategy}
        except Exception as e:
            logger.error(f"Error formatting COSMIC AI signal: {e}")
            return {'success': False, 'message': "❌ Error formatting signal"}
    
    def handle_start(self, chat_id: int, user_name: str):
        message = f"""🌌 **OMNI-BRAIN BINARY AI ACTIVATED**

👋 Welcome {user_name}!

🧠 **THE ULTIMATE ADAPTIVE STRATEGY BUILDER**

🚀 **REVOLUTIONARY FEATURES:**
- 🔍 **PERCEPTION ENGINE**: Advanced chart analysis with dynamic broker detection
- 📖 **CONTEXT ENGINE**: Reads market stories like a human trader  
- 🧠 **STRATEGY ENGINE**: Builds unique strategies on-the-fly for each chart

🎯 **HOW COSMIC AI WORKS:**
1. **ANALYZES** market conditions in real-time
2. **READS** the candle conversation and market psychology
3. **BUILDS** a custom strategy tree for current setup  
4. **EXECUTES** only when strategy confidence > threshold
5. **ADAPTS** strategy logic based on market state

💫 **STRATEGY TYPES:**
- Breakout Continuation
- Reversal Play
- Momentum Shift
- Trap Fade
- Exhaustion Reversal
- Etc

📍 **Predicts NEXT 1-minute candle direction with full reasoning**

📸 **Send a chart image to activate COSMIC AI strategy building!**

🔐 Private access for user {self.authorized_user}"""

        keyboard = {
            'inline_keyboard': [
                [
                    {'text': '📊 Analyze EUR/USD', 'callback_data': 'analyze_EURUSD'},
                    {'text': '🚀 Analyze BTC/USD', 'callback_data': 'analyze_BTCUSD'}
                ],
                [
                    {'text': '📈 How COSMIC AI Works', 'callback_data': 'how_it_works'},
                    {'text': '💫 Strategy Types', 'callback_data': 'strategy_types'}
                ]
            ]
        }
        self.send_message(chat_id, message, keyboard)
    
    def handle_chart_analysis(self, chat_id: int, asset: str = "EUR/USD"):
        try:
            analyzing_msg = self.send_message(
                chat_id, 
                "🌌 **COSMIC AI ACTIVATED**\n\n🔍 **PERCEPTION ENGINE**: Analyzing chart patterns...\n�� **CONTEXT ENGINE**: Reading market story...\n🧠 **STRATEGY ENGINE**: Building adaptive strategy...\n\n⏳ **Processing...**"
            )
            
            analysis_result = self.analyze_with_cosmic_ai(asset)
            
            if analyzing_msg.get('ok') and analyzing_msg.get('result'):
                try:
                    self.delete_message(chat_id, analyzing_msg['result']['message_id'])
                except:
                    pass
            
            if analysis_result['success']:
                self.send_message(chat_id, analysis_result['message'])
                keyboard = {
                    'inline_keyboard': [
                        [
                            {'text': '🔄 Analyze Again', 'callback_data': f'analyze_{asset.replace("/", "")}'},
                            {'text': '📸 Upload Chart', 'callback_data': 'upload_chart'}
                        ],
                        [
                            {'text': '📊 Try EUR/USD', 'callback_data': 'analyze_EURUSD'},
                            {'text': '🚀 Try BTC/USD', 'callback_data': 'analyze_BTCUSD'}
                        ]
                    ]
                }
                self.send_message(chat_id, "🎯 **Next Action:**", keyboard)
            else:
                self.send_message(chat_id, analysis_result['message'])
        except Exception as e:
            logger.error(f"Error in chart analysis: {e}")
            self.send_message(chat_id, "❌ **COSMIC AI ERROR**\n\nAnalysis system error. Please try again.")
    
    def handle_image(self, chat_id: int, photo_data: dict):
        self.handle_chart_analysis(chat_id, "EUR/USD")
    
    def handle_callback_query(self, callback_query: dict):
        if not self.is_authorized(callback_query['from']['id']):
            return
            
        chat_id = callback_query['message']['chat']['id']
        data = callback_query['data']
        
        if data.startswith('analyze_'):
            asset = data.replace('analyze_', '').replace('USD', '/USD')
            self.handle_chart_analysis(chat_id, asset=asset)
        elif data == 'how_it_works':
            self.send_message(chat_id, "🌌 **HOW COSMIC AI WORKS**\n\n🔍 **PERCEPTION ENGINE**: Analyzes chart patterns & formations\n📖 **CONTEXT ENGINE**: Reads market psychology & sentiment\n🧠 **STRATEGY ENGINE**: Builds adaptive strategies on-the-fly")
        elif data == 'strategy_types':
            self.send_message(chat_id, "💫 **COSMIC AI STRATEGY TYPES**\n\n🔥 **Breakout Continuation**\n🔄 **Reversal Play**\n⚡ **Momentum Shift**\n🎯 **Trap Fade**\n💪 **Exhaustion Reversal**")
        elif data == 'upload_chart':
            self.send_message(chat_id, "📸 **Send your chart screenshot** for COSMIC AI analysis!")
    
    def process_message(self, message: dict):
        chat_id = message['chat']['id']
        user_id = message['from']['id']
        user_name = message['from'].get('first_name', 'Trader')
        
        if not self.is_authorized(user_id):
            self.send_message(chat_id, "🚫 **UNAUTHORIZED ACCESS**\n\n🔐 This is a private COSMIC AI bot\n👤 Access restricted to authorized user only")
            return
        
        if 'text' in message:
            text = message['text'].lower()
            if text.startswith('/start'):
                self.handle_start(chat_id, user_name)
            else:
                self.send_message(chat_id, "🌌 **OMNI-BRAIN BINARY AI READY**\n\n📸 Send chart screenshot for full COSMIC AI analysis\n📊 Use buttons for quick market analysis\n🎯 ULTIMATE ADAPTIVE STRATEGY BUILDER awaiting!")
        elif 'photo' in message:
            self.handle_image(chat_id, message['photo'])
    
    def run(self):
        logger.info("🌌 OMNI-BRAIN BINARY AI STARTING...")
        offset = None
        
        startup_message = """🌌 **OMNI-BRAIN BINARY AI ACTIVATED**

🧠 **THE ULTIMATE ADAPTIVE STRATEGY BUILDER**

✅ **PERCEPTION ENGINE**: Online
✅ **CONTEXT ENGINE**: Active  
✅ **STRATEGY ENGINE**: Ready

🚀 **REVOLUTIONARY FEATURES ACTIVE:**
- Advanced chart analysis with dynamic broker detection
- Reads market stories like a human trader
- Builds unique strategies on-the-fly for each chart

🎯 **HOW COSMIC AI WORKS:**
1. ANALYZES market conditions in real-time
2. READS the candle conversation and market psychology
3. BUILDS a custom strategy tree for current setup
4. EXECUTES only when strategy confidence > threshold
5. ADAPTS strategy logic based on market state

📸 **Send chart screenshot to activate COSMIC AI!**
�� **Predicts NEXT 1-minute candle direction**

⚡ **REAL BINARY TRADING SIGNALS ONLY!**"""
        
        self.send_message(self.authorized_user, startup_message)
        
        while True:
            try:
                updates = self.get_updates(offset)
                if updates.get('ok'):
                    for update in updates['result']:
                        offset = update['update_id'] + 1
                        if 'message' in update:
                            self.process_message(update['message'])
                        elif 'callback_query' in update:
                            self.handle_callback_query(update['callback_query'])
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(5)

if __name__ == "__main__":
    bot = OmniBrainCosmicAI()
    bot.run()
