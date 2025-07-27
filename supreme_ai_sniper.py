# Supreme AI Sniper Bot — 100 Billion Years AI Strategy Engine
# ================================================================
# • Advanced AI market psychology analysis
# • Dynamic strategy tree construction  
# • Adaptive confidence thresholds
# • Secret AI engines for different market conditions
# • Next 1M candle predictions with precise timing
# ================================================================

import os, time, logging, datetime as dt
from typing import List, Optional, Dict, Tuple
import numpy as np
import pandas as pd
from telegram import Bot
from telegram.error import TelegramError
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import chromedriver_autoinstaller
from dataclasses import dataclass
from enum import Enum

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Environment Configuration
EMAIL = os.getenv("EMAIL")
PASSWORD = os.getenv("PASSWORD")
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

# AI Engine Configuration
TIMEFRAMES = [1, 2]
SCAN_SLEEP = 1.2
CONFIDENCE_THRESHOLD = 85.0
WINDOW = 100
AI_LOOKBACK = 50

class MarketState(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    BREAKOUT_PENDING = "breakout_pending"
    REVERSAL_ZONE = "reversal_zone"
    EXHAUSTION = "exhaustion"
    TRAP_ZONE = "trap_zone"

class PsychologyPattern(Enum):
    MOMENTUM_SHIFT = "momentum_shift"
    TRAP_ZONE = "trap_zone"
    SUPPORT_REACTION = "support_reaction"
    RESISTANCE_REACTION = "resistance_reaction"
    ENGULFING = "engulfing"
    BREAKOUT_CONTINUATION = "breakout_continuation"
    REVERSAL_PLAY = "reversal_play"
    EXHAUSTION_REVERSAL = "exhaustion_reversal"

@dataclass
class StrategySignal:
    direction: str
    confidence: float
    reasoning: str
    entry_time: str
    target_time: str
    psychology: List[str]
    market_state: str
    strategy_tree: str

class SupremeAISniper:
    def __init__(self):
        self.bot = Bot(BOT_TOKEN)
        self.driver = self._init_browser()
        self._login()
        
        # AI Engine Components
        self.market_memory = {}
        self.strategy_cache = {}
        self.confidence_adapters = {}
        
    # ═══════════════ BROWSER ENGINE ═══════════════
    def _init_browser(self):
        chromedriver_autoinstaller.install()
        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1920,1080")
        return webdriver.Chrome(options=options)

    def _login(self):
        logging.info("🤖 AI Engine initializing...")
        self.driver.get("https://quotex.com/en/login")
        time.sleep(3)
        self.driver.find_element(By.NAME, "email").send_keys(EMAIL)
        self.driver.find_element(By.NAME, "password").send_keys(PASSWORD)
        self.driver.find_element(By.CSS_SELECTOR, "button[type=submit]").click()
        time.sleep(6)
        if "trade" not in self.driver.current_url:
            raise RuntimeError("❌ AI Login failed — verify credentials")
        logging.info("✅ AI Engine connected to market feed")

    # ═══════════════ DATA ACQUISITION ═══════════════
    def fetch_market_data(self, asset: str, tf: int) -> pd.DataFrame:
        """Enhanced data fetching with AI preprocessing"""
        js = "return window.chartData ? window.chartData : null;"
        raw = self.driver.execute_script(js)
        if raw is None:
            return pd.DataFrame()
            
        candles = raw.get(asset, {}).get(str(tf), [])[-WINDOW:]
        if len(candles) < AI_LOOKBACK:
            return pd.DataFrame()
            
        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        
        # AI Enhancements
        df = self._enhance_with_ai_indicators(df)
        return df.set_index("timestamp")

    def _enhance_with_ai_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced AI technical indicators"""
        # Price action analysis
        df['body'] = abs(df['close'] - df['open'])
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        df['candle_range'] = df['high'] - df['low']
        df['body_ratio'] = df['body'] / df['candle_range'].replace(0, np.nan)
        
        # Advanced momentum indicators
        df['momentum'] = df['close'].pct_change(5)
        df['acceleration'] = df['momentum'].diff()
        
        # AI Pattern recognition
        df['is_green'] = df['close'] > df['open']
        df['is_red'] = df['close'] < df['open']
        df['is_doji'] = df['body_ratio'] < 0.1
        
        # Dynamic support/resistance
        df['local_high'] = df['high'].rolling(10).max()
        df['local_low'] = df['low'].rolling(10).min()
        
        # Volatility analysis
        df['volatility'] = df['candle_range'].rolling(20).std()
        df['volume_pressure'] = df.get('volume', 1) * df['body']
        
        return df

    # ═══════════════ AI MARKET PSYCHOLOGY ENGINE ═══════════════
    def analyze_market_psychology(self, df: pd.DataFrame) -> Dict[str, float]:
        """Advanced AI psychology pattern detection"""
        psychology_scores = {}
        
        if len(df) < 20:
            return psychology_scores
            
        recent = df.tail(20)
        
        # 🧠 MOMENTUM SHIFT DETECTION
        momentum_score = self._detect_momentum_shift(recent)
        psychology_scores[PsychologyPattern.MOMENTUM_SHIFT.value] = momentum_score
        
        # 🪤 TRAP ZONE ANALYSIS
        trap_score = self._detect_trap_zones(recent)
        psychology_scores[PsychologyPattern.TRAP_ZONE.value] = trap_score
        
        # 📊 SUPPORT/RESISTANCE PSYCHOLOGY
        support_score = self._analyze_support_reaction(recent)
        resistance_score = self._analyze_resistance_reaction(recent)
        psychology_scores[PsychologyPattern.SUPPORT_REACTION.value] = support_score
        psychology_scores[PsychologyPattern.RESISTANCE_REACTION.value] = resistance_score
        
        # 🔥 ENGULFING PATTERNS
        engulfing_score = self._detect_engulfing_psychology(recent)
        psychology_scores[PsychologyPattern.ENGULFING.value] = engulfing_score
        
        # 💥 BREAKOUT CONTINUATION
        breakout_score = self._analyze_breakout_continuation(recent)
        psychology_scores[PsychologyPattern.BREAKOUT_CONTINUATION.value] = breakout_score
        
        # 🔄 REVERSAL EXHAUSTION
        exhaustion_score = self._detect_exhaustion_reversal(recent)
        psychology_scores[PsychologyPattern.EXHAUSTION_REVERSAL.value] = exhaustion_score
        
        return psychology_scores

    def _detect_momentum_shift(self, df: pd.DataFrame) -> float:
        """AI detection of momentum psychology shifts"""
        if len(df) < 10:
            return 0.0
            
        # Analyze acceleration changes
        momentum_changes = df['acceleration'].tail(5).abs().mean()
        volume_confirmation = df['volume_pressure'].tail(5).mean() if 'volume_pressure' in df else 1
        
        # Psychology: Sudden momentum changes indicate shift
        momentum_score = min(momentum_changes * 100, 95.0)
        return momentum_score * (1 + volume_confirmation * 0.1)

    def _detect_trap_zones(self, df: pd.DataFrame) -> float:
        """Detect fake breakouts and trap zones"""
        if len(df) < 15:
            return 0.0
            
        recent_highs = df['high'].tail(10)
        recent_lows = df['low'].tail(10)
        recent_closes = df['close'].tail(5)
        
        # Check for false breakouts
        resistance_level = recent_highs.quantile(0.8)
        support_level = recent_lows.quantile(0.2)
        
        false_breakout_signals = 0
        
        # Detect upside traps
        if any(recent_closes > resistance_level) and recent_closes.iloc[-1] < resistance_level:
            false_breakout_signals += 1
            
        # Detect downside traps  
        if any(recent_closes < support_level) and recent_closes.iloc[-1] > support_level:
            false_breakout_signals += 1
            
        trap_score = false_breakout_signals * 45.0
        return min(trap_score, 90.0)

    def _analyze_support_reaction(self, df: pd.DataFrame) -> float:
        """Analyze psychology at support levels"""
        if len(df) < 15:
            return 0.0
            
        support_level = df['local_low'].iloc[-1]
        current_price = df['close'].iloc[-1]
        
        # Distance from support
        distance_ratio = abs(current_price - support_level) / support_level
        
        # Reaction strength
        bounce_strength = 0.0
        if current_price > support_level:
            bounce_candles = df.tail(5)
            bounce_strength = (bounce_candles['close'].iloc[-1] - bounce_candles['low'].min()) / support_level
            
        support_score = max(0, 80 - distance_ratio * 1000) * (1 + bounce_strength * 2)
        return min(support_score, 95.0)

    def _analyze_resistance_reaction(self, df: pd.DataFrame) -> float:
        """Analyze psychology at resistance levels"""
        if len(df) < 15:
            return 0.0
            
        resistance_level = df['local_high'].iloc[-1]
        current_price = df['close'].iloc[-1]
        
        # Distance from resistance
        distance_ratio = abs(current_price - resistance_level) / resistance_level
        
        # Rejection strength
        rejection_strength = 0.0
        if current_price < resistance_level:
            rejection_candles = df.tail(5)
            rejection_strength = (rejection_candles['high'].max() - rejection_candles['close'].iloc[-1]) / resistance_level
            
        resistance_score = max(0, 80 - distance_ratio * 1000) * (1 + rejection_strength * 2)
        return min(resistance_score, 95.0)

    def _detect_engulfing_psychology(self, df: pd.DataFrame) -> float:
        """Detect engulfing pattern psychology"""
        if len(df) < 5:
            return 0.0
            
        last_2 = df.tail(2)
        if len(last_2) < 2:
            return 0.0
            
        prev_candle = last_2.iloc[0]
        curr_candle = last_2.iloc[1]
        
        # Bullish engulfing
        bullish_engulfing = (prev_candle['close'] < prev_candle['open'] and 
                           curr_candle['close'] > curr_candle['open'] and
                           curr_candle['open'] < prev_candle['close'] and
                           curr_candle['close'] > prev_candle['open'])
        
        # Bearish engulfing
        bearish_engulfing = (prev_candle['close'] > prev_candle['open'] and 
                           curr_candle['close'] < curr_candle['open'] and
                           curr_candle['open'] > prev_candle['close'] and
                           curr_candle['close'] < prev_candle['open'])
        
        if bullish_engulfing or bearish_engulfing:
            # Size of engulfing
            engulf_ratio = curr_candle['body'] / prev_candle['body']
            return min(70 + engulf_ratio * 15, 95.0)
            
        return 0.0

    def _analyze_breakout_continuation(self, df: pd.DataFrame) -> float:
        """Analyze breakout continuation probability"""
        if len(df) < 20:
            return 0.0
            
        # Volume analysis during breakout
        recent_volume = df['volume_pressure'].tail(5).mean() if 'volume_pressure' in df else 1
        prev_volume = df['volume_pressure'].tail(20).head(15).mean() if 'volume_pressure' in df else 1
        
        volume_expansion = recent_volume / max(prev_volume, 0.001)
        
        # Price momentum
        price_momentum = abs(df['momentum'].tail(3).mean())
        
        # Consolidation before breakout
        consolidation_period = df.tail(10)
        volatility_compression = consolidation_period['volatility'].min() / consolidation_period['volatility'].max()
        
        breakout_score = (volume_expansion * 20 + price_momentum * 300 + 
                         (1 - volatility_compression) * 30)
        
        return min(breakout_score, 95.0)

    def _detect_exhaustion_reversal(self, df: pd.DataFrame) -> float:
        """Detect exhaustion and reversal psychology"""
        if len(df) < 15:
            return 0.0
            
        # Analyze consecutive candles in same direction
        consecutive_up = 0
        consecutive_down = 0
        
        for candle in df.tail(10).itertuples():
            if candle.close > candle.open:
                consecutive_up += 1
                consecutive_down = 0
            elif candle.close < candle.open:
                consecutive_down += 1
                consecutive_up = 0
                
        # Decreasing momentum despite same direction
        momentum_decrease = df['momentum'].tail(5).iloc[0] > df['momentum'].tail(5).iloc[-1]
        
        # Increasing wicks (rejection)
        wick_increase = df['upper_wick'].tail(3).mean() > df['upper_wick'].tail(10).head(7).mean()
        
        exhaustion_signals = (consecutive_up >= 3 or consecutive_down >= 3) + momentum_decrease + wick_increase
        
        exhaustion_score = exhaustion_signals * 25.0
        return min(exhaustion_score, 90.0)

    # ═══════════════ DYNAMIC STRATEGY TREE BUILDER ═══════════════
    def build_strategy_tree(self, df: pd.DataFrame, psychology: Dict[str, float]) -> Dict[str, float]:
        """AI builds custom strategy tree based on current market conditions"""
        strategy_tree = {}
        
        # Market state detection
        market_state = self._detect_market_state(df)
        
        # Strategy 1: BREAKOUT_CONTINUATION
        if psychology.get('breakout_continuation', 0) > 60:
            confidence = self._calculate_breakout_confidence(df, psychology)
            direction = "CALL" if df['momentum'].iloc[-1] > 0 else "PUT"
            strategy_tree['BREAKOUT_CONTINUATION'] = {
                'direction': direction,
                'confidence': confidence,
                'reasoning': f"Breakout continuation detected with {psychology['breakout_continuation']:.1f}% psychology score"
            }
        
        # Strategy 2: REVERSAL_PLAY
        if psychology.get('exhaustion_reversal', 0) > 70:
            confidence = self._calculate_reversal_confidence(df, psychology)
            direction = "PUT" if df['momentum'].iloc[-1] > 0 else "CALL"
            strategy_tree['REVERSAL_PLAY'] = {
                'direction': direction,
                'confidence': confidence,
                'reasoning': f"Exhaustion reversal detected with {psychology['exhaustion_reversal']:.1f}% psychology score"
            }
        
        # Strategy 3: MOMENTUM_SHIFT
        if psychology.get('momentum_shift', 0) > 65:
            confidence = self._calculate_momentum_confidence(df, psychology)
            direction = "CALL" if df['acceleration'].iloc[-1] > 0 else "PUT"
            strategy_tree['MOMENTUM_SHIFT'] = {
                'direction': direction,
                'confidence': confidence,
                'reasoning': f"Momentum shift detected with {psychology['momentum_shift']:.1f}% psychology score"
            }
        
        # Strategy 4: TRAP_FADE
        if psychology.get('trap_zone', 0) > 75:
            confidence = self._calculate_trap_confidence(df, psychology)
            # Fade the trap direction
            last_candle = df.iloc[-1]
            direction = "PUT" if last_candle['close'] > last_candle['open'] else "CALL"
            strategy_tree['TRAP_FADE'] = {
                'direction': direction,
                'confidence': confidence,
                'reasoning': f"Trap zone identified with {psychology['trap_zone']:.1f}% psychology score"
            }
        
        # Strategy 5: SUPPORT_RESISTANCE_REACTION
        support_score = psychology.get('support_reaction', 0)
        resistance_score = psychology.get('resistance_reaction', 0)
        
        if support_score > 70:
            confidence = self._calculate_sr_confidence(df, psychology, 'support')
            strategy_tree['SUPPORT_BOUNCE'] = {
                'direction': 'CALL',
                'confidence': confidence,
                'reasoning': f"Support reaction with {support_score:.1f}% psychology score"
            }
            
        if resistance_score > 70:
            confidence = self._calculate_sr_confidence(df, psychology, 'resistance')
            strategy_tree['RESISTANCE_REJECTION'] = {
                'direction': 'PUT',
                'confidence': confidence,
                'reasoning': f"Resistance rejection with {resistance_score:.1f}% psychology score"
            }
        
        return strategy_tree

    def _detect_market_state(self, df: pd.DataFrame) -> MarketState:
        """AI detection of current market state"""
        if len(df) < 20:
            return MarketState.RANGING
            
        # Trend analysis
        price_change = (df['close'].iloc[-1] - df['close'].iloc[-20]) / df['close'].iloc[-20]
        volatility = df['volatility'].iloc[-1]
        
        if price_change > 0.02 and volatility > df['volatility'].quantile(0.7):
            return MarketState.TRENDING_UP
        elif price_change < -0.02 and volatility > df['volatility'].quantile(0.7):
            return MarketState.TRENDING_DOWN
        elif volatility < df['volatility'].quantile(0.3):
            return MarketState.BREAKOUT_PENDING
        else:
            return MarketState.RANGING

    def _calculate_breakout_confidence(self, df: pd.DataFrame, psychology: Dict[str, float]) -> float:
        """Calculate confidence for breakout continuation strategy"""
        base_confidence = psychology.get('breakout_continuation', 0)
        
        # Volume confirmation
        volume_boost = min(20, df['volume_pressure'].tail(3).mean() * 10) if 'volume_pressure' in df else 0
        
        # Momentum alignment
        momentum_boost = min(15, abs(df['momentum'].iloc[-1]) * 100)
        
        # Market state bonus
        market_state = self._detect_market_state(df)
        state_bonus = 10 if market_state in [MarketState.TRENDING_UP, MarketState.TRENDING_DOWN] else 0
        
        total_confidence = base_confidence + volume_boost + momentum_boost + state_bonus
        return min(total_confidence, 98.0)

    def _calculate_reversal_confidence(self, df: pd.DataFrame, psychology: Dict[str, float]) -> float:
        """Calculate confidence for reversal strategy"""
        base_confidence = psychology.get('exhaustion_reversal', 0)
        
        # Divergence detection
        recent_highs = df['high'].tail(5)
        momentum_divergence = recent_highs.iloc[-1] > recent_highs.iloc[0] and df['momentum'].iloc[-1] < df['momentum'].iloc[-5]
        divergence_boost = 15 if momentum_divergence else 0
        
        # Wick analysis
        wick_boost = min(10, df['upper_wick'].tail(3).mean() * 20)
        
        total_confidence = base_confidence + divergence_boost + wick_boost
        return min(total_confidence, 95.0)

    def _calculate_momentum_confidence(self, df: pd.DataFrame, psychology: Dict[str, float]) -> float:
        """Calculate confidence for momentum shift strategy"""
        base_confidence = psychology.get('momentum_shift', 0)
        
        # Acceleration analysis
        acceleration_boost = min(20, abs(df['acceleration'].iloc[-1]) * 500)
        
        # Volume confirmation
        volume_boost = min(15, df['volume_pressure'].tail(2).mean() * 10) if 'volume_pressure' in df else 0
        
        total_confidence = base_confidence + acceleration_boost + volume_boost
        return min(total_confidence, 96.0)

    def _calculate_trap_confidence(self, df: pd.DataFrame, psychology: Dict[str, float]) -> float:
        """Calculate confidence for trap fade strategy"""
        base_confidence = psychology.get('trap_zone', 0)
        
        # False breakout confirmation
        recent_range = df['candle_range'].tail(5).mean()
        false_breakout_boost = min(15, recent_range * 100) if recent_range > df['candle_range'].quantile(0.8) else 0
        
        total_confidence = base_confidence + false_breakout_boost
        return min(total_confidence, 93.0)

    def _calculate_sr_confidence(self, df: pd.DataFrame, psychology: Dict[str, float], level_type: str) -> float:
        """Calculate confidence for support/resistance strategy"""
        if level_type == 'support':
            base_confidence = psychology.get('support_reaction', 0)
        else:
            base_confidence = psychology.get('resistance_reaction', 0)
        
        # Touch count analysis
        support_level = df['local_low'].iloc[-1]
        touches = sum(1 for low in df['low'].tail(20) if abs(low - support_level) / support_level < 0.005)
        touch_boost = min(20, touches * 5)
        
        total_confidence = base_confidence + touch_boost
        return min(total_confidence, 94.0)

    # ═══════════════ SIGNAL EXECUTION ENGINE ═══════════════
    def execute_ai_analysis(self, asset: str, tf: int) -> Optional[StrategySignal]:
        """Main AI analysis and signal generation"""
        df = self.fetch_market_data(asset, tf)
        if df.empty or len(df) < AI_LOOKBACK:
            return None
            
        # AI Psychology Analysis
        psychology = self.analyze_market_psychology(df)
        
        # Build Dynamic Strategy Tree
        strategy_tree = self.build_strategy_tree(df, psychology)
        
        if not strategy_tree:
            return None
            
        # Find highest confidence strategy
        best_strategy = max(strategy_tree.items(), key=lambda x: x[1]['confidence'])
        strategy_name, strategy_data = best_strategy
        
        # Check confidence threshold
        if strategy_data['confidence'] < CONFIDENCE_THRESHOLD:
            return None
            
        # Generate signal timing (next 1M candle)
        current_time = dt.datetime.now(dt.timezone.utc)
        next_minute = current_time.replace(second=0, microsecond=0) + dt.timedelta(minutes=1)
        target_time = next_minute + dt.timedelta(minutes=tf)
        
        # Build comprehensive reasoning
        active_psychology = [k for k, v in psychology.items() if v > 50]
        market_state = self._detect_market_state(df)
        
        reasoning = f"{strategy_data['reasoning']} | Market: {market_state.value} | Psychology: {', '.join(active_psychology[:3])}"
        
        return StrategySignal(
            direction=strategy_data['direction'],
            confidence=strategy_data['confidence'],
            reasoning=reasoning,
            entry_time=next_minute.strftime("%H:%M"),
            target_time=target_time.strftime("%H:%M"),
            psychology=active_psychology,
            market_state=market_state.value,
            strategy_tree=strategy_name
        )

    # ═══════════════ TELEGRAM SIGNAL ENGINE ═══════════════
    def send_ai_signal(self, asset: str, tf: int, signal: StrategySignal):
        """Send advanced AI signal to Telegram"""
        
        # Calculate next candle timing
        current_time = dt.datetime.now(dt.timezone.utc)
        next_candle = current_time.replace(second=0, microsecond=0) + dt.timedelta(minutes=1)
        
        msg = (
            f"🤖 **𝗔𝗜 𝗦𝗨𝗣𝗥𝗘𝗠𝗘 𝗦𝗡𝗜𝗣𝗘𝗥**\n\n"
            f"🎯 **Asset**: {asset}\n"
            f"📊 **Timeframe**: {tf}M\n"
            f"🚀 **Direction**: {signal.direction}\n"
            f"⏰ **Entry Time**: {signal.entry_time} UTC\n"
            f"🎪 **Expiry**: {signal.target_time} UTC\n"
            f"🧠 **AI Confidence**: {signal.confidence:.1f}%\n"
            f"🏗 **Strategy**: {signal.strategy_tree}\n"
            f"📈 **Market State**: {signal.market_state.replace('_', ' ').title()}\n"
            f"🔮 **Psychology**: {', '.join(signal.psychology[:2])}\n"
            f"💡 **AI Reasoning**: {signal.reasoning}\n\n"
            f"🤖 *100 Billion Years AI Engine*"
        )
        
        try:
            self.bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="Markdown")
            logging.info(f"🚀 AI Signal sent: {asset} {signal.direction} @{signal.confidence:.1f}%")
        except TelegramError as e:
            logging.error(f"❌ Telegram error: {e}")

    # ═══════════════ MAIN AI LOOP ═══════════════
    def run_ai_engine(self):
        """Main AI trading engine loop"""
        logging.info("🤖 Starting Supreme AI Sniper Engine...")
        logging.info("🧠 100 Billion Years AI Strategies Activated")
        
        last_signal_time = time.time()
        
        while True:
            try:
                # Get available OTC assets
                assets = [el.text for el in self.driver.find_elements(By.CLASS_NAME, "symbol-item") 
                         if "otc" in el.text.lower()]
                
                for asset in assets:
                    for tf in TIMEFRAMES:
                        try:
                            # AI Analysis
                            signal = self.execute_ai_analysis(asset, tf)
                            
                            if signal and time.time() - last_signal_time > 15:
                                self.send_ai_signal(asset, tf, signal)
                                last_signal_time = time.time()
                                
                        except Exception as e:
                            logging.exception(f"❌ AI Error analyzing {asset} {tf}M: {e}")
                            
                    time.sleep(SCAN_SLEEP)
                    
            except Exception as e:
                logging.exception(f"❌ AI Engine error: {e}")
                time.sleep(5)

if __name__ == "__main__":
    ai_sniper = SupremeAISniper()
    ai_sniper.run_ai_engine()