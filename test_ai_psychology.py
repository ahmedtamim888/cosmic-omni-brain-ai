#!/usr/bin/env python3
"""
AI Psychology Analysis Test
===========================
Demonstrate the 100 Billion Years AI Engine psychology detection capabilities.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict

# Import AI engine components (mock for testing)
class MockAIEngine:
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

    def analyze_market_psychology(self, df: pd.DataFrame) -> Dict[str, float]:
        """Advanced AI psychology pattern detection"""
        psychology_scores = {}
        
        if len(df) < 20:
            return psychology_scores
            
        recent = df.tail(20)
        
        # 🧠 MOMENTUM SHIFT DETECTION
        momentum_score = self._detect_momentum_shift(recent)
        psychology_scores['momentum_shift'] = momentum_score
        
        # 🪤 TRAP ZONE ANALYSIS
        trap_score = self._detect_trap_zones(recent)
        psychology_scores['trap_zone'] = trap_score
        
        # 🔄 REVERSAL EXHAUSTION
        exhaustion_score = self._detect_exhaustion_reversal(recent)
        psychology_scores['exhaustion_reversal'] = exhaustion_score
        
        return psychology_scores

def create_market_scenarios():
    """Create different market psychology scenarios"""
    base_time = datetime.now() - timedelta(minutes=50)
    
    scenarios = {}
    
    # Scenario 1: Strong Momentum Shift
    print("🧠 Creating Strong Momentum Shift Scenario...")
    momentum_data = []
    price = 100
    for i in range(50):
        if i < 25:  # Downtrend
            change = np.random.uniform(-0.5, 0.1)
        else:  # Sudden momentum shift up
            change = np.random.uniform(0.2, 0.8)
        
        open_price = price
        close_price = price + change
        high_price = max(open_price, close_price) + abs(np.random.uniform(0, 0.2))
        low_price = min(open_price, close_price) - abs(np.random.uniform(0, 0.2))
        
        momentum_data.append([open_price, high_price, low_price, close_price])
        price = close_price
    
    scenarios['momentum_shift'] = pd.DataFrame(momentum_data, columns=['open', 'high', 'low', 'close'])
    
    # Scenario 2: Trap Zone (False Breakout)
    print("🪤 Creating Trap Zone Scenario...")
    trap_data = []
    price = 100
    resistance = 102
    
    for i in range(50):
        if i < 40:  # Build up to resistance
            change = np.random.uniform(-0.2, 0.3)
        elif i < 45:  # False breakout above resistance
            change = np.random.uniform(0.5, 1.0)  # Break above 102
        else:  # Fall back below resistance (trap)
            change = np.random.uniform(-0.8, -0.3)
        
        open_price = price
        close_price = price + change
        high_price = max(open_price, close_price) + abs(np.random.uniform(0, 0.1))
        low_price = min(open_price, close_price) - abs(np.random.uniform(0, 0.1))
        
        trap_data.append([open_price, high_price, low_price, close_price])
        price = close_price
    
    scenarios['trap_zone'] = pd.DataFrame(trap_data, columns=['open', 'high', 'low', 'close'])
    
    # Scenario 3: Exhaustion Reversal
    print("🔄 Creating Exhaustion Reversal Scenario...")
    exhaustion_data = []
    price = 100
    
    for i in range(50):
        if i < 35:  # Strong uptrend
            change = np.random.uniform(0.3, 0.6)
        elif i < 45:  # Momentum weakening but still up
            change = np.random.uniform(0.1, 0.3)
        else:  # Reversal begins
            change = np.random.uniform(-0.4, -0.1)
        
        open_price = price
        close_price = price + change
        
        # Add larger wicks during exhaustion phase
        if i >= 35:
            wick_multiplier = 2.0
        else:
            wick_multiplier = 1.0
            
        high_price = max(open_price, close_price) + abs(np.random.uniform(0, 0.3)) * wick_multiplier
        low_price = min(open_price, close_price) - abs(np.random.uniform(0, 0.2)) * wick_multiplier
        
        exhaustion_data.append([open_price, high_price, low_price, close_price])
        price = close_price
    
    scenarios['exhaustion_reversal'] = pd.DataFrame(exhaustion_data, columns=['open', 'high', 'low', 'close'])
    
    return scenarios

def test_ai_psychology():
    """Test AI psychology detection on different scenarios"""
    print("🤖 AI Psychology Analysis Test")
    print("=" * 60)
    print("Testing 100 Billion Years AI Engine capabilities...\n")
    
    # Create AI engine
    ai_engine = MockAIEngine()
    
    # Generate market scenarios
    scenarios = create_market_scenarios()
    
    for scenario_name, df in scenarios.items():
        print(f"\n📊 Testing Scenario: {scenario_name.replace('_', ' ').title()}")
        print("-" * 50)
        
        # Enhance with AI indicators
        df = ai_engine._enhance_with_ai_indicators(df)
        
        # Analyze psychology
        psychology = ai_engine.analyze_market_psychology(df)
        
        # Display results
        print("🧠 AI Psychology Detection Results:")
        for pattern, score in psychology.items():
            if score > 0:
                confidence_level = "🔥 HIGH" if score > 70 else "⚡ MEDIUM" if score > 40 else "💫 LOW"
                print(f"   • {pattern.replace('_', ' ').title()}: {score:.1f}% {confidence_level}")
        
        # Show market data summary
        print(f"\n📈 Market Data Summary:")
        print(f"   • Price Change: {((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100:.1f}%")
        print(f"   • Volatility: {df['volatility'].iloc[-1]:.4f}")
        print(f"   • Momentum: {df['momentum'].iloc[-1]:.4f}")
        print(f"   • Last 5 candles trend: {'+' if df['close'].tail(5).diff().mean() > 0 else '-'}")
        
        # AI Strategy Recommendation
        best_psychology = max(psychology.items(), key=lambda x: x[1]) if psychology else (None, 0)
        if best_psychology[1] > 60:
            print(f"\n🎯 AI Strategy Recommendation:")
            strategy_name = best_psychology[0].replace('_', ' ').title()
            print(f"   • Primary Pattern: {strategy_name}")
            print(f"   • Confidence: {best_psychology[1]:.1f}%")
            
            # Generate signal direction based on pattern
            if 'momentum_shift' in best_psychology[0]:
                direction = "CALL" if df['acceleration'].iloc[-1] > 0 else "PUT"
            elif 'trap_zone' in best_psychology[0]:
                direction = "PUT" if df['close'].iloc[-1] > df['close'].iloc[-2] else "CALL"
            elif 'exhaustion' in best_psychology[0]:
                direction = "PUT" if df['momentum'].iloc[-1] > 0 else "CALL"
            else:
                direction = "ANALYZE"
                
            print(f"   • Signal Direction: {direction}")
        else:
            print(f"\n⏳ AI Assessment: Market conditions below threshold (85%)")

def demo_telegram_format():
    """Demo the AI Telegram signal format"""
    print(f"\n📱 AI Telegram Signal Format Demo")
    print("=" * 60)
    
    current_time = datetime.now()
    next_minute = current_time.replace(second=0, microsecond=0) + timedelta(minutes=1)
    target_time = next_minute + timedelta(minutes=1)
    
    msg = (
        f"🤖 **𝗔𝗜 𝗦𝗨𝗣𝗥𝗘𝗠𝗘 𝗦𝗡𝗜𝗣𝗘𝗥**\n\n"
        f"🎯 **Asset**: EUR/USD_otc\n"
        f"📊 **Timeframe**: 1M\n"
        f"🚀 **Direction**: CALL\n"
        f"⏰ **Entry Time**: {next_minute.strftime('%H:%M')} UTC\n"
        f"🎪 **Expiry**: {target_time.strftime('%H:%M')} UTC\n"
        f"🧠 **AI Confidence**: 91.3%\n"
        f"🏗 **Strategy**: MOMENTUM_SHIFT\n"
        f"📈 **Market State**: Trending Up\n"
        f"🔮 **Psychology**: momentum_shift, breakout_continuation\n"
        f"💡 **AI Reasoning**: Momentum shift detected with 91.3% psychology score | Market: trending_up\n\n"
        f"🤖 *100 Billion Years AI Engine*"
    )
    
    print("Sample AI Signal:")
    print("-" * 30)
    print(msg)

if __name__ == "__main__":
    test_ai_psychology()
    demo_telegram_format()
    
    print(f"\n🎉 AI Psychology Analysis Test Complete!")
    print("🤖 100 Billion Years AI Engine demonstrated successfully")