"""
Telegram Bot Integration
Sends trading signals and chart screenshots to Telegram
"""

import asyncio
import io
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
import aiohttp
import json
from telegram import Bot
from telegram.error import TelegramError
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import pandas as pd
import numpy as np

class TelegramBot:
    """
    Advanced Telegram bot for trading signals
    Sends formatted signals with charts and analysis
    """
    
    def __init__(self, token: str):
        self.logger = logging.getLogger(__name__)
        self.token = token
        self.bot = Bot(token=token)
        self.chat_ids = []  # Will be populated when users start the bot
        self.signal_history = []
        
        # Chart settings
        plt.style.use('dark_background')
        
    async def send_signal(self, signal: Dict[str, Any], chart_data: Optional[Dict] = None):
        """Send trading signal to Telegram with optional chart"""
        try:
            # Format signal message
            message = self._format_signal_message(signal)
            
            # Generate chart if data provided
            chart_image = None
            if chart_data:
                chart_image = await self._generate_chart(signal, chart_data)
            
            # Send to all registered chats
            if not self.chat_ids:
                # For demo purposes, use a default chat ID or log
                self.logger.info(f"📱 TELEGRAM SIGNAL:\n{message}")
                return
            
            for chat_id in self.chat_ids:
                try:
                    if chart_image:
                        await self.bot.send_photo(
                            chat_id=chat_id,
                            photo=chart_image,
                            caption=message,
                            parse_mode='HTML'
                        )
                    else:
                        await self.bot.send_message(
                            chat_id=chat_id,
                            text=message,
                            parse_mode='HTML'
                        )
                        
                except TelegramError as e:
                    self.logger.error(f"Error sending to chat {chat_id}: {e}")
            
            # Store signal in history
            self.signal_history.append({
                'signal': signal,
                'timestamp': datetime.now(),
                'message': message
            })
            
        except Exception as e:
            self.logger.error(f"Error sending Telegram signal: {e}")
    
    def _format_signal_message(self, signal: Dict[str, Any]) -> str:
        """Format signal into a beautiful Telegram message"""
        try:
            action = signal.get('action', 'NO_TRADE')
            confidence = signal.get('confidence', 0)
            strategy = signal.get('strategy', 'UNKNOWN')
            reason = signal.get('reason', 'No reason provided')
            timestamp = signal.get('timestamp', datetime.now())
            
            # Emojis for different actions
            action_emojis = {
                'CALL': '📈 🟢',
                'PUT': '📉 🔴',
                'NO_TRADE': '⏸️ ⚪'
            }
            
            # Confidence level emojis
            confidence_emoji = self._get_confidence_emoji(confidence)
            
            # Strategy emojis
            strategy_emojis = {
                'GOD_MODE_AI': '⚡🔥',
                'TREND_FOLLOWING': '📊',
                'RANGE_TRADING': '🎯',
                'BREAKOUT_MOMENTUM': '💥',
                'VOLATILITY_EXPANSION': '🌊',
                'CONSOLIDATION_BREAKOUT': '🔓'
            }
            
            strategy_emoji = strategy_emojis.get(strategy, '🤖')
            
            # Format time
            time_str = timestamp.strftime("%H:%M:%S")
            
            # Build message
            message = f"""
{action_emojis.get(action, '🤖')} <b>TRADING SIGNAL</b> {strategy_emoji}

🕐 <b>Time:</b> {time_str}
💎 <b>Action:</b> <b>{action}</b>
{confidence_emoji} <b>Confidence:</b> <b>{confidence:.1%}</b>
🧠 <b>Strategy:</b> {strategy}

📝 <b>Analysis:</b>
{reason}
"""
            
            # Add God Mode specific information
            if strategy == 'GOD_MODE_AI':
                god_metrics = signal.get('god_mode_metrics', {})
                confluences = god_metrics.get('confluences_count', 0)
                quantum_coherence = god_metrics.get('quantum_coherence', 0)
                
                message += f"""
⚡ <b>GOD MODE ACTIVATED</b> ⚡
🔗 Confluences: {confluences}
🌌 Quantum Coherence: {quantum_coherence:.3f}
🧬 Consciousness Score: {god_metrics.get('consciousness_score', 0):.3f}
"""
            
            # Add next candle prediction
            next_candle = signal.get('next_candle_prediction', {})
            if next_candle and next_candle.get('direction') != 'UNKNOWN':
                direction = next_candle.get('direction', 'UNKNOWN')
                strength = next_candle.get('strength', 'UNKNOWN')
                
                direction_emoji = {
                    'UP': '⬆️',
                    'DOWN': '⬇️',
                    'SIDEWAYS': '➡️',
                    'LARGE_MOVE': '💥'
                }.get(direction, '❓')
                
                message += f"""
🔮 <b>Next Candle Prediction:</b>
{direction_emoji} Direction: {direction}
💪 Strength: {strength}
"""
            
            # Add volume condition if available
            volume_condition = signal.get('volume_condition', 'UNKNOWN')
            if volume_condition != 'UNKNOWN':
                volume_emoji = '📊' if volume_condition == 'rising' else '📉'
                message += f"{volume_emoji} Volume: {volume_condition}\n"
            
            # Add trend alignment
            trend_alignment = signal.get('trend_alignment', 'UNKNOWN')
            if trend_alignment != 'UNKNOWN':
                trend_emoji = '📈' if 'bullish' in trend_alignment else '📉' if 'bearish' in trend_alignment else '➡️'
                message += f"{trend_emoji} Trend: {trend_alignment}\n"
            
            message += f"""
━━━━━━━━━━━━━━━━━━━━━
🤖 <b>Ultra-Accurate Trading Bot</b>
🎯 <i>God-Tier Precision</i>
"""
            
            return message
            
        except Exception as e:
            self.logger.error(f"Error formatting signal message: {e}")
            return "Error formatting signal message"
    
    def _get_confidence_emoji(self, confidence: float) -> str:
        """Get emoji based on confidence level"""
        if confidence >= 0.97:
            return '⚡🔥'  # God Mode
        elif confidence >= 0.90:
            return '💎'    # Diamond hands
        elif confidence >= 0.80:
            return '🔥'    # Fire
        elif confidence >= 0.70:
            return '✨'    # Sparkles
        elif confidence >= 0.60:
            return '⭐'    # Star
        else:
            return '⚠️'    # Warning
    
    async def _generate_chart(self, signal: Dict, chart_data: Dict) -> io.BytesIO:
        """Generate chart screenshot for the signal"""
        try:
            candles = chart_data.get('candles', [])
            if not candles:
                return None
            
            # Convert to DataFrame for plotting
            df = pd.DataFrame(candles)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Create figure
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), 
                                         gridspec_kw={'height_ratios': [3, 1]})
            
            # Plot candlestick chart
            await self._plot_candlesticks(ax1, df)
            
            # Plot volume
            await self._plot_volume(ax2, df)
            
            # Add signal indicator
            await self._add_signal_indicator(ax1, signal, df)
            
            # Customize chart
            await self._customize_chart(fig, ax1, ax2, signal)
            
            # Save to BytesIO
            chart_buffer = io.BytesIO()
            plt.savefig(chart_buffer, format='png', dpi=150, bbox_inches='tight',
                       facecolor='#1a1a1a', edgecolor='none')
            chart_buffer.seek(0)
            
            plt.close(fig)
            
            return chart_buffer
            
        except Exception as e:
            self.logger.error(f"Error generating chart: {e}")
            return None
    
    async def _plot_candlesticks(self, ax, df):
        """Plot candlestick chart"""
        try:
            # Plot candlesticks
            for i, (timestamp, row) in enumerate(df.iterrows()):
                open_price = row['open']
                high_price = row['high']
                low_price = row['low']
                close_price = row['close']
                
                # Determine color
                color = '#00ff88' if close_price >= open_price else '#ff4444'
                
                # Draw the wick
                ax.plot([i, i], [low_price, high_price], color='white', linewidth=1, alpha=0.8)
                
                # Draw the body
                body_height = abs(close_price - open_price)
                body_bottom = min(open_price, close_price)
                
                rect = Rectangle((i - 0.3, body_bottom), 0.6, body_height,
                               facecolor=color, edgecolor='white', linewidth=0.5)
                ax.add_patch(rect)
            
            # Customize axes
            ax.set_xlim(-1, len(df))
            ax.set_ylabel('Price', color='white', fontsize=12)
            ax.tick_params(colors='white')
            ax.grid(True, alpha=0.3)
            
            # Format x-axis
            time_labels = [ts.strftime('%H:%M') for ts in df.index[::5]]
            ax.set_xticks(range(0, len(df), 5))
            ax.set_xticklabels(time_labels, rotation=45)
            
        except Exception as e:
            self.logger.error(f"Error plotting candlesticks: {e}")
    
    async def _plot_volume(self, ax, df):
        """Plot volume bars"""
        try:
            volumes = df['volume'].values
            colors = ['#00ff88' if df.iloc[i]['close'] >= df.iloc[i]['open'] 
                     else '#ff4444' for i in range(len(df))]
            
            ax.bar(range(len(volumes)), volumes, color=colors, alpha=0.7)
            ax.set_ylabel('Volume', color='white', fontsize=10)
            ax.tick_params(colors='white')
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            self.logger.error(f"Error plotting volume: {e}")
    
    async def _add_signal_indicator(self, ax, signal: Dict, df):
        """Add signal indicator to chart"""
        try:
            action = signal.get('action', 'NO_TRADE')
            if action == 'NO_TRADE':
                return
            
            # Add signal arrow at the last candle
            last_index = len(df) - 1
            last_price = df.iloc[-1]['close']
            
            if action == 'CALL':
                ax.annotate('📈 CALL', xy=(last_index, last_price),
                           xytext=(last_index - 5, last_price + (df['high'].max() - df['low'].min()) * 0.1),
                           arrowprops=dict(arrowstyle='->', color='#00ff88', lw=2),
                           fontsize=12, color='#00ff88', fontweight='bold')
            elif action == 'PUT':
                ax.annotate('📉 PUT', xy=(last_index, last_price),
                           xytext=(last_index - 5, last_price - (df['high'].max() - df['low'].min()) * 0.1),
                           arrowprops=dict(arrowstyle='->', color='#ff4444', lw=2),
                           fontsize=12, color='#ff4444', fontweight='bold')
            
        except Exception as e:
            self.logger.error(f"Error adding signal indicator: {e}")
    
    async def _customize_chart(self, fig, ax1, ax2, signal: Dict):
        """Customize chart appearance"""
        try:
            # Set background colors
            fig.patch.set_facecolor('#1a1a1a')
            ax1.set_facecolor('#2a2a2a')
            ax2.set_facecolor('#2a2a2a')
            
            # Add title
            strategy = signal.get('strategy', 'UNKNOWN')
            confidence = signal.get('confidence', 0)
            title = f"{strategy} - Confidence: {confidence:.1%}"
            
            if strategy == 'GOD_MODE_AI':
                title = f"⚡ {title} ⚡"
            
            fig.suptitle(title, color='white', fontsize=14, fontweight='bold')
            
            # Remove x-axis labels from top chart
            ax1.set_xticklabels([])
            
            # Adjust layout
            plt.tight_layout()
            
        except Exception as e:
            self.logger.error(f"Error customizing chart: {e}")
    
    async def send_statistics(self, stats: Dict[str, Any]):
        """Send bot statistics to Telegram"""
        try:
            message = self._format_statistics_message(stats)
            
            for chat_id in self.chat_ids:
                try:
                    await self.bot.send_message(
                        chat_id=chat_id,
                        text=message,
                        parse_mode='HTML'
                    )
                except TelegramError as e:
                    self.logger.error(f"Error sending statistics to chat {chat_id}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error sending statistics: {e}")
    
    def _format_statistics_message(self, stats: Dict[str, Any]) -> str:
        """Format statistics into a Telegram message"""
        try:
            message = """
📊 <b>TRADING BOT STATISTICS</b> 📊

🎯 <b>Performance:</b>
• Total Signals: {total_signals}
• High Confidence Signals: {high_conf_signals}
• Success Rate: {success_rate:.1%}

🧠 <b>AI Engine Status:</b>
• Pattern Recognition: {pattern_status}
• Memory Engine: {memory_status}
• God Mode Activations: {god_mode_count}

⚡ <b>Recent Activity (24h):</b>
• Signals Generated: {recent_signals}
• Zones Avoided: {zones_avoided}
• Confidence Average: {avg_confidence:.1%}

🔥 <b>System Status:</b> OPERATIONAL
━━━━━━━━━━━━━━━━━━━━━
🤖 Ultra-Accurate Trading Bot
""".format(
                total_signals=stats.get('total_signals', 0),
                high_conf_signals=stats.get('high_confidence_signals', 0),
                success_rate=stats.get('success_rate', 0),
                pattern_status='✅ Active' if stats.get('pattern_engine_active', True) else '❌ Inactive',
                memory_status='✅ Learning' if stats.get('memory_engine_active', True) else '❌ Inactive',
                god_mode_count=stats.get('god_mode_activations', 0),
                recent_signals=stats.get('recent_signals_24h', 0),
                zones_avoided=stats.get('zones_avoided', 0),
                avg_confidence=stats.get('average_confidence', 0)
            )
            
            return message
            
        except Exception as e:
            self.logger.error(f"Error formatting statistics message: {e}")
            return "Error formatting statistics"
    
    async def add_chat_id(self, chat_id: int):
        """Add a new chat ID for receiving signals"""
        if chat_id not in self.chat_ids:
            self.chat_ids.append(chat_id)
            self.logger.info(f"Added new chat ID: {chat_id}")
    
    async def remove_chat_id(self, chat_id: int):
        """Remove a chat ID from receiving signals"""
        if chat_id in self.chat_ids:
            self.chat_ids.remove(chat_id)
            self.logger.info(f"Removed chat ID: {chat_id}")
    
    async def test_connection(self) -> bool:
        """Test Telegram bot connection"""
        try:
            bot_info = await self.bot.get_me()
            self.logger.info(f"Telegram bot connected: {bot_info.username}")
            return True
            
        except TelegramError as e:
            self.logger.error(f"Telegram connection test failed: {e}")
            return False
    
    def get_signal_history(self) -> List[Dict]:
        """Get recent signal history"""
        return self.signal_history[-20:]  # Return last 20 signals
    
    async def send_startup_message(self):
        """Send startup message to all chats"""
        try:
            message = """
🚀 <b>ULTRA-ACCURATE TRADING BOT STARTED</b> 🚀

⚡ God Mode AI: ACTIVATED
🧠 Pattern Recognition: ONLINE
🎯 Confidence Threshold: 95%
🔥 Ready for God-Tier Trading

━━━━━━━━━━━━━━━━━━━━━
🤖 <i>No lag, no doubt, no mercy</i>
"""
            
            for chat_id in self.chat_ids:
                try:
                    await self.bot.send_message(
                        chat_id=chat_id,
                        text=message,
                        parse_mode='HTML'
                    )
                except TelegramError as e:
                    self.logger.error(f"Error sending startup message to chat {chat_id}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error sending startup message: {e}")