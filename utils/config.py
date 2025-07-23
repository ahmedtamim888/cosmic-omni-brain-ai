"""
Configuration settings for the Ultra-Accurate Trading Bot
"""

import os
from typing import Dict, Any

class Config:
    """
    Configuration class for trading bot settings
    """
    
    def __init__(self):
        # Telegram Bot Configuration
        self.TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', 'YOUR_TELEGRAM_BOT_TOKEN_HERE')
        self.TELEGRAM_CHAT_IDS = self._parse_chat_ids()
        
        # Trading Configuration
        self.CONFIDENCE_THRESHOLD = 0.95  # 95% minimum confidence
        self.GOD_MODE_THRESHOLD = 0.97    # 97% for God Mode activation
        self.MAX_SIGNALS_PER_HOUR = 10    # Limit signals to prevent spam
        self.TIMEFRAME = '1m'             # 1-minute candles
        
        # AI Engine Configuration
        self.PATTERN_LOOKBACK = 8         # Analyze last 8 candles
        self.MEMORY_RETENTION_DAYS = 30   # Remember patterns for 30 days
        self.SUPPORT_RESISTANCE_PERIODS = 100  # Periods for S/R calculation
        
        # Market Data Configuration
        self.SYMBOLS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
        self.DATA_SOURCE = 'synthetic'    # Use synthetic data for demo
        
        # Logging Configuration
        self.LOG_LEVEL = 'INFO'
        self.LOG_FILE = 'trading_bot.log'
        
        # Risk Management
        self.MAX_CONSECUTIVE_LOSSES = 3   # Stop after 3 consecutive losses
        self.COOL_DOWN_MINUTES = 15       # Cool down period after losses
        
        # Performance Tracking
        self.TRACK_PERFORMANCE = True
        self.SAVE_SIGNALS_HISTORY = True
        
    def _parse_chat_ids(self) -> list:
        """Parse Telegram chat IDs from environment variable"""
        chat_ids_str = os.getenv('TELEGRAM_CHAT_IDS', '')
        if chat_ids_str:
            try:
                return [int(chat_id.strip()) for chat_id in chat_ids_str.split(',')]
            except ValueError:
                return []
        return []
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate configuration and return status"""
        issues = []
        
        # Check Telegram configuration
        if self.TELEGRAM_TOKEN == 'YOUR_TELEGRAM_BOT_TOKEN_HERE':
            issues.append("Telegram bot token not configured")
        
        if not self.TELEGRAM_CHAT_IDS:
            issues.append("No Telegram chat IDs configured")
        
        # Check thresholds
        if self.CONFIDENCE_THRESHOLD > 1.0 or self.CONFIDENCE_THRESHOLD < 0.5:
            issues.append("Invalid confidence threshold")
            
        if self.GOD_MODE_THRESHOLD <= self.CONFIDENCE_THRESHOLD:
            issues.append("God Mode threshold should be higher than confidence threshold")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'telegram_configured': self.TELEGRAM_TOKEN != 'YOUR_TELEGRAM_BOT_TOKEN_HERE',
            'chat_ids_count': len(self.TELEGRAM_CHAT_IDS)
        }
    
    def get_telegram_settings(self) -> Dict[str, Any]:
        """Get Telegram-specific settings"""
        return {
            'token': self.TELEGRAM_TOKEN,
            'chat_ids': self.TELEGRAM_CHAT_IDS,
            'configured': self.TELEGRAM_TOKEN != 'YOUR_TELEGRAM_BOT_TOKEN_HERE'
        }
    
    def update_telegram_settings(self, token: str = None, chat_ids: list = None):
        """Update Telegram settings"""
        if token:
            self.TELEGRAM_TOKEN = token
        if chat_ids:
            self.TELEGRAM_CHAT_IDS = chat_ids