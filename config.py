import os
from dotenv import load_dotenv
from typing import Dict, Any

load_dotenv()

class Config:
    """Configuration settings for the binary trading bot"""
    
    # API Keys and Credentials
    BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
    BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', '')
    
    BYBIT_API_KEY = os.getenv('BYBIT_API_KEY', '')
    BYBIT_SECRET_KEY = os.getenv('BYBIT_SECRET_KEY', '')
    
    DERIV_API_TOKEN = os.getenv('DERIV_API_TOKEN', '')
    DERIV_APP_ID = os.getenv('DERIV_APP_ID', '1089')
    
    IQ_OPTION_EMAIL = os.getenv('IQ_OPTION_EMAIL', '')
    IQ_OPTION_PASSWORD = os.getenv('IQ_OPTION_PASSWORD', '')
    
    QUOTEX_EMAIL = os.getenv('QUOTEX_EMAIL', '')
    QUOTEX_PASSWORD = os.getenv('QUOTEX_PASSWORD', '')
    
    # Telegram Bot Settings
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
    
    # Database Settings
    DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://user:password@localhost/trading_bot')
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
    
    # Trading Settings
    DEFAULT_TRADE_AMOUNT = float(os.getenv('DEFAULT_TRADE_AMOUNT', '10.0'))
    MAX_DAILY_LOSS = float(os.getenv('MAX_DAILY_LOSS', '100.0'))
    MAX_CONSECUTIVE_LOSSES = int(os.getenv('MAX_CONSECUTIVE_LOSSES', '3'))
    
    # Risk Management
    RISK_PERCENTAGE = float(os.getenv('RISK_PERCENTAGE', '2.0'))  # Percentage of balance per trade
    STOP_LOSS_PERCENTAGE = float(os.getenv('STOP_LOSS_PERCENTAGE', '5.0'))
    TAKE_PROFIT_PERCENTAGE = float(os.getenv('TAKE_PROFIT_PERCENTAGE', '10.0'))
    
    # Technical Analysis Settings
    RSI_PERIOD = int(os.getenv('RSI_PERIOD', '14'))
    RSI_OVERSOLD = float(os.getenv('RSI_OVERSOLD', '30'))
    RSI_OVERBOUGHT = float(os.getenv('RSI_OVERBOUGHT', '70'))
    
    MACD_FAST = int(os.getenv('MACD_FAST', '12'))
    MACD_SLOW = int(os.getenv('MACD_SLOW', '26'))
    MACD_SIGNAL = int(os.getenv('MACD_SIGNAL', '9'))
    
    # Bollinger Bands
    BB_PERIOD = int(os.getenv('BB_PERIOD', '20'))
    BB_STD = float(os.getenv('BB_STD', '2.0'))
    
    # Moving Averages
    SMA_SHORT = int(os.getenv('SMA_SHORT', '10'))
    SMA_LONG = int(os.getenv('SMA_LONG', '50'))
    EMA_SHORT = int(os.getenv('EMA_SHORT', '12'))
    EMA_LONG = int(os.getenv('EMA_LONG', '26'))
    
    # WebSocket Settings
    WS_HEARTBEAT_INTERVAL = int(os.getenv('WS_HEARTBEAT_INTERVAL', '30'))
    WS_RECONNECT_INTERVAL = int(os.getenv('WS_RECONNECT_INTERVAL', '5'))
    
    # Supported Brokers
    SUPPORTED_BROKERS = {
        'deriv': {
            'name': 'Deriv',
            'ws_url': 'wss://ws.binaryws.com/websockets/v3',
            'api_url': 'https://api.deriv.com',
            'supports_otc': True,
            'min_trade_amount': 0.35,
            'max_trade_amount': 50000,
            'currencies': ['USD', 'EUR', 'GBP', 'BTC'],
            'assets': [
                'frxEURUSD', 'frxGBPUSD', 'frxUSDJPY', 'frxAUDUSD',
                'frxUSDCAD', 'frxUSDCHF', 'frxNZDUSD', 'frxEURGBP',
                'cryBTCUSD', 'cryETHUSD', 'cryLTCUSD', 'cryXRPUSD'
            ]
        },
        'iq_option': {
            'name': 'IQ Option',
            'api_url': 'https://iqoption.com/api',
            'supports_otc': True,
            'min_trade_amount': 1,
            'max_trade_amount': 20000,
            'currencies': ['USD', 'EUR', 'GBP'],
            'assets': [
                'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD',
                'USDCAD', 'USDCHF', 'NZDUSD', 'EURGBP',
                'BTCUSD', 'ETHUSD', 'LTCUSD', 'XRPUSD'
            ]
        },
        'quotex': {
            'name': 'Quotex',
            'api_url': 'https://qxbroker.com/api',
            'supports_otc': True,
            'min_trade_amount': 1,
            'max_trade_amount': 10000,
            'currencies': ['USD', 'EUR'],
            'assets': [
                'EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD',
                'USD/CAD', 'USD/CHF', 'NZD/USD', 'EUR/GBP',
                'BTC/USD', 'ETH/USD', 'LTC/USD', 'XRP/USD'
            ]
        },
        'binance': {
            'name': 'Binance',
            'api_url': 'https://api.binance.com',
            'supports_otc': False,
            'min_trade_amount': 10,
            'max_trade_amount': 1000000,
            'currencies': ['USDT', 'BTC', 'ETH'],
            'assets': [
                'BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT',
                'LINKUSDT', 'LTCUSDT', 'BCHUSDT', 'XLMUSDT'
            ]
        }
    }
    
    # Trading Timeframes
    TIMEFRAMES = {
        '1m': 60,
        '5m': 300,
        '15m': 900,
        '30m': 1800,
        '1h': 3600,
        '4h': 14400,
        '1d': 86400
    }
    
    # Binary Options Expiry Times (in seconds)
    BINARY_EXPIRY_TIMES = [60, 300, 900, 1800, 3600]  # 1m, 5m, 15m, 30m, 1h
    
    @classmethod
    def get_broker_config(cls, broker_name: str) -> Dict[str, Any]:
        """Get configuration for a specific broker"""
        return cls.SUPPORTED_BROKERS.get(broker_name, {})
    
    @classmethod
    def is_broker_supported(cls, broker_name: str) -> bool:
        """Check if a broker is supported"""
        return broker_name in cls.SUPPORTED_BROKERS
    
    @classmethod
    def get_otc_brokers(cls) -> list:
        """Get list of brokers that support OTC trading"""
        return [
            broker for broker, config in cls.SUPPORTED_BROKERS.items()
            if config.get('supports_otc', False)
        ]