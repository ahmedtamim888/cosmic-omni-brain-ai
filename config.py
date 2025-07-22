import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Telegram Bot Configuration
    TELEGRAM_BOT_TOKEN = "8155744845:AAGQ6s8RMSpCAjO16Mzb1xl8_otLnXh3OKQ"
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '-1001234567890')  # Replace with your group chat ID
    
    # AI Engine Configuration
    CONFIDENCE_THRESHOLD = 85.0  # Minimum confidence to send signal
    MIN_CANDLES_REQUIRED = 6     # Minimum candles needed for analysis
    MAX_CANDLES_ANALYZE = 8      # Maximum candles to analyze
    
    # Flask Configuration
    SECRET_KEY = os.getenv('SECRET_KEY', 'cosmic-ai-binary-signal-bot-2024')
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # Analysis Configuration
    CANDLE_DETECTION_SENSITIVITY = 0.8
    MOMENTUM_THRESHOLD = 0.7
    SUPPORT_RESISTANCE_TOLERANCE = 0.02
    
    # Time Configuration
    TIMEZONE_OFFSET = 6  # UTC+6
    
    @staticmethod
    def init_app(app):
        pass