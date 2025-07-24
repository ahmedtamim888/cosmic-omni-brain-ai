import os
from datetime import timezone, timedelta

class Config:
    # Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'cosmic-ai-secret-key-2024'
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # Telegram Configuration
    TELEGRAM_BOT_TOKEN = "8288385434:AAG_RVKnlXDWBZNN38Q3IEfSQXIgxwPlsU0"
    TELEGRAM_CHAT_ID = "7700105638"
    
    # AI Engine Configuration
    CONFIDENCE_THRESHOLD = 85.0  # Minimum confidence to give signal
    MIN_CANDLES_REQUIRED = 6     # Minimum candles needed for analysis
    MAX_CANDLES_ANALYZED = 8     # Maximum candles to analyze
    
    # Trading Configuration
    SIGNAL_TIMEFRAME = "1M"      # 1 minute signals
    TIMEZONE = timezone(timedelta(hours=6))  # UTC+6
    
    # Image Processing Configuration
    MIN_IMAGE_WIDTH = 300
    MIN_IMAGE_HEIGHT = 200
    SUPPORTED_FORMATS = ['jpg', 'jpeg', 'png', 'bmp']
    
    # Strategy Configuration
    STRATEGIES = {
        'BREAKOUT_CONTINUATION': {'weight': 1.2, 'min_confidence': 80},
        'REVERSAL_PLAY': {'weight': 1.1, 'min_confidence': 85},
        'MOMENTUM_SHIFT': {'weight': 1.0, 'min_confidence': 75},
        'TRAP_FADE': {'weight': 1.3, 'min_confidence': 90},
        'EXHAUSTION_REVERSAL': {'weight': 1.15, 'min_confidence': 82}
    }
    
    # Debug Configuration
    DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'
    SAVE_ANALYSIS_IMAGES = True  # Save processed images for debugging