#!/usr/bin/env python3
"""
🌌 OMNI-BRAIN BINARY AI - Ultimate Adaptive Strategy Builder
Main application entry point
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import Config
from api.telegram_bot import TelegramTradingBot
from strategies.cosmic_ai_strategy import CosmicAIStrategy
from brokers.deriv_adapter import DerivAdapter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class BinaryTradingBot:
    """Main Binary Trading Bot Application"""
    
    def __init__(self):
        self.cosmic_ai = CosmicAIStrategy()
        self.telegram_bot = None
        self.brokers = {}
        self.is_running = False
        
        # Use the provided Telegram token
        self.telegram_token = "7604218758:AAHJj2zMDTfVwyJHpLClVCDzukNr2Psj-38"
        
    async def initialize(self):
        """Initialize all components"""
        logger.info("🌌 Initializing OMNI-BRAIN BINARY AI...")
        
        try:
            # Initialize Telegram bot with provided token
            self.telegram_bot = TelegramTradingBot(self.telegram_token)
            logger.info("✅ Telegram bot initialized")
            
            # Initialize brokers if credentials are available
            await self.initialize_brokers()
            
            # Initialize COSMIC AI strategy
            logger.info("✅ COSMIC AI Strategy Engine loaded")
            
            logger.info("🚀 OMNI-BRAIN BINARY AI ready for operation!")
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            raise
    
    async def initialize_brokers(self):
        """Initialize broker connections"""
        try:
            # Initialize Deriv if token is available
            if Config.DERIV_API_TOKEN:
                deriv_adapter = DerivAdapter(
                    api_token=Config.DERIV_API_TOKEN,
                    app_id=Config.DERIV_APP_ID
                )
                self.brokers['deriv'] = deriv_adapter
                logger.info("✅ Deriv adapter initialized")
            else:
                logger.warning("⚠️ Deriv API token not configured")
            
            # Add other brokers here when implemented
            # if Config.IQ_OPTION_EMAIL and Config.IQ_OPTION_PASSWORD:
            #     iq_adapter = IQOptionAdapter(...)
            #     self.brokers['iq_option'] = iq_adapter
            
        except Exception as e:
            logger.error(f"❌ Broker initialization failed: {e}")
    
    async def start(self):
        """Start the binary trading bot"""
        await self.initialize()
        
        logger.info("🚀 Starting OMNI-BRAIN BINARY AI...")
        self.is_running = True
        
        try:
            # Start Telegram bot
            if self.telegram_bot:
                logger.info("📱 Starting Telegram interface...")
                await self.telegram_bot.start_bot()
            else:
                logger.error("❌ Telegram bot not initialized")
                
        except KeyboardInterrupt:
            logger.info("🛑 Shutdown requested by user")
        except Exception as e:
            logger.error(f"❌ Runtime error: {e}")
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the binary trading bot"""
        logger.info("🛑 Stopping OMNI-BRAIN BINARY AI...")
        self.is_running = False
        
        # Disconnect from brokers
        for broker_name, broker in self.brokers.items():
            try:
                if hasattr(broker, 'disconnect'):
                    await broker.disconnect()
                logger.info(f"✅ Disconnected from {broker_name}")
            except Exception as e:
                logger.error(f"❌ Error disconnecting from {broker_name}: {e}")
        
        logger.info("✅ OMNI-BRAIN BINARY AI stopped")

def display_banner():
    """Display application banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║         🌌 OMNI-BRAIN BINARY AI ACTIVATED 🌌                ║
    ║                                                              ║
    ║        🧠 THE ULTIMATE ADAPTIVE STRATEGY BUILDER 🧠         ║
    ║                                                              ║
    ║  🚀 REVOLUTIONARY FEATURES:                                  ║
    ║  - 🔍 PERCEPTION ENGINE: Advanced chart analysis            ║
    ║  - 📖 CONTEXT ENGINE: Reads market stories                  ║
    ║  - 🧠 STRATEGY ENGINE: Builds unique strategies             ║
    ║                                                              ║
    ║  💫 STRATEGY TYPES:                                          ║
    ║  - Breakout Continuation                                     ║
    ║  - Reversal Play                                             ║
    ║  - Momentum Shift                                            ║
    ║  - Trap Fade                                                 ║
    ║  - Exhaustion Reversal                                       ║
    ║                                                              ║
    ║  📍 Predicts NEXT 1-minute candle direction                  ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_dependencies():
    """Check if all required dependencies are available"""
    try:
        import pandas
        import numpy  
        import cv2
        import PIL
        import telegram
        import websockets
        import talib
        logger.info("✅ All dependencies available")
        return True
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        logger.error("Please install requirements: pip install -r requirements.txt")
        return False

async def main():
    """Main application entry point"""
    display_banner()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check if .env file exists
    env_file = Path(".env")
    if not env_file.exists():
        logger.warning("⚠️ .env file not found. Copy .env.example to .env and configure your settings.")
        logger.info("💡 Creating .env file with Telegram token...")
        
        # Create basic .env file with the provided token
        with open(".env", "w") as f:
            f.write("# Binary Trading Bot Configuration\n")
            f.write("TELEGRAM_BOT_TOKEN=7604218758:AAHJj2zMDTfVwyJHpLClVCDzukNr2Psj-38\n")
            f.write("\n# Add your broker credentials here:\n")
            f.write("# DERIV_API_TOKEN=your_deriv_token\n")
            f.write("# IQ_OPTION_EMAIL=your_email\n")
            f.write("# IQ_OPTION_PASSWORD=your_password\n")
        
        logger.info("✅ .env file created with Telegram token")
    
    # Start the bot
    bot = BinaryTradingBot()
    
    try:
        await bot.start()
    except KeyboardInterrupt:
        logger.info("🛑 Application stopped by user")
    except Exception as e:
        logger.error(f"❌ Application error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Application interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)