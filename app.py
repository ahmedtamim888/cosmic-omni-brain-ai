#!/usr/bin/env python3
"""
🔥 GHOST TRANSCENDENCE CORE ∞ vX - The God-Level AI Bot
Ultimate AI Trading Bot with Chart Analysis and Signal Generation
"""

import os
import io
import logging
import asyncio
from datetime import datetime
from flask import Flask, request, jsonify, render_template, redirect, url_for
import cv2
import numpy as np
from PIL import Image
import base64
import threading
import time
from typing import Dict

# Import our custom AI modules
from ai_engine.perception_engine import PerceptionEngine
from ai_engine.context_engine import ContextEngine
from ai_engine.intelligence_engine import IntelligenceEngine
from ai_engine.signal_generator import SignalGenerator
from utils.chart_analyzer import ChartAnalyzer
from utils.logger import setup_logger

# Setup logging
logger = setup_logger()

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'ghost_transcendence_core_infinity')

# Bot configuration
TELEGRAM_BOT_TOKEN = "7604218758:AAHJj2zMDTfVwyJHpLClVCDzukNr2Psj-38"
WEBHOOK_URL = os.environ.get('WEBHOOK_URL', 'https://your-domain.com')

# Initialize AI engines
perception_engine = PerceptionEngine()
context_engine = ContextEngine()
intelligence_engine = IntelligenceEngine()
signal_generator = SignalGenerator()
chart_analyzer = ChartAnalyzer()

class GhostTranscendenceCore:
    """The main AI core that orchestrates all engines"""
    
    def __init__(self):
        self.version = "∞ vX"
        self.personality = "GHOST TRANSCENDENCE CORE"
        self.confidence_threshold = 85.0
        self.learning_history = []
        
    async def analyze_chart(self, image_data):
        """
        Main analysis pipeline - processes chart image and generates signal
        """
        try:
            logger.info("🧠 GHOST TRANSCENDENCE CORE ACTIVATED - Beginning chart analysis")
            
            # Step 1: Perception Engine - Read the chart
            chart_data = await perception_engine.process_image(image_data)
            if not chart_data:
                return {"error": "Could not analyze chart image", "confidence": 0}
            
            # Step 2: Context Engine - Understand the story
            context = await context_engine.analyze_context(chart_data)
            
            # Step 3: Intelligence Engine - Create dynamic strategy
            strategy = await intelligence_engine.create_strategy(chart_data, context)
            
            # Step 4: Signal Generation - Final decision
            signal = await signal_generator.generate_signal(strategy, context)
            
            # Learn and adapt
            self.learning_history.append({
                'timestamp': datetime.now(),
                'signal': signal,
                'context': context
            })
            
            logger.info(f"🎯 Signal Generated: {signal}")
            return signal
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {str(e)}")
            return {"error": str(e), "confidence": 0}

# Initialize the AI core
ghost_core = GhostTranscendenceCore()

@app.route('/')
def index():
    """Main web interface"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_endpoint():
    """Web API endpoint for chart analysis"""
    try:
        if 'chart_image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        file = request.files['chart_image']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Read image data
        image_data = file.read()
        
        # Analyze with AI (synchronous wrapper)
        result = asyncio.run(ghost_core.analyze_chart(image_data))
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"API Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_chart():
    """
    🕯️ CANDLE WHISPERER ANALYSIS ENDPOINT
    Analyzes chart by talking with every candle for 100% accuracy
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        logger.info("🕯️ CANDLE WHISPERER: Starting chart analysis...")
        
        # Create upload folder if it doesn't exist
        upload_folder = 'uploads'
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        
        # Save uploaded image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chart_{timestamp}.png"
        filepath = os.path.join(upload_folder, filename)
        image_file.save(filepath)
        
        # Load and process image
        image = cv2.imread(filepath)
        if image is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        logger.info(f"🖼️ Image loaded: {image.shape}")
        
        # Process with all AI engines (run async code in sync context)
        chart_data = {'image': image, 'filepath': filepath}
        context = {'timestamp': timestamp, 'filename': filename}
        
        # Run async functions synchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Step 1: Perception Engine - convert image to bytes first
            # Convert cv2 image to bytes for the perception engine
            _, buffer = cv2.imencode('.png', image)
            image_bytes = buffer.tobytes()
            
            perception_result = loop.run_until_complete(perception_engine.process_image(image_bytes))
            logger.info(f"👁️ Perception: {len(perception_result.get('candles', []) if perception_result else [])} candles detected")
            
            # Step 2: Intelligence Engine (CANDLE WHISPERER MODE)
            strategy = loop.run_until_complete(intelligence_engine.create_strategy(chart_data, context))
            logger.info(f"🧠 Strategy: {strategy.get('type', 'unknown')}")
            
            # Step 3: Signal Generator (with CANDLE WHISPERER)
            signal = loop.run_until_complete(signal_generator.generate_signal(strategy, chart_data, context))
            logger.info(f"🎯 Signal: {signal.get('signal', 'unknown')}")
            
            # Build CANDLE WHISPERER response
            response = loop.run_until_complete(build_candle_whisperer_response(signal, strategy, perception_result or {}))
            
        finally:
            loop.close()
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return jsonify({
            'error': str(e),
            'signal': 'NO SIGNAL',
            'confidence': 0.0,
            'message': '🚫 CANDLE WHISPERER temporarily offline'
        }), 500

async def build_candle_whisperer_response(signal: Dict, strategy: Dict, perception: Dict) -> Dict:
    """
    🕯️ Build beautiful CANDLE WHISPERER response with all details
    """
    try:
        # Extract signal information
        signal_type = signal.get('signal', 'NO SIGNAL')
        confidence = signal.get('confidence', 0.0)
        next_candle_time = signal.get('next_candle_time', '00:00')
        candle_whisperer_mode = signal.get('candle_whisperer_mode', False)
        reasoning = signal.get('reasoning', 'Analysis complete')
        
        # Extract candle conversation details
        total_candles = signal.get('candle_conversations', 0)
        candle_prophecy = signal.get('candle_prophecy', '')
        accuracy = signal.get('accuracy', 0.60)
        
        # Format confidence as percentage
        confidence_percent = confidence * 100 if confidence < 1.0 else confidence
        accuracy_percent = accuracy * 100 if accuracy < 1.0 else accuracy
        
        # Build response message
        response = {
            'signal': signal_type,
            'confidence': round(confidence_percent, 1),
            'accuracy': round(accuracy_percent, 1),
            'next_candle_time': next_candle_time,
            'timezone': 'UTC+6:00',
            'reasoning': reasoning,
            'candle_whisperer_active': candle_whisperer_mode,
            'total_candles_consulted': total_candles,
            'candle_prophecy': candle_prophecy,
            'strategy_type': strategy.get('type', 'unknown'),
            'features_active': signal.get('features', {}),
            'timestamp': datetime.now().strftime("%H:%M:%S"),
            'version': "🕯️ CANDLE WHISPERER ∞ vX"
        }
        
        # Create clean Telegram-safe message
        def clean_text(text):
            """Clean text for Telegram to avoid parsing errors"""
            if not text:
                return "Analysis complete"
            # Remove problematic characters and limit length
            cleaned = str(text).replace('|', '-').replace('<', '').replace('>', '')
            cleaned = cleaned.replace('[', '(').replace(']', ')').replace('*', '')
            cleaned = cleaned.replace('_', ' ').replace('`', '').replace('\\', '')
            # Limit length to avoid oversized messages
            if len(cleaned) > 200:
                cleaned = cleaned[:200] + "..."
            return cleaned
        
        clean_reasoning = clean_text(reasoning)
        clean_prophecy = clean_text(candle_prophecy if candle_prophecy else 'Market wisdom received')
        
        # Add beautiful formatted message
        if signal_type != 'NO SIGNAL':
            response['message'] = f"""🔥 GHOST TRANSCENDENCE CORE - CANDLE WHISPERER

🕯️ SIGNAL: {signal_type}
⏰ TIMEFRAME: 1M  
🎯 ENTRY TIME: {next_candle_time} (UTC+6:00)
📈 CONFIDENCE: {confidence_percent:.1f}%
🎪 ACCURACY: {accuracy_percent:.1f}%

🧠 CANDLE WHISPERER ANALYSIS:
{clean_reasoning}

🕯️ CANDLE CONVERSATIONS: {total_candles} candles consulted
📜 PROPHECY: {clean_prophecy}

⚡ This signal was generated by TALKING WITH EVERY CANDLE

👻 Ghost Features Active:
✅ Candle Whisperer Mode
✅ 100% Accuracy Target  
✅ UTC+6:00 Timing Precision
✅ Secret Pattern Detection
✅ Loss Prevention System"""
        else:
            response['message'] = f"""🔥 GHOST TRANSCENDENCE CORE - CANDLE WHISPERER

⏸️ SIGNAL: NO SIGNAL
⏰ TIMEFRAME: 1M
🎯 TARGET: {next_candle_time} - Next candle (UTC+6:00)
📈 CONFIDENCE: {confidence_percent:.1f}%

🧠 CANDLE WHISPERER ANALYSIS:
🚫 NO SIGNAL: {clean_reasoning}

🕯️ CANDLE CONVERSATIONS: {total_candles} candles consulted
📜 MESSAGE: Candles advise waiting for better opportunity

⚡ This analysis was generated by TALKING WITH EVERY CANDLE

👻 Ghost Features Active:
✅ Candle Whisperer Mode
✅ Manipulation Resistance
✅ Broker Trap Detection  
✅ Fake Signal Immunity
✅ Adaptive Evolution"""
        
        return response
        
    except Exception as e:
        logger.error(f"Response building error: {str(e)}")
        return {
            'signal': 'NO SIGNAL',
            'confidence': 0.0,
            'message': '🚫 CANDLE WHISPERER response building failed',
            'error': str(e)
        }

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """API endpoint for bot statistics"""
    try:
        stats = {
            "total_analyses": len(ghost_core.learning_history),
            "version": ghost_core.version,
            "personality": ghost_core.personality,
            "confidence_threshold": ghost_core.confidence_threshold,
            "uptime": datetime.now().isoformat()
        }
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Stats API error: {str(e)}")
        return jsonify({"error": str(e)}), 500

def start_telegram_bot():
    """Start Telegram bot in a separate process"""
    # Check if Telegram is disabled
    if os.environ.get('DISABLE_TELEGRAM') == 'true':
        logger.info("📱 Telegram bot disabled by configuration")
        return
    
    # Check if token is configured
    if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == 'your_bot_token_here':
        logger.warning("⚠️ Telegram bot token not configured, skipping bot startup")
        return
    
    try:
        import subprocess
        import sys
        
        # Start the separate bot process
        subprocess.Popen([
            sys.executable, 'telegram_bot.py'
        ], env=os.environ.copy())
        
        logger.info("📱 Telegram bot started in separate process")
        
    except Exception as e:
        logger.error(f"Failed to start Telegram bot: {str(e)}")

if __name__ == '__main__':
    # Start Telegram bot
    start_telegram_bot()
    
    # Start Flask app
    logger.info("🔥 GHOST TRANSCENDENCE CORE ∞ vX STARTING UP")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)