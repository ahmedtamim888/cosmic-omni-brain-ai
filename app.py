import os
import uuid
from datetime import datetime
from flask import Flask, request, render_template, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename
from PIL import Image
import logging

from config import Config
from logic.ai_engine import CosmicAIEngine, MarketSignal
from telegram_bot import send_signal_to_telegram

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Ensure upload directory exists
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)

# Initialize AI Engine
ai_engine = CosmicAIEngine()

def allowed_file(filename):
    """Check if uploaded file has allowed extension"""
    return ('.' in filename and 
            filename.rsplit('.', 1)[1].lower() in Config.SUPPORTED_FORMATS)

def validate_image(file_path):
    """Validate uploaded image meets requirements"""
    try:
        with Image.open(file_path) as img:
            width, height = img.size
            
            if width < Config.MIN_IMAGE_WIDTH or height < Config.MIN_IMAGE_HEIGHT:
                return False, f"Image too small. Minimum size: {Config.MIN_IMAGE_WIDTH}x{Config.MIN_IMAGE_HEIGHT}"
            
            return True, "Valid image"
            
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"

@app.route('/')
def index():
    """Main upload page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_chart():
    """Handle chart upload and analysis"""
    try:
        # Check if file was uploaded
        if 'chart_file' not in request.files:
            return jsonify({
                'error': 'No file uploaded',
                'success': False
            }), 400
        
        file = request.files['chart_file']
        
        # Check if file was selected
        if file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'success': False
            }), 400
        
        # Check file type
        if not allowed_file(file.filename):
            return jsonify({
                'error': f'Invalid file type. Supported formats: {", ".join(Config.SUPPORTED_FORMATS)}',
                'success': False
            }), 400
        
        # Generate unique filename
        unique_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        file_extension = filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{unique_id}.{file_extension}"
        file_path = os.path.join(Config.UPLOAD_FOLDER, unique_filename)
        
        # Save file
        file.save(file_path)
        logger.info(f"File saved: {file_path}")
        
        # Validate image
        is_valid, validation_message = validate_image(file_path)
        if not is_valid:
            os.remove(file_path)  # Clean up invalid file
            return jsonify({
                'error': validation_message,
                'success': False
            }), 400
        
        # Analyze the chart
        logger.info("Starting chart analysis...")
        signal = ai_engine.analyze_chart(file_path)
        logger.info(f"Analysis complete. Signal: {signal.signal}, Confidence: {signal.confidence:.1f}%")
        
        # Prepare response data
        response_data = {
            'success': True,
            'analysis_id': unique_id,
            'signal': {
                'signal': signal.signal,
                'confidence': round(signal.confidence, 1),
                'reasoning': signal.reasoning,
                'strategy': signal.strategy,
                'market_psychology': signal.market_psychology,
                'entry_time': signal.entry_time.strftime('%H:%M:%S'),
                'timeframe': signal.timeframe
            },
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'filename': filename
        }
        
        # Send to Telegram if signal is actionable
        if signal.signal in ['CALL', 'PUT'] and signal.confidence >= Config.CONFIDENCE_THRESHOLD:
            try:
                telegram_sent = send_signal_to_telegram(signal)
                response_data['telegram_sent'] = telegram_sent
                if telegram_sent:
                    logger.info("Signal sent to Telegram successfully")
                else:
                    logger.warning("Failed to send signal to Telegram")
            except Exception as e:
                logger.error(f"Error sending to Telegram: {e}")
                response_data['telegram_sent'] = False
                response_data['telegram_error'] = str(e)
        else:
            response_data['telegram_sent'] = False
            response_data['telegram_reason'] = "Signal not actionable or below confidence threshold"
        
        # Clean up uploaded file (optional - keep for debugging if enabled)
        if not Config.SAVE_ANALYSIS_IMAGES:
            try:
                os.remove(file_path)
            except:
                pass
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error processing upload: {e}")
        return jsonify({
            'error': f'Internal server error: {str(e)}',
            'success': False
        }), 500

@app.route('/test-telegram', methods=['POST'])
def test_telegram():
    """Test Telegram bot configuration"""
    try:
        from telegram_bot import TelegramBot
        
        bot = TelegramBot()
        status = bot.validate_configuration()
        
        if status['bot_token_valid'] and status['chat_id_configured']:
            # Send test message
            test_sent = bot.send_test_message()
            return jsonify({
                'success': True,
                'test_sent': test_sent,
                'status': status,
                'message': 'Test message sent successfully' if test_sent else 'Failed to send test message'
            })
        else:
            return jsonify({
                'success': False,
                'status': status,
                'message': 'Telegram bot not properly configured'
            })
            
    except Exception as e:
        logger.error(f"Error testing Telegram: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'ai_engine': 'initialized',
        'upload_folder': os.path.exists(Config.UPLOAD_FOLDER)
    })

@app.route('/api/status')
def api_status():
    """API status and configuration info"""
    from telegram_bot import TelegramBot
    
    try:
        # Check Telegram bot status
        bot = TelegramBot()
        telegram_status = bot.validate_configuration()
        
        return jsonify({
            'api_version': '1.0',
            'ai_engine': {
                'initialized': True,
                'confidence_threshold': Config.CONFIDENCE_THRESHOLD,
                'min_candles_required': Config.MIN_CANDLES_REQUIRED,
                'max_candles_analyzed': Config.MAX_CANDLES_ANALYZED
            },
            'telegram': {
                'configured': telegram_status['bot_token_valid'] and telegram_status['chat_id_configured'],
                'bot_working': telegram_status['connection_working'],
                'bot_info': telegram_status.get('bot_info', {})
            },
            'upload': {
                'max_file_size': Config.MAX_CONTENT_LENGTH,
                'supported_formats': Config.SUPPORTED_FORMATS,
                'min_image_size': f"{Config.MIN_IMAGE_WIDTH}x{Config.MIN_IMAGE_HEIGHT}"
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting API status: {e}")
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({
        'error': f'File too large. Maximum size: {Config.MAX_CONTENT_LENGTH // (1024*1024)}MB',
        'success': False
    }), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return render_template('index.html'), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle internal errors"""
    logger.error(f"Internal server error: {e}")
    return jsonify({
        'error': 'Internal server error',
        'success': False
    }), 500

# Context processor for template variables
@app.context_processor
def inject_config():
    """Inject configuration variables into templates"""
    return {
        'max_file_size_mb': Config.MAX_CONTENT_LENGTH // (1024*1024),
        'supported_formats': Config.SUPPORTED_FORMATS,
        'confidence_threshold': Config.CONFIDENCE_THRESHOLD
    }

if __name__ == '__main__':
    # Print startup information
    print("🧠 COSMIC AI Binary Signal Bot 🚀")
    print("=" * 50)
    print(f"🔧 Debug mode: {Config.DEBUG}")
    print(f"📁 Upload folder: {Config.UPLOAD_FOLDER}")
    print(f"🎯 Confidence threshold: {Config.CONFIDENCE_THRESHOLD}%")
    print(f"📊 Min candles required: {Config.MIN_CANDLES_REQUIRED}")
    print(f"⏰ Signal timeframe: {Config.SIGNAL_TIMEFRAME}")
    
    # Check Telegram configuration
    try:
        from telegram_bot import TelegramBot
        bot = TelegramBot()
        telegram_status = bot.validate_configuration()
        
        print(f"📱 Telegram bot configured: {telegram_status['bot_token_valid'] and telegram_status['chat_id_configured']}")
        if telegram_status['connection_working']:
            bot_info = telegram_status.get('bot_info', {})
            print(f"🤖 Bot name: {bot_info.get('first_name', 'Unknown')}")
            print(f"👤 Bot username: @{bot_info.get('username', 'unknown')}")
        
    except Exception as e:
        print(f"⚠️ Telegram bot error: {e}")
    
    print("=" * 50)
    print("🌐 Starting Flask server on http://localhost:5000")
    print("📈 Ready to analyze candlestick charts!")
    print("=" * 50)
    
    # Run the app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=Config.DEBUG,
        threaded=True
    )