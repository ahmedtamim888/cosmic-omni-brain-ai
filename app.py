import os
import io
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import logging
from datetime import datetime

# Import our custom modules
from config import Config
from logic.ai_engine import CosmicAIEngine
from telegram_bot import telegram_bot

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_app():
    """Create and configure the Flask application"""
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # Initialize config
    Config.init_app(app)
    
    # Create upload directory if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Initialize AI Engine
    ai_engine = CosmicAIEngine()
    
    return app, ai_engine

# Create the app and AI engine
app, ai_engine = create_app()

@app.route('/')
def index():
    """Main page with upload interface"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_chart():
    """Analyze uploaded chart and return trading signal"""
    try:
        # Check if file is present
        if 'chart' not in request.files:
            return jsonify({
                'error': 'No file uploaded',
                'signal': 'NO TRADE',
                'reason': 'No chart image provided',
                'confidence': 0.0,
                'timestamp': datetime.now().isoformat()
            }), 400
        
        file = request.files['chart']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'signal': 'NO TRADE',
                'reason': 'No chart image selected',
                'confidence': 0.0,
                'timestamp': datetime.now().isoformat()
            }), 400
        
        # Validate file type
        if not _allowed_file(file.filename):
            return jsonify({
                'error': 'Invalid file type',
                'signal': 'NO TRADE',
                'reason': 'Only JPG, PNG, and WEBP images are allowed',
                'confidence': 0.0,
                'timestamp': datetime.now().isoformat()
            }), 400
        
        # Read file data
        file_data = file.read()
        
        # Validate file size
        if len(file_data) > app.config['MAX_CONTENT_LENGTH']:
            return jsonify({
                'error': 'File too large',
                'signal': 'NO TRADE',
                'reason': 'File size exceeds 16MB limit',
                'confidence': 0.0,
                'timestamp': datetime.now().isoformat()
            }), 413
        
        logger.info(f"Analyzing chart: {file.filename}, Size: {len(file_data)} bytes")
        
        # Analyze the chart using AI engine
        analysis_result = ai_engine.analyze_chart(file_data)
        
        logger.info(f"Analysis completed: {analysis_result['signal']} with {analysis_result['confidence']}% confidence")
        
        # Send to Telegram if confidence is high enough
        if (analysis_result['confidence'] >= Config.CONFIDENCE_THRESHOLD and 
            analysis_result['signal'] != 'NO TRADE'):
            
            try:
                telegram_success = telegram_bot.send_signal(analysis_result)
                analysis_result['telegram_sent'] = telegram_success
                
                if telegram_success:
                    logger.info("Signal sent to Telegram successfully")
                else:
                    logger.warning("Failed to send signal to Telegram")
                    
            except Exception as telegram_error:
                logger.error(f"Telegram error: {str(telegram_error)}")
                analysis_result['telegram_sent'] = False
                analysis_result['telegram_error'] = str(telegram_error)
        else:
            analysis_result['telegram_sent'] = False
            analysis_result['telegram_reason'] = 'Confidence below threshold or NO TRADE signal'
        
        return jsonify(analysis_result)
        
    except RequestEntityTooLarge:
        logger.error("File too large error")
        return jsonify({
            'error': 'File too large',
            'signal': 'NO TRADE',
            'reason': 'File size exceeds 16MB limit',
            'confidence': 0.0,
            'timestamp': datetime.now().isoformat()
        }), 413
        
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}", exc_info=True)
        
        # Send error notification to Telegram
        try:
            telegram_bot.send_error_notification(str(e))
        except:
            pass  # Don't fail if Telegram notification fails
        
        return jsonify({
            'error': 'Analysis failed',
            'signal': 'NO TRADE',
            'reason': f'Internal error: {str(e)}',
            'confidence': 0.0,
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    try:
        # Test AI engine
        ai_status = "OK"
        
        # Test Telegram connection
        telegram_status = "OK" if telegram_bot.test_connection() else "ERROR"
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'components': {
                'ai_engine': ai_status,
                'telegram_bot': telegram_status
            },
            'config': {
                'confidence_threshold': Config.CONFIDENCE_THRESHOLD,
                'max_file_size': f"{Config.MAX_CONTENT_LENGTH // (1024*1024)}MB",
                'timezone_offset': f"UTC+{Config.TIMEZONE_OFFSET}"
            }
        })
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/stats')
def get_stats():
    """Get application statistics"""
    try:
        # Basic stats - in a real application, you'd store these in a database
        stats = {
            'total_analyses': 0,  # Would be tracked in database
            'signals_sent': 0,    # Would be tracked in database
            'uptime': datetime.now().isoformat(),
            'version': '1.0.0',
            'ai_engine': 'COSMIC AI v2.0',
            'supported_formats': ['JPG', 'PNG', 'WEBP'],
            'max_file_size': f"{Config.MAX_CONTENT_LENGTH // (1024*1024)}MB"
        }
        
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Stats error: {str(e)}")
        return jsonify({'error': str(e)}), 500

def _allowed_file(filename):
    """Check if uploaded file has allowed extension"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
    return ('.' in filename and 
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS)

@app.errorhandler(413)
def handle_file_too_large(e):
    """Handle file too large error"""
    return jsonify({
        'error': 'File too large',
        'signal': 'NO TRADE',
        'reason': 'File size exceeds 16MB limit',
        'confidence': 0.0,
        'timestamp': datetime.now().isoformat()
    }), 413

@app.errorhandler(404)
def handle_not_found(e):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'The requested endpoint does not exist'
    }), 404

@app.errorhandler(500)
def handle_internal_error(e):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({
        'error': 'Internal server error',
        'signal': 'NO TRADE',
        'reason': 'Server encountered an error',
        'confidence': 0.0,
        'timestamp': datetime.now().isoformat()
    }), 500

if __name__ == '__main__':
    # Test Telegram connection on startup
    logger.info("Starting COSMIC AI Binary Signal Bot...")
    
    try:
        if telegram_bot.test_connection():
            logger.info("Telegram bot connected successfully")
            # Send startup notification
            telegram_bot.send_startup_message()
        else:
            logger.warning("Telegram bot connection failed - check your bot token and chat ID")
    except Exception as e:
        logger.error(f"Telegram startup error: {str(e)}")
    
    # Run the application
    logger.info("Starting Flask server on port 5000...")
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,  # Set to False for production
        threaded=True
    )