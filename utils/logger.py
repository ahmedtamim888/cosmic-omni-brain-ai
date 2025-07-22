#!/usr/bin/env python3
"""
📝 LOGGER - Logging System for Ghost Transcendence Core
Professional logging with emoji support and structured output
"""

import logging
import sys
from datetime import datetime
import os

def setup_logger(name: str = "GHOST_TRANSCENDENCE_CORE", level: str = "INFO") -> logging.Logger:
    """
    Setup logger with custom formatting for the AI trading bot
    """
    
    # Create logger
    logger = logging.getLogger(name)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Set level
    log_levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    logger.setLevel(log_levels.get(level.upper(), logging.INFO))
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    
    # Create formatter with emoji and structured layout
    console_formatter = GhostFormatter()
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    
    # Create file handler for persistent logging
    try:
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        log_file = os.path.join(log_dir, f"ghost_transcendence_{datetime.now().strftime('%Y%m%d')}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # File formatter (without emoji for compatibility)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
    except Exception as e:
        # If file logging fails, continue with console only
        logger.warning(f"⚠️ Could not setup file logging: {str(e)}")
    
    # Prevent duplicate logs
    logger.propagate = False
    
    return logger

class GhostFormatter(logging.Formatter):
    """
    Custom formatter for Ghost Transcendence Core with emoji and colors
    """
    
    def __init__(self):
        super().__init__()
        
        # Color codes for terminal
        self.COLORS = {
            'DEBUG': '\033[36m',      # Cyan
            'INFO': '\033[32m',       # Green
            'WARNING': '\033[33m',    # Yellow
            'ERROR': '\033[31m',      # Red
            'CRITICAL': '\033[35m',   # Magenta
            'RESET': '\033[0m',       # Reset
            'BOLD': '\033[1m',        # Bold
            'DIM': '\033[2m'          # Dim
        }
        
        # Emoji mapping for log levels
        self.EMOJIS = {
            'DEBUG': '🔍',
            'INFO': '💡',
            'WARNING': '⚠️',
            'ERROR': '❌',
            'CRITICAL': '🚨'
        }
        
        # Component emoji mapping
        self.COMPONENT_EMOJIS = {
            'perception_engine': '👁️',
            'context_engine': '🧠',
            'intelligence_engine': '🎯',
            'signal_generator': '⚡',
            'chart_analyzer': '📊',
            'app': '🤖',
            'telegram': '📱'
        }
    
    def format(self, record):
        """
        Format log record with colors and emojis
        """
        try:
            # Get color and emoji for log level
            level_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
            level_emoji = self.EMOJIS.get(record.levelname, '📝')
            
            # Get component emoji
            component_emoji = self._get_component_emoji(record.name, record.funcName)
            
            # Format timestamp
            timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S.%f')[:-3]
            
            # Format level name
            level_name = record.levelname.ljust(8)
            
            # Format module/function info
            module_info = f"{record.name.split('.')[-1]}"
            if hasattr(record, 'funcName') and record.funcName != '<module>':
                module_info += f".{record.funcName}"
            
            # Build the formatted message
            formatted_parts = [
                f"{self.COLORS['DIM']}{timestamp}{self.COLORS['RESET']}",
                f"{level_emoji}",
                f"{level_color}{level_name}{self.COLORS['RESET']}",
                f"{component_emoji}",
                f"{self.COLORS['BOLD']}{module_info}{self.COLORS['RESET']}",
                f"{record.getMessage()}"
            ]
            
            formatted_message = " | ".join(formatted_parts)
            
            # Add exception info if present
            if record.exc_info:
                formatted_message += f"\n{self.formatException(record.exc_info)}"
            
            return formatted_message
            
        except Exception as e:
            # Fallback to basic formatting if custom formatting fails
            return f"{record.levelname}: {record.getMessage()}"
    
    def _get_component_emoji(self, logger_name: str, func_name: str) -> str:
        """
        Get appropriate emoji for component based on logger name or function name
        """
        logger_lower = logger_name.lower()
        func_lower = func_name.lower() if func_name else ""
        
        # Check component patterns
        for component, emoji in self.COMPONENT_EMOJIS.items():
            if component in logger_lower or component in func_lower:
                return emoji
        
        # Check for specific AI patterns
        if any(pattern in logger_lower for pattern in ['perception', 'eye', 'vision']):
            return '👁️'
        elif any(pattern in logger_lower for pattern in ['context', 'memory', 'understand']):
            return '🧠'
        elif any(pattern in logger_lower for pattern in ['intelligence', 'strategy', 'logic']):
            return '🎯'
        elif any(pattern in logger_lower for pattern in ['signal', 'decision', 'generator']):
            return '⚡'
        elif any(pattern in logger_lower for pattern in ['chart', 'analyze', 'pattern']):
            return '📊'
        elif any(pattern in logger_lower for pattern in ['telegram', 'bot', 'message']):
            return '📱'
        elif any(pattern in logger_lower for pattern in ['flask', 'web', 'api']):
            return '🌐'
        elif any(pattern in logger_lower for pattern in ['ghost', 'transcendence', 'core']):
            return '👻'
        else:
            return '🔧'  # Default utility emoji

def get_ghost_logger(component_name: str = "GHOST_COMPONENT") -> logging.Logger:
    """
    Get a logger for a specific Ghost Transcendence Core component
    """
    logger_name = f"GHOST_TRANSCENDENCE_CORE.{component_name}"
    return logging.getLogger(logger_name)

def log_ai_decision(logger: logging.Logger, decision_type: str, confidence: float, details: str = ""):
    """
    Log AI decision with special formatting
    """
    confidence_emoji = "🎯" if confidence > 0.8 else "📈" if confidence > 0.6 else "📊"
    confidence_str = f"{confidence:.1%}"
    
    message = f"{confidence_emoji} AI DECISION: {decision_type} | Confidence: {confidence_str}"
    if details:
        message += f" | {details}"
    
    logger.info(message)

def log_signal_generation(logger: logging.Logger, signal: str, confidence: float, reasoning: str = ""):
    """
    Log signal generation with special formatting
    """
    signal_emoji = "🚀" if signal == "CALL" else "📉" if signal == "PUT" else "⏸️"
    
    message = f"{signal_emoji} SIGNAL GENERATED: {signal} | Confidence: {confidence:.1%}"
    if reasoning:
        # Take first line of reasoning for log
        first_line = reasoning.split('\n')[0]
        message += f" | {first_line}"
    
    logger.info(message)

def log_error_with_context(logger: logging.Logger, error: Exception, context: str = "", component: str = ""):
    """
    Log error with additional context information
    """
    error_msg = f"💥 ERROR in {component}: {str(error)}"
    if context:
        error_msg += f" | Context: {context}"
    
    logger.error(error_msg, exc_info=True)

def log_performance_metric(logger: logging.Logger, metric_name: str, value: float, unit: str = ""):
    """
    Log performance metrics
    """
    metric_emoji = "⚡" if "speed" in metric_name.lower() else "📊"
    message = f"{metric_emoji} METRIC: {metric_name} = {value}"
    if unit:
        message += f" {unit}"
    
    logger.debug(message)

def log_ghost_activation(logger: logging.Logger, feature: str, status: str = "ACTIVATED"):
    """
    Log Ghost Transcendence Core feature activation
    """
    ghost_emoji = "👻"
    message = f"{ghost_emoji} GHOST FEATURE {status}: {feature}"
    logger.info(message)

# Initialize the main logger when module is imported
main_logger = setup_logger()

# Export convenience function
def get_logger(name: str = None) -> logging.Logger:
    """Get logger instance"""
    if name:
        return get_ghost_logger(name)
    return main_logger