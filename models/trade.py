from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime
from enum import Enum as PyEnum
from typing import Optional
import uuid

Base = declarative_base()

class TradeDirection(PyEnum):
    CALL = "call"
    PUT = "put"

class TradeStatus(PyEnum):
    PENDING = "pending"
    ACTIVE = "active"
    WON = "won"
    LOST = "lost"
    CANCELLED = "cancelled"

class BrokerType(PyEnum):
    DERIV = "deriv"
    IQ_OPTION = "iq_option"
    QUOTEX = "quotex"
    BINANCE = "binance"
    BYBIT = "bybit"

class Trade(Base):
    __tablename__ = 'trades'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    broker = Column(Enum(BrokerType), nullable=False)
    asset = Column(String(50), nullable=False)
    direction = Column(Enum(TradeDirection), nullable=False)
    amount = Column(Float, nullable=False)
    entry_price = Column(Float)
    exit_price = Column(Float)
    expiry_time = Column(Integer, nullable=False)  # in seconds
    payout_rate = Column(Float, default=0.8)  # 80% payout
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    opened_at = Column(DateTime)
    closed_at = Column(DateTime)
    
    # Trade status
    status = Column(Enum(TradeStatus), default=TradeStatus.PENDING)
    
    # Strategy information
    strategy_name = Column(String(100))
    strategy_confidence = Column(Float)
    strategy_reasoning = Column(Text)
    
    # Technical indicators at trade time
    rsi_value = Column(Float)
    macd_signal = Column(String(10))  # 'bullish', 'bearish', 'neutral'
    bb_position = Column(String(20))  # 'upper', 'lower', 'middle'
    trend_direction = Column(String(10))  # 'up', 'down', 'sideways'
    
    # Risk management
    risk_reward_ratio = Column(Float)
    max_loss_amount = Column(Float)
    
    # Results
    profit_loss = Column(Float, default=0.0)
    win = Column(Boolean, default=False)
    
    # External trade ID from broker
    external_trade_id = Column(String(100))
    
    def __repr__(self):
        return f"<Trade(id={self.id}, broker={self.broker.value}, asset={self.asset}, direction={self.direction.value}, amount={self.amount}, status={self.status.value})>"

class TradingSession(Base):
    __tablename__ = 'trading_sessions'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    broker = Column(Enum(BrokerType), nullable=False)
    
    # Session timestamps
    started_at = Column(DateTime, default=func.now())
    ended_at = Column(DateTime)
    
    # Session statistics
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    win_rate = Column(Float, default=0.0)
    
    # Financial results
    starting_balance = Column(Float)
    ending_balance = Column(Float)
    total_profit_loss = Column(Float, default=0.0)
    max_drawdown = Column(Float, default=0.0)
    
    # Risk metrics
    largest_win = Column(Float, default=0.0)
    largest_loss = Column(Float, default=0.0)
    consecutive_wins = Column(Integer, default=0)
    consecutive_losses = Column(Integer, default=0)
    
    # Session status
    is_active = Column(Boolean, default=True)
    stop_reason = Column(String(100))  # 'manual', 'max_loss', 'daily_limit', 'error'
    
    def calculate_win_rate(self):
        """Calculate and update win rate"""
        if self.total_trades > 0:
            self.win_rate = (self.winning_trades / self.total_trades) * 100
        return self.win_rate
    
    def update_statistics(self, trade: Trade):
        """Update session statistics with new trade"""
        self.total_trades += 1
        
        if trade.win:
            self.winning_trades += 1
            if trade.profit_loss > self.largest_win:
                self.largest_win = trade.profit_loss
        else:
            self.losing_trades += 1
            if abs(trade.profit_loss) > self.largest_loss:
                self.largest_loss = abs(trade.profit_loss)
        
        self.total_profit_loss += trade.profit_loss
        self.calculate_win_rate()

class MarketData(Base):
    __tablename__ = 'market_data'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    broker = Column(Enum(BrokerType), nullable=False)
    asset = Column(String(50), nullable=False)
    timeframe = Column(String(10), nullable=False)  # '1m', '5m', etc.
    
    # OHLCV data
    timestamp = Column(DateTime, nullable=False)
    open_price = Column(Float, nullable=False)
    high_price = Column(Float, nullable=False)
    low_price = Column(Float, nullable=False)
    close_price = Column(Float, nullable=False)
    volume = Column(Float, default=0.0)
    
    # Technical indicators
    rsi = Column(Float)
    macd_line = Column(Float)
    macd_signal = Column(Float)
    macd_histogram = Column(Float)
    bb_upper = Column(Float)
    bb_middle = Column(Float)
    bb_lower = Column(Float)
    sma_short = Column(Float)
    sma_long = Column(Float)
    ema_short = Column(Float)
    ema_long = Column(Float)
    
    # Additional market metrics
    volatility = Column(Float)
    trend_strength = Column(Float)
    support_level = Column(Float)
    resistance_level = Column(Float)
    
    def __repr__(self):
        return f"<MarketData(asset={self.asset}, timestamp={self.timestamp}, close={self.close_price})>"

class Strategy(Base):
    __tablename__ = 'strategies'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(100), nullable=False, unique=True)
    description = Column(Text)
    
    # Strategy configuration
    config_json = Column(Text)  # JSON string of strategy parameters
    
    # Performance metrics
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    win_rate = Column(Float, default=0.0)
    total_profit_loss = Column(Float, default=0.0)
    max_drawdown = Column(Float, default=0.0)
    sharpe_ratio = Column(Float, default=0.0)
    
    # Strategy status
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # AI/ML specific fields
    model_version = Column(String(50))
    confidence_threshold = Column(Float, default=0.7)
    adaptation_rate = Column(Float, default=0.1)
    
    def calculate_performance_metrics(self):
        """Calculate and update strategy performance metrics"""
        if self.total_trades > 0:
            self.win_rate = (self.winning_trades / self.total_trades) * 100
        # Additional calculations for Sharpe ratio, max drawdown, etc. would go here

class SignalAlert(Base):
    __tablename__ = 'signal_alerts'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    asset = Column(String(50), nullable=False)
    direction = Column(Enum(TradeDirection), nullable=False)
    confidence = Column(Float, nullable=False)
    strategy_name = Column(String(100), nullable=False)
    reasoning = Column(Text)
    
    # Signal details
    entry_price = Column(Float)
    recommended_amount = Column(Float)
    expiry_time = Column(Integer)  # in seconds
    risk_level = Column(String(20))  # 'low', 'medium', 'high'
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    expires_at = Column(DateTime)
    
    # Status
    is_executed = Column(Boolean, default=False)
    trade_id = Column(String)  # Reference to executed trade
    
    def __repr__(self):
        return f"<SignalAlert(asset={self.asset}, direction={self.direction.value}, confidence={self.confidence})>"