"""
Market Data Provider
Fetches real-time and historical market data from multiple sources
"""

import asyncio
import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
import aiohttp
import json
import random

class MarketDataProvider:
    """
    Multi-source market data provider
    Supports real-time and historical data fetching
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.data_sources = ['yahoo', 'alpha_vantage', 'synthetic']
        self.current_source = 'synthetic'  # Default to synthetic for demo
        self.data_cache = {}
        self.cache_duration = 60  # Cache for 60 seconds
        
    async def get_real_time_data(self, symbol: str = 'EURUSD', timeframe: str = '1m') -> Optional[Dict]:
        """Get real-time market data"""
        try:
            # Check cache first
            cache_key = f"{symbol}_{timeframe}_realtime"
            if self._is_cached(cache_key):
                return self.data_cache[cache_key]['data']
            
            # Generate synthetic data for demo purposes
            # In production, this would connect to real data sources
            data = await self._get_synthetic_realtime_data(symbol, timeframe)
            
            # Cache the data
            self._cache_data(cache_key, data)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error getting real-time data: {e}")
            return None
    
    async def get_historical_data(self, symbol: str = 'EURUSD', timeframe: str = '1m', 
                                periods: int = 1000) -> Optional[pd.DataFrame]:
        """Get historical market data"""
        try:
            cache_key = f"{symbol}_{timeframe}_historical_{periods}"
            if self._is_cached(cache_key):
                return self.data_cache[cache_key]['data']
            
            # Generate synthetic historical data for demo
            data = await self._get_synthetic_historical_data(symbol, timeframe, periods)
            
            # Cache the data
            self._cache_data(cache_key, data)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error getting historical data: {e}")
            return None
    
    async def _get_synthetic_realtime_data(self, symbol: str, timeframe: str) -> Dict:
        """Generate synthetic real-time data for demo purposes"""
        try:
            # Base price for different symbols
            base_prices = {
                'EURUSD': 1.0850,
                'GBPUSD': 1.2650,
                'USDJPY': 149.50,
                'BTCUSD': 43500.0,
                'GOLD': 2050.0
            }
            
            base_price = base_prices.get(symbol, 1.0850)
            
            # Generate realistic candle data
            candles = []
            current_time = datetime.now()
            
            # Generate last 50 candles for context
            for i in range(50, 0, -1):
                candle_time = current_time - timedelta(minutes=i)
                
                # Random walk with some trending behavior
                if i == 50:
                    open_price = base_price
                else:
                    open_price = candles[-1]['close']
                
                # Add some volatility and trend
                volatility = base_price * 0.001  # 0.1% volatility
                trend = np.sin(i * 0.1) * 0.0005  # Slight trending
                
                # Generate OHLC
                price_change = np.random.normal(trend, volatility)
                close_price = open_price + price_change
                
                high_extension = abs(np.random.normal(0, volatility * 0.5))
                low_extension = abs(np.random.normal(0, volatility * 0.5))
                
                high_price = max(open_price, close_price) + high_extension
                low_price = min(open_price, close_price) - low_extension
                
                # Generate volume
                base_volume = 1000000
                volume = base_volume + np.random.normal(0, base_volume * 0.3)
                volume = max(100000, volume)  # Minimum volume
                
                candle = {
                    'timestamp': candle_time,
                    'open': round(open_price, 5),
                    'high': round(high_price, 5),
                    'low': round(low_price, 5),
                    'close': round(close_price, 5),
                    'volume': int(volume)
                }
                
                candles.append(candle)
            
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'candles': candles,
                'last_update': current_time
            }
            
        except Exception as e:
            self.logger.error(f"Error generating synthetic real-time data: {e}")
            return None
    
    async def _get_synthetic_historical_data(self, symbol: str, timeframe: str, periods: int) -> pd.DataFrame:
        """Generate synthetic historical data for AI training"""
        try:
            base_prices = {
                'EURUSD': 1.0850,
                'GBPUSD': 1.2650,
                'USDJPY': 149.50,
                'BTCUSD': 43500.0,
                'GOLD': 2050.0
            }
            
            base_price = base_prices.get(symbol, 1.0850)
            
            # Generate historical data
            data = []
            current_price = base_price
            current_time = datetime.now() - timedelta(minutes=periods)
            
            for i in range(periods):
                # More realistic price movements with trends and reversals
                
                # Create trending periods
                trend_cycle = np.sin(i * 0.02) * 0.001  # Longer trend cycles
                volatility_cycle = (0.5 + 0.5 * np.sin(i * 0.05)) * 0.0008  # Volatility cycles
                
                # Random component
                random_component = np.random.normal(0, 0.0003)
                
                # Combine components
                price_change = trend_cycle + random_component
                open_price = current_price
                close_price = open_price + price_change
                
                # Generate realistic wicks
                wick_size = volatility_cycle * np.random.uniform(0.5, 2.0)
                upper_wick = np.random.uniform(0, wick_size)
                lower_wick = np.random.uniform(0, wick_size)
                
                high_price = max(open_price, close_price) + upper_wick
                low_price = min(open_price, close_price) - lower_wick
                
                # Generate volume with correlation to volatility
                base_volume = 1000000
                volatility_factor = abs(price_change) / 0.001
                volume = base_volume * (1 + volatility_factor * 0.5) * np.random.uniform(0.5, 1.5)
                
                data.append({
                    'timestamp': current_time + timedelta(minutes=i),
                    'open': round(open_price, 5),
                    'high': round(high_price, 5),
                    'low': round(low_price, 5),
                    'close': round(close_price, 5),
                    'volume': int(volume)
                })
                
                current_price = close_price
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error generating synthetic historical data: {e}")
            return pd.DataFrame()
    
    async def _get_yahoo_data(self, symbol: str, timeframe: str, periods: int) -> Optional[pd.DataFrame]:
        """Get data from Yahoo Finance (for real implementation)"""
        try:
            # Convert symbol format for Yahoo
            yahoo_symbol = self._convert_symbol_for_yahoo(symbol)
            
            # Convert timeframe
            period_map = {
                '1m': '1m',
                '5m': '5m',
                '15m': '15m',
                '1h': '1h',
                '1d': '1d'
            }
            
            interval = period_map.get(timeframe, '1m')
            
            # Calculate period
            if timeframe == '1m':
                period = f"{periods}m"
            elif timeframe == '5m':
                period = f"{periods * 5}m"
            else:
                period = "1d"
            
            # Fetch data
            ticker = yf.Ticker(yahoo_symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                return None
            
            # Rename columns to match our format
            data.columns = [col.lower() for col in data.columns]
            data.reset_index(inplace=True)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error getting Yahoo data: {e}")
            return None
    
    async def _get_alpha_vantage_data(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """Get data from Alpha Vantage API (for real implementation)"""
        try:
            # This would require an Alpha Vantage API key
            # Implementation would go here
            pass
            
        except Exception as e:
            self.logger.error(f"Error getting Alpha Vantage data: {e}")
            return None
    
    def _convert_symbol_for_yahoo(self, symbol: str) -> str:
        """Convert symbol format for Yahoo Finance"""
        symbol_map = {
            'EURUSD': 'EURUSD=X',
            'GBPUSD': 'GBPUSD=X',
            'USDJPY': 'USDJPY=X',
            'BTCUSD': 'BTC-USD',
            'GOLD': 'GC=F'
        }
        
        return symbol_map.get(symbol, f"{symbol}=X")
    
    def _is_cached(self, cache_key: str) -> bool:
        """Check if data is in cache and still valid"""
        if cache_key not in self.data_cache:
            return False
        
        cache_time = self.data_cache[cache_key]['timestamp']
        return (datetime.now() - cache_time).seconds < self.cache_duration
    
    def _cache_data(self, cache_key: str, data: Any):
        """Cache data with timestamp"""
        self.data_cache[cache_key] = {
            'data': data,
            'timestamp': datetime.now()
        }
        
        # Clean old cache entries
        current_time = datetime.now()
        keys_to_remove = []
        
        for key, cache_data in self.data_cache.items():
            if (current_time - cache_data['timestamp']).seconds > self.cache_duration * 2:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.data_cache[key]
    
    async def test_connection(self) -> bool:
        """Test data source connection"""
        try:
            test_data = await self.get_real_time_data('EURUSD', '1m')
            return test_data is not None
            
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False
    
    def get_supported_symbols(self) -> List[str]:
        """Get list of supported symbols"""
        return ['EURUSD', 'GBPUSD', 'USDJPY', 'BTCUSD', 'GOLD']
    
    def get_supported_timeframes(self) -> List[str]:
        """Get list of supported timeframes"""
        return ['1m', '5m', '15m', '1h', '1d']