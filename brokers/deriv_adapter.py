import asyncio
import websockets
import json
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import pandas as pd
from dataclasses import dataclass

from config import Config
from models.trade import Trade, TradeDirection, TradeStatus, BrokerType

@dataclass
class DerivContract:
    """Deriv binary options contract"""
    contract_id: str
    symbol: str
    contract_type: str
    amount: float
    duration: int
    duration_unit: str
    entry_tick: float
    exit_tick: Optional[float] = None
    profit: Optional[float] = None
    status: str = 'open'

class DerivAdapter:
    """Deriv.com broker adapter for binary options trading"""
    
    def __init__(self, api_token: str, app_id: str = "1089"):
        self.api_token = api_token
        self.app_id = app_id
        self.ws_url = f"wss://ws.binaryws.com/websockets/v3?app_id={app_id}"
        
        self.websocket = None
        self.is_connected = False
        self.account_info = {}
        self.active_contracts = {}
        self.market_data = {}
        
        # Callbacks
        self.on_tick_callback = None
        self.on_contract_update_callback = None
        self.on_connection_callback = None
        
        self.logger = logging.getLogger(__name__)
        
        # Request ID counter for tracking requests
        self.request_id = 1000
        
        # Available assets for binary options
        self.binary_assets = {
            'frxEURUSD': 'EUR/USD',
            'frxGBPUSD': 'GBP/USD', 
            'frxUSDJPY': 'USD/JPY',
            'frxAUDUSD': 'AUD/USD',
            'frxUSDCAD': 'USD/CAD',
            'frxUSDCHF': 'USD/CHF',
            'frxNZDUSD': 'NZD/USD',
            'frxEURGBP': 'EUR/GBP',
            'frxEURJPY': 'EUR/JPY',
            'frxGBPJPY': 'GBP/JPY',
            'cryBTCUSD': 'BTC/USD',
            'cryETHUSD': 'ETH/USD',
            'cryLTCUSD': 'LTC/USD',
            'cryXRPUSD': 'XRP/USD'
        }
    
    async def connect(self) -> bool:
        """Connect to Deriv WebSocket API"""
        try:
            self.websocket = await websockets.connect(self.ws_url)
            self.is_connected = True
            
            # Authorize with API token
            await self.authorize()
            
            # Get account information
            await self.get_account_info()
            
            # Start listening for messages
            asyncio.create_task(self._listen_for_messages())
            
            self.logger.info("Successfully connected to Deriv WebSocket API")
            
            if self.on_connection_callback:
                self.on_connection_callback(True)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Deriv: {e}")
            self.is_connected = False
            
            if self.on_connection_callback:
                self.on_connection_callback(False)
            
            return False
    
    async def disconnect(self):
        """Disconnect from Deriv WebSocket API"""
        if self.websocket and not self.websocket.closed:
            await self.websocket.close()
        
        self.is_connected = False
        self.logger.info("Disconnected from Deriv WebSocket API")
    
    async def authorize(self) -> bool:
        """Authorize API connection"""
        request = {
            "authorize": self.api_token,
            "req_id": self._get_request_id()
        }
        
        await self._send_request(request)
        return True
    
    async def get_account_info(self) -> Dict:
        """Get account information"""
        request = {
            "get_account_status": 1,
            "req_id": self._get_request_id()
        }
        
        await self._send_request(request)
        
        # Also get balance
        balance_request = {
            "balance": 1,
            "account": "all",
            "req_id": self._get_request_id()
        }
        
        await self._send_request(balance_request)
        
        return self.account_info
    
    async def get_available_assets(self) -> List[Dict]:
        """Get available assets for trading"""
        request = {
            "active_symbols": "brief",
            "product_type": "basic",
            "req_id": self._get_request_id()
        }
        
        await self._send_request(request)
        
        # Return predefined binary assets
        return [
            {"symbol": symbol, "display_name": name, "market": "forex" if symbol.startswith('frx') else "crypto"}
            for symbol, name in self.binary_assets.items()
        ]
    
    async def subscribe_to_ticks(self, symbol: str) -> bool:
        """Subscribe to real-time price ticks"""
        request = {
            "ticks": symbol,
            "subscribe": 1,
            "req_id": self._get_request_id()
        }
        
        await self._send_request(request)
        return True
    
    async def unsubscribe_from_ticks(self, symbol: str) -> bool:
        """Unsubscribe from price ticks"""
        request = {
            "forget_all": "ticks",
            "req_id": self._get_request_id()
        }
        
        await self._send_request(request)
        return True
    
    async def get_historical_data(self, symbol: str, timeframe: str = "1m", 
                                count: int = 100) -> pd.DataFrame:
        """Get historical price data"""
        
        # Convert timeframe to Deriv granularity
        granularity_map = {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "30m": 1800,
            "1h": 3600,
            "4h": 14400,
            "1d": 86400
        }
        
        granularity = granularity_map.get(timeframe, 60)
        
        request = {
            "ticks_history": symbol,
            "adjust_start_time": 1,
            "count": count,
            "end": "latest",
            "granularity": granularity,
            "style": "candles",
            "req_id": self._get_request_id()
        }
        
        await self._send_request(request)
        
        # Wait for response and process data
        # This is simplified - in reality you'd wait for the specific response
        return pd.DataFrame()
    
    async def place_binary_option(self, symbol: str, direction: TradeDirection, 
                                amount: float, duration: int = 300) -> Optional[str]:
        """Place a binary options trade"""
        
        contract_type = "CALL" if direction == TradeDirection.CALL else "PUT"
        
        # Get current price for contract
        current_price = await self._get_current_price(symbol)
        if not current_price:
            self.logger.error(f"Could not get current price for {symbol}")
            return None
        
        # Create contract request
        request = {
            "buy": 1,
            "subscribe": 1,
            "price": amount,
            "parameters": {
                "contract_type": contract_type,
                "symbol": symbol,
                "duration": duration,
                "duration_unit": "s",
                "amount": amount,
                "basis": "stake"
            },
            "req_id": self._get_request_id()
        }
        
        try:
            await self._send_request(request)
            
            # In a real implementation, you'd wait for the buy response
            # and extract the contract ID
            contract_id = f"deriv_{datetime.now().timestamp()}"
            
            # Store contract info
            contract = DerivContract(
                contract_id=contract_id,
                symbol=symbol,
                contract_type=contract_type,
                amount=amount,
                duration=duration,
                duration_unit="s",
                entry_tick=current_price
            )
            
            self.active_contracts[contract_id] = contract
            
            self.logger.info(f"Placed {contract_type} contract for {symbol}: {contract_id}")
            return contract_id
            
        except Exception as e:
            self.logger.error(f"Failed to place binary option: {e}")
            return None
    
    async def get_contract_status(self, contract_id: str) -> Optional[Dict]:
        """Get status of a specific contract"""
        if contract_id not in self.active_contracts:
            return None
        
        contract = self.active_contracts[contract_id]
        
        # Request contract status from API
        request = {
            "proposal_open_contract": 1,
            "contract_id": contract_id,
            "req_id": self._get_request_id()
        }
        
        await self._send_request(request)
        
        return {
            "contract_id": contract_id,
            "status": contract.status,
            "entry_tick": contract.entry_tick,
            "exit_tick": contract.exit_tick,
            "profit": contract.profit,
            "amount": contract.amount
        }
    
    async def close_contract(self, contract_id: str) -> bool:
        """Close an active contract (if supported)"""
        if contract_id not in self.active_contracts:
            return False
        
        # Deriv doesn't typically allow early closure of binary options
        # This would be for other contract types
        request = {
            "sell": contract_id,
            "price": 0,  # Market price
            "req_id": self._get_request_id()
        }
        
        try:
            await self._send_request(request)
            return True
        except Exception as e:
            self.logger.error(f"Failed to close contract {contract_id}: {e}")
            return False
    
    async def get_active_contracts(self) -> List[Dict]:
        """Get all active contracts"""
        request = {
            "portfolio": 1,
            "req_id": self._get_request_id()
        }
        
        await self._send_request(request)
        
        return [
            {
                "contract_id": contract_id,
                "symbol": contract.symbol,
                "contract_type": contract.contract_type,
                "amount": contract.amount,
                "status": contract.status,
                "entry_tick": contract.entry_tick
            }
            for contract_id, contract in self.active_contracts.items()
        ]
    
    async def get_balance(self) -> float:
        """Get account balance"""
        request = {
            "balance": 1,
            "req_id": self._get_request_id()
        }
        
        await self._send_request(request)
        
        return self.account_info.get('balance', 0.0)
    
    def set_tick_callback(self, callback: Callable):
        """Set callback for tick updates"""
        self.on_tick_callback = callback
    
    def set_contract_callback(self, callback: Callable):
        """Set callback for contract updates"""
        self.on_contract_update_callback = callback
    
    def set_connection_callback(self, callback: Callable):
        """Set callback for connection status"""
        self.on_connection_callback = callback
    
    async def _listen_for_messages(self):
        """Listen for incoming WebSocket messages"""
        try:
            while self.is_connected and self.websocket:
                message = await self.websocket.recv()
                data = json.loads(message)
                
                await self._handle_message(data)
                
        except websockets.exceptions.ConnectionClosed:
            self.logger.warning("WebSocket connection closed")
            self.is_connected = False
        except Exception as e:
            self.logger.error(f"Error listening for messages: {e}")
            self.is_connected = False
    
    async def _handle_message(self, data: Dict):
        """Handle incoming WebSocket messages"""
        msg_type = data.get('msg_type')
        
        if msg_type == 'tick':
            await self._handle_tick(data)
        elif msg_type == 'buy':
            await self._handle_buy_response(data)
        elif msg_type == 'proposal_open_contract':
            await self._handle_contract_update(data)
        elif msg_type == 'balance':
            await self._handle_balance_update(data)
        elif msg_type == 'authorize':
            await self._handle_authorize_response(data)
        elif msg_type == 'get_account_status':
            await self._handle_account_status(data)
        elif msg_type == 'active_symbols':
            await self._handle_active_symbols(data)
        elif msg_type == 'history':
            await self._handle_historical_data(data)
        elif msg_type == 'error':
            self.logger.error(f"API Error: {data.get('error', {}).get('message', 'Unknown error')}")
    
    async def _handle_tick(self, data: Dict):
        """Handle tick data"""
        tick = data.get('tick', {})
        symbol = tick.get('symbol')
        price = tick.get('quote')
        timestamp = tick.get('epoch')
        
        if symbol and price:
            self.market_data[symbol] = {
                'price': price,
                'timestamp': timestamp,
                'bid': price,  # Simplified
                'ask': price   # Simplified
            }
            
            if self.on_tick_callback:
                self.on_tick_callback(symbol, price, timestamp)
    
    async def _handle_buy_response(self, data: Dict):
        """Handle buy contract response"""
        buy_data = data.get('buy', {})
        contract_id = buy_data.get('contract_id')
        
        if contract_id and contract_id in self.active_contracts:
            contract = self.active_contracts[contract_id]
            contract.status = 'active'
            
            self.logger.info(f"Contract {contract_id} confirmed and active")
    
    async def _handle_contract_update(self, data: Dict):
        """Handle contract status updates"""
        contract_data = data.get('proposal_open_contract', {})
        contract_id = contract_data.get('contract_id')
        
        if contract_id and contract_id in self.active_contracts:
            contract = self.active_contracts[contract_id]
            
            # Update contract status
            if contract_data.get('is_expired'):
                contract.status = 'expired'
                contract.exit_tick = contract_data.get('exit_tick')
                contract.profit = contract_data.get('profit')
                
                if self.on_contract_update_callback:
                    self.on_contract_update_callback(contract_id, contract.status, contract.profit)
    
    async def _handle_balance_update(self, data: Dict):
        """Handle balance updates"""
        balance_data = data.get('balance', {})
        balance = balance_data.get('balance')
        currency = balance_data.get('currency')
        
        if balance is not None:
            self.account_info['balance'] = float(balance)
            self.account_info['currency'] = currency
    
    async def _handle_authorize_response(self, data: Dict):
        """Handle authorization response"""
        authorize_data = data.get('authorize', {})
        
        if authorize_data:
            self.account_info.update({
                'loginid': authorize_data.get('loginid'),
                'email': authorize_data.get('email'),
                'currency': authorize_data.get('currency'),
                'is_virtual': authorize_data.get('is_virtual', False)
            })
            
            self.logger.info(f"Authorized as {authorize_data.get('email')} ({authorize_data.get('loginid')})")
    
    async def _handle_account_status(self, data: Dict):
        """Handle account status response"""
        status_data = data.get('get_account_status', {})
        
        if status_data:
            self.account_info.update({
                'status': status_data.get('status'),
                'risk_classification': status_data.get('risk_classification')
            })
    
    async def _handle_active_symbols(self, data: Dict):
        """Handle active symbols response"""
        symbols_data = data.get('active_symbols', [])
        
        # Process and store available symbols
        for symbol_info in symbols_data:
            symbol = symbol_info.get('symbol')
            if symbol in self.binary_assets:
                self.binary_assets[symbol] = symbol_info.get('display_name', symbol)
    
    async def _handle_historical_data(self, data: Dict):
        """Handle historical data response"""
        history_data = data.get('history', {})
        
        # Process candle data if available
        prices = history_data.get('prices', [])
        times = history_data.get('times', [])
        
        if prices and times:
            # Create DataFrame from historical data
            df_data = []
            for i, (time, price) in enumerate(zip(times, prices)):
                df_data.append({
                    'timestamp': pd.to_datetime(time, unit='s'),
                    'open': price,
                    'high': price,  # Simplified
                    'low': price,   # Simplified
                    'close': price,
                    'volume': 0
                })
            
            # Store in market_data for retrieval
            symbol = history_data.get('symbol')
            if symbol:
                self.market_data[f"{symbol}_history"] = pd.DataFrame(df_data)
    
    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol"""
        if symbol in self.market_data:
            return self.market_data[symbol].get('price')
        
        # Request current tick if not available
        await self.subscribe_to_ticks(symbol)
        
        # Wait a moment for tick data
        await asyncio.sleep(1)
        
        if symbol in self.market_data:
            return self.market_data[symbol].get('price')
        
        return None
    
    async def _send_request(self, request: Dict):
        """Send request to WebSocket API"""
        if not self.is_connected or not self.websocket:
            raise ConnectionError("Not connected to Deriv WebSocket API")
        
        message = json.dumps(request)
        await self.websocket.send(message)
        
        self.logger.debug(f"Sent request: {message}")
    
    def _get_request_id(self) -> int:
        """Get next request ID"""
        self.request_id += 1
        return self.request_id
    
    def is_market_open(self, symbol: str) -> bool:
        """Check if market is open for trading"""
        # Simplified market hours check
        # Forex markets are generally open 24/5
        # Crypto markets are open 24/7
        
        if symbol.startswith('cry'):  # Crypto
            return True
        elif symbol.startswith('frx'):  # Forex
            now = datetime.now()
            # Check if it's weekend
            if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
                return False
            return True
        
        return True
    
    def get_minimum_stake(self, symbol: str) -> float:
        """Get minimum stake amount for symbol"""
        return 0.35  # Deriv minimum stake
    
    def get_maximum_stake(self, symbol: str) -> float:
        """Get maximum stake amount for symbol"""
        return 50000.0  # Deriv maximum stake
    
    def get_payout_rate(self, symbol: str, contract_type: str) -> float:
        """Get payout rate for symbol and contract type"""
        # Simplified payout rate - in reality this varies
        return 0.85  # 85% payout rate