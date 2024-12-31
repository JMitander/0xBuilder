import asyncio
import os
import time
import logging
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from cachetools import TTLCache
from web3 import AsyncWeb3

from api_config import API_Config
from configuration import Configuration

logger = logging.getLogger("0xBuilder")

class Market_Monitor:
    """Advanced market monitoring system for real-time analysis and prediction."""

    # Class Constants
    MODEL_UPDATE_INTERVAL: int = 3600  # Update model every hour
    VOLATILITY_THRESHOLD: float = 0.05  # 5% standard deviation
    LIQUIDITY_THRESHOLD: int = 100_000  # $100,000 in 24h volume
    PRICE_EMA_SHORT_PERIOD: int = 12
    PRICE_EMA_LONG_PERIOD: int = 26
    

    def __init__(
        self,
        web3: "AsyncWeb3",
        configuration: Optional["Configuration"],
        api_config: Optional["API_Config"],
        transaction_core: Optional[Any] = None,  # Add this parameter
    ) -> None:
        """Initialize Market Monitor with required components."""
        self.web3: "AsyncWeb3" = web3
        self.configuration: Optional["Configuration"] = configuration
        self.api_config: Optional["API_Config"] = api_config
        self.transaction_core: Optional[Any] = transaction_core  # Store transaction_core reference
        self.price_model: Optional[LinearRegression] = LinearRegression()
        self.model_last_updated: float = 0
    
        # Get from config or default
        self.linear_regression_path: str = self.configuration.get_config_value("LINEAR_REGRESSION_PATH", "/home/mitander/0xBuilder/python/linear_regression") if self.configuration else "/home/mitander/0xBuilder/python/linear_regression"
        self.model_path: str = self.configuration.get_config_value("MODEL_PATH", "/home/mitander/0xBuilder/python/linear_regression/price_model.joblib") if self.configuration else "/home/mitander/0xBuilder/python/linear_regression/price_model.joblib"
        self.training_data_path: str = self.configuration.get_config_value("TRAINING_DATA_PATH", "/home/mitander/0xBuilder/python/linear_regression/training_data.csv") if self.configuration else "/home/mitander/0xBuilder/python/linear_regression/training_data.csv"

        # Create directory if it doesn't exist
        os.makedirs(self.linear_regression_path, exist_ok=True)
        
        # Add separate caches for different data types
        self.caches: Dict[str, TTLCache] = {
            'price': TTLCache(maxsize=2000, 
                    ttl=300),
            'volume': TTLCache(maxsize=1000, 
                     ttl=900),
            'volatility': TTLCache(maxsize=200, 
                     ttl=600)
        }
        
        # Initialize model variables
        self.price_model: Optional[LinearRegression] = None
        self.last_training_time: float = 0
        self.model_accuracy: float = 0.0
        self.RETRAINING_INTERVAL: int = self.configuration.MODEL_RETRAINING_INTERVAL if self.configuration else 3600 # Retrain every hour
        self.MIN_TRAINING_SAMPLES: int = self.configuration.MIN_TRAINING_SAMPLES if self.configuration else 100
        
        # Initialize data storage
        self.historical_data: pd.DataFrame = pd.DataFrame()
        self.prediction_cache: TTLCache = TTLCache(maxsize=1000, ttl=300)  # 5-minute cache

        # Add data update
        self.update_scheduler = {
            'training_data': 0,  # Last update timestamp
            'model': 0,          # Last model update timestamp
            'UPDATE_INTERVAL': self.configuration.MODEL_RETRAINING_INTERVAL if self.configuration else 3600,  # 1 hour
            'MODEL_INTERVAL': 86400   # 24 hours
        }

    async def initialize(self) -> None:
        """Initialize market monitor components and load model."""
        try:
            self.price_model = LinearRegression()
            self.model_last_updated = 0

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

            # Load existing model if available, or create new one
            model_loaded = False
            if os.path.exists(self.model_path):
                try:
                    self.price_model = joblib.load(self.model_path)
                    logger.debug("Loaded existing price prediction model")
                    model_loaded = True
                except (OSError, KeyError) as e:
                    logger.warning(f"Failed to load model: {e}. Creating new model.")
                    self.price_model = LinearRegression()
            
            if not model_loaded:
                logger.debug("Creating new price prediction model")
                self.price_model = LinearRegression()
                # Save initial model
                try:
                    joblib.dump(self.price_model, self.model_path)
                    logger.debug("Saved initial price prediction model")
                except Exception as e:
                    logger.warning(f"Failed to save initial model: {e}")
            
            # Load or create training data file
            if os.path.exists(self.training_data_path):
                try:
                    self.historical_data = pd.read_csv(self.training_data_path)
                    logger.debug(f"Loaded {len(self.historical_data)} historical data points")
                except Exception as e:
                    logger.warning(f"Failed to load historical data: {e}. Starting with empty dataset.")
                    self.historical_data = pd.DataFrame()
            else:
                self.historical_data = pd.DataFrame()
            
            # Initial model training if needed
            if len(self.historical_data) >= self.MIN_TRAINING_SAMPLES:
                await self._train_model()
            
            logger.debug("Market Monitor initialized ✅")

            # Start update scheduler
            asyncio.create_task(self.schedule_updates())

        except Exception as e:
            logger.critical(f"Market Monitor initialization failed: {e}")
            raise RuntimeError(f"Market Monitor initialization failed: {e}")

    async def schedule_updates(self) -> None:
        """Schedule periodic data and model updates."""
        while True:
            try:
                current_time = time.time()
                
                # Update training data
                if current_time - self.update_scheduler['training_data'] >= self.update_scheduler['UPDATE_INTERVAL']:
                    await self.api_config.update_training_data()
                    self.update_scheduler['training_data'] = current_time
                
                # Retrain model
                if current_time - self.update_scheduler['model'] >= self.update_scheduler['MODEL_INTERVAL']:
                    await self._train_model()
                    self.update_scheduler['model'] = current_time
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in update scheduler: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error

    async def check_market_conditions(self, token_address: str) -> Dict[str, bool]:
        """Analyze current market conditions for a given token."""
        market_conditions = {
            "high_volatility": False,
            "bullish_trend": False,
            "bearish_trend": False,
            "low_liquidity": False,
        }
        
        # Get symbol from address
        token_symbol = self.api_config.get_token_symbol(token_address)
        if not token_symbol:
            logger.debug(f"Cannot get token symbol for address {token_address}")
            return market_conditions

        # Use symbol for API calls
        prices = await self.get_price_data(token_symbol, data_type='historical', timeframe=1)
        if len(prices) < 2:
            logger.debug(f"Not enough price data to analyze market conditions for {token_symbol}")
            return market_conditions

        volatility = self._calculate_volatility(prices)
        if volatility > self.VOLATILITY_THRESHOLD:
            market_conditions["high_volatility"] = True
        logger.debug(f"Calculated volatility for {token_symbol}: {volatility}")

        moving_average = np.mean(prices)
        if prices[-1] > moving_average:
            market_conditions["bullish_trend"] = True
        elif prices[-1] < moving_average:
            market_conditions["bearish_trend"] = True

        volume = await self.get_token_volume(token_symbol)
        if volume < self.LIQUIDITY_THRESHOLD:
            market_conditions["low_liquidity"] = True

        return market_conditions

    # Price Analysis Methods
    async def predict_price_movement(
        self, 
        token_symbol: str
    ) -> float:
        """
        Predict future price movement using linear regression model.
        
        Args:
            token_symbol: Token symbol to analyze
            
        Returns:
            Predicted price value
        """
        try:
            cache_key = f"prediction_{token_symbol}"
            if cache_key in self.prediction_cache:
                return self.prediction_cache[cache_key]

            # Check if model needs retraining
            if time.time() - self.last_training_time > self.RETRAINING_INTERVAL:
                await self._train_model()

            # Get current market data
            market_data = await self._get_market_features(token_symbol)
            if not market_data:
                return 0.0

            # Make prediction
            features = ['market_cap', 'volume_24h', 'percent_change_24h', 'total_supply', 'circulating_supply', 'volatility', 'liquidity_ratio', 'avg_transaction_value', 
                        'trading_pairs', 'exchange_count', 'price_momentum', 'buy_sell_ratio', 'smart_money_flow']
            X = pd.DataFrame([market_data], columns=features)
            prediction = self.price_model.predict(X)[0]

            self.prediction_cache[cache_key] = prediction
            return float(prediction)

        except Exception as e:
            logger.error(f"Error predicting price movement: {e}")
            return 0.0

    async def _get_market_features(self, token_symbol: str) -> Optional[Dict[str, float]]:
        """Get current market features for prediction with enhanced metrics."""
        try:
             # Gather data concurrently
            price, volume, supply_data, market_data, prices = await asyncio.gather(
                self.api_config.get_real_time_price(token_symbol),
                self.api_config.get_token_volume(token_symbol),
                self.api_config.get_token_supply_data(token_symbol),
                self._get_trading_metrics(token_symbol),
                self.get_price_data(token_symbol, data_type='historical', timeframe=1),
                return_exceptions=True
            )

            if any(isinstance(r, Exception) for r in [price, volume, supply_data, market_data, prices]):
                logger.warning(f"Error fetching market data for {token_symbol}")
                return None

            # Basic features
            features = {
                'market_cap': await self.api_config.get_token_market_cap(token_symbol),
                'volume_24h': float(volume),
                'percent_change_24h': await self.api_config.get_price_change_24h(token_symbol),
                'total_supply': supply_data.get('total_supply', 0),
                'circulating_supply': supply_data.get('circulating_supply', 0),
                 'volatility': self._calculate_volatility(prices) if prices else 0,
                'price_momentum': self._calculate_momentum(prices) if prices else 0,
                'liquidity_ratio': await self._calculate_liquidity_ratio(token_symbol),
                **market_data
            }
            
            return features
        
        except Exception as e:
            logger.error(f"Error fetching market features: {e}")
            return None

    def _calculate_momentum(self, prices: List[float]) -> float:
        """Calculate price momentum using exponential moving average."""
        try:
            if  len(prices) < 2:
                return 0.0
            ema_short = np.mean(prices[-self.PRICE_EMA_SHORT_PERIOD:])  # 1-hour EMA
            ema_long = np.mean(prices)  # 24-hour EMA
            return (ema_short / ema_long) - 1 if ema_long != 0 else 0.0
        except Exception as e:
            logger.error(f"Error calculating momentum: {e}")
            return 0.0

    async def _calculate_liquidity_ratio(self, token_symbol: str) -> float:
        """Calculate liquidity ratio using market cap and volume from API config."""
        try:
            volume = await self.api_config.get_token_volume(token_symbol)
            market_cap = await self.api_config.get_token_market_cap(token_symbol)
            return volume / market_cap if market_cap > 0 else 0.0
        except Exception as e:
            logger.error(f"Error calculating liquidity ratio: {e}")
            return 0.0

    async def _get_trading_metrics(self, token_symbol: str) -> Dict[str, float]:
        """Get additional trading metrics."""
        try:
            return {
                'avg_transaction_value': await self._get_avg_transaction_value(token_symbol),
                'trading_pairs': await self._get_trading_pairs_count(token_symbol),
                'exchange_count': await self._get_exchange_count(token_symbol),
                'buy_sell_ratio': await self._get_buy_sell_ratio(token_symbol),
                'smart_money_flow': await self._get_smart_money_flow(token_symbol)
            }
        except Exception as e:
            logger.error(f"Error getting trading metrics: {e}")
            return {
                'avg_transaction_value': 0.0,
                'trading_pairs': 0.0,
                'exchange_count': 0.0,
                'buy_sell_ratio': 1.0,
                'smart_money_flow': 0.0
            }

    # Add methods to calculate new metrics
    async def _get_avg_transaction_value(self, token_symbol: str) -> float:
        """Get average transaction value over last 24h."""
        try:
            volume = await self.api_config.get_token_volume(token_symbol)
            tx_count = await self._get_transaction_count(token_symbol)
            return volume / tx_count if tx_count > 0 else 0.0
        except Exception as e:
            logger.error(f"Error calculating avg transaction value: {e}")
            return 0.0

    # Add helper methods to calculate new metrics
    async def _get_transaction_count(self, token_symbol: str) -> int:
        """Get number of transactions in 24 hrs using api config."""
        try:
            # This data is not available from the api config therefore, this will return 0.
            return 0
        except Exception as e:
             logger.error(f"Error getting transaction count {e}")
             return 0

    async def _get_trading_pairs_count(self, token_symbol: str) -> int:
        """Get number of trading pairs for a token using api config."""
        try:
            metadata = await self.api_config.get_token_metadata(token_symbol)
            return len(metadata.get('trading_pairs', [])) if metadata else 0
        except Exception as e:
            logger.error(f"Error getting trading pairs for {token_symbol}: {e}")
            return 0

    async def _get_exchange_count(self, token_symbol: str) -> int:
        """Get number of exchanges the token is listed on using api config."""
        try:
           metadata = await self.api_config.get_token_metadata(token_symbol)
           return len(metadata.get('exchanges', [])) if metadata else 0
        except Exception as e:
           logger.error(f"Error getting exchange count for {token_symbol}: {e}")
           return 0
        
    async def _get_buy_sell_ratio(self, token_symbol: str) -> float:
        """Get buy/sell ratio from an exchange API (mock)."""
        # Mock implementation returning default value.
        # Real implementation would query an exchange API
        return 1.0

    async def _get_smart_money_flow(self, token_symbol: str) -> float:
        """Calculate smart money flow (mock implementation)."""
        # Mock implementation returnig default value.
        # Real implementation would require on-chain data analysis
        return 0.0
    
    async def update_training_data(self, new_data: Dict[str, Any]) -> None:
        """Update training data with new market information."""
        try:
            # Convert new data to DataFrame row
            df_row = pd.DataFrame([new_data])
            
            # Append to historical data
            self.historical_data = pd.concat([self.historical_data, df_row], ignore_index=True)
            
            # Save updated data
            self.historical_data.to_csv(self.training_data_path, index=False)
            
            # Retrain model if enough new data
            if len(self.historical_data) >= self.MIN_TRAINING_SAMPLES:
                await self._train_model()
                
        except Exception as e:
            logger.error(f"Error updating training data: {e}")

    async def _train_model(self) -> None:
        """Enhanced model training with feature importance analysis."""
        try:
            if len(self.historical_data) < self.MIN_TRAINING_SAMPLES:
                logger.warning("Insufficient data for model training")
                return

            # Define all features we want to use
            features = [
                'market_cap', 'volume_24h', 'percent_change_24h', 
                'total_supply', 'circulating_supply', 'volatility',
                'liquidity_ratio', 'avg_transaction_value', 'trading_pairs',
                'exchange_count', 'price_momentum', 'buy_sell_ratio',
                'smart_money_flow'
            ]

            X = self.historical_data[features]
            y = self.historical_data['price_usd']

            # Train/test split with shuffling
            train_size = int(len(X) * 0.8)
            indices = np.random.permutation(len(X))
            X_train = X.iloc[indices[:train_size]]
            X_test = X.iloc[indices[train_size:]]
            y_train = y.iloc[indices[:train_size]]
            y_test = y.iloc[indices[train_size:]]

            # Train model
            self.price_model = LinearRegression()
            self.price_model.fit(X_train, y_train)

            # Calculate and log feature importance
            importance = pd.DataFrame({
                'feature': features,
                'importance': np.abs(self.price_model.coef_)
            })
            importance = importance.sort_values('importance', ascending=False)
            logger.debug("Feature importance:\n" + str(importance))

            # Calculate accuracy
            self.model_accuracy = self.price_model.score(X_test, y_test)
            
            # Save model and accuracy metrics
            joblib.dump(self.price_model, self.model_path)
            
            self.last_training_time = time.time()
            logger.debug(f"Model trained successfully. Accuracy: {self.model_accuracy:.4f}")

        except Exception as e:
            logger.error(f"Error training model: {e}")

    # Market Data Methods
    async def get_price_data(self, *args, **kwargs):
        """Use centralized price fetching from API_Config."""
        return await self.api_config.get_token_price_data(*args, **kwargs)

    async def get_token_volume(self, token_symbol: str) -> float:
        """
        Get the 24-hour trading volume for a given token symbol.

        :param token_symbol: Token symbol to fetch volume for
        :return: 24-hour trading volume
        """
        cache_key = f"token_volume_{token_symbol}"
        if cache_key in self.caches['volume']:
            logger.debug(f"Returning cached trading volume for {token_symbol}.")
            return self.caches['volume'][cache_key]

        volume = await self._fetch_from_services(
            lambda _: self.api_config.get_token_volume(token_symbol),
            f"trading volume for {token_symbol}"
        )
        if volume is not None:
            self.caches['volume'][cache_key] = volume
        return volume or 0.0

    async def _fetch_from_services(self, fetch_func: Callable[[str], Any], description: str) -> Optional[Union[List[float], float]]:
        """
        Helper method to fetch data from multiple services.

        :param fetch_func: Function to fetch data from a service
        :param description: Description of the data being fetched
        :return: Fetched data or None
        """
        for service in self.api_config.api_configs.keys():
            try:
                logger.debug(f"Fetching {description} using {service}...")
                result = await fetch_func(service)
                if result:
                    return result
            except Exception as e:
                logger.warning(f"failed to fetch {description} using {service}: {e}")
        logger.warning(f"failed to fetch {description}.")
        return None

    # Helper Methods
    def _calculate_volatility(
        self, 
        prices: List[float]
    ) -> float:
        """
        Calculate price volatility using standard deviation of returns.
        
        Args:
            prices: List of historical prices
            
        Returns:
            Volatility measure as float
        """
        prices_array = np.array(prices)
        returns = np.diff(prices_array) / prices_array[:-1]
        return np.std(returns)

    async def _update_price_model(self, token_symbol: str) -> None:
        """
        Update the price prediction model.

        :param token_symbol: Token symbol to update the model for
        """
        prices = await self.get_price_data(token_symbol, data_type='historical')
        if len(prices) > 10:
            X = np.arange(len(prices)).reshape(-1, 1)
            y = np.array(prices)
            self.price_model.fit(X, y)
            self.model_last_updated = time.time()

    async def is_arbitrage_opportunity(self, target_tx: Dict[str, Any]) -> bool:
        """Use transaction_core for decoding but avoid circular dependencies."""
        if not self.transaction_core:
            logger.warning("Transaction core not initialized, cannot check arbitrage")
            return False

        try:
            decoded_tx = await self.transaction_core.decode_transaction_input(
                target_tx["input"], target_tx["to"]
            )
             # Add logic here to check for arbitrage
            if not decoded_tx:
                logger.debug("Transaction input could not be decoded")
                return False
            
            if 'swap' in decoded_tx.get('function_name', '').lower():
                logger.debug("Transaction is a swap, might have arbitrage oppurtunity.")
                # Further check for price differences etc. can be implemented here
                return True

            return False
            
        except Exception as e:
            logger.error(f"Error checking arbitrage opportunity: {e}")
            return False

    async def _get_prices_from_services(self, token_symbol: str) -> List[float]:
        """
        Get real-time prices from different services.

        :param token_symbol: Token symbol to get prices for
        :return: List of real-time prices
        """
        prices = []
        for service in self.api_config.api_configs.keys():
            try:
                price = await self.api_config.get_real_time_price(token_symbol)
                if price is not None:
                    prices.append(price)
            except Exception as e:
                logger.warning(f"failed to get price from {service}: {e}")
        return prices

    async def stop(self) -> None:
        """Clean up resources and stop monitoring."""
        try:
            # Clear caches and clean up resources
            for cache in self.caches.values():
                cache.clear()
            logger.debug("Market Monitor stopped.")
        except Exception as e:
            logger.error(f"Error stopping Market Monitor: {e}")