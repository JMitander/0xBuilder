import time
import numpy as np
from sklearn.linear_model import LinearRegression
from cachetools import TTLCache
from typing import Any, Dict, List, Optional, Union, Tuple
from web3 import AsyncWeb3
import logging
from decimal import Decimal

from configuration.api_config import API_Config
from configuration.configuration import Configuration

logger = logging.getLogger(__name__)

# Add new cache settings
CACHE_SETTINGS = {
    'price': {'ttl': 300, 'size': 1000},
    'volume': {'ttl': 900, 'size': 500},
    'volatility': {'ttl': 600, 'size': 200}
}

class Market_Monitor:
    """Advanced market monitoring system for real-time analysis and prediction."""

    # Class Constants
    MODEL_UPDATE_INTERVAL: int = 3600  # Update model every hour
    VOLATILITY_THRESHOLD: float = 0.05  # 5% standard deviation
    LIQUIDITY_THRESHOLD: int = 100_000  # $100,000 in 24h volume

    def __init__(
        self,
        web3: AsyncWeb3,
        configuration: Optional[Configuration],
        api_config: Optional[API_Config],
        transaction_core: Optional[Any] = None,  # Add this parameter
    ) -> None:
        """Initialize Market Monitor with required components."""
        self.web3 = web3
        self.configuration = configuration
        self.api_config = api_config
        self.transaction_core = transaction_core  # Store transaction_core reference
        self.price_model = LinearRegression()
        self.model_last_updated = 0

        # Add separate caches for different data types
        self.caches = {
            'price': TTLCache(maxsize=CACHE_SETTINGS['price']['size'], 
                            ttl=CACHE_SETTINGS['price']['ttl']),
            'volume': TTLCache(maxsize=CACHE_SETTINGS['volume']['size'], 
                             ttl=CACHE_SETTINGS['volume']['ttl']),
            'volatility': TTLCache(maxsize=CACHE_SETTINGS['volatility']['size'], 
                                 ttl=CACHE_SETTINGS['volatility']['ttl'])
        }

    async def check_market_conditions(
        self, 
        token_address: str
    ) -> Dict[str, bool]:
        """
        Analyze current market conditions for a given token.
        
        Args:
            token_address: Token contract address
            
        Returns:
            Dictionary containing market condition indicators
        """
        market_conditions = {
            "high_volatility": False,
            "bullish_trend": False,
            "bearish_trend": False,
            "low_liquidity": False,
        }
        token_symbol = await self.api_config.get_token_symbol(self.web3, token_address)
        if not token_symbol:
            logger.debug(f"Cannot get token symbol for address {token_address}!")
            return market_conditions

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
        current_time = time.time()
        if current_time - self.model_last_updated > self.MODEL_UPDATE_INTERVAL:
            await self._update_price_model(token_symbol)
        prices = await self.get_price_data(token_symbol, data_type='historical', timeframe=1)
        if not prices:
            logger.debug(f"No recent prices available for {token_symbol}.")
            return 0.0
        next_time = np.array([[len(prices)]])
        predicted_price = self.price_model.predict(next_time)[0]
        logger.debug(f"Price prediction for {token_symbol}: {predicted_price}")
        return float(predicted_price)

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

    async def _fetch_from_services(self, fetch_func, description: str) -> Optional[Union[List[float], float]]:
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
            # ...existing code...
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

    async def decode_transaction_input(self, *args, **kwargs):
        """Use centralized transaction decoding from Transaction_Core."""
        return await self.transaction_core.decode_transaction_input(*args, **kwargs)

    async def initialize(self) -> None:
        """Initialize market monitor components."""
        try:
            self.price_model = LinearRegression()
            self.model_last_updated = 0
            logger.debug("Market Monitor initialized ✅")
        except Exception as e:
            logger.critical(f"Market Monitor initialization failed: {e}")
            raise

    async def stop(self) -> None:
        """Clean up resources and stop monitoring."""
        try:
            # Clear caches and clean up resources
            for cache in self.caches.values():
                cache.clear()
            logger.debug("Market Monitor stopped.")
        except Exception as e:
            logger.error(f"Error stopping Market Monitor: {e}")
