import asyncio
from pathlib import Path
import time
import logging
from typing import Any, Dict, List, Optional, Callable, Union
from decimal import Decimal

import aiohttp
from cachetools import TTLCache
import numpy as np
from web3 import AsyncWeb3

from python.abi_registry import ABI_Registry
from python.configuration import Configuration

logger = logging.getLogger("0xBuilder")


class API_Config:
    """
    Manages interactions with various external APIs for price and market data.
    """

    MAX_REQUEST_ATTEMPTS: int = 5
    REQUEST_BACKOFF_FACTOR: float = 1.5

    def __init__(self, configuration: "Configuration"):
        """
        Initialize API configurations and caches.

        Args:
            configuration: Configuration object containing API keys and settings.
        """
        self.configuration: "Configuration" = configuration
        self.session: Optional[aiohttp.ClientSession] = None
        self.price_cache: TTLCache = TTLCache(maxsize=2000, ttl=300)  # 5 min cache for prices
        self.volume_cache: TTLCache = TTLCache(maxsize=1000, ttl=900)  # 15 min cache for volumes
        self.market_data_cache: TTLCache = TTLCache(maxsize=1000, ttl=1800)  # 30 min cache for market data
        self.token_metadata_cache: TTLCache = TTLCache(maxsize=500, ttl=86400)  # 24h cache for metadata

        # Rate limit tracking
        self.rate_limit_counters: Dict[str, Dict[str, Any]] = {
            "coingecko": {"count": 0, "reset_time": time.time(), "limit": 50},
            "coinmarketcap": {"count": 0, "reset_time": time.time(), "limit": 330},
            "cryptocompare": {"count": 0, "reset_time": time.time(), "limit": 80},
            "binance": {"count": 0, "reset_time": time.time(), "limit": 1200},
        }

        # Priority queues for data fetching
        self.high_priority_tokens: set[str] = set()  # Tokens currently being traded
        self.update_intervals: Dict[str, int] = {
            'price': 30,  # Seconds
            'volume': 300,  # 5 minutes
            'market_data': 1800,  # 30 minutes
            'metadata': 86400  # 24 hours
        }

        # API lock and session
        self.api_lock: asyncio.Lock = asyncio.Lock()
        self.session: Optional[aiohttp.ClientSession] = None

        # API configurations
        self.api_configs: Dict[str, Dict[str, Any]] = {
            "binance": {
                "base_url": "https://api.binance.com/api/v3",
                "market_url": "/ticker/24hr",
                "success_rate": 1.0,
                "weight": 1.0,
                "rate_limit": 1200,
            },
            "coingecko": {
                "base_url": "https://api.coingecko.com/api/v3",
                "market_url": "/coins/{id}/market_chart",
                "volume_url": "/coins/{id}",
                "api_key": configuration.COINGECKO_API_KEY,
                "success_rate": 1.0,
                "weight": 0.8,
                "rate_limit": 50,
            },
            "coinmarketcap": {
                "base_url": "https://pro-api.coinmarketcap.com/v1",
                "ticker_url": "/cryptocurrency/quotes/latest",
                "api_key": configuration.COINMARKETCAP_API_KEY,
                "success_rate": 1.0,
                "weight": 0.7,
                "rate_limit": 333,
            },
            "cryptocompare": {
                "base_url": "https://min-api.cryptocompare.com/data",
                "price_url": "/price",
                "api_key": configuration.CRYPTOCOMPARE_API_KEY,
                "success_rate": 1.0,
                "weight": 0.6,
                "rate_limit": 80,
            },
        }

        # Initialize rate limiters after API configs
        self.rate_limiters: Dict[str, asyncio.Semaphore] = {
            provider: asyncio.Semaphore(config.get("rate_limit", 10))
            for provider, config in self.api_configs.items()
        }
    
    async def __aenter__(self) -> "API_Config":
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.session:
            await self.session.close()
            logger.debug("APIConfig session closed.")

    async def initialize(self) -> None:
        """Initialize API configuration."""
        try:
            self.session = aiohttp.ClientSession()
            logger.info("API_Config initialized ✅")
        except Exception as e:
            logger.critical(f"API_Config initialization failed: {e}")
            raise

    async def close(self) -> None:
        """Close the aiohttp session."""
        if self.session:
            await self.session.close()

    async def get_token_symbol(self, web3: "AsyncWeb3", token_address: str) -> Optional[str]:
        """Get the token symbol for a given token address, using cache or contract."""
        if token_address in self.token_metadata_cache:
            metadata = self.token_metadata_cache[token_address]
            return metadata.get('symbol')

        try:
            erc20_abi =  await self._load_abi() # Load ERC20 abi from the registry
            contract = web3.eth.contract(address=token_address, abi=erc20_abi)
            symbol = await contract.functions.symbol().call()
            self.token_metadata_cache[token_address] = {'symbol': symbol}
            return symbol
        except Exception as e:
            logger.error(f"Error getting symbol for token {token_address}: {e}")
            return None

    async def get_token_metadata(self, token: str) -> Optional[Dict[str, Any]]:
         """Get metadata for a given token symbol."""
         if token in self.token_metadata_cache:
             return self.token_metadata_cache[token]
         metadata = await self._fetch_from_services(
             lambda service: self._fetch_token_metadata(service, token),
             f"metadata for {token}",
         )
         if metadata:
             self.token_metadata_cache[token] = metadata
         return metadata

    async def _fetch_token_metadata(self, source: str, token: str) -> Optional[Dict[str, Any]]:
        """Fetch token metadata from a specific API source."""
        config = self.api_configs.get(source)
        if not config:
            return None

        try:
            if source == "coingecko":
                token_id = token.lower()
                url = f"{config['base_url']}/coins/{token_id}"
                headers = {"x-cg-pro-api-key": config['api_key']} if config['api_key'] else None

                response = await self.make_request(source, url, headers=headers)
                if response:
                    return {
                        'symbol': response.get('symbol', ''),
                        'market_cap': response.get('market_data', {}).get('market_cap', {}).get('usd', 0),
                        'total_supply': response.get('market_data', {}).get('total_supply', 0),
                        'circulating_supply': response.get('market_data', {}).get('circulating_supply', 0),
                        'trading_pairs': len(response.get('tickers', [])),
                        'exchanges': list(set(t.get('market', {}).get('name') for t in response.get('tickers', [])))
                    }

            elif source == "coinmarketcap":
                url = f"{config['base_url']}/cryptocurrency/quotes/latest"
                headers = {
                    "X-CMC_PRO_API_KEY": config['api_key'],
                    "Accept": "application/json"
                }
                params = {"symbol": token.upper()}

                response = await self.make_request(source, url, params=params, headers=headers)
                if response and 'data' in response:
                    data = response['data'].get(token.upper(), {})
                    return {
                        'symbol': data.get('symbol', ''),
                        'market_cap': data.get('quote', {}).get('USD', {}).get('market_cap', 0),
                        'total_supply': data.get('total_supply', 0),
                        'circulating_supply': data.get('circulating_supply', 0),
                        'trading_pairs': len(data.get('tags', [])),
                        'exchanges': []  # CMC doesn't provide exchange list in basic endpoint
                    }

            return None

        except Exception as e:
            logger.error(f"Error fetching metadata from {source}: {e}")
            return None

    async def get_real_time_price(self, token: str, vs_currency: str = "eth") -> Optional[Decimal]:
        """Get real-time price using weighted average from multiple sources."""
        cache_key = f"price_{token}_{vs_currency}"
        if cache_key in self.price_cache:
            return self.price_cache[cache_key]
        prices = []
        weights = []
        async with self.api_lock:
            for source, config in self.api_configs.items():
                try:
                    price = await self._fetch_price(source, token, vs_currency)
                    if price:
                        prices.append(price)
                        weights.append(config["weight"] * config["success_rate"])
                except Exception as e:
                    logger.error(f"Error fetching price from {source}: {e}")
                    config["success_rate"] *= 0.9
        if not prices:
            logger.warning(f"No valid prices found for {token}!")
            return None
        weighted_price = sum(p * w for p, w in zip(prices, weights)) / sum(weights)
        self.price_cache[cache_key] = Decimal(str(weighted_price))
        return self.price_cache[cache_key]

    async def _fetch_price(self, source: str, token: str, vs_currency: str) -> Optional[Decimal]:
        """Fetch the price of a token from a specified source."""
        config = self.api_configs.get(source)
        if not config:
            logger.error(f"API source {source} not configured.")
            return None

        try:
            async with self.session.get(config["base_url"] + f"/simple/price?ids={token}&vs_currencies={vs_currency}") as response:
                if response.status == 200:
                    data = await response.json()
                    if token in data and vs_currency in data[token]:
                        price = Decimal(str(data[token][vs_currency]))
                        logger.debug(f"Fetched price from {source}: {price}")
                        return price
                    else:
                        logger.error(f"Invalid price data format from {source}: {data}")
                        return None
                else:
                    logger.error(f"Failed to fetch price from {source}: Status {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Exception fetching price from {source}: {e}")
            return None

    async def make_request(
        self,
        provider_name: str,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        """Make HTTP request with improved error handling and timeout management."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()

        rate_limiter = self.rate_limiters.get(provider_name)
        if rate_limiter is None:
            logger.error(f"No rate limiter for provider {provider_name}")
            return None

        async with rate_limiter:
            for attempt in range(self.MAX_REQUEST_ATTEMPTS):
                try:
                    # More conservative timeout settings
                    timeout = aiohttp.ClientTimeout(
                        total=30,  # Total timeout
                        connect=10,  # Connection timeout
                        sock_read=10  # Socket read timeout
                    )

                    async with self.session.get(
                        url,
                        params=params,
                        headers=headers,
                        timeout=timeout
                    ) as response:
                        if response.status == 429:  # Rate limit
                            wait_time = self.REQUEST_BACKOFF_FACTOR ** attempt
                            logger.warning(f"Rate limit for {provider_name}, waiting {wait_time}s")
                            await asyncio.sleep(wait_time)
                            continue

                        if response.status >= 400:
                            logger.warning(f"Error {response.status} from {provider_name}")
                            if attempt == self.MAX_REQUEST_ATTEMPTS - 1:
                                return None
                            continue

                        return await response.json()

                except asyncio.TimeoutError:
                    logger.warning(f"Timeout for {provider_name} (attempt {attempt + 1})")
                    if attempt == self.MAX_REQUEST_ATTEMPTS - 1:
                        return None
                except Exception as e:
                    logger.error(f"Error fetching from {provider_name}: {e}")
                    if attempt == self.MAX_REQUEST_ATTEMPTS - 1:
                        return None

                await asyncio.sleep(self.REQUEST_BACKOFF_FACTOR ** attempt)

            return None

    async def fetch_historical_prices(self, token: str, days: int = 30) -> List[float]:
        """Fetch historical price data for a given token symbol."""
        cache_key = f"historical_prices_{token}_{days}"
        if cache_key in self.price_cache:
            logger.debug(f"Returning cached historical prices for {token}.")
            return self.price_cache[cache_key]
        prices = await self._fetch_from_services(
            lambda service: self._fetch_historical_prices(service, token, days),
            f"historical prices for {token}",
        )
        if prices:
            self.price_cache[cache_key] = prices
        return prices or []

    async def _fetch_historical_prices(self, source: str, token: str, days: int) -> Optional[List[float]]:
        """Fetch historical prices from a specified source."""
        config = self.api_configs.get(source)
        if not config:
            logger.error(f"API source {source} not configured.")
            return None

        try:
            url = f"{config['base_url']}/coins/{token}/market_chart?vs_currency=eth&days={days}"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    prices = [price[1] for price in data.get("prices", [])]
                    logger.debug(f"Fetched historical prices from {source}: {prices}")
                    return prices
                else:
                    logger.error(f"Failed to fetch historical prices from {source}: Status {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Exception fetching historical prices from {source}: {e}")
            return None

    async def get_token_volume(self, token: str) -> float:
        """Get the 24-hour trading volume for a given token symbol."""
        cache_key = f"token_volume_{token}"
        if cache_key in self.volume_cache:
            logger.debug(f"Returning cached trading volume for {token}.")
            return self.volume_cache[cache_key]
        volume = await self._fetch_from_services(
            lambda service: self._fetch_token_volume(service, token),
            f"trading volume for {token}",
        )
        if volume is not None:
            self.volume_cache[cache_key] = volume
        return volume or 0.0

    async def _fetch_token_volume(self, source: str, token: str) -> Optional[float]:
        """Enhanced volume fetching with better error handling."""
        config = self.api_configs.get(source)
        if not config:
            return None

        try:
            if source == "binance":
                symbols = await self._get_trading_pairs(token)
                if not symbols:
                    return None
                for symbol in symbols:
                    try:
                        url = f"{config['base_url']}/ticker/24hr"
                        params = {"symbol": symbol}
                        response = await self.make_request(source, url, params=params)
                        if response and 'volume' in response:
                            return float(response['quoteVolume'])
                    except Exception:
                        continue

            elif source == "coingecko":
                token_id = token.lower()
                url = f"{config['base_url']}/simple/price"
                params = {
                    "ids": token_id,
                    "vs_currencies": "usd",
                    "include_24hr_vol": "true"
                }
                if config['api_key']:
                    params['x_cg_pro_api_key'] = config['api_key']

                response = await self.make_request(source, url, params=params)
                if response and token_id in response:
                    return float(response[token_id].get('usd_24h_vol', 0))

            elif source == "coinmarketcap":
                url = f"{config['base_url']}/cryptocurrency/quotes/latest"
                headers = {"X-CMC_PRO_API_KEY": config['api_key']}
                params = {"symbol": token.upper()}
                
                response = await self.make_request(source, url, params=params, headers=headers)
                if response and 'data' in response:
                    token_data = response['data'].get(token.upper(), {})
                    return float(token_data.get('quote', {}).get('USD', {}).get('volume_24h', 0))

            return None

        except Exception as e:
            logger.error(f"Error fetching volume from {source}: {e}")
            return None

    async def _get_trading_pairs(self, token: str) -> List[str]:
        """Get valid trading pairs for a token."""
        quote_currencies = ["USDT", "BUSD", "USD", "ETH", "BTC"]
        symbol_mappings = {
            "WETH": ["ETH"],
            "WBTC": ["BTC"],
            "ETH": ["ETH"],
            "BTC": ["BTC"],
        }

        base_symbols = symbol_mappings.get(token, [token])
        pairs = []

        for base in base_symbols:
            pairs.extend([f"{base}{quote}" for quote in quote_currencies])

        return pairs

    async def _fetch_from_services(self, fetch_func: Callable[[str], Any], description: str) -> Optional[Union[List[float], float]]:
         """Helper method to fetch data from multiple services."""
         for service in self.api_configs.keys():
             try:
                 logger.debug(f"Fetching {description} using {service}...")
                 result = await fetch_func(service)
                 if result:
                     return result
             except Exception as e:
                 logger.warning(f"failed to fetch {description} using {service}: {e}")
         logger.warning(f"failed to fetch {description}.")
         return None
    
    async def _load_abi(self) -> List[Dict[str, Any]]:
        """Load contract abi from a file."""
        try:
            abi_registry = ABI_Registry()
            abi = await abi_registry.load_abi('erc20')
            if not abi:
                 raise ValueError("Failed to load ERC20 ABI using ABI Registry")
            return abi
        except Exception as e:
            logger.error(f"Failed to load abi: {e}")
            raise
    
    async def get_token_price_data(
        self,
        token_symbol: str,
        data_type: str = 'current',
        timeframe: int = 1,
        vs_currency: str = 'eth'
    ) -> Union[float, List[float]]:
        """Centralized price data fetching for all components."""
        cache_key = f"{data_type}_{token_symbol}_{timeframe}_{vs_currency}"
        
        if cache_key in self.price_cache:
            return self.price_cache[cache_key]
            
        try:
            if data_type == 'current':
                data = await self.get_real_time_price(token_symbol, vs_currency)
            elif data_type == 'historical':
                data = await self.fetch_historical_prices(token_symbol, days=timeframe)
            else:
                raise ValueError(f"Invalid data type: {data_type}")
                
            if data is not None:
                self.price_cache[cache_key] = data
            return data
            
        except Exception as e:
            logger.error(f"Error fetching {data_type} price data: {e}")
            return [] if data_type == 'historical' else 0.0

    async def _fetch_with_priority(self, token: str, data_type: str) -> Optional[Any]:
        """Fetch data with priority-based rate limiting."""
        try:
           
            providers = list(self.api_configs.keys())
            
            # Try each provider until we get data
            for provider in providers:
                try:
                    data = await self._fetch_from_provider(provider, token, data_type)
                    if data:
                        return data
                except Exception as e:
                    logger.debug(f"Error fetching from {provider}: {e}")
                    continue
            
            return None
            
        except Exception as e:
            logger.error(f"Error in priority fetch: {e}")
            return None
        
    async def _fetch_from_provider(self, provider: str, token: str, data_type: str) -> Optional[Any]:
        """Fetch data from specific provider with better error handling."""
        try:
            config = self.api_configs.get(provider)
            if not config:
                return None

            if data_type == 'price':
                return await self._fetch_price(provider, token, 'eth')
            elif data_type == 'volume':
                return await self._fetch_token_volume(provider, token)
            elif data_type == 'metadata':
                return await self._fetch_token_metadata(provider, token)
            else:
                logger.warning(f"Unsupported data type: {data_type}")
                return None

        except Exception as e:
            logger.error(f"Error fetching from {provider}: {e}")
            return None

    def _calculate_volatility(self, price_history: List[float]) -> float:
        """Calculate price volatility using standard deviation of returns."""
        if not price_history or len(price_history) < 2:
            return 0.0
        
        try:
            returns = [
                (price_history[i] - price_history[i-1]) / price_history[i-1]
                for i in range(1, len(price_history))
            ]
            return float(np.std(returns))
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return 0.0

    def _calculate_momentum(self, price_history: List[float]) -> float:
        """Calculate price momentum using exponential moving average."""
        if not price_history or len(price_history) < 2:
            return 0.0
        
        try:
            # Calculate short and long term EMAs
            short_period = min(12, len(price_history))
            long_period = min(26, len(price_history))
            
            ema_short = sum(price_history[-short_period:]) / short_period
            ema_long = sum(price_history[-long_period:]) / long_period
            
            momentum = (ema_short / ema_long - 1) if ema_long > 0 else 0
            return float(momentum)
            
        except Exception as e:
            logger.error(f"Error calculating momentum: {e}")
            return 0.0

    async def _gather_training_data(self, token: str) -> Optional[Dict[str, Any]]:
        """Gather all required data for model training."""
        try:
            # Gather data concurrently
            price, volume, market_data = await asyncio.gather(
                self.get_real_time_price(token),
                self.get_token_volume(token),
                self._fetch_market_data(token),
                return_exceptions=True
            )

            # Handle any exceptions
            results = [price, volume, market_data]
            if any(isinstance(r, Exception) for r in results):
                logger.warning(f"Error gathering data for {token}")
                return None

            # Combine all data
            return {
                'timestamp': int(time.time()),
                'symbol': token,
                'price_usd': float(price),
                'volume_24h': float(volume),
                **market_data
            }

        except Exception as e:
            logger.error(f"Error gathering training data: {e}")
            return None

    async def _fetch_market_data(self, token: str) -> Optional[Dict[str, Any]]:
        """Fetch comprehensive market data for a token."""
        try:
            # Cache check
            cache_key = f"market_data_{token}"
            if cache_key in self.market_data_cache:
                return self.market_data_cache[cache_key]

            # Gather data concurrently using asyncio.gather
            data_tasks = [
                self.get_token_metadata(token),
                self.get_token_volume(token),
                self.get_token_price_data(token, 'historical', timeframe=7)  # 7 day price history
            ]
            
            metadata, volume, price_history = await asyncio.gather(*data_tasks, return_exceptions=True)

            # Check for exceptions in results
            results = [metadata, volume, price_history]
            if any(isinstance(r, Exception) for r in results):
                logger.warning(f"Some market data fetching failed for {token}")
                return None

            # Calculate additional metrics
            price_volatility = self._calculate_volatility(price_history) if price_history else 0
            market_data = {
                'market_cap': metadata.get('market_cap', 0) if metadata else 0,
                'volume_24h': float(volume) if volume else 0,
                'percent_change_24h': metadata.get('price_change_24h', 0) if metadata else 0,
                'total_supply': metadata.get('total_supply', 0) if metadata else 0,
                'circulating_supply': metadata.get('circulating_supply', 0) if metadata else 0,
                'volatility': price_volatility,
                'price_momentum': self._calculate_momentum(price_history) if price_history else 0,
                'liquidity_ratio': await self._calculate_liquidity_ratio(token),
                'trading_pairs': len(metadata.get('trading_pairs', [])) if metadata else 0,
                'exchange_count': len(metadata.get('exchanges', [])) if metadata else 0
            }

            # Cache the results
            self.market_data_cache[cache_key] = market_data
            return market_data

        except Exception as e:
            logger.error(f"Error fetching market data for {token}: {e}")
            return None
    
    async def _calculate_liquidity_ratio(self, token: str) -> float:
        """Calculate liquidity ratio using market cap and volume from API config."""
        try:
            volume = await self.get_token_volume(token)
            metadata = await self.get_token_metadata(token)
            market_cap = metadata.get('market_cap', 0) if metadata else 0
            return volume / market_cap if market_cap > 0 else 0.0
        except Exception as e:
            logger.error(f"Error calculating liquidity ratio: {e}")
            return 0.0
    
    async def get_token_supply_data(self, token: str) -> Dict[str, Any]:
            """Gets total and circulating supply for a given token."""
            metadata = await self.get_token_metadata(token)
            if not metadata:
                return {}
            return {
                'total_supply': metadata.get('total_supply', 0),
                'circulating_supply': metadata.get('circulating_supply', 0)
            }

    async def get_token_market_cap(self, token: str) -> float:
        """Gets token market cap."""
        metadata = await self.get_token_metadata(token)
        return metadata.get('market_cap', 0) if metadata else 0

    async def get_price_change_24h(self, token: str) -> float:
        """Gets price change in the last 24h."""
        metadata = await self.get_token_metadata(token)
        return metadata.get('percent_change_24h', 0) if metadata else 0

    async def update_training_data(self) -> None:
        """
        Updates training data for model training by fetching data for all monitored tokens.
        """
        try:
            token_addresses = await self.configuration.get_token_addresses()

             # Gather all training data concurrently
            update_tasks = [self._gather_training_data(token) for token in token_addresses]
            updates = await asyncio.gather(*update_tasks, return_exceptions=True)

            # Filter out failed updates
            valid_updates = [update for update in updates if isinstance(update, dict)]

            if valid_updates:
                await self._write_training_data(valid_updates)

        except Exception as e:
             logger.error(f"Error updating training data: {e}")

    async def _write_training_data(self, updates: List[Dict[str, Any]]) -> None:
        """Write updates to training data file."""
        try:
            import pandas as pd
            from io import StringIO
            import aiofiles

            df = pd.DataFrame(updates)
            training_data_path = Path(self.configuration.TRAINING_DATA_PATH)
            
            # Read existing data, append new data, write back to CSV
            if training_data_path.exists():
                async with aiofiles.open(training_data_path, 'r') as f:
                    old_data = await f.read()
                
                if old_data:
                    df_old = pd.read_csv(StringIO(old_data))
                    df = pd.concat([df_old, df], ignore_index=True)

            async with aiofiles.open(training_data_path, 'w', encoding='utf-8') as file:
                await file.write(df.to_csv(index=False))


            # Keep file size manageable (keep last 30 days)
            await self._cleanup_old_data(training_data_path, days=30)
            
        except Exception as e:
            logger.error(f"Error writing training data: {e}")
            
    async def _cleanup_old_data(self, filepath: Path, days: int) -> None:
        """Remove data older than specified days."""
        try:
            import pandas as pd
            from io import StringIO
            import aiofiles
            
            async with aiofiles.open(filepath, 'r') as f:
                content = await f.read()
            if content:
                df = pd.read_csv(StringIO(content))
                cutoff_time = int(time.time()) - (days * 86400)
                df = df[df['timestamp'] >= cutoff_time]
                async with aiofiles.open(filepath, 'w', encoding='utf-8') as file:
                  await file.write(df.to_csv(index=False))
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")