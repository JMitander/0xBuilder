import asyncio
import json
import logging
import os
import time
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import aiofiles
import aiohttp
from web3 import AsyncWeb3
from cachetools import TTLCache

logger = logging.getLogger(__name__)


class Configuration:
    """
    Loads configuration from environment variables and monitored tokens from a JSON file.
    """

    def __init__(self):
        """Initialize configuration attributes with None values."""
        self.IPC_ENDPOINT = None
        self.HTTP_ENDPOINT = None
        self.WEBSOCKET_ENDPOINT = None
        self.WALLET_KEY = None
        self.WALLET_ADDRESS = None
        self.ETHERSCAN_API_KEY = None
        self.INFURA_PROJECT_ID = None
        self.COINGECKO_API_KEY = None
        self.COINMARKETCAP_API_KEY = None
        self.CRYPTOCOMPARE_API_KEY = None
        self.AAVE_LENDING_POOL_ADDRESS = None
        self.TOKEN_ADDRESSES = None
        self.TOKEN_SYMBOLS = None
        self.ERC20_ABI = None
        self.ERC20_SIGNATURES = None
        self.SUSHISWAP_ROUTER_ABI = None
        self.SUSHISWAP_ROUTER_ADDRESS = None
        self.UNISWAP_ROUTER_ABI = None
        self.UNISWAP_ROUTER_ADDRESS = None
        self.AAVE_FLASHLOAN_ABI = None
        self.AAVE_LENDING_POOL_ABI = None
        self.AAVE_FLASHLOAN_ADDRESS = None
        self.PANCAKESWAP_ROUTER_ABI = None
        self.PANCAKESWAP_ROUTER_ADDRESS = None
        self.BALANCER_ROUTER_ABI = None
        self.BALANCER_ROUTER_ADDRESS = None
        


    async def load(self) -> None:
        """Loads the configuration in the correct order."""
        try:
            logger.info("Loading configuration... ⏳")
            time.sleep(1) # ensuring proper initialization
            await self._load_configuration()
            logger.info ("System reporting go for launch ✅...")
            time.sleep(3) # ensuring proper initialization
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise

    async def _load_configuration(self) -> None:
        """Load configuration in the correct order."""
        try:
            # First load providers and account
            self._load_providers_and_account()
            
            # Then load API keys
            self._load_api_keys()
            
            # Finally load JSON elements
            await self._load_json_elements()
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise

    def _load_api_keys(self) -> None:
        self.ETHERSCAN_API_KEY = self._get_env_variable("ETHERSCAN_API_KEY")
        self.INFURA_PROJECT_ID = self._get_env_variable("INFURA_PROJECT_ID")
        self.COINGECKO_API_KEY = self._get_env_variable("COINGECKO_API_KEY")
        self.COINMARKETCAP_API_KEY = self._get_env_variable("COINMARKETCAP_API_KEY")
        self.CRYPTOCOMPARE_API_KEY = self._get_env_variable("CRYPTOCOMPARE_API_KEY")

    def _load_providers_and_account(self) -> None:
        """Load provider endpoints and account information."""
        try:
            self.IPC_ENDPOINT = self._get_env_variable("IPC_ENDPOINT", default=None)
            self.HTTP_ENDPOINT = self._get_env_variable("HTTP_ENDPOINT", default=None)
            self.WEBSOCKET_ENDPOINT = self._get_env_variable("WEBSOCKET_ENDPOINT", default=None)
            
            # Ensure at least one endpoint is configured
            if not any([self.IPC_ENDPOINT, self.HTTP_ENDPOINT, self.WEBSOCKET_ENDPOINT]):
                raise ValueError("At least one endpoint (IPC, HTTP, or WebSocket) must be configured")

            self.WALLET_KEY = self._get_env_variable("WALLET_KEY")
            self.WALLET_ADDRESS = self._get_env_variable("WALLET_ADDRESS")

            logger.info("Providers OK ✅")
            logger.info("Account OK ✅")
            logger.info("API Keys OK ✅")
            
        except Exception as e:
            logger.error(f"Error loading providers and account: {e}")
            raise

    async def _load_json_elements(self) -> None: 
        try:
            self.AAVE_LENDING_POOL_ADDRESS = self._get_env_variable("AAVE_LENDING_POOL_ADDRESS")
            self.TOKEN_ADDRESSES = await self._load_json_file(
                self._get_env_variable("TOKEN_ADDRESSES"), "monitored tokens"
            )
            self.TOKEN_SYMBOLS = await self._load_json_file(
                self._get_env_variable("TOKEN_SYMBOLS"), "token symbols"
            )
            self.ERC20_ABI = await self._construct_abi_path("abi", "erc20_abi.json")
            self.ERC20_SIGNATURES = await self._load_json_file(
                self._get_env_variable("ERC20_SIGNATURES"), "ERC20 function signatures"
            )
            self.SUSHISWAP_ROUTER_ABI = await self._construct_abi_path("abi", "sushiswap_router_abi.json")
            self.SUSHISWAP_ROUTER_ADDRESS = self._get_env_variable("SUSHISWAP_ROUTER_ADDRESS")
            self.UNISWAP_ROUTER_ABI = await self._construct_abi_path("abi", "uniswap_router_abi.json")
            self.UNISWAP_ROUTER_ADDRESS = self._get_env_variable("UNISWAP_ROUTER_ADDRESS")
            self.AAVE_FLASHLOAN_ABI = await self._construct_abi_path("abi", "aave_flashloan_abi.json")
            self.AAVE_LENDING_POOL_ABI = await self._construct_abi_path("abi", "aave_lending_pool_abi.json")
            self.AAVE_FLASHLOAN_ADDRESS = self._get_env_variable("AAVE_FLASHLOAN_ADDRESS")
            self.PANCAKESWAP_ROUTER_ABI = await self._construct_abi_path("abi", "pancakeswap_router_abi.json")
            self.PANCAKESWAP_ROUTER_ADDRESS = self._get_env_variable("PANCAKESWAP_ROUTER_ADDRESS")
            self.BALANCER_ROUTER_ABI = await self._construct_abi_path("abi", "balancer_router_abi.json")
            self.BALANCER_ROUTER_ADDRESS = self._get_env_variable("BALANCER_ROUTER_ADDRESS")
            logger.debug("JSON elements loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading JSON elements: {e}")
            raise

    def _get_env_variable(self, var_name: str, default: Optional[str] = None) -> str:
        value = os.getenv(var_name, default)
        if value is None:
            raise EnvironmentError(f"Missing environment variable: {var_name}")
        return value

    async def _load_json_file(self, file_path: str, description: str) -> Any:
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                logger.debug(f"Successfully loaded {description} from {file_path}")
                return data
        except FileNotFoundError as e:
            logger.error(f"File not found for {description}: {file_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for {description} in file {file_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading {description} from {file_path}: {e}")
            raise

    async def _construct_abi_path(self, base_path: str, abi_filename: str) -> str:
        abi_path = os.path.join(base_path, abi_filename)
        if not os.path.exists(abi_path):
            logger.error(f"ABI file does not exist: {abi_path}")
            raise FileNotFoundError(f"ABI file not found: {abi_path}")
        logger.debug(f"ABI path constructed: {abi_path}")
        return abi_path

    async def get_token_addresses(self) -> List[str]:
        return self.TOKEN_ADDRESSES

    async def get_token_symbols(self) -> Dict[str, str]:
        return self.TOKEN_SYMBOLS

    def get_abi_path(self, abi_name: str) -> str:
        abi_paths = {
            "erc20_abi": self.ERC20_ABI,
            "sushiswap_router_abi": self.SUSHISWAP_ROUTER_ABI,
            "uniswap_router_abi": self.UNISWAP_ROUTER_ABI,
            "AAVE_FLASHLOAN_ABI": self.AAVE_FLASHLOAN_ABI,
            "AAVE_LENDING_POOL_ABI": self.AAVE_LENDING_POOL_ABI,
            "pancakeswap_router_abi": self.PANCAKESWAP_ROUTER_ABI,
            "balancer_router_abi": self.BALANCER_ROUTER_ABI,
        }
        return abi_paths.get(abi_name.lower(), "")

    async def initialize(self) -> None:
        """Initialize configuration with error handling."""
        try:
            await self._load_configuration()
            logger.debug("Configuration initialized successfully.")
        except Exception as e:
            logger.critical(f"Configuration initialization failed: {e}")
            raise

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Safe configuration value access with default."""
        try:
            return getattr(self, key, default)
        except AttributeError:
            logger.warning(f"Configuration key '{key}' not found, using default: {default}")
            return default
    logger.debug("All Configurations and Environment Variables Loaded Successfully ✅") 

class ABI_Manager:
    """Centralized ABI management system."""

    def __init__(self):
        self.cached_abis: Dict[str, List[Dict[str, Any]]] = {}
        self.abi_paths = {
            'uniswap': 'uniswap_router_abi.json',
            'sushiswap': 'sushiswap_router_abi.json',
            'pancakeswap': 'pancakeswap_router_abi.json',
            'balancer': 'balancer_router_abi.json',
            'erc20': 'erc20_abi.json'
        }

    async def load_abi(self, abi_name: str) -> Optional[List[Dict[str, Any]]]:
        """Load ABI with validation and caching."""
        if abi_name in self.cached_abis:
            return self.cached_abis[abi_name]

        try:
            base_path = Path(__file__).parent.parent.parent / 'abi'
            abi_path = base_path / self.abi_paths.get(abi_name, f"{abi_name}_abi.json")
            
            if not abi_path.exists():
                logger.error(f"ABI file not found: {abi_path}")
                return None

            async with aiofiles.open(abi_path, 'r') as f:
                content = await f.read()
                abi = json.loads(content)

            if not isinstance(abi, list):
                logger.error(f"Invalid ABI format in {abi_path}")
                return None

            self.cached_abis[abi_name] = abi
            logger.debug(f"Successfully loaded {abi_name} ABI")
            return abi

        except Exception as e:
            logger.error(f"Error loading {abi_name} ABI: {e}")
            return None

    def validate_abi(self, abi: List[Dict[str, Any]], required_methods: List[str]) -> bool:
        """Validate ABI contains required methods."""
        try:
            found_methods = {func['name'] for func in abi if 'name' in func}
            missing = set(required_methods) - found_methods
            if missing:
                logger.error(f"ABI missing required methods: {missing}")
                return False
            return True
        except Exception as e:
            logger.error(f"ABI validation error: {e}")
            return False

class API_Config:
    def __init__(self, configuration: Optional[Configuration] = None):
        self.configuration = configuration
        self.session = None 
        self.price_cache = TTLCache(maxsize=1000, ttl=300)  # Cache for 5 minutes
        self.token_symbol_cache = TTLCache(maxsize=1000, ttl=86400)  # Cache for 1 day

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            logger.debug("APIconfig session closed.")

        # API configuration
        self.api_configs = {
            "binance": {
                "base_url": "https://api.binance.com/api/v3",
                "success_rate": 1.0,
                "weight": 1.0,
                "rate_limit": 1200,  # Max requests per minute
            },
            "coingecko": {
                "base_url": "https://api.coingecko.com/api/v3",
                "api_key": self.configuration.COINGECKO_API_KEY,
                "success_rate": 1.0,
                "weight": 0.8,
                "rate_limit": 50,  # Max requests per minute
            },
            "coinmarketcap": {
                "base_url": "https://pro-api.coinmarketcap.com/v1",
                "api_key": self.configuration.COINMARKETCAP_API_KEY,
                "success_rate": 1.0,
                "weight": 0.7,
                "rate_limit": 333,  # Max requests per minute
            },
            "cryptocompare": {
                "base_url": "https://min-api.cryptocompare.com/data",
                "api_key": self.configuration.CRYPTOCOMPARE_API_KEY,
                "success_rate": 1.0,
                "weight": 0.6,
                "rate_limit": 80,  # Max requests per minute
            },
        }

        self.api_lock = asyncio.Lock()
        self.rate_limiters = {
            provider: asyncio.Semaphore(config.get("rate_limit", 10))
            for provider, config in self.api_configs.items()
        }

    async def get_token_symbol(self, web3: AsyncWeb3, token_address: str) -> Optional[str]:
        """Get the token symbol for a given token address."""
        if token_address in self.token_symbol_cache:
            return self.token_symbol_cache[token_address]
        if token_address in self.configuration.TOKEN_SYMBOLS:
            symbol = self.configuration.TOKEN_SYMBOLS[token_address]
            self.token_symbol_cache[token_address] = symbol
            return symbol
        try:
            erc20_abi = await self._load_abi(self.configuration.ERC20_ABI)
            contract = web3.eth.contract(address=token_address, abi=erc20_abi)
            symbol = await contract.functions.symbol().call()
            self.token_symbol_cache[token_address] = symbol
            return symbol
        except Exception as e:
            logger.error(f"Error getting symbol for token {token_address}: {e}")
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
                    price = Decimal(str(data[token][vs_currency]))
                    logger.debug(f"Fetched price from {source}: {price}")
                    return price
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
        max_attempts: int = 5,
        backoff_factor: float = 1.5,
    ) -> Any:
        """Make HTTP request with exponential backoff and rate limit per provider."""
        rate_limiter = self.rate_limiters.get(provider_name)
        
        if rate_limiter is None:
            logger.error(f"No rate limiter for provider {provider_name}")
            return None
        
        async with rate_limiter:
            for attempt in range(max_attempts):
                try:
                    timeout = aiohttp.ClientTimeout(total=10 * (attempt + 1))
                    async with self.session.get(url, params=params, headers=headers, timeout=timeout) as response:
                        if response.status == 429:
                            wait_time = backoff_factor ** attempt
                            logger.warning(f"Rate limit exceeded for {provider_name}, retrying in {wait_time}s...")
                            await asyncio.sleep(wait_time)
                            continue
                        response.raise_for_status()
                        return await response.json()

                except aiohttp.ClientResponseError as e:
                    if attempt == max_attempts - 1:
                        logger.error(f"Request failed after {max_attempts} attempts: {e}")
                        raise
                    wait_time = backoff_factor ** attempt
                    logger.warning(f"Request attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                except aiohttp.ClientConnectionError as e:
                    if attempt == max_attempts - 1:
                        logger.error(f"Connection error after {max_attempts} attempts: {e}")
                        raise
                    wait_time = backoff_factor ** attempt
                    logger.warning(f"Connection error on attempt {attempt + 1}: {e}. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                except Exception as e:
                    logger.error(f"Unexpected error: {e}")
                    raise

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
            async with self.session.get(config["base_url"] + f"/coins/{token}/market_chart?vs_currency=eth&days={days}") as response:
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
        if cache_key in self.price_cache:
            logger.debug(f"Returning cached trading volume for {token}.")
            return self.price_cache[cache_key]
        volume = await self._fetch_from_services(
            lambda service: self._fetch_token_volume(service, token),
            f"trading volume for {token}",
        )
        if volume is not None:
            self.price_cache[cache_key] = volume
        return volume or 0.0

    async def _fetch_token_volume(self, source: str, token: str) -> Optional[float]:
        """Fetch token volume from a specified source."""
        config = self.api_configs.get(source)
        if not config:
            logger.debug(f"API configuration for {source} not found.")
            return None
        if source == "coingecko":
            url = f"{config['base_url']}/coins/markets"
            params = {"vs_currency": "usd", "ids": token}
            response = await self.make_request(source, url, params=params)
            return response[0]["total_volume"] if response else None
        else:
            logger.debug(f"Unsupported volume source: {source}")
            return None

    async def _fetch_from_services(self, fetch_func, description: str):
        """Helper method to fetch data from multiple services."""
        for service in self.api_configs.keys():
            try:
                logger.debug(f"Fetching {description} using {service}...")
                result = await fetch_func(service)
                if result:
                    return result
            except Exception as e:
                logger.warning(f"Failed to fetch {description} using {service}: {e}")
        logger.warning(f"Failed to fetch {description}.")
        return None

    async def _load_abi(self, abi_path: str) -> List[Dict[str, Any]]:
        """Load contract abi from a file."""
        try:
            async with aiofiles.open(abi_path, 'r') as file:
                content = await file.read()
                abi = json.loads(content)
            logger.debug(f"Loaded abi from {abi_path} successfully.")
            return abi
        except Exception as e:
            logger.error(f"Failed to load abi from {abi_path}: {e}")
            raise

    async def close(self):
        """Close the aiohttp session."""
        await self.session.close()

    async def initialize(self) -> None:
        """Initialize API configuration."""
        try:
            self.session = aiohttp.ClientSession()
            logger.info("API_Config initialized ✅")
        except Exception as e:
            logger.critical(f"API_Config initialization failed: {e}")
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