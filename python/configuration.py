import asyncio
import json
import logging
import os
import time
import aiofiles
import aiohttp
import pandas as pd
import dotenv

from web3 import AsyncWeb3
from cachetools import TTLCache
from abi_registry import ABI_Registry
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

dotenv.load_dotenv()


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
        
        # Add ML model configuration
        self.MODEL_RETRAINING_INTERVAL = 3600  # 1 hour
        self.MIN_TRAINING_SAMPLES = 100
        self.MODEL_ACCURACY_THRESHOLD = 0.7
        self.PREDICTION_CACHE_TTL = 300  # 5 minutes
        
        self.abi_registry = ABI_Registry()

        # Add WETH and USDC addresses with checksum conversion
        self.WETH_ADDRESS = AsyncWeb3.to_checksum_address("0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2")  # Mainnet WETH
        self.USDC_ADDRESS = AsyncWeb3.to_checksum_address("0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48")  # Mainnet USDC
        self.USDT_ADDRESS = AsyncWeb3.to_checksum_address("0xdAC17F958D2ee523a2206206994597C13D831ec7")  # Mainnet USDT
        self.DAI_ADDRESS = AsyncWeb3.to_checksum_address("0x6B175474E89094C44Da98b954EedeAC495271d0F") # Mainnet DAI
        self.WBTC_ADDRESS = AsyncWeb3.to_checksum_address("0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599") # Mainnet WBTC
        self.SUSHI_ADDRESS = AsyncWeb3.to_checksum_address("0x6B3595068778DD592e39A122f4f5a5cF09C90fE2") # Mainnet SUSHI
        self.UNI_ADDRESS = AsyncWeb3.to_checksum_address("0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984") # Mainnet UNI
        self.BAL_ADDRESS = AsyncWeb3.to_checksum_address("0xba100000625a3754423978a60c9317c58a424e3D") # Mainnet BAL
        self.AAVE_ADDRESS = AsyncWeb3.to_checksum_address("0x7Fc66500c84A76Ad7e9c93437bFc5Ac33E2DDaE9") # Mainnet AAVE
        self.CRV_ADDRESS = AsyncWeb3.to_checksum_address("0xD533a949740bb3306d119CC777fa900bA034cd52") # Mainnet CRV
        self.YFI_ADDRESS = AsyncWeb3.to_checksum_address("0x0bc529c00C6401aEF6D220BE8C6Ea1667F6Ad93e") # Mainnet YFI
        self.REN_ADDRESS = AsyncWeb3.to_checksum_address("0x408e41876cCCDC0F92210600ef50372656052a38") # Mainnet REN
        self.LINK_ADDRESS = AsyncWeb3.to_checksum_address("0x514910771AF9Ca656af840dff83E8264EcF986CA") # Mainnet LINK
        self.MKR_ADDRESS = AsyncWeb3.to_checksum_address("0x9f8F72aA9304c8B593d555F12eF6589cC3A579A2") # Mainnet MKR


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
            # First ensure ABI registry is loaded
            if not self.abi_registry.abis:
                raise ValueError("Failed to load ABIs")
            
            # Then load the rest of the configuration
            self._load_providers_and_account()
            self._load_api_keys()
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
        self.BINANCE_API_KEY = self._get_env_variable("BINANCE_API_KEY")

    def _load_providers_and_account(self) -> None:
        """Load provider endpoints and account information."""
        try:
            # Load all possible endpoints
            self.IPC_ENDPOINT = os.getenv("IPC_ENDPOINT")
            self.HTTP_ENDPOINT = os.getenv("HTTP_ENDPOINT")
            self.WEBSOCKET_ENDPOINT = os.getenv("WEBSOCKET_ENDPOINT")
            
            # Count active endpoints
            active_endpoints = sum(1 for endpoint in [
                self.IPC_ENDPOINT, 
                self.HTTP_ENDPOINT, 
                self.WEBSOCKET_ENDPOINT
            ] if endpoint is not None and endpoint.strip() != '')

            if active_endpoints != 1:
                active = []
                if self.IPC_ENDPOINT: active.append("IPC")
                if self.HTTP_ENDPOINT: active.append("HTTP")
                if self.WEBSOCKET_ENDPOINT: active.append("WebSocket")
                raise ValueError(
                    f"Exactly one endpoint (IPC, HTTP, or WebSocket) must be configured. "
                    f"Found {active_endpoints} active endpoints: {', '.join(active)}"
                )

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

    async def get_abi(self, abi_type: str) -> Optional[List[Dict]]:
        """Get ABI from registry."""
        return self.abi_registry.get_abi(abi_type)

    logger.debug("All Configurations and Environment Variables Loaded Successfully ✅") 

class API_Config:
    def __init__(self, configuration: Optional[Configuration] = None):
        self.configuration = configuration
        self.session = None 
        self.price_cache = TTLCache(maxsize=2000, ttl=300)  # 5 min cache for prices
        self.market_data_cache = TTLCache(maxsize=1000, ttl=1800)  # 30 min cache for market data
        self.token_metadata_cache = TTLCache(maxsize=500, ttl=86400)  # 24h cache for metadata
        
        # Add rate limit tracking
        self.rate_limit_counters = {
            "coingecko": {"count": 0, "reset_time": time.time(), "limit": 50},
            "coinmarketcap": {"count": 0, "reset_time": time.time(), "limit": 330},
            "cryptocompare": {"count": 0, "reset_time": time.time(), "limit": 80}
        }
        
        # Add priority queues for data fetching
        self.high_priority_tokens = set()  # Tokens currently being traded
        self.update_intervals = {
            'price': 30,  # Seconds
            'volume': 300,  # 5 minutes
            'market_data': 1800,  # 30 minutes
            'metadata': 86400  # 24 hours
        }

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

    async def token_symbol_cache(self, web3: AsyncWeb3, token_address: str) -> Optional[str]:
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
        """Fetch token metadata from a specified source."""
        config = self.api_configs.get(source)
        if not config:
            logger.debug(f"API configuration for {source} not found.")
            return None
        if source == "coingecko":
            url = f"{config['base_url']}/coins/{token}"
            response = await self.make_request(source, url)
            return response
        else:
            logger.debug(f"Unsupported metadata source: {source}")
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

    async def _fetch_with_priority(self, token: str, data_type: str) -> Any:
        """Fetch data with priority-based rate limiting."""
        try:
            is_priority = token in self.high_priority_tokens
            providers = self._get_sorted_providers(is_priority)
            
            for provider, config in providers:
                if await self._can_make_request(provider):
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

    def _get_sorted_providers(self, is_priority: bool) -> List[Tuple[str, dict]]:
        """Get providers sorted by reliability and rate limits."""
        providers = list(self.api_configs.items())
        
        def provider_score(provider_data):
            name, config = provider_data
            counter = self.rate_limit_counters[name]
            rate_left = 1 - (counter['count'] / counter['limit'])
            reliability = config['success_rate']
            
            # Priority tokens get faster, more reliable providers
            if is_priority:
                return (reliability * 2 + rate_left) * config['weight']
            return (reliability + rate_left) * config['weight']
            
        return sorted(providers, key=provider_score, reverse=True)

    async def _can_make_request(self, provider: str) -> bool:
        """Check if we can make a request within rate limits."""
        counter = self.rate_limit_counters[provider]
        current_time = time.time()
        
        # Reset counter if time window passed
        if current_time - counter['reset_time'] >= 60:
            counter['count'] = 0
            counter['reset_time'] = current_time
            
        return counter['count'] < counter['limit']

    async def update_training_data(self) -> None:
        """Smart update of training data."""
        try:
            # Get all required tokens
            tokens = await self.configuration.get_token_addresses()
            current_time = int(time.time())
            
            # Prepare batch updates
            updates = []
            for token in tokens:
                # Get latest data point timestamp for this token
                last_update = await self._get_last_update_time(token)
                
                # Only update if enough time has passed
                if current_time - last_update >= self.update_intervals['market_data']:
                    data = await self._gather_training_data(token)
                    if data:
                        updates.append(data)
                        
                # Avoid hitting rate limits
                await asyncio.sleep(0.1)
                
            # Batch write updates to CSV
            if updates:
                await self._write_training_data(updates)
                
        except Exception as e:
            logger.error(f"Error updating training data: {e}")

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

    async def _write_training_data(self, updates: List[Dict[str, Any]]) -> None:
        """Write updates to training data file."""
        try:
            df = pd.DataFrame(updates)
            training_data_path = Path(__file__).parent.parent / "linear_regression" / "training_data.csv"
            
            # Append new data
            df.to_csv(training_data_path, mode='a', header=False, index=False)
            
            # Keep file size manageable (keep last 30 days)
            self._cleanup_old_data(training_data_path, days=30)
            
        except Exception as e:
            logger.error(f"Error writing training data: {e}")

    def _cleanup_old_data(self, filepath: Path, days: int) -> None:
        """Remove data older than specified days."""
        try:
            df = pd.read_csv(filepath)
            cutoff_time = int(time.time()) - (days * 86400)
            df = df[df['timestamp'] >= cutoff_time]
            df.to_csv(filepath, index=False)
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")