#========================= 0xBuilder - By NordChain (github.com/user/JMitander) =========================#

import os
import logging
import random
import sys
import dotenv
import time
import json
import asyncio
import aiofiles
import aiohttp
import numpy as np
import hexbytes
import psutil
import traceback

from cachetools import TTLCache
from sklearn.linear_model import LinearRegression
from decimal import Decimal
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from dataclasses import *

from web3 import AsyncWeb3
from web3.exceptions import TransactionNotFound, ContractLogicError
from eth_account import Account
from web3.providers import AsyncIPCProvider, AsyncHTTPProvider, WebSocketProvider
from web3.middleware import ExtraDataToPOAMiddleware, SignAndSendRawMiddlewareBuilder
from web3.eth import AsyncEth



#========================== Logging and console output ==========================

# ANSI color codes for different log levels
COLORS = {
    "DEBUG": "\033[94m",     # Blue
    "INFO": "\033[92m",      # Green
    "WARNING": "\033[93m",   # Yellow
    "ERROR": "\033[91m",     # Red
    "CRITICAL": "\033[95m",  # Magenta
    "DETAILED": "\033[96m",  # Cyan for detailed messages
    "RESET": "\033[0m"       # Reset to default color
}

# Custom formatter to add color based on log level
class ColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        color = COLORS.get(record.levelname, COLORS["RESET"])
        reset = COLORS["RESET"]
        record.levelname = f"{color}{record.levelname}{reset}"  # Colorize level name
        record.msg = f"{color}{record.msg}{reset}"              # Colorize message
        return super().format(record)

# Configure the logging once
def configure_logging():
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColorFormatter("%(asctime)s [%(levelname)s] %(message)s"))

    logging.basicConfig(
        level=logging.INFO,  # Global logging level
        handlers=[handler]
    )

# Helper function to get the logger
def get_logger(name: Optional[str] = None) -> logging.Logger:
    if not logging.getLogger().hasHandlers():
        configure_logging()
        
    logger = logging.getLogger(name if name else "0xBuilder")
    return logger

# Initialize the logger globally so it can be used throughout the script
logger = get_logger("0xBuilder")

dotenv.load_dotenv()

_loading_bar_active = False

async def loading_bar(
    message: str,
    total_time: int,
    success_message: Optional[str] = None,
) -> None:
    """Displays a loading bar in the console, ensuring no duplicate runs, with color output and specific success messages."""
    global _loading_bar_active

    # ANSI escape codes for color
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    RESET = "\033[0m"

    
    _loading_bar_active = True  # Set the flag to active
    bar_length = 20

    try:
        for i in range(101):
            await asyncio.sleep(total_time / 100)
            percent = i / 100
            filled_length = int(percent * bar_length)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            sys.stdout.write(f"\r{GREEN}{message} [{bar}] {i}%{RESET}")
            sys.stdout.flush()

        sys.stdout.write("\n")
        sys.stdout.flush()

        if success_message:
            logger.debug(f"{YELLOW}{success_message}{RESET}")
        else:
            return
    except Exception:
        raise

#========================== Environment variable configuration ==========================

class Configuration:
    """
    Loads configuration from environment variables and monitored tokens from a JSON file.
    """

    async def load(self) -> None:
        """Loads the configuration."""
        await self._load_configuration()

    async def _load_configuration(self) -> None:
        try:
            await loading_bar("Loading Environment Variables", 2)
            self._load_api_keys()
            self._load_providers_and_account()
            await self._load_json_elements()
            logger.info("Configuration loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration: {e}") from e
        
    def _load_api_keys(self) -> None:
        self.ETHERSCAN_API_KEY = self._get_env_variable("ETHERSCAN_API_KEY")
        self.INFURA_PROJECT_ID = self._get_env_variable("INFURA_PROJECT_ID")
        self.COINGECKO_API_KEY = self._get_env_variable("COINGECKO_API_KEY")
        self.COINMARKETCAP_API_KEY = self._get_env_variable("COINMARKETCAP_API_KEY")
        self.CRYPTOCOMPARE_API_KEY = self._get_env_variable("CRYPTOCOMPARE_API_KEY")

    def _load_providers_and_account(self) -> None:
        self.HTTP_ENDPOINT = self._get_env_variable("HTTP_ENDPOINT")
        self.IPC_ENDPOINT = self._get_env_variable("IPC_ENDPOINT")
        self.WEBSOCKET_ENDPOINT = self._get_env_variable("WEBSOCKET_ENDPOINT")
        self.WALLET_KEY = self._get_env_variable("WALLET_KEY")
        self.WALLET_ADDRESS = self._get_env_variable("WALLET_ADDRESS")

    async def _load_json_elements(self) -> None:
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

    def _get_env_variable(self, var_name: str, default: Optional[str] = None) -> str:
        value = os.getenv(var_name, default)
        if value is None:
            raise EnvironmentError(f"Missing environment variable: {var_name}")
        return value

    async def _load_json_file(self, file_path: str, description: str) -> Any:
        try:
            async with aiofiles.open(file_path, "r") as f:
                content = await f.read()
                data = json.loads(content)
                await loading_bar(f"Loading {len(data)} {description} from {file_path}", 3)
                return data
        except FileNotFoundError as e:
            logger.error(f"{description.capitalize()} file not found: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding {description} JSON: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load {description} from {file_path}: {e}")
            raise

    async def _construct_abi_path(self, base_path: str, abi_filename: str) -> str:
        abi_path = os.path.join(base_path, abi_filename)
        if not os.path.exists(abi_path):
            logger.error(f"abi file not found at path: {abi_path}")
            raise FileNotFoundError(f"abi file '{abi_filename}' not found in path '{base_path}'")
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
            "aave_flashloan_abi": self.AAVE_FLASHLOAN_ABI,
            "aave_lending_pool_abi": self.AAVE_LENDING_POOL_ABI,
            "pancakeswap_router_abi": self.PANCAKESWAP_ROUTER_ABI,
            "balancer_router_abi": self.BALANCER_ROUTER_ABI,
        }
        return abi_paths.get(abi_name.lower(), "")

#========================== Nonce management system for Ethereum transactions ==========================

class Nonce_Core:
    """
    Advanced nonce management system for Ethereum transactions with caching,
    auto-recovery, and comprehensive error handling.
    """

    MAX_RETRIES = 3
    RETRY_DELAY = 1.0
    CACHE_TTL = 300  # Cache TTL in seconds

    def __init__(
        self,
        web3: AsyncWeb3,
        address: str,
        configuration: Configuration,
    ):
        self.pending_transactions = set()
        self.web3 = web3
        self.configuration = configuration
        self.address = address
        self.lock = asyncio.Lock()
        self.nonce_cache = TTLCache(maxsize=1, ttl=self.CACHE_TTL)
        self.last_sync = time.monotonic()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the nonce manager with error recovery."""
        try:
            async with self.lock:
                if not self._initialized:
                    await self._init_nonce()
                    self._initialized = True
                    logger.debug(f"Nonce_Core initialized for {self.address[:10]}...")
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise RuntimeError("Nonce_Core initialization failed") from e

    async def _init_nonce(self) -> None:
        """Initialize nonce with fallback mechanisms."""
        current_nonce = await self._fetch_current_nonce_with_retries()
        pending_nonce = await self._get_pending_nonce()
        # Use the higher of current or pending nonce
        self.nonce_cache[self.address] = max(current_nonce, pending_nonce)
        self.last_sync = time.monotonic()

    async def get_nonce(self, force_refresh: bool = False) -> int:
        """Get next available nonce with optional force refresh."""
        if not self._initialized:
            await self.initialize()
        async with self.lock:
            try:
                if force_refresh or self._should_refresh_cache():
                    await self.refresh_nonce()
                current_nonce = self.nonce_cache.get(self.address, 0)
                next_nonce = current_nonce
                self.nonce_cache[self.address] = current_nonce + 1
                logger.debug(f"Allocated nonce {next_nonce} for {self.address[:10]}...")
                return next_nonce
            except KeyError as e:
                logger.error(f"Nonce cache key error: {e}")
                await self._handle_nonce_error()
                raise
            except Exception as e:
                logger.error(f"Error getting nonce: {e}")
                await self._handle_nonce_error()
                raise

    async def refresh_nonce(self) -> None:
        """Refresh nonce from chain with conflict resolution."""
        async with self.lock:
            try:
                chain_nonce = await self._fetch_current_nonce_with_retries()
                cached_nonce = self.nonce_cache.get(self.address, 0)
                pending_nonce = await self._get_pending_nonce()
                # Take the highest nonce to avoid conflicts
                new_nonce = max(chain_nonce, cached_nonce, pending_nonce)
                self.nonce_cache[self.address] = new_nonce
                self.last_sync = time.monotonic()
                logger.debug(f"Nonce refreshed to {new_nonce}")
            except Exception as e:
                logger.error(f"Nonce refresh failed: {e}")
                raise

    async def _fetch_current_nonce_with_retries(self) -> int:
        """Fetch current nonce with exponential backoff."""
        backoff = self.RETRY_DELAY
        for attempt in range(self.MAX_RETRIES):
            try:
                return await self.web3.eth.get_transaction_count(
                    self.address, block_identifier="pending"
                )
            except Exception as e:
                if attempt == self.MAX_RETRIES - 1:
                    logger.error(f"Nonce fetch failed after retries: {e}")
                    raise
                logger.warning(f"Nonce fetch attempt {attempt + 1} failed: {e}. Retrying in {backoff}s...")
                await asyncio.sleep(backoff)
                backoff *= 2

    async def _get_pending_nonce(self) -> int:
        """Get highest nonce from pending transactions."""
        try:
            pending_nonces = [int(nonce) for nonce in self.pending_transactions]
            return max(pending_nonces) + 1 if pending_nonces else 0
        except Exception as e:
            logger.error(f"Error getting pending nonce: {e}")
            return 0

    async def track_transaction(self, tx_hash: str, nonce: int) -> None:
        """Track pending transaction for nonce management."""
        self.pending_transactions.add(nonce)
        try:
            # Wait for transaction confirmation
            await self.web3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            self.pending_transactions.discard(nonce)
        except Exception as e:
            logger.error(f"Transaction tracking failed: {e}")
        finally:
            self.pending_transactions.discard(nonce)

    async def _handle_nonce_error(self) -> None:
        """Handle nonce-related errors with recovery attempt."""
        try:
            await self.sync_nonce_with_chain()
        except Exception as e:
            logger.error(f"Nonce error recovery failed: {e}")
            raise

    async def sync_nonce_with_chain(self) -> None:
        """Force synchronization with chain state."""
        async with self.lock:
            try:
                new_nonce = await self._fetch_current_nonce_with_retries()
                self.nonce_cache[self.address] = new_nonce
                self.last_sync = time.monotonic()
                self.pending_transactions.clear()
                logger.debug(f"Nonce synchronized to {new_nonce}")
            except Exception as e:
                logger.error(f"Nonce synchronization failed: {e}")
                raise

    def _should_refresh_cache(self) -> bool:
        """Determine if cache refresh is needed."""
        return time.monotonic() - self.last_sync > (self.CACHE_TTL / 2)

    async def reset(self) -> None:
        """Complete reset of nonce manager state."""
        async with self.lock:
            try:
                self.nonce_cache.clear()
                self.pending_transactions.clear()
                self.last_sync = time.monotonic()
                self._initialized = False
                await self.initialize()
                logger.debug("Nonce Core reset complete")
            except Exception as e:
                logger.error(f"Reset failed: {e}")
                raise

    async def stop(self) -> None:
        """Gracefully stop the nonce manager."""
        try:
            await self.reset()
            logger.debug("Nonce Core stopped successfully.")
        except Exception as e:
            logger.error(f"Error stopping nonce core: {e}")
            raise

#================================= API setup and configuration =================================#

class API_Config:
    def __init__(self, configuration: Optional[Configuration] = None):
        self.configuration = Configuration
        self.session = None
        self.price_cache = TTLCache(maxsize=1000, ttl=300)  # Cache for 5 minutes
        self.token_symbol_cache = TTLCache(maxsize=1000, ttl=86400)  # Cache for 1 day

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self):
        if self.session:
            await self.session.close()

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
            logger.debug(f"API configuration for {source} not found.")
            return None
        if source == "coingecko":
            url = f"{config['base_url']}/simple/price"
            params = {"ids": token, "vs_currencies": vs_currency}
            response = await self.make_request(source, url, params=params)
            return Decimal(str(response[token][vs_currency]))
        elif source == "coinmarketcap":
            url = f"{config['base_url']}/cryptocurrency/quotes/latest"
            params = {"symbol": token.upper(), "convert": vs_currency.upper()}
            headers = {"X-CMC_PRO_API_KEY": config["api_key"]}
            response = await self.make_request(source, url, params=params, headers=headers)
            data = response["data"][token.upper()]["quote"][vs_currency.upper()]["price"]
            return Decimal(str(data))
        elif source == "cryptocompare":
            url = f"{config['base_url']}/price"
            params = {"fsym": token.upper(), "tsyms": vs_currency.upper(), "api_key": config["api_key"]}
            response = await self.make_request(source, url, params=params)
            return Decimal(str(response[vs_currency.upper()]))
        elif source == "binance":
            url = f"{config['base_url']}/ticker/price"
            symbol = f"{token.upper()}{vs_currency.upper()}"
            params = {"symbol": symbol}
            response = await self.make_request(source, url, params=params)
            return Decimal(str(response["price"]))
        else:
            logger.warning(f"Unsupported price source: {source}")
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
            rate_limiter = asyncio.Semaphore(10)
            self.rate_limiters[provider_name] = rate_limiter
        
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
            logger.debug(f"API configuration for {source} not found.")
            return None
        if source == "coingecko":
            url = f"{config['base_url']}/coins/{token}/market_chart"
            params = {"vs_currency": "usd", "days": days}
            response = await self.make_request(source, url, params=params)
            return [price[1] for price in response["prices"]]
        else:
            logger.debug(f"Unsupported historical price source: {source}")
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

#=========================== 0xBuilder's Safety controller and risk management Engine ===========================#

class Safety_Net:
    """
    Safety_Net provides risk management and price verification functionality
    with multiple data sources, automatic failover, and dynamic adjustments.
    """

    CACHE_TTL = 300  # Cache TTL in seconds
    GAS_PRICE_CACHE_TTL = 15  # 15 sec cache for gas prices

    SLIPPAGE_CONFIG = {
        "default": 0.1,
        "min": 0.01,
        "max": 0.5,
        "high_congestion": 0.05,
        "low_congestion": 0.2,
    }

    GAS_CONFIG = {
        "max_gas_price_gwei": 500,
        "min_profit_multiplier": 2.0,
        "base_gas_limit": 21000,
    }

    def __init__(
        self,
        web3: AsyncWeb3,
        configuration: Optional[Configuration] = None,
        address: Optional[str] = None,
        account: Optional[Account] = None,
        api_config: Optional[API_Config] = None,
    ):
        self.web3 = web3
        self.address = address
        self.configuration = configuration
        self.account = account
        self.api_config = api_config
        self.price_cache = TTLCache(maxsize=1000, ttl=self.CACHE_TTL)
        self.gas_price_cache = TTLCache(maxsize=1, ttl=self.GAS_PRICE_CACHE_TTL)

        self.price_lock = asyncio.Lock()
        logger.debug("Safety Net initialized with enhanced configuration")

    async def get_balance(self, account: Any) -> Decimal:
        """Get account balance with retries and caching."""
        cache_key = f"balance_{account.address}"
        if cache_key in self.price_cache:
            return self.price_cache[cache_key]

        for attempt in range(3):
            try:
                balance_wei = await self.web3.eth.get_balance(account.address)
                balance_eth = Decimal(self.web3.from_wei(balance_wei, "ether"))
                self.price_cache[cache_key] = balance_eth
                logger.debug(f"Balance for {account.address[:10]}...: {balance_eth:.4f} ETH")
                return balance_eth
            except Exception as e:
                if attempt == 2:
                    logger.error(f"Failed to get balance after 3 attempts: {e}")
                    return Decimal(0)
                await asyncio.sleep(1 * (attempt + 1))

    async def ensure_profit(
        self,
        transaction_data: Dict[str, Any],
        minimum_profit_eth: Optional[float] = None,
    ) -> bool:
        """Enhanced profit verification with dynamic thresholds and risk assessment."""
        try:
            if minimum_profit_eth is None:
                account_balance = await self.get_balance(self.account)
                minimum_profit_eth = (
                    0.003 if account_balance < Decimal("0.5") else 0.01
                )

            gas_price_gwei = await self.get_dynamic_gas_price()
            gas_used = await self.estimate_gas(transaction_data)

            if not self._validate_gas_parameters(gas_price_gwei, gas_used):
                return False

            gas_cost_eth = self._calculate_gas_cost(gas_price_gwei, gas_used)
            slippage = await self.adjust_slippage_tolerance()

            output_token = transaction_data.get("output_token")
            real_time_price = await self.api_config.get_real_time_price(output_token)
            if not real_time_price:
                return False

            profit = await self._calculate_profit(
                transaction_data, real_time_price, slippage, gas_cost_eth
            )

            self._log_profit_calculation(
                transaction_data, real_time_price, gas_cost_eth, profit, minimum_profit_eth
            )

            return profit > Decimal(minimum_profit_eth)
        except KeyError as e:
            logger.warning(f"Missing required transaction data key: {e}")
        except Exception as e:
            logger.error(f"Error in profit calculation: {e}")
        return False

    def _validate_gas_parameters(self, gas_price_gwei: Decimal, gas_used: int) -> bool:
        """Validate gas parameters against safety thresholds."""
        if gas_used == 0:
            logger.error("Gas estimation returned zero")
            return False
        if gas_price_gwei > self.GAS_CONFIG["max_gas_price_gwei"]:
            logger.warning(f"Gas price {gas_price_gwei} gwei exceeds maximum threshold")
            return False
        return True

    def _calculate_gas_cost(self, gas_price_gwei: Decimal, gas_used: int) -> Decimal:
        """Calculate total gas cost in ETH."""
        return gas_price_gwei * Decimal(gas_used) * Decimal("1e-9")

    async def _calculate_profit(
        self,
        transaction_data: Dict[str, Any],
        real_time_price: Decimal,
        slippage: float,
        gas_cost_eth: Decimal,
    ) -> Decimal:
        """Calculate expected profit considering slippage and gas costs."""
        expected_output = real_time_price * Decimal(transaction_data["amountOut"])
        input_amount = Decimal(transaction_data["amountIn"])
        slippage_adjusted_output = expected_output * (1 - Decimal(slippage))
        return slippage_adjusted_output - input_amount - gas_cost_eth

    def _log_profit_calculation(
        self,
        transaction_data: Dict[str, Any],
        real_time_price: Decimal,
        gas_cost_eth: Decimal,
        profit: Decimal,
        minimum_profit_eth: float,
    ) -> None:
        """Log detailed profit calculation metrics."""
        profitable = "Yes" if profit > Decimal(minimum_profit_eth) else "No"
        logger.debug(
            f"Profit Calculation Summary:\n"
            f"Token: {transaction_data['output_token']}\n"
            f"Real-time Price: {real_time_price:.6f} ETH\n"
            f"Input Amount: {transaction_data['amountIn']:.6f} ETH\n"
            f"Expected Output: {transaction_data['amountOut']:.6f} tokens\n"
            f"Gas Cost: {gas_cost_eth:.6f} ETH\n"
            f"Calculated Profit: {profit:.6f} ETH\n"
            f"Minimum Required: {minimum_profit_eth} ETH\n"
            f"Profitable: {profitable}"
        )

    async def get_dynamic_gas_price(self) -> Decimal:
        """Get the current gas price dynamically."""
        if "gas_price" in self.gas_price_cache:
            return self.gas_price_cache["gas_price"]
        try:
            gas_price = await self.web3.eth.generate_gas_price()
            if gas_price is None:
                gas_price = await self.web3.eth.gas_price
            gas_price_gwei = Decimal(self.web3.from_wei(gas_price, "gwei"))
            self.gas_price_cache["gas_price"] = gas_price_gwei
            return gas_price_gwei
        except Exception as e:
            logger.error(f"Error fetching dynamic gas price: {e}")
            return Decimal(0)

    async def estimate_gas(self, transaction_data: Dict[str, Any]) -> int:
        """Estimate the gas required for a transaction."""
        try:
            gas_estimate = await self.web3.eth.estimate_gas(transaction_data)
            return gas_estimate
        except Exception as e:
            logger.error(f"Gas estimation failed: {e}")
            return 0

    async def adjust_slippage_tolerance(self) -> float:
        """Adjust slippage tolerance based on network conditions."""
        try:
            congestion_level = await self.get_network_congestion()
            if congestion_level > 0.8:
                slippage = self.SLIPPAGE_CONFIG["high_congestion"]
            elif congestion_level < 0.2:
                slippage = self.SLIPPAGE_CONFIG["low_congestion"]
            else:
                slippage = self.SLIPPAGE_CONFIG["default"]
            slippage = min(
                max(slippage, self.SLIPPAGE_CONFIG["min"]), self.SLIPPAGE_CONFIG["max"]
            )
            logger.debug(f"Adjusted slippage tolerance to {slippage * 100}%")
            return slippage
        except Exception as e:
            logger.error(f"Error adjusting slippage tolerance: {e}")
            return self.SLIPPAGE_CONFIG["default"]

    async def get_network_congestion(self) -> float:
        """Estimate the current network congestion level."""
        try:
            latest_block = await self.web3.eth.get_block("latest")
            gas_used = latest_block["gasUsed"]
            gas_limit = latest_block["gasLimit"]
            congestion_level = gas_used / gas_limit
            logger.debug(f"Network congestion level: {congestion_level * 100}%")
            return congestion_level
        except Exception as e:
            logger.error(f"Error fetching network congestion: {e}")
            return 0.5  # Assume medium congestion if unknown

    async def stop(self) -> None:
         """Stops the 0xBuilder gracefully."""
         try:
            await self.api_config.close()
            logger.debug("Safety Net stopped successfully.")
         except Exception as e:
             logger.error(f"Error stopping safety net: {e}")
             raise

#=============================== Txpool/Mempool Monitoring ===============================

class Mempool_Monitor:
    """
    Advanced mempool monitoring system that identifies and analyzes profitable transactions.
    Includes sophisticated profit estimation, caching, and parallel processing capabilities.
    """

    MAX_RETRIES = 3
    RETRY_DELAY = 1.0
    BATCH_SIZE = 10
    MAX_PARALLEL_TASKS = 50

    def __init__(
        self,
        web3: AsyncWeb3,
        safety_net: Safety_Net,
        nonce_core: Nonce_Core,
        api_config: API_Config,
        monitored_tokens: Optional[List[str]] = None,
        erc20_abi: List[Dict[str, Any]] = None,
        configuration: Optional[Configuration] = None,
    ):
        # Core components
        self.web3 = web3
        self.configuration = configuration
        self.safety_net = safety_net
        self.nonce_core = nonce_core
        self.api_config = api_config

        # Monitoring state
        self.running = False
        self.pending_transactions = asyncio.Queue()
        self.monitored_tokens = set(monitored_tokens or [])
        self.profitable_transactions = asyncio.Queue()
        self.processed_transactions = set()

        # Configuration
        self.erc20_abi = erc20_abi or []
        self.minimum_profit_threshold = Decimal("0.001")
        self.max_parallel_tasks = self.MAX_PARALLEL_TASKS
        self.retry_attempts = self.MAX_RETRIES
        self.backoff_factor = 1.5

        # Concurrency control
        self.semaphore = asyncio.Semaphore(self.max_parallel_tasks)
        self.task_queue = asyncio.Queue()

        logger.debug("Mempool_Monitor initialized with enhanced configuration.")

    async def start_monitoring(self) -> None:
        """Start monitoring the mempool with improved error handling."""
        if self.running:
            logger.debug("Monitoring is already active.")
            return

        try:
            self.running = True
            monitoring_task = asyncio.create_task(self._run_monitoring())
            processor_task = asyncio.create_task(self._process_task_queue())

            logger.info("Mempool monitoring started.")
            await asyncio.gather(monitoring_task, processor_task)

        except Exception as e:
            self.running = False
            logger.error(f"Failed to start monitoring: {e}")
            raise

    async def stop_monitoring(self) -> None:
        """Gracefully stop monitoring activities."""
        if not self.running:
            return

        self.running = False
        try:
            # Wait for remaining tasks to complete
            while not self.task_queue.empty():
                await asyncio.sleep(0.1)
            logger.info("Mempool monitoring stopped gracefully.")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Error during monitoring shutdown: {e}")

    async def _run_monitoring(self) -> None:
        """Enhanced mempool monitoring with automatic recovery."""
        retry_count = 0

        while self.running:
            try:
                pending_filter = await self._setup_pending_filter()
                if not pending_filter:
                    continue

                while self.running:
                    tx_hashes = await pending_filter.get_new_entries()
                    if tx_hashes:
                        await self._handle_new_transactions(tx_hashes)
                    await asyncio.sleep(0.1)

            except Exception as e:
                retry_count += 1
                wait_time = min(self.backoff_factor ** retry_count, 30)
                logger.error(
                    f"Monitoring error (attempt {retry_count}): {e}"
                )
                await asyncio.sleep(wait_time)

    async def _setup_pending_filter(self) -> Optional[Any]:
        """Set up pending transaction filter with validation."""
        try:
            pending_filter = await self.web3.eth.filter("pending")
            logger.debug(
                f"Connected to network via {self.web3.provider.__class__.__name__}"
            )
            return pending_filter

        except Exception as e:
            logger.warning(f"Failed to setup pending filter: {e}")
            return None

    async def _handle_new_transactions(self, tx_hashes: List[str]) -> None:
        """Process new transactions in parallel with rate limiting."""
        async def process_batch(batch):
            await asyncio.gather(
                *(self._queue_transaction(tx_hash) for tx_hash in batch)
            )

        try:
            # Process transactions in batches
            for i in range(0, len(tx_hashes), self.BATCH_SIZE):
                batch = tx_hashes[i: i + self.BATCH_SIZE]
                await process_batch(batch)

        except Exception as e:
            logger.error(f"Error processing transaction batch: {e}")

    async def _queue_transaction(self, tx_hash: str) -> None:
        """Queue transaction for processing with deduplication."""
        tx_hash_hex = tx_hash.hex() if isinstance(tx_hash, bytes) else tx_hash
        if tx_hash_hex not in self.processed_transactions:
            self.processed_transactions.add(tx_hash_hex)
            await self.task_queue.put(tx_hash_hex)

    async def _process_task_queue(self) -> None:
        """Process queued transactions with concurrency control."""
        while self.running:
            try:
                tx_hash = await self.task_queue.get()
                async with self.semaphore:
                    await self.process_transaction(tx_hash)
                self.task_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Task processing error: {e}")

    async def process_transaction(self, tx_hash: str) -> None:
        """Process individual transactions with enhanced error handling."""
        try:
            tx = await self._get_transaction_with_retry(tx_hash)
            if not tx:
                return

            analysis = await self.analyze_transaction(tx)
            if analysis.get("is_profitable"):
                await self._handle_profitable_transaction(analysis)

        except Exception as e:
            logger.debug(f"Error processing transaction {tx_hash}: {e}")

    async def _get_transaction_with_retry(self, tx_hash: str) -> Optional[Any]:
        """Fetch transaction details with exponential backoff."""
        for attempt in range(self.retry_attempts):
            try:
                return await self.web3.eth.get_transaction(tx_hash)
            except TransactionNotFound:
                if attempt == self.retry_attempts - 1:
                    return None
                await asyncio.sleep(self.backoff_factor ** attempt)
            except Exception as e:
                logger.error(f"Error fetching transaction {tx_hash}: {e}")
                return None

    async def _handle_profitable_transaction(self, analysis: Dict[str, Any]) -> None:
        """Process and queue profitable transactions."""
        try:
            await self.profitable_transactions.put(analysis)
            logger.debug(
                f"Profitable transaction identified: {analysis['tx_hash']} "
                f"(Estimated profit: {analysis.get('profit', 'Unknown')} ETH)"
            )
        except Exception as e:
            logger.debug(f"Error handling profitable transaction: {e}")

    async def analyze_transaction(self, tx) -> Dict[str, Any]:
        if not tx.hash or not tx.input:
            logger.debug(
                f"Transaction {tx.hash.hex()} is missing essential fields. Skipping."
            )
            return {"is_profitable": False}
        try:
            if tx.value > 0:
                return await self._analyze_eth_transaction(tx)
            return await self._analyze_token_transaction(tx)
        except Exception as e:
            logger.error(
                f"Error analyzing transaction {tx.hash.hex()}: {e}"
            )
            return {"is_profitable": False}

    async def _analyze_eth_transaction(self, tx) -> Dict[str, Any]:
        try:
            if await self._is_profitable_eth_transaction(tx):
                await self._log_transaction_details(tx, is_eth=True)
                return {
                    "is_profitable": True,
                    "tx_hash": tx.hash.hex(),
                    "value": tx.value,
                    "to": tx.to,
                    "from": tx["from"],
                    "input": tx.input,
                    "gasPrice": tx.gasPrice,
                }
            return {"is_profitable": False}
        except Exception as e:
            logger.error(
                f"Error analyzing ETH transaction {tx.hash.hex()}: {e}"
            )
            return {"is_profitable": False}

    async def _analyze_token_transaction(self, tx) -> Dict[str, Any]:
        try:
            if not self.erc20_abi:
                logger.warning("ERC20 ABI not loaded. Cannot analyze token transaction.")
                return {"is_profitable": False}

            contract = self.web3.eth.contract(address=tx.to, abi=self.erc20_abi)
            function_abi, function_params = contract.decode_function_input(tx.input)
            function_name = function_abi.name
            if function_name in self.configuration.ERC20_SIGNATURES:
                estimated_profit = await self._estimate_profit(tx, function_params)
                if estimated_profit > self.minimum_profit_threshold:
                    logger.debug(
                        f"Identified profitable transaction {tx.hash.hex()} with estimated profit: {estimated_profit:.4f} ETH"
                    )
                    await self._log_transaction_details(tx)
                    return {
                        "is_profitable": True,
                        "profit": estimated_profit,
                        "function_name": function_name,
                        "params": function_params,
                        "tx_hash": tx.hash.hex(),
                        "to": tx.to,
                        "input": tx.input,
                        "value": tx.value,
                        "gasPrice": tx.gasPrice,
                    }
                else:
                    logger.debug(
                        f"Transaction {tx.hash.hex()} is below threshold. Skipping..."
                    )
                    return {"is_profitable": False}
            else:
                logger.debug(
                    f"Function {function_name} not in ERC20_SIGNATURES. Skipping."
                )
                return {"is_profitable": False}
        except Exception as e:
            logger.debug(
                f"Error decoding function input for transaction {tx.hash.hex()}: {e}"
            )
            return {"is_profitable": False}

    async def _is_profitable_eth_transaction(self, tx) -> bool:
        try:
            potential_profit = await self._estimate_eth_transaction_profit(tx)
            return potential_profit > self.minimum_profit_threshold
        except Exception as e:
            logger.debug(
                f"Error estimating ETH transaction profit for transaction {tx.hash.hex()}: {e}"
            )
            return False

    async def _estimate_eth_transaction_profit(self, tx: Any) -> Decimal:
        try:
            gas_price_gwei = await self.safety_net.get_dynamic_gas_price()
            gas_used = tx.gas if tx.gas else await self.web3.eth.estimate_gas(tx)
            gas_cost_eth = Decimal(gas_price_gwei) * Decimal(gas_used) * Decimal("1e-9")
            eth_value = Decimal(self.web3.from_wei(tx.value, "ether"))
            potential_profit = eth_value - gas_cost_eth
            return potential_profit if potential_profit > 0 else Decimal(0)
        except Exception as e:
            logger.error(f"Error estimating ETH transaction profit: {e}")
            return Decimal(0)

    async def _estimate_profit(self, tx, function_params: Dict[str, Any]) -> Decimal:
        try:
            gas_price_gwei = Decimal(self.web3.from_wei(tx.gasPrice, "gwei"))
            gas_used = tx.gas if tx.gas else await self.web3.eth.estimate_gas(tx)
            gas_cost_eth = gas_price_gwei * Decimal(gas_used) * Decimal("1e-9")
            input_amount_wei = Decimal(function_params.get("amountIn", 0))
            output_amount_min_wei = Decimal(function_params.get("amountOutMin", 0))
            path = function_params.get("path", [])
            if len(path) < 2:
                logger.debug(
                    f"Transaction {tx.hash.hex()} has an invalid path for swapping. Skipping."
                )
                return Decimal(0)
            output_token_address = path[-1]
            output_token_symbol = await self.api_config.get_token_symbol(self.web3, output_token_address)
            if not output_token_symbol:
                logger.debug(
                    f"Output token symbol not found for address {output_token_address}. Skipping."
                )
                return Decimal(0)
            market_price = await self.api_config.get_real_time_price(
                output_token_symbol.lower()
            )
            if market_price is None or market_price == 0:
                logger.debug(
                    f"Market price not available for token {output_token_symbol}. Skipping."
                )
                return Decimal(0)
            input_amount_eth = Decimal(self.web3.from_wei(input_amount_wei, "ether"))
            output_amount_eth = Decimal(self.web3.from_wei(output_amount_min_wei, "ether"))
            expected_output_value = output_amount_eth * market_price
            profit = expected_output_value - input_amount_eth - gas_cost_eth
            return profit if profit > 0 else Decimal(0)
        except Exception as e:
            logger.debug(
                f"Error estimating profit for transaction {tx.hash.hex()}: {e}"
            )
            return Decimal(0)

    async def _log_transaction_details(self, tx, is_eth=False) -> None:
        try:
            transaction_info = {
                "transaction hash": tx.hash.hex(),
                "value": self.web3.from_wei(tx.value, "ether") if is_eth else tx.value,
                "from": tx["from"],
                "to": (tx.to[:10] + "..." + tx.to[-10:]) if tx.to else None,
                "input": tx.input,
                "gas price": self.web3.from_wei(tx.gasPrice, "gwei"),
            }
            if is_eth:
                logger.debug(f"Pending ETH Transaction Details: {transaction_info}")
            else:
                logger.debug(f"Pending Token Transaction Details: {transaction_info}")
        except Exception as e:
            logger.debug(
                f"Error logging transaction details for {tx.hash.hex()}: {e}"
            )

#============================= Transaction Orchestrator and Core =============================

class Transaction_Core:
    """
    Transaction_Core is the main transaction engine that handles all transaction-related
    Builds and executes transactions, including front-run, back-run, and sandwich attack strategies.
    It interacts with smart contracts, manages transaction signing, gas price estimation, and handles flashloans
    """
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0  # Base delay in seconds for retries

    def __init__(
        self,
        web3: AsyncWeb3,
        account: Account,
        aave_flashloan_address: str,
        aave_flashloan_abi: List[Dict[str, Any]],
        aave_lending_pool_address: str,
        aave_lending_pool_abi: List[Dict[str, Any]],
        api_config: Optional[API_Config] = None,
        monitor: Optional[Mempool_Monitor] = None,
        nonce_core: Optional[Nonce_Core] = None,
        safety_net: Optional[Safety_Net] = None,
        configuration: Optional[Configuration] = None,
        gas_price_multiplier: float = 1.1,
        erc20_abi: Optional[List[Dict[str, Any]]] = None,
    ):
        self.web3 = web3
        self.account = account
        self.configuration = configuration
        self.monitor = monitor
        self.api_config = api_config
        self.nonce_core = nonce_core
        self.safety_net = safety_net
        self.gas_price_multiplier = gas_price_multiplier
        self.RETRY_ATTEMPTS = self.MAX_RETRIES
        self.erc20_abi = erc20_abi or []
        self.current_profit = Decimal("0")
        self.aave_flashloan_address = aave_flashloan_address
        self.aave_flashloan_abi = aave_flashloan_abi
        self.aave_lending_pool_address = aave_lending_pool_address
        self.aave_lending_pool_abi = aave_lending_pool_abi

    async def initialize(self):
        """Initializes all required contracts."""
        self.flashloan_contract = await self._initialize_contract(
            self.aave_flashloan_address,
            self.aave_flashloan_abi,
            "Flashloan Contract",
        )
        self.lending_pool_contract = await self._initialize_contract(
            self.aave_lending_pool_address,
            self.aave_lending_pool_abi,
            "Lending Pool Contract",
        )
        self.uniswap_router_contract = await self._initialize_contract(
            self.configuration.UNISWAP_ROUTER_ADDRESS,
            self.configuration.UNISWAP_ROUTER_ABI,
            "Uniswap Router Contract",
        )
        self.sushiswap_router_contract = await self._initialize_contract(
            self.configuration.SUSHISWAP_ROUTER_ADDRESS,
            self.configuration.SUSHISWAP_ROUTER_ABI,
            "Sushiswap Router Contract",
        )
        self.pancakeswap_router_contract = await self._initialize_contract(
            self.configuration.PANCAKESWAP_ROUTER_ADDRESS,
            self.configuration.PANCAKESWAP_ROUTER_ABI,
            "Pancakeswap Router Contract",
        )
        self.balancer_router_contract = await self._initialize_contract(
            self.configuration.BALANCER_ROUTER_ADDRESS,
            self.configuration.BALANCER_ROUTER_ABI,
            "Balancer Router Contract",
        )

        self.erc20_abi = self.erc20_abi or await self._load_erc20_abi()

    async def _initialize_contract(
        self,
        contract_address: str,
        contract_abi: List[Dict[str, Any]],
        contract_name: str,
    ) -> Any:
        """Initializes a contract instance with  error handling."""
        try:
            # Load ABI from file if it's a string path
            if isinstance(contract_abi, str):
                async with aiofiles.open(contract_abi, 'r') as f:
                    abi_content = await f.read()
                    contract_abi = json.loads(abi_content)

            contract_instance = self.web3.eth.contract(
                address=self.web3.to_checksum_address(contract_address),
                abi=contract_abi,
            )
            logger.info(f"Loaded {contract_name} successfully.")
            return contract_instance
        except FileNotFoundError:
            logger.error(f"ABI file for {contract_name} not found at {contract_abi}.")
            raise
        except json.JSONDecodeError:
            logger.error(f"ABI file for {contract_name} is not valid JSON.")
            raise
        except Exception as e:
            logger.error(
                f"Failed to load {contract_name} at {contract_address}: {e}"
            )
            raise ValueError(
                f"Contract initialization failed for {contract_name}"
            ) from e

    async def _load_erc20_abi(self) -> List[Dict[str, Any]]:
        """Load the ERC20 ABI."""
        try:
            erc20_abi = await self.api_config._load_abi(self.configuration.ERC20_ABI)
            logger.info("ERC20 ABI loaded successfully.")
            return erc20_abi
        except FileNotFoundError:
            logger.error(f"ERC20 ABI file not found at {self.configuration.ERC20_ABI}.")
            raise
        except json.JSONDecodeError:
            logger.error("ERC20 ABI file is not valid JSON.")
            raise
        except Exception as e:
            logger.error(f"Failed to load ERC20 ABI: {e}")
            raise ValueError("ERC20 ABI loading failed") from e

    async def build_transaction(
        self, function_call: Any, additional_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Builds a transaction dictionary from a contract function call.

        :param function_call: Contract function call object.
        :param additional_params: Additional transaction parameters.
        :return: Transaction dictionary.
        """
        additional_params = additional_params or {}
        try:
            tx_details = function_call.buildTransaction({
                'chainId': await self.web3.eth.chain_id,
                'nonce': await self.nonce_core.get_nonce(),
                'from': self.account.address,
            })
            tx_details.update(additional_params)
            tx_details["gas"] = await self.estimate_gas_smart(tx_details)
            tx_details.update(await self.get_dynamic_gas_price())
            logger.debug(f"Built transaction: {tx_details}")
            return tx_details
        except KeyError as e:
            logger.error(f"Missing transaction parameter: {e}")
            raise
        except Exception as e:
            logger.error(f"Error building transaction: {e}")
            raise

    async def get_dynamic_gas_price(self) -> Dict[str, int]:
        """
        Gets dynamic gas price adjusted by the multiplier.

        :return: Dictionary containing 'gasPrice'.
        """
        try:
            gas_price_gwei = await self.safety_net.get_dynamic_gas_price()
            logger.debug(f"Fetched gas price: {gas_price_gwei} Gwei")
        except Exception as e:
            logger.error(
                f"Error fetching dynamic gas price: {e}. Using default gas price."
            )
            gas_price_gwei = Decimal(100.0)  # Default gas price in Gwei

        gas_price = int(
            self.web3.to_wei(gas_price_gwei * self.gas_price_multiplier, "gwei")
        )
        return {"gasPrice": gas_price}

    async def estimate_gas_smart(self, tx: Dict[str, Any]) -> int:
        """
        Estimates gas with fallback to a default value.

        :param tx: Transaction dictionary.
        :return: Estimated gas.
        """
        try:
            gas_estimate = await self.web3.eth.estimate_gas(tx)
            logger.debug(f"Estimated gas: {gas_estimate}")
            return gas_estimate
        except ContractLogicError as e:
            logger.warning(f"Contract logic error during gas estimation: {e}. Using default gas limit.")
            return 100_000  # Default gas limit
        except TransactionNotFound:
            logger.warning("Transaction not found during gas estimation. Using default gas limit.")
            return 100_000
        except Exception as e:
            logger.error(f"Gas estimation failed: {e}. Using default gas limit.")
            return 100_000  # Default gas limit

    async def execute_transaction(self, tx: Dict[str, Any]) -> Optional[str]:
        """
        Executes a transaction with retries.

        :param tx: Transaction dictionary.
        :return: Transaction hash if successful, else None.
        """
        try:
            for attempt in range(1, self.MAX_RETRIES + 1):
                signed_tx = await self.sign_transaction(tx)
                tx_hash = await self.web3.eth.send_raw_transaction(signed_tx)
                tx_hash_hex = (
                    tx_hash.hex()
                    if isinstance(tx_hash, hexbytes.HexBytes)
                    else tx_hash
                )
                logger.info(
                    f"Transaction sent successfully with hash: {tx_hash_hex}"
                )
                await self.nonce_core.refresh_nonce()
                return tx_hash_hex
        except TransactionNotFound as e:
                logger.error(
                    f"Transaction not found: {e}. Attempt {attempt} of {self.retry_attempts}"
                )
        except ContractLogicError as e:
                logger.error(
                    f"Contract logic error: {e}. Attempt {attempt} of {self.retry_attempts}"
                )
                
        except Exception as e:
                logger.error(
                    f"Error executing transaction: {e}. Attempt {attempt} of {self.retry_attempts}"
                )
                if attempt < self.MAX_RETRIES:
                    sleep_time = self.RETRY_DELAY * attempt
                    logger.warning(f"Retrying in {sleep_time} seconds...")
                    await asyncio.sleep(sleep_time)
                    logger.warning("Failed to execute transaction after multiple attempts.")
                    return None

    async def sign_transaction(self, transaction: Dict[str, Any]) -> bytes:
        """
        Signs a transaction with the account's private key.

        :param transaction: Transaction dictionary.
        :return: Signed transaction bytes.
        """
        try:
            signed_tx = self.web3.eth.account.sign_transaction(
                transaction,
                private_key=self.account.key,
            )
            logger.info(
                f"Transaction signed successfully: Nonce {transaction['nonce']}."
            )
            return signed_tx.rawTransaction
        except KeyError as e:
            logger.error(f"Missing transaction parameter for signing: {e}")
            raise
        except Exception as e:
            logger.error(f"Error signing transaction: {e}")
            raise

    async def handle_eth_transaction(self, target_tx: Dict[str, Any]) -> bool:
        """
        Handles an ETH transfer transaction.

        :param target_tx: Target transaction dictionary.
        :return: True if successful, else False.
        """
        tx_hash = target_tx.get("tx_hash", "Unknown")
        logger.debug(f"Handling ETH transaction {tx_hash}")
        try:
            eth_value = target_tx.get("value", 0)
            if eth_value <= 0:
                logger.debug("Transaction value is zero or negative. Skipping.")
                return False

            tx_details = {
                "to": target_tx.get("to", ""),
                "value": eth_value,
                "gas": 21_000,
                "nonce": await self.nonce_core.get_nonce(),
                "chainId": self.web3.eth.chain_id,
                "from": self.account.address,
            }
            original_gas_price = int(target_tx.get("gasPrice", 0))
            if original_gas_price <= 0:
                logger.warning("Original gas price is zero or negative. Skipping.")
                return False
            tx_details["gasPrice"] = int(
                original_gas_price * 1.1  # Increase gas price by 10%
            )

            eth_value_ether = self.web3.from_wei(eth_value, "ether")
            logger.info(
                f"Building ETH front-run transaction for {eth_value_ether} ETH to {tx_details['to']}"
            )
            tx_hash_executed = await self.execute_transaction(tx_details)
            if tx_hash_executed:
                logger.info(
                    f"Successfully executed ETH transaction with hash: {tx_hash_executed}"
                )
                return True
            else:
                logger.warning("Failed to execute ETH transaction.")
                return False
        except KeyError as e:
            logger.error(f"Missing required transaction parameter: {e}")
            return False
        except Exception as e:
            logger.error(f"Error handling ETH transaction: {e}")
            return False

    def calculate_flashloan_amount(self, target_tx: Dict[str, Any]) -> int:
        """
        Calculates the flashloan amount based on estimated profit.

        :param target_tx: Target transaction dictionary.
        :return: Flashloan amount in Wei.
        """
        estimated_profit = target_tx.get("profit", 0)
        if estimated_profit > 0:
            flashloan_amount = int(
                Decimal(estimated_profit) * Decimal("0.8") * Decimal("1e18")
            )  # Convert ETH to Wei
            logger.debug(
                f"Calculated flashloan amount: {flashloan_amount} Wei based on estimated profit."
            )
            return flashloan_amount
        else:
            logger.debug("No estimated profit. Setting flashloan amount to 0.")
            return 0

    async def simulate_transaction(self, transaction: Dict[str, Any]) -> bool:
        """
        Simulates a transaction to check if it will succeed.

        :param transaction: Transaction dictionary.
        :return: True if simulation succeeds, else False.
        """
        logger.debug(
            f"Simulating transaction with nonce {transaction.get('nonce', 'Unknown')}."
        )
        try:
            await self.web3.eth.call(transaction, block_identifier="pending")
            logger.debug("Transaction simulation succeeded.")
            return True
        except ContractLogicError as e:
            logger.debug(f"Transaction simulation failed due to contract logic error: {e}")
            return False
        except Exception as e:
            logger.debug(f"Transaction simulation failed: {e}")
            return False

    async def prepare_flashloan_transaction(
        self, flashloan_asset: str, flashloan_amount: int
    ) -> Optional[Dict[str, Any]]:
        """
        Prepares a flashloan transaction.

        :param flashloan_asset: Asset address to borrow.
        :param flashloan_amount: Amount to borrow in Wei.
        :return: Transaction dictionary if successful, else None.
        """
        if flashloan_amount <= 0:
            logger.debug(
                "Flashloan amount is 0 or less, skipping flashloan transaction preparation."
            )
            return None
        try:
            flashloan_function = self.flashloan_contract.functions.RequestFlashLoan(
                self.web3.to_checksum_address(flashloan_asset), flashloan_amount
            )
            logger.debug(
                f"Preparing flashloan transaction for {flashloan_amount} Wei of {flashloan_asset}."
            )
            flashloan_tx = await self.build_transaction(flashloan_function)
            return flashloan_tx
        except ContractLogicError as e:
            logger.error(
                f"Contract logic error preparing flashloan transaction: {e}"
            )
            return None
        except Exception as e:
            logger.error(f"Error preparing flashloan transaction: {e}")
            return None

    async def send_bundle(self, transactions: List[Dict[str, Any]]) -> bool:
        """
        Sends a bundle of transactions to MEV relays.

        :param transactions: List of transaction dictionaries.
        :return: True if bundle sent successfully, else False.
        """
        try:
            signed_txs = [await self.sign_transaction(tx) for tx in transactions]
            bundle_payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "eth_sendBundle",
                "params": [
                    {
                        "txs": [signed_tx.hex() for signed_tx in signed_txs],
                        "blockNumber": hex(await self.web3.eth.block_number + 1),
                    }
                ],
            }

            # List of MEV builders to try
            mev_builders = [
                {
                    "name": "Flashbots",
                    "url": "https://relay.flashbots.net",
                    "auth_header": "X-Flashbots-Signature"
                },
                # Add other builders as needed...
            ]

            # Track successful submissions
            successes = []

            # Try sending to each builder
            for builder in mev_builders:
                headers = {
                    "Content-Type": "application/json",
                     builder["auth_header"]: f"{self.account.address}:{self.account.key}",
               }

                for attempt in range(1, self.retry_attempts + 1):
                    try:
                        logger.debug(f"Attempt {attempt} to send bundle via {builder['name']}.")
                        async with aiohttp.ClientSession() as session:
                            async with session.post(
                                builder["url"],
                                json=bundle_payload,
                                headers=headers,
                                timeout=30,
                            ) as response:
                                response.raise_for_status()
                                response_data = await response.json()

                                if "error" in response_data:
                                    logger.error(
                                        f"Bundle submission error via {builder['name']}: {response_data['error']}"
                                    )
                                    raise ValueError(response_data["error"])

                                logger.info(f"Bundle sent successfully via {builder['name']}.")
                                successes.append(builder['name'])
                                break  # Success, move to next builder

                    except aiohttp.ClientResponseError as e:
                        logger.error(
                            f"HTTP error sending bundle via {builder['name']}: {e}. Attempt {attempt} of {self.retry_attempts}"
                        )
                        if attempt < self.retry_attempts:
                            sleep_time = self.retry_delay * attempt
                            logger.warning(f"Retrying in {sleep_time} seconds...")
                            await asyncio.sleep(sleep_time)
                    except ValueError as e:
                        logger.error(f"Bundle submission error via {builder['name']}: {e}")
                        break  # Move to next builder
                    except Exception as e:
                        logger.error(f"Unexpected error with {builder['name']}: {e}. Attempt {attempt} of {self.retry_attempts}")
                        if attempt < self.retry_attempts:
                            sleep_time = self.retry_delay * attempt
                            logger.warning(f"Retrying in {sleep_time} seconds...")
                            await asyncio.sleep(sleep_time)

            # Update nonce if any submissions succeeded
            if successes:
                await self.nonce_core.refresh_nonce()
                logger.info(f"Bundle successfully sent to builders: {', '.join(successes)}")
                return True
            else:
                logger.warning("Failed to send bundle to any MEV builders.")
                return False

        except Exception as e:
            logger.error(f"Unexpected error in send_bundle: {e}")
            return False

    async def front_run(self, target_tx: Dict[str, Any]) -> bool:
        """
        Executes a front-run transaction with  validation and error handling.

        :param target_tx: Target transaction dictionary.
        :return: True if successful, else False.
        """
        if not isinstance(target_tx, dict):
            logger.debug("Invalid transaction format provided!")
            return False

        tx_hash = target_tx.get("tx_hash", "Unknown")
        logger.debug(f"Attempting front-run on target transaction: {tx_hash}")

        # Validate required transaction parameters
        required_fields = ["input", "to", "value", "gasPrice"]
        if not all(field in target_tx for field in required_fields):
            missing = [field for field in required_fields if field not in target_tx]
            logger.debug(f"Missing required transaction parameters: {missing}. Skipping...")
            return False

        try:
            # Decode transaction input with validation
            decoded_tx = await self.decode_transaction_input(
                target_tx.get("input", "0x"),
                self.web3.to_checksum_address(target_tx.get("to", ""))
            )
            if not decoded_tx or "params" not in decoded_tx:
                logger.debug("Failed to decode transaction input for front-run.")
                return False

            # Extract and validate path parameter
            path = decoded_tx["params"].get("path", [])
            if not path or not isinstance(path, list) or len(path) < 2:
                logger.debug("Transaction has invalid or no path parameter. Skipping...")
                return False

            # Prepare flashloan
            try:
                flashloan_asset = self.web3.to_checksum_address(path[0])
                flashloan_amount = self.calculate_flashloan_amount(target_tx)

                if flashloan_amount <= 0:
                    logger.debug("Insufficient flashloan amount calculated.")
                    return False

                flashloan_tx = await self.prepare_flashloan_transaction(
                    flashloan_asset, flashloan_amount
                )
                if not flashloan_tx:
                    logger.debug("Failed to prepare flashloan transaction!")
                    return False

            except (ValueError, IndexError) as e:
                logger.error(f"Error preparing flashloan: {str(e)}")
                return False

            # Prepare front-run transaction
            front_run_tx_details = await self._prepare_front_run_transaction(target_tx)
            if not front_run_tx_details:
                logger.warning("Failed to prepare front-run transaction!")
                return False

            # Simulate transactions
            simulation_success = await asyncio.gather(
                self.simulate_transaction(flashloan_tx),
                self.simulate_transaction(front_run_tx_details)
            )

            if not all(simulation_success):
                logger.error("Transaction simulation failed!")
                return False

            # Send transaction bundle
            if await self.send_bundle([flashloan_tx, front_run_tx_details]):
                logger.info("Front-run transaction bundle sent successfully.")
                return True
            else:
                logger.warning("Failed to send front-run transaction bundle!")
                return False

        except KeyError as e:
            logger.error(f"Missing required transaction parameter: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error in front-run execution: {str(e)}")
            return False

    async def back_run(self, target_tx: Dict[str, Any]) -> bool:
        """
        Executes a back-run transaction with validation and error handling.

        :param target_tx: Target transaction dictionary.
        :return: True if successful, else False.
        """
        if not isinstance(target_tx, dict):
            logger.debug("Invalid transaction format provided!")
            return False

        tx_hash = target_tx.get("tx_hash", "Unknown")
        logger.debug(f"Attempting back-run on target transaction: {tx_hash}")

        # Validate required transaction parameters
        required_fields = ["input", "to", "value", "gasPrice"]
        if not all(field in target_tx for field in required_fields):
            missing = [field for field in required_fields if field not in target_tx]
            logger.debug(f"Missing required transaction parameters: {missing}. Skipping...")
            return False

        try:
            # Decode transaction input with validation
            decoded_tx = await self.decode_transaction_input(
                target_tx.get("input", "0x"),
                self.web3.to_checksum_address(target_tx.get("to", ""))
            )
            if not decoded_tx or "params" not in decoded_tx:
                logger.debug("Failed to decode transaction input for back-run.")
                return False

            # Extract and validate path parameter
            path = decoded_tx["params"].get("path", [])
            if not path or not isinstance(path, list) or len(path) < 2:
                logger.debug("Transaction has invalid or no path parameter. Skipping...")
                return False

            # Reverse the path for back-run
            reversed_path = path[::-1]
            decoded_tx["params"]["path"] = reversed_path

            # Prepare back-run transaction
            back_run_tx_details = await self._prepare_back_run_transaction(target_tx, decoded_tx)
            if not back_run_tx_details:
                logger.warning("Failed to prepare back-run transaction!")
                return False

            # Simulate back-run transaction
            simulation_success = await self.simulate_transaction(back_run_tx_details)

            if not simulation_success:
                logger.error("Back-run transaction simulation failed!")
                return False

            # Send back-run transaction
            if await self.send_bundle([back_run_tx_details]):
                logger.info("Back-run transaction bundle sent successfully.")
                return True
            else:
                logger.warning("Failed to send back-run transaction bundle!")
                return False

        except KeyError as e:
            logger.error(f"Missing required transaction parameter: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error in back-run execution: {str(e)}")
            return False

    async def execute_sandwich_attack(self, target_tx: Dict[str, Any]) -> bool:
        """
        Executes a sandwich attack on the target transaction.

        :param target_tx: Target transaction dictionary.
        :return: True if successful, else False.
        """
        if not isinstance(target_tx, dict):
            logger.debug("Invalid transaction format provided!")
            return False

        tx_hash = target_tx.get("tx_hash", "Unknown")
        logger.debug(f"Attempting sandwich attack on target transaction: {tx_hash}")

        # Validate required transaction parameters
        required_fields = ["input", "to", "value", "gasPrice"]
        if not all(field in target_tx for field in required_fields):
            missing = [field for field in required_fields if field not in target_tx]
            logger.debug(f"Missing required transaction parameters: {missing}. Skipping...")
            return False

        try:
            # Decode transaction input with validation
            decoded_tx = await self.decode_transaction_input(
                target_tx.get("input", "0x"),
                self.web3.to_checksum_address(target_tx.get("to", ""))
            )
            if not decoded_tx or "params" not in decoded_tx:
                logger.debug("Failed to decode transaction input for sandwich attack.")
                return False

            # Extract and validate path parameter
            path = decoded_tx["params"].get("path", [])
            if not path or not isinstance(path, list) or len(path) < 2:
                logger.debug("Transaction has invalid or no path parameter. Skipping...")
                return False

            flashloan_asset = self.web3.to_checksum_address(path[0])
            flashloan_amount = self.calculate_flashloan_amount(target_tx)

            if flashloan_amount <= 0:
                logger.debug("Insufficient flashloan amount calculated.")
                return False

            # Prepare flashloan transaction
            flashloan_tx = await self.prepare_flashloan_transaction(
                flashloan_asset, flashloan_amount
            )
            if not flashloan_tx:
                logger.debug("Failed to prepare flashloan transaction!")
                return False

            # Prepare front-run transaction
            front_run_tx_details = await self._prepare_front_run_transaction(target_tx)
            if not front_run_tx_details:
                logger.warning("Failed to prepare front-run transaction!")
                return False

            # Prepare back-run transaction
            back_run_tx_details = await self._prepare_back_run_transaction(target_tx, decoded_tx)
            if not back_run_tx_details:
                logger.warning("Failed to prepare back-run transaction!")
                return False

            # Simulate all transactions
            simulation_results = await asyncio.gather(
                self.simulate_transaction(flashloan_tx),
                self.simulate_transaction(front_run_tx_details),
                self.simulate_transaction(back_run_tx_details),
                return_exceptions=True
            )

            if any(isinstance(result, Exception) for result in simulation_results):
                logger.warning("One or more transaction simulations failed!")
                return False

            if not all(simulation_results):
                logger.warning("Not all transaction simulations were successful!")
                return False

            # Execute transaction bundle
            if await self.send_bundle([flashloan_tx, front_run_tx_details, back_run_tx_details]):
                logger.info("Sandwich attack transaction bundle sent successfully.")
                return True
            else:
                logger.warning("Failed to send sandwich attack transaction bundle!")
                return False

        except KeyError as e:
            logger.error(f"Missing required transaction parameter: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error in sandwich attack execution: {e}")
            return False

    async def _prepare_front_run_transaction(
        self, target_tx: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Prepares the front-run transaction based on the target transaction.

        :param target_tx: Target transaction dictionary.
        :return: Front-run transaction dictionary if successful, else None.
        """
        try:
            decoded_tx = await self.decode_transaction_input(
                target_tx.get("input", "0x"),
                self.web3.to_checksum_address(target_tx.get("to", ""))
            )
            if not decoded_tx:
                logger.debug("Failed to decode target transaction input for front-run.")
                return None

            function_name = decoded_tx.get("function_name")
            if not function_name:
                logger.debug("Missing function name in decoded transaction.")
                return None

            function_params = decoded_tx.get("params", {})
            to_address = self.web3.to_checksum_address(target_tx.get("to", ""))

            # Router address mapping
            routers = {
                self.configuration.UNISWAP_ROUTER_ADDRESS: (self.uniswap_router_contract, "Uniswap"),
                self.configuration.SUSHISWAP_ROUTER_ADDRESS: (self.sushiswap_router_contract, "Sushiswap"),
                self.configuration.PANCAKESWAP_ROUTER_ADDRESS: (self.pancakeswap_router_contract, "Pancakeswap"),
                self.configuration.BALANCER_ROUTER_ADDRESS: (self.balancer_router_contract, "Balancer")
            }

            if to_address not in routers:
                logger.warning(f"Unknown router address {to_address}. Cannot determine exchange.")
                return None

            router_contract, exchange_name = routers[to_address]
            if not router_contract:
                logger.warning(f"Router contract not initialized for {exchange_name}.")
                return None

            # Get the function object by name
            try:
                front_run_function = getattr(router_contract.functions, function_name)(**function_params)
            except AttributeError:
                logger.debug(f"Function {function_name} not found in {exchange_name} router ABI.")
                return None

            # Build the transaction
            front_run_tx = await self.build_transaction(front_run_function)
            logger.info(f"Prepared front-run transaction on {exchange_name} successfully.")
            return front_run_tx

        except Exception as e:
            logger.error(f"Error preparing front-run transaction: {e}")
            return None


    async def _prepare_back_run_transaction(
        self, target_tx: Dict[str, Any], decoded_tx: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Prepare the back-run transaction based on the target transaction.

        :param target_tx: Target transaction dictionary.
        :param decoded_tx: Decoded target transaction dictionary.
        :return: Back-run transaction dictionary if successful, else None.
        """
        try:
            function_name = decoded_tx.get("function_name")
            if not function_name:
                logger.debug("Missing function name in decoded transaction.")
                return None

            function_params = decoded_tx.get("params", {})

            # Handle path parameter for back-run
            path = function_params.get("path", [])
            if not path or not isinstance(path, list) or len(path) < 2:
                logger.debug("Transaction has invalid or no path parameter for back-run.")
                return None

            # Reverse the path for back-run
            reversed_path = path[::-1]
            function_params["path"] = reversed_path

            to_address = self.web3.to_checksum_address(target_tx.get("to", ""))

            # Router address mapping
            routers = {
                self.configuration.UNISWAP_ROUTER_ADDRESS: (self.uniswap_router_contract, "Uniswap"),
                self.configuration.SUSHISWAP_ROUTER_ADDRESS: (self.sushiswap_router_contract, "Sushiswap"),
                self.configuration.PANCAKESWAP_ROUTER_ADDRESS: (self.pancakeswap_router_contract, "Pancakeswap"),
                self.configuration.BALANCER_ROUTER_ADDRESS: (self.balancer_router_contract, "Balancer")
            }

            if to_address not in routers:
                logger.debug(f"Unknown router address {to_address}. Cannot determine exchange.")
                return None

            router_contract, exchange_name = routers[to_address]
            if not router_contract:
                logger.debug(f"Router contract not initialized for {exchange_name}.")
                return None

            # Get the function object by name
            try:
                back_run_function = getattr(router_contract.functions, function_name)(**function_params)
            except AttributeError:
                logger.debug(f"Function {function_name} not found in {exchange_name} router ABI.")
                return None

            # Build the transaction
            back_run_tx = await self.build_transaction(back_run_function)
            logger.info(f"Prepared back-run transaction on {exchange_name} successfully.")
            return back_run_tx

        except Exception as e:
            logger.error(f"Error preparing back-run transaction: {e}")
            return None
        
    async def decode_transaction_input(self, input_data: str, contract_address: str) -> Optional[Dict[str, Any]]:
        """
        Decodes the input data of a transaction.

        :param input_data: Hexadecimal input data of the transaction.
        :param contract_address: Address of the contract being interacted with.
        :return: Dictionary containing function name and parameters if successful, else None.
        """
        try:
            # Load the contract ABI
            contract = self.web3.eth.contract(address=contract_address, abi=self.erc20_abi)
            function_obj, function_params = contract.decode_function_input(input_data)
            decoded_data = {
                "function_name": function_obj.fn_name,
                "params": function_params,
            }
            logger.debug(
                f"Decoded transaction input: {decoded_data}"
            )
            return decoded_data
        except ContractLogicError as e:
            logger.debug(f"Contract logic error during decoding: {e}")
            return None
        except Exception as e:
            logger.error(f"Error decoding transaction input: {e}")
            return None

    async def cancel_transaction(self, nonce: int) -> bool:
            """
            Cancels a stuck transaction by sending a zero-value transaction with the same nonce.
    
            :param nonce: Nonce of the transaction to cancel.
            :return: True if cancellation was successful, else False.
            """
            cancel_tx = {
                "to": self.account.address,
                "value": 0,
                "gas": 21_000,
                "gasPrice": self.web3.to_wei("150", "gwei"),  # Higher than the stuck transaction
                "nonce": nonce,
                "chainId": await self.web3.eth.chain_id,
                "from": self.account.address,
            }
            
            try:
                signed_cancel_tx = await self.sign_transaction(cancel_tx)
                tx_hash = await self.web3.eth.send_raw_transaction(signed_cancel_tx)
                tx_hash_hex = (
                    tx_hash.hex()
                    if isinstance(tx_hash, hexbytes.HexBytes)
                    else tx_hash
                )
                logger.debug(
                    f"Cancellation transaction sent successfully: {tx_hash_hex}"
                )
                return True
            except Exception as e:
                logger.warning(f"Failed to cancel transaction: {e}")
                return False

    async def estimate_gas_limit(self, tx: Dict[str, Any]) -> int:
        """
        Estimates the gas limit for a transaction.

        :param tx: Transaction dictionary.
        :return: Estimated gas limit.
        """
        try:
            gas_estimate = await self.web3.eth.estimate_gas(tx)
            logger.debug(f"Estimated gas: {gas_estimate}")
            return gas_estimate
        except Exception as e:
            logger.debug(
                f"Gas estimation failed: {e}. Using default gas limit of 100000."
            )
            return 100_000  # Default gas limit

    async def get_current_profit(self) -> Decimal:
        """
        Fetches the current profit from the safety net.

        :return: Current profit as Decimal.
        """
        try:
            current_profit = await self.safety_net.get_balance(self.account)
            self.current_profit = Decimal(current_profit)
            logger.debug(f"Current profit: {self.current_profit} ETH")
            return self.current_profit
        except Exception as e:
            logger.error(f"Error fetching current profit: {e}")
            return Decimal("0")

    async def withdraw_eth(self) -> bool:
        """
        Withdraws ETH from the flashloan contract.

        :return: True if successful, else False.
        """
        try:
            withdraw_function = self.flashloan_contract.functions.withdrawETH()
            tx = await self.build_transaction(withdraw_function)
            tx_hash = await self.execute_transaction(tx)
            if tx_hash:
                logger.debug(
                    f"ETH withdrawal transaction sent with hash: {tx_hash}"
                )
                return True
            else:
                logger.warning("Failed to send ETH withdrawal transaction.")
                return False
        except ContractLogicError as e:
            logger.error(f"Contract logic error during ETH withdrawal: {e}")
            return False
        except Exception as e:
            logger.error(f"Error withdrawing ETH: {e}")
            return False

    async def withdraw_token(self, token_address: str) -> bool:
        """
        Withdraws a specific token from the flashloan contract.

        :param token_address: Address of the token to withdraw.
        :return: True if successful, else False.
        """
        try:
            withdraw_function = self.flashloan_contract.functions.withdrawToken(
                self.web3.to_checksum_address(token_address)
            )
            tx = await self.build_transaction(withdraw_function)
            tx_hash = await self.execute_transaction(tx)
            if tx_hash:
                logger.debug(
                    f"Token withdrawal transaction sent with hash: {tx_hash}"
                )
                return True
            else:
                logger.warning("Failed to send token withdrawal transaction.")
                return False
        except ContractLogicError as e:
            logger.error(f"Contract logic error during token withdrawal: {e}")
            return False
        except Exception as e:
            logger.error(f"Error withdrawing token: {e}")
            return False

    async def transfer_profit_to_account(self, amount: Decimal, account: str) -> bool:
        """
        Transfers profit to another account.

        :param amount: Amount of ETH to transfer.
        :param account: Recipient account address.
        :return: True if successful, else False.
        """
        try:
            transfer_function = self.flashloan_contract.functions.transfer(
                self.web3.to_checksum_address(account), int(amount * Decimal("1e18"))
            )
            tx = await self.build_transaction(transfer_function)
            tx_hash = await self.execute_transaction(tx)
            if tx_hash:
                logger.debug(
                    f"Profit transfer transaction sent with hash: {tx_hash}"
                )
                return True
            else:
                logger.warning("Failed to send profit transfer transaction.")
                return False
        except ContractLogicError as e:
            logger.error(f"Contract logic error during profit transfer: {e}")
            return False
        except Exception as e:
            logger.error(f"Error transferring profit: {e}")
            return False

    async def stop(self) -> None:
        try:
            await self.safety_net.stop()
            await self.nonce_core.stop()
            logger.debug("Stopped 0xBuilder. ")
        except Exception as e:
            logger.error(f"Error stopping 0xBuilder: {e} !")
            raise


#================================== Market Metrics and Monitoring ==================================#

class Market_Monitor:
    MODEL_UPDATE_INTERVAL = 3600  # Update model every hour
    VOLATILITY_THRESHOLD = 0.05  # 5% standard deviation
    LIQUIDITY_THRESHOLD = 100000  # $100,000 in 24h volume

    def __init__(
        self,
        web3: AsyncWeb3,
        configuration: Optional[Configuration],
        api_config: Optional[API_Config],
    ):
        self.web3 = web3
        self.configuration = configuration
        self.api_config = api_config
        self.price_model = LinearRegression()
        self.model_last_updated = 0
        self.price_cache = TTLCache(maxsize=1000, ttl=300)  # Cache for 5 minutes

    async def check_market_conditions(self, token_address: str) -> Dict[str, Any]:
        """
        Check various market conditions for a given token

        :param token_address: Address of the token to check
        :return: Dictionary of market conditions
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

        prices = await self.fetch_historical_prices(token_symbol, days=1)
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

    def _calculate_volatility(self, prices: List[float]) -> float:
        """
        Calculate the volatility of a list of prices.

        :param prices: List of prices
        :return: Volatility as standard deviation of returns
        """
        prices_array = np.array(prices)
        returns = np.diff(prices_array) / prices_array[:-1]
        return np.std(returns)

    async def fetch_historical_prices(self, token_symbol: str, days: int = 30) -> List[float]:
        """
        Fetch historical price data for a given token symbol.

        :param token_symbol: Token symbol to fetch prices for
        :param days: Number of days to fetch prices for
        :return: List of historical prices
        """
        cache_key = f"historical_prices_{token_symbol}_{days}"
        if cache_key in self.price_cache:
            logger.debug(f"Returning cached historical prices for {token_symbol}.")
            return self.price_cache[cache_key]

        prices = await self._fetch_from_services(
            lambda _: self.api_config.fetch_historical_prices(token_symbol, days=days),
            f"historical prices for {token_symbol}"
        )
        if prices:
            self.price_cache[cache_key] = prices
        return prices or []

    async def get_token_volume(self, token_symbol: str) -> float:
        """
        Get the 24-hour trading volume for a given token symbol.

        :param token_symbol: Token symbol to fetch volume for
        :return: 24-hour trading volume
        """
        cache_key = f"token_volume_{token_symbol}"
        if cache_key in self.price_cache:
            logger.debug(f"Returning cached trading volume for {token_symbol}.")
            return self.price_cache[cache_key]

        volume = await self._fetch_from_services(
            lambda _: self.api_config.get_token_volume(token_symbol),
            f"trading volume for {token_symbol}"
        )
        if volume is not None:
            self.price_cache[cache_key] = volume
        return volume or 0.0

    async def _fetch_from_services(self, fetch_func, description: str):
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
                logger.warning(f"Failed to fetch {description} using {service}: {e}")
        logger.warning(f"Failed to fetch {description}.")
        return None

    async def predict_price_movement(self, token_symbol: str) -> float:
        """Predict the next price movement for a given token symbol."""
        current_time = time.time()
        if current_time - self.model_last_updated > self.MODEL_UPDATE_INTERVAL:
            await self._update_price_model(token_symbol)
        prices = await self.fetch_historical_prices(token_symbol, days=1)
        if not prices:
            logger.debug(f"No recent prices available for {token_symbol}.")
            return 0.0
        next_time = np.array([[len(prices)]])
        predicted_price = self.price_model.predict(next_time)[0]
        logger.debug(f"Price prediction for {token_symbol}: {predicted_price}")
        return float(predicted_price)

    async def _update_price_model(self, token_symbol: str):
        """
        Update the price prediction model.

        :param token_symbol: Token symbol to update the model for
        """
        prices = await self.fetch_historical_prices(token_symbol)
        if len(prices) > 10:
            X = np.arange(len(prices)).reshape(-1, 1)
            y = np.array(prices)
            self.price_model.fit(X, y)
            self.model_last_updated = time.time()

    async def is_arbitrage_opportunity(self, target_tx: Dict[str, Any]) -> bool:
        """
        Check if there's an arbitrage opportunity based on the target transaction.

        :param target_tx: Target transaction dictionary
        :return: True if arbitrage opportunity detected, else False
        """

        decoded_tx = await self.decode_transaction_input(target_tx["input"], target_tx["to"])
        if not decoded_tx:
            return False
        path = decoded_tx["params"].get("path", [])
        if len(path) < 2:
            return False
        token_address = path[-1]  # The token being bought
        token_symbol = await self.api_config.get_token_symbol(self.web3, token_address)
        if not token_symbol:
            return False

        prices = await self._get_prices_from_services(token_symbol)
        if len(prices) < 2:
            return False

        price_difference = abs(prices[0] - prices[1])
        average_price = sum(prices) / len(prices)
        if average_price == 0:
            return False
        price_difference_percentage = price_difference / average_price
        if price_difference_percentage > 0.01:
            logger.debug(f"Arbitrage opportunity detected for {token_symbol}")
            return True
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
                logger.warning(f"Failed to get price from {service}: {e}")
        return prices

    async def decode_transaction_input(
        self, input_data: str, contract_address: str
    ) -> Optional[Dict[str, Any]]:
        """
        Decode the input data of a transaction.

        :param input_data: Hexadecimal input data of the transaction.
        :param contract_address: Address of the contract being interacted with.
        :return: Dictionary containing function name and parameters if successful, else None.
        """
        try:
            erc20_abi = await self.api_config._load_abi(self.configuration.ERC20_ABI)
            contract = self.web3.eth.contract(address=contract_address, abi=erc20_abi)
            function_abi, params = contract.decode_function_input(input_data)
            return {"function_name": function_abi["name"], "params": params}
        except Exception as e:
            logger.warning(f"Failed in decoding transaction input: {e}")
            return None

#//////////////////////////////////////////////////////////////////////////////

#============================= Strategy Orchestrator =============================

@dataclass
class StrategyPerformanceMetrics:
    successes: int = 0
    failures: int = 0
    profit: Decimal = Decimal("0")
    avg_execution_time: float = 0.0
    success_rate: float = 0.0
    total_executions: int = 0

@dataclass
class StrategyConfiguration:
    decay_factor: float = 0.95
    min_profit_threshold: Decimal = Decimal("0.01")
    learning_rate: float = 0.01
    exploration_rate: float = 0.1

@dataclass
class StrategyExecutionError(Exception):
    """Exception raised when a strategy execution fails."""
    message: str

    def __str__(self) -> str:
        return self.message

class Strategy_Net:
    def __init__(
        self,
        transaction_core: Optional['Transaction_Core'] = None,
        market_monitor: Optional[Market_Monitor] = None,
        safety_net: Optional['Safety_Net'] = None,
        api_config: Optional[API_Config] = None,
    ) -> None:
        self.transaction_core = transaction_core
        self.market_monitor = market_monitor
        self.safety_net = safety_net
        self.api_config = api_config

        self.strategy_types = [
            "eth_transaction",
            "front_run",
            "back_run",
            "sandwich_attack"
        ]

        self.strategy_performance: Dict[str, StrategyPerformanceMetrics] = {
            strategy_type: StrategyPerformanceMetrics()
            for strategy_type in self.strategy_types
        }

        self.reinforcement_weights: Dict[str, np.ndarray] = {
            strategy_type: np.ones(len(self.get_strategies(strategy_type)))
            for strategy_type in self.strategy_types
        }

        self.configuration = StrategyConfiguration()

        self.history_data: List[Dict[str, Any]] = []

        # Dynamic strategy registry
        self._strategy_registry: Dict[str, List[Callable[[Dict[str, Any]], asyncio.Future]]] = {
            "eth_transaction": [self.high_value_eth_transfer],
            "front_run": [
                self.aggressive_front_run,
                self.predictive_front_run,
                self.volatility_front_run,
                self.advanced_front_run,
            ],
            "back_run": [
                self.price_dip_back_run,
                self.flashloan_back_run,
                self.high_volume_back_run,
                self.advanced_back_run,
            ],
            "sandwich_attack": [
                self.flash_profit_sandwich,
                self.price_boost_sandwich,
                self.arbitrage_sandwich,
                self.advanced_sandwich_attack,
            ],
        }

        logger.debug("Strategy_Net initialized with enhanced configuration")

    def register_strategy(self, strategy_type: str, strategy_func: Callable[[Dict[str, Any]], asyncio.Future]) -> None:
        """Register a new strategy dynamically."""
        if strategy_type not in self.strategy_types:
            logger.warning(f"Attempted to register unknown strategy type: {strategy_type}")
            return
        self._strategy_registry[strategy_type].append(strategy_func)
        self.reinforcement_weights[strategy_type] = np.ones(len(self._strategy_registry[strategy_type]))
        logger.info(f"Registered new strategy '{strategy_func.__name__}' under '{strategy_type}'")

    def get_strategies(self, strategy_type: str) -> List[Callable[[Dict[str, Any]], asyncio.Future]]:
        """Retrieve strategies for a given strategy type."""
        return self._strategy_registry.get(strategy_type, [])

    async def execute_best_strategy(
        self, target_tx: Dict[str, Any], strategy_type: str
    ) -> bool:
        """
        Execute the best strategy for the given strategy type.

        :param target_tx: Target transaction dictionary.
        :param strategy_type: Type of strategy to execute
        :return: True if successful, else False.
        """
        strategies = self.get_strategies(strategy_type)
        if not strategies:
            logger.debug(f"No strategies available for type: {strategy_type}")
            return False

        try:
            start_time = time.time()
            selected_strategy = await self._select_best_strategy(strategies, strategy_type)

            profit_before = await self.transaction_core.get_current_profit()
            success = await selected_strategy(target_tx)
            profit_after = await self.transaction_core.get_current_profit()

            execution_time = time.time() - start_time
            profit_made = profit_after - profit_before

            await self._update_strategy_metrics(
                selected_strategy.__name__,
                strategy_type,
                success,
                profit_made,
                execution_time,
            )

            return success

        except StrategyExecutionError as e:
            logger.error(f"Strategy execution failed: {str(e)}", exc_info=True)
            return False
        except Exception as e:
            logger.exception(f"Unexpected error during strategy execution: {e}")
            return False

    async def _select_best_strategy(
        self, strategies: List[Callable[[Dict[str, Any]], asyncio.Future]], strategy_type: str
    ) -> Callable[[Dict[str, Any]], asyncio.Future]:
        """Select the best strategy based on reinforcement learning weights."""
        weights = self.reinforcement_weights[strategy_type]

        if random.random() < self.configuration.exploration_rate:
            logger.debug("Using exploration for strategy selection")
            return random.choice(strategies)

        # Numerical stability for softmax
        max_weight = np.max(weights)
        exp_weights = np.exp(weights - max_weight)
        probabilities = exp_weights / exp_weights.sum()

        selected_index = np.random.choice(len(strategies), p=probabilities)
        selected_strategy = strategies[selected_index]
        logger.debug(f"Selected strategy '{selected_strategy.__name__}' with weight {weights[selected_index]:.4f}")
        return selected_strategy

    async def _update_strategy_metrics(
        self,
        strategy_name: str,
        strategy_type: str,
        success: bool,
        profit: Decimal,
        execution_time: float,
    ) -> None:
        """Update metrics for the executed strategy."""
        metrics = self.strategy_performance[strategy_type]
        metrics.total_executions += 1

        if success:
            metrics.successes += 1
            metrics.profit += profit
        else:
            metrics.failures += 1

        metrics.avg_execution_time = (
            metrics.avg_execution_time * self.configuration.decay_factor
            + execution_time * (1 - self.configuration.decay_factor)
        )
        metrics.success_rate = metrics.successes / metrics.total_executions

        strategy_index = self.get_strategy_index(strategy_name, strategy_type)
        if strategy_index >= 0:
            reward = self._calculate_reward(success, profit, execution_time)
            self._update_reinforcement_weight(strategy_type, strategy_index, reward)

        self.history_data.append(
            {
                "timestamp": time.time(),
                "strategy_name": strategy_name,
                "success": success,
                "profit": float(profit),
                "execution_time": execution_time,
                "total_profit": float(metrics.profit),
            }
        )

    def get_strategy_index(self, strategy_name: str, strategy_type: str) -> int:
        """Get the index of a strategy in the strategy list."""
        strategies = self.get_strategies(strategy_type)
        for index, strategy in enumerate(strategies):
            if strategy.__name__ == strategy_name:
                return index
        logger.warning(f"Strategy '{strategy_name}' not found in type '{strategy_type}'")
        return -1

    def _calculate_reward(
        self, success: bool, profit: Decimal, execution_time: float
    ) -> float:
        """Calculate the reward for a strategy execution."""
        base_reward = float(profit) if success else -0.1
        time_penalty = -0.01 * execution_time
        total_reward = base_reward + time_penalty
        logger.debug(f"Calculated reward: {total_reward:.4f} (Base: {base_reward}, Time Penalty: {time_penalty})")
        return total_reward

    def _update_reinforcement_weight(
        self, strategy_type: str, index: int, reward: float
    ) -> None:
        """Update the reinforcement learning weight for a strategy."""
        lr = self.configuration.learning_rate
        current_weight = self.reinforcement_weights[strategy_type][index]
        new_weight = current_weight * (1 - lr) + reward * lr
        self.reinforcement_weights[strategy_type][index] = max(0.1, new_weight)
        logger.debug(f"Updated weight for strategy index {index} in '{strategy_type}': {new_weight:.4f}")

    async def _decode_transaction(self, target_tx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Decode transaction input."""
        try:
            decoded = await self.transaction_core.decode_transaction_input(
                target_tx["input"], target_tx["to"]
            )
            logger.debug(f"Decoded transaction: {decoded}")
            return decoded
        except Exception as e:
            logger.error(f"Error decoding transaction: {e}")
            return None

    async def _get_token_symbol(self, token_address: str) -> Optional[str]:
        """Get token symbol from address."""
        try:
            symbol = await self.api_config.get_token_symbol(
                self.transaction_core.web3, token_address
            )
            logger.debug(f"Retrieved token symbol '{symbol}' for address '{token_address}'")
            return symbol
        except Exception as e:
            logger.error(f"Error fetching token symbol: {e}")
            return None

    # ========================= Strategy Implementations =========================

    async def high_value_eth_transfer(self, target_tx: Dict[str, Any]) -> bool:
        """
        Execute high-value ETH transfer strategy with advanced validation and dynamic thresholds.
        
        :param target_tx: Target transaction dictionary
        :return: True if transaction was executed successfully, else False
        """
        logger.debug("Initiating High-Value ETH Transfer Strategy...")

        try:
            # Basic transaction validation
            if not isinstance(target_tx, dict) or not target_tx:
                logger.debug("Invalid transaction format provided!")
                return False

            # Extract transaction details
            eth_value_in_wei = int(target_tx.get("value", 0))
            gas_price = int(target_tx.get("gasPrice", 0))
            to_address = target_tx.get("to", "")

            # Convert values to ETH for readability
            eth_value = self.transaction_core.web3.from_wei(eth_value_in_wei, "ether")
            gas_price_gwei = self.transaction_core.web3.from_wei(gas_price, "gwei")

            # Dynamic threshold based on current gas prices
            base_threshold = self.transaction_core.web3.to_wei(10, "ether")
            if gas_price_gwei > 200:  # High gas price scenario
                threshold = base_threshold * 2  # Double threshold when gas is expensive
            elif gas_price_gwei > 100:
                threshold = base_threshold * 1.5
            else:
                threshold = base_threshold

            # Log detailed transaction analysis
            threshold_eth = self.transaction_core.web3.from_wei(threshold, 'ether')
            logger.debug(
                f"Transaction Analysis:\n"
                f"Value: {eth_value:.4f} ETH\n"
                f"Gas Price: {gas_price_gwei:.2f} Gwei\n"
                f"To Address: {to_address[:10]}...\n"
                f"Current Threshold: {threshold_eth} ETH"
            )

            # Additional validation checks
            if eth_value_in_wei <= 0:
                logger.debug("Transaction value is zero or negative. Skipping...")
                return False

            if not self.transaction_core.web3.is_address(to_address):
                logger.debug("Invalid recipient address. Skipping...")
                return False

            # Check contract interaction
            is_contract = await self._is_contract_address(to_address)
            if is_contract:
                logger.debug("Recipient is a contract. Additional validation required...")
                if not await self._validate_contract_interaction(to_address):
                    return False

            # Execute if value exceeds threshold
            if eth_value_in_wei > threshold:
                logger.info(
                    f"High-value ETH transfer detected:\n"
                    f"Value: {eth_value:.4f} ETH\n"
                    f"Threshold: {threshold_eth} ETH"
                )
                return await self.transaction_core.handle_eth_transaction(target_tx)

            logger.debug(
                f"ETH transaction value ({eth_value:.4f} ETH) below threshold "
                f"({threshold_eth} ETH). Skipping..."
            )
            return False

        except Exception as e:
            logger.error(f"Error in high-value ETH transfer strategy: {e}")
            return False

    async def _is_contract_address(self, address: str) -> bool:
        """Check if address is a contract."""
        try:
            code = await self.transaction_core.web3.eth.get_code(address)
            is_contract = len(code) > 0
            logger.debug(f"Address '{address}' is_contract: {is_contract}")
            return is_contract
        except Exception as e:
            logger.error(f"Error checking if address is contract: {e}")
            return False

    async def _validate_contract_interaction(self, contract_address: str) -> bool:
        """Validate interaction with contract address."""
        try:
            # Example validation: check if it's a known contract
            token_symbols = await self.api_config.get_token_symbols()
            is_valid = contract_address in token_symbols
            logger.debug(f"Contract address '{contract_address}' validation result: {is_valid}")
            return is_valid
        except Exception as e:
            logger.error(f"Error validating contract interaction: {e}")
            return False

    async def aggressive_front_run(self, target_tx: Dict[str, Any]) -> bool:
        """
        Execute aggressive front-run strategy with comprehensive validation,
        dynamic thresholds, and risk assessment.
        """
        logger.debug("Initiating Aggressive Front-Run Strategy...")

        try:
            # Step 1: Basic transaction validation
            if not isinstance(target_tx, dict) or not target_tx:
                logger.debug("Invalid transaction format. Skipping...")
                return False

            # Step 2: Extract and validate key transaction parameters
            tx_value = int(target_tx.get("value", 0))
            tx_hash = target_tx.get("tx_hash", "Unknown")[:10]
            gas_price = int(target_tx.get("gasPrice", 0))

            # Step 3: Calculate value metrics
            value_eth = self.transaction_core.web3.from_wei(tx_value, "ether")
            threshold = self._calculate_dynamic_threshold(gas_price)

            logger.debug(
                f"Transaction Analysis:\n"
                f"Hash: {tx_hash}\n"
                f"Value: {value_eth:.4f} ETH\n"
                f"Gas Price: {self.transaction_core.web3.from_wei(gas_price, 'gwei'):.2f} Gwei\n"
                f"Threshold: {threshold:.4f} ETH"
            )

            # Step 4: Risk assessment
            risk_score = await self._assess_front_run_risk(target_tx)
            if risk_score < 0.5:  # Risk score below threshold
                logger.debug(f"Risk score too high ({risk_score:.2f}). Skipping front-run.")
                return False

            # Step 5: Check opportunity value
            if value_eth >= threshold:
                # Additional validation for high-value transactions
                if value_eth > 10:  # Extra checks for very high value transactions
                    if not await self._validate_high_value_transaction(target_tx):
                        logger.debug("High-value transaction validation failed. Skipping...")
                        return False

                logger.info(
                    f"Executing aggressive front-run:\n"
                    f"Transaction: {tx_hash}\n"
                    f"Value: {value_eth:.4f} ETH\n"
                    f"Risk Score: {risk_score:.2f}"
                )
                return await self.transaction_core.front_run(target_tx)

            logger.debug(
                f"Transaction value {value_eth:.4f} ETH below threshold {threshold:.4f} ETH. Skipping..."
            )
            return False

        except Exception as e:
            logger.error(f"Error in aggressive front-run strategy: {e}")
            return False

    def _calculate_dynamic_threshold(self, gas_price: int) -> float:
        """Calculate dynamic threshold based on current gas prices."""
        gas_price_gwei = float(self.transaction_core.web3.from_wei(gas_price, "gwei"))

        # Base threshold adjusts with gas price
        if gas_price_gwei > 200:
            threshold = 2.0  # Higher threshold when gas is expensive
        elif gas_price_gwei > 100:
            threshold = 1.5
        elif gas_price_gwei > 50:
            threshold = 1.0
        else:
            threshold = 0.5  # Minimum threshold

        logger.debug(f"Dynamic threshold based on gas price {gas_price_gwei} Gwei: {threshold} ETH")
        return threshold

    async def _assess_front_run_risk(self, tx: Dict[str, Any]) -> float:
        """
        Calculate risk score for front-running (0-1 scale).
        Lower score indicates higher risk.
        """
        try:
            risk_score = 1.0

            # Gas price impact
            gas_price = int(tx.get("gasPrice", 0))
            gas_price_gwei = float(self.transaction_core.web3.from_wei(gas_price, "gwei"))
            if gas_price_gwei > 300:
                risk_score *= 0.7  # High gas price increases risk

            # Contract interaction check
            input_data = tx.get("input", "0x")
            if len(input_data) > 10:  # Complex contract interaction
                risk_score *= 0.8

            # Check market conditions
            market_conditions = await self.market_monitor.check_market_conditions(tx.get("to", ""))
            if market_conditions.get("high_volatility", False):
                risk_score *= 0.7
            if market_conditions.get("low_liquidity", False):
                risk_score *= 0.6

            risk_score = max(risk_score, 0.0)  # Ensure non-negative
            logger.debug(f"Assessed front-run risk score: {risk_score:.2f}")
            return round(risk_score, 2)

        except Exception as e:
            logger.error(f"Error assessing front-run risk: {e}")
            return 0.0  # Return maximum risk on error

    async def _validate_high_value_transaction(self, tx: Dict[str, Any]) -> bool:
        """Additional validation for high-value transactions."""
        try:
            # Check if the target address is a known contract
            to_address = tx.get("to", "")
            if not to_address:
                logger.debug("Transaction missing 'to' address.")
                return False

            # Verify code exists at the address
            code = await self.transaction_core.web3.eth.get_code(to_address)
            if not code:
                logger.warning(f"No contract code found at {to_address}")
                return False

            # Check if it's a known token or DEX contract
            token_symbols = await self.api_config.get_token_symbols()
            if to_address not in token_symbols:
                logger.warning(f"Address {to_address} not in known token list")
                return False

            logger.debug(f"High-value transaction validated for address {to_address}")
            return True

        except Exception as e:
            logger.error(f"Error validating high-value transaction: {e}")
            return False

    async def predictive_front_run(self, target_tx: Dict[str, Any]) -> bool:
        """
        Execute predictive front-run strategy based on advanced price prediction analysis
        and multiple market indicators.
        """
        logger.debug("Initiating Enhanced Predictive Front-Run Strategy...")

        try:
            # Step 1: Validate and decode transaction
            decoded_tx = await self._decode_transaction(target_tx)
            if not decoded_tx:
                logger.debug("Failed to decode transaction. Skipping...")
                return False

            path = decoded_tx.get("params", {}).get("path", [])
            if not path or len(path) < 2:
                logger.debug("Invalid or missing path parameter. Skipping...")
                return False

            # Step 2: Get token details and validate
            token_address = path[0]
            token_symbol = await self._get_token_symbol(token_address)
            if not token_symbol:
                logger.debug(f"Cannot get token symbol for {token_address}. Skipping...")
                return False

            # Step 3: Gather market data asynchronously
            try:
                data = await asyncio.gather(
                    self.market_monitor.predict_price_movement(token_symbol),
                    self.api_config.get_real_time_price(token_symbol),
                    self.market_monitor.check_market_conditions(target_tx["to"]),
                    self.market_monitor.fetch_historical_prices(token_symbol, days=1),
                    return_exceptions=True
                )
                predicted_price, current_price, market_conditions, historical_prices = data

                if any(isinstance(x, Exception) for x in data):
                    logger.warning("Failed to gather complete market data.")
                    return False

                if current_price is None or predicted_price is None:
                    logger.debug("Missing price data for analysis.")
                    return False

            except Exception as e:
                logger.error(f"Error gathering market data: {e}")
                return False

            # Step 4: Calculate price metrics
            price_change = (predicted_price / float(current_price) - 1) * 100
            volatility = np.std(historical_prices) / np.mean(historical_prices) if historical_prices else 0

            # Step 5: Score the opportunity (0-100)
            opportunity_score = await self._calculate_opportunity_score(
                price_change=price_change,
                volatility=volatility,
                market_conditions=market_conditions,
                current_price=current_price,
                historical_prices=historical_prices
            )

            # Log detailed analysis
            logger.debug(
                f"Predictive Analysis for {token_symbol}:\n"
                f"Current Price: {current_price:.6f}\n"
                f"Predicted Price: {predicted_price:.6f}\n"
                f"Expected Change: {price_change:.2f}%\n"
                f"Volatility: {volatility:.2f}\n"
                f"Opportunity Score: {opportunity_score}/100\n"
                f"Market Conditions: {market_conditions}"
            )

            # Step 6: Execute if conditions are favorable
            if opportunity_score >= 75:  # High confidence threshold
                logger.info(
                    f"Executing predictive front-run for {token_symbol} "
                    f"(Score: {opportunity_score}/100, Expected Change: {price_change:.2f}%)"
                )
                return await self.transaction_core.front_run(target_tx)

            logger.debug(
                f"Opportunity score {opportunity_score}/100 below threshold. Skipping front-run."
            )
            return False

        except Exception as e:
            logger.error(f"Error in predictive front-run strategy: {e}")
            return False

    async def _calculate_opportunity_score(
        self,
        price_change: float,
        volatility: float,
        market_conditions: Dict[str, bool],
        current_price: float,
        historical_prices: List[float]
    ) -> float:
        """
        Calculate comprehensive opportunity score (0-100) based on multiple metrics.
        Higher score indicates more favorable conditions for front-running.
        """
        score = 0

        # Price change component (0-40 points)
        if price_change > 5.0:        # Very strong upward prediction
            score += 40
        elif price_change > 3.0:      # Strong upward prediction
            score += 30
        elif price_change > 1.0:      # Moderate upward prediction
            score += 20
        elif price_change > 0.5:      # Slight upward prediction
            score += 10

        # Volatility component (0-20 points)
        # Lower volatility is better for predictable outcomes
        if volatility < 0.02:         # Very low volatility
            score += 20
        elif volatility < 0.05:       # Low volatility
            score += 15
        elif volatility < 0.08:       # Moderate volatility
            score += 10

        # Market conditions component (0-20 points)
        if market_conditions.get("bullish_trend", False):
            score += 10
        if not market_conditions.get("high_volatility", True):
            score += 5
        if not market_conditions.get("low_liquidity", True):
            score += 5

        # Price trend component (0-20 points)
        if historical_prices and len(historical_prices) > 1:
            recent_trend = (historical_prices[-1] / historical_prices[0] - 1) * 100
            if recent_trend > 0:      # Upward trend
                score += 20
            elif recent_trend > -1:    # Stable trend
                score += 10

        logger.debug(f"Calculated opportunity score: {score}/100")
        return score

    async def volatility_front_run(self, target_tx: Dict[str, Any]) -> bool:
        """
        Execute front-run strategy based on market volatility analysis with 
        advanced risk assessment and dynamic thresholds.
        """
        logger.debug("Initiating Enhanced Volatility Front-Run Strategy...")

        try:
            # Extract and validate transaction data
            decoded_tx = await self._decode_transaction(target_tx)
            if not decoded_tx:
                logger.debug("Failed to decode transaction. Skipping...")
                return False

            path = decoded_tx.get("params", {}).get("path", [])
            if not path or len(path) < 2:
                logger.debug("Invalid or missing path parameter. Skipping...")
                return False

            # Get token details and price data
            token_address = path[0]
            token_symbol = await self._get_token_symbol(token_address)
            if not token_symbol:
                logger.debug(f"Cannot get token symbol for {token_address}. Skipping...")
                return False

            # Gather market data asynchronously
            results = await asyncio.gather(
                self.market_monitor.check_market_conditions(target_tx["to"]),
                self.api_config.get_real_time_price(token_symbol),
                self.market_monitor.fetch_historical_prices(token_symbol, days=1),
                return_exceptions=True
            )

            market_conditions, current_price, historical_prices = results

            if any(isinstance(result, Exception) for result in results):
                logger.warning("Failed to gather complete market data")
                return False

            # Calculate volatility metrics
            volatility_score = await self._calculate_volatility_score(
                historical_prices=historical_prices,
                current_price=current_price,
                market_conditions=market_conditions
            )

            # Log detailed analysis
            logger.debug(
                f"Volatility Analysis for {token_symbol}:\n"
                f"Volatility Score: {volatility_score:.2f}/100\n"
                f"Current Price: {current_price}\n"
                f"24h Price Range: {min(historical_prices):.4f} - {max(historical_prices):.4f}\n"
                f"Market Conditions: {market_conditions}"
            )

            # Execute based on volatility thresholds
            if volatility_score >= 75:  # High volatility threshold
                logger.info(
                    f"Executing volatility-based front-run for {token_symbol} "
                    f"(Volatility Score: {volatility_score:.2f}/100)"
                )
                return await self.transaction_core.front_run(target_tx)

            logger.debug(
                f"Volatility score {volatility_score:.2f}/100 below threshold. Skipping front-run."
            )
            return False

        except Exception as e:
            logger.error(f"Error in volatility front-run strategy: {e}")
            return False

    async def _calculate_volatility_score(
        self,
        historical_prices: List[float],
        current_price: float,
        market_conditions: Dict[str, bool]
    ) -> float:
        """
        Calculate comprehensive volatility score (0-100) based on multiple metrics.
        Higher score indicates more favorable volatility conditions.
        """
        score = 0

        # Calculate price volatility metrics
        if len(historical_prices) > 1:
            price_changes = np.diff(historical_prices) / np.array(historical_prices[:-1])
            volatility = np.std(price_changes)
            price_range = (max(historical_prices) - min(historical_prices)) / np.mean(historical_prices)

            # Volatility component (0-40 points)
            if volatility > 0.1:  # Very high volatility
                score += 40
            elif volatility > 0.05:  # High volatility
                score += 30
            elif volatility > 0.02:  # Moderate volatility
                score += 20

            # Price range component (0-30 points)
            if price_range > 0.15:  # Wide price range
                score += 30
            elif price_range > 0.08:  # Moderate price range
                score += 20
            elif price_range > 0.03:  # Narrow price range
                score += 10

        # Market conditions component (0-30 points)
        if market_conditions.get("high_volatility", False):
            score += 15
        if market_conditions.get("bullish_trend", False):
            score += 10
        if not market_conditions.get("low_liquidity", True):
            score += 5

        logger.debug(f"Calculated volatility score: {score}/100")
        return score

    async def advanced_front_run(self, target_tx: Dict[str, Any]) -> bool:
        """
        Execute advanced front-run strategy with comprehensive analysis, risk management,
        and multi-factor decision making.
        """
        logger.debug("Initiating Advanced Front-Run Strategy...")

        # Step 1: Validate transaction and decode
        try:
            decoded_tx = await self._decode_transaction(target_tx)
            if not decoded_tx:
                logger.debug("Failed to decode transaction. Skipping...")
                return False

            # Extract and validate path
            path = decoded_tx.get("params", {}).get("path", [])
            if not path or len(path) < 2:
                logger.debug("Invalid or missing path parameter. Skipping...")
                return False

            # Get token details
            token_address = path[0]
            token_symbol = await self._get_token_symbol(token_address)
            if not token_symbol:
                logger.debug(f"Cannot get token symbol for {token_address}. Skipping...")
                return False

        except Exception as e:
            logger.error(f"Error in transaction validation: {e}")
            return False

        try:
            # Step 2: Multi-factor analysis
            analysis_results = await asyncio.gather(
                self.market_monitor.predict_price_movement(token_symbol),
                self.market_monitor.check_market_conditions(target_tx["to"]),
                self.api_config.get_real_time_price(token_symbol),
                self.api_config.get_token_volume(token_symbol),
                return_exceptions=True
            )

            predicted_price, market_conditions, current_price, volume = analysis_results

            if any(isinstance(result, Exception) for result in analysis_results):
                logger.warning("Failed to gather complete market data.")
                return False

            if current_price is None or predicted_price is None:
                logger.debug("Missing price data for analysis. Skipping...")
                return False

            # Step 3: Advanced decision making
            price_increase = (predicted_price / float(current_price) - 1) * 100
            is_bullish = market_conditions.get("bullish_trend", False)
            is_volatile = market_conditions.get("high_volatility", False)
            has_liquidity = not market_conditions.get("low_liquidity", True)

            # Calculate risk score (0-100)
            risk_score = self._calculate_risk_score(
                price_increase=price_increase,
                is_bullish=is_bullish,
                is_volatile=is_volatile,
                has_liquidity=has_liquidity,
                volume=volume
            )

            # Log detailed analysis
            logger.debug(
                f"Analysis for {token_symbol}:\n"
                f"Price Increase: {price_increase:.2f}%\n"
                f"Market Trend: {'Bullish' if is_bullish else 'Bearish'}\n"
                f"Volatility: {'High' if is_volatile else 'Low'}\n"
                f"Liquidity: {'Adequate' if has_liquidity else 'Low'}\n"
                f"24h Volume: ${volume:,.2f}\n"
                f"Risk Score: {risk_score}/100"
            )

            # Step 4: Execute if conditions are favorable
            if risk_score >= 75:  # Minimum risk score threshold
                logger.info(
                    f"Executing advanced front-run for {token_symbol} "
                    f"(Risk Score: {risk_score}/100)"
                )
                return await self.transaction_core.front_run(target_tx)

            logger.debug(
                f"Risk score {risk_score}/100 below threshold. Skipping front-run."
            )
            return False

        except Exception as e:
            logger.error(f"Error in advanced front-run analysis: {e}")
            return False

    def _calculate_risk_score(
        self,
        price_increase: float,
        is_bullish: bool,
        is_volatile: bool,
        has_liquidity: bool,
        volume: float
    ) -> int:
        """
        Calculate a risk score from 0-100 based on multiple factors.
        Higher score indicates more favorable conditions.
        """
        score = 0

        # Price momentum (0-30 points)
        if price_increase >= 5.0:
            score += 30
        elif price_increase >= 3.0:
            score += 20
        elif price_increase >= 1.0:
            score += 10

        # Market trend (0-20 points)
        if is_bullish:
            score += 20

        # Volatility (0-15 points)
        if not is_volatile:
            score += 15

        # Liquidity (0-20 points)
        if has_liquidity:
            score += 20

        # Volume-based score (0-15 points)
        if volume >= 1_000_000:  # $1M+ daily volume
            score += 15
        elif volume >= 500_000:   # $500k+ daily volume
            score += 10
        elif volume >= 100_000:   # $100k+ daily volume
            score += 5

        logger.debug(f"Calculated risk score: {score}/100")
        return score

    async def price_dip_back_run(self, target_tx: Dict[str, Any]) -> bool:
        """Execute back-run strategy based on price dip prediction."""
        logger.debug("Initiating Price Dip Back-Run Strategy...")
        decoded_tx = await self._decode_transaction(target_tx)
        if not decoded_tx:
            return False
        path = decoded_tx.get("params", {}).get("path", [])
        if not path:
            logger.debug("Transaction has no path parameter. Skipping...")
            return False
        token_symbol = await self._get_token_symbol(path[-1])
        if not token_symbol:
            return False
        current_price = await self.api_config.get_real_time_price(token_symbol)
        if current_price is None:
            return False
        predicted_price = await self.market_monitor.predict_price_movement(token_symbol)
        if predicted_price < float(current_price) * 0.99:
            logger.debug("Predicted price decrease exceeds threshold, proceeding with back-run.")
            return await self.transaction_core.back_run(target_tx)
        logger.debug("Predicted price decrease does not meet threshold. Skipping back-run.")
        return False

    async def flashloan_back_run(self, target_tx: Dict[str, Any]) -> bool:
        """Execute back-run strategy using flash loans."""
        logger.debug("Initiating Flashloan Back-Run Strategy...")
        estimated_amount = await self.transaction_core.calculate_flashloan_amount(target_tx)
        estimated_profit = estimated_amount * Decimal("0.02")
        if estimated_profit > self.configuration.min_profit_threshold:
            logger.debug(f"Estimated profit: {estimated_profit} ETH meets threshold.")
            return await self.transaction_core.back_run(target_tx)
        logger.debug("Profit is insufficient for flashloan back-run. Skipping.")
        return False

    async def high_volume_back_run(self, target_tx: Dict[str, Any]) -> bool:
        """Execute back-run strategy based on high trading volume."""
        logger.debug("Initiating High Volume Back-Run Strategy...")
        token_address = target_tx.get("to")
        token_symbol = await self._get_token_symbol(token_address)
        if not token_symbol:
            return False
        volume_24h = await self.api_config.get_token_volume(token_symbol)
        volume_threshold = self._get_volume_threshold(token_symbol)
        if volume_24h > volume_threshold:
            logger.debug(f"High volume detected (${volume_24h:,.2f} USD), proceeding with back-run.")
            return await self.transaction_core.back_run(target_tx)
        logger.debug(f"Volume (${volume_24h:,.2f} USD) below threshold (${volume_threshold:,.2f} USD). Skipping.")
        return False

    def _get_volume_threshold(self, token_symbol: str) -> float:
        """
        Determine the volume threshold for a token based on market cap tiers and liquidity.
        Returns threshold in USD.
        """
        # Define volume thresholds for different token tiers
        tier1_tokens = {
            "WETH": 15_000_000,
            "ETH": 15_000_000,
            "WBTC": 25_000_000,
            "USDT": 50_000_000,
            "USDC": 50_000_000,
            "DAI": 20_000_000,
        }

        tier2_tokens = {
            "UNI": 5_000_000,
            "LINK": 8_000_000,
            "AAVE": 3_000_000,
            "MKR": 2_000_000,
            "CRV": 4_000_000,
            "SUSHI": 2_000_000,
            "SNX": 2_000_000,
            "COMP": 2_000_000,
        }

        tier3_tokens = {
            "1INCH": 1_000_000,
            "YFI": 1_500_000,
            "BAL": 1_000_000,
            "PERP": 800_000,
            "DYDX": 1_200_000,
            "LDO": 1_500_000,
            "RPL": 700_000,
        }

        volatile_tokens = {
            "SHIB": 8_000_000,
            "PEPE": 5_000_000,
            "DOGE": 10_000_000,
            "FLOKI": 3_000_000,
        }

        # Check each tier in order
        if token_symbol in tier1_tokens:
            threshold = tier1_tokens[token_symbol]
        elif token_symbol in tier2_tokens:
            threshold = tier2_tokens[token_symbol]
        elif token_symbol in tier3_tokens:
            threshold = tier3_tokens[token_symbol]
        elif token_symbol in volatile_tokens:
            threshold = volatile_tokens[token_symbol]
        else:
            threshold = 500_000  # Conservative default for unknown tokens

        logger.debug(f"Volume threshold for '{token_symbol}': ${threshold:,.2f} USD")
        return threshold

    async def advanced_back_run(self, target_tx: Dict[str, Any]) -> bool:
        """Execute advanced back-run strategy with comprehensive analysis."""
        logger.debug("Initiating Advanced Back-Run Strategy...")
        decoded_tx = await self._decode_transaction(target_tx)
        if not decoded_tx:
            return False
        market_conditions = await self.market_monitor.check_market_conditions(
            target_tx["to"]
        )
        if market_conditions.get("high_volatility", False) and market_conditions.get(
            "bullish_trend", False
        ):
            logger.debug("Market conditions favorable for advanced back-run.")
            return await self.transaction_core.back_run(target_tx)
        logger.debug("Market conditions unfavorable for advanced back-run. Skipping.")
        return False

    async def flash_profit_sandwich(self, target_tx: Dict[str, Any]) -> bool:
        """Execute sandwich attack strategy using flash loans."""
        logger.debug("Initiating Flash Profit Sandwich Strategy...")
        estimated_amount = await self.transaction_core.calculate_flashloan_amount(target_tx)
        estimated_profit = estimated_amount * Decimal("0.02")
        if estimated_profit > self.configuration.min_profit_threshold:
            gas_price = await self.transaction_core.get_dynamic_gas_price()
            if gas_price > 200:
                logger.debug(f"Gas price too high for sandwich attack: {gas_price} Gwei")
                return False
            logger.debug(f"Executing sandwich with estimated profit: {estimated_profit:.4f} ETH")
            return await self.transaction_core.execute_sandwich_attack(target_tx)
        logger.debug("Insufficient profit potential for flash sandwich. Skipping.")
        return False

    async def price_boost_sandwich(self, target_tx: Dict[str, Any]) -> bool:
        """Execute sandwich attack strategy based on price momentum."""
        logger.debug("Initiating Price Boost Sandwich Strategy...")
        decoded_tx = await self._decode_transaction(target_tx)
        if not decoded_tx:
            return False
        path = decoded_tx.get("params", {}).get("path", [])
        if not path:
            logger.debug("Transaction has no path parameter. Skipping...")
            return False
        token_symbol = await self._get_token_symbol(path[0])
        if not token_symbol:
            return False
        historical_prices = await self.market_monitor.fetch_historical_prices(token_symbol)
        if not historical_prices:
            return False
        momentum = await self._analyze_price_momentum(historical_prices)
        if momentum > 0.02:
            logger.debug(f"Strong price momentum detected: {momentum:.2%}")
            return await self.transaction_core.execute_sandwich_attack(target_tx)
        logger.debug(f"Insufficient price momentum: {momentum:.2%}. Skipping.")
        return False

    async def _analyze_price_momentum(self, prices: List[float]) -> float:
        """Analyze price momentum from historical prices."""
        if not prices or len(prices) < 2:
            logger.debug("Insufficient historical prices for momentum analysis.")
            return 0.0
        price_changes = [prices[i] / prices[i - 1] - 1 for i in range(1, len(prices))]
        momentum = sum(price_changes) / len(price_changes)
        logger.debug(f"Calculated price momentum: {momentum:.4f}")
        return momentum

    async def arbitrage_sandwich(self, target_tx: Dict[str, Any]) -> bool:
        """Execute sandwich attack strategy based on arbitrage opportunities."""
        logger.debug("Initiating Arbitrage Sandwich Strategy...")
        decoded_tx = await self._decode_transaction(target_tx)
        if not decoded_tx:
            return False
        path = decoded_tx.get("params", {}).get("path", [])
        if not path:
            logger.debug("Transaction has no path parameter. Skipping...")
            return False
        token_symbol = await self._get_token_symbol(path[-1])
        if not token_symbol:
            return False
        is_arbitrage = await self.market_monitor.is_arbitrage_opportunity(target_tx)
        if is_arbitrage:
            logger.debug(f"Arbitrage opportunity detected for {token_symbol}")
            return await self.transaction_core.execute_sandwich_attack(target_tx)
        logger.debug("No profitable arbitrage opportunity found. Skipping.")
        return False

    async def advanced_sandwich_attack(self, target_tx: Dict[str, Any]) -> bool:
        """Execute advanced sandwich attack strategy with risk management."""
        logger.debug("Initiating Advanced Sandwich Attack...")
        decoded_tx = await self._decode_transaction(target_tx)
        if not decoded_tx:
            return False
        market_conditions = await self.market_monitor.check_market_conditions(
            target_tx["to"]
        )
        if market_conditions.get("high_volatility", False) and market_conditions.get(
            "bullish_trend", False
        ):
            logger.debug("Conditions favorable for sandwich attack.")
            return await self.transaction_core.execute_sandwich_attack(target_tx)
        logger.debug("Conditions unfavorable for sandwich attack. Skipping.")
        return False

    # ========================= End of Strategy Implementations =========================



    # ========================= Main Execution Loop =========================
class Main_Core:
    """
    Builds and manages the entire MEV bot, initializing all components,
    managing connections, and orchestrating the main execution loop.
    """

    def __init__(self, configuration: Optional[Configuration] = None) -> None:
        self.configuration = configuration
        self.web3: Optional[AsyncWeb3] = None
        self.account: Optional[Account] = None
        self.components: Dict[str, Any] = {
            'api_config': None, 
            'nonce_core': None,
            'safety_net': None,
            'market_monitor': None,
            'mempool_monitor': None,
            'transaction_core': None,
            'strategy_net': None,
        }
        logger.info("Main_Core initialized successfully.")

    async def initialize(self) -> None:
        """Initialize all components with error handling."""
        try:
            # Initialize account first
            wallet_key = self.configuration.WALLET_KEY
            if not wallet_key:
                raise ValueError("Wallet key is not set in configuration.")

            try:
                # Remove '0x' prefix if present and ensure the key is valid hex
                cleaned_key = wallet_key[2:] if wallet_key.startswith('0x') else wallet_key
                if not all(c in '0123456789abcdefABCDEF' for c in cleaned_key):
                    raise ValueError("Invalid wallet key format - must be hexadecimal")
                # Add '0x' prefix back if it was removed
                full_key = f"0x{cleaned_key}" if not wallet_key.startswith('0x') else wallet_key
                self.account = Account.from_key(full_key)
            except Exception as e:
                raise ValueError(f"Invalid wallet key format: {e}")

            # Initialize web3 after account is set up
            self.web3 = await self._initialize_web3()
            if not self.web3:
                raise RuntimeError("Failed to initialize Web3 connection")

            if not self.account:
                raise RuntimeError("Failed to initialize account")

            await self._check_account_balance()
            await self._initialize_components()
            logger.info("All components initialized successfully.")
        except Exception as e:
            logger.error(f"Fatal error during initialization: {e}!")
            await self.stop()

    async def _initialize_web3(self) -> Optional[AsyncWeb3]:
        """Initialize Web3 connection with multiple provider fallback."""
        providers = self._get_providers()
        if not providers:
            logger.error("No valid endpoints provided.")
            return None

        for provider_name, provider in providers:
            try:
                logger.debug(f"Attempting connection with {provider_name}...")
                web3 = AsyncWeb3(provider, modules={"eth": (AsyncEth,)})

                if await self._test_connection(web3, provider_name):
                    await self._add_middleware(web3)
                    return web3

            except Exception as e:
                logger.warning(f"{provider_name} connection failed: {e}")
                continue

        return None

    def _get_providers(self) -> List[Tuple[str, Union[AsyncIPCProvider, AsyncHTTPProvider, WebSocketProvider]]]:
        """Get list of available providers with validation."""
        providers = []
        if self.configuration.IPC_ENDPOINT and os.path.exists(self.configuration.IPC_ENDPOINT):
            providers.append(("IPC", AsyncIPCProvider(self.configuration.IPC_ENDPOINT)))
        if self.configuration.HTTP_ENDPOINT:
            providers.append(("HTTP", AsyncHTTPProvider(self.configuration.HTTP_ENDPOINT)))
        if self.configuration.WEBSOCKET_ENDPOINT:
            providers.append(("WebSocket", WebSocketProvider(self.configuration.WEBSOCKET_ENDPOINT)))
        return providers

    async def _test_connection(self, web3: AsyncWeb3, name: str) -> bool:
        """Test Web3 connection with retries."""
        for attempt in range(3):
            try:
                if await web3.is_connected():
                    chain_id = await web3.eth.chain_id
                    logger.info(f"Connected to network {name} (Chain ID: {chain_id})")
                    return True
            except Exception as e:
                logger.error(f"Connection attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(1)
        return False

    async def _add_middleware(self, web3: AsyncWeb3) -> None:
        """Add appropriate middleware based on network."""
        try:
            chain_id = await web3.eth.chain_id
            if chain_id in {99, 100, 77, 7766, 56}:  # POA networks
                web3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
                logger.debug("Injected POA middleware.")
            elif chain_id in {1, 3, 4, 5, 42, 420}:  # ETH networks
                web3.middleware_onion.add(SignAndSendRawMiddlewareBuilder(self.account))
                logger.debug("Injected middleware for ETH network.")
            else:
                logger.warning("Unknown network; no middleware injected.")
        except Exception as e:
            logger.error(f"Middleware configuration failed: {e}")
            raise

    async def _check_account_balance(self) -> None:
        """Check the Ethereum account balance."""
        try:
            if not self.account:
                raise ValueError("Account not initialized")

            balance = await self.web3.eth.get_balance(self.account.address)
            balance_eth = self.web3.from_wei(balance, 'ether')

            logger.info(f"Account {self.account.address} initialized")
            logger.debug(f"Balance: {balance_eth:.4f} ETH")

            if balance_eth < 0.01:
                logger.warning("Low account balance (<0.01 ETH)")

        except Exception as e:
            logger.error(f"Balance check failed: {e}")
            raise

    async def _initialize_components(self) -> None:
        """Initialize all bot components with error handling."""
        try:
            # Initialize core components
            self.components['api_config'] = API_Config(self.configuration)
            self.components['nonce_core'] = Nonce_Core(
                self.web3, self.account.address, self.configuration
            )
            await self.components['nonce_core'].initialize()

            self.components['safety_net'] = Safety_Net(
                self.web3, self.configuration, self.account, self.components['api_config']
            )

            # Load contract ABIs
            erc20_abi = await self._load_abi(self.configuration.ERC20_ABI)
            aave_flashloan_abi = await self._load_abi(self.configuration.AAVE_FLASHLOAN_ABI)
            aave_lending_pool_abi = await self._load_abi(self.configuration.AAVE_LENDING_POOL_ABI)

            # Initialize analysis components
            self.components['market_monitor'] = Market_Monitor(
                self.web3, self.configuration, self.components['api_config']
            )

            # Initialize monitoring components
            self.components['mempool_monitor'] = Mempool_Monitor(
                web3=self.web3,
                safety_net=self.components['safety_net'],
                nonce_core=self.components['nonce_core'],
                api_config=self.components['api_config'],
                monitored_tokens=await self.configuration.get_token_addresses(),
                erc20_abi=erc20_abi,
                configuration=self.configuration
            )

            # Initialize transaction components
            self.components['transaction_core'] = Transaction_Core(
                web3=self.web3,
                account=self.account,
                aave_flashloan_address=self.configuration.AAVE_FLASHLOAN_ADDRESS,
                aave_flashloan_abi=aave_flashloan_abi,
                aave_lending_pool_address=self.configuration.AAVE_LENDING_POOL_ADDRESS,
                aave_lending_pool_abi=aave_lending_pool_abi,
                monitor=self.components['mempool_monitor'],
                nonce_core=self.components['nonce_core'],
                safety_net=self.components['safety_net'],
                api_config=self.components['api_config'],
                configuration=self.configuration,
                erc20_abi=erc20_abi
            )
            await self.components['transaction_core'].initialize()

            # Initialize strategy components
            self.components['strategy_net'] = Strategy_Net(
                transaction_core=self.components['transaction_core'],
                market_monitor=self.components['market_monitor'],
                safety_net=self.components['safety_net'],
                api_config=self.components['api_config'],
            )

        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            raise

    async def run(self) -> None:
        """Main execution loop with improved error handling."""
        logger.debug("Starting Main_Core...")

        try:
            required_components = ['mempool_monitor', 'strategy_net', 'transaction_core']
            if not all(component in self.components and self.components[component] for component in required_components):
                raise RuntimeError("Required components are not properly initialized")

            await self.components['mempool_monitor'].start_monitoring()

            while True:
                try:
                    await self._process_profitable_transactions()
                    await asyncio.sleep(1)
                except asyncio.CancelledError:
                    logger.info("Received cancellation signal")
                    break
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    await asyncio.sleep(5)  # Back off on error

        except KeyboardInterrupt:
            logger.warning("Received shutdown signal...")
        except Exception as e:
            logger.error(f"Critical error in main loop: {e}")
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Graceful shutdown of all components."""
        logger.warning("Shutting down Main_Core...")

        try:
            if 'mempool_monitor' in self.components and self.components['mempool_monitor']:
                await self.components['mempool_monitor'].stop()

            if 'transaction_core' in self.components and self.components['transaction_core']:
                await self.components['transaction_core'].stop()
            
            if 'nonce_core' in self.components and self.components['nonce_core']:
                await self.components['nonce_core'].stop()

            if self.web3:  # Close Web3 connection
                await self.web3.provider.disconnect()
                
        except Exception as e: 
            logger.error(f"Error during shutdown: {e}")
        finally:
            sys.exit(0)
                
    async def _process_profitable_transactions(self) -> None:
        """
        Process profitable transactions from the queue with enhanced validation,
        performance monitoring, and error recovery.
        """
        monitor = self.components['mempool_monitor']
        strategy = self.components['strategy_net']
        
        while not monitor.profitable_transactions.empty():
            start_time = time.time()
            tx = None
            
            try:
                # Get transaction with timeout
                tx = await asyncio.wait_for(
                    monitor.profitable_transactions.get(), 
                    timeout=5.0
                )
                
                # Validate transaction format
                if not self._validate_transaction(tx):
                    logger.warning("Invalid transaction format, skipping...")
                    continue
                    
                tx_hash = tx.get('tx_hash', 'Unknown')[:10]
                strategy_type = self._determine_strategy_type(tx)
                
                # Log detailed transaction info
                logger.debug(
                    f"Processing Transaction:\n"
                    f"Hash: {tx_hash}\n"
                    f"Strategy: {strategy_type}\n"
                    f"Value: {self.web3.from_wei(tx.get('value', 0), 'ether'):.4f} ETH\n"
                    f"Gas Price: {self.web3.from_wei(tx.get('gasPrice', 0), 'gwei'):.1f} Gwei"
                )

                # Check if transaction is still valid
                if not await self._is_tx_still_valid(tx):
                    logger.debug(f"Transaction {tx_hash} is no longer valid, skipping...")
                    continue

                # Execute strategy with timeout
                success = await asyncio.wait_for(
                    strategy.execute_best_strategy(tx, strategy_type),
                    timeout=30.0
                )

                # Log execution metrics
                execution_time = time.time() - start_time
                self._log_execution_metrics(tx_hash, success, execution_time)

                if success:
                    logger.info(
                        f"Successfully executed strategy for {tx_hash} "
                        f"({execution_time:.2f}s)"
                    )
                else:
                    logger.warning(
                        f"Strategy execution failed for {tx_hash} "
                        f"({execution_time:.2f}s)"
                    )

            except asyncio.TimeoutError:
                logger.warning(
                    f"Timeout processing transaction after {time.time() - start_time:.2f}s"
                )
                
            except Exception as e:
                tx_hash = tx.get('tx_hash', 'Unknown')[:10] if tx else 'Unknown'
                logger.error(
                    f"Error processing tx {tx_hash}: {str(e)}\n"
                    f"Stack trace: {traceback.format_exc()}"
                )
                
            finally:
                # Always mark task as done
                if tx:
                    monitor.profitable_transactions.task_done()
                    
        # Log queue statistics
        logger.debug(
            f"Queue status: {monitor.profitable_transactions.qsize()} transactions remaining"
        )

    def _validate_transaction(self, tx: Dict[str, Any]) -> bool:
        """Validate transaction format and required fields."""
        required_fields = ['tx_hash', 'value', 'gasPrice', 'to']
        return (
            isinstance(tx, dict) 
            and all(field in tx for field in required_fields)
            and all(tx[field] is not None for field in required_fields)
        )

    def _determine_strategy_type(self, tx: Dict[str, Any]) -> str:
        """Determine best strategy type based on transaction properties."""
        if tx.get('value', 0) > 0:
            return 'eth_transaction'
        elif self._is_token_swap(tx):
            if tx.get('gasPrice', 0) > self.web3.to_wei('200', 'gwei'):
                return 'back_run'
            else:
                return 'front_run' 
        return 'sandwich_attack'

    async def _is_tx_still_valid(self, tx: Dict[str, Any]) -> bool:
        """Check if transaction is still valid and pending."""
        try:
            tx_hash = tx.get('tx_hash')
            if not tx_hash:
                return False
            tx_status = await self.web3.eth.get_transaction(tx_hash)
            return tx_status is not None and tx_status.blockNumber is None
        except Exception:
            return False

    def _is_token_swap(self, tx: Dict[str, Any]) -> bool:
        """Check if transaction is a token swap."""
        return (
            len(tx.get('input', '0x')) > 10 
            and tx.get('value', 0) == 0
        )

    def _log_execution_metrics(self, tx_hash: str, success: bool, execution_time: float) -> None:
        """Log detailed execution metrics."""
        try:
            import psutil
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
        except ImportError:
            memory_usage = 0.0  # psutil not available

        logger.debug(
            f"Execution Metrics for {tx_hash}:\n"
            f"Success: {success}\n"
            f"Execution Time: {execution_time:.2f}s\n"
            f"Memory Usage: {memory_usage:.1f}MB"
        )

    async def _load_abi(self, abi_path: str) -> List[Dict[str, Any]]:
        """Load contract abi from a file."""
        try:
            with open(abi_path, 'r') as file:
                abi = json.load(file)
            await loading_bar(f"Loaded ABI from {abi_path} successfully.", 0.1)
            return abi
        except Exception as e:
            logger.warning(f"Failed to load ABI from {abi_path}: {e}!")
            raise

# ////////////////////////////////////////////////////////////////////////////

async def main():
    """Main entry point with setup and error handling."""
    global logger
    try:
        # Initialize configuration
        configuration = Configuration()
        await configuration.load()

        # Initialize and run the bot
        main_core = Main_Core(configuration)
        await main_core.initialize()
        await main_core.run()

    except KeyboardInterrupt:
        logger.debug("Shutdown complete.")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())