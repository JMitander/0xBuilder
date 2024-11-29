import os
import sys
import json
import time
import random
import asyncio
import aiohttp
import aiofiles
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional, List, Dict, Any, Callable, Union, Tuple

from web3 import AsyncWeb3, AsyncHTTPProvider, AsyncIPCProvider, WebSocketProvider  
from web3.exceptions import ContractLogicError, TransactionNotFound
from web3.middleware import ExtraDataToPOAMiddleware, SignAndSendRawMiddlewareBuilder
from web3.contract import *
from web3.eth import AsyncEth
from eth_account import Account
from web3.geth import *

from cachetools import TTLCache

import psutil
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from joblib import dump, load

# For handling hex bytes
from hexbytes import HexBytes

# Define a global variable for loading bar state
_loading_bar_active = False

class LoadingBarError(Exception):
    """Custom exception for loading bar errors."""
    pass

async def loading_bar(
    message: str,
    total_time: int,
    success_message: Optional[str] = None,
) -> None:
    """
    Display an asynchronous loading bar in the console.

    Args:
        message (str): The message to display alongside the loading bar.
        total_time (int): Total time in seconds for the loading bar to complete.
        success_message (Optional[str]): Message to display upon completion.

    Raises:
        LoadingBarError: If there's an error during loading bar execution.
        ValueError: If invalid parameters are provided.
    """
    global _loading_bar_active
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    RESET = "\033[0m"
    
    # Input validation
    if not isinstance(message, str) or not message:
        raise ValueError("Message must be a non-empty string")
    if not isinstance(total_time, int) or total_time <= 0:
        raise ValueError("Total time must be a positive integer")
    if success_message is not None and not isinstance(success_message, str):
        raise ValueError("Success message must be a string if provided")

    _loading_bar_active = True
    bar_length = 20
    try:
        for i in range(101):
            try:
                await asyncio.sleep(total_time / 100)
                percent = i / 100
                filled_length = int(percent * bar_length)
                bar = '█' * filled_length + '-' * (bar_length - filled_length)
                sys.stdout.write(f"\r{GREEN}{message} [{bar}] {i}%{RESET}")
                sys.stdout.flush()
            except asyncio.CancelledError:
                sys.stdout.write(f"\r{RED}Loading bar cancelled{RESET}\n")
                sys.stdout.flush()
                raise
            except Exception as e:
                raise LoadingBarError(f"Error during loading bar progress: {str(e)}")

        sys.stdout.write("\n")
        sys.stdout.flush()

        if success_message:
            print(f"{GREEN}{success_message}{RESET}")

    except LoadingBarError as e:
        print(f"{RED}Loading bar error: {str(e)}{RESET}")
        raise
    except Exception as e:
        print(f"{RED}Unexpected error in loading_bar: {str(e)}{RESET}")
        raise LoadingBarError(f"Unexpected error: {str(e)}")
    finally:
        _loading_bar_active = False

@dataclass
class StrategyPerformanceMetrics:
    """Data class for tracking strategy performance metrics."""
    avg_execution_time: float = field(default=0.0)
    success_rate: float = field(default=0.0)
    total_executions: int = field(default=0)
    successes: int = field(default=0)
    failures: int = field(default=0)
    profit: Decimal = field(default=Decimal("0.0"))

    def __post_init__(self):
        """Validate metrics after initialization."""
        if self.avg_execution_time < 0:
            raise ValueError("Average execution time cannot be negative")
        if not 0 <= self.success_rate <= 1:
            raise ValueError("Success rate must be between 0 and 1")
        if self.total_executions < 0:
            raise ValueError("Total executions cannot be negative")
        if self.successes < 0:
            raise ValueError("Successes cannot be negative")
        if self.failures < 0:
            raise ValueError("Failures cannot be negative")
        if self.successes + self.failures > self.total_executions:
            raise ValueError("Sum of successes and failures cannot exceed total executions")

@dataclass
class StrategyConfiguration:
    """Data class for strategy configuration parameters."""
    decay_factor: float = field(default=0.95)
    min_profit_threshold: Decimal = field(default=Decimal("0.01"))
    learning_rate: float = field(default=0.01)
    exploration_rate: float = field(default=0.1)

    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        if not 0 < self.decay_factor <= 1:
            raise ValueError("Decay factor must be between 0 and 1")
        if self.min_profit_threshold <= 0:
            raise ValueError("Minimum profit threshold must be positive")
        if not 0 < self.learning_rate <= 1:
            raise ValueError("Learning rate must be between 0 and 1")
        if not 0 <= self.exploration_rate <= 1:
            raise ValueError("Exploration rate must be between 0 and 1")

@dataclass
class StrategyExecutionError(Exception):
    """Custom exception for strategy execution errors."""
    message: str
    error_code: int = field(default=500)
    timestamp: float = field(default_factory=time.time)
    details: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """Format the error message with additional details."""
        error_msg = f"Strategy Execution Error ({self.error_code}): {self.message}"
        if self.details:
            error_msg += f"\nDetails: {json.dumps(self.details, indent=2)}"
        return error_msg

class Configuration:
    """
    Configuration class to load and store environment variables and settings.
    """
    STREAMLIT_ENABLED = True

    def __init__(self):
        self.INFURA_PROJECT_ID = ""
        self.COINGECKO_API_KEY = ""
        self.COINMARKETCAP_API_KEY = ""
        self.CRYPTOCOMPARE_API_KEY = ""

        self.HTTP_ENDPOINT = ""
        self.IPC_ENDPOINT = ""
        self.WEBSOCKET_ENDPOINT = ""
        self.WALLET_KEY = ""
        self.WALLET_ADDRESS = ""

        self.AAVE_LENDING_POOL_ADDRESS = ""
        self.TOKEN_ADDRESSES = {}
        self.TOKEN_SYMBOLS = {}

        self.ERC20_ABI = ""
        self.ERC20_SIGNATURES = {}
        self.SUSHISWAP_ROUTER_ABI = ""
        self.SUSHISWAP_ROUTER_ADDRESS = ""
        self.UNISWAP_ROUTER_ABI = ""
        self.UNISWAP_ROUTER_ADDRESS = ""
        self.AAVE_FLASHLOAN_ABI = ""
        self.AAVE_LENDING_POOL_ABI = ""
        self.AAVE_FLASHLOAN_ADDRESS = ""
        self.PANCAKESWAP_ROUTER_ABI = ""
        self.PANCAKESWAP_ROUTER_ADDRESS = ""
        self.BALANCER_ROUTER_ABI = ""
        self.BALANCER_ROUTER_ADDRESS = ""

        self.ML_MODEL_PATH = "models/price_model.joblib"
        self.ML_TRAINING_DATA_PATH = "data/training_data.csv"

    async def load(self) -> None:
        """
        Load the configuration by calling internal methods.
        """
        try:
            await self._load_configuration()
        except Exception as e:
            print(f"[Configuration] Initialization failed: {e}")
            raise

    async def _load_configuration(self) -> None:
        """
        Internal method to load configuration.
        """
        try:
            await loading_bar("Loading Environment Variables", 2, "Environment Variables Loaded")
            self._load_api_keys()
            self._load_providers_and_account()
            self._load_ML_models()
            await self._load_json_elements()
        except EnvironmentError as e:
            print(f"[Configuration] Environment error: {e}")
            raise
        except FileNotFoundError as e:
            print(f"[Configuration] File not found: {e}")
            raise
        except json.JSONDecodeError as e:
            print(f"[Configuration] JSON decode error in file: {e}")
            raise
        except Exception as e:
            print(f"[Configuration] Unexpected error: {e}")
            raise RuntimeError(f"Failed to load configuration: {e}") from e

    def _load_ML_models(self) -> None:
        """
        Load ML model paths.
        """
        try:
            self.ML_MODEL_PATH = "models/price_model.joblib"
            self.ML_TRAINING_DATA_PATH = "data/training_data.csv"
        except Exception as e:
            print(f"[Configuration] Error loading ML models: {e}")
            raise

    def _load_api_keys(self) -> None:
        """
        Load API keys from environment variables.
        """
        try:
            self.ETHERSCAN_API_KEY = self._get_env_variable("ETHERSCAN_API_KEY")
            self.INFURA_PROJECT_ID = self._get_env_variable("INFURA_PROJECT_ID")
            self.COINGECKO_API_KEY = self._get_env_variable("COINGECKO_API_KEY")
            self.COINMARKETCAP_API_KEY = self._get_env_variable("COINMARKETCAP_API_KEY")
            self.CRYPTOCOMPARE_API_KEY = self._get_env_variable("CRYPTOCOMPARE_API_KEY")
        except EnvironmentError as e:
            print(f"[Configuration] API key loading error: {e}")
            raise

    def _load_providers_and_account(self) -> None:
        """
        Load providers and account information from environment variables.
        """
        try:
            self.HTTP_ENDPOINT = self._get_env_variable("HTTP_ENDPOINT")
            self.IPC_ENDPOINT = self._get_env_variable("IPC_ENDPOINT")
            self.WEBSOCKET_ENDPOINT = self._get_env_variable("WEBSOCKET_ENDPOINT")
            self.WALLET_KEY = self._get_env_variable("WALLET_KEY")
            self.WALLET_ADDRESS = self._get_env_variable("WALLET_ADDRESS")
        except EnvironmentError as e:
            print(f"[Configuration] Provider/account loading error: {e}")
            raise

    async def _load_json_elements(self) -> None:
        """
        Load JSON elements from files specified in environment variables.
        """
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
        except EnvironmentError as e:
            print(f"[Configuration] JSON elements loading error: {e}")
            raise
        except FileNotFoundError as e:
            print(f"[Configuration] JSON file not found: {e}")
            raise
        except json.JSONDecodeError as e:
            print(f"[Configuration] JSON decode error: {e}")
            raise
        except Exception as e:
            print(f"[Configuration] Unexpected error loading JSON elements: {e}")
            raise

    def _get_env_variable(self, var_name: str, default: Optional[str] = None) -> str:
        """
        Retrieve environment variable or raise error if missing.

        Args:
            var_name (str): The name of the environment variable.
            default (Optional[str]): Default value if variable is not set.

        Returns:
            str: The value of the environment variable.

        Raises:
            EnvironmentError: If the variable is missing and no default is provided.
        """
        value = os.getenv(var_name, default)
        if value is None:
            error_msg = f"Missing environment variable: {var_name}"
            print(f"[Configuration] {error_msg}")
            raise EnvironmentError(error_msg)
        return value

    async def _load_json_file(self, file_path: str, description: str) -> Any:
        """
        Asynchronously load a JSON file.

        Args:
            file_path (str): Path to the JSON file.
            description (str): Description of the JSON content.

        Returns:
            Any: The loaded JSON data.

        Raises:
            FileNotFoundError: If the file does not exist.
            json.JSONDecodeError: If the JSON is invalid.
            Exception: For other errors.
        """
        try:
            async with aiofiles.open(file_path, "r") as f:
                content = await f.read()
                data = json.loads(content)
                await loading_bar(f"Loading {len(data)} {description} from {file_path}", 3, f"Loaded {description}")
                return data
        except FileNotFoundError as e:
            error_msg = f"File not found: {file_path}"
            print(f"[Configuration] {error_msg}")
            raise FileNotFoundError(error_msg) from e
        except json.JSONDecodeError as e:
            error_msg = f"JSON decode error in file {file_path}: {e}"
            print(f"[Configuration] {error_msg}")
            raise json.JSONDecodeError(error_msg, e.doc, e.pos) from e
        except Exception as e:
            error_msg = f"Error loading JSON file {file_path}: {e}"
            print(f"[Configuration] {error_msg}")
            raise Exception(error_msg) from e

    async def _construct_abi_path(self, base_path: str, abi_filename: str) -> str:
        """
        Construct and verify the path to an ABI file.

        Args:
            base_path (str): Base directory path.
            abi_filename (str): ABI filename.

        Returns:
            str: The full path to the ABI file.

        Raises:
            FileNotFoundError: If the ABI file does not exist.
        """
        abi_path = os.path.join(base_path, abi_filename)
        if not os.path.exists(abi_path):
            error_msg = f"ABI file '{abi_filename}' not found in path '{base_path}'"
            print(f"[Configuration] {error_msg}")
            raise FileNotFoundError(error_msg)
        return abi_path

    async def get_token_addresses(self) -> List[str]:
        """
        Get the list of token addresses.

        Returns:
            List[str]: List of token addresses.
        """
        try:
            return list(self.TOKEN_ADDRESSES.values())
        except Exception as e:
            print(f"[Configuration] Error retrieving token addresses: {e}")
            return []

    async def get_token_symbols(self) -> Dict[str, str]:
        """
        Get the mapping of token symbols.

        Returns:
            Dict[str, str]: Mapping from token address to symbol.
        """
        try:
            return self.TOKEN_SYMBOLS
        except Exception as e:
            print(f"[Configuration] Error retrieving token symbols: {e}")
            return {}

    def get_abi_path(self, abi_name: str) -> str:
        """
        Get the ABI path for a given ABI name.

        Args:
            abi_name (str): The name of the ABI.

        Returns:
            str: The path to the ABI file or empty string if not found.
        """
        abi_paths = {
            "erc20_abi": self.ERC20_ABI,
            "sushiswap_router_abi": self.SUSHISWAP_ROUTER_ABI,
            "uniswap_router_abi": self.UNISWAP_ROUTER_ABI,
            "aave_flashloan_abi": self.AAVE_FLASHLOAN_ABI,
            "aave_lending_pool_abi": self.AAVE_LENDING_POOL_ABI,
            "pancakeswap_router_abi": self.PANCAKESWAP_ROUTER_ABI,
            "balancer_router_abi": self.BALANCER_ROUTER_ABI,
        }
        path = abi_paths.get(abi_name.lower(), "")
        if not path:
            print(f"[Configuration] ABI path for '{abi_name}' not found.")
        return path

class NonceCore:
    """
    Core class to manage transaction nonces with caching and synchronization.
    """
    MAX_RETRIES = 5
    RETRY_DELAY = 1.0
    CACHE_TTL = 300

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
        """
        Initialize the nonce core by fetching the current nonce.
        """
        try:
            async with self.lock:
                if not self._initialized:
                    await self._init_nonce()
                    self._initialized = True
        except (ConnectionError, TimeoutError) as e:
            print(f"Connection error during NonceCore initialization: {e}")
            raise RuntimeError("NonceCore initialization failed due to connection issues.") from e
        except Exception as e:
            print(f"Unexpected error during NonceCore initialization: {e}")
            raise RuntimeError("NonceCore initialization failed.") from e

    async def _init_nonce(self) -> None:
        """
        Initialize nonce by fetching from the chain and pending transactions.
        """
        try:
            current_nonce = await self._fetch_current_nonce_with_retries()
            pending_nonce = await self._get_pending_nonce()
            self.nonce_cache[self.address] = max(current_nonce, pending_nonce)
            self.last_sync = time.monotonic()
        except Exception as e:
            print(f"Error initializing nonce: {e}")
            raise

    async def get_nonce(self, force_refresh: bool = False) -> int:
        """
        Get the next nonce, optionally forcing a refresh from the chain.

        Args:
            force_refresh (bool): Whether to force refresh the nonce.

        Returns:
            int: The next nonce.
        """
        if not self._initialized:
            await self.initialize()
        async with self.lock:
            try:
                if force_refresh or self._should_refresh_cache():
                    await self.refresh_nonce()
                current_nonce = self.nonce_cache.get(self.address, 0)
                next_nonce = current_nonce
                self.nonce_cache[self.address] = current_nonce + 1
                return next_nonce
            except KeyError as e:
                print(f"NonceCore KeyError: {e}")
                await self._handle_nonce_error()
                raise RuntimeError("Failed to retrieve nonce due to KeyError.") from e
            except Exception as e:
                print(f"NonceCore Exception in get_nonce: {e}")
                await self._handle_nonce_error()
                raise RuntimeError("Failed to retrieve nonce.") from e

    async def refresh_nonce(self) -> None:
        """
        Refresh the nonce by fetching from the chain and pending transactions.
        """
        async with self.lock:
            try:
                chain_nonce = await self._fetch_current_nonce_with_retries()
                cached_nonce = self.nonce_cache.get(self.address, 0)
                pending_nonce = await self._get_pending_nonce()
                new_nonce = max(chain_nonce, cached_nonce, pending_nonce)
                self.nonce_cache[self.address] = new_nonce
                self.last_sync = time.monotonic()
            except (ConnectionError, TimeoutError) as e:
                print(f"Connection error while refreshing nonce: {e}")
                raise RuntimeError("Failed to refresh nonce due to connection issues.") from e
            except Exception as e:
                print(f"Error refreshing nonce: {e}")
                raise

    async def _fetch_current_nonce_with_retries(self) -> int:
        """
        Fetch the current nonce from the blockchain with retries.

        Returns:
            int: The current nonce.

        Raises:
            Exception: If unable to fetch after retries.
        """
        backoff = self.RETRY_DELAY
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                nonce = await self.web3.eth.get_transaction_count(
                    self.address, block_identifier="pending"
                )
                return nonce
            except (ConnectionError, TimeoutError) as e:
                print(f"Network error fetching nonce (attempt {attempt}): {e}")
                if attempt == self.MAX_RETRIES:
                    raise RuntimeError("Max retries reached while fetching nonce.") from e
                await asyncio.sleep(backoff)
                backoff *= 2
            except Exception as e:
                print(f"Unexpected error fetching nonce (attempt {attempt}): {e}")
                if attempt == self.MAX_RETRIES:
                    raise
                await asyncio.sleep(backoff)
                backoff *= 2
        raise Exception("Failed to fetch current nonce after retries")

    async def _get_pending_nonce(self) -> int:
        """
        Get the highest pending nonce.

        Returns:
            int: The next pending nonce.
        """
        try:
            if not self.pending_transactions:
                return 0
            pending_nonces = list(self.pending_transactions)
            return max(pending_nonces) + 1 if pending_nonces else 0
        except Exception as e:
            print(f"Error getting pending nonce: {e}")
            return 0

    async def track_transaction(self, tx_hash: str, nonce: int) -> None:
        """
        Track a transaction's nonce.

        Args:
            tx_hash (str): The transaction hash.
            nonce (int): The nonce of the transaction.
        """
        self.pending_transactions.add(nonce)
        try:
            await self.web3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            self.pending_transactions.discard(nonce)
        except asyncio.TimeoutError:
            print(f"Timeout waiting for transaction receipt: {tx_hash}")
            self.pending_transactions.discard(nonce)
            raise
        except Exception as e:
            print(f"Error tracking transaction {tx_hash}: {e}")
            self.pending_transactions.discard(nonce)
            raise

    async def _handle_nonce_error(self) -> None:
        """
        Handle errors related to nonce by syncing with the chain.
        """
        try:
            await self.sync_nonce_with_chain()
        except Exception as e:
            print(f"Error handling nonce error: {e}")
            raise RuntimeError("Failed to handle nonce error.") from e

    async def sync_nonce_with_chain(self) -> None:
        """
        Synchronize the nonce with the chain by fetching from the blockchain.
        """
        async with self.lock:
            try:
                new_nonce = await self._fetch_current_nonce_with_retries()
                self.nonce_cache[self.address] = new_nonce
                self.last_sync = time.monotonic()
                self.pending_transactions.clear()
            except (ConnectionError, TimeoutError) as e:
                print(f"Connection error syncing nonce with chain: {e}")
                raise RuntimeError("Failed to sync nonce due to connection issues.") from e
            except Exception as e:
                print(f"Error syncing nonce with chain: {e}")
                raise

    def _should_refresh_cache(self) -> bool:
        """
        Determine if the cache should be refreshed based on TTL.

        Returns:
            bool: Whether to refresh the cache.
        """
        return time.monotonic() - self.last_sync > (self.CACHE_TTL / 2)

    async def reset(self) -> None:
        """
        Reset the nonce cache and pending transactions.
        """
        async with self.lock:
            try:
                self.nonce_cache.clear()
                self.pending_transactions.clear()
                self.last_sync = time.monotonic()
                self._initialized = False
                await self.initialize()
            except Exception as e:
                print(f"Error resetting NonceCore: {e}")
                raise RuntimeError("Failed to reset NonceCore.") from e

    async def stop(self) -> None:
        """
        Stop the NonceCore by resetting it.
        """
        try:
            await self.reset()
        except Exception as e:
            print(f"Error stopping NonceCore: {e}")
            raise RuntimeError("Failed to stop NonceCore.") from e

class APIConfig:
    """
    API Configuration and management class.
    """
    def __init__(self, configuration: Optional['Configuration'] = None):
        self.apiconfig = {}
        self.configuration = configuration
        self.session = None
        self.price_cache = TTLCache(maxsize=1000, ttl=300)
        self.token_symbol_cache = TTLCache(maxsize=1000, ttl=86400)
        self.api_lock = asyncio.Lock()
        self.rate_limiters = {}

    async def __aenter__(self):
        """
        Initialize the API session and configurations.
        """
        self.session = aiohttp.ClientSession()
        self.apiconfig = {
            "binance": {
                "base_url": "https://api.binance.com/api/v3",
                "success_rate": 1.0,
                "weight": 1.0,
                "rate_limit": 1200,
            },
            "coingecko": {
                "base_url": "https://api.coingecko.com/api/v3",
                "api_key": self.configuration.COINGECKO_API_KEY,
                "success_rate": 1.0,
                "weight": 0.8,
                "rate_limit": 50,
            },
            "coinmarketcap": {
                "base_url": "https://pro-api.coinmarketcap.com/v1",
                "api_key": self.configuration.COINMARKETCAP_API_KEY,
                "success_rate": 1.0,
                "weight": 0.7,
                "rate_limit": 333,
            },
            "cryptocompare": {
                "base_url": "https://min-api.cryptocompare.com/data",
                "api_key": self.configuration.CRYPTOCOMPARE_API_KEY,
                "success_rate": 1.0,
                "weight": 0.6,
                "rate_limit": 80,
            },
            "primary": {  # Added 'primary' for default usage
                "base_url": "https://api.coingecko.com/api/v3",
                "api_key": self.configuration.COINGECKO_API_KEY,
                "success_rate": 1.0,
                "weight": 1.0,
                "rate_limit": 50,
            },
        }
        self.rate_limiters = {
            provider: asyncio.Semaphore(config.get("rate_limit", 10))
            for provider, config in self.apiconfig.items()
        }
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Cleanup the API session.
        """
        if self.session:
            await self.session.close()

    async def get_token_symbol(self, web3: 'AsyncWeb3', token_address: str) -> Optional[str]:
        """
        Get the symbol of a token, either from cache or by querying the blockchain.

        Args:
            web3 (AsyncWeb3): The web3 instance.
            token_address (str): The token's contract address.

        Returns:
            Optional[str]: The token symbol or None if not found.
        """
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
            print(f"Error getting token symbol for {token_address}: {e}")
            return None

    async def get_real_time_price(self, token: str, service: str = "primary", vs_currency: str = "eth") -> Optional[Decimal]:
        """
        Get the real-time price of a token using a specified service.

        Args:
            token (str): The token symbol or ID.
            service (str): The service to use for fetching the price.
            vs_currency (str): The currency to compare against.

        Returns:
            Optional[Decimal]: The price or None if not available.
        """
        cache_key = f"price_{token}_{vs_currency}"
        if cache_key in self.price_cache:
            return self.price_cache[cache_key]
        prices = []
        weights = []
        async with self.api_lock:
            for source, config in self.apiconfig.items():
                if source != service and service != "primary":
                    continue  # Only use the specified service
                try:
                    price = await self._fetch_price(source, token, vs_currency)
                    if price:
                        prices.append(price)
                        weights.append(config["weight"] * config["success_rate"])
                except Exception as e:
                    print(f"Error fetching price from {source}: {e}")
                    config["success_rate"] *= 0.9  # Reduce success rate on failure
        if not prices:
            return None
        weighted_price = sum(p * w for p, w in zip(prices, weights)) / sum(weights)
        self.price_cache[cache_key] = Decimal(str(weighted_price))
        return self.price_cache[cache_key]

    async def _fetch_price(self, source: str, token: str, vs_currency: str) -> Optional[Decimal]:
        """
        Fetch price from a specific source.

        Args:
            source (str): The API source.
            token (str): Token identifier.
            vs_currency (str): Currency to compare against.

        Returns:
            Optional[Decimal]: The fetched price or None.
        """
        config = self.apiconfig.get(source)
        if not config:
            return None
        try:
            if source == "coingecko" or (source == "primary" and config["base_url"] == "https://api.coingecko.com/api/v3"):
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
                return None
        except Exception as e:
            print(f"Error fetching price from {source}: {e}")
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
        """
        Make an HTTP GET request with rate limiting and retries.

        Args:
            provider_name (str): The API provider.
            url (str): The URL to fetch.
            params (Optional[Dict[str, Any]]): Query parameters.
            headers (Optional[Dict[str, str]]): Request headers.
            max_attempts (int): Maximum number of attempts.
            backoff_factor (float): Backoff multiplier.

        Returns:
            Any: The JSON response.

        Raises:
            Exception: If all attempts fail.
        """
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
                            print(f"Rate limited by {provider_name}. Waiting for {wait_time} seconds.")
                            await asyncio.sleep(wait_time)
                            continue
                        response.raise_for_status()
                        return await response.json()
                except aiohttp.ClientResponseError as e:
                    print(f"Client response error from {provider_name}: {e}")
                    if attempt == max_attempts - 1:
                        raise
                    wait_time = backoff_factor ** attempt
                    await asyncio.sleep(wait_time)
                except aiohttp.ClientConnectionError as e:
                    print(f"Client connection error from {provider_name}: {e}")
                    if attempt == max_attempts - 1:
                        raise
                    wait_time = backoff_factor ** attempt
                    await asyncio.sleep(wait_time)
                except Exception as e:
                    print(f"Unexpected error from {provider_name}: {e}")
                    if attempt == max_attempts - 1:
                        raise
                    wait_time = backoff_factor ** attempt
                    await asyncio.sleep(wait_time)

    async def fetch_historical_prices_for_model(self, token: str, days: int = 30, service: str = "primary") -> List[float]:
        """
        Fetch historical prices for ML model training.

        Args:
            token (str): Token symbol or ID.
            days (int): Number of days to fetch.
            service (str): The service to use.

        Returns:
            List[float]: List of historical prices.
        """
        return await self.fetch_historical_prices(token, days, service)

    async def get_token_volume(self, token: str, service: str = "primary") -> float:
        """
        Get the trading volume of a token.

        Args:
            token (str): The token symbol or ID.
            service (str): The service to use.

        Returns:
            float: The trading volume or 0.0 if not available.
        """
        cache_key = f"token_volume_{token}"
        if cache_key in self.price_cache:
            return self.price_cache[cache_key]
        try:
            if service == "coingecko":
                url = f"{self.apiconfig['coingecko']['base_url']}/coins/markets"
                params = {"vs_currency": "usd", "ids": token}
                response = await self.make_request(service, url, params=params)
                volume = response[0]["total_volume"] if response else 0.0
                self.price_cache[cache_key] = volume
                return volume
            else:
                # Add other services if needed
                return 0.0
        except Exception as e:
            print(f"Error fetching token volume from {service}: {e}")
            return 0.0

    async def fetch_historical_prices(self, token: str, days: int = 30, service: str = "primary") -> List[float]:
        """
        Fetch historical prices for a token.

        Args:
            token (str): The token symbol or ID.
            days (int): Number of days to fetch.
            service (str): The service to use.

        Returns:
            List[float]: List of historical prices.
        """
        cache_key = f"historical_prices_{token}_{days}"
        if cache_key in self.price_cache:
            return self.price_cache[cache_key]
        try:
            if service == "coingecko" or service == "primary":
                url = f"{self.apiconfig[service]['base_url']}/coins/{token}/market_chart"
                params = {"vs_currency": "usd", "days": days}
                response = await self.make_request(service, url, params=params)
                prices = [price[1] for price in response["prices"]] if "prices" in response else []
                self.price_cache[cache_key] = prices
                return prices
            else:
                # Add other services if needed
                return []
        except Exception as e:
            print(f"Error fetching historical prices from {service}: {e}")
            return []

    async def _fetch_from_services(self, fetch_func: Callable[[str], Any], description: str) -> Optional[Any]:
        """
        Fetch data from multiple services until success.

        Args:
            fetch_func (Callable[[str], Any]): The function to fetch data.
            description (str): Description for logging.

        Returns:
            Optional[Any]: The fetched data or None.
        """
        for service in self.apiconfig.keys():
            try:
                result = await fetch_func(service)
                if result:
                    return result
            except Exception as e:
                print(f"Error fetching {description} from {service}: {e}")
                continue
        return None

    async def _load_abi(self, abi_path: str) -> List[Dict[str, Any]]:
        """
        Load ABI from a file.

        Args:
            abi_path (str): Path to the ABI file.

        Returns:
            List[Dict[str, Any]]: The ABI.

        Raises:
            Exception: If unable to load.
        """
        try:
            async with aiofiles.open(abi_path, 'r') as file:
                content = await file.read()
                abi = json.loads(content)
            return abi
        except Exception as e:
            print(f"Error loading ABI from {abi_path}: {e}")
            raise

    async def close(self):
        """
        Close the API session.
        """
        if self.session:
            await self.session.close()

class SafetyNet:
    """
    Safety controller and risk management engine.
    """
    CACHE_TTL = 300
    GAS_PRICE_CACHE_TTL = 15
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
        apiconfig: Optional[APIConfig] = None,
    ):
        self.web3 = web3
        self.address = address
        self.configuration = configuration
        self.account = account
        self.apiconfig = apiconfig
        self.price_cache = TTLCache(maxsize=1000, ttl=self.CACHE_TTL)
        self.gas_price_cache = TTLCache(maxsize=1, ttl=self.GAS_PRICE_CACHE_TTL)
        self.price_lock = asyncio.Lock()

    async def get_balance(self, account: Account) -> Decimal:
        """
        Get the ETH balance of an account.

        Args:
            account (Account): The account.

        Returns:
            Decimal: The balance in ETH.
        """
        cache_key = f"balance_{account.address}"
        if cache_key in self.price_cache:
            return self.price_cache[cache_key]
        for attempt in range(3):
            try:
                balance_wei = await self.web3.eth.get_balance(account.address)
                balance_eth = Decimal(self.web3.from_wei(balance_wei, "ether"))
                self.price_cache[cache_key] = balance_eth
                return balance_eth
            except Exception as e:
                print(f"Error getting balance for {account.address} (attempt {attempt + 1}): {e}")
                if attempt == 2:
                    return Decimal(0)
                await asyncio.sleep(1 * (attempt + 1))
        return Decimal(0)

    async def ensure_profit(
        self,
        transaction_data: Dict[str, Any],
        minimum_profit_eth: Optional[float] = None,
    ) -> bool:
        """
        Ensure that executing a transaction would result in a minimum profit.

        Args:
            transaction_data (Dict[str, Any]): The transaction data.
            minimum_profit_eth (Optional[float]): The minimum profit threshold.

        Returns:
            bool: True if profit is ensured, False otherwise.
        """
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
            real_time_price = await self.apiconfig.get_real_time_price(output_token)
            if not real_time_price:
                print("Real-time price not available.")
                return False
            profit = await self._calculate_profit(
                transaction_data, real_time_price, slippage, gas_cost_eth
            )
            await self._log_profit_calculation(
                transaction_data, real_time_price, gas_cost_eth, profit, minimum_profit_eth
            )
            return profit > Decimal(minimum_profit_eth)
        except KeyError as e:
            print(f"KeyError in ensure_profit: {e}")
            return False
        except Exception as e:
            print(f"Exception in ensure_profit: {e}")
            return False

    def _validate_gas_parameters(self, gas_price_gwei: Decimal, gas_used: int) -> bool:
        """
        Validate gas parameters to ensure they are within acceptable limits.

        Args:
            gas_price_gwei (Decimal): The gas price in Gwei.
            gas_used (int): The estimated gas used.

        Returns:
            bool: True if valid, False otherwise.
        """
        if gas_used == 0:
            return False
        if gas_price_gwei > self.GAS_CONFIG["max_gas_price_gwei"]:
            print(f"Gas price {gas_price_gwei} Gwei exceeds maximum allowed.")
            return False
        return True

    def _calculate_gas_cost(self, gas_price_gwei: Decimal, gas_used: int) -> Decimal:
        """
        Calculate the gas cost in ETH.

        Args:
            gas_price_gwei (Decimal): The gas price in Gwei.
            gas_used (int): The estimated gas used.

        Returns:
            Decimal: The gas cost in ETH.
        """
        return gas_price_gwei * Decimal(gas_used) * Decimal("1e-9")

    async def _calculate_profit(
        self,
        transaction_data: Dict[str, Any],
        real_time_price: Decimal,
        slippage: float,
        gas_cost_eth: Decimal,
    ) -> Decimal:
        """
        Calculate the expected profit from a transaction.

        Args:
            transaction_data (Dict[str, Any]): The transaction data.
            real_time_price (Decimal): The real-time price of the output token.
            slippage (float): The slippage tolerance.
            gas_cost_eth (Decimal): The gas cost in ETH.

        Returns:
            Decimal: The expected profit.
        """
        try:
            expected_output = real_time_price * Decimal(transaction_data["amountOut"])
            input_amount = Decimal(transaction_data["amountIn"])
            slippage_adjusted_output = expected_output * (1 - Decimal(slippage))
            return slippage_adjusted_output - input_amount - gas_cost_eth
        except Exception as e:
            print(f"Error calculating profit: {e}")
            return Decimal(0)

    async def get_dynamic_gas_price(self) -> Decimal:
        """
        Get the dynamic gas price, utilizing caching.

        Returns:
            Decimal: The gas price in Gwei.
        """
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
            print(f"Error getting dynamic gas price: {e}")
            return Decimal(0)

    async def estimate_gas(self, transaction_data: Dict[str, Any]) -> int:
        """
        Estimate the gas required for a transaction.

        Args:
            transaction_data (Dict[str, Any]): The transaction data.

        Returns:
            int: The estimated gas.
        """
        try:
            gas_estimate = await self.web3.eth.estimate_gas(transaction_data)
            return gas_estimate
        except Exception as e:
            print(f"Error estimating gas: {e}")
            return self.GAS_CONFIG["base_gas_limit"]

    async def adjust_slippage_tolerance(self) -> float:
        """
        Adjust slippage tolerance based on network congestion.

        Returns:
            float: The adjusted slippage tolerance.
        """
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
            return slippage
        except Exception as e:
            print(f"Error adjusting slippage tolerance: {e}")
            return self.SLIPPAGE_CONFIG["default"]

    async def get_network_congestion(self) -> float:
        """
        Get the current network congestion level.

        Returns:
            float: The network congestion level (0 to 1).
        """
        try:
            latest_block = await self.web3.eth.get_block("latest")
            gas_used = latest_block["gasUsed"]
            gas_limit = latest_block["gasLimit"]
            congestion_level = gas_used / gas_limit
            return congestion_level
        except Exception as e:
            print(f"Error getting network congestion: {e}")
            return 0.5

    async def _log_profit_calculation(
        self,
        transaction_data: Dict[str, Any],
        real_time_price: Decimal,
        gas_cost_eth: Decimal,
        profit: Decimal,
        minimum_profit_eth: float,
    ) -> None:
        """
        Log the profit calculation details.

        Args:
            transaction_data (Dict[str, Any]): The transaction data.
            real_time_price (Decimal): The real-time price of the output token.
            gas_cost_eth (Decimal): The gas cost in ETH.
            profit (Decimal): The calculated profit.
            minimum_profit_eth (float): The minimum profit threshold.
        """
        # Implement logging as needed
        print(f"Profit Calculation:")
        print(f"  Real-time Price: {real_time_price} ETH")
        print(f"  Gas Cost: {gas_cost_eth} ETH")
        print(f"  Calculated Profit: {profit} ETH")
        print(f"  Minimum Profit Required: {minimum_profit_eth} ETH")

class MempoolMonitor:
    """
    Mempool monitoring class to watch for new transactions.
    """
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0
    BATCH_SIZE = 10
    MAX_PARALLEL_TASKS = 50

    def __init__(
        self,
        web3: AsyncWeb3,
        safetynet: 'SafetyNet',
        noncecore: 'NonceCore',
        apiconfig: 'APIConfig',
        monitored_tokens: Optional[List[str]] = None,
        erc20_abi: List[Dict[str, Any]] = None,
        configuration: Optional['Configuration'] = None,
    ):
        self.web3 = web3
        self.configuration = configuration
        self.safetynet = safetynet
        self.noncecore = noncecore
        self.apiconfig = apiconfig
        self.running = False
        self.monitored_tokens = set(monitored_tokens or [])
        self.profitable_transactions = asyncio.Queue()
        self.processed_transactions = set()
        self.erc20_abi = erc20_abi or []
        self.minimum_profit_threshold = Decimal("0.001")
        self.max_parallel_tasks = self.MAX_PARALLEL_TASKS
        self.retry_attempts = self.MAX_RETRIES
        self.backoff_factor = 1.5
        self.semaphore = asyncio.Semaphore(self.max_parallel_tasks)
        self.task_queue = asyncio.Queue()

    async def start_monitoring(self) -> None:
        """
        Start monitoring the mempool for new transactions.
        """
        if self.running:
            return
        try:
            self.running = True
            monitoring_task = asyncio.create_task(self._run_monitoring())
            processor_task = asyncio.create_task(self._process_task_queue())
            await asyncio.gather(monitoring_task, processor_task)
        except Exception as e:
            print(f"Error starting mempool monitoring: {e}")
            self.running = False
            raise

    async def stop_monitoring(self) -> None:
        """
        Stop monitoring the mempool.
        """
        if not self.running:
            return
        self.running = False
        try:
            while not self.task_queue.empty():
                await asyncio.sleep(0.1)
            # Gracefully cancel tasks
            tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            print("Mempool monitoring stopped.")
        except Exception as e:
            print(f"Error stopping mempool monitoring: {e}")
            raise

    async def _run_monitoring(self) -> None:
        """
        Internal method to run the monitoring loop.
        """
        retry_count = 0
        while self.running:
            try:
                pending_filter = await self._setup_pending_filter()
                if not pending_filter:
                    await asyncio.sleep(5)
                    continue
                while self.running:
                    tx_hashes = await pending_filter.get_new_entries()
                    if tx_hashes:
                        await self._handle_new_transactions(tx_hashes)
                    await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                retry_count += 1
                wait_time = min(self.backoff_factor ** retry_count, 30)
                await asyncio.sleep(wait_time)

    async def _setup_pending_filter(self) -> Optional[Any]:
        """
        Set up a filter for pending transactions.

        Returns:
            Optional[Any]: The pending transaction filter or None if failed.
        """
        try:
            pending_filter = await self.web3.eth.filter("pending")
            return pending_filter
        except Exception as e:
            print(f"Error setting up pending filter: {e}")
            return None

    async def _handle_new_transactions(self, tx_hashes: List[str]) -> None:
        """
        Handle a batch of new transaction hashes.

        Args:
            tx_hashes (List[str]): List of transaction hashes.
        """
        async def process_batch(batch):
            await asyncio.gather(
                *(self._queue_transaction(tx_hash) for tx_hash in batch)
            )
        try:
            for i in range(0, len(tx_hashes), self.BATCH_SIZE):
                batch = tx_hashes[i: i + self.BATCH_SIZE]
                await process_batch(batch)
        except Exception as e:
            print(f"Error handling new transactions: {e}")

    async def _queue_transaction(self, tx_hash: str) -> None:
        """
        Queue a transaction hash for processing.

        Args:
            tx_hash (str): The transaction hash.
        """
        tx_hash_hex = tx_hash.hex() if isinstance(tx_hash, bytes) else tx_hash
        if tx_hash_hex not in self.processed_transactions:
            self.processed_transactions.add(tx_hash_hex)
            await self.task_queue.put(tx_hash_hex)

    async def _process_task_queue(self) -> None:
        """
        Process the queued transaction hashes.
        """
        while self.running:
            try:
                tx_hash = await self.task_queue.get()
                async with self.semaphore:
                    await self.process_transaction(tx_hash)
                self.task_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error processing task queue: {e}")

    async def process_transaction(self, tx_hash: str) -> None:
        """
        Process a single transaction hash.

        Args:
            tx_hash (str): The transaction hash.
        """
        try:
            tx = await self._get_transaction_with_retry(tx_hash)
            if not tx:
                return
            analysis = await self.analyze_transaction(tx)
            if analysis.get("is_profitable"):
                await self._handle_profitable_transaction(analysis)
        except Exception as e:
            print(f"Error processing transaction {tx_hash}: {e}")

    async def _get_transaction_with_retry(self, tx_hash: str) -> Optional[Any]:
        """
        Get transaction details with retries.

        Args:
            tx_hash (str): The transaction hash.

        Returns:
            Optional[Any]: The transaction object or None.
        """
        for attempt in range(self.retry_attempts):
            try:
                tx = await self.web3.eth.get_transaction(tx_hash)
                return tx
            except TransactionNotFound:
                if attempt == self.retry_attempts - 1:
                    print(f"Transaction {tx_hash} not found after {self.retry_attempts} attempts.")
                    return None
                await asyncio.sleep(self.backoff_factor ** attempt)
            except Exception as e:
                print(f"Error fetching transaction {tx_hash} (attempt {attempt + 1}): {e}")
                if attempt == self.retry_attempts - 1:
                    return None
                await asyncio.sleep(self.backoff_factor ** attempt)
        return None

    async def _handle_profitable_transaction(self, analysis: Dict[str, Any]) -> None:
        """
        Handle a profitable transaction by adding it to the queue.

        Args:
            analysis (Dict[str, Any]): The analysis result of the transaction.
        """
        try:
            await self.profitable_transactions.put(analysis)
        except Exception as e:
            print(f"Error handling profitable transaction: {e}")

    async def analyze_transaction(self, tx: Any) -> Dict[str, Any]:
        """
        Analyze a transaction to determine if it's profitable.

        Args:
            tx (Any): The transaction object.

        Returns:
            Dict[str, Any]: Analysis result.
        """
        if not tx.hash or not tx.input:
            return {"is_profitable": False}
        try:
            if tx.value > 0:
                return await self._analyze_eth_transaction(tx)
            return await self._analyze_token_transaction(tx)
        except Exception as e:
            print(f"Error analyzing transaction: {e}")
            return {"is_profitable": False}

    async def _analyze_eth_transaction(self, tx: Any) -> Dict[str, Any]:
        """
        Analyze an ETH transaction for profitability.

        Args:
            tx (Any): The transaction object.

        Returns:
            Dict[str, Any]: Analysis result.
        """
        try:
            if await self._is_profitable_eth_transaction(tx):
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
            print(f"Error analyzing ETH transaction: {e}")
            return {"is_profitable": False}

    async def _analyze_token_transaction(self, tx: Any) -> Dict[str, Any]:
        """
        Analyze a token transaction for profitability.

        Args:
            tx (Any): The transaction object.

        Returns:
            Dict[str, Any]: Analysis result.
        """
        try:
            if not self.erc20_abi:
                return {"is_profitable": False}
            contract = self.web3.eth.contract(address=tx.to, abi=self.erc20_abi)
            function_abi, function_params = contract.decode_function_input(tx.input)
            function_name = function_abi.name
            if function_name in self.configuration.ERC20_SIGNATURES:
                estimated_profit = await self._estimate_profit(tx, function_params)
                if estimated_profit > self.minimum_profit_threshold:
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
                    return {"is_profitable": False}
            else:
                return {"is_profitable": False}
        except Exception as e:
            print(f"Error analyzing token transaction: {e}")
            return {"is_profitable": False}

    async def _is_profitable_eth_transaction(self, tx: Any) -> bool:
        """
        Determine if an ETH transaction is profitable.

        Args:
            tx (Any): The transaction object.

        Returns:
            bool: True if profitable, False otherwise.
        """
        try:
            potential_profit = await self.safetynet.ensure_profit(tx, None)
            return potential_profit
        except Exception as e:
            print(f"Error checking ETH transaction profitability: {e}")
            return False

    async def _estimate_profit(
        self,
        tx: Any,
        function_params: Dict[str, Any],
    ) -> Decimal:
        """
        Estimate the profit from a token transaction.

        Args:
            tx (Any): The transaction object.
            function_params (Dict[str, Any]): The function parameters.

        Returns:
            Decimal: The estimated profit.
        """
        try:
            gas_price_gwei = Decimal(self.web3.from_wei(tx.gasPrice, "gwei"))
            gas_used = tx.gas if tx.gas else await self.web3.eth.estimate_gas(tx)
            gas_cost_eth = gas_price_gwei * Decimal(gas_used) * Decimal("1e-9")
            input_amount_wei = Decimal(function_params.get("amountIn", 0))
            output_amount_min_wei = Decimal(function_params.get("amountOutMin", 0))
            path = function_params.get("path", [])
            if len(path) < 2:
                return Decimal(0)
            output_token_address = path[-1]
            output_token_symbol = await self.apiconfig.get_token_symbol(self.web3, output_token_address)
            if not output_token_symbol:
                return Decimal(0)
            market_price = await self.apiconfig.get_real_time_price(
                output_token_symbol.lower()
            )
            if market_price is None or market_price == 0:
                return Decimal(0)
            input_amount_eth = Decimal(self.web3.from_wei(input_amount_wei, "ether"))
            output_amount_eth = Decimal(self.web3.from_wei(output_amount_min_wei, "ether"))
            expected_output_value = output_amount_eth * market_price
            profit = expected_output_value - input_amount_eth - gas_cost_eth
            return profit if profit > 0 else Decimal(0)
        except Exception as e:
            print(f"Error estimating profit: {e}")
            return Decimal(0)

class TransactionCore:
    """
    Transaction orchestrator and core class.
    """
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0

    def __init__(
        self,
        web3: AsyncWeb3,
        account: Account,
        aave_flashloan_address: str,
        aave_flashloan_abi: List[Dict[str, Any]],
        aave_lending_pool_address: str,
        aave_lending_pool_abi: List[Dict[str, Any]],
        apiconfig: Optional['APIConfig'] = None,
        monitor: Optional['MempoolMonitor'] = None,
        noncecore: Optional['NonceCore'] = None,
        safetynet: Optional['SafetyNet'] = None,
        configuration: Optional['Configuration'] = None,
        gas_price_multiplier: float = 1.1,
        erc20_abi: Optional[List[Dict[str, Any]]] = [],
    ):
        self.web3 = web3
        self.account = account
        self.configuration = configuration
        self.monitor = monitor
        self.apiconfig = apiconfig
        self.noncecore = noncecore
        self.safetynet = safetynet
        self.gas_price_multiplier = gas_price_multiplier
        self.retry_attempts = self.MAX_RETRIES
        self.erc20_abi = erc20_abi or []
        self.current_profit = Decimal("0")
        self.aave_flashloan_address = aave_flashloan_address
        self.aave_flashloan_abi = aave_flashloan_abi
        self.aave_lending_pool_address = aave_lending_pool_address
        self.aave_lending_pool_abi = aave_lending_pool_abi

    async def initialize(self):
        """
        Initialize contracts and load ABIs.
        """
        try:
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

        except Exception as e:
            print(f"Error initializing TransactionCore: {e}")
            raise

    async def _initialize_contract(
        self,
        contract_address: str,
        contract_abi: Union[str, List[Dict[str, Any]]],
        contract_name: str,
    ) -> Any:
        """
        Initialize a contract instance.

        Args:
            contract_address (str): The contract's address.
            contract_abi (Union[str, List[Dict[str, Any]]]): The ABI or path to the ABI file.
            contract_name (str): The name of the contract.

        Returns:
            Any: The contract instance.

        Raises:
            ValueError: If initialization fails.
        """
        try:
            if isinstance(contract_abi, str):
                if contract_abi in self._abi_cache:
                    contract_abi_content = self._abi_cache[contract_abi]
                else:
                    async with aiofiles.open(contract_abi, 'r') as f:
                        contract_abi_content = await f.read()
                        self._abi_cache[contract_abi] = contract_abi_content
                    contract_abi = json.loads(contract_abi_content)
            contract_instance = self.web3.eth.contract(
                address=self.web3.to_checksum_address(contract_address),
                abi=contract_abi,
            )
            return contract_instance
        except FileNotFoundError:
            print(f"ABI file not found for {contract_name}.")
            raise
        except json.JSONDecodeError:
            print(f"Invalid JSON ABI for {contract_name}.")
            raise
        except Exception as e:
            print(f"Contract initialization failed for {contract_name}: {e}")
            raise ValueError(
                f"Contract initialization failed for {contract_name}"
            ) from e

    _abi_cache = {}

    async def _load_erc20_abi(self) -> List[Dict[str, Any]]:
        """
        Load the ERC20 ABI.

        Returns:
            List[Dict[str, Any]]: The ERC20 ABI.

        Raises:
            Exception: If loading fails.
        """
        try:
            erc20_abi = await self.apiconfig._load_abi(self.configuration.ERC20_ABI)
            return erc20_abi
        except FileNotFoundError:
            print("ERC20 ABI file not found.")
            raise
        except json.JSONDecodeError:
            print("Invalid JSON in ERC20 ABI.")
            raise
        except Exception as e:
            print(f"ERC20 ABI loading failed: {e}")
            raise

    async def build_transaction(
        self, function_call: Any, additional_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Build a transaction dictionary.

        Args:
            function_call (Any): The contract function call.
            additional_params (Optional[Dict[str, Any]]): Additional transaction parameters.

        Returns:
            Dict[str, Any]: The transaction dictionary.

        Raises:
            Exception: If building the transaction fails.
        """
        additional_params = additional_params or {}
        try:
            tx_details = function_call.buildTransaction({
                'chainId': await self.web3.eth.chain_id,
                'nonce': await self.noncecore.get_nonce(),
                'from': self.account.address,
            })
            tx_details.update(additional_params)
            tx_details["gas"] = await self.estimate_gas_smart(tx_details)
            tx_details.update(await self.get_dynamic_gas_price())
            return tx_details
        except KeyError as e:
            print(f"KeyError in build_transaction: {e}")
            raise
        except Exception as e:
            print(f"Error building transaction: {e}")
            raise

    async def get_dynamic_gas_price(self) -> Dict[str, int]:
        """
        Get the dynamic gas price with a multiplier.

        Returns:
            Dict[str, int]: Dictionary containing the gas price.
        """
        try:
            gas_price_gwei = await self.safetynet.get_dynamic_gas_price()
        except Exception as e:
            print(f"Error getting dynamic gas price: {e}")
            gas_price_gwei = Decimal(100.0)
        gas_price = int(
            self.web3.to_wei(gas_price_gwei * self.gas_price_multiplier, "gwei")
        )
        return {"gasPrice": gas_price}

    async def estimate_gas_smart(self, tx: Dict[str, Any]) -> int:
        """
        Estimate gas with smart fallback.

        Args:
            tx (Dict[str, Any]): The transaction dictionary.

        Returns:
            int: The estimated gas.
        """
        try:
            gas_estimate = await self.web3.eth.estimate_gas(tx)
            return gas_estimate
        except ContractLogicError as e:
            print(f"Contract logic error estimating gas: {e}")
            return 100_000
        except TransactionNotFound as e:
            print(f"Transaction not found estimating gas: {e}")
            return 100_000
        except Exception as e:
            print(f"General error estimating gas: {e}")
            return 100_000

    async def execute_transaction(self, tx: Dict[str, Any]) -> Optional[str]:
        """
        Execute a transaction with retries.

        Args:
            tx (Dict[str, Any]): The transaction dictionary.

        Returns:
            Optional[str]: The transaction hash in hex or None if failed.
        """
        for attempt in range(1, self.retry_attempts + 1):
            try:
                signed_tx = await self.sign_transaction(tx)
                tx_hash = await self.web3.eth.send_raw_transaction(signed_tx)
                tx_hash_hex = tx_hash.hex() if isinstance(tx_hash, HexBytes) else tx_hash
                await self.noncecore.refresh_nonce()
                return tx_hash_hex
            except TransactionNotFound as e:
                print(f"Transaction not found error: {e}")
                if attempt < self.MAX_RETRIES:
                    sleep_time = self.RETRY_DELAY * attempt
                    await asyncio.sleep(sleep_time)
                    continue
                else:
                    return None
            except ContractLogicError as e:
                print(f"Contract logic error: {e}")
                return None
            except Exception as e:
                print(f"Error executing transaction (attempt {attempt}): {e}")
                if attempt < self.MAX_RETRIES:
                    sleep_time = self.RETRY_DELAY * attempt
                    await asyncio.sleep(sleep_time)
                    continue
                else:
                    return None
        return None

    async def sign_transaction(self, transaction: Dict[str, Any]) -> bytes:
        """
        Sign a transaction.

        Args:
            transaction (Dict[str, Any]): The transaction dictionary.

        Returns:
            bytes: The signed transaction.
        """
        try:
            signed_tx = self.web3.eth.account.sign_transaction(
                transaction,
                private_key=self.account.key,
            )
            return signed_tx.rawTransaction
        except KeyError as e:
            print(f"KeyError in sign_transaction: {e}")
            raise
        except Exception as e:
            print(f"Error signing transaction: {e}")
            raise

    async def handle_eth_transaction(self, target_tx: Dict[str, Any]) -> bool:
        """
        Handle a high-value ETH transaction.

        Args:
            target_tx (Dict[str, Any]): The target transaction data.

        Returns:
            bool: True if handled successfully, False otherwise.
        """
        tx_hash = target_tx.get("tx_hash", "Unknown")
        try:
            eth_value = target_tx.get("value", 0)
            if eth_value <= 0:
                return False
            tx_details = {
                "to": target_tx.get("to", ""),
                "value": eth_value,
                "gas": 21_000,
                "nonce": await self.noncecore.get_nonce(),
                "chainId": await self.web3.eth.chain_id,
                "from": self.account.address,
            }
            original_gas_price = int(target_tx.get("gasPrice", 0))
            if original_gas_price <= 0:
                return False
            tx_details["gasPrice"] = int(
                original_gas_price * 1.1
            )
            eth_value_ether = self.web3.from_wei(eth_value, "ether")
            tx_hash_executed = await self.execute_transaction(tx_details)
            if tx_hash_executed:
                print(f"Executed ETH transaction: {tx_hash_executed}")
                return True
            else:
                print(f"Failed to execute ETH transaction for {tx_hash}")
                return False
        except KeyError as e:
            print(f"KeyError in handle_eth_transaction: {e}")
            return False
        except Exception as e:
            print(f"Exception in handle_eth_transaction: {e}")
            return False

    def calculate_flashloan_amount(self, target_tx: Dict[str, Any]) -> int:
        """
        Calculate the flashloan amount based on estimated profit.

        Args:
            target_tx (Dict[str, Any]): The target transaction data.

        Returns:
            int: The flashloan amount in wei.
        """
        estimated_profit = target_tx.get("profit", 0)
        if estimated_profit > 0:
            flashloan_amount = int(
                Decimal(estimated_profit) * Decimal("0.8") * Decimal("1e18")
            )
            return flashloan_amount
        else:
            return 0

    async def simulate_transaction(self, transaction: Dict[str, Any]) -> bool:
        """
        Simulate a transaction to check for success.

        Args:
            transaction (Dict[str, Any]): The transaction dictionary.

        Returns:
            bool: True if simulation is successful, False otherwise.
        """
        try:
            await self.web3.eth.call(transaction, block_identifier="pending")
            return True
        except ContractLogicError as e:
            print(f"Contract logic error during simulation: {e}")
            return False
        except Exception as e:
            print(f"Exception during transaction simulation: {e}")
            return False

    async def prepare_flashloan_transaction(
        self, flashloan_asset: str, flashloan_amount: int
    ) -> Optional[Dict[str, Any]]:
        """
        Prepare a flashloan transaction.

        Args:
            flashloan_asset (str): The asset to flashloan.
            flashloan_amount (int): The amount to flashloan.

        Returns:
            Optional[Dict[str, Any]]: The flashloan transaction dictionary or None.
        """
        if flashloan_amount <= 0:
            return None
        try:
            flashloan_function = self.flashloan_contract.functions.RequestFlashLoan(
                self.web3.to_checksum_address(flashloan_asset), flashloan_amount
            )
            flashloan_tx = await self.build_transaction(flashloan_function)
            return flashloan_tx
        except ContractLogicError as e:
            print(f"Contract logic error preparing flashloan: {e}")
            return None
        except Exception as e:
            print(f"Exception preparing flashloan transaction: {e}")
            return None

    async def send_bundle(self, transactions: List[Dict[str, Any]]) -> bool:
        """
        Send a bundle of transactions to an MEV relay.

        Args:
            transactions (List[Dict[str, Any]]): List of transaction dictionaries.

        Returns:
            bool: True if the bundle was sent successfully, False otherwise.
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
            mev_builders = [
                {
                    "name": "Flashbots",
                    "url": "https://relay.flashbots.net",
                    "auth_header": "X-Flashbots-Signature"
                },
            ]
            successes = []
            for builder in mev_builders:
                headers = {
                    "Content-Type": "application/json",
                     builder["auth_header"]: f"{self.account.address}:{self.account.key}",
               }
                for attempt in range(1, self.retry_attempts + 1):
                    try:
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
                                    raise ValueError(response_data["error"])
                                successes.append(builder['name'])
                                break
                    except aiohttp.ClientResponseError as e:
                        print(f"Client response error sending bundle to {builder['name']}: {e}")
                        if attempt < self.retry_attempts:
                            sleep_time = self.RETRY_DELAY * attempt
                            await asyncio.sleep(sleep_time)
                    except ValueError as e:
                        print(f"ValueError sending bundle to {builder['name']}: {e}")
                        break
                    except Exception as e:
                        print(f"Unexpected error sending bundle to {builder['name']}: {e}")
                        if attempt < self.retry_attempts:
                            sleep_time = self.RETRY_DELAY * attempt
                            await asyncio.sleep(sleep_time)
            if successes:
                await self.noncecore.refresh_nonce()
                print(f"Successfully sent bundle to: {', '.join(successes)}")
                return True
            else:
                print("Failed to send bundle to any MEV builders.")
                return False
        except Exception as e:
            print(f"Error sending bundle: {e}")
            return False

    async def front_run(self, target_tx: Dict[str, Any]) -> bool:
        """
        Execute a front-run strategy.

        Args:
            target_tx (Dict[str, Any]): The target transaction data.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            return await self.execute_sandwich_attack(target_tx)
        except Exception as e:
            print(f"Error executing front-run: {e}")
            return False

    async def back_run(self, target_tx: Dict[str, Any]) -> bool:
        """
        Execute a back-run strategy.

        Args:
            target_tx (Dict[str, Any]): The target transaction data.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            return await self.execute_sandwich_attack(target_tx)
        except Exception as e:
            print(f"Error executing back-run: {e}")
            return False
        
    async def sandwich_attack(self, target_tx: Dict[str, Any]) -> bool:
        """
        Execute a sandwich attack. This method is a wrapper for the sandwich attack.

        Args:
            target_tx (Dict[str, Any]): The target transaction data.

        Returns:    
            bool: True if successful, False otherwise.
        """
        try:
            return await self.execute_sandwich_attack(target_tx)
        except Exception as e:
            print(f"Error executing sandwich attack: {e}")
            return False
        

    async def execute_sandwich_attack(self, target_tx: Dict[str, Any]) -> bool:
        """
        Execute a sandwich attack consisting of front-run and back-run transactions.

        Args:
            target_tx (Dict[str, Any]): The target transaction data.

        Returns:
            bool: True if the sandwich attack was successful, False otherwise.
        """
        try:
            decoded_tx = await self.decode_transaction_input(
                target_tx.get("input", "0x"),
                self.web3.to_checksum_address(target_tx.get("to", ""))
            )
            if not decoded_tx or "params" not in decoded_tx:
                print("Invalid transaction data for sandwich attack.")
                return False
            path = decoded_tx["params"].get("path", [])
            if len(path) < 2:
                print("Insufficient path length for sandwich attack.")
                return False
            flashloan_asset = self.web3.to_checksum_address(path[0])
            flashloan_amount = self.calculate_flashloan_amount(target_tx)
            if flashloan_amount <= 0:
                print("Calculated flashloan amount is non-positive.")
                return False
            flashloan_tx = await self.prepare_flashloan_transaction(
                flashloan_asset, flashloan_amount
            )
            if not flashloan_tx:
                print("Failed to prepare flashloan transaction.")
                return False
            front_run_tx_details = await self._prepare_front_run_transaction(target_tx)
            if not front_run_tx_details:
                print("Failed to prepare front-run transaction.")
                return False
            back_run_tx_details = await self._prepare_back_run_transaction(target_tx, decoded_tx)
            if not back_run_tx_details:
                print("Failed to prepare back-run transaction.")
                return False
            simulation_results = await asyncio.gather(
                self.simulate_transaction(flashloan_tx),
                self.simulate_transaction(front_run_tx_details),
                self.simulate_transaction(back_run_tx_details),
                return_exceptions=True
            )
            if any(isinstance(result, Exception) for result in simulation_results):
                print("Simulation failed for one or more transactions.")
                return False
            if not all(simulation_results):
                print("One or more transaction simulations failed.")
                return False
            if await self.send_bundle([flashloan_tx, front_run_tx_details, back_run_tx_details]):
                print("Sandwich attack executed successfully.")
                return True
            else:
                print("Failed to execute sandwich attack.")
                return False
        except Exception as e:
            print(f"Error executing sandwich attack: {e}")
            return False

    async def _prepare_front_run_transaction(
        self, target_tx: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Prepare the front-run transaction.

        Args:
            target_tx (Dict[str, Any]): The target transaction data.

        Returns:
            Optional[Dict[str, Any]]: The front-run transaction dictionary or None.
        """
        try:
            decoded_tx = await self.decode_transaction_input(
                target_tx.get("input", "0x"),
                self.web3.to_checksum_address(target_tx.get("to", ""))
            )
            if not decoded_tx:
                print("Failed to decode transaction for front-run.")
                return None
            function_name = decoded_tx.get("function_name")
            if not function_name:
                print("Function name missing in decoded transaction for front-run.")
                return None
            function_params = decoded_tx.get("params", {})
            to_address = self.web3.to_checksum_address(target_tx.get("to", ""))
            routers = {
                self.configuration.UNISWAP_ROUTER_ADDRESS: (self.uniswap_router_contract, "Uniswap"),
                self.configuration.SUSHISWAP_ROUTER_ADDRESS: (self.sushiswap_router_contract, "Sushiswap"),
                self.configuration.PANCAKESWAP_ROUTER_ADDRESS: (self.pancakeswap_router_contract, "Pancakeswap"),
                self.configuration.BALANCER_ROUTER_ADDRESS: (self.balancer_router_contract, "Balancer")
            }
            if to_address not in routers:
                print(f"Router {to_address} not recognized for front-run.")
                return None
            router_contract, exchange_name = routers[to_address]
            if not router_contract:
                print(f"No router contract found for {exchange_name}.")
                return None
            try:
                front_run_function = getattr(router_contract.functions, function_name)(**function_params)
            except AttributeError:
                print(f"Function {function_name} not found in {exchange_name} router.")
                return None
            front_run_tx = await self.build_transaction(front_run_function)
            return front_run_tx
        except Exception as e:
            print(f"Error preparing front-run transaction: {e}")
            return None

    async def _prepare_back_run_transaction(
        self, target_tx: Dict[str, Any], decoded_tx: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Prepare the back-run transaction.

        Args:
            target_tx (Dict[str, Any]): The target transaction data.
            decoded_tx (Dict[str, Any]): The decoded transaction data.

        Returns:
            Optional[Dict[str, Any]]: The back-run transaction dictionary or None.
        """
        try:
            function_name = decoded_tx.get("function_name")
            if not function_name:
                print("Function name missing in decoded transaction for back-run.")
                return None
            function_params = decoded_tx.get("params", {})
            path = function_params.get("path", [])
            if not path or not isinstance(path, list) or len(path) < 2:
                print("Insufficient path length for back-run.")
                return None
            reversed_path = path[::-1]
            function_params["path"] = reversed_path
            to_address = self.web3.to_checksum_address(target_tx.get("to", ""))
            routers = {
                self.configuration.UNISWAP_ROUTER_ADDRESS: (self.uniswap_router_contract, "Uniswap"),
                self.configuration.SUSHISWAP_ROUTER_ADDRESS: (self.sushiswap_router_contract, "Sushiswap"),
                self.configuration.PANCAKESWAP_ROUTER_ADDRESS: (self.pancakeswap_router_contract, "Pancakeswap"),
                self.configuration.BALANCER_ROUTER_ADDRESS: (self.balancer_router_contract, "Balancer")
            }
            if to_address not in routers:
                print(f"Router {to_address} not recognized for back-run.")
                return None
            router_contract, exchange_name = routers[to_address]
            if not router_contract:
                print(f"No router contract found for {exchange_name}.")
                return None
            try:
                back_run_function = getattr(router_contract.functions, function_name)(**function_params)
            except AttributeError:
                print(f"Function {function_name} not found in {exchange_name} router.")
                return None
            back_run_tx = await self.build_transaction(back_run_function)
            return back_run_tx
        except Exception as e:
            print(f"Error preparing back-run transaction: {e}")
            return None

    async def decode_transaction_input(self, input_data: str, contract_address: str) -> Optional[Dict[str, Any]]:
        """
        Decode the input data of a transaction.

        Args:
            input_data (str): The input data in hex.
            contract_address (str): The contract address.

        Returns:
            Optional[Dict[str, Any]]: Decoded transaction data or None.
        """
        try:
            contract = self.web3.eth.contract(address=contract_address, abi=self.erc20_abi)
            function_obj, function_params = contract.decode_function_input(input_data)
            decoded_data = {
                "function_name": function_obj.fn_name,
                "params": function_params,
            }
            return decoded_data
        except ContractLogicError as e:
            print(f"Contract logic error decoding transaction input: {e}")
            return None
        except Exception as e:
            print(f"Exception decoding transaction input: {e}")
            return None

    async def cancel_transaction(self, nonce: int) -> bool:
        """
        Cancel a pending transaction by sending a 0 ETH transaction with the same nonce.

        Args:
            nonce (int): The nonce of the transaction to cancel.

        Returns:
            bool: True if canceled successfully, False otherwise.
        """
        cancel_tx = {
            "to": self.account.address,
            "value": 0,
            "gas": 21_000,
            "gasPrice": self.web3.to_wei("150", "gwei"),
            "nonce": nonce,
            "chainId": await self.web3.eth.chain_id,
            "from": self.account.address,
        }
        try:
            signed_cancel_tx = await self.sign_transaction(cancel_tx)
            tx_hash = await self.web3.eth.send_raw_transaction(signed_cancel_tx)
            tx_hash_hex = tx_hash.hex() if isinstance(tx_hash, HexBytes) else tx_hash
            print(f"Canceled transaction with nonce {nonce}: {tx_hash_hex}")
            return True
        except Exception as e:
            print(f"Error canceling transaction with nonce {nonce}: {e}")
            return False

    async def estimate_gas_limit(self, tx: Dict[str, Any]) -> int:
        """
        Estimate the gas limit for a transaction.

        Args:
            tx (Dict[str, Any]): The transaction dictionary.

        Returns:
            int: The estimated gas limit.
        """
        try:
            gas_estimate = await self.web3.eth.estimate_gas(tx)
            return gas_estimate
        except Exception as e:
            print(f"Error estimating gas limit: {e}")
            return 100_000

    async def get_current_profit(self) -> Decimal:
        """
        Get the current profit from the account balance.

        Returns:
            Decimal: The current profit in ETH.
        """
        try:
            current_profit = await self.safetynet.get_balance(self.account)
            self.current_profit = Decimal(current_profit)
            return self.current_profit
        except Exception as e:
            print(f"Error getting current profit: {e}")
            return Decimal("0")

    async def withdraw_eth(self) -> bool:
        """
        Withdraw ETH from the flashloan contract.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            withdraw_function = self.flashloan_contract.functions.withdrawETH()
            tx = await self.build_transaction(withdraw_function)
            tx_hash = await self.execute_transaction(tx)
            if tx_hash:
                print(f"Withdrew ETH: {tx_hash}")
                return True
            else:
                print("Failed to withdraw ETH.")
                return False
        except ContractLogicError as e:
            print(f"Contract logic error withdrawing ETH: {e}")
            return False
        except Exception as e:
            print(f"Exception withdrawing ETH: {e}")
            return False

    async def withdraw_token(self, token_address: str) -> bool:
        """
        Withdraw a specific token from the flashloan contract.

        Args:
            token_address (str): The token's contract address.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            withdraw_function = self.flashloan_contract.functions.withdrawToken(
                self.web3.to_checksum_address(token_address)
            )
            tx = await self.build_transaction(withdraw_function)
            tx_hash = await self.execute_transaction(tx)
            if tx_hash:
                print(f"Withdrew token {token_address}: {tx_hash}")
                return True
            else:
                print(f"Failed to withdraw token {token_address}.")
                return False
        except ContractLogicError as e:
            print(f"Contract logic error withdrawing token {token_address}: {e}")
            return False
        except Exception as e:
            print(f"Exception withdrawing token {token_address}: {e}")
            return False

    async def transfer_profit_to_account(self, amount: Decimal, account: str) -> bool:
        """
        Transfer profit to another account.

        Args:
            amount (Decimal): The amount to transfer in ETH.
            account (str): The recipient's account address.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            transfer_function = self.flashloan_contract.functions.transfer(
                self.web3.to_checksum_address(account), int(amount * Decimal("1e18"))
            )
            tx = await self.build_transaction(transfer_function)
            tx_hash = await self.execute_transaction(tx)
            if tx_hash:
                print(f"Transferred profit to {account}: {tx_hash}")
                return True
            else:
                print(f"Failed to transfer profit to {account}.")
                return False
        except ContractLogicError as e:
            print(f"Contract logic error transferring profit: {e}")
            return False
        except Exception as e:
            print(f"Exception transferring profit: {e}")
            return False

class MarketMonitor:
    """
    Market monitoring class with machine learning capabilities.
    """
    MODEL_UPDATE_INTERVAL: int = 3600  # seconds
    VOLATILITY_THRESHOLD: float = 0.05
    LIQUIDITY_THRESHOLD: float = 100000.0

    def __init__(
        self,
        web3: AsyncWeb3,
        configuration: Optional[Configuration] = None,
        apiconfig: Optional[APIConfig] = None,
    ) -> None:
        self.web3 = web3
        self.configuration = configuration or Configuration()
        self.apiconfig = apiconfig or APIConfig(self.configuration)
        self.price_model: LinearRegression = LinearRegression()
        self.model_last_updated: float = 0.0
        self.price_cache: TTLCache = TTLCache(maxsize=1000, ttl=300)
        self.model_path: str = self.configuration.ML_MODEL_PATH
        self.training_data_path: str = self.configuration.ML_TRAINING_DATA_PATH
        self.model_lock: asyncio.Lock = asyncio.Lock()

    async def load_model(self) -> None:
        """
        Load the ML model from disk.
        """
        async with self.model_lock:
            if os.path.exists(self.model_path) and os.path.exists(self.training_data_path):
                try:
                    data = load(self.model_path)
                    self.price_model = data['model']
                    self.model_last_updated = data.get('model_last_updated', 0.0)
                    print("ML model loaded successfully.")
                except Exception as e:
                    print(f"Error loading ML model: {e}")
            else:
                print("ML model or training data not found. Starting with a new model.")

    async def save_model(self) -> None:
        """
        Save the ML model to disk.
        """
        async with self.model_lock:
            try:
                data = {
                    'model': self.price_model,
                    'model_last_updated': self.model_last_updated
                }
                dump(data, self.model_path)
                print("ML model saved successfully.")
            except Exception as e:
                print(f"Error saving ML model: {e}")

    async def _update_price_model(self, token_symbol: str) -> None:
        """
        Update the ML price prediction model with new data.

        Args:
            token_symbol (str): The token symbol to update the model for.
        """
        async with self.model_lock:
            try:
                prices = await self.fetch_historical_prices(token_symbol)
                if len(prices) > 10:
                    X = np.arange(len(prices)).reshape(-1, 1)
                    y = np.array(prices)
                    if os.path.exists(self.training_data_path):
                        existing_data = pd.read_csv(self.training_data_path)
                        X_existing = existing_data[['time']].values
                        y_existing = existing_data['price'].values
                        X = np.vstack((X_existing, X))
                        y = np.concatenate((y_existing, y))
                    training_df = pd.DataFrame({'time': X.flatten(), 'price': y})
                    training_df.to_csv(self.training_data_path, index=False)
                    self.price_model.fit(X, y)
                    self.model_last_updated = time.time()
                    await self.save_model()
                    print(f"ML model updated for {token_symbol}.")
                else:
                    print(f"Not enough data to update ML model for {token_symbol}.")
            except Exception as e:
                print(f"Error updating ML model for {token_symbol}: {e}")

    async def periodic_model_training(self, token_symbol: str) -> None:
        """
        Periodically train the ML model.

        Args:
            token_symbol (str): The token symbol to train the model for.
        """
        while True:
            try:
                current_time = time.time()
                if current_time - self.model_last_updated > self.MODEL_UPDATE_INTERVAL:
                    await self._update_price_model(token_symbol)
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in periodic model training: {e}")

    async def start_periodic_training(self, token_symbol: str) -> None:
        """
        Start the periodic training task.

        Args:
            token_symbol (str): The token symbol to train the model for.
        """
        asyncio.create_task(self.periodic_model_training(token_symbol))

    async def check_market_conditions(self, token_address: str) -> Dict[str, Any]:
        """
        Check various market conditions for a token.

        Args:
            token_address (str): The token's contract address.

        Returns:
            Dict[str, Any]: A dictionary of market conditions.
        """
        market_conditions: Dict[str, Any] = {
            "high_volatility": False,
            "bullish_trend": False,
            "bearish_trend": False,
            "low_liquidity": False,
        }
        token_symbol = await self.apiconfig.get_token_symbol(self.web3, token_address)
        if not token_symbol:
            return market_conditions
        prices = await self.fetch_historical_prices(token_symbol, days=1)
        if len(prices) < 2:
            return market_conditions
        volatility = self._calculate_volatility(prices)
        if volatility > self.VOLATILITY_THRESHOLD:
            market_conditions["high_volatility"] = True
        moving_average = np.mean(prices)
        if prices[-1] > moving_average:
            market_conditions["bullish_trend"] = True
        elif prices[-1] < moving_average:
            market_conditions["bearish_trend"] = True
        volume = await self.apiconfig.get_token_volume(token_symbol)
        if volume < self.LIQUIDITY_THRESHOLD:
            market_conditions["low_liquidity"] = True
        return market_conditions

    def _calculate_volatility(self, prices: List[float]) -> float:
        """
        Calculate the volatility of a token based on historical prices.

        Args:
            prices (List[float]): List of historical prices.

        Returns:
            float: The calculated volatility.
        """
        try:
            prices_array = np.array(prices)
            returns = np.diff(prices_array) / prices_array[:-1]
            return np.std(returns)
        except Exception as e:
            print(f"Error calculating volatility: {e}")
            return 0.0

    async def fetch_historical_prices(self, token_symbol: str, days: int = 30) -> List[float]:
        """
        Fetch historical prices for a token.

        Args:
            token_symbol (str): The token symbol.
            days (int): Number of days to fetch.

        Returns:
            List[float]: List of historical prices.
        """
        return await self.apiconfig.fetch_historical_prices(token_symbol, days, service="primary")

    async def get_token_volume(self, token_symbol: str) -> float:
        """
        Get the trading volume of a token.

        Args:
            token_symbol (str): The token symbol.

        Returns:
            float: The trading volume.
        """
        return await self.apiconfig.get_token_volume(token_symbol, service="primary")

    async def _fetch_from_services(
        self,
        fetch_func: Callable[[str], Any],
        description: str
    ) -> Optional[Any]:
        """
        Fetch data from multiple services until success.

        Args:
            fetch_func (Callable[[str], Any]): The function to fetch data.
            description (str): Description for logging.

        Returns:
            Optional[Any]: The fetched data or None.
        """
        return await self.apiconfig._fetch_from_services(fetch_func, description)

    async def predict_price_movement(self, token_symbol: str) -> float:
        """
        Predict the future price movement of a token.

        Args:
            token_symbol (str): The token symbol.

        Returns:
            float: The predicted price.
        """
        current_time = time.time()
        if current_time - self.model_last_updated > self.MODEL_UPDATE_INTERVAL:
            await self._update_price_model(token_symbol)
        prices = await self.fetch_historical_prices(token_symbol, days=1)
        if not prices:
            return 0.0
        try:
            X_pred = np.array([[len(prices)]])
            predicted_price = self.price_model.predict(X_pred)[0]
            return float(predicted_price)
        except Exception as e:
            print(f"Error predicting price movement for {token_symbol}: {e}")
            return 0.0

    async def is_arbitrage_opportunity(self, target_tx: Dict[str, Any]) -> bool:
        """
        Determine if there's an arbitrage opportunity based on the target transaction.

        Args:
            target_tx (Dict[str, Any]): The target transaction data.

        Returns:
            bool: True if an arbitrage opportunity exists, False otherwise.
        """
        decoded_tx = await self.decode_transaction_input(target_tx.get("input", ""), target_tx.get("to", ""))
        if not decoded_tx:
            return False
        path = decoded_tx.get("params", {}).get("path", [])
        if len(path) < 2:
            return False
        token_address = path[-1]
        token_symbol = await self.apiconfig.get_token_symbol(self.web3, token_address)
        if not token_symbol:
            return False
        prices = await self.apiconfig.get_real_time_price(token_symbol, service="primary", vs_currency="eth")
        if not prices:
            return False
        # Implement arbitrage logic as needed
        # Placeholder logic
        return False

class StrategyNet:
    """
    Strategy execution and performance tracking class.
    """
    def __init__(
        self,
        transactioncore: Optional['TransactionCore'] = None,
        marketmonitor: Optional['MarketMonitor'] = None,
        safetynet: Optional['SafetyNet'] = None,
        apiconfig: Optional['APIConfig'] = None,
        configuration: Optional['StrategyConfiguration'] = None,
    ) -> None:
        self.transactioncore = transactioncore
        self.marketmonitor = marketmonitor
        self.safetynet = safetynet
        self.apiconfig = apiconfig
        self.configuration = configuration or StrategyConfiguration()

        self.strategy_types: List[str] = [
            "eth_transaction",
            "front_run",
            "back_run",
            "sandwich_attack"
        ]

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

        self.strategy_performance: Dict[str, StrategyPerformanceMetrics] = {
            strategy_type: StrategyPerformanceMetrics()
            for strategy_type in self.strategy_types
        }

        self.reinforcement_weights: Dict[str, List[float]] = {
            strategy_type: [1.0 for _ in self.get_strategies(strategy_type)]
            for strategy_type in self.strategy_types
        }

        self.history_data: List[Dict[str, Any]] = []

    def register_strategy(
        self,
        strategy_type: str,
        strategy_func: Callable[[Dict[str, Any]], asyncio.Future]
    ) -> None:
        """
        Register a new strategy.

        Args:
            strategy_type (str): The type/category of the strategy.
            strategy_func (Callable[[Dict[str, Any]], asyncio.Future]): The strategy function.
        """
        if strategy_type not in self.strategy_types:
            print(f"Strategy type {strategy_type} is not recognized.")
            return
        self._strategy_registry[strategy_type].append(strategy_func)
        self.reinforcement_weights[strategy_type].append(1.0)

    def get_strategies(
        self,
        strategy_type: str
    ) -> List[Callable[[Dict[str, Any]], asyncio.Future]]:
        """
        Get the list of strategies for a given type.

        Args:
            strategy_type (str): The strategy type.

        Returns:
            List[Callable[[Dict[str, Any]], asyncio.Future]]: List of strategy functions.
        """
        return self._strategy_registry.get(strategy_type, [])

    async def execute_best_strategy(
        self,
        target_tx: Dict[str, Any],
        strategy_type: str
    ) -> bool:
        """
        Execute the best-performing strategy for a given transaction and strategy type.

        Args:
            target_tx (Dict[str, Any]): The target transaction data.
            strategy_type (str): The strategy type to execute.

        Returns:
            bool: True if successful, False otherwise.
        """
        strategies = self.get_strategies(strategy_type)
        if not strategies:
            print(f"No strategies registered for type {strategy_type}.")
            return False

        try:
            start_time = time.time()
            selected_strategy = await self._select_best_strategy(strategies, strategy_type)

            profit_before = await self.transactioncore.get_current_profit()

            success = await selected_strategy(target_tx)

            profit_after = await self.transactioncore.get_current_profit()

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
            print(f"Strategy execution error: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error executing strategy: {e}")
            return False

    async def _select_best_strategy(
        self,
        strategies: List[Callable[[Dict[str, Any]], asyncio.Future]],
        strategy_type: str
    ) -> Callable[[Dict[str, Any]], asyncio.Future]:
        """
        Select the best strategy based on reinforcement weights.

        Args:
            strategies (List[Callable[[Dict[str, Any]], asyncio.Future]]): List of strategy functions.
            strategy_type (str): The strategy type.

        Returns:
            Callable[[Dict[str, Any]], asyncio.Future]: The selected strategy function.
        """
        weights = self.reinforcement_weights[strategy_type]
        if not weights:
            print(f"No reinforcement weights available for strategy type {strategy_type}. Selecting randomly.")
            return random.choice(strategies)

        if random.random() < self.configuration.exploration_rate:
            print(f"Exploration: Selecting random strategy for {strategy_type}.")
            return random.choice(strategies)

        max_weight = max(weights)
        exp_weights = [np.exp(w - max_weight) for w in weights]
        sum_exp = sum(exp_weights)
        probabilities = [w / sum_exp for w in exp_weights]

        selected_index = np.random.choice(len(strategies), p=probabilities)
        selected_strategy = strategies[selected_index]
        print(f"Selected strategy {selected_strategy.__name__} for {strategy_type}.")
        return selected_strategy

    async def _update_strategy_metrics(
        self,
        strategy_name: str,
        strategy_type: str,
        success: bool,
        profit: Decimal,
        execution_time: float
    ) -> None:
        """
        Update the performance metrics of a strategy.

        Args:
            strategy_name (str): The name of the strategy.
            strategy_type (str): The strategy type.
            success (bool): Whether the strategy succeeded.
            profit (Decimal): The profit made.
            execution_time (float): The time taken to execute.
        """
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
        print(f"Updated metrics for {strategy_name}: Success={success}, Profit={profit}, Execution Time={execution_time}s")

    def get_strategy_index(self, strategy_name: str, strategy_type: str) -> int:
        """
        Get the index of a strategy within its type.

        Args:
            strategy_name (str): The name of the strategy.
            strategy_type (str): The strategy type.

        Returns:
            int: The index of the strategy or -1 if not found.
        """
        strategies = self.get_strategies(strategy_type)
        for index, strategy in enumerate(strategies):
            if strategy.__name__ == strategy_name:
                return index
        return -1

    def _calculate_reward(
        self,
        success: bool,
        profit: Decimal,
        execution_time: float
    ) -> float:
        """
        Calculate the reinforcement learning reward.

        Args:
            success (bool): Whether the strategy succeeded.
            profit (Decimal): The profit made.
            execution_time (float): The time taken to execute.

        Returns:
            float: The calculated reward.
        """
        base_reward = float(profit) if success else -0.1
        time_penalty = -0.01 * execution_time
        total_reward = base_reward + time_penalty
        return total_reward

    def _update_reinforcement_weight(
        self,
        strategy_type: str,
        index: int,
        reward: float
    ) -> None:
        """
        Update the reinforcement weight for a strategy based on the reward.

        Args:
            strategy_type (str): The strategy type.
            index (int): The index of the strategy.
            reward (float): The reward to apply.
        """
        lr = self.configuration.learning_rate
        current_weight = self.reinforcement_weights[strategy_type][index]
        new_weight = current_weight * (1 - lr) + reward * lr
        self.reinforcement_weights[strategy_type][index] = max(0.1, new_weight)
        print(f"Updated weight for strategy {index} in {strategy_type}: {new_weight}")

    async def _decode_transaction(self, target_tx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Decode a target transaction.

        Args:
            target_tx (Dict[str, Any]): The target transaction data.

        Returns:
            Optional[Dict[str, Any]]: The decoded transaction data or None.
        """
        try:
            decoded = await self.transactioncore.decode_transaction_input(
                target_tx.get("input", ""), target_tx.get("to", "")
            )
            return decoded
        except Exception as e:
            print(f"Error decoding transaction: {e}")
            return None

    async def _get_token_symbol(self, token_address: str) -> Optional[str]:
        """
        Get the symbol of a token.

        Args:
            token_address (str): The token's contract address.

        Returns:
            Optional[str]: The token symbol or None.
        """
        try:
            symbol = await self.apiconfig.get_token_symbol(
                self.transactioncore.web3, token_address
            )
            return symbol
        except Exception as e:
            print(f"Error getting token symbol for {token_address}: {e}")
            return None

    async def high_value_eth_transfer(self, target_tx: Dict[str, Any]) -> bool:
        """
        Strategy for handling high-value ETH transfers.

        Args:
            target_tx (Dict[str, Any]): The target transaction data.

        Returns:
            bool: True if handled successfully, False otherwise.
        """
        try:
            if not self._is_valid_transaction(target_tx):
                return False

            eth_value_in_wei, gas_price, to_address = self._extract_transaction_details(target_tx)
            eth_value, gas_price_gwei, threshold = self._calculate_thresholds(eth_value_in_wei, gas_price)

            if not await self._additional_validation_checks(eth_value_in_wei, to_address):
                return False

            if eth_value_in_wei > threshold:
                return await self.transactioncore.handle_eth_transaction(target_tx)

            return False

        except Exception as e:
            print(f"Error in high_value_eth_transfer strategy: {e}")
            return False

    def _is_valid_transaction(self, target_tx: Dict[str, Any]) -> bool:
        """
        Validate the structure of a transaction.

        Args:
            target_tx (Dict[str, Any]): The transaction data.

        Returns:
            bool: True if valid, False otherwise.
        """
        if not isinstance(target_tx, dict) or not target_tx:
            print("Invalid transaction format.")
            return False
        return True

    def _extract_transaction_details(self, target_tx: Dict[str, Any]) -> Tuple[int, int, str]:
        """
        Extract transaction details.

        Args:
            target_tx (Dict[str, Any]): The transaction data.

        Returns:
            Tuple[int, int, str]: ETH value in wei, gas price, and recipient address.
        """
        eth_value_in_wei = int(target_tx.get("value", 0))
        gas_price = int(target_tx.get("gasPrice", 0))
        to_address = target_tx.get("to", "")
        return eth_value_in_wei, gas_price, to_address

    def _calculate_thresholds(self, eth_value_in_wei: int, gas_price: int) -> Tuple[float, float, int]:
        """
        Calculate thresholds for high-value ETH transfers.

        Args:
            eth_value_in_wei (int): ETH value in wei.
            gas_price (int): Gas price in wei.

        Returns:
            Tuple[float, float, int]: ETH value, gas price in gwei, and threshold.
        """
        eth_value = eth_value_in_wei / 1e18
        gas_price_gwei = gas_price / 1e9
        threshold = self.configuration.HIGH_VALUE_ETH_THRESHOLD
        return eth_value, gas_price_gwei, threshold * 1e18 
    
    async def _additional_validation_checks(self, eth_value_in_wei: int, to_address: str) -> bool:
        """
        Perform additional validation checks.

        Args:
            eth_value_in_wei (int): ETH value in wei.
            to_address (str): The recipient address.

        Returns:
            bool: True if all checks pass, False otherwise.
        """
        if eth_value_in_wei <= 0:
            print("Invalid ETH value.")
            return False
        if not to_address:
            print("Recipient address missing.")
            return False
        return True
    
    async def aggressive_front_run(self, target_tx: Dict[str, Any]) -> bool:
        """
        Strategy for aggressive front-running.

        Args:
            target_tx (Dict[str, Any]): The target transaction data.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            if not self._is_valid_transaction(target_tx):
                return False

            decoded_tx = await self._decode_transaction(target_tx)
            if not decoded_tx:
                return False

            path = decoded_tx.get("params", {}).get("path", [])
            if len(path) < 2:
                return False

            token_address = path[-1]
            token_symbol = await self._get_token_symbol(token_address)
            if not token_symbol:
                return False

            market_conditions = await self.marketmonitor.check_market_conditions(token_address)
            if market_conditions.get("high_volatility"):
                return await self.transactioncore.front_run(target_tx)

            return False

        except Exception as e:
            print(f"Error in aggressive_front_run strategy: {e}")
            return False
    
    async def predictive_front_run(self, target_tx: Dict[str, Any]) -> bool:
        """
        Strategy for predictive front-running.

        Args:
            target_tx (Dict[str, Any]): The target transaction data.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            if not self._is_valid_transaction(target_tx):
                return False

            decoded_tx = await self._decode_transaction(target_tx)
            if not decoded_tx:
                return False

            path = decoded_tx.get("params", {}).get("path", [])
            if len(path) < 2:
                return False

            token_address = path[-1]
            token_symbol = await self._get_token_symbol(token_address)
            if not token_symbol:
                return False

            predicted_price = await self.marketmonitor.predict_price_movement(token_symbol)
            if predicted_price > 0:
                return await self.transactioncore.front_run(target_tx)

            return False

        except Exception as e:
            print(f"Error in predictive_front_run strategy: {e}")
            return False
    
    async def volatility_front_run(self, target_tx: Dict[str, Any]) -> bool:
        """
        Strategy for front-running based on volatility.

        Args:
            target_tx (Dict[str, Any]): The target transaction data.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            if not self._is_valid_transaction(target_tx):
                return False

            decoded_tx = await self._decode_transaction(target_tx)
            if not decoded_tx:
                return False

            path = decoded_tx.get("params", {}).get("path", [])
            if len(path) < 2:
                return False

            token_address = path[-1]
            token_symbol = await self._get_token_symbol(token_address)
            if not token_symbol:
                return False

            market_conditions = await self.marketmonitor.check_market_conditions(token_address)
            if market_conditions.get("high_volatility"):
                return await self.transactioncore.front_run(target_tx)

            return False

        except Exception as e:
            print(f"Error in volatility_front_run strategy: {e}")
            return False
    
    async def advanced_front_run(self, target_tx: Dict[str, Any]) -> bool:
        """
        Advanced front-running strategy.

        Args:
            target_tx (Dict[str, Any]): The target transaction data.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            if not self._is_valid_transaction(target_tx):
                return False

            decoded_tx = await self._decode_transaction(target_tx)
            if not decoded_tx:
                return False

            path = decoded_tx.get("params", {}).get("path", [])
            if len(path) < 2:
                return False

            token_address = path[-1]
            token_symbol = await self._get_token_symbol(token_address)
            if not token_symbol:
                return False

            predicted_price = await self.marketmonitor.predict_price_movement(token_symbol)
            if predicted_price > 0:
                return await self.transactioncore.front_run(target_tx)

            return False

        except Exception as e:
            print(f"Error in advanced_front_run strategy: {e}")
            return False
    
    async def price_dip_back_run(self, target_tx: Dict[str, Any]) -> bool:
        """
        Strategy for back-running based on price dips.

        Args:
            target_tx (Dict[str, Any]): The target transaction data.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            if not self._is_valid_transaction(target_tx):
                return False

            decoded_tx = await self._decode_transaction(target_tx)
            if not decoded_tx:
                return False

            path = decoded_tx.get("params", {}).get("path", [])
            if len(path) < 2:
                return False

            token_address = path[-1]
            token_symbol = await self._get_token_symbol(token_address)
            if not token_symbol:
                return False

            predicted_price = await self.marketmonitor.predict_price_movement(token_symbol)
            if predicted_price > 0:
                return await self.transactioncore.back_run(target_tx)

            return False

        except Exception as e:
            print(f"Error in price_dip_back_run strategy: {e}")
            return False
    
    async def flashloan_back_run(self, target_tx: Dict[str, Any]) -> bool:
        """
        Strategy for back-running with flashloans.

        Args:
            target_tx (Dict[str, Any]): The target transaction data.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            if not self._is_valid_transaction(target_tx):
                return False

            decoded_tx = await self._decode_transaction(target_tx)
            if not decoded_tx:
                return False

            path = decoded_tx.get("params", {}).get("path", [])
            if len(path) < 2:
                return False

            token_address = path[-1]
            token_symbol = await self._get_token_symbol(token_address)
            if not token_symbol:
                return False

            market_conditions = await self.marketmonitor.check_market_conditions(token_address)
            if market_conditions.get("low_liquidity"):
                return await self.transactioncore.back_run(target_tx)

            return False

        except Exception as e:
            print(f"Error in flashloan_back_run strategy: {e}")
            return False
        
    async def high_volume_back_run(self, target_tx: Dict[str, Any]) -> bool:
        """
        Strategy for back-running based on high trading volume.

        Args:
            target_tx (Dict[str, Any]): The target transaction data.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            if not self._is_valid_transaction(target_tx):
                return False

            decoded_tx = await self._decode_transaction(target_tx)
            if not decoded_tx:
                return False

            path = decoded_tx.get("params", {}).get("path", [])
            if len(path) < 2:
                return False

            token_address = path[-1]
            token_symbol = await self._get_token_symbol(token_address)
            if not token_symbol:
                return False

            volume = await self.marketmonitor.get_token_volume(token_symbol)
            if volume > self.configuration.HIGH_VOLUME_THRESHOLD:
                return await self.transactioncore.back_run(target_tx)

            return False

        except Exception as e:
            print(f"Error in high_volume_back_run strategy: {e}")
            return False
        
    async def advanced_back_run(self, target_tx: Dict[str, Any]) -> bool:
        """
        Advanced back-running strategy.

        Args:
            target_tx (Dict[str, Any]): The target transaction data.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            if not self._is_valid_transaction(target_tx):
                return False

            decoded_tx = await self._decode_transaction(target_tx)
            if not decoded_tx:
                return False

            path = decoded_tx.get("params", {}).get("path", [])
            if len(path) < 2:
                return False

            token_address = path[-1]
            token_symbol = await self._get_token_symbol(token_address)
            if not token_symbol:
                return False

            market_conditions = await self.marketmonitor.check_market_conditions(token_address)
            if market_conditions.get("low_liquidity"):
                return await self.transactioncore.back_run(target_tx)

            return False

        except Exception as e:
            print(f"Error in advanced_back_run strategy: {e}")
            return False
    
    async def flash_profit_sandwich(self, target_tx: Dict[str, Any]) -> bool:
        """
        Strategy for sandwich attacks with flash profits.

        Args:
            target_tx (Dict[str, Any]): The target transaction data.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            if not self._is_valid_transaction(target_tx):
                return False

            decoded_tx = await self._decode_transaction(target_tx)
            if not decoded_tx:
                return False

            path = decoded_tx.get("params", {}).get("path", [])
            if len(path) < 2:
                return False

            token_address = path[-1]
            token_symbol = await self._get_token_symbol(token_address)
            if not token_symbol:
                return False

            market_conditions = await self.marketmonitor.check_market_conditions(token_address)
            if market_conditions.get("high_volatility"):
                return await self.transactioncore.sandwich_attack(target_tx)

            return False

        except Exception as e:
            print(f"Error in flash_profit_sandwich strategy: {e}")
            return False
        
    async def price_boost_sandwich(self, target_tx: Dict[str, Any]) -> bool:
        """
        Strategy for sandwich attacks with price boosts.

        Args:
            target_tx (Dict[str, Any]): The target transaction data.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            if not self._is_valid_transaction(target_tx):
                return False

            decoded_tx = await self._decode_transaction(target_tx)
            if not decoded_tx:
                return False

            path = decoded_tx.get("params", {}).get("path", [])
            if len(path) < 2:
                return False

            token_address = path[-1]
            token_symbol = await self._get_token_symbol(token_address)
            if not token_symbol:
                return False

            predicted_price = await self.marketmonitor.predict_price_movement(token_symbol)
            if predicted_price > 0:
                return await self.transactioncore.sandwich_attack(target_tx)

            return False

        except Exception as e:
            print(f"Error in price_boost_sandwich strategy: {e}")
            return False
    
    async def arbitrage_sandwich(self, target_tx: Dict[str, Any]) -> bool:
        """
        Strategy for sandwich attacks with arbitrage opportunities.

        Args:
            target_tx (Dict[str, Any]): The target transaction data.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            if not self._is_valid_transaction(target_tx):
                return False

            return await self.transactioncore.sandwich_attack(target_tx)

        except Exception as e:
            print(f"Error in arbitrage_sandwich strategy: {e}")
            return False
        
    async def advanced_sandwich_attack(self, target_tx: Dict[str, Any]) -> bool:
        """
        Advanced sandwich attack strategy.

        Args:
            target_tx (Dict[str, Any]): The target transaction data.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            if not self._is_valid_transaction(target_tx):
                return False

            decoded_tx = await self._decode_transaction(target_tx)
            if not decoded_tx:
                return False

            path = decoded_tx.get("params", {}).get("path", [])
            if len(path) < 2:
                return False

            token_address = path[-1]
            token_symbol = await self._get_token_symbol(token_address)
            if not token_symbol:
                return False

            predicted_price = await self.marketmonitor.predict_price_movement(token_symbol)
            if predicted_price > 0:
                return await self.transactioncore.sandwich_attack(target_tx)

            return False

        except Exception as e:
            print(f"Error in advanced_sandwich_attack strategy: {e}")
            return False
    
    async def execute_strategy(
        self,
        target_tx: Dict[str, Any],
        strategy_type: str
    ) -> bool:
        """
        Execute a strategy for a given transaction and strategy type.

        Args:
            target_tx (Dict[str, Any]): The target transaction data.
            strategy_type (str): The strategy type to execute.

        Returns:
            bool: True if successful, False otherwise.
        """
        strategies = self.get_strategies(strategy_type)
        if not strategies:
            print(f"No strategies registered for type {strategy_type}.")
            return False

        for strategy in strategies:
            try:
                success = await strategy(target_tx)
                if success:
                    return True
            except Exception as e:
                print(f"Error executing strategy {strategy.__name__}: {e}")

        return False
    
    async def execute_random_strategy(self, target_tx: Dict[str, Any]) -> bool:
        """
        Execute a random strategy for a given transaction.

        Args:
            target_tx (Dict[str, Any]): The target transaction data.

        Returns:
            bool: True if successful, False otherwise.
        """
        strategy_type = random.choice(self.strategy_types)
        return await self.execute_strategy(target_tx, strategy_type)
    
    async def execute_best_strategy(self, target_tx: Dict[str, Any]) -> bool:
        """
        Execute the best-performing strategy for a given transaction.

        Args:
            target_tx (Dict[str, Any]): The target transaction data.

        Returns:
            bool: True if successful, False otherwise.
        """
        best_strategy = self._select_best_strategy()
        return await best_strategy(target_tx)
    
    async def _select_best_strategy(self) -> Callable[[Dict[str, Any]], bool]:
        """
        Select the best strategy based on performance metrics.

        Returns:
            Callable[[Dict[str, Any]], bool]: The selected strategy function.
        """
        best_strategy = self._get_best_strategy()
        print(f"Selected best strategy: {best_strategy.__name__}")
        return best_strategy
    
    def _get_best_strategy(self) -> Callable[[Dict[str, Any]], bool]:
        """
        Get the best-performing strategy based on performance metrics.

        Returns:
            Callable[[Dict[str, Any]], bool]: The best strategy function.
        """
        best_strategy = max(self.strategy_performance, key=self.strategy_performance.get)
        strategies = self.get_strategies(best_strategy)
        return random.choice(strategies) if strategies else self.execute_random_strategy
    
    async def _update_strategy_metrics(
        self,
        strategy_name: str,
        strategy_type: str,
        success: bool,
        profit: Decimal,
        execution_time: float
    ) -> None:
        """
        Update the performance metrics of a strategy.

        Args:
            strategy_name (str): The name of the strategy.
            strategy_type (str): The strategy type.
            success (bool): Whether the strategy succeeded.
            profit (Decimal): The profit made.
            execution_time (float): The time taken to execute.
        """
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
        print(f"Updated metrics for {strategy_name}: Success={success}, Profit={profit}, Execution Time={execution_time}s")

    def _calculate_reward(
        self,
        success: bool,
        profit: Decimal,
        execution_time: float
    ) -> float:
        """
        Calculate the reinforcement learning reward.

        Args:
            success (bool): Whether the strategy succeeded.
            profit (Decimal): The profit made.
            execution_time (float): The time taken to execute.

        Returns:
            float: The calculated reward.
        """
        base_reward = float(profit) if success else -0.1
        time_penalty = -0.01 * execution_time
        total_reward = base_reward + time_penalty
        return total_reward
    
    def _update_reinforcement_weight(
        self,
        strategy_type: str,
        index: int,
        reward: float
    ) -> None:
        """
        Update the reinforcement weight for a strategy based on the reward.

        Args:
            strategy_type (str): The strategy type.
            index (int): The index of the strategy.
            reward (float): The reward to apply.
        """
        lr = self.configuration.learning_rate
        current_weight = self.reinforcement_weights[strategy_type][index]
        new_weight = current_weight * (1 - lr) + reward * lr
        self.reinforcement_weights[strategy_type][index] = max(0.1, new_weight)
        print(f"Updated weight for strategy {index} in {strategy_type}: {new_weight}")

class MainCore:
    def __init__(self, configuration: Optional[Configuration] = None) -> None:
        self.configuration: Configuration = configuration or Configuration()
        self.apiconfig: Optional['APIConfig'] = None
        self.web3: Optional[AsyncWeb3] = None
        self.mempoolmonitor: Optional['MempoolMonitor'] = None
        self.account: Optional['Account'] = None
        self.marketmonitor: Optional[MarketMonitor] = None
        self.transactioncore: Optional['TransactionCore'] = None
        self.strategynet: Optional[StrategyNet] = None
        self.safetynet: Optional['SafetyNet'] = None
        self.noncecore: Optional['NonceCore'] = None

    async def initialize(self) -> None:
        try:
            wallet_key = self.configuration.WALLET_KEY
            if not wallet_key:
                raise ValueError("Wallet key is not set in configuration.")

            try:
                cleaned_key = wallet_key[2:] if wallet_key.startswith('0x') else wallet_key
                if not all(c in '0123456789abcdefABCDEF' for c in cleaned_key) or len(cleaned_key) != 64:
                    raise ValueError("Invalid wallet key format - must be a 64-character hexadecimal string")

                full_key = f"0x{cleaned_key}" if not wallet_key.startswith('0x') else wallet_key
                self.account = Account.from_key(full_key)
            except Exception as e:
                raise ValueError(f"Invalid wallet key format: {e}")

            self.web3 = await self._initialize_web3()
            if not self.web3:
                raise RuntimeError("Failed to initialize Web3 connection")

            await self._check_account_balance()
            await self._initialize_components()
        except Exception:
            await self.stop()

    async def _initialize_web3(self) -> Optional[AsyncWeb3]:
        providers = self._get_providers()
        if not providers:
            return None

        for provider_name, provider in providers:
            try:
                web3 = AsyncWeb3(provider, modules={"eth": (AsyncEth,)})

                if await self._test_connection(web3, provider_name):
                    await self._add_middleware(web3)
                    return web3

            except Exception:
                continue

        return None

    def _get_providers(self) -> List[Tuple[str, Any]]:
        providers: List[Tuple[str, Any]] = []
        if self.configuration.IPC_ENDPOINT and os.path.exists(self.configuration.IPC_ENDPOINT):
            providers.append(("IPC", AsyncIPCProvider(self.configuration.IPC_ENDPOINT)))
        if self.configuration.HTTP_ENDPOINT:
            providers.append(("HTTP", AsyncHTTPProvider(self.configuration.HTTP_ENDPOINT)))
        if self.configuration.WEBSOCKET_ENDPOINT:
            providers.append(("WebSocket", WebSocketProvider(self.configuration.WEBSOCKET_ENDPOINT)))
        return providers

    async def _test_connection(self, web3: AsyncWeb3, name: str) -> bool:
        for attempt in range(3):
            try:
                if await web3.is_connected():
                    chain_id = await web3.eth.chain_id
                    return True
            except Exception:
                await asyncio.sleep(1)
        return False

    async def _add_middleware(self, web3: AsyncWeb3) -> None:
        try:
            chain_id = await web3.eth.chain_id
            if chain_id in {99, 100, 77, 7766, 56}:
                web3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
            elif chain_id in {1}:
                web3.middleware_onion.add(SignAndSendRawMiddlewareBuilder.build(self.account))
            elif chain_id in {61}:
                web3.middleware_onion.add(SignAndSendRawMiddlewareBuilder.build(self.account))
            elif chain_id in {56, 97, 42, 80001}:
                web3.middleware_onion.add(SignAndSendRawMiddlewareBuilder.build(self.account))
            else:
                pass
        except Exception:
            raise

    async def _check_account_balance(self) -> None:
        try:
            if not self.account:
                raise ValueError("Account not initialized")

            balance = await self.web3.eth.get_balance(self.account.address)
            balance_eth = self.web3.from_wei(balance, 'ether')

            if balance_eth < 0.001:
                pass
        except Exception:
            raise

    async def _initialize_components(self) -> None:
        try:
            self.apiconfig = APIConfig(self.configuration)

            self.noncecore = NonceCore(
                self.web3, self.account.address, self.configuration
            )
            await self.noncecore.initialize()

            self.safetynet = SafetyNet(
                self.web3, self.configuration, self.account, self.apiconfig
            )

            erc20_abi = await self._load_abi(self.configuration.ERC20_ABI)
            aave_flashloan_abi = await self._load_abi(self.configuration.AAVE_FLASHLOAN_ABI)
            aave_lending_pool_abi = await self._load_abi(self.configuration.AAVE_LENDING_POOL_ABI)

            self.marketmonitor = MarketMonitor(
                self.web3, self.configuration, self.apiconfig
            )

            tokens_to_monitor = await self.configuration.get_token_addresses()
            for token_address in tokens_to_monitor:
                token_symbol = await self.apiconfig.get_token_symbol(self.web3, token_address)
                if token_symbol:
                    await self.marketmonitor.start_periodic_training(token_symbol)

            self.mempoolmonitor = MempoolMonitor(
                web3=self.web3,
                safetynet=self.safetynet,
                noncecore=self.noncecore,
                apiconfig=self.apiconfig,
                monitored_tokens=tokens_to_monitor,
                erc20_abi=erc20_abi,
                configuration=self.configuration
            )

            self.transactioncore = TransactionCore(
                web3=self.web3,
                account=self.account,
                aave_flashloan_address=self.configuration.AAVE_FLASHLOAN_ADDRESS,
                aave_flashloan_abi=aave_flashloan_abi,
                aave_lending_pool_address=self.configuration.AAVE_LENDING_POOL_ADDRESS,
                aave_lending_pool_abi=aave_lending_pool_abi,
                monitor=self.mempoolmonitor,
                noncecore=self.noncecore,
                safetynet=self.safetynet,
                apiconfig=self.apiconfig,
                configuration=self.configuration,
                erc20_abi=erc20_abi
            )
            await self.transactioncore.initialize()

            self.strategynet = StrategyNet(
                transactioncore=self.transactioncore,
                marketmonitor=self.marketmonitor,
                safetynet=self.safetynet,
                apiconfig=self.apiconfig,
                configuration=self.configuration,
            )
        except Exception:
            raise

    async def run(self) -> None:
        try:
            if not all([
                self.configuration,
                self.noncecore,
                self.apiconfig,
                self.safetynet,
                self.mempoolmonitor,
                self.transactioncore,
                self.strategynet,
                self.marketmonitor,
            ]):
                raise RuntimeError("Required components are not properly initialized")

            await self.mempoolmonitor.start_monitoring()

            while True:
                try:
                    await self._process_profitable_transactions()
                    await asyncio.sleep(1)
                except asyncio.CancelledError:
                    break
                except Exception:
                    await asyncio.sleep(5)

        except KeyboardInterrupt:
            pass
        except Exception:
            pass
        finally:
            await self.stop()

    async def stop(self) -> None:
        try:
            if self.mempoolmonitor:
                await self.mempoolmonitor.stop_monitoring()

            if self.transactioncore:
                await self.transactioncore.stop()

            if self.noncecore:
                await self.noncecore.stop()

            if self.safetynet:
                await self.safetynet.stop()

            if self.marketmonitor:
                pass

            if self.apiconfig:
                await self.apiconfig.close()

            if self.web3:
                await self.web3.provider.disconnect()

            event_loop = asyncio.get_event_loop()
            event_loop.stop() or event_loop.close()

            if exception := sys.exc_info():
                pass

        except Exception:
            pass
        finally:
            sys.exit(0)

    async def _process_profitable_transactions(self) -> None:
        monitor = self.mempoolmonitor
        strategy = self.strategynet

        while not monitor.profitable_transactions.empty():
            start_time = time.time()
            tx = None

            try:
                tx = await asyncio.wait_for(
                    monitor.profitable_transactions.get(),
                    timeout=5.0
                )

                if not self._validate_transaction(tx):
                    continue

                tx_hash = tx.get('tx_hash', 'Unknown')[:10]
                strategy_type = self._determine_strategy_type(tx)

                if not await self._is_tx_still_valid(tx):
                    continue

                success = await asyncio.wait_for(
                    strategy.execute_best_strategy(tx, strategy_type),
                    timeout=30.0
                )

                execution_time = time.time() - start_time
                self._log_execution_metrics(tx_hash, success, execution_time)

                if success:
                    pass
                else:
                    pass

            except asyncio.TimeoutError:
                pass

            except Exception:
                tx_hash = tx.get('tx_hash', 'Unknown')[:10] if tx else 'Unknown'

            finally:
                if tx:
                    monitor.profitable_transactions.task_done()

    def _validate_transaction(self, tx: Dict[str, Any]) -> bool:
        required_fields = ['tx_hash', 'value', 'gasPrice', 'to']
        is_valid = (
            isinstance(tx, dict)
            and all(field in tx for field in required_fields)
            and all(tx[field] is not None for field in required_fields)
        )
        return is_valid

    def _determine_strategy_type(self, tx: Dict[str, Any]) -> str:
        if tx.get('value', 0) > 0:
            return 'eth_transaction'
        elif self._is_token_swap(tx):
            if tx.get('gasPrice', 0) > self.web3.to_wei(200, 'gwei'):
                return 'back_run'
            else:
                return 'front_run'
        return 'sandwich_attack'

    async def _is_tx_still_valid(self, tx: Dict[str, Any]) -> bool:
        try:
            tx_hash = tx.get('tx_hash')
            if not tx_hash:
                return False
            tx_status = await self.web3.eth.get_transaction(tx_hash)
            return tx_status is not None and tx_status.block_number is None
        except Exception:
            return False

    def _is_token_swap(self, tx: Dict[str, Any]) -> bool:
        return (
            len(tx.get('input', '0x')) > 10
            and tx.get('value', 0) == 0
        )

    def _log_execution_metrics(self, tx_hash: str, success: bool, execution_time: float) -> None:
        try:
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
        except ImportError:
            memory_usage = 0.0

    async def _load_abi(self, abi_path: str) -> List[Dict[str, Any]]:
        try:
            with open(abi_path, 'r') as file:
                abi = json.load(file)
            return abi
        except Exception:
            raise

async def main():
    try:
        configuration = Configuration()
        await configuration.load()

        main_core = MainCore(configuration)
        await main_core.initialize()
        await main_core.run()
    except KeyboardInterrupt:
        pass
    except Exception:
        sys.exit(1)

def run_standard():
    asyncio.run(main())

if __name__ == "__main__":
    run_standard()

    
