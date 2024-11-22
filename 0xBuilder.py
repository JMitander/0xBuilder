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
import traceback
import joblib 
import pandas as pd
import psutil
import queue
import threading
import matplotlib as plt
import streamlit as st
from cachetools import TTLCache
from sklearn.linear_model import LinearRegression
from decimal import Decimal
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass

from web3 import AsyncWeb3
from web3.exceptions import TransactionNotFound, ContractLogicError
from eth_account import Account
from web3.providers import AsyncIPCProvider, AsyncHTTPProvider, WebSocketProvider
from web3.middleware import ExtraDataToPOAMiddleware, SignAndSendRawMiddlewareBuilder
from web3.eth import AsyncEth



#========================== Logging and console output ==========================


class StreamlitHandler(logging.Handler):
    """
    Custom logging handler that sends logs to a Streamlit text area via a queue.
    """
    def __init__(self, log_queue: 'queue.Queue'):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        msg = self.format(record)
        self.log_queue.put(msg)

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
    The Configuration class is responsible for loading and managing all
    configuration parameters required by the MEV bot. It sources environment
    variables, loads JSON files for monitored tokens and contract ABIs,
    and sets up paths for machine learning models and training data.
    """

    def __init__(self):
        """
        Initializes the Configuration instance with default attributes.
        Attributes are populated during the load process.
        """
        
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

    async def load(self) -> None:
        """
        Public method to initiate the loading of all configuration parameters.
        It sequentially loads API keys, provider details, ML models, and JSON elements.
        """
        await self._load_configuration()

    async def _load_configuration(self) -> None:
        """
        Private method that orchestrates the loading of different configuration aspects.
        It handles exceptions and ensures that the configuration is loaded successfully.
        """
        try:
            # Display a loading bar while loading environment variables
            await loading_bar("Loading Environment Variables", 2)
            self._load_api_keys()  # Load API keys from environment variables
            self._load_providers_and_account()  # Load provider endpoints and account details
            self._load_ML_models()  # Load machine learning model paths
            await self._load_json_elements()  # Load monitored tokens and contract ABIs from JSON
            logger.info("Configuration loaded successfully.")
        except Exception as e:
            # Raise a runtime error if any part of the configuration fails to load
            raise RuntimeError(f"Failed to load configuration: {e}") from e

    def _load_ML_models(self) -> None:
        """
        Loads the file paths for machine learning models and training data.
        These paths are used later for model training and prediction tasks.
        """
        self.ML_MODEL_PATH = "models/price_model.joblib"  # Path to the trained price prediction model
        self.ML_TRAINING_DATA_PATH = "data/training_data.csv"  # Path to the training data CSV file

    def _load_api_keys(self) -> None:
        """
        Loads API keys from environment variables required for interacting with external services.
        Raises an error if any key is missing.
        """
        self.ETHERSCAN_API_KEY = self._get_env_variable("ETHERSCAN_API_KEY")
        self.INFURA_PROJECT_ID = self._get_env_variable("INFURA_PROJECT_ID")
        self.COINGECKO_API_KEY = self._get_env_variable("COINGECKO_API_KEY")
        self.COINMARKETCAP_API_KEY = self._get_env_variable("COINMARKETCAP_API_KEY")
        self.CRYPTOCOMPARE_API_KEY = self._get_env_variable("CRYPTOCOMPARE_API_KEY")

    def _load_providers_and_account(self) -> None:
        """
        Loads blockchain provider endpoints and account details from environment variables.
        Ensures that the wallet key and address are correctly retrieved.
        """
        self.HTTP_ENDPOINT = self._get_env_variable("HTTP_ENDPOINT")
        self.IPC_ENDPOINT = self._get_env_variable("IPC_ENDPOINT")
        self.WEBSOCKET_ENDPOINT = self._get_env_variable("WEBSOCKET_ENDPOINT")
        self.WALLET_KEY = self._get_env_variable("WALLET_KEY")
        self.WALLET_ADDRESS = self._get_env_variable("WALLET_ADDRESS")

    async def _load_json_elements(self) -> None:
        """
        Asynchronously loads various JSON-based configuration elements such as token addresses,
        token symbols, and contract ABIs. It constructs paths to ABI files and loads
        function signatures from JSON files.

        Did you know?
        ERC20 stands for "Ethereum Request for Comments 20" and is a standard interface for
        fungible tokens on the Ethereum blockchain. ERC20 tokens are used for a wide range of
        applications, including decentralized finance (DeFi), gaming, and non-fungible tokens (NFTs).

        """
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
        """
        Retrieves the value of an environment variable. Raises an error if the variable
        is not set and no default is provided.

        Args:
            var_name (str): The name of the environment variable.
            default (Optional[str], optional): The default value if the variable is not set. Defaults to None.

        Returns:
            str: The value of the environment variable.

        Raises:
            EnvironmentError: If the environment variable is not set and no default is provided.
        """
        value = os.getenv(var_name, default)
        if value is None:
            raise EnvironmentError(f"Missing environment variable: {var_name}")
        return value

    async def _load_json_file(self, file_path: str, description: str) -> Any:
        """
        Asynchronously loads and parses a JSON file.

        Args:
            file_path (str): The path to the JSON file.
            description (str): A description of the JSON content for logging purposes.

        Returns:
            Any: The parsed JSON content.

        Raises:
            FileNotFoundError: If the JSON file does not exist.
            json.JSONDecodeError: If the JSON file contains invalid JSON.
            Exception: For any other errors during file loading.
        """
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
        """
        Constructs the full path to an ABI file and verifies its existence.

        Args:
            base_path (str): The base directory where ABI files are stored.
            abi_filename (str): The filename of the ABI.

        Returns:
            str: The full path to the ABI file.

        Raises:
            FileNotFoundError: If the ABI file does not exist at the constructed path.
        """
        abi_path = os.path.join(base_path, abi_filename)
        if not os.path.exists(abi_path):
            logger.error(f"abi file not found at path: {abi_path}")
            raise FileNotFoundError(f"abi file '{abi_filename}' not found in path '{base_path}'")
        return abi_path

    async def get_token_addresses(self) -> List[str]:
        """
        Retrieves the list of monitored token addresses.

        Returns:
            List[str]: A list of token addresses.
        """
        return self.TOKEN_ADDRESSES

    async def get_token_symbols(self) -> Dict[str, str]:
        """
        Retrieves the mapping of token addresses to their symbols.

        Did you know?
        A token symbol is a unique identifier that represents a specific token on a blockchain. 
        for example SHIB would be the symbol for Shiba Inu and WETH for Wrapped Ethereum.
        
        Returns:
            Dict[str, str]: A dictionary mapping token addresses to symbols.
        """
        return self.TOKEN_SYMBOLS

    def get_abi_path(self, abi_name: str) -> str:
        """
        Retrieves the path to a specific ABI based on its name.

        Args:
            abi_name (str): The name identifier of the ABI.

        Returns:
            str: The path to the requested ABI. Returns an empty string if not found.
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
        return abi_paths.get(abi_name.lower(), "")

        
#========================== Nonce management system for Ethereum transactions ==========================

class NonceCore:
    """
    The NonceCore class manages Ethereum transaction nonces with advanced features such as
    caching, auto-recovery, and comprehensive error handling. It ensures that transactions
    are sent with the correct nonce to prevent conflicts and double-spending.

    Did you know?
    In the context of Ethereum transactions, a nonce is a unique identifier assigned to each
    transaction to prevent replay attacks and ensure that transactions are processed in the
    correct order. Nonces are used to track the number of transactions sent from an address
    and are incremented with each new transaction. By managing nonces effectively, the bot
    can ensure that transactions are executed reliably and without conflicts.
    """

    MAX_RETRIES = 3  # Maximum number of retry attempts for fetching nonces
    RETRY_DELAY = 1.0  # Base delay in seconds between retries
    CACHE_TTL = 300  # Time-to-live for nonce cache in seconds

    def __init__(
        self,
        web3: AsyncWeb3,
        address: str,
        configuration: Configuration,
    ):
        """
        Initializes the NonceCore instance.

        Args:
            web3 (AsyncWeb3): The AsyncWeb3 instance for interacting with the Ethereum network.
            address (str): The Ethereum address for which to manage nonces.
            configuration (Configuration): The configuration instance containing settings.
        """
        self.pending_transactions = set()  # Set of nonces for pending transactions
        self.web3 = web3  # Web3 instance for blockchain interactions
        self.configuration = configuration  # Configuration settings
        self.address = address  # Ethereum address
        self.lock = asyncio.Lock()  # Asyncio lock to manage concurrency
        self.nonce_cache = TTLCache(maxsize=1, ttl=self.CACHE_TTL)  # Cache for the latest nonce
        self.last_sync = time.monotonic()  # Timestamp of the last nonce synchronization
        self._initialized = False  # Flag indicating if the nonce manager is initialized

    async def initialize(self) -> None:
        """
        Initializes the nonce manager with error recovery mechanisms.
        It fetches the current nonce from the blockchain and sets up the cache.
        """
        try:
            async with self.lock:
                if not self._initialized:
                    await self._init_nonce()
                    self._initialized = True
                    logger.debug(f"NonceCore initialized for {self.address[:10]}...")
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise RuntimeError("NonceCore initialization failed") from e

    async def _init_nonce(self) -> None:
        """
        Private method to initialize the nonce by fetching the current nonce from the blockchain
        and considering any pending transactions.
        """
        current_nonce = await self._fetch_current_nonce_with_retries()
        pending_nonce = await self._get_pending_nonce()
        # Use the highest nonce among current and pending to avoid conflicts
        self.nonce_cache[self.address] = max(current_nonce, pending_nonce)
        self.last_sync = time.monotonic()

    async def get_nonce(self, force_refresh: bool = False) -> int:
        """
        Retrieves the next available nonce. Optionally forces a refresh of the nonce cache.

        Args:
            force_refresh (bool, optional): If True, refreshes the nonce from the blockchain. Defaults to False.

        Returns:
            int: The next available nonce.

        Raises:
            KeyError: If the nonce cache does not contain the address.
            Exception: For any other errors during nonce retrieval.
        """
        if not self._initialized:
            await self.initialize()
        async with self.lock:
            try:
                if force_refresh or self._should_refresh_cache():
                    await self.refresh_nonce()
                current_nonce = self.nonce_cache.get(self.address, 0)
                next_nonce = current_nonce
                self.nonce_cache[self.address] = current_nonce + 1  # Increment nonce for the next transaction
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
        """
        Refreshes the nonce by fetching the current nonce from the blockchain and
        resolving any conflicts with pending transactions.
        """
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
        """
        Fetches the current nonce from the blockchain with retry logic and exponential backoff.

        Returns:
            int: The current nonce.

        Raises:
            Exception: If all retry attempts fail.
        """
        backoff = self.RETRY_DELAY
        for attempt in range(self.MAX_RETRIES):
            try:
                # Fetch the nonce including pending transactions to get the latest
                return await self.web3.eth.get_transaction_count(
                    self.address, block_identifier="pending"
                )
            except Exception as e:
                if attempt == self.MAX_RETRIES - 1:
                    logger.error(f"Nonce fetch failed after retries: {e}")
                    raise
                logger.warning(f"Nonce fetch attempt {attempt + 1} failed: {e}. Retrying in {backoff}s...")
                await asyncio.sleep(backoff)
                backoff *= 2  # Exponential backoff

    async def _get_pending_nonce(self) -> int:
        """
        Retrieves the highest nonce from the set of pending transactions.

        Returns:
            int: The next available pending nonce.
        """
        try:
            pending_nonces = [int(nonce) for nonce in self.pending_transactions]
            return max(pending_nonces) + 1 if pending_nonces else 0
        except Exception as e:
            logger.error(f"Error getting pending nonce: {e}")
            return 0

    async def track_transaction(self, tx_hash: str, nonce: int) -> None:
        """
        Tracks a pending transaction by adding its nonce to the set of pending transactions.
        It waits for the transaction to be confirmed and then removes it from tracking.

        Args:
            tx_hash (str): The hash of the transaction to track.
            nonce (int): The nonce associated with the transaction.
        """
        self.pending_transactions.add(nonce)
        try:
            # Wait for the transaction to be mined with a timeout
            await self.web3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            self.pending_transactions.discard(nonce)
        except Exception as e:
            logger.error(f"Transaction tracking failed: {e}")
        finally:
            self.pending_transactions.discard(nonce)

    async def _handle_nonce_error(self) -> None:
        """
        Handles nonce-related errors by attempting to synchronize the nonce with the blockchain.
        """
        try:
            await self.sync_nonce_with_chain()
        except Exception as e:
            logger.error(f"Nonce error recovery failed: {e}")
            raise

    async def sync_nonce_with_chain(self) -> None:
        """
        Forces synchronization of the nonce with the current state of the blockchain.
        It fetches the latest nonce and clears any pending transactions.
        """
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
        """
        Determines whether the nonce cache should be refreshed based on the elapsed time
        since the last synchronization.

        Returns:
            bool: True if the cache should be refreshed, else False.
        """
        return time.monotonic() - self.last_sync > (self.CACHE_TTL / 2)

    async def reset(self) -> None:
        """
        Resets the nonce manager's state by clearing the cache and pending transactions,
        and re-initializing the nonce.
        """
        async with self.lock:
            try:
                self.nonce_cache.clear()  # Clear the nonce cache
                self.pending_transactions.clear()  # Clear pending transactions
                self.last_sync = time.monotonic()  # Reset synchronization timestamp
                self._initialized = False  # Mark as uninitialized
                await self.initialize()  # Re-initialize nonce manager
                logger.debug("Nonce Core reset complete")
            except Exception as e:
                logger.error(f"Reset failed: {e}")
                raise

    async def stop(self) -> None:
        """
        Gracefully stops the nonce manager by resetting its state.
        """
        try:
            await self.reset()
            logger.debug("Nonce Core stopped successfully.")
        except Exception as e:
            logger.error(f"Error stopping nonce core: {e}")
            raise


#================================= API setup and configuration =================================#

class APIConfig: 
    """
    The APIConfig class manages interactions with various cryptocurrency and market data APIs.
    It handles API requests, caching of responses, and implements rate limiting and failover mechanisms
    to ensure reliable data retrieval for price information, historical data, and token volumes.

    Did you know?
    API stands for Application Programming Interface, which is a set of rules and protocols
    that allows one software application to interact with another. APIs are used to facilitate
    communication between different systems, enabling data exchange and functionality integration
    across platforms. In the context of this bot, APIs are used to fetch real-time price data
    for tokens and cryptocurrencies from external sources.
    """

    def __init__(self, configuration: Optional[Configuration] = None):
        """
        Initializes the APIConfig instance with necessary configurations.

        Args:
            configuration (Optional[Configuration], optional): The configuration instance containing API keys and settings.
                Defaults to None.
        """
        self.apiconfig = {}  # Configuration settings for different API providers
        self.configuration = configuration  # Configuration settings
        self.session = None  # aiohttp ClientSession for making HTTP requests
        self.price_cache = TTLCache(maxsize=1000, ttl=300)  # Cache for real-time prices (5 minutes)
        self.token_symbol_cache = TTLCache(maxsize=1000, ttl=86400)  # Cache for token symbols (1 day)

    async def __aenter__(self):
        """
        Asynchronous context manager entry. Initializes the aiohttp session.
        """
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Asynchronous context manager exit. Closes the aiohttp session.
        """
        if self.session:
            await self.session.close()

        # API configuration detailing different API providers and their settings
        self.apiconfig = {
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

        # Initialize rate limiters for each API provider based on their rate limits
        self.api_lock = asyncio.Lock()  # Lock to manage concurrent API access
        self.rate_limiters = {
            provider: asyncio.Semaphore(config.get("rate_limit", 10))
            for provider, config in self.apiconfig.items()
        }

    async def get_token_symbol(self, web3: AsyncWeb3, token_address: str) -> Optional[str]:
        """
        Retrieves the symbol of a token given its blockchain address. Utilizes caching to minimize redundant API calls.

        Args:
            web3 (AsyncWeb3): The Web3 instance for interacting with the blockchain.
            token_address (str): The blockchain address of the token.

        Returns:
            Optional[str]: The token symbol if found, else None.
        """
        if token_address in self.token_symbol_cache:
            return self.token_symbol_cache[token_address]
        if token_address in self.configuration.TOKEN_SYMBOLS:
            symbol = self.configuration.TOKEN_SYMBOLS[token_address]
            self.token_symbol_cache[token_address] = symbol
            return symbol
        try:
            erc20_abi = await self._load_abi(self.configuration.ERC20_ABI)  # Load ERC20 ABI
            contract = web3.eth.contract(address=token_address, abi=erc20_abi)
            symbol = await contract.functions.symbol().call()  # Call the symbol function
            self.token_symbol_cache[token_address] = symbol
            return symbol
        except Exception as e:
            logger.error(f"Error getting symbol for token {token_address}: {e}")
            return None

    async def get_real_time_price(self, token: str, vs_currency: str = "eth") -> Optional[Decimal]:
        """
        Retrieves the real-time price of a token against a specified currency using a weighted average
        from multiple API sources. Implements caching to reduce API load and improve performance.

        Args:
            token (str): The symbol of the token (e.g., 'btc', 'eth').
            vs_currency (str, optional): The currency to compare against (default is 'eth'). Defaults to "eth".

        Returns:
            Optional[Decimal]: The real-time price if successful, else None.
        """
        cache_key = f"price_{token}_{vs_currency}"
        if cache_key in self.price_cache:
            return self.price_cache[cache_key]
        prices = []
        weights = []
        async with self.api_lock:
            for source, config in self.apiconfig.items():
                try:
                    price = await self._fetch_price(source, token, vs_currency)
                    if price:
                        prices.append(price)
                        weights.append(config["weight"] * config["success_rate"])
                except Exception as e:
                    logger.error(f"Error fetching price from {source}: {e}")
                    config["success_rate"] *= 0.9  # Penalize failed source
        if not prices:
            logger.warning(f"No valid prices found for {token}!")
            return None
        # Calculate weighted average price
        weighted_price = sum(p * w for p, w in zip(prices, weights)) / sum(weights)
        self.price_cache[cache_key] = Decimal(str(weighted_price))
        return self.price_cache[cache_key]

    async def _fetch_price(self, source: str, token: str, vs_currency: str) -> Optional[Decimal]:
        """
        Fetches the price of a token from a specific API source.

        Args:
            source (str): The name of the API source (e.g., 'binance', 'coingecko').
            token (str): The symbol of the token.
            vs_currency (str): The currency to compare against.

        Returns:
            Optional[Decimal]: The price if fetched successfully, else None.
        """
        config = self.apiconfig.get(source)
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
        """
        Makes an HTTP GET request to a specified URL with parameters and headers.
        Implements rate limiting, retries with exponential backoff, and error handling.

        Args:
            provider_name (str): The name of the API provider for logging and rate limiting.
            url (str): The full URL to send the GET request to.
            params (Optional[Dict[str, Any]], optional): Query parameters for the request. Defaults to None.
            headers (Optional[Dict[str, str]], optional): HTTP headers for the request. Defaults to None.
            max_attempts (int, optional): Maximum number of retry attempts. Defaults to 5.
            backoff_factor (float, optional): Factor by which the wait time increases after each retry. Defaults to 1.5.

        Returns:
            Any: The JSON-decoded response content.

        Raises:
            aiohttp.ClientResponseError: If the HTTP response contains an error status.
            Exception: For any other unexpected errors.
        """
        rate_limiter = self.rate_limiters.get(provider_name)

        if rate_limiter is None:
            # Initialize a default rate limiter if not configured
            rate_limiter = asyncio.Semaphore(10)
            self.rate_limiters[provider_name] = rate_limiter

        async with rate_limiter:
            for attempt in range(max_attempts):
                try:
                    timeout = aiohttp.ClientTimeout(total=10 * (attempt + 1))  # Dynamic timeout based on attempt
                    async with self.session.get(url, params=params, headers=headers, timeout=timeout) as response:
                        if response.status == 429:
                            # Handle rate limit exceeded
                            wait_time = backoff_factor ** attempt
                            logger.warning(f"Rate limit exceeded for {provider_name}, retrying in {wait_time}s...")
                            await asyncio.sleep(wait_time)
                            continue
                        response.raise_for_status()  # Raise an error for bad status codes
                        return await response.json()  # Return the JSON-decoded response
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
        """
        Fetches historical price data for a given token symbol over a specified number of days.
        Utilizes caching to avoid redundant API calls.

        Args:
            token (str): The symbol of the token (e.g., 'btc', 'eth').
            days (int, optional): The number of days of historical data to fetch. Defaults to 30.

        Returns:
            List[float]: A list of historical prices.
        """
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
        """
        Fetches historical prices from a specified API source.

        Args:
            source (str): The name of the API source.
            token (str): The symbol of the token.
            days (int): Number of days of historical data to fetch.

        Returns:
            Optional[List[float]]: A list of historical prices if successful, else None.
        """
        config = self.apiconfig.get(source)
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
        """
        Retrieves the 24-hour trading volume for a given token symbol.
        Utilizes caching to minimize API calls.

        Args:
            token (str): The symbol of the token.

        Returns:
            float: The 24-hour trading volume in USD.
        """
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
        """
        Fetches the trading volume for a token from a specified API source.

        Args:
            source (str): The name of the API source.
            token (str): The symbol of the token.

        Returns:
            Optional[float]: The trading volume if fetched successfully, else None.
        """
        config = self.apiconfig.get(source)
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
        """
        Helper method to fetch data from multiple API services.

        Args:
            fetch_func (Callable): A lambda function that takes a service name and fetches data.
            description (str): Description of the data being fetched for logging.

        Returns:
            Any: The fetched data if successful, else None.
        """
        for service in self.apiconfig.keys():
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
        """
        Asynchronously loads a contract's ABI from a specified file path.

        Args:
            abi_path (str): The file path to the ABI JSON file.

        Returns:
            List[Dict[str, Any]]: The loaded ABI.

        Raises:
            Exception: If loading the ABI fails.
        """
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
        """
        Closes the aiohttp session to free up resources.
        """
        await self.session.close()


#=========================== 0xBuilder's Safety controller and risk management Engine ===========================#

#=========================== 0xBuilder's Safety controller and risk management Engine ===========================#

class SafetyNet:
    """
    The SafetyNet class provides robust risk management and price verification functionalities.
    It leverages multiple data sources, implements automatic failover mechanisms, and dynamically
    adjusts trading parameters based on real-time network and market conditions. This ensures that
    all transactions executed by the MEV bot are secure, profitable, and compliant with predefined
    risk thresholds.
    """

    CACHE_TTL = 300  # Time-to-live for general caches in seconds
    GAS_PRICE_CACHE_TTL = 15  # Time-to-live for gas price cache in seconds

    SLIPPAGE_CONFIG = {
        "default": 0.1,  # Default slippage tolerance (10%)
        "min": 0.01,  # Minimum slippage tolerance (1%)
        "max": 0.5,  # Maximum slippage tolerance (50%)
        "high_congestion": 0.05,  # Slippage during high network congestion (5%)
        "low_congestion": 0.2,  # Slippage during low network congestion (20%)
    }

    GAS_CONFIG = {
        "max_gas_price_gwei": 500,  # Maximum allowable gas price in Gwei
        "min_profit_multiplier": 2.0,  # Minimum profit multiplier to consider a transaction
        "base_gas_limit": 21000,  # Base gas limit for standard transactions
    }

    def __init__(
        self,
        web3: AsyncWeb3,
        configuration: Optional[Configuration] = None,
        address: Optional[str] = None,
        account: Optional[Account] = None,
        apiconfig: Optional[APIConfig] = None,
    ):
        """
        Initializes the SafetyNet instance with necessary components and configurations.

        Args:
            web3 (AsyncWeb3): The Web3 instance for blockchain interactions.
            configuration (Optional[Configuration], optional): Configuration settings. Defaults to None.
            address (Optional[str], optional): Ethereum address to monitor. Defaults to None.
            account (Optional[Account], optional): Ethereum account instance. Defaults to None.
            apiconfig (Optional[APIConfig], optional): API configuration for data retrieval. Defaults to None.
        """
        self.web3 = web3  # Web3 instance for blockchain interactions
        self.address = address  # Ethereum address
        self.configuration = configuration  # Configuration settings
        self.account = account  # Ethereum account instance
        self.apiconfig = apiconfig  # API configuration for data retrieval
        self.price_cache = TTLCache(maxsize=1000, ttl=self.CACHE_TTL)  # Cache for various price data
        self.gas_price_cache = TTLCache(maxsize=1, ttl=self.GAS_PRICE_CACHE_TTL)  # Cache for gas prices

        self.price_lock = asyncio.Lock()  # Lock to manage concurrent access to price data
        logger.debug("Safety Net initialized with enhanced configuration")

    async def get_balance(self, account: Any) -> Decimal:
        """
        Retrieves the ETH balance of the specified account with retries and caching.

        Args:
            account (Any): The Ethereum account instance.

        Returns:
            Decimal: The ETH balance of the account.
        """
        cache_key = f"balance_{account.address}"
        if cache_key in self.price_cache:
            return self.price_cache[cache_key]

        for attempt in range(3):
            try:
                balance_wei = await self.web3.eth.get_balance(account.address)
                balance_eth = Decimal(self.web3.from_wei(balance_wei, "ether"))
                self.price_cache[cache_key] = balance_eth  # Cache the balance
                logger.debug(f"Balance for {account.address[:10]}...: {balance_eth:.4f} ETH")
                return balance_eth
            except Exception as e:
                if attempt == 2:
                    logger.error(f"Failed to get balance after 3 attempts: {e}")
                    return Decimal(0)
                await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff

    async def ensure_profit(
        self,
        transaction_data: Dict[str, Any],
        minimum_profit_eth: Optional[float] = None,
    ) -> bool:
        """
        Ensures that a transaction will yield a minimum profit before execution.
        It considers dynamic thresholds and performs risk assessments.

        Did you know?
        Gas fees are the cost of doing business on the Ethereum network. They are paid to miners for processing transactions and executing smart contracts. 
        Gas fees are calculated based on the computational resources required to execute a transaction or contract function. 
        The higher the gas price, the faster the transaction will be processed. (most of the time)

        Args:
            transaction_data (Dict[str, Any]): The transaction data dictionary.
            minimum_profit_eth (Optional[float], optional): The minimum profit threshold in ETH. Defaults to None.

        Returns:
            bool: True if the transaction is expected to be profitable, else False.
        """
        try:
            if minimum_profit_eth is None:
                account_balance = await self.get_balance(self.account)
                # Set minimum profit based on account balance to ensure sustainability
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
        """
        Validates gas parameters against predefined safety thresholds.

        Args:
            gas_price_gwei (Decimal): The gas price in Gwei.
            gas_used (int): The estimated gas used by the transaction.

        Returns:
            bool: True if gas parameters are within acceptable limits, else False.
        """
        if gas_used == 0:
            logger.error("Gas estimation returned zero")
            return False
        if gas_price_gwei > self.GAS_CONFIG["max_gas_price_gwei"]:
            logger.warning(f"Gas price {gas_price_gwei} gwei exceeds maximum threshold")
            return False
        return True

    def _calculate_gas_cost(self, gas_price_gwei: Decimal, gas_used: int) -> Decimal:
        """
        Calculates the total gas cost in ETH based on gas price and gas used.

        Args:
            gas_price_gwei (Decimal): The gas price in Gwei.
            gas_used (int): The estimated gas used by the transaction.

        Returns:
            Decimal: The total gas cost in ETH.
        """
        return gas_price_gwei * Decimal(gas_used) * Decimal("1e-9")  # Convert Gwei to ETH

    async def _calculate_profit(
        self,
        transaction_data: Dict[str, Any],
        real_time_price: Decimal,
        slippage: float,
        gas_cost_eth: Decimal,
    ) -> Decimal:
        """
        Calculates the expected profit of a transaction considering slippage and gas costs.

        Args:
            transaction_data (Dict[str, Any]): The transaction data dictionary.
            real_time_price (Decimal): The real-time price of the output token.
            slippage (float): The slippage tolerance.
            gas_cost_eth (Decimal): The estimated gas cost in ETH.

        Returns:
            Decimal: The calculated profit in ETH.
        """
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
        """
        Logs a detailed summary of the profit calculation for auditing and debugging purposes.

        Args:
            transaction_data (Dict[str, Any]): The transaction data dictionary.
            real_time_price (Decimal): The real-time price of the output token.
            gas_cost_eth (Decimal): The estimated gas cost in ETH.
            profit (Decimal): The calculated profit in ETH.
            minimum_profit_eth (float): The minimum required profit threshold in ETH.
        """
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
        """
        Retrieves the current gas price dynamically, applying a multiplier to adjust for network conditions.

        Returns:
            Decimal: The adjusted gas price in Gwei.
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
            logger.error(f"Error fetching dynamic gas price: {e}")
            return Decimal(0)

    async def estimate_gas(self, transaction_data: Dict[str, Any]) -> int:
        """
        Estimates the gas required for a given transaction.

        Args:
            transaction_data (Dict[str, Any]): The transaction data dictionary.

        Returns:
            int: The estimated gas required. Returns 0 if estimation fails.
        """
        try:
            gas_estimate = await self.web3.eth.estimate_gas(transaction_data)
            return gas_estimate
        except Exception as e:
            logger.error(f"Gas estimation failed: {e}")
            return 0

    async def adjust_slippage_tolerance(self) -> float:
        """
        Adjusts the slippage tolerance dynamically based on current network congestion.

        Returns:
            float: The adjusted slippage tolerance as a decimal (e.g., 0.1 for 10%).
        """
        try:
            congestion_level = await self.get_network_congestion()
            if congestion_level > 0.8:
                slippage = self.SLIPPAGE_CONFIG["high_congestion"]
            elif congestion_level < 0.2:
                slippage = self.SLIPPAGE_CONFIG["low_congestion"]
            else:
                slippage = self.SLIPPAGE_CONFIG["default"]
            # Ensure slippage is within defined min and max bounds
            slippage = min(
                max(slippage, self.SLIPPAGE_CONFIG["min"]), self.SLIPPAGE_CONFIG["max"]
            )
            logger.debug(f"Adjusted slippage tolerance to {slippage * 100}%")
            return slippage
        except Exception as e:
            logger.error(f"Error adjusting slippage tolerance: {e}")
            return self.SLIPPAGE_CONFIG["default"]

    async def get_network_congestion(self) -> float:
        """
        Estimates the current network congestion by analyzing the latest block's gas usage.

        Returns:
            float: The network congestion level as a decimal (e.g., 0.5 for 50%).
        """
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

    async def _calculate_profit(
        self,
        transaction_data: Dict[str, Any],
        real_time_price: Decimal,
        slippage: float,
        gas_cost_eth: Decimal,
    ) -> Decimal:
        """
        Calculates the expected profit of a transaction after accounting for slippage and gas costs.

        Args:
            transaction_data (Dict[str, Any]): The transaction data dictionary.
            real_time_price (Decimal): The real-time price of the output token.
            slippage (float): The slippage tolerance.
            gas_cost_eth (Decimal): The estimated gas cost in ETH.

        Returns:
            Decimal: The calculated profit in ETH.
        """
        expected_output = real_time_price * Decimal(transaction_data["amountOut"])
        input_amount = Decimal(transaction_data["amountIn"])
        slippage_adjusted_output = expected_output * (1 - Decimal(slippage))
        return slippage_adjusted_output - input_amount - gas_cost_eth

    async def stop(self) -> None:
        """
        Gracefully stops the SafetyNet by closing the API session and performing any necessary cleanup.
        """
        try:
            await self.apiconfig.close()
            logger.debug("Safety Net stopped successfully.")
        except Exception as e:
            logger.error(f"Error stopping safety net: {e}")
            raise


#=============================== Txpool/Mempool Monitoring ===============================

#============================= Txpool/Mempool Monitoring ===============================

class MempoolMonitor:
    """
    The MempoolMonitor class provides an advanced monitoring system for the Ethereum mempool.
    It identifies and analyzes profitable transactions in real-time, utilizing sophisticated
    profit estimation, caching, and parallel processing capabilities. This ensures that the MEV
    bot can swiftly detect and act upon lucrative opportunities, optimizing transaction
    execution strategies such as front-running, back-running, and sandwich attacks.
    """

    MAX_RETRIES = 3  # Maximum number of retry attempts for transaction fetching
    RETRY_DELAY = 1.0  # Base delay in seconds between retries
    BATCH_SIZE = 10  # Number of transactions to process in a single batch
    MAX_PARALLEL_TASKS = 50  # Maximum number of parallel tasks for processing transactions

    def __init__(
        self,
        web3: AsyncWeb3,
        safetynet: SafetyNet,
        noncecore: NonceCore,
        apiconfig: APIConfig,
        monitored_tokens: Optional[List[str]] = None,
        erc20_abi: List[Dict[str, Any]] = None,
        configuration: Optional[Configuration] = None,
    ):
        """
        Initializes the MempoolMonitor instance with necessary components and configurations.

        Args:
            web3 (AsyncWeb3): The Web3 instance for interacting with the Ethereum network.
            safetynet (SafetyNet): The SafetyNet instance for risk management and price verification.
            noncecore (NonceCore): The NonceCore instance for managing transaction nonces.
            apiconfig (APIConfig): The APIConfig instance for fetching market data.
            monitored_tokens (Optional[List[str]], optional): List of token symbols to monitor. Defaults to None.
            erc20_abi (List[Dict[str, Any]], optional): The ABI of the ERC20 token contract. Defaults to None.
            configuration (Optional[Configuration], optional): Configuration settings. Defaults to None.
        """
        # Core components required for monitoring and analysis
        self.web3 = web3
        self.configuration = configuration
        self.safetynet = safetynet
        self.noncecore = noncecore
        self.apiconfig = apiconfig

        # Monitoring state and data structures
        self.running = False  # Flag indicating whether monitoring is active
        self.pending_transactions = asyncio.Queue()  # Queue for incoming transaction hashes
        self.monitored_tokens = set(monitored_tokens or [])  # Set of tokens to monitor
        self.profitable_transactions = asyncio.Queue()  # Queue for identified profitable transactions
        self.processed_transactions = set()  # Set to keep track of processed transaction hashes

        # Configuration settings for transaction analysis
        self.erc20_abi = erc20_abi or []  # ERC20 contract ABI for decoding transactions
        self.minimum_profit_threshold = Decimal("0.001")  # Minimum profit in ETH to consider a transaction profitable
        self.max_parallel_tasks = self.MAX_PARALLEL_TASKS  # Concurrency limit for processing tasks
        self.retry_attempts = self.MAX_RETRIES  # Number of retry attempts for fetching transactions
        self.backoff_factor = 1.5  # Exponential backoff factor for retries

        # Concurrency control using semaphores to limit the number of parallel tasks
        self.semaphore = asyncio.Semaphore(self.max_parallel_tasks)
        self.task_queue = asyncio.Queue()  # Queue for tasks to be processed

        logger.debug("MempoolMonitor initialized with enhanced configuration.")

    async def start_monitoring(self) -> None:
        """
        Starts the mempool monitoring process. It initializes the monitoring flag and
        launches asynchronous tasks for running the monitoring loop and processing the task queue.

        Raises:
            Exception: If monitoring fails to start.
        """
        if self.running:
            logger.debug("Monitoring is already active.")
            return

        try:
            self.running = True
            # Launch the monitoring and processor tasks concurrently
            monitoring_task = asyncio.create_task(self._run_monitoring())
            processor_task = asyncio.create_task(self._process_task_queue())

            logger.info("Mempool monitoring started.")
            # Wait for both tasks to complete (which they won't until stopped)
            await asyncio.gather(monitoring_task, processor_task)

        except Exception as e:
            self.running = False
            logger.error(f"Failed to start monitoring: {e}")
            raise

    async def stop_monitoring(self) -> None:
        """
        Gracefully stops the mempool monitoring process. It sets the running flag to False
        and waits for any remaining tasks in the queue to be processed before shutting down.
        """
        if not self.running:
            return

        self.running = False
        try:
            # Wait for the task queue to be emptied
            while not self.task_queue.empty():
                await asyncio.sleep(0.1)
            logger.info("Mempool monitoring stopped gracefully.")
            sys.exit(0)  # Exit the application gracefully
        except Exception as e:
            logger.error(f"Error during monitoring shutdown: {e}")

    async def _run_monitoring(self) -> None:
        """
        The main monitoring loop that continuously fetches new pending transactions from the mempool.
        It sets up a pending transaction filter and processes incoming transaction hashes as they appear.
        Implements automatic recovery with retry logic in case of failures.
        """
        retry_count = 0

        while self.running:
            try:
                pending_filter = await self._setup_pending_filter()
                if not pending_filter:
                    continue

                while self.running:
                    # Fetch new pending transaction hashes
                    tx_hashes = await pending_filter.get_new_entries()
                    if tx_hashes:
                        await self._handle_new_transactions(tx_hashes)
                    await asyncio.sleep(0.1)  # Small delay to prevent tight loop

            except Exception as e:
                retry_count += 1
                wait_time = min(self.backoff_factor ** retry_count, 30)  # Cap wait time at 30 seconds
                logger.error(
                    f"Monitoring error (attempt {retry_count}): {e}"
                )
                await asyncio.sleep(wait_time)  # Wait before retrying

    async def _setup_pending_filter(self) -> Optional[Any]:
        """
        Sets up a filter to listen for new pending transactions in the mempool.

        Did you know?:
        A pending transaction is a transaction that has been broadcast to the network but has not
        yet been included in a block. These transactions are considered unconfirmed until they are
        included in a block and become part of the blockchain.
    
        Returns:
            Optional[Any]: The pending transaction filter object if successful, else None.
        """
        try:
            pending_filter = await self.web3.eth.filter("pending")  # Create a pending transaction filter
            logger.debug(
                f"Connected to network via {self.web3.provider.__class__.__name__}"
            )
            return pending_filter

        except Exception as e:
            logger.warning(f"Failed to setup pending filter: {e}")
            return None

    async def _handle_new_transactions(self, tx_hashes: List[str]) -> None:
        """
        Processes a batch of new transaction hashes by queuing them for further analysis.
    
        Args:
            tx_hashes (List[str]): A list of new pending transaction hashes.
        """
        async def process_batch(batch):
            # Concurrently queue each transaction hash in the batch
            await asyncio.gather(
                *(self._queue_transaction(tx_hash) for tx_hash in batch)
            )

        try:
            # Process transactions in defined batch sizes
            for i in range(0, len(tx_hashes), self.BATCH_SIZE):
                batch = tx_hashes[i: i + self.BATCH_SIZE]
                await process_batch(batch)

        except Exception as e:
            logger.error(f"Error processing transaction batch: {e}")

    async def _queue_transaction(self, tx_hash: str) -> None:
        """
        Adds a transaction hash to the processing queue if it hasn't been processed yet.
        This prevents duplicate processing of the same transaction.
    
        Args:
            tx_hash (str): The hash of the transaction to queue.
        """
        tx_hash_hex = tx_hash.hex() if isinstance(tx_hash, bytes) else tx_hash
        if tx_hash_hex not in self.processed_transactions:
            self.processed_transactions.add(tx_hash_hex)  # Mark as processed
            await self.task_queue.put(tx_hash_hex)  # Add to the processing queue

    async def _process_task_queue(self) -> None:
        """
        Continuously processes transaction hashes from the task queue. It ensures that
        the number of concurrent processing tasks does not exceed the defined limit.
        """
        while self.running:
            try:
                tx_hash = await self.task_queue.get()
                async with self.semaphore:  # Limit concurrency
                    await self.process_transaction(tx_hash)
                self.task_queue.task_done()  # Mark the task as done
            except asyncio.CancelledError:
                break  # Exit if the task is cancelled
            except Exception as e:
                logger.error(f"Task processing error: {e}")

    async def process_transaction(self, tx_hash: str) -> None:
        """
        Processes an individual transaction by fetching its details, analyzing its profitability,
        and handling it accordingly if it's identified as profitable.
    
        Args:
            tx_hash (str): The hash of the transaction to process.
        """
        try:
            tx = await self._get_transaction_with_retry(tx_hash)  # Fetch transaction details
            if not tx:
                return

            analysis = await self.analyze_transaction(tx)  # Analyze transaction profitability
            if analysis.get("is_profitable"):
                await self._handle_profitable_transaction(analysis)  # Handle profitable transaction

        except Exception as e:
            logger.debug(f"Error processing transaction {tx_hash}: {e}")

    async def _get_transaction_with_retry(self, tx_hash: str) -> Optional[Any]:
        """
        Attempts to fetch transaction details with retry logic and exponential backoff.

        Did you know?:
        Exponential backoff is a networking algorithm that uses feedback to multiplicatively
        decrease the rate of some process, in order to gradually find an acceptable rate.
    
        Args:
            tx_hash (str): The hash of the transaction to fetch.
    
        Returns:
            Optional[Any]: The transaction object if successful, else None.
        """
        for attempt in range(self.retry_attempts):
            try:
                return await self.web3.eth.get_transaction(tx_hash)
            except TransactionNotFound:
                if attempt == self.retry_attempts - 1:
                    return None
                await asyncio.sleep(self.backoff_factor ** attempt)  # Exponential backoff
            except Exception as e:
                logger.error(f"Error fetching transaction {tx_hash}: {e}")
                return None

    async def _handle_profitable_transaction(self, analysis: Dict[str, Any]) -> None:
        """
        Handles a profitable transaction by adding it to the profitable transactions queue
        and logging relevant details for further action.
    
        Args:
            analysis (Dict[str, Any]): The analysis result of the transaction.
        """
        try:
            await self.profitable_transactions.put(analysis)  # Queue for profitable transactions
            logger.debug(
                f"Profitable transaction identified: {analysis['tx_hash']} "
                f"(Estimated profit: {analysis.get('profit', 'Unknown')} ETH)"
            )
        except Exception as e:
            logger.debug(f"Error handling profitable transaction: {e}")

    async def analyze_transaction(self, tx) -> Dict[str, Any]:
        """
        Analyzes a transaction to determine if it is profitable based on certain criteria.
        Differentiates between ETH transfers and token transactions.
    
        Args:
            tx: The transaction object fetched from the blockchain.
    
        Returns:
            Dict[str, Any]: Analysis result indicating profitability and related details.
        """
        if not tx.hash or not tx.input:
            logger.debug(
                f"Transaction {tx.hash.hex()} is missing essential fields. Skipping."
            )
            return {"is_profitable": False}
        try:
            if tx.value > 0:
                return await self._analyze_eth_transaction(tx)  # Analyze ETH transfer
            return await self._analyze_token_transaction(tx)  # Analyze token transaction
        except Exception as e:
            logger.error(
                f"Error analyzing transaction {tx.hash.hex()}: {e}"
            )
            return {"is_profitable": False}

    async def _analyze_eth_transaction(self, tx) -> Dict[str, Any]:
        """
        Specifically analyzes an ETH transfer transaction for profitability.
    
        Args:
            tx: The ETH transfer transaction object.
    
        Returns:
            Dict[str, Any]: Analysis result indicating profitability and transaction details.
        """
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
        """
        Specifically analyzes a token transaction for profitability by decoding the transaction
        input and estimating potential profits.

        Did you know?:
        ERC20 tokens are a type of Ethereum token standard that defines certain rules and
        functions that all Ethereum tokens must follow. This includes how tokens are transferred,
        how users can access data about a token, and how users can access the total supply of tokens.
    
        Args:
            tx: The token transaction object.
    
        Returns:
            Dict[str, Any]: Analysis result indicating profitability and transaction details.
        """
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

    async def _is_profitable_eth_transaction(self, tx: Any) -> bool:
        """
        Determines if an ETH transfer transaction is profitable by estimating the potential profit.
    
        Args:
            tx (Any): The ETH transfer transaction object.
    
        Returns:
            bool: True if the transaction is expected to be profitable, else False.
        """
        try:
            potential_profit = await self._estimate_eth_transaction_profit(tx)
            return potential_profit > self.minimum_profit_threshold
        except Exception as e:
            logger.debug(
                f"Error estimating ETH transaction profit for transaction {tx.hash.hex()}: {e}"
            )
            return False

    async def _estimate_eth_transaction_profit(self, tx: Any) -> Decimal:
        """
        Estimates the profit from an ETH transfer transaction by considering the input ETH,
        expected output, and gas costs.
    
        Args:
            tx (Any): The ETH transfer transaction object.
    
        Returns:
            Decimal: The estimated profit in ETH.
        """
        try:
            gas_price_gwei = await self.safetynet.get_dynamic_gas_price()
            gas_used = tx.gas if tx.gas else await self.web3.eth.estimate_gas(tx)
            gas_cost_eth = Decimal(gas_price_gwei) * Decimal(gas_used) * Decimal("1e-9")
            eth_value = Decimal(self.web3.from_wei(tx.value, "ether"))
            potential_profit = eth_value - gas_cost_eth
            return potential_profit if potential_profit > 0 else Decimal(0)
        except Exception as e:
            logger.error(f"Error estimating ETH transaction profit: {e}")
            return Decimal(0)

    async def _estimate_profit(
        self,
        tx: Any,
        function_params: Dict[str, Any],
    ) -> Decimal:
        """
        Estimates the potential profit of a token transaction by analyzing the input parameters,
        market prices, slippage, and gas costs.
        
        Did you know?:
        Slippage is the difference between the expected price of a trade and the price at which
        the trade is executed.
    
        Args:
            tx (Any): The token transaction object.
            function_params (Dict[str, Any]): The decoded function parameters of the transaction.
    
        Returns:
            Decimal: The estimated profit in ETH.
        """
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
            output_token_symbol = await self.apiconfig.get_token_symbol(self.web3, output_token_address)
            if not output_token_symbol:
                logger.debug(
                    f"Output token symbol not found for address {output_token_address}. Skipping."
                )
                return Decimal(0)
            market_price = await self.apiconfig.get_real_time_price(
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

    def _log_profit_calculation(
        self,
        transaction_data: Dict[str, Any],
        real_time_price: Decimal,
        gas_cost_eth: Decimal,
        profit: Decimal,
        minimum_profit_eth: float,
    ) -> None:
        """
        Logs a detailed summary of the profit calculation for a transaction.
    
        Args:
            transaction_data (Dict[str, Any]): The transaction data dictionary.
            real_time_price (Decimal): The real-time price of the output token.
            gas_cost_eth (Decimal): The estimated gas cost in ETH.
            profit (Decimal): The calculated profit in ETH.
            minimum_profit_eth (float): The minimum required profit threshold in ETH.
        """
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

    async def _fetch_from_services(self, fetch_func, description: str):
        """
        Helper method to fetch data from multiple API services.
    
        Args:
            fetch_func (Callable): A lambda function that takes a service name and fetches data.
            description (str): Description of the data being fetched for logging.
    
        Returns:
            Any: The fetched data if successful, else None.
        """
        for service in self.apiconfig.keys():
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
        """
        Asynchronously loads a contract's ABI from a specified file path.

        ABI (Application Binary Interface) is a JSON file that defines the methods and properties of a smart contract.
    
        Args:
            abi_path (str): The file path to the ABI JSON file.
    
        Returns:
            List[Dict[str, Any]]: The loaded ABI.
    
        Raises:
            Exception: If loading the ABI fails.
        """
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
        """
        Closes any open resources, such as the aiohttp session, to free up system resources.
        """
        await self.apiconfig.close()

    async def _log_transaction_details(self, tx: Any, is_eth: bool = False) -> None:
        """
        Logs detailed information about a transaction for auditing and debugging purposes.
    
        Args:
            tx (Any): The transaction object to log.
            is_eth (bool, optional): True if the transaction is an ETH transfer. Defaults to False.
        """
        if is_eth:
            logger.debug(
                f"ETH Transaction Details:\n"
                f"Hash: {tx.hash.hex()}\n"
                f"Value: {self.web3.from_wei(tx.value, 'ether')} ETH\n"
                f"To: {tx.to}\n"
                f"From: {tx['from']}\n"
                f"Gas Price: {self.web3.from_wei(tx.gasPrice, 'gwei')} Gwei"
            )
        else:
            logger.debug(
                f"Token Transaction Details:\n"
                f"Hash: {tx.hash.hex()}\n"
                f"Value: {self.web3.from_wei(tx.value, 'ether')} ETH\n"
                f"To: {tx.to}\n"
                f"Input: {tx.input}\n"
                f"Gas Price: {self.web3.from_wei(tx.gasPrice, 'gwei')} Gwei"
            )


#============================= Transaction Orchestrator and Core =============================

class TransactionCore:
    """
    TransactionCore is the main transaction engine that Builds and executes transactions,
    including front-run, back-run, and sandwich attack strategies. It interacts with smart contracts 
    manages transaction signing, gas price estimation, and handles flashloans.
    
    Flashloans are a type of uncollateralized loan that must be 
    borrowed and repaid within the same transaction.
    
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
        apiconfig: Optional[APIConfig] = None,
        monitor: Optional[MempoolMonitor] = None,
        noncecore: Optional[NonceCore] = None,
        safetynet: Optional[SafetyNet] = None,
        configuration: Optional[Configuration] = None,
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

    _abi_cache = {}

    async def _initialize_contract(
        self,
        contract_address: str,
        contract_abi: Union[str, List[Dict[str, Any]]],
        contract_name: str,
    ) -> Any:
        """Initializes a contract instance with error handling and ABI caching."""
        try:
            # Load ABI from file if it's a string path and cache it
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
            erc20_abi = await self.apiconfig._load_abi(self.configuration.ERC20_ABI)
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
                'nonce': await self.noncecore.get_nonce(),
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
            gas_price_gwei = await self.safetynet.get_dynamic_gas_price()
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
            for attempt in range(1, self.retry_attempts + 1):
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
                await self.noncecore.refresh_nonce()
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

        did you know?:
        A transaction is a message that is sent from one account to another account on the blockchain.
        The reason for signing a transaction is to prove that the sender is the owner of the account.

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
                "nonce": await self.noncecore.get_nonce(),
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
                await self.noncecore.refresh_nonce()
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

        did you know?:
        A front-running attack is a type of transaction ordering dependence (TOD) attack where
        the attacker attempts to exploit the time delay between the transaction submission and
        its confirmation on the blockchain. The attacker can insert a transaction in the same
        block as the target transaction to manipulate the order of execution and gain an advantage.

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
            current_profit = await self.safetynet.get_balance(self.account)
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
            await self.safetynet.stop()
            await self.noncecore.stop()
            logger.debug("Stopped 0xBuilder. ")
        except Exception as e:
            logger.error(f"Error stopping 0xBuilder: {e} !")
            raise



import asyncio
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from cachetools import TTLCache
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression

# Assuming logger is predefined
# from your_logging_module import logger

class MarketMonitor:
    """
    The MarketMonitor class is responsible for monitoring and analyzing market data,
    updating predictive models, and assessing market conditions such as volatility
    and liquidity for specified tokens.
    """

    MODEL_UPDATE_INTERVAL: int = 3600  # Update model every hour (in seconds)
    VOLATILITY_THRESHOLD: float = 0.05   # 5% standard deviation as volatility threshold
    LIQUIDITY_THRESHOLD: float = 100000.0  # $100,000 in 24h volume as liquidity threshold

    def __init__(
        self,
        web3: 'AsyncWeb3',
        configuration: Optional['Configuration'] = None,
        apiconfig: Optional['APIConfig'] = None,
    ) -> None:
        """
        Initialize the MarketMonitor with Web3 instance, configuration, and API configurations.

        Args:
            web3 (AsyncWeb3): The asynchronous Web3 instance for blockchain interactions.
            configuration (Optional[Configuration], optional): Configuration settings.
            apiconfig (Optional[APIConfig], optional): API configuration for data fetching services.
        """
        self.web3 = web3
        self.configuration = configuration or Configuration()
        self.apiconfig = apiconfig or APIConfig(self.configuration)

        # Initialize the Linear Regression model for price prediction
        self.price_model: LinearRegression = LinearRegression()
        self.model_last_updated: float = 0.0  # Timestamp of the last model update

        # Initialize a cache for price data with a TTL of 5 minutes and max size of 1000 entries
        self.price_cache: TTLCache = TTLCache(maxsize=1000, ttl=300)

        # Paths for saving/loading the ML model and training data
        self.model_path: str = self.configuration.ML_MODEL_PATH
        self.training_data_path: str = self.configuration.ML_TRAINING_DATA_PATH

        # Lock to ensure that model updates are thread-safe and prevent concurrent updates
        self.model_lock: asyncio.Lock = asyncio.Lock()

        # Asynchronously load existing model and training data if available
        asyncio.create_task(self.load_model())

    async def load_model(self) -> None:
        """
        Asynchronously load the machine learning model and training data from disk.
        Utilizes a lock to prevent concurrent access during loading.
        """
        async with self.model_lock:
            if os.path.exists(self.model_path) and os.path.exists(self.training_data_path):
                try:
                    data = joblib.load(self.model_path)
                    self.price_model = data['model']
                    self.model_last_updated = data.get('model_last_updated', 0.0)
                    logger.info("ML model and training data loaded successfully.")
                except Exception as e:
                    logger.error(f"Failed to load ML model: {e}")
            else:
                logger.info("No existing ML model found. Starting fresh.")

    async def save_model(self) -> None:
        """
        Asynchronously save the machine learning model and training data to disk.
        Utilizes a lock to ensure thread-safe operations during saving.
        """
        async with self.model_lock:
            try:
                data = {
                    'model': self.price_model,
                    'model_last_updated': self.model_last_updated
                }
                joblib.dump(data, self.model_path)
                logger.info(f"ML model saved to {self.model_path}.")
            except Exception as e:
                logger.error(f"Failed to save ML model: {e}")

    async def _update_price_model(self, token_symbol: str) -> None:
        """
        Update the price prediction model with new historical data for a specific token.
        Fetches historical prices, prepares training data, fits the model, and saves it.

        Args:
            token_symbol (str): The symbol of the token to update the model for.
        """
        async with self.model_lock:
            try:
                prices = await self.fetch_historical_prices(token_symbol)
                if len(prices) > 10:
                    # Prepare training data with time steps as features
                    X = np.arange(len(prices)).reshape(-1, 1)  # Time steps
                    y = np.array(prices)  # Corresponding prices

                    # Load existing training data if available
                    if os.path.exists(self.training_data_path):
                        existing_data = pd.read_csv(self.training_data_path)
                        X_existing = existing_data[['time']].values
                        y_existing = existing_data['price'].values
                        X = np.vstack((X_existing, X))
                        y = np.concatenate((y_existing, y))

                    # Save the updated training data to CSV
                    training_df = pd.DataFrame({'time': X.flatten(), 'price': y})
                    training_df.to_csv(self.training_data_path, index=False)
                    logger.debug(f"Training data for {token_symbol} updated with {len(prices)} new points.")

                    # Fit the Linear Regression model with the new data
                    self.price_model.fit(X, y)
                    self.model_last_updated = time.time()
                    logger.info(f"ML model updated and trained for {token_symbol}.")

                    # Persist the updated model to disk
                    await self.save_model()
                else:
                    logger.debug(f"Not enough price data to train the model for {token_symbol}.")
            except Exception as e:
                logger.error(f"Error updating price model: {e}")

    async def periodic_model_training(self, token_symbol: str) -> None:
        """
        Periodically train the ML model based on the defined interval.
        This coroutine runs indefinitely, checking if the model needs an update
        and triggering the update process accordingly.

        Args:
            token_symbol (str): The symbol of the token to train the model for.
        """
        while True:
            try:
                current_time = time.time()
                if current_time - self.model_last_updated > self.MODEL_UPDATE_INTERVAL:
                    logger.debug(f"Time to update ML model for {token_symbol}.")
                    await self._update_price_model(token_symbol)
                await asyncio.sleep(60)  # Check every minute
            except asyncio.CancelledError:
                logger.info("Periodic model training task cancelled.")
                break
            except Exception as e:
                logger.error(f"Error in periodic model training: {e}")

    async def start_periodic_training(self, token_symbol: str) -> None:
        """
        Start the background task for periodic model training for a specific token.

        Args:
            token_symbol (str): The symbol of the token to train the model for.
        """
        asyncio.create_task(self.periodic_model_training(token_symbol))

    async def check_market_conditions(self, token_address: str) -> Dict[str, Any]:
        """
        Assess various market conditions for a given token, such as volatility,
        trend direction, and liquidity.

        Args:
            token_address (str): The blockchain address of the token to check.

        Returns:
            Dict[str, Any]: A dictionary containing the evaluated market conditions.
        """
        market_conditions: Dict[str, Any] = {
            "high_volatility": False,
            "bullish_trend": False,
            "bearish_trend": False,
            "low_liquidity": False,
        }

        token_symbol = await self.apiconfig.get_token_symbol(self.web3, token_address)
        if not token_symbol:
            logger.debug(f"Cannot get token symbol for address {token_address}!")
            return market_conditions

        prices = await self.fetch_historical_prices(token_symbol, days=1)
        if len(prices) < 2:
            logger.debug(f"Not enough price data to analyze market conditions for {token_symbol}")
            return market_conditions

        # Calculate volatility and assess if it exceeds the threshold
        volatility = self._calculate_volatility(prices)
        if volatility > self.VOLATILITY_THRESHOLD:
            market_conditions["high_volatility"] = True
        logger.debug(f"Calculated volatility for {token_symbol}: {volatility:.4f}")

        # Determine trend based on the moving average
        moving_average = np.mean(prices)
        if prices[-1] > moving_average:
            market_conditions["bullish_trend"] = True
        elif prices[-1] < moving_average:
            market_conditions["bearish_trend"] = True

        # Assess liquidity based on trading volume
        volume = await self.get_token_volume(token_symbol)
        if volume < self.LIQUIDITY_THRESHOLD:
            market_conditions["low_liquidity"] = True

        return market_conditions

    def _calculate_volatility(self, prices: List[float]) -> float:
        """
        Calculate the volatility of a list of prices as the standard deviation of returns.

        Args:
            prices (List[float]): A list of historical prices.

        Returns:
            float: The calculated volatility.
        """
        prices_array = np.array(prices)
        returns = np.diff(prices_array) / prices_array[:-1]
        return np.std(returns)

    async def fetch_historical_prices(self, token_symbol: str, days: int = 30) -> List[float]:
        """
        Fetch historical price data for a given token symbol.

        Args:
            token_symbol (str): The symbol of the token to fetch prices for.
            days (int, optional): Number of days to fetch prices for. Defaults to 30.

        Returns:
            List[float]: A list of historical prices.
        """
        cache_key = f"historical_prices_{token_symbol}_{days}"
        if cache_key in self.price_cache:
            logger.debug(f"Returning cached historical prices for {token_symbol}.")
            return self.price_cache[cache_key]

        # Fetch historical prices from available services
        prices = await self._fetch_from_services(
            lambda service: self.apiconfig.fetch_historical_prices(token_symbol, days=days, service=service),
            f"historical prices for {token_symbol}"
        )
        if prices:
            self.price_cache[cache_key] = prices  # Cache the fetched prices
        return prices or []

    async def get_token_volume(self, token_symbol: str) -> float:
        """
        Retrieve the 24-hour trading volume for a specified token.

        Args:
            token_symbol (str): The symbol of the token to fetch volume for.

        Returns:
            float: The 24-hour trading volume in USD.
        """
        cache_key = f"token_volume_{token_symbol}"
        if cache_key in self.price_cache:
            logger.debug(f"Returning cached trading volume for {token_symbol}.")
            return self.price_cache[cache_key]

        # Fetch trading volume from available services
        volume = await self._fetch_from_services(
            lambda service: self.apiconfig.get_token_volume(token_symbol, service=service),
            f"trading volume for {token_symbol}"
        )
        if volume is not None:
            self.price_cache[cache_key] = volume  # Cache the fetched volume
        return volume or 0.0

    async def _fetch_from_services(
        self,
        fetch_func: Callable[[str], Any],
        description: str
    ) -> Optional[Any]:
        """
        Helper method to fetch data from multiple configured services.

        Args:
            fetch_func (Callable[[str], Any]): The function to fetch data from a service.
            description (str): Description of the data being fetched.

        Returns:
            Optional[Any]: The fetched data or None if all services fail.
        """
        for service in self.apiconfig.apiconfig.keys():
            try:
                logger.debug(f"Fetching {description} using {service}...")
                result = await fetch_func(service)
                if result:
                    logger.debug(f"Successfully fetched {description} using {service}.")
                    return result
            except Exception as e:
                logger.warning(f"Failed to fetch {description} using {service}: {e}")
        logger.warning(f"All services failed to fetch {description}.")
        return None

    async def predict_price_movement(self, token_symbol: str) -> float:
        """
        Predict the next price movement for a given token symbol using the ML model.

        Args:
            token_symbol (str): The symbol of the token to predict price movement for.

        Returns:
            float: The predicted price movement.
        """
        current_time = time.time()
        if current_time - self.model_last_updated > self.MODEL_UPDATE_INTERVAL:
            logger.debug(f"Model needs updating for {token_symbol}. Triggering update.")
            await self._update_price_model(token_symbol)

        prices = await self.fetch_historical_prices(token_symbol, days=1)
        if not prices:
            logger.debug(f"No recent prices available for {token_symbol}.")
            return 0.0

        try:
            # Predict the next price based on the current data
            X_pred = np.array([[len(prices)]])
            predicted_price = self.price_model.predict(X_pred)[0]
            logger.debug(f"Price prediction for {token_symbol}: {predicted_price:.6f}")
            return float(predicted_price)
        except Exception as e:
            logger.error(f"Error predicting price movement: {e}")
            return 0.0

    async def is_arbitrage_opportunity(self, target_tx: Dict[str, Any]) -> bool:
        """
        Determine if there's an arbitrage opportunity based on the target transaction.

        Args:
            target_tx (Dict[str, Any]): The target transaction dictionary.

        Returns:
            bool: True if an arbitrage opportunity is detected, else False.
        """
        decoded_tx = await self.decode_transaction_input(target_tx.get("input", ""), target_tx.get("to", ""))
        if not decoded_tx:
            return False

        path = decoded_tx["params"].get("path", [])
        if len(path) < 2:
            return False

        token_address = path[-1]  # The token being bought
        token_symbol = await self.apiconfig.get_token_symbol(self.web3, token_address)
        if not token_symbol:
            return False

        # Fetch real-time prices from different services
        prices = await self._get_prices_from_services(token_symbol)
        if len(prices) < 2:
            return False

        # Calculate the difference and percentage to assess arbitrage potential
        price_difference = abs(prices[0] - prices[1])
        average_price = sum(prices) / len(prices)
        if average_price == 0:
            return False

        price_difference_percentage = price_difference / average_price
        if price_difference_percentage > 0.01:  # Arbitrage threshold set at 1%
            logger.debug(f"Arbitrage opportunity detected for {token_symbol} ({price_difference_percentage*100:.2f}%)")
            return True
        return False

    async def _get_prices_from_services(self, token_symbol: str) -> List[float]:
        """
        Retrieve real-time prices for a token from different services.

        Args:
            token_symbol (str): The symbol of the token to get prices for.

        Returns:
            List[float]: A list of real-time prices from various services.
        """
        prices: List[float] = []
        for service in self.apiconfig.apiconfig.keys():
            try:
                price = await self.apiconfig.get_real_time_price(token_symbol, service=service)
                if price is not None:
                    prices.append(price)
                    logger.debug(f"Price from {service}: {price}")
            except Exception as e:
                logger.warning(f"Failed to get price from {service}: {e}")
        return prices

    async def decode_transaction_input(
        self, input_data: str, contract_address: str
    ) -> Optional[Dict[str, Any]]:
        """
        Decode the input data of a blockchain transaction.

        Args:
            input_data (str): Hexadecimal input data of the transaction.
            contract_address (str): Address of the contract being interacted with.

        Returns:
            Optional[Dict[str, Any]]: Decoded transaction details if successful, else None.
        """
        try:
            # Assuming ERC20 ABI is loaded elsewhere or passed appropriately
            erc20_abi = self.configuration.ERC20_ABI  # Replace with actual ABI retrieval
            contract = self.web3.eth.contract(address=contract_address, abi=erc20_abi)
            function_abi, params = contract.decode_function_input(input_data)
            logger.debug(f"Decoded transaction input: {function_abi['name']} with params {params}")
            return {"function_name": function_abi["name"], "params": params}
        except Exception as e:
            logger.warning(f"Failed to decode transaction input: {e}")
            return None


@dataclass
class StrategyPerformanceMetrics:
    """
    Dataclass to track performance metrics for each strategy type.
    """
    avg_execution_time: float = 0.0
    success_rate: float = 0.0
    total_executions: int = 0
    successes: int = 0
    failures: int = 0
    profit: Decimal = Decimal("0.0")

@dataclass
class StrategyConfiguration:
    """
    Dataclass to hold configuration parameters for strategy execution and learning.
    """
    decay_factor: float = 0.95
    min_profit_threshold: Decimal = Decimal("0.01")
    learning_rate: float = 0.01
    exploration_rate: float = 0.1

@dataclass
class StrategyExecutionError(Exception):
    """
    Custom exception raised when a strategy execution fails.
    """
    message: str

    def __str__(self) -> str:
        return self.message

class StrategyNet:
    """
    The StrategyNet class orchestrates various trading strategies, manages their execution,
    and employs reinforcement learning to optimize strategy selection based on performance.
    """

    def __init__(
        self,
        transactioncore: Optional['TransactionCore'] = None,
        marketmonitor: Optional['MarketMonitor'] = None,
        safetynet: Optional['SafetyNet'] = None,
        apiconfig: Optional['APIConfig'] = None,
        configuration: Optional['Configuration'] = None,
    ) -> None:
        """
        Initialize the StrategyNet with required components.

        Args:
            transactioncore (Optional[TransactionCore], optional): Core transaction handler.
            marketmonitor (Optional[MarketMonitor], optional): Market monitoring and analysis component.
            safetynet (Optional[SafetyNet], optional): Safety mechanisms to prevent harmful operations.
            apiconfig (Optional[APIConfig], optional): API configurations for data fetching.
            configuration (Optional[Configuration], optional): Configuration settings.
        """
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
        Register a new strategy dynamically under a specific strategy type.

        Args:
            strategy_type (str): The type/category of the strategy.
            strategy_func (Callable[[Dict[str, Any]], asyncio.Future]): The strategy function to register.
        """
        if strategy_type not in self.strategy_types:
            logger.warning(f"Attempted to register unknown strategy type: {strategy_type}")
            return
        self._strategy_registry[strategy_type].append(strategy_func)
        self.reinforcement_weights[strategy_type].append(1.0)
        logger.info(f"Registered new strategy '{strategy_func.__name__}' under '{strategy_type}'")

    def get_strategies(
        self,
        strategy_type: str
    ) -> List[Callable[[Dict[str, Any]], asyncio.Future]]:
        """
        Retrieve all strategies registered under a specific strategy type.

        Args:
            strategy_type (str): The type/category of the strategy.

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
        Execute the most suitable strategy for the given strategy type based on performance metrics.

        Args:
            target_tx (Dict[str, Any]): The target transaction dictionary.
            strategy_type (str): The type/category of the strategy to execute.

        Returns:
            bool: True if the strategy was executed successfully, else False.
        """
        strategies = self.get_strategies(strategy_type)
        if not strategies:
            logger.debug(f"No strategies available for type: {strategy_type}")
            return False

        try:
            start_time = time.time()
            selected_strategy = await self._select_best_strategy(strategies, strategy_type)

            # Capture profit before strategy execution
            profit_before = await self.transactioncore.get_current_profit()

            # Execute the selected strategy
            success = await selected_strategy(target_tx)

            # Capture profit after strategy execution
            profit_after = await self.transactioncore.get_current_profit()

            # Calculate execution metrics
            execution_time = time.time() - start_time
            profit_made = profit_after - profit_before

            # Update strategy performance metrics
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
        self,
        strategies: List[Callable[[Dict[str, Any]], asyncio.Future]],
        strategy_type: str
    ) -> Callable[[Dict[str, Any]], asyncio.Future]:
        """
        Select the best strategy based on reinforcement learning weights and exploration rate.

        Args:
            strategies (List[Callable[[Dict[str, Any]], asyncio.Future]]): List of strategy functions.
            strategy_type (str): The type/category of the strategy.

        Returns:
            Callable[[Dict[str, Any]], asyncio.Future]: The selected strategy function.
        """
        weights = self.reinforcement_weights[strategy_type]
        if not weights:
            logger.debug("No weights available, using random strategy selection")
            return random.choice(strategies)

        # Decide between exploration and exploitation based on exploration rate
        if random.random() < self.configuration.exploration_rate:
            logger.debug("Using exploration for strategy selection")
            return random.choice(strategies)

        # Normalize weights using softmax for probability distribution
        max_weight = max(weights)
        exp_weights = [np.exp(w - max_weight) for w in weights]
        sum_exp = sum(exp_weights)
        probabilities = [w / sum_exp for w in exp_weights]

        # Select strategy based on calculated probabilities
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
        execution_time: float
    ) -> None:
        """
        Update performance metrics for a strategy based on its execution outcome.

        Args:
            strategy_name (str): The name of the strategy function executed.
            strategy_type (str): The type/category of the strategy.
            success (bool): Whether the strategy execution was successful.
            profit (Decimal): Profit made from the strategy execution.
            execution_time (float): Time taken to execute the strategy.
        """
        metrics = self.strategy_performance[strategy_type]
        metrics.total_executions += 1

        if success:
            metrics.successes += 1
            metrics.profit += profit
        else:
            metrics.failures += 1

        # Update average execution time using exponential moving average
        metrics.avg_execution_time = (
            metrics.avg_execution_time * self.configuration.decay_factor
            + execution_time * (1 - self.configuration.decay_factor)
        )

        # Update success rate
        metrics.success_rate = metrics.successes / metrics.total_executions

        # Retrieve the index of the strategy in the registry for weight updating
        strategy_index = self.get_strategy_index(strategy_name, strategy_type)
        if strategy_index >= 0:
            # Calculate reward based on execution outcome
            reward = self._calculate_reward(success, profit, execution_time)
            # Update reinforcement learning weight for the strategy
            self._update_reinforcement_weight(strategy_type, strategy_index, reward)

        # Append execution details to history data for tracking
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
        """
        Retrieve the index of a strategy within its strategy type's list.

        Args:
            strategy_name (str): The name of the strategy function.
            strategy_type (str): The type/category of the strategy.

        Returns:
            int: The index of the strategy, or -1 if not found.
        """
        strategies = self.get_strategies(strategy_type)
        for index, strategy in enumerate(strategies):
            if strategy.__name__ == strategy_name:
                return index
        logger.warning(f"Strategy '{strategy_name}' not found in type '{strategy_type}'")
        return -1

    def _calculate_reward(
        self,
        success: bool,
        profit: Decimal,
        execution_time: float
    ) -> float:
        """
        Calculate the reward for a strategy execution to be used in reinforcement learning.

        Args:
            success (bool): Whether the strategy was successful.
            profit (Decimal): Profit made from the strategy.
            execution_time (float): Time taken to execute the strategy.

        Returns:
            float: The calculated reward.
        """
        # Base reward is the profit if successful, else a negative penalty
        base_reward = float(profit) if success else -0.1
        # Time penalty to discourage strategies that take too long
        time_penalty = -0.01 * execution_time
        total_reward = base_reward + time_penalty
        logger.debug(f"Calculated reward: {total_reward:.4f} (Base: {base_reward}, Time Penalty: {time_penalty})")
        return total_reward

    def _update_reinforcement_weight(
        self,
        strategy_type: str,
        index: int,
        reward: float
    ) -> None:
        """
        Update the reinforcement learning weight for a specific strategy based on the reward.

        Args:
            strategy_type (str): The type/category of the strategy.
            index (int): The index of the strategy within its strategy type's list.
            reward (float): The reward to apply to the strategy's weight.
        """
        lr = self.configuration.learning_rate  # Learning rate for weight updates
        current_weight = self.reinforcement_weights[strategy_type][index]
        # Update weight using a simple learning rule
        new_weight = current_weight * (1 - lr) + reward * lr
        # Ensure that the weight does not fall below a minimum threshold
        self.reinforcement_weights[strategy_type][index] = max(0.1, new_weight)
        logger.debug(f"Updated weight for strategy index {index} in '{strategy_type}': {new_weight:.4f}")

    async def _decode_transaction(self, target_tx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Decode the input data of a transaction to understand its purpose.

        Args:
            target_tx (Dict[str, Any]): The target transaction dictionary.

        Returns:
            Optional[Dict[str, Any]]: Decoded transaction details if successful, else None.
        """
        try:
            decoded = await self.transactioncore.decode_transaction_input(
                target_tx.get("input", ""), target_tx.get("to", "")
            )
            logger.debug(f"Decoded transaction: {decoded}")
            return decoded
        except Exception as e:
            logger.error(f"Error decoding transaction: {e}")
            return None

    async def _get_token_symbol(self, token_address: str) -> Optional[str]:
        """
        Retrieve the token symbol given its blockchain address.

        Args:
            token_address (str): The blockchain address of the token.

        Returns:
            Optional[str]: The token symbol if found, else None.
        """
        try:
            symbol = await self.apiconfig.get_token_symbol(
                self.transactioncore.web3, token_address
            )
            logger.debug(f"Retrieved token symbol '{symbol}' for address '{token_address}'")
            return symbol
        except Exception as e:
            logger.error(f"Error fetching token symbol: {e}")
            return None

    # ========================= Strategy Implementations =========================

    async def high_value_eth_transfer(self, target_tx: Dict[str, Any]) -> bool:
        """
        Execute the High-Value ETH Transfer Strategy with advanced validation and dynamic thresholds.

        Args:
            target_tx (Dict[str, Any]): The target transaction dictionary.

        Returns:
            bool: True if the transaction was executed successfully, else False.
        """
        logger.debug("Initiating High-Value ETH Transfer Strategy...")

        try:
            if not self._is_valid_transaction(target_tx):
                return False

            eth_value_in_wei, gas_price, to_address = self._extract_transaction_details(target_tx)
            eth_value, gas_price_gwei, threshold = self._calculate_thresholds(eth_value_in_wei, gas_price)

            self._log_transaction_analysis(eth_value, gas_price_gwei, to_address, threshold)

            if not await self._additional_validation_checks(eth_value_in_wei, to_address):
                return False

            if eth_value_in_wei > threshold:
                logger.info(
                    f"High-value ETH transfer detected:\n"
                    f"Value: {eth_value:.4f} ETH\n"
                    f"Threshold: {self.transactioncore.web3.from_wei(threshold, 'ether')} ETH"
                )
                return await self.transactioncore.handle_eth_transaction(target_tx)

            logger.debug(
                f"ETH transaction value ({eth_value:.4f} ETH) below threshold "
                f"({self.transactioncore.web3.from_wei(threshold, 'ether')} ETH). Skipping..."
            )
            return False

        except Exception as e:
            logger.error(f"Error in high-value ETH transfer strategy: {e}")
            return False

    def _is_valid_transaction(self, target_tx: Dict[str, Any]) -> bool:
        """
        Validate the transaction format.

        Args:
            target_tx (Dict[str, Any]): The target transaction dictionary.

        Returns:
            bool: True if valid, else False.
        """
        if not isinstance(target_tx, dict) or not target_tx:
            logger.debug("Invalid transaction format provided!")
            return False
        return True

    def _extract_transaction_details(self, target_tx: Dict[str, Any]) -> Tuple[int, int, str]:
        """
        Extract essential details from the transaction.

        Args:
            target_tx (Dict[str, Any]): The target transaction dictionary.

        Returns:
            Tuple[int, int, str]: ETH value in Wei, gas price, and recipient address.
        """
        eth_value_in_wei = int(target_tx.get("value", 0))
        gas_price = int(target_tx.get("gasPrice", 0))
        to_address = target_tx.get("to", "")
        return eth_value_in_wei, gas_price, to_address

    def _calculate_thresholds(self, eth_value_in_wei: int, gas_price: int) -> Tuple[float, float, int]:
        """
        Calculate dynamic thresholds based on gas price.

        Args:
            eth_value_in_wei (int): ETH value in Wei.
            gas_price (int): Gas price in Wei.

        Returns:
            Tuple[float, float, int]: ETH value, gas price in Gwei, and threshold in Wei.
        """
        eth_value = self.transactioncore.web3.from_wei(eth_value_in_wei, "ether")
        gas_price_gwei = self.transactioncore.web3.from_wei(gas_price, "gwei")

        base_threshold = self.transactioncore.web3.to_wei(10, "ether")
        if gas_price_gwei > 200:
            threshold = base_threshold * 2
        elif gas_price_gwei > 100:
            threshold = base_threshold * 1.5
        else:
            threshold = base_threshold

        return eth_value, gas_price_gwei, int(threshold)

    def _log_transaction_analysis(
        self,
        eth_value: float,
        gas_price_gwei: float,
        to_address: str,
        threshold: int
    ) -> None:
        """
        Log the transaction analysis details.

        Args:
            eth_value (float): ETH value.
            gas_price_gwei (float): Gas price in Gwei.
            to_address (str): Recipient address.
            threshold (int): Threshold in Wei.
        """
        threshold_eth = self.transactioncore.web3.from_wei(threshold, 'ether')
        logger.debug(
            f"Transaction Analysis:\n"
            f"Value: {eth_value:.4f} ETH\n"
            f"Gas Price: {gas_price_gwei:.2f} Gwei\n"
            f"To Address: {to_address[:10]}...\n"
            f"Current Threshold: {threshold_eth} ETH"
        )

    async def _additional_validation_checks(self, eth_value_in_wei: int, to_address: str) -> bool:
        """
        Perform additional validation checks on the transaction.

        Args:
            eth_value_in_wei (int): ETH value in Wei.
            to_address (str): Recipient address.

        Returns:
            bool: True if all checks pass, else False.
        """
        if eth_value_in_wei <= 0:
            logger.debug("Transaction value is zero or negative. Skipping...")
            return False

        if not self.transactioncore.web3.is_address(to_address):
            logger.debug("Invalid recipient address. Skipping...")
            return False

        is_contract = await self._is_contract_address(to_address)
        if is_contract:
            logger.debug("Recipient is a contract. Additional validation required...")
            if not await self._validate_contract_interaction(to_address):
                return False

        return True

    async def _is_contract_address(self, address: str) -> bool:
        """
        Determine if a given address is a smart contract.

        Args:
            address (str): The blockchain address to check.

        Returns:
            bool: True if the address is a contract, else False.
        """
        try:
            code = await self.transactioncore.web3.eth.get_code(address)
            is_contract = len(code) > 0
            logger.debug(f"Address '{address}' is_contract: {is_contract}")
            return is_contract
        except Exception as e:
            logger.error(f"Error checking if address is contract: {e}")
            return False

    async def _validate_contract_interaction(self, contract_address: str) -> bool:
        """
        Validate interactions with a contract address to ensure legitimacy.

        Args:
            contract_address (str): The address of the contract to validate.

        Returns:
            bool: True if the interaction is valid, else False.
        """
        try:
            # Example validation: check if it's a known contract
            token_symbols = await self.configuration.get_token_symbols()
            is_valid = contract_address in token_symbols
            logger.debug(f"Contract address '{contract_address}' validation result: {is_valid}")
            return is_valid
        except Exception as e:
            logger.error(f"Error validating contract interaction: {e}")
            return False

    async def aggressive_front_run(self, target_tx: Dict[str, Any]) -> bool:
        """
        Execute the Aggressive Front-Run Strategy with comprehensive validation,
        dynamic thresholds, and risk assessment.

        Args:
            target_tx (Dict[str, Any]): The target transaction dictionary.

        Returns:
            bool: True if the strategy was executed successfully, else False.
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
            value_eth = self.transactioncore.web3.from_wei(tx_value, "ether")
            threshold = self._calculate_dynamic_threshold(gas_price)

            logger.debug(
                f"Transaction Analysis:\n"
                f"Hash: {tx_hash}\n"
                f"Value: {value_eth:.4f} ETH\n"
                f"Gas Price: {self.transactioncore.web3.from_wei(gas_price, 'gwei'):.2f} Gwei\n"
                f"Threshold: {threshold:.4f} ETH"
            )

            # Step 4: Risk assessment
            risk_score = await self._assess_front_run_risk(target_tx)
            if risk_score < 0.5:  # Risk score below threshold indicates high risk
                logger.debug(f"Risk score too high ({risk_score:.2f}). Skipping front-run.")
                return False

            # Step 5: Check if opportunity value meets the threshold
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
                return await self.transactioncore.front_run(target_tx)

            # Skip execution if the transaction value is below the threshold
            logger.debug(
                f"Transaction value {value_eth:.4f} ETH below threshold {threshold:.4f} ETH. Skipping..."
            )
            return False

        except Exception as e:
            logger.error(f"Error in aggressive front-run strategy: {e}")
            return False

    def _calculate_dynamic_threshold(self, gas_price: int) -> float:
        """
        Calculate a dynamic threshold based on current gas prices to adjust strategy aggressiveness.

        Args:
            gas_price (int): Gas price in Wei.

        Returns:
            float: The calculated threshold in ETH.
        """
        gas_price_gwei = float(self.transactioncore.web3.from_wei(gas_price, "gwei"))

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
        Calculate the risk score for front-running a transaction.

        Args:
            tx (Dict[str, Any]): The transaction dictionary.

        Returns:
            float: Risk score on a scale from 0 to 1, where lower scores indicate higher risk.
        """
        try:
            risk_score = 1.0  # Start with maximum risk

            # Gas price impact
            gas_price = int(tx.get("gasPrice", 0))
            gas_price_gwei = float(self.transactioncore.web3.from_wei(gas_price, "gwei"))
            if gas_price_gwei > 300:
                risk_score *= 0.7  # High gas price increases risk

            # Contract interaction check
            input_data = tx.get("input", "0x")
            if len(input_data) > 10:  # Complex contract interaction implies higher risk
                risk_score *= 0.8

            # Check market conditions
            market_conditions = await self.marketmonitor.check_market_conditions(tx.get("to", ""))
            if market_conditions.get("high_volatility", False):
                risk_score *= 0.7
            if market_conditions.get("low_liquidity", False):
                risk_score *= 0.6

            risk_score = max(risk_score, 0.0)  # Ensure non-negative score
            logger.debug(f"Assessed front-run risk score: {risk_score:.2f}")
            return round(risk_score, 2)
        except Exception as e:
            logger.error(f"Error assessing front-run risk: {e}")
            return 0.0  # Return maximum risk on error

    async def _validate_high_value_transaction(self, tx: Dict[str, Any]) -> bool:
        """
        Perform additional validation for high-value transactions to ensure legitimacy.

        Args:
            tx (Dict[str, Any]): The transaction dictionary.

        Returns:
            bool: True if the transaction passes additional validations, else False.
        """
        try:
            # Check if the target address is a known contract
            to_address = tx.get("to", "")
            if not to_address:
                logger.debug("Transaction missing 'to' address.")
                return False

            # Verify that code exists at the target address
            code = await self.transactioncore.web3.eth.get_code(to_address)
            if not code:
                logger.warning(f"No contract code found at {to_address}")
                return False

            # Check if the address corresponds to a known token or DEX contract
            token_symbols = await self.configuration.get_token_symbols()
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
        Execute the Predictive Front-Run Strategy based on advanced price prediction
        and multiple market indicators.

        Args:
            target_tx (Dict[str, Any]): The target transaction dictionary.

        Returns:
            bool: True if the strategy was executed successfully, else False.
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
                predicted_price, current_price, market_conditions, historical_prices = await asyncio.gather(
                    self.marketmonitor.predict_price_movement(token_symbol),
                    self.apiconfig.get_real_time_price(token_symbol, service="primary"),
                    self.marketmonitor.check_market_conditions(target_tx.get("to", "")),
                    self.marketmonitor.fetch_historical_prices(token_symbol, days=1),
                    return_exceptions=True
                )

                # Handle potential exceptions from gathered tasks
                if isinstance(predicted_price, Exception) or \
                   isinstance(current_price, Exception) or \
                   isinstance(market_conditions, Exception) or \
                   isinstance(historical_prices, Exception):
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
                return await self.transactioncore.front_run(target_tx)

            # Skip execution if opportunity score is below the threshold
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
        Calculate a comprehensive opportunity score based on multiple metrics.

        Args:
            price_change (float): Expected percentage change in price.
            volatility (float): Calculated volatility.
            market_conditions (Dict[str, bool]): Current market conditions.
            current_price (float): Current price of the token.
            historical_prices (List[float]): Historical prices for trend analysis.

        Returns:
            float: The calculated opportunity score out of 100.
        """
        score = 0.0

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
        Execute the Volatility Front-Run Strategy based on market volatility analysis
        with advanced risk assessment and dynamic thresholds.

        Args:
            target_tx (Dict[str, Any]): The target transaction dictionary.

        Returns:
            bool: True if the strategy was executed successfully, else False.
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

            token_symbol = await self._get_token_symbol(path[0])
            if not token_symbol:
                logger.debug(f"Cannot get token symbol for {token_symbol}. Skipping...")
                return False

            # Gather market data asynchronously
            results = await asyncio.gather(
                self.marketmonitor.check_market_conditions(target_tx.get("to", "")),
                self.apiconfig.get_real_time_price(token_symbol, service="primary"),
                self.marketmonitor.fetch_historical_prices(token_symbol, days=1),
                return_exceptions=True
            )

            market_conditions, current_price, historical_prices = results

            # Handle potential exceptions from gathered tasks
            if any(isinstance(result, Exception) for result in results):
                logger.warning("Failed to gather complete market data.")
                return False

            # Calculate volatility score based on historical data and market conditions
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
                return await self.transactioncore.front_run(target_tx)

            # Skip execution if volatility score is below the threshold
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
        Calculate a comprehensive volatility score based on multiple metrics.

        Args:
            historical_prices (List[float]): List of historical prices.
            current_price (float): Current price of the token.
            market_conditions (Dict[str, bool]): Current market conditions.

        Returns:
            float: The calculated volatility score out of 100.
        """
        score = 0.0

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
        Execute the Advanced Front-Run Strategy with comprehensive analysis,
        risk management, and multi-factor decision making.

        Args:
            target_tx (Dict[str, Any]): The target transaction dictionary.

        Returns:
            bool: True if the strategy was executed successfully, else False.
        """
        logger.debug("Initiating Advanced Front-Run Strategy...")
        # TODO: Implement advanced front-run strategy
        return False

    async def price_dip_back_run(self, target_tx: Dict[str, Any]) -> bool:
        """
        Execute the Price Dip Back-Run Strategy based on price dip prediction.

        Args:
            target_tx (Dict[str, Any]): The target transaction dictionary.

        Returns:
            bool: True if the strategy was executed successfully, else False.
        """
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

        current_price = await self.apiconfig.get_real_time_price(token_symbol, service="primary")
        if current_price is None:
            return False

        predicted_price = await self.marketmonitor.predict_price_movement(token_symbol)
        if predicted_price < float(current_price) * 0.99:
            logger.debug("Predicted price decrease exceeds threshold, proceeding with back-run.")
            return await self.transactioncore.back_run(target_tx)

        logger.debug("Predicted price decrease does not meet threshold. Skipping back-run.")
        return False

    async def flashloan_back_run(self, target_tx: Dict[str, Any]) -> bool:
        """
        Execute the Flashloan Back-Run Strategy using flash loans.

        Args:
            target_tx (Dict[str, Any]): The target transaction dictionary.

        Returns:
            bool: True if the strategy was executed successfully, else False.
        """
        logger.debug("Initiating Flashloan Back-Run Strategy...")
        estimated_amount = await self.transactioncore.calculate_flashloan_amount(target_tx)
        estimated_profit = estimated_amount * Decimal("0.02")  # Example profit calculation
        if estimated_profit > self.configuration.min_profit_threshold:
            logger.debug(f"Estimated profit: {estimated_profit} ETH meets threshold.")
            return await self.transactioncore.back_run(target_tx)
        logger.debug("Profit is insufficient for flashloan back-run. Skipping.")
        return False

    async def high_volume_back_run(self, target_tx: Dict[str, Any]) -> bool:
        """
        Execute the High Volume Back-Run Strategy based on trading volume.

        Args:
            target_tx (Dict[str, Any]): The target transaction dictionary.

        Returns:
            bool: True if the strategy was executed successfully, else False.
        """
        logger.debug("Initiating High Volume Back-Run Strategy...")
        token_address = target_tx.get("to")
        token_symbol = await self._get_token_symbol(token_address)
        if not token_symbol:
            return False

        volume_24h = await self.apiconfig.get_token_volume(token_symbol, service="primary")
        volume_threshold = self._get_volume_threshold(token_symbol)
        if volume_24h > volume_threshold:
            logger.debug(f"High volume detected (${volume_24h:,.2f} USD), proceeding with back-run.")
            return await self.transactioncore.back_run(target_tx)

        logger.debug(f"Volume (${volume_24h:,.2f} USD) below threshold (${volume_threshold:,.2f} USD). Skipping.")
        return False

    def _get_volume_threshold(self, token_symbol: str) -> float:
        """
        Determine the volume threshold for a token based on predefined tiers.

        Args:
            token_symbol (str): The symbol of the token.

        Returns:
            float: The volume threshold in USD.
        """
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

        # Check each tier in order to assign the appropriate threshold
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
        """
        Execute the Advanced Back-Run Strategy with comprehensive analysis.

        Args:
            target_tx (Dict[str, Any]): The target transaction dictionary.

        Returns:
            bool: True if the strategy was executed successfully, else False.
        """
        logger.debug("Initiating Advanced Back-Run Strategy...")
        decoded_tx = await self._decode_transaction(target_tx)
        if not decoded_tx:
            return False

        market_conditions = await self.marketmonitor.check_market_conditions(target_tx.get("to", ""))
        if market_conditions.get("high_volatility", False) and market_conditions.get("bullish_trend", False):
            logger.debug("Market conditions favorable for advanced back-run.")
            return await self.transactioncore.back_run(target_tx)

        logger.debug("Market conditions unfavorable for advanced back-run. Skipping.")
        return False

    async def flash_profit_sandwich(self, target_tx: Dict[str, Any]) -> bool:
        """
        Execute the Flash Profit Sandwich Strategy using flash loans.

        Args:
            target_tx (Dict[str, Any]): The target transaction dictionary.

        Returns:
            bool: True if the strategy was executed successfully, else False.
        """
        logger.debug("Initiating Flash Profit Sandwich Strategy...")
        estimated_amount = await self.transactioncore.calculate_flashloan_amount(target_tx)
        estimated_profit = estimated_amount * Decimal("0.02")  # Example profit calculation
        if estimated_profit > self.configuration.min_profit_threshold:
            gas_price = await self.transactioncore.get_dynamic_gas_price()
            if gas_price > 200:
                logger.debug(f"Gas price too high for sandwich attack: {gas_price} Gwei")
                return False
            logger.debug(f"Executing sandwich with estimated profit: {estimated_profit:.4f} ETH")
            return await self.transactioncore.execute_sandwich_attack(target_tx)
        logger.debug("Insufficient profit potential for flash sandwich. Skipping.")
        return False

    async def price_boost_sandwich(self, target_tx: Dict[str, Any]) -> bool:
        """
        Execute the Price Boost Sandwich Strategy based on price momentum.

        Args:
            target_tx (Dict[str, Any]): The target transaction dictionary.

        Returns:
            bool: True if the strategy was executed successfully, else False.
        """
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

        historical_prices = await self.marketmonitor.fetch_historical_prices(token_symbol, days=1)
        if not historical_prices:
            return False

        momentum = await self._analyze_price_momentum(historical_prices)
        if momentum > 0.02:
            logger.debug(f"Strong price momentum detected: {momentum:.2%}")
            return await self.transactioncore.execute_sandwich_attack(target_tx)

        logger.debug(f"Insufficient price momentum: {momentum:.2%}. Skipping.")
        return False

    async def _analyze_price_momentum(self, prices: List[float]) -> float:
        """
        Analyze the price momentum from historical prices.

        Args:
            prices (List[float]): List of historical prices.

        Returns:
            float: Calculated price momentum.
        """
        if not prices or len(prices) < 2:
            logger.debug("Insufficient historical prices for momentum analysis.")
            return 0.0
        price_changes = [prices[i] / prices[i - 1] - 1 for i in range(1, len(prices))]
        momentum = sum(price_changes) / len(price_changes)
        logger.debug(f"Calculated price momentum: {momentum:.4f}")
        return momentum

    async def arbitrage_sandwich(self, target_tx: Dict[str, Any]) -> bool:
        """
        Execute the Arbitrage Sandwich Strategy based on arbitrage opportunities.

        Args:
            target_tx (Dict[str, Any]): The target transaction dictionary.

        Returns:
            bool: True if the strategy was executed successfully, else False.
        """
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

        is_arbitrage = await self.marketmonitor.is_arbitrage_opportunity(target_tx)
        if is_arbitrage:
            logger.debug(f"Arbitrage opportunity detected for {token_symbol}")
            return await self.transactioncore.execute_sandwich_attack(target_tx)

        logger.debug("No profitable arbitrage opportunity found. Skipping.")
        return False

    async def advanced_sandwich_attack(self, target_tx: Dict[str, Any]) -> bool:
        """
        Execute the Advanced Sandwich Attack Strategy with risk management.

        Args:
            target_tx (Dict[str, Any]): The target transaction dictionary.

        Returns:
            bool: True if the strategy was executed successfully, else False.
        """
        logger.debug("Initiating Advanced Sandwich Attack...")
        decoded_tx = await self._decode_transaction(target_tx)
        if not decoded_tx:
            return False

        market_conditions = await self.marketmonitor.check_market_conditions(target_tx.get("to", ""))
        if market_conditions.get("high_volatility", False) and market_conditions.get("bullish_trend", False):
            logger.debug("Conditions favorable for sandwich attack.")
            return await self.transactioncore.execute_sandwich_attack(target_tx)

        logger.debug("Conditions unfavorable for sandwich attack. Skipping.")
        return False

    # ========================= End of Strategy Implementations =========================



class MainCore:
    """
    The MainCore class orchestrates the entire MEV bot operation, initializing all components,
    managing connections, and handling the main execution loop.
    """

    def __init__(self, configuration: Optional[Configuration] = None) -> None:
        """
        Initialize the MainCore with optional configuration.

        Args:
            configuration (Optional[Configuration], optional): Configuration settings. Defaults to None.
        """
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

        logger.info("MainCore initialized successfully.")

    async def initialize(self) -> None:
        """
        Initialize all components of the MEV bot with comprehensive error handling.
        This includes setting up the account, Web3 connection, and initializing all sub-components.
        """
        try:
            # Initialize account first
            wallet_key = self.configuration.WALLET_KEY
            if not wallet_key:
                raise ValueError("Wallet key is not set in configuration.")

            try:
                # Clean and validate the wallet key format
                cleaned_key = wallet_key[2:] if wallet_key.startswith('0x') else wallet_key
                if not all(c in '0123456789abcdefABCDEF' for c in cleaned_key) or len(cleaned_key) != 64:
                    raise ValueError("Invalid wallet key format - must be a 64-character hexadecimal string")

                # Ensure the key has the '0x' prefix
                full_key = f"0x{cleaned_key}" if not wallet_key.startswith('0x') else wallet_key
                self.account = Account.from_key(full_key)
                logger.info(f"Account initialized: {self.account.address}")
            except Exception as e:
                raise ValueError(f"Invalid wallet key format: {e}")

            # Initialize Web3 after account is set up
            self.web3 = await self._initialize_web3()
            if not self.web3:
                raise RuntimeError("Failed to initialize Web3 connection")

            # Check the account balance to ensure sufficient funds
            await self._check_account_balance()

            # Initialize all other components
            await self._initialize_components()
            logger.info("All components initialized successfully.")
        except Exception as e:
            logger.critical(f"Fatal error during initialization: {e}!")
            await self.stop()

    async def _initialize_web3(self) -> Optional[AsyncWeb3]:
        """
        Initialize the Web3 connection using multiple providers with fallback mechanisms.

        Returns:
            Optional[AsyncWeb3]: The initialized AsyncWeb3 instance if successful, else None.
        """
        providers = self._get_providers()
        if not providers:
            logger.error("No valid endpoints provided.")
            return None

        for provider_name, provider in providers:
            try:
                logger.debug(f"Attempting connection with {provider_name}...")
                web3 = AsyncWeb3(provider, modules={"eth": (AsyncEth,)})

                # Test the connection with retries
                if await self._test_connection(web3, provider_name):
                    await self._add_middleware(web3)
                    return web3

            except Exception as e:
                logger.warning(f"{provider_name} connection failed: {e}")
                continue

        return None

    def _get_providers(self) -> List[Tuple[str, Any]]:
        """
        Retrieve a list of available Web3 providers based on the configuration.

        Returns:
            List[Tuple[str, Any]]: A list of tuples containing provider names and their corresponding instances.
        """
        providers: List[Tuple[str, Any]] = []
        if self.configuration.IPC_ENDPOINT and os.path.exists(self.configuration.IPC_ENDPOINT):
            providers.append(("IPC", AsyncIPCProvider(self.configuration.IPC_ENDPOINT)))
        if self.configuration.HTTP_ENDPOINT:
            providers.append(("HTTP", AsyncHTTPProvider(self.configuration.HTTP_ENDPOINT)))
        if self.configuration.WEBSOCKET_ENDPOINT:
            providers.append(("WebSocket", WebSocketProvider(self.configuration.WEBSOCKET_ENDPOINT)))
        return providers

    async def _test_connection(self, web3: AsyncWeb3, name: str) -> bool:
        """
        Test the Web3 connection with a specified provider, retrying up to 3 times.

        Args:
            web3 (AsyncWeb3): The AsyncWeb3 instance to test.
            name (str): The name of the provider for logging purposes.

        Returns:
            bool: True if the connection was successful, else False.
        """
        for attempt in range(3):
            try:
                if await web3.is_connected():
                    chain_id = await web3.eth.chain_id
                    logger.info(f"Connected to network {name} (Chain ID: {chain_id})")
                    return True
            except Exception as e:
                logger.error(f"Connection attempt {attempt + 1} to {name} failed: {e}")
                await asyncio.sleep(1)  # Wait before retrying
        return False

    async def _add_middleware(self, web3: AsyncWeb3) -> None:
        """
        Add appropriate middleware to the Web3 instance based on the network's chain ID.

        Args:
            web3 (AsyncWeb3): The Web3 instance to configure.

        Raises:
            Exception: If middleware configuration fails.
        """
        try:
            chain_id = await web3.eth.chain_id
            if chain_id in {99, 100, 77, 7766, 56}:  # POA networks
                web3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
                logger.debug("Injected POA middleware.")
            elif chain_id in {1}:  # Ethereum network
                web3.middleware_onion.add(SignAndSendRawMiddlewareBuilder.build(self.account))
                logger.info("Injected middleware for ETH network.")
            elif chain_id in {61}: # ETH Classic network
                web3.middleware_onion.add(SignAndSendRawMiddlewareBuilder.build(self.account))
                logger.debug("Injected middleware for ETH Classic network.")
            elif chain_id in {56, 97, 42, 80001}: # Binance Smart Chain networks
                web3.middleware_onion.add(SignAndSendRawMiddlewareBuilder.build(self.account))
                logger.debug("Injected middleware for BSC network.")         
            
            else:
                logger.warning("Unknown network; no middleware injected.")
        except Exception as e:
            logger.error(f"Middleware configuration failed: {e}")
            raise

    async def _check_account_balance(self) -> None:
        """
        Check the Ethereum account balance to ensure it has sufficient funds for operations.

        Raises:
            Exception: If balance check fails or account is not initialized.
        """
        try:
            if not self.account:
                raise ValueError("Account not initialized")

            balance = await self.web3.eth.get_balance(self.account.address)
            balance_eth = self.web3.from_wei(balance, 'ether')

            logger.info(f"Account {self.account.address} initialized with balance: {balance_eth:.4f} ETH")

            if balance_eth < 0.001:
                logger.warning("Low account balance (<0.001 ETH)")
        except Exception as e:
            logger.error(f"Balance check failed: {e}")
            raise

    async def _initialize_components(self) -> None:
        """
        Initialize all bot components including API configurations, NonceCore,
        SafetyNet, MarketMonitor, MempoolMonitor, TransactionCore, and StrategyNet.

        Raises:
            Exception: If any component initialization fails.
        """
        try:
            # Initialize API Config
            self.apiconfig = APIConfig(self.configuration)

            # Initialize Nonce Core to manage transaction nonces
            self.noncecore = NonceCore(
                self.web3, self.account.address, self.configuration
            )
            await self.noncecore.initialize()

            # Initialize Safety Net to provide safety mechanisms
            self.safetynet = SafetyNet(
                self.web3, self.configuration, self.account, self.apiconfig
            )

            # Load contract ABIs required for interacting with smart contracts
            erc20_abi = await self._load_abi(self.configuration.ERC20_ABI)
            aave_flashloan_abi = await self._load_abi(self.configuration.AAVE_FLASHLOAN_ABI_PATH)
            aave_lending_pool_abi = await self._load_abi(self.configuration.AAVE_LENDING_POOL_ABI_PATH)

            # Initialize Market Monitor for market data analysis and model training
            self.marketmonitor = MarketMonitor(
                self.web3, self.configuration, self.apiconfig
            )

            # Start periodic training for each monitored token
            tokens_to_monitor = await self.configuration.get_token_addresses()
            for token_address in tokens_to_monitor:
                token_symbol = await self.apiconfig.get_token_symbol(self.web3, token_address)
                if token_symbol:
                    await self.marketmonitor.start_periodic_training(token_symbol)

            # Initialize Mempool Monitor to monitor pending transactions
            self.mempoolmonitor = MempoolMonitor(
                web3=self.web3,
                safetynet=self.safetynet,
                noncecore=self.noncecore,
                apiconfig=self.apiconfig,
                monitored_tokens=tokens_to_monitor,
                erc20_abi=erc20_abi,
                configuration=self.configuration
            )

            # Initialize Transaction Core to handle transaction executions
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

            # Initialize Strategy Net to manage and execute strategies
            self.strategynet = StrategyNet(
                transactioncore=self.transactioncore,
                marketmonitor=self.marketmonitor,
                safetynet=self.safetynet,
                apiconfig=self.apiconfig,
                configuration=self.configuration,
            )

            logger.info("All sub-components initialized successfully.")
        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            raise

    async def run(self) -> None:
        """
        The main execution loop of the MEV bot, which continuously processes
        profitable transactions from the mempool and executes appropriate strategies.

        Incorporates improved error handling to ensure robustness.
        """
        logger.debug("Starting MainCore...")

        try:
            # Ensure all required components are initialized
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

            # Start monitoring the mempool for transactions
            await self.mempoolmonitor.start_monitoring()

            # Enter the main loop to process transactions
            while True:
                try:
                    await self._process_profitable_transactions()
                    await asyncio.sleep(1)  # Short sleep to yield control
                except asyncio.CancelledError:
                    logger.info("Received cancellation signal")
                    break
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    await asyncio.sleep(5)  # Back off on error to prevent rapid retries

        except KeyboardInterrupt:
            logger.warning("Received shutdown signal...")
        except Exception as e:
            logger.error(f"Critical error in main loop: {e}")
        finally:
            # Ensure graceful shutdown of all components
            await self.stop()

    async def stop(self) -> None:
        """
        Gracefully shutdown all components of the MEV bot, ensuring that
        all tasks are properly terminated and resources are released.
        """
        logger.warning("Shutting down MainCore...")

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
                # Cancel all periodic training tasks if implemented
                # Assuming you have a way to track and cancel these tasks
                pass

            if self.apiconfig:
                await self.apiconfig.close()

            if self.web3:  # Close Web3 connection
                await self.web3.provider.disconnect()

            event_loop = asyncio.get_event_loop()
            event_loop.stop() or event_loop.close()

            if exception := sys.exc_info():
                logger.error(f"Exception during shutdown: {exception}")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        finally:
            logger.warning("Shutdown Complete")
            sys.exit(0)

    async def _process_profitable_transactions(self) -> None:
        """
        Process profitable transactions from the mempool queue with enhanced validation,
        performance monitoring, and error recovery mechanisms.
        """
        monitor = self.mempoolmonitor
        strategy = self.strategynet

        while not monitor.profitable_transactions.empty():
            start_time = time.time()
            tx = None

            try:
                # Retrieve a transaction with a timeout to prevent indefinite blocking
                tx = await asyncio.wait_for(
                    monitor.profitable_transactions.get(),
                    timeout=5.0
                )

                # Validate transaction format and required fields
                if not self._validate_transaction(tx):
                    logger.warning("Invalid transaction format, skipping...")
                    continue

                tx_hash = tx.get('tx_hash', 'Unknown')[:10]
                strategy_type = self._determine_strategy_type(tx)

                # Log detailed transaction information for debugging
                logger.debug(
                    f"Processing Transaction:\n"
                    f"Hash: {tx_hash}\n"
                    f"Strategy: {strategy_type}\n"
                    f"Value: {self.web3.from_wei(tx.get('value', 0), 'ether'):.4f} ETH\n"
                    f"Gas Price: {self.web3.from_wei(tx.get('gasPrice', 0), 'gwei'):.1f} Gwei"
                )

                # Check if the transaction is still valid and pending
                if not await self._is_tx_still_valid(tx):
                    logger.debug(f"Transaction {tx_hash} is no longer valid, skipping...")
                    continue

                # Execute the selected strategy with a timeout to prevent long-running tasks
                success = await asyncio.wait_for(
                    strategy.execute_best_strategy(tx, strategy_type),
                    timeout=30.0
                )

                # Calculate execution time for metrics
                execution_time = time.time() - start_time
                # Log execution metrics for monitoring and debugging
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
                # Always mark the task as done to prevent queue blockage
                if tx:
                    monitor.profitable_transactions.task_done()

        # Log the remaining number of transactions in the queue
        logger.debug(
            f"Queue status: {monitor.profitable_transactions.qsize()} transactions remaining"
        )

    def _validate_transaction(self, tx: Dict[str, Any]) -> bool:
        """
        Validate the format and required fields of a transaction.

        Args:
            tx (Dict[str, Any]): The transaction dictionary.

        Returns:
            bool: True if the transaction is valid, else False.
        """
        required_fields = ['tx_hash', 'value', 'gasPrice', 'to']
        is_valid = (
            isinstance(tx, dict)
            and all(field in tx for field in required_fields)
            and all(tx[field] is not None for field in required_fields)
        )
        if not is_valid:
            logger.debug(f"Transaction validation failed for tx: {tx}")
        return is_valid

    def _determine_strategy_type(self, tx: Dict[str, Any]) -> str:
        """
        Determine the appropriate strategy type based on transaction properties.

        Args:
            tx (Dict[str, Any]): The transaction dictionary.

        Returns:
            str: The determined strategy type.
        """
        if tx.get('value', 0) > 0:
            return 'eth_transaction'
        elif self._is_token_swap(tx):
            if tx.get('gasPrice', 0) > self.web3.to_wei(200, 'gwei'):
                return 'back_run'
            else:
                return 'front_run'
        return 'sandwich_attack'

    async def _is_tx_still_valid(self, tx: Dict[str, Any]) -> bool:
        """
        Check if a transaction is still valid and pending on the network.

        Args:
            tx (Dict[str, Any]): The transaction dictionary.

        Returns:
            bool: True if the transaction is still valid and pending, else False.
        """
        try:
            tx_hash = tx.get('tx_hash')
            if not tx_hash:
                return False
            tx_status = await self.web3.eth.get_transaction(tx_hash)
            return tx_status is not None and tx_status.block_number is None
        except Exception:
            return False

    def _is_token_swap(self, tx: Dict[str, Any]) -> bool:
        """
        Determine if a transaction is a token swap based on its input data.

        Args:
            tx (Dict[str, Any]): The transaction dictionary.

        Returns:
            bool: True if the transaction is a token swap, else False.
        """
        return (
            len(tx.get('input', '0x')) > 10
            and tx.get('value', 0) == 0
        )

    def _log_execution_metrics(self, tx_hash: str, success: bool, execution_time: float) -> None:
        """
        Log detailed execution metrics for a strategy execution.

        Args:
            tx_hash (str): The hash of the transaction.
            success (bool): Whether the strategy execution was successful.
            execution_time (float): Time taken to execute the strategy.
        """
        try:
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # Convert to MB
        except ImportError:
            memory_usage = 0.0  # psutil not available

        logger.debug(
            f"Execution Metrics for {tx_hash}:\n"
            f"Success: {success}\n"
            f"Execution Time: {execution_time:.2f}s\n"
            f"Memory Usage: {memory_usage:.1f}MB"
        )

    async def _load_abi(self, abi_path: str) -> List[Dict[str, Any]]:
        """
        Load a contract's ABI from a specified file path.

        Args:
            abi_path (str): The file path to the ABI JSON file.

        Returns:
            List[Dict[str, Any]]: The loaded ABI.

        Raises:
            Exception: If loading the ABI fails.
        """
        try:
            with open(abi_path, 'r') as file:
                abi = json.load(file)
            logger.debug(f"Loaded ABI from {abi_path} successfully.")
            return abi
        except Exception as e:
            logger.warning(f"Failed to load ABI from {abi_path}: {e}")
            raise

async def main():
    """
    The main entry point of the MEV bot application.
    Initializes configuration, sets up MainCore, and starts the bot.
    Handles graceful shutdown and critical error logging.
    """
    global logger
    try:
        # Initialize configuration settings
        configuration = Configuration()
        await configuration.load()

        # Initialize and run the MEV bot
        main_core = MainCore(configuration)
        await main_core.initialize()
        await main_core.run()

    except KeyboardInterrupt:
        logger.debug("Shutdown complete.")
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        sys.exit(1)

def run_standard():
    """
    Run the MEV bot in standard (command-line) mode.
    """
    asyncio.run(main())

def run_streamlit():
    """
    Run the MEV bot with a Streamlit GUI.
    """
    import plotly.express as px  # For interactive plots
    import streamlit as st
    import threading
    import queue
    import psutil

    # Initialize Streamlit page
    st.set_page_config(page_title="MEV Bot Control Panel", layout="wide")
    st.title("MEV Bot Control Panel")

    # Initialize a queue to hold log messages
    log_queue = queue.Queue()

    # Add StreamlitHandler to the logger
    stream_handler = StreamlitHandler(log_queue)
    stream_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Placeholder for logs
    log_placeholder = st.empty()

    # Placeholder for metrics and graphs
    metrics_placeholder = st.empty()
    graph_placeholder = st.empty()
    system_placeholder = st.empty()

    # Initialize session state variables
    if 'bot_running' not in st.session_state:
        st.session_state.bot_running = False
    if 'main_core' not in st.session_state:
        st.session_state.main_core = None
    if 'thread' not in st.session_state:
        st.session_state.thread = None
    if 'history_data' not in st.session_state:
        st.session_state.history_data = []
    if 'metrics' not in st.session_state:
        st.session_state.metrics = {
            "current_profit": 0.0,
            "total_profit": 0.0,
            "account_balance": 0.0,
            "strategy_performance": {}
        }

    # Sidebar for Configuration Inputs
    with st.sidebar:
        st.header("Configuration Settings")

        # Example configuration inputs
        wallet_key = st.text_input("Wallet Key", type="password")
        gas_price_multiplier = st.slider("Gas Price Multiplier", 1.0, 2.0, 1.1, 0.1)
        min_profit_threshold = st.number_input("Minimum Profit Threshold (ETH)", value=0.01, step=0.001)

        # Button to apply configuration
        if st.button("Apply Configuration"):
            if st.session_state.main_core:
                st.session_state.main_core.configuration.WALLET_KEY = wallet_key
                st.session_state.main_core.configuration.gas_price_multiplier = gas_price_multiplier
                st.session_state.main_core.configuration.min_profit_threshold = Decimal(min_profit_threshold)
                logger.info("Configuration updated successfully!")
            else:
                logger.warning("Bot is not running. Start the bot to apply configurations.")

    # Function to continuously update logs
    def update_logs():
        logs = ""
        while not log_queue.empty():
            msg = log_queue.get_nowait()
            logs += msg + "\n"
        if logs:
            existing_logs = st.session_state.get('logs', "")
            st.session_state.logs = existing_logs + logs
            log_placeholder.text_area("Logs", value=st.session_state.logs, height=300, max_chars=None, key="logs", disabled=True)

    # Function to periodically update metrics and graphs
    def update_metrics_and_graphs():
        # Update metrics
        metrics = st.session_state.metrics
        metrics_placeholder.markdown("### Current Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Profit (ETH)", f"{metrics['current_profit']:.4f}")
        col2.metric("Total Profit (ETH)", f"{metrics['total_profit']:.4f}")
        col3.metric("Account Balance (ETH)", f"{metrics['account_balance']:.4f}")

        # Update strategy performance metrics
        st.subheader("Strategy Performance")
        strategy_data = metrics.get("strategy_performance", {})
        if strategy_data:
            df = pd.DataFrame(strategy_data).T
            df = df.reset_index().rename(columns={"index": "Strategy"})
            st.dataframe(df, height=300)

            # Plotting Success Rate using Plotly
            fig1 = px.bar(df, x='Strategy', y='success_rate', title='Success Rate per Strategy',
                          labels={'success_rate': 'Success Rate'}, range_y=[0,1])
            graph_placeholder.plotly_chart(fig1, use_container_width=True)

            # Plotting Average Execution Time using Plotly
            fig2 = px.bar(df, x='Strategy', y='avg_execution_time', title='Average Execution Time (s) per Strategy',
                          labels={'avg_execution_time': 'Avg Execution Time (s)'}, range_y=[0, df['avg_execution_time'].max()*1.1])
            graph_placeholder.plotly_chart(fig2, use_container_width=True)

            # Plotting Total Profit using Plotly
            fig3 = px.bar(df, x='Strategy', y='profit', title='Total Profit (ETH) per Strategy',
                          labels={'profit': 'Total Profit (ETH)'}, range_y=[0, df['profit'].max()*1.1])
            graph_placeholder.plotly_chart(fig3, use_container_width=True)
        else:
            st.write("No strategy performance data available.")

    # Function to monitor system resources
    def update_system_metrics():
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        system_placeholder.markdown("### System Metrics")
        col1, col2 = system_placeholder.columns(2)
        col1.metric("CPU Usage (%)", f"{cpu_percent}%")
        col2.metric("Memory Usage (%)", f"{memory.percent}%")

    # Function to run the bot in a separate thread
    def start_bot():
        async def bot_runner():
            try:
                # Initialize configuration settings
                configuration = Configuration()
                await configuration.load()

                # Initialize and run the MEV bot
                main_core = MainCore(configuration)
                st.session_state.main_core = main_core
                await main_core.initialize()
                await main_core.run()
            except Exception as e:
                logger.critical(f"Fatal error: {e}")
                st.session_state.bot_running = False

        asyncio.run(bot_runner())

    # Start Bot Button
    if not st.session_state.bot_running:
        if st.button("Start Bot"):
            st.session_state.bot_running = True
            # Start the bot in a new thread to prevent blocking
            st.session_state.thread = threading.Thread(target=start_bot, daemon=True)
            st.session_state.thread.start()
            logger.info("Bot started successfully!")
    else:
        if st.button("Stop Bot"):
            if st.session_state.main_core:
                # Define a function to stop the bot asynchronously
                def stop_bot():
                    asyncio.run(st.session_state.main_core.stop())
                    st.session_state.bot_running = False
                    logger.info("Bot stopped successfully!")

                # Start the stop function in a new thread to prevent blocking
                stop_thread = threading.Thread(target=stop_bot, daemon=True)
                stop_thread.start()
            else:
                logger.warning("Bot core not initialized.")

    # Display Bot Status
    st.subheader("Bot Status")
    status = "🟢 Running" if st.session_state.bot_running else "🔴 Stopped"
    st.write(f"**Status:** {status}")

    # Display Logs
    st.subheader("Logs")
    update_logs()

    # Display Metrics and Graphs
    st.subheader("Performance Metrics")
    update_metrics_and_graphs()

    # Display System Metrics
    st.subheader("System Metrics")
    update_system_metrics()

    # Auto-refresh logs and metrics every second using Streamlit's experimental features
    if st.session_state.bot_running:
        # Use Streamlit's experimental function to rerun the script after a delay
        st.experimental_rerun()

class StreamlitHandler(logging.Handler):
    """
    Custom logging handler to capture logs and send them to Streamlit.
    """

    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record: logging.LogRecord) -> None:
        log_entry = self.format(record)
        self.log_queue.put(log_entry)

if __name__ == "__main__":
    if 'streamlit' in sys.argv[0]:
        run_streamlit()
    else:
        run_standard()