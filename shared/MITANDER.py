import json
import os
import asyncio
import aiofiles
import aiohttp
import dotenv
import time
import sys
import tracemalloc
import async_timeout
import hexbytes
import scheduling
import random
import joblib
import logging
import pandas as pd
import numpy as np

# Data processing and machine learning libraries
from sklearn.linear_model import LinearRegression
from colorama import Fore, Style, init
from io import StringIO
from cachetools import TTLCache

# Type hints, dataclasses, and decorators
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

# Web3 and Ethereum imports
from web3 import AsyncWeb3, AsyncIPCProvider, AsyncHTTPProvider, WebSocketProvider
from web3.eth import AsyncEth
from web3.exceptions import TransactionNotFound, ContractLogicError, Web3ValueError
from web3.middleware import ExtraDataToPOAMiddleware

from eth_account import Account
from eth_utils import function_signature_to_4byte_selector

# Initialize colorama
init(autoreset=True)

class CustomFormatter(logging.Formatter):
    """Custom logging formatter with colors."""
    LEVEL_COLORS = {
        logging.DEBUG: f"{Fore.MAGENTA}",    # Blue
        logging.INFO: f"{Fore.GREEN}",     # Green
        logging.WARNING: f"{Fore.YELLOW}",  # Yellow
        logging.ERROR: f"{Fore.RED}",    # Red
        logging.CRITICAL: f"{Fore.RED}", # Bold Red
        "RESET": "\033[0m",     # Reset
    }

    COLORS = {
        "RESET": "\033[0m",
        "RED": "\033[31m",
        "GREEN": "\033[32m",
        "YELLOW": "\033[33m",
        "MAGENTA": "\033[35m",

    }

    def format(self, record: logging.LogRecord) -> str:
        """Formats a log record with colors."""
        color = self.LEVEL_COLORS.get(record.levelname, self.COLORS["RESET"])
        reset = self.COLORS["RESET"]
        record.levelname = f"{color}{record.levelname}{reset}"  # Colorize level name
        record.msg = f"{color}{record.msg}{reset}"              # Colorize message
        return super().format(record)

# Configure the logging once
def configure_logging(level: int = logging.DEBUG) -> None:
    """Configures logging with a colored formatter."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(CustomFormatter("%(asctime)s [%(levelname)s] %(message)s"))
    handler.stream.reconfigure(encoding='utf-8') # explicitly specify UTF-8 encoding
    logging.basicConfig(
        level=level,  # Global logging level
        handlers=[handler]
    )

# Factory function to get a logger instance
def getLogger(name: Optional[str] = None, level: int = logging.DEBUG) -> logging.Logger:
    """Returns a logger instance, configuring logging if it hasn't been yet."""
    if not logging.getLogger().hasHandlers():
        configure_logging(level)
        
    logger = logging.getLogger(name if name else "0xBuilder")
    return logger



dotenv.load_dotenv() 

logger = getLogger("0xBuilder")

# Configuration Constants
MIN_PROFIT_THRESHOLD: Decimal = Decimal("0.01")
DEFAULT_GAS_THRESHOLD: int = 200
STRATEGY_SCORE_THRESHOLD: int = 75
BULLISH_THRESHOLD: float = 0.02
HIGH_VOLUME_DEFAULT: int = 500_000  # USD

CACHE_SETTINGS: Dict[str, Dict[str, int]] = {
    'price': {'ttl': 300, 'size': 1000},
    'volume': {'ttl': 900, 'size': 500},
    'volatility': {'ttl': 600, 'size': 200}
}

# Add risk thresholds
RISK_THRESHOLDS: Dict[str, Union[int, float]] = {
    'gas_price': 500,  # Gwei
    'min_profit': 0.01,  # ETH
    'max_slippage': 0.05,  # 5%
    'congestion': 0.8  # 80%
}

# Error codes
ERROR_MARKET_MONITOR_INIT: int = 1001
ERROR_MODEL_LOAD: int = 1002
ERROR_DATA_LOAD: int = 1003
ERROR_MODEL_TRAIN: int = 1004
ERROR_CORE_INIT: int = 1005
ERROR_WEB3_INIT: int = 1006
ERROR_CONFIG_LOAD: int = 1007
ERROR_STRATEGY_EXEC: int = 1008

# Error messages with default fallbacks
ERROR_MESSAGES: Dict[int, str] = {
    ERROR_MARKET_MONITOR_INIT: "Market Monitor initialization failed",
    ERROR_MODEL_LOAD: "Failed to load price prediction model",
    ERROR_DATA_LOAD: "Failed to load historical training data",
    ERROR_MODEL_TRAIN: "Failed to train price prediction model",
    ERROR_CORE_INIT: "Core initialization failed",
    ERROR_WEB3_INIT: "Web3 connection failed",
    ERROR_CONFIG_LOAD: "Configuration loading failed",
    ERROR_STRATEGY_EXEC: "Strategy execution failed",
}

# Add a helper function to get error message with fallback
def get_error_message(code: int, default: str = "Unknown error") -> str:
    """Get error message for error code with fallback to default message."""
    return ERROR_MESSAGES.get(code, default)

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiofiles
from eth_utils import function_signature_to_4byte_selector

# Initialize logging (ensure this is done before any logging is performed)
logger = logging.getLogger("0xBuilder")

class ABI_Registry:
    """Centralized ABI registry with validation and signature mapping."""

    REQUIRED_METHODS = {
        'erc20': {'transfer', 'approve', 'transferFrom', 'balanceOf'},
        'uniswap': {'swapExactTokensForTokens', 'swapTokensForExactTokens', 'addLiquidity', 'getAmountsOut'},
        'sushiswap': {'swapExactTokensForTokens', 'swapTokensForExactTokens', 'addLiquidity', 'getAmountsOut'},
        'pancakeswap': {'swapExactTokensForTokens', 'swapTokensForExactTokens', 'addLiquidity', 'getAmountsOut'},
        'balancer': {'swap', 'addLiquidity'},
        'aave_flashloan': {'fn_RequestFlashLoan', 'executeOperation', 'ADDRESSES_PROVIDER', 'POOL'},
        'aave': {'ADDRESSES_PROVIDER', 'getReservesList', 'getReserveData'}
    }

    def __init__(self):
        self.abis: Dict[str, List[Dict]] = {}
        self.signatures: Dict[str, Dict[str, str]] = {}
        self.method_selectors: Dict[str, Dict[str, str]] = {}
        self._initialized: bool = False

    async def initialize(self) -> None:
        """Async method to load and validate all ABIs at initialization."""
        if self._initialized:
            logger.debug("ABI_Registry already initialized.")
            return
        await self._load_all_abis()
        self._initialized = True
        logger.debug("ABI_Registry initialization complete.")

    async def _load_all_abis(self) -> None:
        """Load and validate all ABIs at initialization."""
        abi_dir = Path(__file__).parent.parent / 'abi'

        abi_files = {
            'erc20': 'erc20_abi.json',
            'uniswap': 'uniswap_abi.json',
            'sushiswap': 'sushiswap_router_abi.json',
            'pancakeswap': 'pancakeswap_router_abi.json',
            'balancer': 'balancer_router_abi.json',
            'aave_flashloan': 'aave_flashloan.abi.json',
            'aave': 'aave_pool_abi.json'
        }

        # Define critical ABIs that are essential for the application
        critical_abis = {'erc20', 'uniswap'}

        for abi_type, filename in abi_files.items():
            abi_path = abi_dir / filename
            try:
                abi = await self._load_abi_from_path(abi_path, abi_type)
                self.abis[abi_type] = abi
                self._extract_signatures(abi, abi_type)
                logger.debug(f"Loaded and validated {abi_type} ABI from {abi_path}")
            except FileNotFoundError:
                logger.error(f"ABI file not found for {abi_type}: {abi_path}")
                if abi_type in critical_abis:
                    raise
                else:
                    logger.warning(f"Skipping non-critical ABI: {abi_type}")
            except ValueError as ve:
                logger.error(f"Validation failed for {abi_type} ABI: {ve}")
                if abi_type in critical_abis:
                    raise
                else:
                    logger.warning(f"Skipping non-critical ABI: {abi_type}")
            except json.JSONDecodeError as je:
                logger.error(f"JSON decode error for {abi_type} ABI: {je}")
                if abi_type in critical_abis:
                    raise
                else:
                    logger.warning(f"Skipping non-critical ABI: {abi_type}")
            except Exception as e:
                logger.error(f"Unexpected error loading {abi_type} ABI: {e}")
                if abi_type in critical_abis:
                    raise
                else:
                    logger.warning(f"Skipping non-critical ABI: {abi_type}")

    async def _load_abi_from_path(self, abi_path: Path, abi_type: str) -> List[Dict]:
        """Loads and validates the ABI from a given path."""
        try:
            if not abi_path.exists():
                logger.error(f"ABI file not found: {abi_path}")
                raise FileNotFoundError(f"ABI file not found: {abi_path}")

            async with aiofiles.open(abi_path, 'r', encoding='utf-8') as f:
                abi_content = await f.read()
                abi = json.loads(abi_content)
                logger.debug(f"ABI content loaded from {abi_path}")

            if not self._validate_abi(abi, abi_type):
                raise ValueError(f"Validation failed for {abi_type} ABI from file {abi_path}")

            return abi
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for {abi_type} in file {abi_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading ABI {abi_type}: {e}")
            raise

    async def load_abi(self, abi_type: str) -> Optional[List[Dict]]:
        """Load specific ABI type with validation."""
        try:
            abi_dir = Path(__file__).parent.parent / 'abi'
            abi_files = {
                'erc20': 'erc20_abi.json',
                'uniswap': 'uniswap_abi.json',
                'sushiswap': 'sushiswap_router_abi.json',
                'pancakeswap': 'pancakeswap_router_abi.json',
                'balancer': 'balancer_router_abi.json',
                'aave_flashloan': 'aave_flashloan.abi.json',
                'aave': 'aave_pool_abi.json'
            }
            
            if abi_type not in abi_files:
                logger.error(f"Unknown ABI type: {abi_type}")
                return None

            abi_path = abi_dir / abi_files[abi_type]
            abi = await self._load_abi_from_path(abi_path, abi_type)
            self.abis[abi_type] = abi
            self._extract_signatures(abi, abi_type)
            logger.debug(f"Loaded and validated {abi_type} ABI")
            return abi
        except Exception as e:
            logger.error(f"Error loading ABI {abi_type}: {e}")
            return None

    def _validate_abi(self, abi: List[Dict], abi_type: str) -> bool:
        """Validate ABI structure and required methods."""
        if not isinstance(abi, list):
            logger.error(f"Invalid ABI format for {abi_type}")
            return False

        found_methods = {
            item.get('name') for item in abi
            if item.get('type') == 'function' and 'name' in item
        }

        required = self.REQUIRED_METHODS.get(abi_type, set())
        if not required.issubset(found_methods):
            missing = required - found_methods
            logger.error(f"Missing required methods in {abi_type} ABI: {missing}")
            return False

        return True

    def _extract_signatures(self, abi: List[Dict], abi_type: str) -> None:
        """Extract function signatures and method selectors."""
        signatures = {}
        selectors = {}

        for item in abi:
            if item.get('type') == 'function':
                name = item.get('name')
                if name:
                    # Create function signature
                    inputs = ','.join(inp.get('type', '') for inp in item.get('inputs', []))
                    signature = f"{name}({inputs})"

                    # Generate selector
                    selector = function_signature_to_4byte_selector(signature)
                    hex_selector = selector.hex()

                    signatures[name] = signature
                    selectors[hex_selector] = name

        self.signatures[abi_type] = signatures
        self.method_selectors[abi_type] = selectors

    def get_abi(self, abi_type: str) -> Optional[List[Dict]]:
        """Get validated ABI by type."""
        return self.abis.get(abi_type)

    def get_method_selector(self, selector: str) -> Optional[str]:
        """Get method name from selector, checking all ABIs."""
        for abi_type, selectors in self.method_selectors.items():
            if selector in selectors:
                return selectors[selector]
        return None

    def get_function_signature(self, abi_type: str, method_name: str) -> Optional[str]:
        """Get function signature by ABI type and method name."""
        return self.signatures.get(abi_type, {}).get(method_name)


class Configuration:
    """
    Loads configuration from environment variables and monitored tokens from a JSON file.
    """

    def __init__(self):
        """Initialize configuration attributes with None values."""
        self.IPC_ENDPOINT: Optional[str] = None
        self.HTTP_ENDPOINT: Optional[str] = None
        self.WEBSOCKET_ENDPOINT: Optional[str] = None
        self.WALLET_KEY: Optional[str] = None
        self.WALLET_ADDRESS: Optional[str] = None
        self.ETHERSCAN_API_KEY: Optional[str] = None
        self.INFURA_PROJECT_ID: Optional[str] = None
        self.COINGECKO_API_KEY: Optional[str] = None
        self.COINMARKETCAP_API_KEY: Optional[str] = None
        self.CRYPTOCOMPARE_API_KEY: Optional[str] = None
        self.AAVE_POOL_ADDRESS: Optional[str] = None
        self.TOKEN_ADDRESSES: Optional[List[str]] = None
        self.TOKEN_SYMBOLS: Optional[Dict[str, str]] = None
        self.ERC20_ABI: Optional[str] = None
        self.ERC20_SIGNATURES: Optional[Dict[str, str]] = None
        self.SUSHISWAP_ABI: Optional[str] = None
        self.SUSHISWAP_ADDRESS: Optional[str] = None
        self.UNISWAP_ABI: Optional[str] = None
        self.UNISWAP_ADDRESS: Optional[str] = None
        self.AAVE_FLASHLOAN_ADDRESS: Optional[str] = None
        self.AAVE_FLASHLOAN_ABI: Optional[Any] = None
        self.AAVE_POOL_ABI: Optional[Any] = None
        self.AAVE_POOL_ADDRESS: Optional[str] = None
        
        
        # Add ML model configuration
        self.MODEL_RETRAINING_INTERVAL: int = 3600  # 1 hour
        self.MIN_TRAINING_SAMPLES: int = 100
        self.MODEL_ACCURACY_THRESHOLD: float = 0.7
        self.PREDICTION_CACHE_TTL: int = 300  # 5 minutes
        
        # Add default config values for strategies
        self.SLIPPAGE_DEFAULT: float = 0.1
        self.SLIPPAGE_MIN: float = 0.01
        self.SLIPPAGE_MAX: float = 0.5
        self.SLIPPAGE_HIGH_CONGESTION: float = 0.05
        self.SLIPPAGE_LOW_CONGESTION: float = 0.2
        self.MAX_GAS_PRICE_GWEI: int = 500
        self.MIN_PROFIT_MULTIPLIER: float = 2.0
        self.BASE_GAS_LIMIT: int = 21000
        self.LINEAR_REGRESSION_PATH: str = "/linear_regression"
        self.MODEL_PATH: str = "/linear_regression/price_model.joblib"
        self.TRAINING_DATA_PATH: str = "/linear_regression/training_data.csv"


        self.abi_registry = ABI_Registry()

        # Add WETH and USDC addresses
        self.WETH_ADDRESS: str = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"  # Mainnet WETH
        self.USDC_ADDRESS: str = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"  # Mainnet USDC
        self.USDT_ADDRESS: str = "0xdAC17F958D2ee523a2206206994597C13D831ec7"  # Mainnet USDT


    async def load(self) -> None:
        """Loads the configuration in the correct order."""
        try:
            logger.info("Loading configuration... ⏳")
            await asyncio.sleep(1)  # ensuring proper initialization

            # Initialize ABI Registry
            await self.abi_registry.initialize()

            # Proceed with configuration loading
            await self._load_configuration()
            logger.info("System reporting go for launch ✅...")
            await asyncio.sleep(3)  # ensuring proper initialization

            logger.debug("All Configurations and Environment Variables Loaded Successfully ✅")
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
            self.AAVE_POOL_ADDRESS = self._get_env_variable("AAVE_POOL_ADDRESS")
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
            self.SUSHISWAP_ABI = await self._construct_abi_path("abi", "sushiswap_abi.json")
            self.SUSHISWAP_ADDRESS = self._get_env_variable("SUSHISWAP_ADDRESS")
            self.UNISWAP_ABI = await self._construct_abi_path("abi", "uniswap_abi.json")
            self.UNISWAP_ADDRESS = self._get_env_variable("UNISWAP_ADDRESS")
            self.AAVE_FLASHLOAN_ABI = await self._load_json_file(
                await self._construct_abi_path("abi", "aave_flashloan_abi.json"),
                "Aave Flashloan ABI"
            )
            self.AAVE_POOL_ABI = await self._load_json_file(
               await self._construct_abi_path("abi", "aave_pool_abi.json"),
                "Aave Lending Pool ABI"
            )
            self.AAVE_FLASHLOAN_ADDRESS = self._get_env_variable("AAVE_FLASHLOAN_ADDRESS")
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
            async with aiofiles.open(file_path, 'r') as file:
                data = json.loads(await file.read())
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
            "sushiswap_abi": self.SUSHISWAP_ABI,
            "uniswap_abi": self.UNISWAP_ABI,
            "AAVE_FLASHLOAN_ABI": self.AAVE_FLASHLOAN_ABI,
            "AAVE_POOL_ABI": self.AAVE_POOL_ABI,
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
    """
    Manages interactions with various external APIs for price and market data.
    """
    
    MAX_REQUEST_ATTEMPTS: int = 5
    REQUEST_BACKOFF_FACTOR: float = 1.5


    def __init__(self, configuration: Optional["Configuration"] = None):
        self.configuration: Optional["Configuration"] = configuration
        self.session: Optional[aiohttp.ClientSession] = None
        self.price_cache: TTLCache = TTLCache(maxsize=2000, ttl=300)  # 5 min cache for prices
        self.volume_cache: TTLCache = TTLCache(maxsize=1000, ttl=900) # 15 min cache for volumes
        self.market_data_cache: TTLCache = TTLCache(maxsize=1000, ttl=1800)  # 30 min cache for market data
        self.token_metadata_cache: TTLCache = TTLCache(maxsize=500, ttl=86400)  # 24h cache for metadata
        
        # Add rate limit tracking
        self.rate_limit_counters: Dict[str, Dict[str, Any]] = {
            "coingecko": {"count": 0, "reset_time": time.time(), "limit": 50},
            "coinmarketcap": {"count": 0, "reset_time": time.time(), "limit": 330},
            "cryptocompare": {"count": 0, "reset_time": time.time(), "limit": 80},
            "binance": {"count": 0, "reset_time": time.time(), "limit": 1200},
        }
        
        # Add priority queues for data fetching
        self.high_priority_tokens: set[str] = set()  # Tokens currently being traded
        self.update_intervals: Dict[str, int] = {
            'price': 30,  # Seconds
            'volume': 300,  # 5 minutes
            'market_data': 1800,  # 30 minutes
            'metadata': 86400  # 24 hours
        }

        # Initialize API lock and session
        self.api_lock: asyncio.Lock = asyncio.Lock()
        self.session: Optional[aiohttp.ClientSession] = None

        # Initialize API configurations
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
                "api_key": configuration.COINGECKO_API_KEY if configuration else None,
                "success_rate": 1.0,
                "weight": 0.8,
                "rate_limit": 50,
            },
            "coinmarketcap": {
                "base_url": "https://pro-api.coinmarketcap.com/v1",
                "ticker_url": "/cryptocurrency/quotes/latest",
                "api_key": configuration.COINMARKETCAP_API_KEY if configuration else None,
                "success_rate": 1.0,
                "weight": 0.7,
                "rate_limit": 333,
            },
            "cryptocompare": {
                "base_url": "https://min-api.cryptocompare.com/data",
                "price_url": "/price",
                "api_key": configuration.CRYPTOCOMPARE_API_KEY if configuration else None,
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
            logger.debug("APIconfig session closed.")

    async def get_token_symbol(self, web3: AsyncWeb3, token_address: str) -> Optional[str]:
        """Get the token symbol for a given token address."""
        if token_address in self.token_metadata_cache:
            metadata = self.token_metadata_cache[token_address]
            return metadata.get('symbol')
        if token_address in self.configuration.TOKEN_SYMBOLS:
            symbol = self.configuration.TOKEN_SYMBOLS[token_address]
            self.token_metadata_cache[token_address] = {'symbol':symbol}
            return symbol
        try:
            erc20_abi = await self._load_abi(self.configuration.ERC20_ABI)
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
        """Fetch token metadata with improved error handling."""
        config = self.api_configs.get(source)
        if not config:
            return None

        try:
            if source == "coingecko":
                # Handle both token address and symbol
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
                # Use symbol mapping with fallback
                symbols = await self._get_trading_pairs(token)
                if not symbols:
                    return None

                # Try each symbol pair
                for symbol in symbols:
                    try:
                        url = f"{config['base_url']}/ticker/24hr"
                        params = {"symbol": symbol}
                        response = await self.make_request(source, url, params=params)
                        
                        if response and 'volume' in response:
                            return float(response['quoteVolume'])  # Use quote volume for USD value
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
        # Common base pairs
        quote_currencies = ["USDT", "BUSD", "USD", "ETH", "BTC"]
        
        # Symbol mappings for common tokens
        symbol_mappings = {
            "WETH": ["ETH"],
            "WBTC": ["BTC"],
            "ETH": ["ETH"],
            "BTC": ["BTC"],
            # Add more mappings as needed
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
    
    async def _load_abi(self, abi_path: str) -> List[Dict[str, Any]]:
        """Load contract abi from a file."""
        try:
            abi_registry = ABI_Registry()
            abi = await abi_registry.load_abi('erc20')
            if not abi:
                 raise ValueError("Failed to load ERC20 ABI using ABI Registry")
            return abi
        except Exception as e:
            logger.error(f"Failed to load abi from {abi_path}: {e}")
            raise

    async def close(self) -> None:
        """Close the aiohttp session."""
        if self.session:
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

    async def _get_last_update_time(self, token: str) -> int:
        """Get the timestamp of last data update for a token."""
        try:
            training_data_path = Path(__file__).parent.parent / "linear_regression" / "training_data.csv"
            if not training_data_path.exists():
                return 0
                
            df = pd.read_csv(training_data_path)
            if df.empty or 'timestamp' not in df.columns or 'symbol' not in df.columns:
                return 0
                
            token_data = df[df['symbol'] == token]
            if token_data.empty:
                return 0
                
            return int(token_data['timestamp'].max())
            
        except Exception as e:
            logger.error(f"Error getting last update time for {token}: {e}")
            return 0

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


class Nonce_Core:
    """
    Advanced nonce management system for Ethereum transactions with caching,
    auto-recovery, and comprehensive error handling.
    """

    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 1.0
    CACHE_TTL: int = 300  # Cache TTL in seconds
    TRANSACTION_TIMEOUT: int = 120  # Transaction receipt timeout in seconds

    def __init__(
        self,
        web3: AsyncWeb3,
        address: str,
        configuration: "Configuration",
    ):
        self.pending_transactions: set[int] = set()
        self.web3: AsyncWeb3 = web3
        self.configuration: "Configuration" = configuration
        self.address: str = address
        self.lock: asyncio.Lock = asyncio.Lock()
        self.nonce_cache: TTLCache = TTLCache(maxsize=1, ttl=self.CACHE_TTL)
        self.last_sync: float = time.monotonic()
        self._initialized: bool = False

    async def initialize(self) -> None:
        """Initialize the nonce manager with error recovery."""
        try:
            await self._init_nonce()
            self._initialized = True
            logger.debug("Noncecore initialized ✅")
        except Exception as e:
            logger.error(f"Failed to initialize Nonce_Core: {e}")
            raise

    async def _init_nonce(self) -> None:
         """Initialize nonce with fallback mechanisms."""
         current_nonce = await self._fetch_current_nonce_with_retries()
         pending_nonce = await self._get_pending_nonce()
          # Use the higher of current or pending nonce
         self.nonce_cache[self.address] = max(current_nonce, pending_nonce)
         self.last_sync = time.monotonic()
         logger.info(f"Initial nonce set to {self.nonce_cache[self.address]}")


    async def get_nonce(self, force_refresh: bool = False) -> int:
        """Get next available nonce with optional force refresh."""
        if not self._initialized:
            await self.initialize()

        if force_refresh or self._should_refresh_cache():
            await self.refresh_nonce()

        return self.nonce_cache.get(self.address, 0)

    async def refresh_nonce(self) -> None:
        """Refresh nonce from chain with conflict resolution."""
        async with self.lock:
            current_nonce = await self.web3.eth.get_transaction_count(self.address)
            self.nonce_cache[self.address] = current_nonce
            self.last_sync = time.monotonic()
            logger.debug(f"Nonce refreshed to {current_nonce}.")

    async def _fetch_current_nonce_with_retries(self) -> int:
        """Fetch current nonce with exponential backoff."""
        backoff = self.RETRY_DELAY
        for attempt in range(self.MAX_RETRIES):
            try:
                nonce = await self.web3.eth.get_transaction_count(self.address)
                logger.debug(f"Fetched nonce: {nonce}")
                return nonce
            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed to fetch nonce: {e}")
                await asyncio.sleep(backoff)
                backoff *= 2
        raise Web3ValueError("Failed to fetch current nonce after retries")

    async def _get_pending_nonce(self) -> int:
        """Get highest nonce from pending transactions."""
        try:
            pending = await self.web3.eth.get_transaction_count(self.address, 'pending')
            logger.info(f"NonceCore Reports pending nonce: {pending}")
            return pending
        except Exception as e:
            logger.error(f"Error fetching pending nonce: {e}")
            # Instead of retrying, raise exception for upper layer to manage it.
            raise Web3ValueError(f"Failed to fetch pending nonce: {e}")

    async def track_transaction(self, tx_hash: str, nonce: int) -> None:
        """Track pending transaction for nonce management."""
        self.pending_transactions.add(nonce)
        try:
            receipt = await self.web3.eth.wait_for_transaction_receipt(tx_hash, timeout=self.TRANSACTION_TIMEOUT)
            if receipt.status == 1:
                logger.info(f"Transaction {tx_hash} succeeded.")
            else:
                logger.error(f"Transaction {tx_hash} failed.")
        except Exception as e:
            logger.error(f"Error tracking transaction {tx_hash}: {e}")
        finally:
            self.pending_transactions.discard(nonce)

    async def _handle_nonce_error(self) -> None:
        """Handle nonce-related errors with recovery attempt."""
        logger.warning("Handling nonce-related error. Refreshing nonce.")
        await self.refresh_nonce()

    async def sync_nonce_with_chain(self) -> None:
        """Force synchronization with chain state."""
        async with self.lock:
            await self.refresh_nonce()

    async def reset(self) -> None:
        """Complete reset of nonce manager state."""
        async with self.lock:
            self.nonce_cache.clear()
            self.pending_transactions.clear()
            await self.refresh_nonce()
            logger.debug("NonceCore reset. OK ✅")

    async def stop(self) -> None:
        """Stop nonce manager operations."""
        if not self._initialized:
            return
        try:
            await self.reset()
            logger.debug("Nonce Core stopped successfully.")
        except Exception as e:
            logger.error(f"Error stopping Nonce Core: {e}")

    def _should_refresh_cache(self) -> bool:
        """Check if the nonce cache should be refreshed based on time."""
        return time.monotonic() - self.last_sync > self.CACHE_TTL

class Safety_Net:
    """
    Enhanced safety system for risk management and transaction validation.
    """

    CACHE_TTL: int = 300  # Cache TTL in seconds
    GAS_PRICE_CACHE_TTL: int = 15  # 15 sec cache for gas prices

    def __init__(
        self,
        web3: AsyncWeb3,
        configuration: Optional["Configuration"] = None,
        address: Optional[str] = None,
        account: Optional[Account] = None,
        api_config: Optional["API_Config"] = None,
        market_monitor: Optional[Any] = None,  # Add this parameter
    ):
        self.web3: AsyncWeb3 = web3
        self.address: Optional[str] = address
        self.configuration: Optional["Configuration"] = configuration
        self.account: Optional[Account] = account
        self.api_config: Optional["API_Config"] = api_config
        self.price_cache: TTLCache = TTLCache(maxsize=1000, ttl=self.CACHE_TTL)
        self.gas_price_cache: TTLCache = TTLCache(maxsize=1, ttl=self.GAS_PRICE_CACHE_TTL)
        self.market_monitor: Optional[Any] = market_monitor  # Store market_monitor reference

        self.price_lock: asyncio.Lock = asyncio.Lock()
        logger.info("SafetyNet is reporting for duty 🛡️")
        time.sleep(3) # ensuring proper initialization

        # Add safety checks cache
        self.safety_cache: TTLCache = TTLCache(maxsize=100, ttl=60)  # 1 minute cache

        # Load settings from config object
        if self.configuration:
            self.SLIPPAGE_CONFIG: Dict[str, float] = {
                "default": self.configuration.get_config_value("SLIPPAGE_DEFAULT", 0.1),
                "min": self.configuration.get_config_value("SLIPPAGE_MIN", 0.01),
                "max": self.configuration.get_config_value("SLIPPAGE_MAX", 0.5),
                "high_congestion": self.configuration.get_config_value("SLIPPAGE_HIGH_CONGESTION", 0.05),
                "low_congestion": self.configuration.get_config_value("SLIPPAGE_LOW_CONGESTION", 0.2),
            }
            self.GAS_CONFIG: Dict[str, Union[int, float]] = {
                "max_gas_price_gwei": self.configuration.get_config_value("MAX_GAS_PRICE_GWEI", 500),
                "min_profit_multiplier": self.configuration.get_config_value("MIN_PROFIT_MULTIPLIER", 2.0),
                "base_gas_limit": self.configuration.get_config_value("BASE_GAS_LIMIT", 21000)
            }
        else: #Defaults for testing when config class is not available.
             self.SLIPPAGE_CONFIG: Dict[str, float] = {
                "default":  0.1,
                "min": 0.01,
                "max": 0.5,
                "high_congestion":  0.05,
                "low_congestion": 0.2,
            }
             self.GAS_CONFIG: Dict[str, Union[int, float]] = {
                "max_gas_price_gwei":  500,
                "min_profit_multiplier": 2.0,
                "base_gas_limit":  21000
            }
        

    async def initialize(self) -> None:
        """Initialize Safety Net components."""
        try:
            # Initialize price cache
            self.price_cache = TTLCache(maxsize=1000, ttl=self.CACHE_TTL)
            
            # Initialize gas price cache
            self.gas_price_cache = TTLCache(maxsize=1, ttl=self.GAS_PRICE_CACHE_TTL)
            
            # Initialize safety checks cache
            self.safety_cache = TTLCache(maxsize=100, ttl=60)
            
            # Verify web3 connection
            if not self.web3:
                raise RuntimeError("Web3 not initialized in Safety_Net")
                
            # Test connection
            if not await self.web3.is_connected():
                raise RuntimeError("Web3 connection failed in Safety_Net")

            logger.info("SafetyNet initialized successfully ✅")
        except Exception as e:
            logger.critical(f"Safety Net initialization failed: {e}")
            raise

    async def get_balance(self, account: Account) -> Decimal:
        """Get account balance with retries and caching."""
        cache_key = f"balance_{account.address}"
        if cache_key in self.price_cache:
            logger.debug("Balance fetched from cache.")
            return self.price_cache[cache_key]

        for attempt in range(3):
            try:
                balance = Decimal(await self.web3.eth.get_balance(account.address)) / Decimal("1e18")
                self.price_cache[cache_key] = balance
                logger.debug(f"Fetched balance: {balance} ETH")
                return balance
            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed to fetch balance: {e}")
                await asyncio.sleep(2 ** attempt)
        logger.error("Failed to fetch account balance after retries.")
        return Decimal("0")

    async def ensure_profit(
        self,
        transaction_data: Dict[str, Any],
        minimum_profit_eth: Optional[float] = None,
    ) -> bool:
        """Enhanced profit verification with dynamic thresholds and risk assessment."""
        try:
            real_time_price = await self.api_config.get_real_time_price(transaction_data['output_token'])
            if real_time_price is None:
                logger.error("Real-time price unavailable.")
                return False

            gas_cost_eth = self._calculate_gas_cost(
                Decimal(transaction_data["gas_price"]),
                transaction_data["gas_used"]
            )

            slippage = await self.adjust_slippage_tolerance()
            profit = await self._calculate_profit(
                transaction_data, real_time_price, slippage, gas_cost_eth
            )

            self._log_profit_calculation(transaction_data, real_time_price, gas_cost_eth, profit, minimum_profit_eth or 0.001)

            return profit > Decimal(minimum_profit_eth or 0.001)
        except KeyError as e:
            logger.error(f"Missing key in transaction data: {e}")
            return False
        except Exception as e:
            logger.error(f"Error in ensure_profit: {e}")
            return False

    def _validate_gas_parameters(self, gas_price_gwei: Decimal, gas_used: int) -> bool:
        """Validate gas parameters against safety thresholds."""
        if gas_used == 0:
            logger.error("Gas used cannot be zero.")
            return False
        if gas_price_gwei > self.GAS_CONFIG["max_gas_price_gwei"]:
            logger.warning(f"Gas price {gas_price_gwei} Gwei exceeds maximum threshold.")
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
        """Get the current gas price dynamically with fallback."""
        if "gas_price" in self.gas_price_cache:
            return self.gas_price_cache["gas_price"]
        try:
            if not self.web3:
                logger.error("Web3 not initialized in Safety_Net")
                return Decimal("50")  # Default fallback gas price in Gwei
                
            gas_price = await self.web3.eth.gas_price
            if gas_price is None:
                logger.warning("Failed to get gas price, using fallback")
                return Decimal("50")  # Fallback gas price
                
            gas_price_decimal = Decimal(gas_price) / Decimal(10**9)  # Convert to Gwei
            self.gas_price_cache["gas_price"] = gas_price_decimal
            return gas_price_decimal
        except Exception as e:
            logger.error(f"Failed to get dynamic gas price: {e}")
            return Decimal("50")  # Fallback gas price on error

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

    async def check_transaction_safety(
        self, 
        tx_data: Dict[str, Any],
        check_type: str = 'all'
    ) -> Tuple[bool, Dict[str, Any]]:
        """Unified safety check method for transactions."""
        try:
             is_safe = True
             messages = []
             
             if check_type in ['all', 'gas']:
                gas_price = await self.get_dynamic_gas_price()
                if gas_price > RISK_THRESHOLDS['gas_price']:
                     is_safe = False
                     messages.append(f"Gas price too high: {gas_price} Gwei")

             # Check profit potential
             if check_type in ['all', 'profit']:
                profit = await self._calculate_profit(
                    tx_data,
                    await self.api_config.get_real_time_price(tx_data['output_token']),
                    await self.adjust_slippage_tolerance(),
                    self._calculate_gas_cost(
                        Decimal(tx_data['gas_price']),
                        tx_data['gas_used']
                    )
                )
                if profit < RISK_THRESHOLDS['min_profit']:
                     is_safe = False
                     messages.append(f"Insufficient profit: {profit} ETH")

             # Check network congestion
             if check_type in ['all', 'network']:
                congestion = await self.get_network_congestion()
                if congestion > RISK_THRESHOLDS['congestion']:
                     is_safe = False
                     messages.append(f"High network congestion: {congestion:.1%}")


             return is_safe, {
                'is_safe': is_safe,
                'gas_ok': is_safe if check_type not in ['all', 'gas'] else gas_price <= RISK_THRESHOLDS['gas_price'],
                'profit_ok': is_safe if check_type not in ['all', 'profit'] else profit >= RISK_THRESHOLDS['min_profit'],
                'slippage_ok': True, # Not yet implemented slippage checks
                'congestion_ok': is_safe if check_type not in ['all', 'network'] else congestion <= RISK_THRESHOLDS['congestion'],
                'messages': messages
            }

        except Exception as e:
            logger.error(f"Safety check error: {e}")
            return False, {'is_safe': False, 'messages': [str(e)]}

    async def stop(self) -> None:
         """Stops the 0xBuilder gracefully."""
         try:
            if self.api_config:
                await self.api_config.close()
            logger.debug("Safety Net stopped successfully.")
         except Exception as e:
             logger.error(f"Error stopping safety net: {e}")
             raise
             logger.error(f"Error stopping safety net: {e}")
             raise

    async def assess_transaction_risk(
        self,
        tx: Dict[str, Any],
        token_symbol: str,
        market_conditions: Optional[Dict[str, bool]] = None,
        price_change: float = 0,
        volume: float = 0
    ) -> Tuple[float, Dict[str, Any]]:
        """Centralized risk assessment with proper error handling."""
        try:
            risk_score = 1.0
            
            # Get market conditions if not provided and market_monitor exists
            if not market_conditions and self.market_monitor:
                market_conditions = await self.market_monitor.check_market_conditions(tx.get("to", ""))
            elif not market_conditions:
                market_conditions = {}  # Default empty if no market_monitor
                
            # Gas price impact
            gas_price = int(tx.get("gasPrice", 0))
            gas_price_gwei = float(self.web3.from_wei(gas_price, "gwei"))
            if gas_price_gwei > RISK_THRESHOLDS['gas_price']:
                risk_score *= 0.7
                
            # Market conditions impact
            if market_conditions.get("high_volatility", False):
                risk_score *= 0.7
            if market_conditions.get("low_liquidity", False):
                risk_score *= 0.6
            if market_conditions.get("bullish_trend", False):
                risk_score *= 1.2
                
            # Price change impact    
            if price_change > 0:
                risk_score *= min(1.3, 1 + (price_change / 100))
                
            # Volume impact
            if volume >= 1_000_000:
                risk_score *= 1.2
            elif volume <= 100_000:
                risk_score *= 0.8
                
            return risk_score, market_conditions                    
        
        except Exception as e:
            logger.error(f"Error in risk assessment: {e}")
            return 0.0, {}


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
        web3: AsyncWeb3,
        configuration: Optional["Configuration"],
        api_config: Optional["API_Config"],
        transaction_core: Optional[Any] = None,  # Add this parameter
    ) -> None:
        """Initialize Market Monitor with required components."""
        self.web3: AsyncWeb3 = web3
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
            'price': TTLCache(maxsize=CACHE_SETTINGS['price']['size'], 
                    ttl=CACHE_SETTINGS['price']['ttl']),
            'volume': TTLCache(maxsize=CACHE_SETTINGS['volume']['size'], 
                     ttl=CACHE_SETTINGS['volume']['ttl']),
            'volatility': TTLCache(maxsize=CACHE_SETTINGS['volatility']['size'], 
                     ttl=CACHE_SETTINGS['volatility']['ttl'])
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
        scheduling
        self.update_scheduler = {
            'training_data': 0,  # Last update timestamp
            'model': 0,          # Last model update timestamp
            'UPDATE_INTERVAL': self.configuration.MODEL_RETRAINING_INTERVAL if self.configuration else 3600,  # 1 hour
            'MODEL_INTERVAL': 86400   # 24 hours
        }
        self.abi_registry = ABI_Registry()

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

    async def _get_contract(self, address: str, abi_type: str) -> Optional[Any]:
        """Get contract instance using ABI registry."""
        try:
            abi = self.abi_registry.get_abi(abi_type)
            if not abi:
                return None
            return self.web3.eth.contract(
                address=self.web3.to_checksum_address(address),
                abi=abi
            )
        except Exception as e:
            logger.error(f"Error creating contract instance: {e}")
            return None

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
        configuration: Optional[Configuration] = None,
        erc20_abi: Optional[List[Dict[str, Any]]] = None,  # Changed to accept loaded ABI
        market_monitor: Optional[Market_Monitor] = None # Market monitor for arbitrage check
    ):
        # Core components
        self.web3 = web3
        self.configuration = configuration
        self.safety_net = safety_net
        self.nonce_core = nonce_core
        self.api_config = api_config
        self.market_monitor = market_monitor

        # Monitoring state
        self.running = False
        self.pending_transactions = asyncio.Queue()
        self.monitored_tokens = set(monitored_tokens or [])
        self.profitable_transactions = asyncio.Queue()
        self.processed_transactions = set()

        # Configuration
        # Validate ERC20 ABI
        if not erc20_abi or not isinstance(erc20_abi, list):
            logger.error("Invalid or missing ERC20 ABI")
            self.erc20_abi = []
        else:
            self.erc20_abi = erc20_abi
            logger.debug(f"Loaded ERC20 ABI with {len(self.erc20_abi)} entries")
        self.minimum_profit_threshold = Decimal("0.001")
        self.max_parallel_tasks = self.MAX_PARALLEL_TASKS
        self.retry_attempts = self.MAX_RETRIES
        self.backoff_factor = 1.5

        # Concurrency control
        self.semaphore = asyncio.Semaphore(self.max_parallel_tasks)
        self.task_queue = asyncio.Queue()

        # Add function signature mappings
        self.function_signatures = {
            '0xa9059cbb': 'transfer',
            '0x095ea7b3': 'approve',
            '0x23b872dd': 'transferFrom',
            # Add more common ERC20 function signatures as needed
        }
        if configuration and hasattr(configuration, 'ERC20_SIGNATURES'):
            self.function_signatures.update(configuration.ERC20_SIGNATURES)

        self.abi_registry = ABI_Registry()

        logger.info("Go for main engine start! ✅...")
        time.sleep(3) # ensuring proper initialization

    async def start_monitoring(self) -> None:
        """Start monitoring the mempool with improved error handling."""
        if self.running:
            logger.debug("Monitoring is already active.")
            return

        try:
            self.running = True
            monitoring_task = asyncio.create_task(self._run_monitoring())
            processor_task = asyncio.create_task(self._process_task_queue())

            logger.info("Lift-off 🚀🚀🚀")
            logger.info("Monitoring mempool activities... 📡")
            await asyncio.gather(monitoring_task, processor_task)
            

        except Exception as e:
            self.running = False
    async def stop(self) -> None:
        """Gracefully stop monitoring activities.""" 
        if not self.running:
            return

        self.running = False
        self.stopping = True
        try:
            # Wait for remaining tasks with timeout
            try:
                # Set a timeout of 5 seconds for remaining tasks
                await asyncio.wait_for(self.task_queue.join(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Timed out waiting for tasks to complete")
            
            # Cancel any remaining tasks
            while not self.task_queue.empty():
                try:
                    self.task_queue.get_nowait()
                    self.task_queue.task_done()
                except asyncio.QueueEmpty:
                    break
                
            logger.debug("Mempool monitoring stopped gracefully.")
        except Exception as e:
            logger.error(f"Error during monitoring shutdown: {e}")
            raise

    async def _run_monitoring(self) -> None:
        """Enhanced mempool monitoring with automatic recovery and fallback.""" 
        retry_count = 0
        
        while self.running:
            try:
                # Try setting up filter first
                try:
                    pending_filter = await self._setup_pending_filter()
                    if pending_filter:
                        while self.running:
                            tx_hashes = await pending_filter.get_new_entries()
                            await self._handle_new_transactions(tx_hashes)
                            await asyncio.sleep(0.1)  # Prevent tight loop
                    else:
                        # Fallback to polling if filter not available
                        await self._poll_pending_transactions()
                except Exception as filter_error:
                    logger.warning(f"Filter-based monitoring failed: {filter_error}")
                    logger.info("Switching to polling-based monitoring...")
                    await self._poll_pending_transactions()

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                retry_count += 1
                await asyncio.sleep(self.RETRY_DELAY * retry_count)

    async def _poll_pending_transactions(self) -> None:
        """Enhanced polling method with better error handling."""
        last_block = await self.web3.eth.block_number
        
        while self.running:
            try:
                current_block = await self.web3.eth.block_number
                if current_block <= last_block:
                    await asyncio.sleep(1)
                    continue

                # Process new blocks
                for block_num in range(last_block + 1, current_block + 1):
                    try:
                        block = await self.web3.eth.get_block(block_num, full_transactions=True)
                        if block and block.transactions:
                            tx_hashes = [tx.hash.hex() if hasattr(tx, 'hash') else tx['hash'].hex() 
                                       for tx in block.transactions]
                            await self._handle_new_transactions(tx_hashes)
                    except Exception as e:
                        logger.error(f"Error processing block {block_num}: {e}")
                        continue

                last_block = current_block
                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Error in polling loop: {e}")
                await asyncio.sleep(2)

    async def _setup_pending_filter(self) -> Optional[Any]:
        """Set up pending transaction filter with validation and fallback."""
        try:
            # Try to create a filter
            pending_filter = await self.web3.eth.filter("pending")
            
            # Test filter with timeout
            try:
                async with async_timeout.timeout(5):
                    await pending_filter.get_new_entries()
                    logger.debug("Successfully set up pending transaction filter")
                    return pending_filter
            except asyncio.TimeoutError:
                logger.warning("Filter setup timed out, falling back to polling")
                return None
            except Exception as e:
                logger.warning(f"Filter validation failed: {e}, falling back to polling")
                return None

        except Exception as e:
            logger.warning(f"Failed to setup pending filter: {e}, falling back to polling")
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
            logger.error(f"Error handling new transactions: {e}")

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
                logger.error(f"Error processing task queue: {e}")

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
        """Enhanced profitable transaction handling with validation."""
        try:
            # Validate profit value
            profit = analysis.get('profit', Decimal(0))
            if isinstance(profit, (int, float)):
                profit = Decimal(str(profit))
            elif not isinstance(profit, Decimal):
                logger.warning(f"Invalid profit type: {type(profit)}")
                profit = Decimal(0)

            # Format profit for logging
            profit_str = f"{float(profit):.6f}" if profit > 0 else 'Unknown'
            
            # Add additional analysis data
            analysis['profit'] = profit
            analysis['timestamp'] = time.time()
            analysis['gas_price'] = self.web3.from_wei(
                analysis.get('gasPrice', 0), 
                'gwei'
            )

            await self.profitable_transactions.put(analysis)
            
            logger.info(
                f"Profitable transaction identified: {analysis['tx_hash']} "
                f"(Estimated profit: {profit_str} ETH)"
            )

        except Exception as e:
            logger.error(f"Error handling profitable transaction: {e}")

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
        """Enhanced token transaction analysis with better validation."""
        try:
            if not self.erc20_abi or not tx.input or len(tx.input) < 10:
                logger.debug("Missing ERC20 ABI or invalid transaction input")
                return {"is_profitable": False}

            # Extract function selector
            function_selector = tx.input[:10]  # includes '0x'
            selector_no_prefix = function_selector[2:]  # remove '0x'

            # Initialize variables
            function_name = None
            function_params = {}
            decoded = False

            # Method 1: Try local signature lookup first (fastest)
            if selector_no_prefix in self.function_signatures:
                try:
                    function_name = self.function_signatures[selector_no_prefix]
                    if len(tx.input) >= 138:  # Standard ERC20 call length
                        params_data = tx.input[10:]
                        if function_name == 'transfer':
                            to_address = '0x' + params_data[:64][-40:]
                            amount = int(params_data[64:128], 16)
                            function_params = {'to': to_address, 'amount': amount}
                            decoded = True
                except Exception as e:
                    logger.debug(f"Error in direct signature lookup: {e}")

            # Method 2: Try contract decode only if direct lookup failed
            if not decoded:
                try:
                    # Ensure we have a valid contract address and ABI
                    if not tx.to or not self.erc20_abi:
                        logger.debug("Missing contract address or ABI")
                        return {"is_profitable": False}

                    contract = self.web3.eth.contract(
                        address=self.web3.to_checksum_address(tx.to),
                        abi=self.erc20_abi
                    )

                    try:
                        func_obj, decoded_params = contract.decode_function_input(tx.input)
                        function_name = (
                            getattr(func_obj, 'fn_name', None) or
                            getattr(func_obj, 'function_identifier', None)
                        )
                        if function_name:
                            function_params = decoded_params
                            decoded = True
                    except Web3ValueError as e:
                        logger.debug(f"Could not decode function input: {e}")
                except Exception as e:
                    logger.debug(f"Contract decode error: {e}")

            # Method 3: Configuration fallback
            if not decoded and hasattr(self.configuration, 'ERC20_SIGNATURES'):
                try:
                    function_name = self.configuration.ERC20_SIGNATURES.get(function_selector)
                    if function_name:
                        decoded = True
                except Exception as e:
                    logger.debug(f"Error in configuration lookup: {e}")

            # Process decoded transaction if successful
            if decoded and function_name in ('transfer', 'transferFrom', 'swap'):
                # Enhanced parameter validation
                params = await self._extract_transaction_params(tx, function_name, function_params)
                if not params:
                    logger.debug(f"Could not extract valid parameters for {function_name}")
                    return {"is_profitable": False}

                amounts = await self._validate_token_amounts(params)
                if not amounts['valid']:
                    logger.debug(f"Invalid token amounts: {amounts['reason']}")
                    if 'details' in amounts:
                        logger.debug(f"Validation details: {amounts['details']}")
                    return {"is_profitable": False}

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

            if not decoded:
                logger.debug(f"Could not decode transaction with selector: {function_selector}")

            return {"is_profitable": False}

        except Exception as e:
            logger.error(f"Error analyzing token transaction {tx.hash.hex()}: {e}")
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
        """Enhanced profit estimation with improved precision and market analysis."""
        try:
            # Validate and get gas costs with increased precision
            gas_data = await self._calculate_gas_costs(tx)
            if not gas_data['valid']:
                logger.debug(f"Invalid gas data: {gas_data['reason']}")
                return Decimal(0)

            # Get token amounts with validation
            amounts = await self._validate_token_amounts(function_params)
            if not amounts['valid']:
                logger.debug(f"Invalid token amounts: {amounts['reason']}")
                return Decimal(0)

            # Get market data with validation
            market_data = await self._get_market_data(function_params['path'][-1])
            if not market_data['valid']:
                logger.debug(f"Invalid market data: {market_data['reason']}")
                return Decimal(0)

            # Calculate profit with all factors
            profit = await self._calculate_final_profit(
                amounts=amounts['data'],
                gas_costs=gas_data['data'],
                market_data=market_data['data']
            )

            # Log comprehensive calculation details
            self._log_profit_calculation(profit, amounts['data'], gas_data['data'], market_data['data'])

            return Decimal(max(0, profit))

        except Exception as e:
            logger.error(f"Error in profit estimation: {e}")
            return Decimal(0)

    async def _calculate_gas_costs(self, tx: Any) -> Dict[str, Any]:
        """Calculate gas costs with improved precision."""
        try:
            gas_price_wei = Decimal(tx.gasPrice)
            gas_price_gwei = Decimal(self.web3.from_wei(gas_price_wei, "gwei"))
            
            # Get dynamic gas estimate
            gas_used = tx.gas if tx.gas else await self.web3.eth.estimate_gas(tx)
            gas_used = Decimal(gas_used)

            # Add safety margin for gas estimation (10%)
            gas_with_margin = gas_used * Decimal("1.1")
            
            # Calculate total gas cost in ETH
            gas_cost_eth = (gas_price_gwei * gas_with_margin * Decimal("1e-9")).quantize(Decimal("0.000000001"))

            return {
                'valid': True,
                'data': {
                    'gas_price_gwei': gas_price_gwei,
                    'gas_used': gas_used,
                    'gas_with_margin': gas_with_margin,
                    'gas_cost_eth': gas_cost_eth
                }
            }
        except Exception as e:
            return {'valid': False, 'reason': str(e)}

    async def _validate_token_amounts(self, function_params: Dict[str, Any]) -> Dict[str, Any]:
        """Improved token amount validation with better hex value handling."""
        try:
            # Extract amounts with more comprehensive fallbacks
            input_amount = function_params.get("amountIn", 
                function_params.get("value", 
                function_params.get("amount",
                function_params.get("_value", 0)
            )))
            output_amount = function_params.get("amountOutMin",
                function_params.get("amountOut",
                function_params.get("amount",
                function_params.get("_amount", 0)
                )
                )
            )

            # Enhanced hex string handling
            def parse_amount(amount: Any) -> int:
                if isinstance(amount, str):
                    if amount.startswith("0x"):
                        return int(amount, 16)
                    if amount.isnumeric():
                        return int(amount)
                return int(amount) if amount else 0

            # Parse amounts
            try:
                input_amount_wei = Decimal(str(parse_amount(input_amount)))
                output_amount_wei = Decimal(str(parse_amount(output_amount)))
            except (ValueError, TypeError) as e:
                return {
                    'valid': False,
                    'reason': f'Amount parsing error: {str(e)}',
                    'details': {
                        'input_raw': input_amount,
                        'output_raw': output_amount
                    }
                }

            # Validate amounts
            if input_amount_wei <= 0 and output_amount_wei <= 0:
                return {
                    'valid': False,
                    'reason': 'Both input and output amounts are zero or negative',
                    'details': {
                        'input_wei': str(input_amount_wei),
                        'output_wei': str(output_amount_wei)
                    }
                }

            # Convert to ETH with higher precision
            input_amount_eth = Decimal(self.web3.from_wei(input_amount_wei, "ether")).quantize(Decimal("0.000000001"))
            output_amount_eth = Decimal(self.web3.from_wei(output_amount_wei, "ether")).quantize(Decimal("0.000000001"))

            return {
                'valid': True,
                'data': {
                    'input_eth': input_amount_eth,
                    'output_eth': output_amount_eth,
                    'input_wei': input_amount_wei,
                    'output_wei': output_amount_wei
                }
            }

        except Exception as e:
            logger.error(f"Unexpected error in token amount validation: {e}")
            return {
                'valid': False,
                'reason': 'Unknown validation error',
                'details': {
                    'error': str(e),
                    'params': str(function_params)
                }
            }

    async def _get_market_data(self, token_address: str) -> Dict[str, Any]:
        """Get comprehensive market data for profit calculation."""
        try:
            token_symbol = await self.api_config.get_token_symbol(self.web3, token_address)
            if not token_symbol:
                return {'valid': False, 'reason': 'Token symbol not found'}

            # Get market price and liquidity data
            price = await self.api_config.get_real_time_price(token_symbol.lower())
            if not price or price <= 0:
                return {'valid': False, 'reason': 'Invalid market price'}

            # Calculate dynamic slippage based on liquidity
            slippage = await self._calculate_dynamic_slippage(token_symbol)

            return {
                'valid': True,
                'data': {
                    'price': Decimal(str(price)),
                    'slippage': slippage,
                    'symbol': token_symbol
                }
            }
        except Exception as e:
            return {'valid': False, 'reason': str(e)}

    async def _calculate_dynamic_slippage(self, token_symbol: str) -> Decimal:
        """Calculate dynamic slippage based on market conditions."""
        try:
            volume = await self.api_config.get_token_volume(token_symbol)
            # Adjust slippage based on volume (higher volume = lower slippage)
            if (volume > 1_000_000):  # High volume
                return Decimal("0.995")  # 0.5% slippage
            elif volume > 500_000:  # Medium volume
                return Decimal("0.99")   # 1% slippage
            else:  # Low volume
                return Decimal("0.98")   # 2% slippage
        except Exception:
            return Decimal("0.99")  # Default to 1% slippage

    async def _calculate_final_profit(
        self,
        amounts: Dict[str, Decimal],
        gas_costs: Dict[str, Decimal],
        market_data: Dict[str, Any]
    ) -> Decimal:
        """Calculate final profit with all factors considered."""
        try:
            # Calculate expected output value with slippage
            expected_output_value = (
                amounts['output_eth'] * 
                market_data['price'] * 
                market_data['slippage']
            ).quantize(Decimal("0.000000001"))

            # Calculate net profit
            profit = (
                expected_output_value - 
                amounts['input_eth'] - 
                gas_costs['gas_cost_eth']
            ).quantize(Decimal("0.000000001"))

            return profit

        except Exception as e:
            logger.error(f"Error in final profit calculation: {e}")
            return Decimal(0)

    def _log_profit_calculation(
        self,
        profit: Decimal,
        amounts: Dict[str, Decimal],
        gas_costs: Dict[str, Decimal],
        market_data: Dict[str, Any]
    ) -> None:
        """Log detailed profit calculation metrics."""
        logger.debug(
            f"Profit Calculation Details:\n"
            f"Token: {market_data['symbol']}\n"
            f"Input Amount: {amounts['input_eth']:.9f} ETH\n"
            f"Expected Output: {amounts['output_eth']:.9f} tokens\n"
            f"Market Price: {market_data['price']:.9f}\n"
            f"Slippage: {(1 - float(market_data['slippage'])) * 100:.2f}%\n"
            f"Gas Cost: {gas_costs['gas_cost_eth']:.9f} ETH\n"
            f"Gas Price: {gas_costs['gas_price_gwei']:.2f} Gwei\n"
            f"Final Profit: {profit:.9f} ETH"
        )

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

    async def _extract_transaction_params(
        self,
        tx: Any,
        function_name: str,
        decoded_params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Extract and validate transaction parameters with improved parsing.
        """
        try:
            params = {}
            
            # Enhanced transfer parameter handling
            if function_name in ['transfer', 'transferFrom']:
                # Handle different parameter naming conventions
                amount = decoded_params.get('amount', 
                            decoded_params.get('_value', 
                                decoded_params.get('value', 
                                    decoded_params.get('wad', 0))))
                to_addr = decoded_params.get('to', 
                            decoded_params.get('_to', 
                                decoded_params.get('dst', 
                                    decoded_params.get('recipient'))))
                
                if function_name == 'transferFrom':
                    from_addr = decoded_params.get('from', 
                                decoded_params.get('_from', 
                                    decoded_params.get('src', 
                                        decoded_params.get('sender'))))
                    params['from'] = from_addr
                
                params.update({
                    'amount': amount,
                    'to': to_addr
                })
            
            # Enhanced swap parameter handling
            elif function_name in ['swap', 'swapExactTokensForTokens', 'swapTokensForExactTokens']:
                # Handle different swap parameter formats
                params = {
                    'amountIn': decoded_params.get('amountIn',
                                decoded_params.get('amount0',
                                    decoded_params.get('amountInMax', 0))),
                    'amountOutMin': decoded_params.get('amountOutMin',
                                decoded_params.get('amount1',
                                    decoded_params.get('amountOut', 0))),
                    'path': decoded_params.get('path', [])
                }
            
            # Validate parameters presence and format
            if not self._validate_params_format(params, function_name):
                logger.debug(f"Invalid parameter format for {function_name}")
                return None
            
            return params
        
        except Exception as e:
            logger.error(f"Error extracting transaction parameters: {e}")
            return None

    def _validate_params_format(self, params: Dict[str, Any], function_name: str) -> bool:
        """
        Validate parameter format based on function type.
        """
        try:
            if function_name in ['transfer', 'transferFrom']:
                required = ['amount', 'to']
                if function_name == 'transferFrom':
                    required.append('from')
            elif function_name in ['swap', 'swapExactTokensForTokens', 'swapTokensForExactTokens']:
                required = ['amountIn', 'amountOutMin', 'path']
            else:
                logger.debug(f"Unsupported function name: {function_name}")
                return False

            # Check required fields are present and non-empty
            for field in required:
                if params.get(field) is None or params.get(field) == '':
                    logger.debug(f"Missing or empty field '{field}' for function '{function_name}'")
                    return False
            
            return True

        except Exception as e:
            logger.error(f"Parameter validation error: {e}")
            return False

    async def initialize(self) -> None:
        """
        Initialize with proper ABI loading.
        """
        try:
            # Load ERC20 ABI through ABI manager
            self.erc20_abi = await self.abi_registry.load_abi('erc20')
            if not self.erc20_abi:
                raise ValueError("Failed to load ERC20 ABI")
            
            # Validate required methods
            required_methods = ['transfer', 'approve', 'transferFrom', 'balanceOf']
            if not self.abi_registry.validate_abi(self.erc20_abi, required_methods):
                raise ValueError("Invalid ERC20 ABI")
            
            # Initialize other attributes
            self.running = False
            self.pending_transactions = asyncio.Queue()
            self.profitable_transactions = asyncio.Queue()
            self.processed_transactions = set()
            self.task_queue = asyncio.Queue()
            
            logger.info("MempoolMonitor initialized ✅")
        
        except Exception as e:
            logger.critical(f"Mempool Monitor initialization failed: {e}")
            raise

    async def stop(self) -> None:
        """
        Gracefully stop the Mempool Monitor.
        """ 
        try:
            self.running = False
            self.stopping = True
            await self.task_queue.join()
            logger.debug("Mempool Monitor stopped gracefully.")
        
        except Exception as e:
            logger.error(f"Error stopping Mempool Monitor: {e}")
            raise
            
class Transaction_Core:
    """
    Transaction_Core is the main transaction engine that handles all transaction-related
    Builds and executes transactions, including front-run, back-run, and sandwich attack strategies.
    It interacts with smart contracts, manages transaction signing, gas price estimation, and handles flashloans
    """
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 1.0  # Base delay in seconds for retries
    DEFAULT_GAS_LIMIT: int = 100_000 # Default gas limit
    DEFAULT_CANCEL_GAS_PRICE_GWEI: int = 150
    DEFAULT_PROFIT_TRANSFER_MULTIPLIER: int = 10**18
    DEFAULT_GAS_PRICE_GWEI: int = 50

    def __init__(
        self,
        web3: AsyncWeb3,
        account: Account,
        AAVE_FLASHLOAN_ADDRESS: str,
        AAVE_FLASHLOAN_ABI: List[Dict[str, Any]],
        AAVE_POOL_ADDRESS: str,
        AAVE_POOL_ABI: List[Dict[str, Any]],
        api_config: Optional["API_Config"] = None,
        market_monitor: Optional["Market_Monitor"] = None,
        mempool_monitor: Optional["Mempool_Monitor"] = None,
        nonce_core: Optional["Nonce_Core"] = None,
        safety_net: Optional["Safety_Net"] = None,
        configuration: Optional["Configuration"] = None,
        gas_price_multiplier: float = 1.1,
        erc20_abi: Optional[List[Dict[str, Any]]] = None,
        uniswap_address: Optional[str] = None,
        uniswap_abi: Optional[List[Dict[str, Any]]] = None,
    ):
        self.web3: AsyncWeb3 = web3
        self.account: Account = account
        self.configuration: Optional["Configuration"] = configuration
        self.market_monitor: Optional["Market_Monitor"] = market_monitor
        self.mempool_monitor: Optional["Mempool_Monitor"] = mempool_monitor
        self.api_config: Optional["API_Config"] = api_config
        self.nonce_core: Optional["Nonce_Core"] = nonce_core
        self.safety_net: Optional["Safety_Net"] = safety_net
        self.gas_price_multiplier: float = gas_price_multiplier
        self.RETRY_ATTEMPTS: int = self.MAX_RETRIES
        self.erc20_abi: List[Dict[str, Any]] = erc20_abi or []
        self.current_profit: Decimal = Decimal("0")
        self.AAVE_FLASHLOAN_ADDRESS: str = AAVE_FLASHLOAN_ADDRESS
        self.AAVE_FLASHLOAN_ABI: List[Dict[str, Any]] = AAVE_FLASHLOAN_ABI
        self.AAVE_POOL_ADDRESS: str = AAVE_POOL_ADDRESS
        self.AAVE_POOL_ABI: List[Dict[str, Any]] = AAVE_POOL_ABI
        self.abi_registry: ABI_Registry = ABI_Registry()
        self.uniswap_address: str = uniswap_address
        self.uniswap_abi: List[Dict[str, Any]] = uniswap_abi or []
        


    def normalize_address(self, address: str) -> str:
        """Normalize Ethereum address to checksum format."""
        try:
            # Directly convert to checksum without changing case
            return self.web3.to_checksum_address(address)
        except Exception as e:
            logger.error(f"Error normalizing address {address}: {e}")
            raise

    async def initialize(self) -> None:
        """Initialize with proper ABI loading."""
        try:
            # Initialize contracts using ABIs from registry
            router_configs = [
                (self.configuration.UNISWAP_ADDRESS, 'uniswap', 'Uniswap'),
                (self.configuration.SUSHISWAP_ROUTER_ADDRESS, 'sushiswap', 'Sushiswap'),
            ]

            for address, abi_type, name in router_configs:
                try:
                    # Normalize address before creating contract
                    normalized_address = self.normalize_address(address)
                    abi = self.abi_registry.get_abi(abi_type)
                    if not abi:
                        raise ValueError(f"Failed to load {name} ABI")
                    contract = self.web3.eth.contract(
                        address=normalized_address,
                        abi=abi
                    )

                    # Perform validation
                    await self._validate_contract(contract, name, abi_type)

                    setattr(self, f"{name.lower()}_router_contract", contract)
                except Exception as e:
                    logger.error(f"Failed to initialize {name} router: {e}")
                    raise

            # Initialize Aave contracts with correct functions
            self.aave_flashloan = self.web3.eth.contract(
                address=self.normalize_address(self.configuration.AAVE_FLASHLOAN_ADDRESS),
                abi=self.configuration.AAVE_FLASHLOAN_ABI
            )

            await self._validate_contract(self.aave_flashloan, "Aave Flashloan", 'aave_flashloan')

            self.aave_pool = self.web3.eth.contract(
                address=self.normalize_address(self.AAVE_POOL_ADDRESS),
                abi=self.AAVE_POOL_ABI
            )
            await self._validate_contract(self.aave_pool, "Aave Lending Pool", 'aave')

            logger.info("Transaction Core initialized successfully")

        except Exception as e:
            logger.error(f"Transaction Core initialization failed: {e}")
            raise

    async def _validate_contract(self, contract: Any, name: str, abi_type: str) -> None:
        """Validates contracts using a shared pattern."""
        try:
            if 'Lending Pool' in name:
                await contract.functions.getReservesList().call()
                logger.debug(f"{name} contract validated successfully via getReservesList()")
            elif 'Flashloan' in name:
                await contract.functions.ADDRESSES_PROVIDER().call()
                logger.debug(f"{name} contract validated successfully via ADDRESSES_PROVIDER()")
            elif abi_type in ['uniswap', 'sushiswap']:
                path = [self.configuration.WETH_ADDRESS, self.configuration.USDC_ADDRESS]
                await contract.functions.getAmountsOut(1000000, path).call()
                logger.debug(f"{name} contract validated successfully")
            else:
                logger.debug(f"No specific validation for {name}, but initialized")


        except Exception as e:
             logger.warning(f"Contract validation warning for {name}: {e}")

    async def _load_erc20_abi(self) -> List[Dict[str, Any]]:
        """Load the ERC20 ABI with better path handling."""
        try:
            return await self.abi_registry.get_abi('erc20')
            
        except Exception as e:
            logger.error(f"Failed to load ERC20 ABI: {e}")
            raise

    async def _validate_signatures(self) -> None:
        """Validate loaded ERC20 signatures."""
        # This function was not used, and it is now removed
        pass

    async def build_transaction(self, function_call: Any, additional_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Enhanced transaction building with EIP-1559 support and proper gas estimation.""" 
        additional_params = additional_params or {}
        try:
            # Get chain ID
            chain_id = await self.web3.eth.chain_id
            
            # Check if chain supports EIP-1559
            latest_block = await self.web3.eth.get_block('latest')
            supports_eip1559 = 'baseFeePerGas' in latest_block
            
            # Base transaction parameters
            tx_params = {
                'chainId': chain_id,
                'nonce': await self.nonce_core.get_nonce(),
                'from': self.account.address,
            }
            
            # Add EIP-1559 specific parameters if supported
            if supports_eip1559:
                base_fee = latest_block['baseFeePerGas']
                priority_fee = await self.web3.eth.max_priority_fee
                
                tx_params.update({
                    'maxFeePerGas': int(base_fee * 2),  # Double the base fee
                    'maxPriorityFeePerGas': int(priority_fee)
                })
            else:
                # Legacy gas price
                tx_params.update(await self._get_dynamic_gas_parameters())
            
            # Build transaction
            tx_details = function_call.buildTransaction(tx_params)
            
            # Add additional parameters
            tx_details.update(additional_params)
            
            # Estimate gas with buffer
            estimated_gas = await self.estimate_gas_smart(tx_details)
            tx_details['gas'] = int(estimated_gas * 1.1)  # Add 10% buffer
            
            return tx_details
            
        except Exception as e:
            logger.error(f"Error building transaction: {e}")
            raise

    async def _get_dynamic_gas_parameters(self) -> Dict[str, int]:
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
            gas_price_gwei = Decimal(self.DEFAULT_GAS_PRICE_GWEI)  # Default gas price in Gwei

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
            return self.DEFAULT_GAS_LIMIT  # Default gas limit
        except TransactionNotFound:
            logger.warning("Transaction not found during gas estimation. Using default gas limit.")
            return self.DEFAULT_GAS_LIMIT
        except Exception as e:
            logger.error(f"Gas estimation failed: {e}. Using default gas limit.")
            return self.DEFAULT_GAS_LIMIT  # Default gas limit

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
                logger.debug(f"Transaction sent successfully: {tx_hash.hex()}")
                return tx_hash.hex()
        except TransactionNotFound as e:
            logger.error(
                f"Transaction not found: {e}. Attempt {attempt} of {self.retry_attempts}"
            )
        except ContractLogicError as e:
            logger.error(
                f"Contract logic error: {e}. Attempt {attempt} of {self.retry_attempts}"
            )
        except Exception as e:
            logger.warning(f"Attempt {attempt+1}: Failed to execute transaction - {e}")
            await asyncio.sleep(self.RETRY_DELAY * (attempt + 1))
        logger.error("Failed to execute transaction after retries")
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
            logger.debug(
                f"Transaction signed successfully: Nonce {transaction['nonce']}. ✍️ "
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
            logger.debug(
                f"Building ETH front-run transaction for {eth_value_ether} ETH to {tx_details['to']}"
            )
            tx_hash_executed = await self.execute_transaction(tx_details)
            if tx_hash_executed:
                logger.debug(
                    f"Successfully executed ETH transaction with hash: {tx_hash_executed} ✅ "
                )
                return True
            else:
                logger.warning("Failed to execute ETH transaction. Retrying... ⚠️ ")
        
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
            function_call = self.aave_flashloan.functions.fn_RequestFlashLoan(
                flashloan_asset,
                flashloan_amount
            )
            tx = await self.build_transaction(function_call)
            return tx
        except ContractLogicError as e:
            logger.error(
                f"Contract logic error preparing flashloan transaction: {e} ⚠️ "
            )
            return None
        except Exception as e:
            logger.error(f"Error preparing flashloan transaction: {e} ⚠️ ")
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
                        logger.debug(f"Attempt {attempt} to send bundle via {builder['name']}. ℹ️ ")
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

                                logger.info(f"Bundle sent successfully via {builder['name']}. ✅ ")
                                successes.append(builder['name'])
                                break  # Success, move to next builder

                    except aiohttp.ClientResponseError as e:
                        logger.error(
                            f"HTTP error sending bundle via {builder['name']}: {e}. Attempt {attempt} of {self.retry_attempts}"
                        )
                        if attempt < self.retry_attempts:
                            sleep_time = self.RETRY_DELAY * attempt
                            logger.warning(f"Retrying in {sleep_time} seconds...")
                            await asyncio.sleep(sleep_time)
                    except ValueError as e:
                        logger.error(f"Bundle submission error via {builder['name']}: {e} ⚠️ ")
                        break  # Move to next builder
                    except Exception as e:
                        logger.error(f"Unexpected error with {builder['name']}: {e}. Attempt {attempt} of {self.retry_attempts} ⚠️ ")
                        if attempt < self.retry_attempts:
                            sleep_time = self.RETRY_DELAY * attempt
                            logger.warning(f"Retrying in {sleep_time} seconds...")
                            await asyncio.sleep(sleep_time)

            # Update nonce if any submissions succeeded
            if successes:
                await self.nonce_core.refresh_nonce()
                logger.info(f"Bundle successfully sent to builders: {', '.join(successes)} ✅ ")
                return True
            else:
                logger.warning("Failed to send bundle to any MEV builders. ⚠️ ")
                return False

        except Exception as e:
            logger.error(f"Unexpected error in send_bundle: {e} ⚠️ ")
            return False

    async def _validate_transaction(self, tx: Dict[str, Any], operation: str) -> Optional[Dict[str, Any]]:
        """Common transaction validation logic."""
        if not isinstance(tx, dict):
            logger.debug("Invalid transaction format provided!")            
            return None

        required_fields = ["input", "to", "value", "gasPrice"]
        if not all(field in tx for field in required_fields):
            missing = [field for field in required_fields if field not in tx]
            logger.debug(f"Missing required parameters for {operation}: {missing}")
            return None

        # Decode and validate transaction input
        decoded_tx = await self.decode_transaction_input(
            tx.get("input", "0x"),
            self.web3.to_checksum_address(tx.get("to", ""))
        )
        if not decoded_tx or "params" not in decoded_tx:
            logger.debug(f"Failed to decode transaction input for {operation}")
            return None

        # Validate path parameter
        path = decoded_tx["params"].get("path", [])
        if not path or not isinstance(path, list) or len(path) < 2:
            logger.debug(f"Invalid path parameter for {operation}")
            return None

        return decoded_tx

    async def front_run(self, target_tx: Dict[str, Any]) -> bool:
        """Execute front-run transaction with validation."""
        decoded_tx = await self._validate_transaction(target_tx, "front-run")
        if not decoded_tx:
            return False

        try:
            path = decoded_tx["params"]["path"]
            flashloan_tx = await self._prepare_flashloan(path[0], target_tx)
            front_run_tx = await self._prepare_front_run_transaction(target_tx)

            if not all([flashloan_tx, front_run_tx]):
                return False

            # Validate and send transaction bundle
            if await self._validate_and_send_bundle([flashloan_tx, front_run_tx]):
                logger.info("Front-run executed successfully ✅")
                return True

            return False
        except Exception as e:
            logger.error(f"Front-run execution failed: {e}")
            return False

    async def back_run(self, target_tx: Dict[str, Any]) -> bool:
        """Execute back-run transaction with validation."""
        decoded_tx = await self._validate_transaction(target_tx, "back-run")
        if not decoded_tx:
            return False

        try:
            back_run_tx = await self._prepare_back_run_transaction(target_tx, decoded_tx)
            if not back_run_tx:
                return False

            if await self._validate_and_send_bundle([back_run_tx]):
                logger.info("Back-run executed successfully ✅")
                return True

            return False
        except Exception as e:
            logger.error(f"Back-run execution failed: {e}")
            return False

    async def execute_sandwich_attack(self, target_tx: Dict[str, Any]) -> bool:
        """Execute sandwich attack with validation."""
        decoded_tx = await self._validate_transaction(target_tx, "sandwich")
        if not decoded_tx:
            return False

        try:
            path = decoded_tx["params"]["path"]
            flashloan_tx = await self._prepare_flashloan(path[0], target_tx)
            front_tx = await self._prepare_front_run_transaction(target_tx)
            back_tx = await self._prepare_back_run_transaction(target_tx, decoded_tx)

            if not all([flashloan_tx, front_tx, back_tx]):
                return False

            if await self._validate_and_send_bundle([flashloan_tx, front_tx, back_tx]):
                logger.info("Sandwich attack executed successfully 🥪✅")
                return True

            return False
        except Exception as e:
            logger.error(f"Sandwich attack execution failed: {e}")
            return False

    async def _prepare_flashloan(self, asset: str, target_tx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Helper to prepare flashloan transaction."""
        flashloan_amount = self.calculate_flashloan_amount(target_tx)
        if flashloan_amount <= 0:
            return None
        return await self.prepare_flashloan_transaction(
            self.web3.to_checksum_address(asset),
            flashloan_amount
        )

    async def _validate_and_send_bundle(self, transactions: List[Dict[str, Any]]) -> bool:
        """Validate and send a bundle of transactions."""
        # Simulate all transactions
        simulations = await asyncio.gather(
            *[self.simulate_transaction(tx) for tx in transactions],
            return_exceptions=True
        )

        if any(isinstance(result, Exception) or not result for result in simulations):
            logger.warning("Transaction simulation failed")
            return False

        return await self.send_bundle(transactions)

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
                logger.debug("Failed to decode target transaction input for front-run. Skipping.")
                return None

            function_name = decoded_tx.get("function_name")
            if not function_name:
                logger.debug("Missing function name in decoded transaction.  🚨")
                return None

            function_params = decoded_tx.get("params", {})
            to_address = self.web3.to_checksum_address(target_tx.get("to", ""))

            # Router address mapping
            routers = {
                self.configuration.UNISWAP_ADDRESS: (self.uniswap_router_contract, "Uniswap"),
                self.configuration.SUSHISWAP_ROUTER_ADDRESS: (self.sushiswap_router_contract, "Sushiswap"),
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
            logger.info(f"Prepared front-run transaction on {exchange_name} successfully. ✅ ")
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
                self.configuration.UNISWAP_ADDRESS: (self.uniswap_router_contract, "Uniswap"),
                self.configuration.SUSHISWAP_ROUTER_ADDRESS: (self.sushiswap_router_contract, "Sushiswap"),
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
        """Decode transaction input using ABI registry."""
        try:
            # Get selector from input
            selector = input_data[:10][2:]  # Remove '0x' prefix
            
            # Get method name from registry
            method_name = self.abi_registry.get_method_selector(selector)
            if not method_name:
                logger.debug(f"Unknown method selector: {selector}")
                return None

            # Get appropriate ABI for decoding
            for abi_type, abi in self.abi_registry.abis.items():
                try:
                    contract = self.web3.eth.contract(
                        address=self.web3.to_checksum_address(contract_address),
                        abi=abi
                    )
                    func_obj, decoded_params = contract.decode_function_input(input_data)
                    return {
                        "function_name": method_name,
                        "params": decoded_params,
                        "signature": selector,
                        "abi_type": abi_type
                    }
                except Exception:
                    continue

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
                "gasPrice": self.web3.to_wei(self.DEFAULT_CANCEL_GAS_PRICE_GWEI, "gwei"),  # Higher than the stuck transaction
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
                f"Gas estimation failed: {e}. Using default gas limit of {self.DEFAULT_GAS_LIMIT}."
            )
            return self.DEFAULT_GAS_LIMIT  # Default gas limit

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
            withdraw_function = self.aave_flashloan.functions.withdrawETH()
            tx = await self.build_transaction(withdraw_function)
            tx_hash = await self.execute_transaction(tx)
            if (tx_hash):
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
    
    async def estimate_transaction_profit(self, tx: Dict[str, Any]) -> Decimal:
        """
        Estimates the profit of a transaction based on the current gas price.

        :param tx: Transaction dictionary.
        :return: Estimated profit as Decimal.
        """
        try:
            gas_price = await self._get_dynamic_gas_parameters()
            gas_limit = await self.estimate_gas_limit(tx)
            gas_cost = gas_price["gasPrice"] * gas_limit
            profit = self.current_profit - gas_cost
            logger.debug(f"Estimated profit: {profit} ETH")
            return profit
        except Exception as e:
            logger.error(f"Error estimating transaction profit: {e}")
            return Decimal("0")
        
    async def withdraw_token(self, token_address: str) -> bool:
        """
        Withdraws a specific token from the flashloan contract.

        :param token_address: Address of the token to withdraw.
        :return: True if successful, else False.
        """
        try:
            withdraw_function = self.aave_flashloan.functions.withdrawToken(
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
            transfer_function = self.aave_flashloan.functions.transfer(
                self.web3.to_checksum_address(account), int(amount * Decimal(self.DEFAULT_PROFIT_TRANSFER_MULTIPLIER))
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

    async def calculate_gas_parameters(
        self,
        tx: Dict[str, Any],
        gas_limit: Optional[int] = None
    ) -> Dict[str, int]:
        """Centralized gas parameter calculation."""
        try:
            gas_params = await self._get_dynamic_gas_parameters()
            estimated_gas = gas_limit or await self.estimate_gas_smart(tx)
            return {
                'gasPrice': gas_params['gasPrice'],
                'gas': int(estimated_gas * 1.1)  # Add 10% buffer
            }
        except Exception as e:
            logger.error(f"Error calculating gas parameters: {e}")
            # Fallback to Default values
            return {
                "gasPrice": int(self.web3.to_wei(self.DEFAULT_GAS_PRICE_GWEI * self.gas_price_multiplier, "gwei")),
                "gas": self.DEFAULT_GAS_LIMIT 
            }

class Main_Core:
    """
    Builds and manages the entire MEV bot, initializing all components,
    managing connections, and orchestrating the main execution loop.
    """
    WEB3_MAX_RETRIES: int = 3
    WEB3_RETRY_DELAY: int = 2
    
    def __init__(self, configuration: "Configuration") -> None:
        # Take first memory snapshot after initialization
        self.memory_snapshot: tracemalloc.Snapshot = tracemalloc.take_snapshot()
        self.configuration: "Configuration" = configuration
        self.web3: Optional[AsyncWeb3] = None
        self.account: Optional[Account] = None
        self.running: bool = False
        self.components: Dict[str, Any] = {
            'api_config': None,
            'nonce_core': None, 
            'safety_net': None,
            'market_monitor': None,
            'mempool_monitor': None,
            'transaction_core': None,
            'strategy_net': None,
        }
        logger.info("Starting 0xBuilder...")
        
    async def _initialize_components(self) -> None:
        """Initialize all components in the correct dependency order."""
        try:
            # 1. First initialize configuration and load ABIs
            await self._load_configuration()
            
            # Load and validate ERC20 ABI
            erc20_abi = await self._load_abi(self.configuration.ERC20_ABI)
            if not erc20_abi:
                raise ValueError("Failed to load ERC20 ABI")
            
            self.web3 = await self._initialize_web3()
            if not self.web3:
                raise RuntimeError("Failed to initialize Web3 connection")

            self.account = Account.from_key(self.configuration.WALLET_KEY)
            await self._check_account_balance()

            # 2. Initialize API config
            self.components['api_config'] = API_Config(self.configuration)
            await self.components['api_config'].initialize()

            # 3. Initialize nonce core
            self.components['nonce_core'] = Nonce_Core(
                self.web3, 
                self.account.address, 
                self.configuration
            )
            await self.components['nonce_core'].initialize()

            # 4. Initialize safety net
            self.components['safety_net'] = Safety_Net(
                self.web3,
                self.configuration,
                self.account,
                self.components['api_config'],
                market_monitor = self.components.get('market_monitor')
            )
            await self.components['safety_net'].initialize()

            # 5. Initialize transaction core
            self.components['transaction_core'] = Transaction_Core(
                self.web3,
                self.account,
                self.configuration.AAVE_FLASHLOAN_ADDRESS,
                self.configuration.AAVE_FLASHLOAN_ABI,
                self.configuration.AAVE_POOL_ADDRESS,
                self.configuration.AAVE_POOL_ABI,
                api_config=self.components['api_config'],
                nonce_core=self.components['nonce_core'],
                safety_net=self.components['safety_net'],
                configuration=self.configuration
            )
            await self.components['transaction_core'].initialize()

            # 6. Initialize market monitor
            self.components['market_monitor'] = Market_Monitor(
                web3=self.web3,
                configuration=self.configuration,
                api_config=self.components['api_config'],
                transaction_core=self.components['transaction_core']
            )
            await self.components['market_monitor'].initialize()

            # 7. Initialize mempool monitor with validated ABI
            self.components['mempool_monitor'] = Mempool_Monitor(
                web3=self.web3,
                safety_net=self.components['safety_net'],
                nonce_core=self.components['nonce_core'],
                api_config=self.components['api_config'],
                monitored_tokens=await self.configuration.get_token_addresses(),
                configuration=self.configuration,
                erc20_abi=erc20_abi,  # Pass the loaded ABI
                market_monitor=self.components['market_monitor']
            )
            await self.components['mempool_monitor'].initialize()

            # 8. Finally initialize strategy net
            self.components['strategy_net'] = Strategy_Net(
                self.components['transaction_core'],
                self.components['market_monitor'],
                self.components['safety_net'],
                self.components['api_config']
            )
            await self.components['strategy_net'].initialize()

            logger.info("All components initialized successfully ✅")

        except Exception as e:
            logger.critical(f"Component initialization failed: {e}")
            raise

    async def initialize(self) -> None:
        """Initialize all components with proper error handling."""
        try:
            before_snapshot = tracemalloc.take_snapshot()
            await self._initialize_components()
            after_snapshot = tracemalloc.take_snapshot()
            
            # Log memory usage
            top_stats = after_snapshot.compare_to(before_snapshot, 'lineno')
            logger.debug("Memory allocation during initialization:")
            for stat in top_stats[:3]:
                logger.debug(str(stat))

            logger.debug("Main Core initialization complete ✅")
            
        except Exception as e:
            logger.critical(f"Main Core initialization failed: {e}")
            raise

    async def _load_configuration(self) -> None:
        """Load all configuration elements in the correct order."""
        try:
            # First load the configuration itself
            await self.configuration.load()
            
            logger.debug("Configuration loaded ✅ ")
        except Exception as e:
            logger.critical(f"Failed to load configuration: {e}")
            raise

    async def _initialize_web3(self) -> Optional[AsyncWeb3]:
        """Initialize Web3 connection with error handling and retries."""
        providers = await self._get_providers()
        if not providers:
            logger.error("No valid endpoints provided!")
            return None

        for provider_name, provider in providers:
            for attempt in range(self.WEB3_MAX_RETRIES):
                try:
                    logger.debug(f"Attempting connection with {provider_name} (attempt {attempt + 1})...")
                    web3 = AsyncWeb3(provider, modules={"eth": (AsyncEth,)})
                    
                    # Test connection with timeout
                    try:
                        async with async_timeout.timeout(10):
                            if await web3.is_connected():
                                chain_id = await web3.eth.chain_id
                                logger.debug(f"Connected to network via {provider_name} (Chain ID: {chain_id})")
                                await self._add_middleware(web3)
                                return web3
                    except asyncio.TimeoutError:
                        logger.warning(f"Connection timeout with {provider_name}")
                        continue
                        
                except Exception as e:
                    logger.warning(f"{provider_name} connection attempt {attempt + 1} failed: {e}")
                    if attempt < self.WEB3_MAX_RETRIES - 1:
                        await asyncio.sleep(self.WEB3_RETRY_DELAY * (attempt + 1))
                    continue

            logger.error(f"All attempts failed for {provider_name}")
            
        logger.error("Failed to initialize Web3 with any provider")
        return None

    async def _get_providers(self) -> List[Tuple[str, Union[AsyncIPCProvider, AsyncHTTPProvider, WebSocketProvider]]]:
        """Get list of available providers with validation."""
        providers = []
    
        if self.configuration.HTTP_ENDPOINT:
            try:
                http_provider = AsyncHTTPProvider(self.configuration.HTTP_ENDPOINT)
                await http_provider.make_request('eth_blockNumber', [])
                providers.append(("HTTP Provider", http_provider))
                logger.info("Linked to Ethereum network via HTTP Provider. ✅")
                return providers
            except Exception as e:
                logger.warning(f"HTTP Provider failed. {e} ❌ - Attempting WebSocket... ")
    
        if self.configuration.WEBSOCKET_ENDPOINT:
            try:
                ws_provider = WebSocketProvider(self.configuration.WEBSOCKET_ENDPOINT)
                await ws_provider.connect()
                providers.append(("WebSocket Provider", ws_provider))
                logger.info("Linked to Ethereum network via WebSocket Provider. ✅")
                return providers
            except Exception as e:
                logger.warning(f"WebSocket Provider failed. {e} ❌ - Attempting IPC... ")
            
        if self.configuration.IPC_ENDPOINT:
            try:
                ipc_provider = AsyncIPCProvider(self.configuration.IPC_ENDPOINT)
                await ipc_provider.make_request('eth_blockNumber', [])
                providers.append(("IPC Provider", ipc_provider))
                logger.info("Linked to Ethereum network via IPC Provider. ✅")
                return providers
            except Exception as e:
                logger.warning(f"IPC Provider failed. {e} ❌ - All providers failed.")

        logger.critical("No more providers are available! ❌")
        return providers
    
    async def _test_connection(self, web3: AsyncWeb3, name: str) -> bool:
        """Test Web3 connection with retries."""
        for attempt in range(3):
            try:
                if await web3.is_connected():
                    chain_id = await web3.eth.chain_id
                    logger.debug(f"Connected to network {name} (Chain ID: {chain_id}) ")
                    return True
            except Exception as e:
                logger.error(f"Connection attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(1)

    async def _add_middleware(self, web3: AsyncWeb3) -> None:
        """Add appropriate middleware based on network."""
        try:
            chain_id = await web3.eth.chain_id
            if (chain_id in {99, 100, 77, 7766, 56}):  # POA networks
                web3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
                logger.debug(f"Injected POA middleware.")
            elif chain_id in {1, 3, 4, 5, 42, 420}:  # ETH networks
                logger.debug(f"ETH network.")
                pass
            else:
                logger.warning(f"Unknown network; no middleware injected.")
        except Exception as e:
            logger.error(f"Middleware configuration failed: {e}")
            raise

    async def _check_account_balance(self) -> None:
        """Check the Ethereum account balancer_router_abi."""
        try:
            if not self.account:
                raise ValueError("Account not initialized")

            balancer_router_abi = await self.web3.eth.get_balance(self.account.address)
            balance_eth = self.web3.from_wei(balancer_router_abi, 'ether')

            logger.debug(f"Account {self.account.address} initialized ")
            logger.debug(f"Balance: {balance_eth:.4f} ETH")

            if balance_eth < 0.01:
                logger.warning(f"Low account balance (<0.01 ETH)")

        except Exception as e:
            logger.error(f"Balance check failed: {e}")
            raise

    async def _initialize_component(self, name: str, component: Any) -> None:
        """Initialize a single component with error handling."""
        try:
            if hasattr(component, 'initialize'):
                await component.initialize()
            self.components[name] = component
            logger.debug(f"Initialized {name} successfully")
        except Exception as e:
            logger.error(f"Failed to initialize {name}: {e}")
            raise

    async def _initialize_monitoring_components(self) -> None:
        """Initialize monitoring components in the correct order."""
        # First initialize market monitor with transaction core
        try:
            await self._initialize_component('market_monitor', Market_Monitor(
                web3=self.web3, 
                configuration=self.configuration, 
                api_config=self.components['api_config'],
                transaction_core=self.components.get('transaction_core')  # Add this
            ))

            # Then initialize mempool monitor with required dependencies
            await self._initialize_component('mempool_monitor', Mempool_Monitor(
                web3=self.web3,
                safety_net=self.components['safety_net'],
                nonce_core=self.components['nonce_core'],
                api_config=self.components['api_config'],
                monitored_tokens=await self.configuration.get_token_addresses(),
                market_monitor=self.components['market_monitor'],
                configuration=self.configuration,
                erc20_abi = await self._load_abi(self.configuration.ERC20_ABI)
            ))

            # 7. Finally initialize strategy net
            await self._initialize_component('strategy_net', Strategy_Net(
                transaction_core=self.components['transaction_core'],
                market_monitor=self.components['market_monitor'],
                safety_net=self.components['safety_net'],
                api_config=self.components['api_config']
            ))

        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            raise
            

    async def run(self) -> None:
        """Main execution loop with improved task management."""
        logger.debug("Starting 0xBuilder...")
        self.running = True

        try:
            if not self.components['mempool_monitor']:
                raise RuntimeError("Mempool monitor not properly initialized")

            # Take initial memory snapshot
            initial_snapshot = tracemalloc.take_snapshot()
            last_memory_check = time.time()
            MEMORY_CHECK_INTERVAL = 300

            # Create task groups for different operations
            async with asyncio.TaskGroup() as tg:
                # Start monitoring task
                monitoring_task = tg.create_task(
                    self.components['mempool_monitor'].start_monitoring()
                )
                
                # Start processing task
                processing_task = tg.create_task(
                    self._process_profitable_transactions()
                )

                # Start memory monitoring task
                memory_task = tg.create_task(
                    self._monitor_memory(initial_snapshot)
                )

            # Tasks will be automatically cancelled when leaving the context
                
        except* asyncio.CancelledError:
            logger.info("Tasks cancelled during shutdown")
        except* Exception as e:
            logger.error(f"Fatal error in run loop: {e}")
        finally:
            await self.stop()

    async def _monitor_memory(self, initial_snapshot: tracemalloc.Snapshot) -> None:
        """Separate task for memory monitoring."""
        while self.running:
            try:
                current_snapshot = tracemalloc.take_snapshot()
                top_stats = current_snapshot.compare_to(initial_snapshot, 'lineno')
                
                logger.debug("Memory allocation changes:")
                for stat in top_stats[:3]:
                    logger.debug(str(stat))
                                        
                await asyncio.sleep(300)  # Check every 5 minutes
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in memory monitoring: {e}")

    async def stop(self) -> None:
        """Gracefully stop all components in the correct order."""
        logger.warning("Shutting down Core...")
        self.running = False

        try:
            shutdown_order = [
                'mempool_monitor',  # Stop monitoring first
                'strategy_net',     # Stop strategies
                'transaction_core', # Stop transactions
                'market_monitor',   # Stop market monitoring
                'safety_net',      # Stop safety checks
                'nonce_core',      # Stop nonce management
                'api_config'       # Stop API connections last
            ]

            # Stop components in parallel where possible
            stop_tasks = []
            for component_name in shutdown_order:
                component = self.components.get(component_name)
                if component and hasattr(component, 'stop'):
                    stop_tasks.append(self._stop_component(component_name, component))
            
            if stop_tasks:
                await asyncio.gather(*stop_tasks, return_exceptions=True)

            # Clean up web3 connection
            if self.web3 and hasattr(self.web3.provider, 'disconnect'):
                await self.web3.provider.disconnect()

            # Final memory snapshot
            final_snapshot = tracemalloc.take_snapshot()
            top_stats = final_snapshot.compare_to(self.memory_snapshot, 'lineno')
            
            logger.debug("Final memory allocation changes:")
            for stat in top_stats[:5]:
                logger.debug(str(stat))

            logger.debug("Core shutdown complete.")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        finally:
            tracemalloc.stop()

    async def _stop_component(self, name: str, component: Any) -> None:
        """Stop a single component with error handling."""
        try:
            await component.stop()
            logger.debug(f"Stopped {name}")
        except Exception as e:
            logger.error(f"Error stopping {name}: {e}")

    async def _process_profitable_transactions(self) -> None:
        """Process profitable transactions from the queue."""
        strategy = self.components['strategy_net']
        monitor = self.components['mempool_monitor']
        
        while self.running:
            try:
                try:
                    tx = await asyncio.wait_for(monitor.profitable_transactions.get(), timeout=1.0)
                    tx_hash = tx.get('tx_hash', 'Unknown')
                    strategy_type = tx.get('strategy_type', 'Unknown')
                except asyncio.TimeoutError:
                    continue
                
                logger.debug(f"Processing transaction {tx_hash} with strategy type {strategy_type}")
                success = await strategy.execute_best_strategy(tx, strategy_type)

                if success:
                    logger.debug(f"Strategy execution successful for tx: {tx_hash}")
                else:
                    logger.warning(f"Strategy execution failed for tx: {tx_hash}")

                # Mark task as done
                monitor.profitable_transactions.task_done()

            except Exception as e:
                logger.error(f"Error processing transaction: {e}")

    async def _load_abi(self, abi_path: str) -> List[Dict[str, Any]]:
       """Load contract ABI from a file with better path handling."""
       try:
            abi_registry = ABI_Registry()
            abi = await abi_registry.load_abi('erc20')
            if not abi:
                raise ValueError("Failed to load ERC20 ABI using ABI Registry")
            return abi
       except Exception as e:
            logger.error(f"Error loading ABI from {abi_path}: {e}")
            raise

# Logging and helper function
class ColorFormatter(logging.Formatter):
    """Custom formatter for colored log output."""
    COLORS: Dict[str, str] = {
        "DEBUG": "\033[94m",    # Blue
        "INFO": "\033[92m",     # Green
        "WARNING": "\033[93m",  # Yellow
        "ERROR": "\033[91m",    # Red
        "CRITICAL": "\033[91m\033[1m", # Bold Red
        "RESET": "\033[0m",     # Reset
    }

    def format(self, record: logging.LogRecord) -> str:
        """Formats a log record with colors."""
        color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        reset = self.COLORS["RESET"]
        record.levelname = f"{color}{record.levelname}{reset}"  # Colorize level name
        record.msg = f"{color}{record.msg}{reset}"              # Colorize message
        return super().format(record)

# Configure the logging once
def configure_logging(level: int = logging.DEBUG) -> None:
    """Configures logging with a colored formatter."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColorFormatter("%(asctime)s [%(levelname)s] %(message)s"))

    logging.basicConfig(
        level=level,  # Global logging level
        handlers=[handler]
    )

# Factory function to get a logger instance
def getLogger(name: Optional[str] = None, level: int = logging.DEBUG) -> logging.Logger:
    """Returns a logger instance, configuring logging if it hasn't been yet."""
    if not logging.getLogger().hasHandlers():
        configure_logging(level)
        
    logger = logging.getLogger(name if name else "0xBuilder")
    return logger

# Initialize the logger globally so it can be used throughout the script
logger = getLogger("0xBuilder")


# Add new cache settings
CACHE_SETTINGS: Dict[str, Dict[str, int]] = {
    'price': {'ttl': 300, 'size': 1000},
    'volume': {'ttl': 900, 'size': 500},
    'volatility': {'ttl': 600, 'size': 200}
}


# Add risk thresholds
RISK_THRESHOLDS: Dict[str, Union[int, float]] = {
    'gas_price': 500,  # Gwei
    'min_profit': 0.01,  # ETH
    'max_slippage': 0.05,  # 5%
    'congestion': 0.8  # 80%
}

# Error codes
ERROR_MARKET_MONITOR_INIT: int = 1001
ERROR_MODEL_LOAD: int = 1002
ERROR_DATA_LOAD: int = 1003
ERROR_MODEL_TRAIN: int = 1004
ERROR_CORE_INIT: int = 1005
ERROR_WEB3_INIT: int = 1006
ERROR_CONFIG_LOAD: int = 1007
ERROR_STRATEGY_EXEC: int = 1008

# Error messages with default fallbacks
ERROR_MESSAGES: Dict[int, str] = {
    ERROR_MARKET_MONITOR_INIT: "Market Monitor initialization failed",
    ERROR_MODEL_LOAD: "Failed to load price prediction model",
    ERROR_DATA_LOAD: "Failed to load historical training data",
    ERROR_MODEL_TRAIN: "Failed to train price prediction model",
    ERROR_CORE_INIT: "Core initialization failed",
    ERROR_WEB3_INIT: "Web3 connection failed",
    ERROR_CONFIG_LOAD: "Configuration loading failed",
    ERROR_STRATEGY_EXEC: "Strategy execution failed",
}

# Add a helper function to get error message with fallback
def get_error_message(code: int, default: str = "Unknown error") -> str:
    """Get error message for error code with fallback to default message."""
    return ERROR_MESSAGES.get(code, default)

class Strategy_Net:
    """Advanced strategy network for MEV operations including front-running, back-running, and sandwich attacks."""
    
    REWARD_BASE_MULTIPLIER: float = -0.1
    REWARD_TIME_PENALTY: float = -0.01


    def __init__(
        self,
        transaction_core: Optional[Any],
        market_monitor: Optional[Any],
        safety_net: Optional[Any],
        api_config: Optional[Any],
    ) -> None:
        self.transaction_core: Optional[Any] = transaction_core
        self.market_monitor: Optional[Any] = market_monitor
        self.safety_net: Optional[Any] = safety_net
        self.api_config: Optional[Any] = api_config

        # Initialize strategy types
        self.strategy_types: List[str] = [
            "eth_transaction",
            "front_run",
            "back_run",
            "sandwich_attack"
        ]

        # Initialize strategy registry before using it
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

        # Initialize performance metrics after strategy registry
        self.strategy_performance: Dict[str, "StrategyPerformanceMetrics"] = {
            strategy_type: StrategyPerformanceMetrics()
            for strategy_type in self.strategy_types
        }

        # Initialize reinforcement weights after strategy registry
        self.reinforcement_weights: Dict[str, np.ndarray] = {
            strategy_type: np.ones(len(self._strategy_registry[strategy_type]))
            for strategy_type in self.strategy_types
        }

        self.configuration: "StrategyConfiguration" = StrategyConfiguration()
        self.history_data: List[Dict[str, Any]] = []

        logger.debug("StrategyNet initialized with enhanced configuration")

    async def initialize(self) -> None:
        """Initialize strategy network with performance metrics and reinforcement weights."""
        try:
            # Initialize performance metrics
            self.strategy_performance = {
                strategy_type: StrategyPerformanceMetrics()
                for strategy_type in self.strategy_types
            }
            # Initialize reinforcement weights
            self.reinforcement_weights = {
                strategy_type: np.ones(len(self.get_strategies(strategy_type)))
                for strategy_type in self.strategy_types
            }
            logger.info("StrategyNet initialized ✅")
        except Exception as e:
            logger.critical(f"Strategy Net initialization failed: {e}")
            raise

    def register_strategy(self, strategy_type: str, strategy_func: Callable[[Dict[str, Any]], asyncio.Future]) -> None:
        """Register a new strategy dynamically.""" 
        if strategy_type not in self.strategy_types:
            logger.warning(f"Attempted to register unknown strategy type: {strategy_type}")
            return
        self._strategy_registry[strategy_type].append(strategy_func)
        self.reinforcement_weights[strategy_type] = np.ones(len(self._strategy_registry[strategy_type]))
        logger.debug(f"Registered new strategy '{strategy_func.__name__}' under '{strategy_type}'")

    def get_strategies(self, strategy_type: str) -> List[Callable[[Dict[str, Any]], asyncio.Future]]:
        """Retrieve strategies for a given strategy type.""" 
        return self._strategy_registry.get(strategy_type, [])

    async def execute_best_strategy(
        self, 
        target_tx: Dict[str, Any], 
        strategy_type: str
    ) -> bool:
        """
        Execute the optimal strategy based on current market conditions and historical performance.
        
        Args:
            target_tx: Target transaction details
            strategy_type: Type of strategy to execute
            
        Returns:
            bool: True if execution was successful, False otherwise
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
        base_reward = float(profit) if success else self.REWARD_BASE_MULTIPLIER
        time_penalty = self.REWARD_TIME_PENALTY * execution_time
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

    # Consolidate duplicate risk assessment methods into one
    async def _assess_risk(
        self,
        tx: Dict[str, Any],
        token_symbol: str,
        price_change: float = 0,
        volume: float = 0
    ) -> Tuple[float, Dict[str, Any]]:
        """Centralized risk assessment for all strategies."""
        try:
            risk_score = 1.0
            market_conditions = await self.market_monitor.check_market_conditions(tx.get("to", ""))
            
            # Gas price impact 
            gas_price = int(tx.get("gasPrice", 0))
            gas_price_gwei = float(self.transaction_core.web3.from_wei(gas_price, "gwei"))
            if gas_price_gwei > 300:
                risk_score *= 0.7

            # Market conditions impact
            if market_conditions.get("high_volatility", False):
                risk_score *= 0.7
            if market_conditions.get("low_liquidity", False):
                risk_score *= 0.6
            if market_conditions.get("bullish_trend", False):
                risk_score *= 1.2
                
            # Price change impact
            if price_change > 0:
                risk_score *= min(1.3, 1 + (price_change / 100))
                
            # Volume impact    
            if volume >= 1_000_000:
                risk_score *= 1.2
            elif volume <= 100_000:
                risk_score *= 0.8

            risk_score = max(0.0, min(1.0, risk_score))
            logger.debug(f"Risk assessment for {token_symbol}: {risk_score:.2f}")
            
            return risk_score, market_conditions
            
        except Exception as e:
            logger.error(f"Error in risk assessment: {e}")
            return 0.0, {}

    # Remove duplicate validation methods and consolidate into one
    async def _validate_transaction(
        self,
        tx: Dict[str, Any],
        strategy_type: str,
        min_value: float = 0
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """Centralized transaction validation for all strategies."""
        try:
            if not isinstance(tx, dict) or not tx:
                logger.debug("Invalid transaction format")
                return False, None, None

            decoded_tx = await self._decode_transaction(tx)
            if not decoded_tx:
                return False, None, None

            # Extract and validate path
            path = decoded_tx.get("params", {}).get("path", [])
            if not path or len(path) < 2:
                logger.debug("Invalid transaction path")
                return False, None, None

            # Get token details
            token_address = path[0] if strategy_type in ["front_run", "sandwich_attack"] else path[-1]
            token_symbol = await self._get_token_symbol(token_address)
            if not token_symbol:
                return False, None, None

            # Validate value if required
            if min_value > 0:
                tx_value = self.transaction_core.web3.from_wei(int(tx.get("value", 0)), "ether")
                if float(tx_value) < min_value:
                    logger.debug(f"Transaction value {tx_value} below minimum {min_value}")
                    return False, None, None

            return True, decoded_tx, token_symbol

        except Exception as e:
            logger.error(f"Transaction validation error: {e}")
            return False, None, None
    
    async def high_value_eth_transfer(self, target_tx: Dict[str, Any]) -> bool:
         """
        Execute high-value ETH transfer strategy with advanced validation and dynamic thresholds.
        
        :param target_tx: Target transaction dictionary
        :return: True if transaction was executed successfully, else False
        """
         logger.info("Initiating High-Value ETH Transfer Strategy...")

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
                logger.debug(
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
        Execute aggressive front-running strategy with dynamic gas pricing and risk assessment.
        
        Args:
            target_tx: Target transaction details
            
        Returns:
            bool: True if front-run was successful, False otherwise
        """
        logger.debug("Initiating Aggressive Front-Run Strategy...")

        # Validate transaction
        valid, decoded_tx, token_symbol = await self._validate_transaction(
            target_tx, "front_run", min_value=0.1
        )
        if not valid:
            return False

        # Assess risk
        risk_score, market_conditions = await self._assess_risk(
            target_tx, 
            token_symbol,
            price_change=await self.api_config.get_price_change_24h(token_symbol)
        )

        if risk_score >= 0.7:  # High confidence threshold
            logger.debug(f"Executing aggressive front-run (Risk: {risk_score:.2f})")
            return await self.transaction_core.front_run(target_tx)

        return False

    async def predictive_front_run(self, target_tx: Dict[str, Any]) -> bool:
        """
        Execute predictive front-run strategy based on advanced price prediction analysis
        and multiple market indicators.
        """
        logger.debug("Initiating Enhanced Predictive Front-Run Strategy...")

        # Validate transaction
        valid, decoded_tx, token_symbol = await self._validate_transaction(
            target_tx, "front_run"
        )
        if not valid:
            return False

        # Gather market data asynchronously
        try:
            data = await asyncio.gather(
                self.market_monitor.predict_price_movement(token_symbol),
                self.api_config.get_real_time_price(token_symbol),
                self.market_monitor.check_market_conditions(target_tx["to"]),
                self.api_config.get_token_price_data(token_symbol, 'historical', timeframe=1),
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

        # Calculate price metrics
        price_change = (predicted_price / float(current_price) - 1) * 100
        volatility = np.std(historical_prices) / np.mean(historical_prices) if historical_prices else 0

        # Score the opportunity (0-100)
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

        # Execute if conditions are favorable
        if opportunity_score >= 75:  # High confidence threshold
            logger.debug(
                f"Executing predictive front-run for {token_symbol} "
                f"(Score: {opportunity_score}/100, Expected Change: {price_change:.2f}%)"
            )
            return await self.transaction_core.front_run(target_tx)

        logger.debug(
            f"Opportunity score {opportunity_score}/100 below threshold. Skipping front-run."
        )
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
        
        # Define score components with weights.
        components = {
           "price_change": {
               "very_strong": {"threshold": 5.0, "points": 40},
               "strong": {"threshold": 3.0, "points": 30},
               "moderate": {"threshold": 1.0, "points": 20},
               "slight": {"threshold": 0.5, "points": 10}
           },
           "volatility": {
               "very_low": {"threshold": 0.02, "points": 20},
               "low": {"threshold": 0.05, "points": 15},
               "moderate": {"threshold": 0.08, "points": 10},
           },
           "market_conditions": {
                "bullish_trend": {"points": 10},
                "not_high_volatility": {"points": 5},
                "not_low_liquidity": {"points": 5},
           },
            "price_trend": {
                "upward": {"points": 20},
                "stable": {"points": 10},
           }
       }

        # Price change component
        if price_change > components["price_change"]["very_strong"]["threshold"]:
            score += components["price_change"]["very_strong"]["points"]
        elif price_change > components["price_change"]["strong"]["threshold"]:
            score += components["price_change"]["strong"]["points"]
        elif price_change > components["price_change"]["moderate"]["threshold"]:
            score += components["price_change"]["moderate"]["points"]
        elif price_change > components["price_change"]["slight"]["threshold"]:
            score += components["price_change"]["slight"]["points"]

        # Volatility component
        if volatility < components["volatility"]["very_low"]["threshold"]:
           score += components["volatility"]["very_low"]["points"]
        elif volatility < components["volatility"]["low"]["threshold"]:
           score += components["volatility"]["low"]["points"]
        elif volatility < components["volatility"]["moderate"]["threshold"]:
           score += components["volatility"]["moderate"]["points"]

        # Market conditions component
        if market_conditions.get("bullish_trend", False):
            score += components["market_conditions"]["bullish_trend"]["points"]
        if not market_conditions.get("high_volatility", True):
            score += components["market_conditions"]["not_high_volatility"]["points"]
        if not market_conditions.get("low_liquidity", True):
            score += components["market_conditions"]["not_low_liquidity"]["points"]

        # Price trend component
        if historical_prices and len(historical_prices) > 1:
            recent_trend = (historical_prices[-1] / historical_prices[0] - 1) * 100
            if recent_trend > 0:
                score += components["price_trend"]["upward"]["points"]
            elif recent_trend > -1:
                score += components["price_trend"]["stable"]["points"]

        logger.debug(f"Calculated opportunity score: {score}/100")
        return score

    async def volatility_front_run(self, target_tx: Dict[str, Any]) -> bool:
        """
        Execute front-run strategy based on market volatility analysis with 
        advanced risk assessment and dynamic thresholds.
        """
        logger.debug("Initiating Enhanced Volatility Front-Run Strategy...")

        # Validate transaction
        valid, decoded_tx, token_symbol = await self._validate_transaction(
            target_tx, "front_run"
        )
        if not valid:
            return False

        # Gather market data asynchronously
        try:
            results = await asyncio.gather(
                self.market_monitor.check_market_conditions(target_tx["to"]),
                self.api_config.get_real_time_price(token_symbol),
                 self.api_config.get_token_price_data(token_symbol, 'historical', timeframe=1),
                return_exceptions=True
            )

            market_conditions, current_price, historical_prices = results

            if any(isinstance(result, Exception) for result in results):
                logger.warning("Failed to gather complete market data")
                return False

        except Exception as e:
            logger.error(f"Error gathering market data: {e}")
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
            logger.debug(
                f"Executing volatility-based front-run for {token_symbol} "
                f"(Volatility Score: {volatility_score:.2f}/100)"
            )
            return await self.transaction_core.front_run(target_tx)

        logger.debug(
            f"Volatility score {volatility_score:.2f}/100 below threshold. Skipping front-run."
        )
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

        # Validate transaction
        valid, decoded_tx, token_symbol = await self._validate_transaction(
            target_tx, "front_run"
        )
        if not valid:
            return False

        # Multi-factor analysis
        try:
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

        except Exception as e:
            logger.error(f"Error gathering market data: {e}")
            return False

        # Advanced decision making
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

        # Execute if conditions are favorable
        if risk_score >= 75:  # Minimum risk score threshold
            logger.debug(
                f"Executing advanced front-run for {token_symbol} "
                f"(Risk Score: {risk_score}/100)"
            )
            return await self.transaction_core.front_run(target_tx)

        logger.debug(
            f"Risk score {risk_score}/100 below threshold. Skipping front-run."
        )
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
        Calculate comprehensive risk score based on multiple market factors.
        
        Args:
            price_increase: Percentage price increase
            is_bullish: Market trend indicator
            is_volatile: Volatility indicator
            has_liquidity: Liquidity indicator
            volume: Trading volume in USD
            
        Returns:
            int: Risk score between 0-100
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

        # Validate transaction
        valid, decoded_tx, token_symbol = await self._validate_transaction(
            target_tx, "back_run"
        )
        if not valid:
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

        # Validate transaction
        valid, decoded_tx, token_symbol = await self._validate_transaction(
            target_tx, "back_run"
        )
        if not valid:
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
            threshold = HIGH_VOLUME_DEFAULT  # Conservative default for unknown tokens

        logger.debug(f"Volume threshold for '{token_symbol}': ${threshold:,.2f} USD")
        return threshold

    async def advanced_back_run(self, target_tx: Dict[str, Any]) -> bool:
        """Execute advanced back-run strategy with comprehensive analysis."""
        logger.debug("Initiating Advanced Back-Run Strategy...")

        # Validate transaction
        valid, decoded_tx, token_symbol = await self._validate_transaction(
            target_tx, "back_run"
        )
        if not valid:
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
            if (gas_price > 200):
                logger.debug(f"Gas price too high for sandwich attack: {gas_price} Gwei")
                return False
            logger.debug(f"Executing sandwich with estimated profit: {estimated_profit:.4f} ETH")
            return await self.transaction_core.execute_sandwich_attack(target_tx)
        logger.debug("Insufficient profit potential for flash sandwich. Skipping.")
        return False

    async def price_boost_sandwich(self, target_tx: Dict[str, Any]) -> bool:
        """Execute sandwich attack strategy based on price momentum."""
        logger.debug("Initiating Price Boost Sandwich Strategy...")

        # Validate transaction
        valid, decoded_tx, token_symbol = await self._validate_transaction(
            target_tx, "sandwich_attack"
        )
        if not valid:
            return False

        historical_prices = await self.api_config.get_token_price_data(token_symbol, 'historical')
        if not historical_prices:
            logger.debug("No historical price data available, skipping price boost sandwich attack")
            return False

        momentum = await self._analyze_price_momentum(historical_prices)
        if momentum > BULLISH_THRESHOLD:
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

        # Validate transaction
        valid, decoded_tx, token_symbol = await self._validate_transaction(
            target_tx, "sandwich_attack"
        )
        if not valid:
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

        # Validate transaction
        valid, decoded_tx, token_symbol = await self._validate_transaction(
            target_tx, "sandwich_attack"
        )
        if not valid:
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

    async def stop(self) -> None:
        """Stop strategy network operations.""" 
        try:
            # Clean up any running strategies
            self.strategy_performance.clear()
            self.reinforcement_weights.clear()
            self.history_data.clear()
            logger.info("Strategy Net stopped successfully.")
        except Exception as e:
            logger.error(f"Error stopping Strategy Net: {e}")

    async def _estimate_profit(self, tx: Any, decoded_params: Dict[str, Any]) -> Decimal:
        """Estimate potential profit from transaction."""
        try:
            # Extract key parameters
            path = decoded_params.get('path', [])
            value = getattr(tx, 'value', 0)
            gas_price = getattr(tx, 'gasPrice', 0)

            # Calculate estimated profit based on path and value
            estimated_profit = await self.transaction_core.estimate_transaction_profit(
                tx, path, value, gas_price
            )
            logger.debug(f"Estimated profit: {estimated_profit:.4f} ETH")
            return estimated_profit
        except Exception as e:
            logger.error(f"Error estimating profit: {e}")
            return Decimal("0")

class StrategyConfiguration:
    """Configuration parameters for strategy execution."""
    
    def __init__(self):
        self.decay_factor: float = 0.95
        self.min_profit_threshold: Decimal = MIN_PROFIT_THRESHOLD
        self.learning_rate: float = 0.01
        self.exploration_rate: float = 0.1

class StrategyPerformanceMetrics:
    """Metrics for tracking strategy performance."""
    successes: int = 0
    failures: int = 0
    profit: Decimal = Decimal("0")
    avg_execution_time: float = 0.0
    success_rate: float = 0.0
    total_executions: int = 0


class StrategyExecutionError(Exception):
    """Custom exception for strategy execution failures."""
    def __init__(self, message: str = "Strategy execution failed"):
        self.message: str = message
        super().__init__(self.message)


class ColorFormatter(logging.Formatter):
    """Custom formatter for colored log output."""
    COLORS = {
        "DEBUG": "\033[94m",    # Blue
        "INFO": "\033[92m",     # Green
        "WARNING": "\033[93m",  # Yellow
        "ERROR": "\033[91m",    # Red
        "CRITICAL": "\033[91m\033[1m", # Bold Red
        "RESET": "\033[0m",     # Reset
    }

    def format(self, record: logging.LogRecord) -> str:
        """Formats a log record with colors."""
        color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        reset = self.COLORS["RESET"]
        record.levelname = f"{color}{record.levelname}{reset}"  # Colorize level name
        record.msg = f"{color}{record.msg}{reset}"              # Colorize message
        return super().format(record)

# Configure the logging once
def configure_logging(level: int = logging.DEBUG) -> None:
    """Configures logging with a colored formatter."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColorFormatter("%(asctime)s [%(levelname)s] %(message)s"))

    logging.basicConfig(
        level=level,  # Global logging level
        handlers=[handler]
    )

# Factory function to get a logger instance
def getLogger(name: Optional[str] = None, level: int = logging.DEBUG) -> logging.Logger:
    """Returns a logger instance, configuring logging if it hasn't been yet."""
    if not logging.getLogger().hasHandlers():
        configure_logging(level)
        
    logger = logging.getLogger(name if name else "0xBuilder")
    return logger

# Initialize the logger globally so it can be used throughout the script
logger = getLogger("0xBuilder")# Add new cache settings
CACHE_SETTINGS = {
    'price': {'ttl': 300, 'size': 1000},
    'volume': {'ttl': 900, 'size': 500},
    'volatility': {'ttl': 600, 'size': 200}
}


# Add risk thresholds
RISK_THRESHOLDS = {
    'gas_price': 500,  # Gwei
    'min_profit': 0.01,  # ETH
    'max_slippage': 0.05,  # 5%
    'congestion': 0.8  # 80%
}

# Error codes
ERROR_MARKET_MONITOR_INIT: int = 1001
ERROR_MODEL_LOAD: int = 1002
ERROR_DATA_LOAD: int = 1003
ERROR_MODEL_TRAIN: int = 1004
ERROR_CORE_INIT: int = 1005
ERROR_WEB3_INIT: int = 1006
ERROR_CONFIG_LOAD: int = 1007
ERROR_STRATEGY_EXEC: int = 1008

# Error messages with default fallbacks
ERROR_MESSAGES: Dict[int, str] = {
    ERROR_MARKET_MONITOR_INIT: "Market Monitor initialization failed",
    ERROR_MODEL_LOAD: "Failed to load price prediction model",
    ERROR_DATA_LOAD: "Failed to load historical training data",
    ERROR_MODEL_TRAIN: "Failed to train price prediction model",
    ERROR_CORE_INIT: "Core initialization failed",
    ERROR_WEB3_INIT: "Web3 connection failed",
    ERROR_CONFIG_LOAD: "Configuration loading failed",
    ERROR_STRATEGY_EXEC: "Strategy execution failed",
}

# Add a helper function to get error message with fallback
def get_error_message(code: int, default: str = "Unknown error") -> str:
    """Get error message for error code with fallback to default message."""
    return ERROR_MESSAGES.get(code, default)



class Main_Core:
    """
    Builds and manages the entire MEV bot, initializing all components,
    managing connections, and orchestrating the main execution loop.
    """
    WEB3_MAX_RETRIES: int = 3
    WEB3_RETRY_DELAY: int = 2
    
    def __init__(self, configuration: "Configuration") -> None:
        # Take first memory snapshot after initialization
        self.memory_snapshot: tracemalloc.Snapshot = tracemalloc.take_snapshot()
        self.configuration: "Configuration" = configuration
        self.web3: Optional[AsyncWeb3] = None
        self.account: Optional[Account] = None
        self.running: bool = False
        self.components: Dict[str, Any] = {
            'api_config': None,
            'nonce_core': None, 
            'safety_net': None,
            'market_monitor': None,
            'mempool_monitor': None,
            'transaction_core': None,
            'strategy_net': None,
        }
        logger.info("Starting 0xBuilder...")
        
    async def _initialize_components(self) -> None:
        """Initialize all components in the correct dependency order."""
        try:
             # 1. First initialize configuration and load ABIs
            await self._load_configuration()

            # Initialize ABI Registry and load ABIs
            await self.configuration.abi_registry.initialize()
            
            # Load and validate ERC20 ABI
            erc20_abi = await self._load_abi(self.configuration.ERC20_ABI)
            if not erc20_abi:
                raise ValueError("Failed to load ERC20 ABI")
            
            self.web3 = await self._initialize_web3()
            if not self.web3:
                raise RuntimeError("Failed to initialize Web3 connection")

            self.account = Account.from_key(self.configuration.WALLET_KEY)
            await self._check_account_balance()

            # 2. Initialize API config
            self.components['api_config'] = API_Config(self.configuration)
            await self.components['api_config'].initialize()

            # 3. Initialize nonce core
            self.components['nonce_core'] = Nonce_Core(
                self.web3, 
                self.account.address, 
                self.configuration
            )
            await self.components['nonce_core'].initialize()

            # 4. Initialize safety net
            self.components['safety_net'] = Safety_Net(
                self.web3,
                self.configuration,
                self.account,
                self.components['api_config'],
                market_monitor = self.components.get('market_monitor')
            )
            await self.components['safety_net'].initialize()

            # 5. Initialize transaction core
            self.components['transaction_core'] = Transaction_Core(
                self.web3,
                self.account,
                self.configuration.AAVE_FLASHLOAN_ADDRESS,
                self.configuration.AAVE_FLASHLOAN_ABI,
                self.configuration.AAVE_POOL_ADDRESS,
                self.configuration.AAVE_POOL_ABI,
                api_config=self.components['api_config'],
                nonce_core=self.components['nonce_core'],
                safety_net=self.components['safety_net'],
                configuration=self.configuration
            )
            await self.components['transaction_core'].initialize()

            # 6. Initialize market monitor
            self.components['market_monitor'] = Market_Monitor(
                web3=self.web3,
                configuration=self.configuration,
                api_config=self.components['api_config'],
                transaction_core=self.components['transaction_core']
            )
            await self.components['market_monitor'].initialize()

            # 7. Initialize mempool monitor with validated ABI
            self.components['mempool_monitor'] = Mempool_Monitor(
                web3=self.web3,
                safety_net=self.components['safety_net'],
                nonce_core=self.components['nonce_core'],
                api_config=self.components['api_config'],
                monitored_tokens=await self.configuration.get_token_addresses(),
                configuration=self.configuration,
                erc20_abi=erc20_abi,  # Pass the loaded ABI
                market_monitor=self.components['market_monitor']
            )
            await self.components['mempool_monitor'].initialize()

            # 8. Finally initialize strategy net
            self.components['strategy_net'] = Strategy_Net(
                self.components['transaction_core'],
                self.components['market_monitor'],
                self.components['safety_net'],
                self.components['api_config']
            )
            await self.components['strategy_net'].initialize()

            logger.info("All components initialized successfully ✅")

        except Exception as e:
            logger.critical(f"Component initialization failed: {e}")
            raise

    async def initialize(self) -> None:
        """Initialize all components with proper error handling."""
        try:
            before_snapshot = tracemalloc.take_snapshot()
            await self._initialize_components()
            after_snapshot = tracemalloc.take_snapshot()
            
            # Log memory usage
            top_stats = after_snapshot.compare_to(before_snapshot, 'lineno')
            logger.debug("Memory allocation during initialization:")
            for stat in top_stats[:3]:
                logger.debug(str(stat))

            logger.debug("Main Core initialization complete ✅")
            
        except Exception as e:
            logger.critical(f"Main Core initialization failed: {e}")
            raise

    async def _load_configuration(self) -> None:
        """Load all configuration elements in the correct order."""
        try:
            # First load the configuration itself
            await self.configuration.load()
            
            logger.debug("Configuration loaded ✅ ")
        except Exception as e:
            logger.critical(f"Failed to load configuration: {e}")
            raise

    async def _initialize_web3(self) -> Optional[AsyncWeb3]:
        """Initialize Web3 connection with error handling and retries."""
        providers = await self._get_providers()
        if not providers:
            logger.error("No valid endpoints provided!")
            return None

        for provider_name, provider in providers:
            for attempt in range(self.WEB3_MAX_RETRIES):
                try:
                    logger.debug(f"Attempting connection with {provider_name} (attempt {attempt + 1})...")
                    web3 = AsyncWeb3(provider, modules={"eth": (AsyncEth,)})
                    
                    # Test connection with timeout
                    try:
                        async with async_timeout.timeout(10):
                            if await web3.is_connected():
                                chain_id = await web3.eth.chain_id
                                logger.debug(f"Connected to network via {provider_name} (Chain ID: {chain_id})")
                                await self._add_middleware(web3)
                                return web3
                    except asyncio.TimeoutError:
                        logger.warning(f"Connection timeout with {provider_name}")
                        continue
                        
                except Exception as e:
                    logger.warning(f"{provider_name} connection attempt {attempt + 1} failed: {e}")
                    if attempt < self.WEB3_MAX_RETRIES - 1:
                        await asyncio.sleep(self.WEB3_RETRY_DELAY * (attempt + 1))
                    continue

            logger.error(f"All attempts failed for {provider_name}")
            
        logger.error("Failed to initialize Web3 with any provider")
        return None

    async def _get_providers(self) -> List[Tuple[str, Union[AsyncIPCProvider, AsyncHTTPProvider, WebSocketProvider]]]:
        """Get list of available providers with validation."""
        providers = []
    
        if self.configuration.HTTP_ENDPOINT:
            try:
                http_provider = AsyncHTTPProvider(self.configuration.HTTP_ENDPOINT)
                await http_provider.make_request('eth_blockNumber', [])
                providers.append(("HTTP Provider", http_provider))
                logger.info("Linked to Ethereum network via HTTP Provider. ✅")
                return providers
            except Exception as e:
                logger.warning(f"HTTP Provider failed. {e} ❌ - Attempting WebSocket... ")
    
        if self.configuration.WEBSOCKET_ENDPOINT:
            try:
                ws_provider = WebSocketProvider(self.configuration.WEBSOCKET_ENDPOINT)
                await ws_provider.connect()
                providers.append(("WebSocket Provider", ws_provider))
                logger.info("Linked to Ethereum network via WebSocket Provider. ✅")
                return providers
            except Exception as e:
                logger.warning(f"WebSocket Provider failed. {e} ❌ - Attempting IPC... ")
            
        if self.configuration.IPC_ENDPOINT:
            try:
                ipc_provider = AsyncIPCProvider(self.configuration.IPC_ENDPOINT)
                await ipc_provider.make_request('eth_blockNumber', [])
                providers.append(("IPC Provider", ipc_provider))
                logger.info("Linked to Ethereum network via IPC Provider. ✅")
                return providers
            except Exception as e:
                logger.warning(f"IPC Provider failed. {e} ❌ - All providers failed.")

        logger.critical("No more providers are available! ❌")
        return providers
    
    async def _test_connection(self, web3: AsyncWeb3, name: str) -> bool:
        """Test Web3 connection with retries."""
        for attempt in range(3):
            try:
                if await web3.is_connected():
                    chain_id = await web3.eth.chain_id
                    logger.debug(f"Connected to network {name} (Chain ID: {chain_id}) ")
                    return True
            except Exception as e:
                logger.error(f"Connection attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(1)

    async def _add_middleware(self, web3: AsyncWeb3) -> None:
        """Add appropriate middleware based on network."""
        try:
            chain_id = await web3.eth.chain_id
            if (chain_id in {99, 100, 77, 7766, 56}):  # POA networks
                web3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
                logger.debug(f"Injected POA middleware.")
            elif chain_id in {1, 3, 4, 5, 42, 420}:  # ETH networks
                logger.debug(f"ETH network.")
                pass
            else:
                logger.warning(f"Unknown network; no middleware injected.")
        except Exception as e:
            logger.error(f"Middleware configuration failed: {e}")
            raise

    async def _check_account_balance(self) -> None:
        """Check the Ethereum account balancer_router_abi."""
        try:
            if not self.account:
                raise ValueError("Account not initialized")

            balancer_router_abi = await self.web3.eth.get_balance(self.account.address)
            balance_eth = self.web3.from_wei(balancer_router_abi, 'ether')

            logger.debug(f"Account {self.account.address} initialized ")
            logger.debug(f"Balance: {balance_eth:.4f} ETH")

            if balance_eth < 0.01:
                logger.warning(f"Low account balance (<0.01 ETH)")

        except Exception as e:
            logger.error(f"Balance check failed: {e}")
            raise

    async def _initialize_component(self, name: str, component: Any) -> None:
        """Initialize a single component with error handling."""
        try:
            if hasattr(component, 'initialize'):
                await component.initialize()
            self.components[name] = component
            logger.debug(f"Initialized {name} successfully")
        except Exception as e:
            logger.error(f"Failed to initialize {name}: {e}")
            raise

    async def _initialize_monitoring_components(self) -> None:
        """Initialize monitoring components in the correct order."""
        # First initialize market monitor with transaction core
        try:
            await self._initialize_component('market_monitor', Market_Monitor(
                web3=self.web3, 
                configuration=self.configuration, 
                api_config=self.components['api_config'],
                transaction_core=self.components.get('transaction_core')  # Add this
            ))

            # Then initialize mempool monitor with required dependencies
            await self._initialize_component('mempool_monitor', Mempool_Monitor(
                web3=self.web3,
                safety_net=self.components['safety_net'],
                nonce_core=self.components['nonce_core'],
                api_config=self.components['api_config'],
                monitored_tokens=await self.configuration.get_token_addresses(),
                market_monitor=self.components['market_monitor'],
                configuration=self.configuration,
                erc20_abi = await self._load_abi(self.configuration.ERC20_ABI)
            ))

            # 7. Finally initialize strategy net
            await self._initialize_component('strategy_net', Strategy_Net(
                transaction_core=self.components['transaction_core'],
                market_monitor=self.components['market_monitor'],
                safety_net=self.components['safety_net'],
                api_config=self.components['api_config']
            ))

        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            raise
            

    async def run(self) -> None:
        """Main execution loop with improved task management."""
        logger.debug("Starting 0xBuilder...")
        self.running = True

        try:
            if not self.components['mempool_monitor']:
                raise RuntimeError("Mempool monitor not properly initialized")

            # Take initial memory snapshot
            initial_snapshot = tracemalloc.take_snapshot()
            last_memory_check = time.time()
            MEMORY_CHECK_INTERVAL = 300

            # Create task groups for different operations
            async with asyncio.TaskGroup() as tg:
                # Start monitoring task
                monitoring_task = tg.create_task(
                    self.components['mempool_monitor'].start_monitoring()
                )
                
                # Start processing task
                processing_task = tg.create_task(
                    self._process_profitable_transactions()
                )

                # Start memory monitoring task
                memory_task = tg.create_task(
                    self._monitor_memory(initial_snapshot)
                )

            # Tasks will be automatically cancelled when leaving the context
                
        except* asyncio.CancelledError:
            logger.info("Tasks cancelled during shutdown")
        except* Exception as e:
            logger.error(f"Fatal error in run loop: {e}")
        finally:
            await self.stop()

    async def _monitor_memory(self, initial_snapshot: tracemalloc.Snapshot) -> None:
        """Separate task for memory monitoring."""
        while self.running:
            try:
                current_snapshot = tracemalloc.take_snapshot()
                top_stats = current_snapshot.compare_to(initial_snapshot, 'lineno')
                
                logger.debug("Memory allocation changes:")
                for stat in top_stats[:3]:
                    logger.debug(str(stat))
                    
                await asyncio.sleep(300)  # Check every 5 minutes
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in memory monitoring: {e}")

    async def stop(self) -> None:
        """Gracefully stop all components in the correct order."""
        logger.warning("Shutting down Core...")
        self.running = False

        try:
            shutdown_order = [
                'mempool_monitor',  # Stop monitoring first
                'strategy_net',     # Stop strategies
                'transaction_core', # Stop transactions
                'market_monitor',   # Stop market monitoring
                'safety_net',      # Stop safety checks
                'nonce_core',      # Stop nonce management
                'api_config'       # Stop API connections last
            ]

            # Stop components in parallel where possible
            stop_tasks = []
            for component_name in shutdown_order:
                component = self.components.get(component_name)
                if component and hasattr(component, 'stop'):
                    stop_tasks.append(self._stop_component(component_name, component))
            
            if stop_tasks:
                await asyncio.gather(*stop_tasks, return_exceptions=True)

            # Clean up web3 connection
            if self.web3 and hasattr(self.web3.provider, 'disconnect'):
                await self.web3.provider.disconnect()

            # Final memory snapshot
            final_snapshot = tracemalloc.take_snapshot()
            top_stats = final_snapshot.compare_to(self.memory_snapshot, 'lineno')
            
            logger.debug("Final memory allocation changes:")
            for stat in top_stats[:5]:
                logger.debug(str(stat))

            logger.debug("Core shutdown complete.")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        finally:
            tracemalloc.stop()

    async def _stop_component(self, name: str, component: Any) -> None:
        """Stop a single component with error handling."""
        try:
            await component.stop()
            logger.debug(f"Stopped {name}")
        except Exception as e:
            logger.error(f"Error stopping {name}: {e}")

    async def _process_profitable_transactions(self) -> None:
        """Process profitable transactions from the queue."""
        strategy = self.components['strategy_net']
        monitor = self.components['mempool_monitor']
        
        while self.running:
            try:
                try:
                    tx = await asyncio.wait_for(monitor.profitable_transactions.get(), timeout=1.0)
                    tx_hash = tx.get('tx_hash', 'Unknown')
                    strategy_type = tx.get('strategy_type', 'Unknown')
                except asyncio.TimeoutError:
                    continue
                
                logger.debug(f"Processing transaction {tx_hash} with strategy type {strategy_type}")
                success = await strategy.execute_best_strategy(tx, strategy_type)

                if success:
                    logger.debug(f"Strategy execution successful for tx: {tx_hash}")
                else:
                    logger.warning(f"Strategy execution failed for tx: {tx_hash}")

                # Mark task as done
                monitor.profitable_transactions.task_done()

            except Exception as e:
                logger.error(f"Error processing transaction: {e}")

    async def _load_abi(self, abi_path: str) -> List[Dict[str, Any]]:
        """Load contract ABI from a file with better path handling."""
        try:
            abi_registry = ABI_Registry()
            abi = await abi_registry.load_abi('erc20')
            if not abi:
                raise ValueError("Failed to load ERC20 ABI using ABI Registry")
            return abi
        except Exception as e:
            logger.error(f"Error loading ABI from {abi_path}: {e}")
            raise


# Initialize the logger before everything else
logger = getLogger("0xBuilder")

async def main():
    """Main entry point with graceful shutdown handling."""
    loop = asyncio.get_running_loop()
    shutdown_handler_task: Optional[asyncio.Task] = None


    try:
        # Start memory tracking
        tracemalloc.start()
        logger.info("Starting 0xBuilder...")

        # Initialize configuration
        configuration = Configuration()
        
        # Create and initialize main core
        core = Main_Core(configuration)

        # Initialize and run
        await core.initialize()
        await core.run()

    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        if tracemalloc.is_tracing():
            snapshot = tracemalloc.take_snapshot()
            logger.debug("Top 10 memory allocations:")
            for stat in snapshot.statistics('lineno')[:10]:
                logger.debug(str(stat))
    finally: 
            # Stop memory tracking
            tracemalloc.stop()
            logger.info("0xBuilder shutdown complete")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        # Get current memory snapshot on error
        snapshot = tracemalloc.take_snapshot()
        logger.critical(f"Program terminated with an error: {e}")
        logger.debug("Top 10 memory allocations at error:")
        top_stats = snapshot.statistics('lineno')
        for stat in top_stats[:10]:
            logger.debug(str(stat))