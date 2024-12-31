import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import aiofiles
import dotenv

# Initialize logger (this can be moved to a dedicated logging module later)
logger = logging.getLogger("0xBuilder")

class Configuration:
    """
    Loads configuration from environment variables and JSON files.
    Provides methods to access and validate configuration data.
    """

    def __init__(self, env_path: Optional[str] = None) -> None:
        """
        Initialize Configuration with default values.
        Args:
            env_path: Optional path to the .env file. Defaults to the current directory.
        """
        self.env_path = env_path if env_path else ".env"
        self._load_env()
        # General settings
        self.MAX_GAS_PRICE: int = self._get_env_int("MAX_GAS_PRICE", 100_000_000_000)  # 100 Gwei in wei
        self.GAS_LIMIT: int = self._get_env_int("GAS_LIMIT", 1_000_000)
        self.MAX_SLIPPAGE: float = self._get_env_float("MAX_SLIPPAGE", 0.01)
        self.MIN_PROFIT: float = self._get_env_float("MIN_PROFIT", 0.001)
        self.MIN_BALANCE: float = self._get_env_float("MIN_BALANCE", 0.000001)

        # API Keys and Endpoints
        self.ETHERSCAN_API_KEY: str = self._get_env_str("ETHERSCAN_API_KEY")
        self.INFURA_PROJECT_ID: str = self._get_env_str("INFURA_PROJECT_ID")
        self.INFURA_API_KEY: str = self._get_env_str("INFURA_API_KEY")
        self.COINGECKO_API_KEY: str = self._get_env_str("COINGECKO_API_KEY")
        self.COINMARKETCAP_API_KEY: str = self._get_env_str("COINMARKETCAP_API_KEY")
        self.CRYPTOCOMPARE_API_KEY: str = self._get_env_str("CRYPTOCOMPARE_API_KEY")
        self.HTTP_ENDPOINT: Optional[str] = self._get_env_str("HTTP_ENDPOINT", None)
        self.WEBSOCKET_ENDPOINT: Optional[str] = self._get_env_str("WEBSOCKET_ENDPOINT", None)
        self.IPC_ENDPOINT: Optional[str] = self._get_env_str("IPC_ENDPOINT", None)
        
        # Account
        self.WALLET_ADDRESS: str = self._get_env_str("WALLET_ADDRESS")
        self.WALLET_KEY: str = self._get_env_str("WALLET_KEY")

        # Paths
        self.BASE_PATH: Path = Path(__file__).parent.parent
        self.ERC20_ABI: Path = self._resolve_path("ERC20_ABI")
        self.AAVE_FLASHLOAN_ABI: Path = self._resolve_path("AAVE_FLASHLOAN_ABI")
        self.AAVE_POOL_ABI: Path = self._resolve_path("AAVE_POOL_ABI")
        self.UNISWAP_ABI: Path = self._resolve_path("UNISWAP_ABI")
        self.SUSHISWAP_ABI: Path = self._resolve_path("SUSHISWAP_ABI")
        self.ERC20_SIGNATURES: Path = self._resolve_path("ERC20_SIGNATURES")
        self.TOKEN_ADDRESSES: Path = self._resolve_path("TOKEN_ADDRESSES")
        self.TOKEN_SYMBOLS: Path = self._resolve_path("TOKEN_SYMBOLS")

        # Router Addresses
        self.UNISWAP_ADDRESS: str = self._get_env_str("UNISWAP_ADDRESS")
        self.SUSHISWAP_ADDRESS: str = self._get_env_str("SUSHISWAP_ADDRESS")
        self.AAVE_POOL_ADDRESS: str = self._get_env_str("AAVE_POOL_ADDRESS")
        self.AAVE_FLASHLOAN_ADDRESS: str = self._get_env_str("AAVE_FLASHLOAN_ADDRESS")
    
        # Set default values for strategies
        self.SLIPPAGE_DEFAULT: float = 0.1
        self.SLIPPAGE_MIN: float = 0.01
        self.SLIPPAGE_MAX: float = 0.5
        self.SLIPPAGE_HIGH_CONGESTION: float = 0.05
        self.SLIPPAGE_LOW_CONGESTION: float = 0.2
        self.MAX_GAS_PRICE_GWEI: int = 500
        self.MIN_PROFIT_MULTIPLIER: float = 2.0
        self.BASE_GAS_LIMIT: int = 21000
        
        # Add ML model configuration
        self.MODEL_RETRAINING_INTERVAL: int = 3600  # 1 hour
        self.MIN_TRAINING_SAMPLES: int = 100
        self.MODEL_ACCURACY_THRESHOLD: float = 0.7
        self.PREDICTION_CACHE_TTL: int = 300  # 5 minutes
        self.LINEAR_REGRESSION_PATH: str = str(self.BASE_PATH / "linear_regression")
        self.MODEL_PATH: str = str(self.BASE_PATH / "linear_regression" / "price_model.joblib")
        self.TRAINING_DATA_PATH: str = str(self.BASE_PATH / "linear_regression" / "training_data.csv")
        
        # WETH and USDC addresses
        self.WETH_ADDRESS: str = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"  # Mainnet WETH
        self.USDC_ADDRESS: str = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"  # Mainnet USDC
        self.USDT_ADDRESS: str = "0xdAC17F958D2ee523a2206206994597C13D831ec7"  # Mainnet USDT

    def _load_env(self) -> None:
        """Load environment variables from .env file."""
        dotenv.load_dotenv(dotenv_path=self.env_path)
        logger.debug(f"Environment variables loaded from: {self.env_path}")

    def _validate_ethereum_address(self, address: str, var_name: str) -> str:
        """Validate Ethereum address format."""
        if not isinstance(address, str):
            raise ValueError(f"Invalid {var_name}: must be a string")
        
        # Remove '0x' prefix if present for length check
        clean_address = address.lower().replace('0x', '')
        
        if len(clean_address) != 40:
            raise ValueError(f"Invalid {var_name} length: {address}")
            
        try:
            # Check if address contains only valid hex characters
            int(clean_address, 16)
            return f"0x{clean_address}"
        except ValueError:
            raise ValueError(f"Invalid hex format for {var_name}: {address}")

    def _get_env_str(self, var_name: str, default: Optional[str] = None) -> str:
        """Get an environment variable as string, raising error if missing."""
        value = os.getenv(var_name, default)
        if value is None:
            logger.error(f"Missing environment variable: {var_name}")
            raise ValueError(f"Missing environment variable: {var_name}")
            
        # Only validate Ethereum addresses for contract addresses
        if 'ADDRESS' in var_name and not any(path_suffix in var_name for path_suffix in ['ABI', 'PATH', 'ADDRESSES', 'SIGNATURES', 'SYMBOLS']):
            return self._validate_ethereum_address(value, var_name)
            
        return value

    def _parse_numeric_string(self, value: str) -> str:
        """Remove underscores and comments from numeric strings."""
        # Remove any comments (starting with #)
        value = value.split('#')[0].strip()
        # Remove underscores from the number
        return value.replace('_', '')

    def _get_env_int(self, var_name: str, default: Optional[int] = None) -> int:
        """Get an environment variable as int, raising error if missing or invalid."""
        try:
            value = os.getenv(var_name)
            if value is None:
                if default is None:
                    raise ValueError(f"Missing environment variable: {var_name}")
                return default
            # Clean the value before parsing
            cleaned_value = self._parse_numeric_string(value)
            return int(cleaned_value)
        except ValueError as e:
            logger.error(f"Invalid or missing integer environment variable: {var_name} - {e}")
            raise

    def _get_env_float(self, var_name: str, default: Optional[float] = None) -> float:
        """Get an environment variable as float, raising error if missing or invalid."""
        try:
            value = os.getenv(var_name)
            if value is None:
                if default is None:
                    raise ValueError(f"Missing environment variable: {var_name}")
                return default
            # Clean the value before parsing
            cleaned_value = self._parse_numeric_string(value)
            return float(cleaned_value)
        except ValueError as e:
            logger.error(f"Invalid or missing float environment variable: {var_name} - {e}")
            raise

    def _resolve_path(self, path_env_var: str) -> Path:
        """Resolve a path from environment variable and ensure the file exists."""
        path_str = self._get_env_str(path_env_var)
        full_path = self.BASE_PATH / path_str
        if not full_path.exists():
            logger.error(f"File not found: {full_path}")
            raise FileNotFoundError(f"File not found: {full_path}")
        logger.debug(f"Resolved path: {full_path}")
        return full_path

    async def _load_json(self, file_path: Path, description: str) -> Any:
        """Load JSON data from a file with proper async."""
        try:
             async with aiofiles.open(file_path, 'r') as f:
                data = json.loads(await f.read())
                logger.debug(f"Successfully loaded {description} from {file_path}")
                return data
        except FileNotFoundError as e:
            logger.error(f"File not found for {description}: {file_path} - {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for {description} in file {file_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading {description} from {file_path}: {e}")
            raise

    async def get_token_addresses(self) -> List[str]:
         """Get the list of monitored token addresses from the config file."""
         data = await self._load_json(self.TOKEN_ADDRESSES, "monitored tokens")
         if not isinstance(data, dict):
             logger.error("Invalid format for token addresses: must be a dictionary")
             raise ValueError("Invalid format for token addresses: must be a dictionary")
             
         # Extract addresses from dictionary keys
         addresses = list(data.keys())
         logger.debug(f"Loaded {len(addresses)} token addresses")
         return addresses

    async def get_token_symbols(self) -> Dict[str, str]:
        """Get the mapping of token addresses to symbols from the config file."""
        data = await self._load_json(self.TOKEN_SYMBOLS, "token symbols")
        if not isinstance(data, dict):
            logger.error("Invalid format for token symbols: must be a dict")
            raise ValueError("Invalid format for token symbols: must be a dict")
        return data

    async def get_erc20_signatures(self) -> Dict[str, str]:
        """Load ERC20 function signatures from JSON."""
        data = await self._load_json(self.ERC20_SIGNATURES, "ERC20 function signatures")
        if not isinstance(data, dict):
            logger.error("Invalid format for ERC20 signatures: must be a dict")
            raise ValueError("Invalid format for ERC20 signatures: must be a dict")
        return data

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """
        Safe configuration value access with default.
        """
        try:
            return getattr(self, key, default)
        except AttributeError:
            logger.warning(f"Configuration key '{key}' not found, using default: {default}")
            return default
        
    def get_all_config_values(self) -> Dict[str, Any]:
         """Returns all configuration values as a dictionary."""
         return vars(self)

    async def load_abi_from_path(self, path: Path) -> List[Dict[str, Any]]:
        """Load and return ABI content from a file path."""
        try:
            async with aiofiles.open(path, 'r') as f:
                content = await f.read()
                abi = json.loads(content)
                if not isinstance(abi, list):
                    raise ValueError(f"Invalid ABI format in {path}: not a list")
                return abi
        except Exception as e:
            logger.error(f"Failed to load ABI from {path}: {e}")
            raise

    async def load(self) -> None:
        """Load and validate all configuration data."""
        try:
            # Create required directories if they don't exist
            os.makedirs(self.LINEAR_REGRESSION_PATH, exist_ok=True)
            os.makedirs(self.BASE_PATH / "abi", exist_ok=True)
            os.makedirs(self.BASE_PATH / "utils", exist_ok=True)

            # Load critical ABIs
            self.AAVE_FLASHLOAN_ABI = await self.load_abi_from_path(self._resolve_path("AAVE_FLASHLOAN_ABI"))
            self.AAVE_POOL_ABI = await self.load_abi_from_path(self._resolve_path("AAVE_POOL_ABI"))
            
            # Validate API keys are set
            required_keys = [
                'ETHERSCAN_API_KEY',
                'INFURA_API_KEY',
                'COINGECKO_API_KEY',
                'COINMARKETCAP_API_KEY',
                'CRYPTOCOMPARE_API_KEY'
            ]
            
            for key in required_keys:
                if not getattr(self, key):
                    logger.warning(f"Missing API key: {key}")

            logger.info("Configuration loaded successfully")

        except Exception as e:
            logger.error(f"Configuration load failed: {e}")
            raise