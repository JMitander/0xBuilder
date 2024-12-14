import os
import json
import time
from typing import Any, Dict, List, Optional
import logging

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

#//////////////////////////////////////////////////////////////////////////////