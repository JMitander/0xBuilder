import os
import sys
import dotenv
import time
import logging
import json
import asyncio
import aiofiles
import aiohttp
import joblib
import plotly.express as px 
import numpy as np
import tracemalloc
import hexbytes
import pandas as pd
from cachetools import TTLCache, cached
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from decimal import Decimal
from typing import Set, List, Dict, Any, Optional
from eth_account.messages import encode_defunct
from web3.exceptions import TransactionNotFound, ContractLogicError
from web3 import AsyncWeb3, AsyncIPCProvider, AsyncHTTPProvider, Web3
from web3.middleware import SignAndSendRawMiddlewareBuilder, ExtraDataToPOAMiddleware
from web3.geth import isinstance
from web3.eth import AsyncEth, Contract
from eth_account import Account

dotenv.load_dotenv()

async def loading_bar(message: str, total_time: int):
    bar_length = 20
    try:
        for i in range(101):
            time.sleep(total_time / 100)
            percent = i / 100
            bar = '█' * int(percent * bar_length) + '-' * (bar_length - int(percent * bar_length))
            print(f"\r{message} [{bar}] {i}%", end='', flush=True)
        print()
    except Exception as e:
        print(f"\r{message} [{'█' * bar_length}] 100% - ❌ Error: {e}", flush=True)

async def setup_logging():
    # Create a root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Set the lowest level for the logger
    tracemalloc.start()  # Start memory tracing
    # Create console handler for INFO level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)  # Console logs only INFO and above
    console_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    # Create file handler for DEBUG level
    file_handler = logging.FileHandler("0xplorer_log.txt", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)  # File logs DEBUG and above
    file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    # Add both handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.info("Logging setup completed. 📝✅")

class Config:
    """
    Loads configuration from environment variables and monitored tokens from a JSON file.
    """
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    async def load(self):
        await self._load_config()

    async def _load_config(self):
        try:
            # API Keys
            await loading_bar("Loading Environment Variables", 2)
            self.ETHERSCAN_API_KEY = self._get_env_variable("ETHERSCAN_API_KEY")
            self.INFURA_PROJECT_ID = self._get_env_variable("INFURA_PROJECT_ID")
            self.COINGECKO_API_KEY = self._get_env_variable("COINGECKO_API_KEY")
            self.COINMARKETCAP_API_KEY = self._get_env_variable("COINMARKETCAP_API_KEY")
            self.CRYPTOCOMPARE_API_KEY = self._get_env_variable("CRYPTOCOMPARE_API_KEY")
            # Providers and Account
            self.HTTP_ENDPOINT = self._get_env_variable("HTTP_ENDPOINT")
            self.IPC_ENDPOINT = self._get_env_variable("IPC_ENDPOINT")
            self.WEBSOCKET_ENDPOINT = self._get_env_variable("WEBSOCKET_ENDPOINT")
            self.WALLET_KEY = self._get_env_variable("WALLET_KEY")
            self.WALLET_ADDRESS = self._get_env_variable("WALLET_ADDRESS")
            # JSON Elements
            self.AAVE_V3_LENDING_POOL_ADDRESS = self._get_env_variable(
                "AAVE_V3_LENDING_POOL_ADDRESS"
            )
            self.TOKEN_ADDRESSES = await self._load_monitored_tokens(
                self._get_env_variable("TOKEN_ADDRESSES")
            )
            self.TOKEN_SYMBOLS = self._get_env_variable("TOKEN_SYMBOLS")
            self.ERC20_ABI = await self._construct_ABI_path("ABI", "erc20_ABI.json")
            self.ERC20_SIGNATURES = await self._load_erc20_function_signatures(
                self._get_env_variable("ERC20_SIGNATURES")
            )
            self.SUSHISWAP_ROUTER_ABI = await self._construct_ABI_path(
                "ABI", "sushiswap_router_ABI.json"
            )
            self.SUSHISWAP_ROUTER_ADDRESS = self._get_env_variable(
                "SUSHISWAP_ROUTER_ADDRESS"
            )
            self.UNISWAP_V2_ROUTER_ABI = await self._construct_ABI_path(
                "ABI", "uniswap_v2_router_ABI.json"
            )
            self.UNISWAP_V2_ROUTER_ADDRESS = self._get_env_variable(
                "UNISWAP_V2_ROUTER_ADDRESS"
            )
            self.AAVE_V3_FLASHLOAN_ABI = await self._construct_ABI_path(
                "ABI", "aave_v3_flashloan_contract_ABI.json"
            )
            self.AAVE_V3_LENDING_POOL_ABI = await self._construct_ABI_path(
                "ABI", "aave_v3_lending_pool_ABI.json"
            )
            self.AAVE_V3_FLASHLOAN_CONTRACT_ADDRESS = self._get_env_variable(
                "AAVE_V3_FLASHLOAN_CONTRACT_ADDRESS"
            )
            self.PANCAKESWAP_ROUTER_ABI = await self._construct_ABI_path(
                "ABI", "pancakeswap_router_ABI.json"
            )
            self.PANCAKESWAP_ROUTER_ADDRESS = self._get_env_variable(
                "PANCAKESWAP_ROUTER_ADDRESS"
            )
            self.BALANCER_ROUTER_ABI = await self._construct_ABI_path(
                "ABI", "balancer_router_ABI.json"
            )
            self.BALANCER_ROUTER_ADDRESS = self._get_env_variable(
                "BALANCER_ROUTER_ADDRESS")
            self.logger.info("Configuration loaded successfully. ✅")
        except EnvironmentError as e:
            self.logger.error(f"Environment variable error: {e} ❌")
            raise
        except FileNotFoundError as e:
            self.logger.error(f"File not found: {e} ❌")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error loading configuration: {e} ❌")
            raise

    def _get_env_variable(self, var_name: str, default: Optional[str] = None) -> str:
        value = os.getenv(var_name, default)
        if value is None:
            self.logger.error(f"Missing environment variable: {var_name} ❌")
            raise EnvironmentError(f"Missing environment variable: {var_name}")
        return value

    async def _load_monitored_tokens(self, file_path: str) -> List[str]:
        await loading_bar("Loading Monitored Tokens", 1)
        try:
            async with aiofiles.open(file_path, "r") as f:
                content = await f.read()
                tokens = json.loads(content)
                self.logger.debug(
                    f"Loaded {len(tokens)} monitored tokens from {file_path} ✅"
                )
                return tokens
        except Exception as e:
            self.logger.error(
                f"Failed to load monitored tokens from {file_path}: {e} ❌"
            )
            return []
        
    async def _load_erc20_function_signatures(self, file_path: str) -> Dict[str, str]:
        await loading_bar("Loading ERC20 Function Signatures", 1)
        try:
            async with aiofiles.open(file_path, "r") as f:
                content = await f.read()
                signatures = json.loads(content)
                self.logger.debug(
                    f"Loaded {len(signatures)} ERC20 function signatures from {file_path} ✅"
                )
                return signatures
        except Exception as e:
            self.logger.error(
                f"Failed to load ERC20 function signatures from {file_path}: {e} ❌"
            )
            return {}

    async def _construct_ABI_path(self, base_path: str, ABI_filename: str) -> str:
        ABI_path = os.path.join(base_path, ABI_filename)
        await loading_bar(f"Constructing '{ABI_filename}'", 1)
        if not os.path.exists(ABI_path):
            self.logger.error(f"ABI file not found at path: {ABI_path} ❌")
            raise FileNotFoundError(
                f"ABI file '{ABI_filename}' not found in path '{base_path}' ❌"
            )
        return ABI_path

    def get_ABI_path(self, ABI_name: str) -> str:
        ABI_paths = {
            "erc20": self.ERC20_ABI,
            "sushiswap": self.SUSHISWAP_ROUTER_ABI,
            "uniswap_v2": self.UNISWAP_V2_ROUTER_ABI,
            "aave_v3_flashloan": self.AAVE_V3_FLASHLOAN_ABI,
            "lending_pool": self.AAVE_V3_LENDING_POOL_ABI,
            "pancakeswap": self.PANCAKESWAP_ROUTER_ABI,
            "balancer": self.BALANCER_ROUTER_ABI,
        }
        return ABI_paths.get(ABI_name.lower(), "")

    async def get_token_addresses(self) -> List[str]:
        return self.TOKEN_ADDRESSES

    async def get_token_symbols(self) -> str:
        return self.TOKEN_SYMBOLS
