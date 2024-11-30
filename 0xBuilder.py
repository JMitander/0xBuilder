import os
import sys
import json
import time
import asyncio
import logging
import random
import aiohttp
import aiofiles
import psutil
import numpy as np
import pandas as pd
import joblib
from decimal import Decimal
from typing import Optional, List, Dict, Any, Callable, Union, Tuple
from web3.eth import AsyncEth
from web3 import AsyncWeb3
from web3.providers import AsyncHTTPProvider, AsyncIPCProvider, WebSocketProvider
from web3.middleware import ExtraDataToPOAMiddleware, SignAndSendRawMiddlewareBuilder
from web3.exceptions import ContractLogicError, TransactionNotFound
from web3.types import HexBytes
from eth_account import Account
from cachetools import TTLCache
from sklearn.linear_model import LinearRegression
from dataclasses import dataclass





async def loading_bar(message: str, total_time: int, success_message: Optional[str] = None) -> None:
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    RESET = "\033[0m"
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
            print(f"{GREEN}{success_message}{RESET}")
    except Exception as e:
        print(f"{YELLOW}Loading bar error: {e}{RESET}")
        raise

class Configuration:
    STREAMLIT_ENABLED: bool = True

    def __init__(self):
        self.INFURA_PROJECT_ID: str = ""
        self.COINGECKO_API_KEY: str = ""
        self.COINMARKETCAP_API_KEY: str = ""
        self.CRYPTOCOMPARE_API_KEY: str = ""

        self.HTTP_ENDPOINT: str = ""
        self.IPC_ENDPOINT: str = ""
        self.WEBSOCKET_ENDPOINT: str = ""
        self.WALLET_KEY: str = ""
        self.WALLET_ADDRESS: str = ""

        self.AAVE_LENDING_POOL_ADDRESS: str = ""
        self.TOKEN_ADDRESSES: Dict[str, str] = {}
        self.TOKEN_SYMBOLS: Dict[str, str] = {}

        self.ERC20_ABI: str = ""
        self.ERC20_SIGNATURES: Dict[str, Any] = {}
        self.SUSHISWAP_ROUTER_ABI: str = ""
        self.SUSHISWAP_ROUTER_ADDRESS: str = ""
        self.UNISWAP_ROUTER_ABI: str = ""
        self.UNISWAP_ROUTER_ADDRESS: str = ""
        self.AAVE_FLASHLOAN_ABI: str = ""
        self.AAVE_LENDING_POOL_ABI: str = ""
        self.AAVE_FLASHLOAN_ADDRESS: str = ""
        self.PANCAKESWAP_ROUTER_ABI: str = ""
        self.PANCAKESWAP_ROUTER_ADDRESS: str = ""
        self.BALANCER_ROUTER_ABI: str = ""
        self.BALANCER_ROUTER_ADDRESS: str = ""

        self.ML_MODEL_PATH: str = "models/price_model.joblib"
        self.ML_TRAINING_DATA_PATH: str = "data/training_data.csv"

    async def load(self) -> None:
        await loading_bar("Loading Environment Variables", 2, "Environment Variables Loaded")
        self._load_api_keys()
        self._load_providers_and_account()
        self._load_ML_models()
        await self._load_json_elements()

    def _load_ML_models(self) -> None:
        self.ML_MODEL_PATH = "models/price_model.joblib"
        self.ML_TRAINING_DATA_PATH = "data/training_data.csv"

    def _load_api_keys(self) -> None:
        self.ETHERSCAN_API_KEY = self._get_env_variable("ETHERSCAN_API_KEY")
        self.INFURA_PROJECT_ID = self._get_env_variable("INFURA_PROJECT_ID")
        self.COINGECKO_API_KEY = self._get_env_variable("COINGECKO_API_KEY")
        self.COINMARKETCAP_API_KEY = self._get_env_variable("COINMARKETCAP_API_KEY")
        self.CRYPTOCOMPARE_API_KEY = self._get_env_variable("CRYPTOCOMPARE_API_KEY")

    def _load_providers_and_account(self) -> None:
        self.HTTP_ENDPOINT = self._get_env_variable("HTTP_ENDPOINT")
        self.IPC_ENDPOINT = self._get_env_variable("IPC_ENDPOINT", default="")
        self.WEBSOCKET_ENDPOINT = self._get_env_variable("WEBSOCKET_ENDPOINT", default="")
        self.WALLET_KEY = self._get_env_variable("WALLET_KEY")
        self.WALLET_ADDRESS = self._get_env_variable("WALLET_ADDRESS")

    async def _load_json_elements(self) -> None:
        self.AAVE_LENDING_POOL_ADDRESS = self._get_env_variable("AAVE_LENDING_POOL_ADDRESS")
        self.TOKEN_ADDRESSES = await self._load_json_file(self._get_env_variable("TOKEN_ADDRESSES"), "monitored tokens")
        self.TOKEN_SYMBOLS = await self._load_json_file(self._get_env_variable("TOKEN_SYMBOLS"), "token symbols")
        self.ERC20_ABI = await self._construct_abi_path("abi", "erc20_abi.json")
        self.ERC20_SIGNATURES = await self._load_json_file(self._get_env_variable("ERC20_SIGNATURES"), "ERC20 function signatures")
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
                await loading_bar(f"Loading {len(data)} {description} from {file_path}", 3, f"{description.capitalize()} Loaded")
                return data
        except FileNotFoundError:
            print(f"{description.capitalize()} file not found at {file_path}")
            raise
        except json.JSONDecodeError:
            print(f"Failed to decode JSON for {description} from {file_path}")
            raise
        except Exception as e:
            print(f"Error loading {description} from {file_path}: {e}")
            raise

    async def _construct_abi_path(self, base_path: str, abi_filename: str) -> str:
        abi_path = os.path.join(base_path, abi_filename)
        if not os.path.exists(abi_path):
            raise FileNotFoundError(f"ABI file '{abi_filename}' not found in path '{base_path}'")
        return abi_path

    async def get_token_addresses(self) -> List[str]:
        return list(self.TOKEN_ADDRESSES.values())

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

class APIConfig:
    def __init__(self, configuration: Optional[Configuration] = None):
        self.apiconfig: Dict[str, Dict[str, Any]] = {}
        self.configuration = configuration
        self.session: Optional[aiohttp.ClientSession] = None
        self.price_cache: TTLCache = TTLCache(maxsize=1000, ttl=300)
        self.token_symbol_cache: TTLCache = TTLCache(maxsize=1000, ttl=86400)
        self.api_lock: asyncio.Lock = asyncio.Lock()
        self.rate_limiters: Dict[str, asyncio.Semaphore] = {}

    async def __aenter__(self):
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
            "primary": {
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
        if self.session:
            await self.session.close()

    async def get_token_symbol(self, web3: AsyncWeb3, token_address: str) -> Optional[str]:
        if token_address in self.token_symbol_cache:
            return self.token_symbol_cache[token_address]
        if token_address in self.configuration.TOKEN_SYMBOLS:
            symbol = self.configuration.TOKEN_SYMBOLS[token_address]
            self.token_symbol_cache[token_address] = symbol
            return symbol
        try:
            erc20_abi = await self._load_abi(self.configuration.ERC20_ABI)
            contract = self.web3.eth.contract(address=token_address, abi=erc20_abi)
            symbol = await contract.functions.symbol().call()
            self.token_symbol_cache[token_address] = symbol
            return symbol
        except Exception:
            return None

    async def get_real_time_price(self, token: str, vs_currency: str = "eth", service: Optional[str] = None) -> Optional[Decimal]:
        cache_key = f"price_{token}_{vs_currency}"
        if cache_key in self.price_cache:
            return self.price_cache[cache_key]
        
        services = [service] if service else list(self.apiconfig.keys())
        prices = []
        weights = []
        async with self.api_lock:
            for source in services:
                config = self.apiconfig.get(source)
                if not config:
                    continue
                try:
                    price = await self._fetch_price(source, token, vs_currency)
                    if price:
                        prices.append(price)
                        weights.append(config["weight"] * config["success_rate"])
                except Exception:
                    continue

        if not prices:
            return None
        weighted_price = sum(p * w for p, w in zip(prices, weights)) / sum(weights)
        self.price_cache[cache_key] = Decimal(str(weighted_price))
        return self.price_cache[cache_key]

    async def _fetch_price(self, source: str, token: str, vs_currency: str) -> Optional[Decimal]:
        config = self.apiconfig.get(source)
        if not config:
            return None

        if source == "coingecko":
            url = f"{config['base_url']}/simple/price"
            params = {"ids": token, "vs_currencies": vs_currency}
            response = await self.make_request(source, url, params=params)
            return Decimal(str(response.get(token, {}).get(vs_currency, 0)))
        elif source == "coinmarketcap":
            url = f"{config['base_url']}/cryptocurrency/quotes/latest"
            params = {"symbol": token.upper(), "convert": vs_currency.upper()}
            headers = {"X-CMC_PRO_API_KEY": config["api_key"]}
            response = await self.make_request(source, url, params=params, headers=headers)
            data = response.get("data", {}).get(token.upper(), {}).get("quote", {}).get(vs_currency.upper(), {}).get("price", 0)
            return Decimal(str(data))
        elif source == "cryptocompare":
            url = f"{config['base_url']}/price"
            params = {"fsym": token.upper(), "tsyms": vs_currency.upper(), "api_key": config["api_key"]}
            response = await self.make_request(source, url, params=params)
            return Decimal(str(response.get(vs_currency.upper(), 0)))
        elif source == "binance":
            url = f"{config['base_url']}/ticker/price"
            symbol = f"{token.upper()}{vs_currency.upper()}"
            params = {"symbol": symbol}
            response = await self.make_request(source, url, params=params)
            return Decimal(str(response.get("price", 0)))
        elif source == "primary":
            url = f"{config['base_url']}/simple/price"
            params = {"ids": token, "vs_currencies": vs_currency}
            response = await self.make_request(source, url, params=params)
            return Decimal(str(response.get(token, {}).get(vs_currency, 0)))
        else:
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
        rate_limiter = self.rate_limiters.get(provider_name, asyncio.Semaphore(10))
        async with rate_limiter:
            for attempt in range(max_attempts):
                try:
                    timeout = aiohttp.ClientTimeout(total=10 * (attempt + 1))
                    async with self.session.get(url, params=params, headers=headers, timeout=timeout) as response:
                        if response.status == 429:
                            wait_time = backoff_factor ** attempt
                            await asyncio.sleep(wait_time)
                            continue
                        response.raise_for_status()
                        return await response.json()
                except aiohttp.ClientResponseError:
                    if attempt == max_attempts - 1:
                        raise
                    wait_time = backoff_factor ** attempt
                    await asyncio.sleep(wait_time)
                except aiohttp.ClientConnectionError:
                    if attempt == max_attempts - 1:
                        raise
                    wait_time = backoff_factor ** attempt
                    await asyncio.sleep(wait_time)
                except asyncio.TimeoutError:
                    if attempt == max_attempts - 1:
                        raise
                    wait_time = backoff_factor ** attempt
                    await asyncio.sleep(wait_time)
                except Exception:
                    raise

    async def fetch_historical_prices(self, token: str, days: int = 30, service: Optional[str] = None) -> List[float]:
        cache_key = f"historical_prices_{token}_{days}"
        if cache_key in self.price_cache:
            return self.price_cache[cache_key]
        
        services = [service] if service else list(self.apiconfig.keys())
        for source in services:
            config = self.apiconfig.get(source)
            if not config:
                continue
            try:
                if source == "coingecko":
                    url = f"{config['base_url']}/coins/{token}/market_chart"
                    params = {"vs_currency": "usd", "days": days}
                    response = await self.make_request(source, url, params=params)
                    prices = [price[1] for price in response.get("prices", [])]
                    if prices:
                        self.price_cache[cache_key] = prices
                        return prices
            except Exception:
                continue
        return []

    async def get_token_volume(self, token: str, service: Optional[str] = None) -> float:
        cache_key = f"token_volume_{token}"
        if cache_key in self.price_cache:
            return self.price_cache[cache_key]
        
        services = [service] if service else list(self.apiconfig.keys())
        for source in services:
            config = self.apiconfig.get(source)
            if not config:
                continue
            try:
                if source == "coingecko":
                    url = f"{config['base_url']}/coins/markets"
                    params = {"vs_currency": "usd", "ids": token}
                    response = await self.make_request(source, url, params=params)
                    if response and isinstance(response, list):
                        volume = response[0].get("total_volume", 0.0)
                        self.price_cache[cache_key] = float(volume)
                        return float(volume)
            except Exception:
                continue
        return 0.0

    async def _fetch_from_services(
        self,
        fetch_func: Callable[[str], Any],
        description: str
    ) -> Optional[Any]:
        for service in self.apiconfig.keys():
            try:
                result = await fetch_func(service)
                if result:
                    return result
            except Exception:
                continue
        return None

    async def _load_abi(self, abi_path: str) -> List[Dict[str, Any]]:
        try:
            async with aiofiles.open(abi_path, 'r') as file:
                content = await file.read()
                abi = json.loads(content)
            return abi
        except Exception:
            raise

    async def _load_abi_sync(self, abi_path: str) -> List[Dict[str, Any]]:
        try:
            with open(abi_path, 'r') as file:
                abi = json.load(file)
            return abi
        except Exception:
            raise

class NonceCore:
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 1.0
    CACHE_TTL: int = 300

    def __init__(
        self,
        web3: AsyncWeb3,
        address: str,
        configuration: Configuration,
    ):
        self.pending_transactions: set = set()
        self.web3 = web3
        self.configuration = configuration
        self.address = address
        self.lock = asyncio.Lock()
        self.nonce_cache: TTLCache = TTLCache(maxsize=1, ttl=self.CACHE_TTL)
        self.last_sync: float = time.monotonic()
        self._initialized: bool = False

    async def initialize(self) -> None:
        async with self.lock:
            if not self._initialized:
                await self._init_nonce()
                self._initialized = True

    async def _init_nonce(self) -> None:
        current_nonce = await self._fetch_current_nonce_with_retries()
        pending_nonce = await self._get_pending_nonce()
        self.nonce_cache[self.address] = max(current_nonce, pending_nonce)
        self.last_sync = time.monotonic()

    async def get_nonce(self, force_refresh: bool = False) -> int:
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
            except KeyError:
                await self._handle_nonce_error()
                raise
            except Exception:
                await self._handle_nonce_error()
                raise

    async def refresh_nonce(self) -> None:
        async with self.lock:
            try:
                chain_nonce = await self._fetch_current_nonce_with_retries()
                cached_nonce = self.nonce_cache.get(self.address, 0)
                pending_nonce = await self._get_pending_nonce()
                new_nonce = max(chain_nonce, cached_nonce, pending_nonce)
                self.nonce_cache[self.address] = new_nonce
                self.last_sync = time.monotonic()
            except Exception:
                raise

    async def _fetch_current_nonce_with_retries(self) -> int:
        backoff = self.RETRY_DELAY
        for attempt in range(self.MAX_RETRIES):
            try:
                return await self.web3.eth.get_transaction_count(
                    self.address, block_identifier="pending"
                )
            except Exception:
                if attempt == self.MAX_RETRIES - 1:
                    raise
                await asyncio.sleep(backoff)
                backoff *= 2

    async def _get_pending_nonce(self) -> int:
        try:
            pending_nonces = [int(nonce) for nonce in self.pending_transactions]
            return max(pending_nonces) + 1 if pending_nonces else 0
        except Exception:
            return 0

    async def track_transaction(self, tx_hash: str, nonce: int) -> None:
        self.pending_transactions.add(nonce)
        try:
            await self.web3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            self.pending_transactions.discard(nonce)
        except Exception:
            self.pending_transactions.discard(nonce)

    async def _handle_nonce_error(self) -> None:
        try:
            await self.sync_nonce_with_chain()
        except Exception:
            raise

    async def sync_nonce_with_chain(self) -> None:
        async with self.lock:
            try:
                new_nonce = await self._fetch_current_nonce_with_retries()
                self.nonce_cache[self.address] = new_nonce
                self.last_sync = time.monotonic()
                self.pending_transactions.clear()
            except Exception:
                raise

    def _should_refresh_cache(self) -> bool:
        return time.monotonic() - self.last_sync > (self.CACHE_TTL / 2)

    async def reset(self) -> None:
        async with self.lock:
            try:
                self.nonce_cache.clear()
                self.pending_transactions.clear()
                self.last_sync = time.monotonic()
                self._initialized = False
                await self.initialize()
            except Exception:
                raise

    async def stop(self) -> None:
        try:
            await self.reset()
        except Exception:
            raise

class SafetyNet:
    CACHE_TTL: int = 300
    GAS_PRICE_CACHE_TTL: int = 15
    SLIPPAGE_CONFIG: Dict[str, float] = {
        "default": 0.1,
        "min": 0.01,
        "max": 0.5,
        "high_congestion": 0.05,
        "low_congestion": 0.2,
    }
    GAS_CONFIG: Dict[str, Union[int, float]] = {
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
        self.price_cache: TTLCache = TTLCache(maxsize=1000, ttl=self.CACHE_TTL)
        self.gas_price_cache: TTLCache = TTLCache(maxsize=1, ttl=self.GAS_PRICE_CACHE_TTL)
        self.price_lock: asyncio.Lock = asyncio.Lock()

    async def get_balance(self) -> Decimal:
        cache_key = f"balance_{self.address}"
        if cache_key in self.price_cache:
            return self.price_cache[cache_key]
        for attempt in range(3):
            try:
                balance_wei = await self.web3.eth.get_balance(self.address)
                balance_eth = Decimal(self.web3.from_wei(balance_wei, "ether"))
                self.price_cache[cache_key] = balance_eth
                return balance_eth
            except Exception:
                if attempt < 2:
                    await asyncio.sleep(1 * (attempt + 1))
        return Decimal(0)

    async def ensure_profit(
        self,
        transaction_data: Dict[str, Any],
        minimum_profit_eth: Optional[float] = None,
    ) -> bool:
        try:
            if minimum_profit_eth is None:
                account_balance = await self.get_balance()
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
            return profit > Decimal(minimum_profit_eth)
        except Exception:
            return False

    def _validate_gas_parameters(self, gas_price_gwei: Decimal, gas_used: int) -> bool:
        if gas_used == 0:
            return False
        if gas_price_gwei > Decimal(self.GAS_CONFIG["max_gas_price_gwei"]):
            return False
        return True

    def _calculate_gas_cost(self, gas_price_gwei: Decimal, gas_used: int) -> Decimal:
        return gas_price_gwei * Decimal(gas_used) * Decimal("1e-9")

    async def _calculate_profit(
        self,
        transaction_data: Dict[str, Any],
        real_time_price: Decimal,
        slippage: float,
        gas_cost_eth: Decimal,
    ) -> Decimal:
        expected_output = real_time_price * Decimal(transaction_data["amountOut"])
        input_amount = Decimal(transaction_data["amountIn"])
        slippage_adjusted_output = expected_output * (1 - Decimal(slippage))
        return slippage_adjusted_output - input_amount - gas_cost_eth

    async def get_dynamic_gas_price(self) -> Decimal:
        if "gas_price" in self.gas_price_cache:
            return self.gas_price_cache["gas_price"]
        try:
            gas_price = await self.web3.eth.gas_price
            gas_price_gwei = Decimal(self.web3.from_wei(gas_price, "gwei"))
            self.gas_price_cache["gas_price"] = gas_price_gwei
            return gas_price_gwei
        except Exception:
            return Decimal(0)

    async def estimate_gas(self, transaction: Dict[str, Any]) -> int:
        try:
            gas_estimate = await self.web3.eth.estimate_gas(transaction)
            return gas_estimate
        except Exception:
            return 0

    async def adjust_slippage_tolerance(self) -> float:
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
        except Exception:
            return self.SLIPPAGE_CONFIG["default"]

    async def get_network_congestion(self) -> float:
        try:
            latest_block = await self.web3.eth.get_block("latest")
            gas_used = latest_block["gasUsed"]
            gas_limit = latest_block["gasLimit"]
            congestion_level = gas_used / gas_limit
            return congestion_level
        except Exception:
            return 0.5

    async def stop(self) -> None:
        pass

class MempoolMonitor:
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 1.0
    BATCH_SIZE: int = 10
    MAX_PARALLEL_TASKS: int = 50

    def __init__(
        self,
        web3: AsyncWeb3,
        safetynet: SafetyNet,
        noncecore: NonceCore,
        apiconfig: APIConfig,
        monitored_tokens: Optional[List[str]] = None,
        erc20_abi: Optional[List[Dict[str, Any]]] = None,
        configuration: Optional[Configuration] = None,
    ):
        self.web3 = web3
        self.configuration = configuration
        self.safetynet = safetynet
        self.noncecore = noncecore
        self.apiconfig = apiconfig
        self.running: bool = False
        self.pending_transactions: asyncio.Queue = asyncio.Queue()
        self.monitored_tokens: set = set(monitored_tokens or [])
        self.profitable_transactions: asyncio.Queue = asyncio.Queue()
        self.processed_transactions: set = set()
        self.erc20_abi: List[Dict[str, Any]] = erc20_abi or []
        self.minimum_profit_threshold: Decimal = Decimal("0.001")
        self.max_parallel_tasks: int = self.MAX_PARALLEL_TASKS
        self.retry_attempts: int = self.MAX_RETRIES
        self.backoff_factor: float = 1.5
        self.semaphore: asyncio.Semaphore = asyncio.Semaphore(self.max_parallel_tasks)
        self.task_queue: asyncio.Queue = asyncio.Queue()

    async def start_monitoring(self) -> None:
        if self.running:
            return
        try:
            self.running = True
            monitoring_task = asyncio.create_task(self._run_monitoring())
            processor_task = asyncio.create_task(self._process_task_queue())
            await asyncio.gather(monitoring_task, processor_task)
        except Exception:
            self.running = False
            raise

    async def stop_monitoring(self) -> None:
        if not self.running:
            return
        self.running = False
        try:
            while not self.task_queue.empty():
                await asyncio.sleep(0.1)
        except Exception:
            raise

    async def _run_monitoring(self) -> None:
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
            except Exception:
                retry_count += 1
                wait_time = min(self.backoff_factor ** retry_count, 30)
                await asyncio.sleep(wait_time)

    async def _setup_pending_filter(self) -> Optional[Any]:
        try:
            pending_filter = await self.web3.eth.filter("pending")
            return pending_filter
        except Exception:
            return None

    async def _handle_new_transactions(self, tx_hashes: List[str]) -> None:
        async def process_batch(batch):
            await asyncio.gather(
                *(self._queue_transaction(tx_hash) for tx_hash in batch)
            )
        try:
            for i in range(0, len(tx_hashes), self.BATCH_SIZE):
                batch = tx_hashes[i: i + self.BATCH_SIZE]
                await process_batch(batch)
        except Exception:
            pass

    async def _queue_transaction(self, tx_hash: str) -> None:
        tx_hash_hex = tx_hash.hex() if isinstance(tx_hash, bytes) else tx_hash
        if tx_hash_hex not in self.processed_transactions:
            self.processed_transactions.add(tx_hash_hex)
            await self.task_queue.put(tx_hash_hex)

    async def _process_task_queue(self) -> None:
        while self.running:
            try:
                tx_hash = await self.task_queue.get()
                async with self.semaphore:
                    await self.process_transaction(tx_hash)
                self.task_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception:
                pass

    async def process_transaction(self, tx_hash: str) -> None:
        try:
            tx = await self._get_transaction_with_retry(tx_hash)
            if not tx:
                return
            analysis = await self.analyze_transaction(tx)
            if analysis.get("is_profitable"):
                await self._handle_profitable_transaction(analysis)
        except Exception:
            pass

    async def _get_transaction_with_retry(self, tx_hash: str) -> Optional[Any]:
        for attempt in range(self.retry_attempts):
            try:
                return await self.web3.eth.get_transaction(tx_hash)
            except TransactionNotFound:
                if attempt == self.retry_attempts - 1:
                    return None
                await asyncio.sleep(self.backoff_factor ** attempt)
            except Exception:
                return None
        return None

    async def _handle_profitable_transaction(self, analysis: Dict[str, Any]) -> None:
        try:
            await self.profitable_transactions.put(analysis)
        except Exception:
            pass

    async def analyze_transaction(self, tx: Any) -> Dict[str, Any]:
        if not tx.hash or not tx.input:
            return {"is_profitable": False}
        try:
            if tx.value > 0:
                return await self._analyze_eth_transaction(tx)
            return await self._analyze_token_transaction(tx)
        except Exception:
            return {"is_profitable": False}

    async def _analyze_eth_transaction(self, tx: Any) -> Dict[str, Any]:
        try:
            is_profitable = await self.safetynet.ensure_profit({
                "to": tx.to,
                "value": tx.value,
                "gasPrice": tx.gasPrice,
                "output_token": "",
                "amountOut": 0,
                "amountIn": tx.value,
            })
            if is_profitable:
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
        except Exception:
            return {"is_profitable": False}

    async def _analyze_token_transaction(self, tx: Any) -> Dict[str, Any]:
        if not self.erc20_abi:
            return {"is_profitable": False}
        try:
            contract = self.web3.eth.contract(address=tx.to, abi=self.erc20_abi)
            function_abi, function_params = contract.decode_function_input(tx.input)
            function_name = function_abi.name
            if function_name in self.configuration.ERC20_SIGNATURES:
                estimated_profit = await self.safetynet.ensure_profit({
                    "to": tx.to,
                    "value": tx.value,
                    "gasPrice": tx.gasPrice,
                    "output_token": "",
                    "amountOut": function_params.get("amountOutMin", 0),
                    "amountIn": function_params.get("amountIn", 0),
                })
                if estimated_profit:
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
        except Exception:
            return {"is_profitable": False}

class TransactionCore:
    """
    Core class for building, signing, and executing transactions.
    """

    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 1.0

    def __init__(
        self,
        web3: AsyncWeb3,
        account: Account,
        aave_flashloan_address: str,
        aave_flashloan_abi: List[Dict[str, Any]],
        aave_lending_pool_address: str,
        aave_lending_pool_abi: List[Dict[str, Any]],
        apiconfig: APIConfig,
        noncecore: NonceCore,
        safetynet: SafetyNet,
        configuration: Configuration,
        gas_price_multiplier: float = 1.1,
        erc20_abi: Optional[List[Dict[str, Any]]] = None,
    ):
        self.web3 = web3
        self.account = account
        self.configuration = configuration
        self.apiconfig = apiconfig
        self.noncecore = noncecore
        self.safetynet = safetynet
        self.gas_price_multiplier = gas_price_multiplier
        self.retry_attempts = self.MAX_RETRIES
        self.erc20_abi: List[Dict[str, Any]] = erc20_abi or []
        self.current_profit: Decimal = Decimal("0")
        self.aave_flashloan_address: str = aave_flashloan_address
        self.aave_flashloan_abi: List[Dict[str, Any]] = aave_flashloan_abi
        self.aave_lending_pool_address: str = aave_lending_pool_address
        self.aave_lending_pool_abi: List[Dict[str, Any]] = aave_lending_pool_abi

        # Contract instances will be initialized later
        self.flashloan_contract: Optional[Any] = None
        self.lending_pool_contract: Optional[Any] = None
        self.uniswap_router_contract: Optional[Any] = None
        self.sushiswap_router_contract: Optional[Any] = None
        self.pancakeswap_router_contract: Optional[Any] = None
        self.balancer_router_contract: Optional[Any] = None

        self._abi_cache: Dict[str, Any] = {}

    async def initialize(self) -> None:
        """
        Initializes contract instances.
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
                await self.apiconfig._load_abi(self.configuration.UNISWAP_ROUTER_ABI),
                "Uniswap Router Contract",
            )
            self.sushiswap_router_contract = await self._initialize_contract(
                self.configuration.SUSHISWAP_ROUTER_ADDRESS,
                await self.apiconfig._load_abi(self.configuration.SUSHISWAP_ROUTER_ABI),
                "Sushiswap Router Contract",
            )
            self.pancakeswap_router_contract = await self._initialize_contract(
                self.configuration.PANCAKESWAP_ROUTER_ADDRESS,
                await self.apiconfig._load_abi(self.configuration.PANCAKESWAP_ROUTER_ABI),
                "Pancakeswap Router Contract",
            )
            self.balancer_router_contract = await self._initialize_contract(
                self.configuration.BALANCER_ROUTER_ADDRESS,
                await self.apiconfig._load_abi(self.configuration.BALANCER_ROUTER_ABI),
                "Balancer Router Contract",
            )
            if not self.erc20_abi:
                self.erc20_abi = await self._load_erc20_abi()
            print("TransactionCore initialized successfully.")
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
        Initializes a smart contract instance.

        Args:
            contract_address (str): The address of the contract.
            contract_abi (Union[str, List[Dict[str, Any]]]): The ABI of the contract or path to ABI file.
            contract_name (str): The name of the contract for logging purposes.

        Returns:
            Any: The contract instance.

        Raises:
            FileNotFoundError: If ABI file is not found.
            json.JSONDecodeError: If ABI file is invalid.
            Exception: For any other exceptions.
        """
        try:
            if isinstance(contract_abi, str):
                # Assume it's a path to the ABI file
                if contract_abi in self._abi_cache:
                    contract_abi_content = self._abi_cache[contract_abi]
                else:
                    contract_abi_content = await self.apiconfig._load_abi(contract_abi)
                    self._abi_cache[contract_abi] = contract_abi_content
                contract_abi = contract_abi_content
            contract_instance = self.web3.eth.contract(
                address=self.web3.to_checksum_address(contract_address),
                abi=contract_abi,
            )
            print(f"{contract_name} initialized at {contract_address}")
            return contract_instance
        except FileNotFoundError:
            print(f"ABI file not found for {contract_name} at {contract_abi}")
            raise
        except json.JSONDecodeError:
            print(f"Invalid ABI JSON for {contract_name} at {contract_abi}")
            raise
        except Exception as e:
            print(f"Error initializing {contract_name}: {e}")
            raise

    async def _load_erc20_abi(self) -> List[Dict[str, Any]]:
        """
        Loads the ERC20 ABI.

        Returns:
            List[Dict[str, Any]]: The ERC20 ABI.
        """
        try:
            erc20_abi = await self.apiconfig._load_abi(self.configuration.ERC20_ABI)
            return erc20_abi
        except Exception as e:
            print(f"Error loading ERC20 ABI: {e}")
            raise

    async def build_transaction(
        self, function_call: Any, additional_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Builds a transaction dictionary with necessary parameters.

        Args:
            function_call (Any): The contract function call.
            additional_params (Optional[Dict[str, Any]], optional): Additional transaction parameters.
                Defaults to None.

        Returns:
            Dict[str, Any]: The transaction dictionary.

        Raises:
            Exception: If building the transaction fails.
        """
        additional_params = additional_params or {}
        try:
            tx_details = await function_call.build_transaction({
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
        Retrieves and adjusts the gas price based on the multiplier.

        Returns:
            Dict[str, int]: The gas price parameter.
        """
        try:
            gas_price_gwei = await self.safetynet.get_dynamic_gas_price()
            gas_price = int(
                self.web3.to_wei(gas_price_gwei * Decimal(str(self.gas_price_multiplier)), "gwei")
            )
            return {"gasPrice": gas_price}
        except Exception as e:
            print(f"Error getting dynamic gas price: {e}")
            # Default to 100 Gwei if unable to fetch
            return {"gasPrice": int(self.web3.to_wei(100 * self.gas_price_multiplier, "gwei"))}

    async def estimate_gas_smart(self, tx: Dict[str, Any]) -> int:
        """
        Estimates the gas required for a transaction, handling exceptions gracefully.

        Args:
            tx (Dict[str, Any]): The transaction details.

        Returns:
            int: The estimated gas limit.
        """
        try:
            gas_estimate = await self.web3.eth.estimate_gas(tx)
            return gas_estimate
        except ContractLogicError as e:
            print(f"ContractLogicError during gas estimation: {e}")
            return 100_000  # Fallback gas limit
        except TransactionNotFound as e:
            print(f"TransactionNotFound during gas estimation: {e}")
            return 100_000  # Fallback gas limit
        except Exception as e:
            print(f"Error estimating gas: {e}")
            return 100_000  # Fallback gas limit

    async def execute_transaction(self, tx: Dict[str, Any]) -> Optional[str]:
        """
        Executes a signed transaction with retry logic.

        Args:
            tx (Dict[str, Any]): The transaction details.

        Returns:
            Optional[str]: The transaction hash if successful, else None.
        """
        for attempt in range(1, self.retry_attempts + 1):
            try:
                signed_tx = self.sign_transaction(tx)
                tx_hash = await self.web3.eth.send_raw_transaction(signed_tx)
                tx_hash_hex = tx_hash.hex() if isinstance(tx_hash, HexBytes) else tx_hash
                await self.noncecore.refresh_nonce()
                print(f"Transaction executed with hash: {tx_hash_hex}")
                return tx_hash_hex
            except TransactionNotFound as e:
                print(f"TransactionNotFound: {e}")
            except ContractLogicError as e:
                print(f"ContractLogicError: {e}")
            except Exception as e:
                print(f"Error executing transaction: {e}")
                if attempt < self.MAX_RETRIES:
                    sleep_time = self.RETRY_DELAY * attempt
                    await asyncio.sleep(sleep_time)
                else:
                    print(f"Failed to execute transaction after {self.MAX_RETRIES} attempts.")
        return None

    def sign_transaction(self, transaction: Dict[str, Any]) -> bytes:
        """
        Signs a transaction with the account's private key.

        Args:
            transaction (Dict[str, Any]): The transaction details.

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
        Handles an ETH transaction by potentially front-running it.

        Args:
            target_tx (Dict[str, Any]): The target transaction details.

        Returns:
            bool: True if the transaction was handled successfully, False otherwise.
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
                print(f"ETH Transaction handled successfully: {tx_hash_executed}")
                return True
            else:
                print(f"Failed to handle ETH transaction: {tx_hash}")
                return False
        except KeyError as e:
            print(f"KeyError in handle_eth_transaction: {e}")
            return False
        except Exception as e:
            print(f"Error handling ETH transaction {tx_hash}: {e}")
            return False

    def calculate_flashloan_amount(self, target_tx: Dict[str, Any]) -> int:
        """
        Calculates the amount for a flashloan based on estimated profit.

        Args:
            target_tx (Dict[str, Any]): The target transaction details.

        Returns:
            int: The flashloan amount in Wei.
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
        Simulates a transaction to check if it would succeed.

        Args:
            transaction (Dict[str, Any]): The transaction details.

        Returns:
            bool: True if the simulation succeeds, False otherwise.
        """
        try:
            await self.web3.eth.call(transaction, block_identifier="pending")
            return True
        except ContractLogicError as e:
            print(f"ContractLogicError during simulation: {e}")
            return False
        except Exception as e:
            print(f"Error simulating transaction: {e}")
            return False

    async def prepare_flashloan_transaction(
        self, flashloan_asset: str, flashloan_amount: int
    ) -> Optional[Dict[str, Any]]:
        """
        Prepares a flashloan transaction.

        Args:
            flashloan_asset (str): The asset address for the flashloan.
            flashloan_amount (int): The amount for the flashloan in Wei.

        Returns:
            Optional[Dict[str, Any]]: The prepared transaction, or None if failed.
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
            print(f"ContractLogicError in prepare_flashloan_transaction: {e}")
            return None
        except Exception as e:
            print(f"Error preparing flashloan transaction: {e}")
            return None

    async def send_bundle(self, transactions: List[Dict[str, Any]]) -> bool:
        """
        Sends a bundle of transactions via Flashbots or similar services.

        Args:
            transactions (List[Dict[str, Any]]): List of signed transaction dictionaries.

        Returns:
            bool: True if the bundle was successfully sent, False otherwise.
        """
        try:
            signed_txs = [self.sign_transaction(tx) for tx in transactions]
            bundle_payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "eth_sendBundle",
                "params": [
                    {
                        "txs": [signed_tx.hex() for signed_tx in signed_txs],
                        "blockNumber": hex(await self.web3.eth.block_number() + 1),
                    }
                ],
            }
            mev_builders = [
                {
                    "name": "Flashbots",
                    "url": "https://relay.flashbots.net",
                    "auth_header": "X-Flashbots-Signature"
                },
                # Add more builders if needed
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
                                break  # Exit retry loop on success
                    except aiohttp.ClientResponseError as e:
                        print(f"ClientResponseError with {builder['name']}: {e}")
                        if attempt < self.retry_attempts:
                            sleep_time = self.RETRY_DELAY * attempt
                            await asyncio.sleep(sleep_time)
                    except ValueError as e:
                        print(f"ValueError with {builder['name']}: {e}")
                        break  # Do not retry on value errors
                    except Exception as e:
                        print(f"Unexpected error with {builder['name']}: {e}")
                        if attempt < self.retry_attempts:
                            sleep_time = self.RETRY_DELAY * attempt
                            await asyncio.sleep(sleep_time)
            if successes:
                await self.noncecore.refresh_nonce()
                print(f"Bundle sent successfully via: {', '.join(successes)}")
                return True
            else:
                print("Failed to send bundle via all builders.")
                return False
        except Exception as e:
            print(f"Error sending bundle: {e}")
            return False

    async def front_run(self, target_tx: Dict[str, Any]) -> bool:
        """
        Executes a front-run transaction based on the target transaction.

        Args:
            target_tx (Dict[str, Any]): The target transaction details.

        Returns:
            bool: True if the front-run was successful, False otherwise.
        """
        try:
            if not isinstance(target_tx, dict):
                return False
            tx_hash = target_tx.get("tx_hash", "Unknown")
            required_fields = ["input", "to", "value", "gasPrice"]
            if not all(field in target_tx for field in required_fields):
                print(f"Transaction {tx_hash} is missing required fields.")
                return False
            decoded_tx = await self.decode_transaction_input(
                target_tx.get("input", "0x"),
                self.web3.to_checksum_address(target_tx.get("to", ""))
            )
            if not decoded_tx or "params" not in decoded_tx:
                return False
            path = decoded_tx["params"].get("path", [])
            if len(path) < 2:
                return False
            flashloan_asset = self.web3.to_checksum_address(path[0])
            flashloan_amount = self.calculate_flashloan_amount(target_tx)
            if flashloan_amount <= 0:
                return False
            flashloan_tx = await self.prepare_flashloan_transaction(
                flashloan_asset, flashloan_amount
            )
            if not flashloan_tx:
                return False
            front_run_tx_details = await self._prepare_front_run_transaction(target_tx)
            if not front_run_tx_details:
                return False
            simulation_success = await asyncio.gather(
                self.simulate_transaction(flashloan_tx),
                self.simulate_transaction(front_run_tx_details)
            )
            if not all(simulation_success):
                return False
            if await self.send_bundle([flashloan_tx, front_run_tx_details]):
                print(f"Front-run transaction executed successfully for {tx_hash}")
                return True
            else:
                print(f"Failed to execute front-run transaction for {tx_hash}")
                return False
        except Exception as e:
            print(f"Error executing front_run for {tx_hash}: {e}")
            return False

    async def back_run(self, target_tx: Dict[str, Any]) -> bool:
        """
        Executes a back-run transaction based on the target transaction.

        Args:
            target_tx (Dict[str, Any]): The target transaction details.

        Returns:
            bool: True if the back-run was successful, False otherwise.
        """
        try:
            if not isinstance(target_tx, dict):
                return False
            tx_hash = target_tx.get("tx_hash", "Unknown")
            required_fields = ["input", "to", "value", "gasPrice"]
            if not all(field in target_tx for field in required_fields):
                print(f"Transaction {tx_hash} is missing required fields.")
                return False
            decoded_tx = await self.decode_transaction_input(
                target_tx.get("input", "0x"),
                self.web3.to_checksum_address(target_tx.get("to", ""))
            )
            if not decoded_tx or "params" not in decoded_tx:
                return False
            path = decoded_tx["params"].get("path", [])
            if len(path) < 2:
                return False
            reversed_path = path[::-1]
            decoded_tx["params"]["path"] = reversed_path
            to_address = self.web3.to_checksum_address(target_tx.get("to", ""))
            routers = {
                self.configuration.UNISWAP_ROUTER_ADDRESS: (self.uniswap_router_contract, "Uniswap"),
                self.configuration.SUSHISWAP_ROUTER_ADDRESS: (self.sushiswap_router_contract, "Sushiswap"),
                self.configuration.PANCAKESWAP_ROUTER_ADDRESS: (self.pancakeswap_router_contract, "Pancakeswap"),
                self.configuration.BALANCER_ROUTER_ADDRESS: (self.balancer_router_contract, "Balancer")
            }
            if to_address not in routers:
                print(f"Router address {to_address} not recognized.")
                return False
            router_contract, exchange_name = routers[to_address]
            if not router_contract:
                print(f"Router contract for {exchange_name} is not initialized.")
                return False
            try:
                back_run_function = getattr(router_contract.functions, decoded_tx["function_name"])(**decoded_tx["params"])
            except AttributeError as e:
                print(f"AttributeError: {e}")
                return False
            back_run_tx = await self.build_transaction(back_run_function)
            if not back_run_tx:
                return False
            simulation_success = await self.simulate_transaction(back_run_tx)
            if not simulation_success:
                print(f"Back-run transaction simulation failed for {tx_hash}")
                return False
            if await self.send_bundle([back_run_tx]):
                print(f"Back-run transaction executed successfully for {tx_hash}")
                return True
            else:
                print(f"Failed to execute back-run transaction for {tx_hash}")
                return False
        except Exception as e:
            print(f"Error executing back_run for {tx_hash}: {e}")
            return False

    async def execute_sandwich_attack(self, target_tx: Dict[str, Any]) -> bool:
        """
        Executes a sandwich attack based on the target transaction.

        Args:
            target_tx (Dict[str, Any]): The target transaction details.

        Returns:
            bool: True if the sandwich attack was successful, False otherwise.
        """
        try:
            if not isinstance(target_tx, dict):
                return False
            tx_hash = target_tx.get("tx_hash", "Unknown")
            required_fields = ["input", "to", "value", "gasPrice"]
            if not all(field in target_tx for field in required_fields):
                print(f"Transaction {tx_hash} is missing required fields.")
                return False
            decoded_tx = await self.decode_transaction_input(
                target_tx.get("input", "0x"),
                self.web3.to_checksum_address(target_tx.get("to", ""))
            )
            if not decoded_tx or "params" not in decoded_tx:
                return False
            path = decoded_tx["params"].get("path", [])
            if len(path) < 2:
                return False
            flashloan_asset = self.web3.to_checksum_address(path[0])
            flashloan_amount = self.calculate_flashloan_amount(target_tx)
            if flashloan_amount <= 0:
                return False
            flashloan_tx = await self.prepare_flashloan_transaction(
                flashloan_asset, flashloan_amount
            )
            if not flashloan_tx:
                return False
            front_run_tx_details = await self._prepare_front_run_transaction(target_tx)
            if not front_run_tx_details:
                return False
            back_run_tx_details = await self._prepare_back_run_transaction(target_tx, decoded_tx)
            if not back_run_tx_details:
                return False
            simulation_results = await asyncio.gather(
                self.simulate_transaction(flashloan_tx),
                self.simulate_transaction(front_run_tx_details),
                self.simulate_transaction(back_run_tx_details),
                return_exceptions=True
            )
            if any(isinstance(result, Exception) for result in simulation_results):
                print(f"Simulation failed for sandwich attack on transaction {tx_hash}")
                return False
            if not all(simulation_results):
                print(f"One or more simulations failed for sandwich attack on transaction {tx_hash}")
                return False
            if await self.send_bundle([flashloan_tx, front_run_tx_details, back_run_tx_details]):
                print(f"Sandwich attack executed successfully for {tx_hash}")
                return True
            else:
                print(f"Failed to execute sandwich attack for {tx_hash}")
                return False
        except Exception as e:
            print(f"Error executing sandwich attack for {tx_hash}: {e}")
            return False

    async def _prepare_front_run_transaction(
        self, target_tx: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Prepares the front-run transaction based on the target transaction.

        Args:
            target_tx (Dict[str, Any]): The target transaction details.

        Returns:
            Optional[Dict[str, Any]]: The prepared front-run transaction, or None if failed.
        """
        try:
            decoded_tx = await self.decode_transaction_input(
                target_tx.get("input", "0x"),
                self.web3.to_checksum_address(target_tx.get("to", ""))
            )
            if not decoded_tx:
                return None
            function_name = decoded_tx.get("function_name")
            if not function_name:
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
                print(f"Router address {to_address} not recognized for front-run.")
                return None
            router_contract, exchange_name = routers[to_address]
            if not router_contract:
                print(f"Router contract for {exchange_name} is not initialized.")
                return None
            try:
                front_run_function = getattr(router_contract.functions, function_name)(**function_params)
            except AttributeError as e:
                print(f"AttributeError: {e}")
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
        Prepares the back-run transaction based on the target transaction.

        Args:
            target_tx (Dict[str, Any]): The target transaction details.
            decoded_tx (Dict[str, Any]): The decoded transaction details.

        Returns:
            Optional[Dict[str, Any]]: The prepared back-run transaction, or None if failed.
        """
        try:
            function_name = decoded_tx.get("function_name")
            if not function_name:
                return None
            function_params = decoded_tx.get("params", {})
            path = function_params.get("path", [])
            if not path or not isinstance(path, list) or len(path) < 2:
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
                print(f"Router address {to_address} not recognized for back-run.")
                return None
            router_contract, exchange_name = routers[to_address]
            if not router_contract:
                print(f"Router contract for {exchange_name} is not initialized.")
                return None
            try:
                back_run_function = getattr(router_contract.functions, function_name)(**function_params)
            except AttributeError as e:
                print(f"AttributeError: {e}")
                return None
            back_run_tx = await self.build_transaction(back_run_function)
            return back_run_tx
        except Exception as e:
            print(f"Error preparing back-run transaction: {e}")
            return None

    async def decode_transaction_input(self, input_data: str, contract_address: str) -> Optional[Dict[str, Any]]:
        """
        Decodes the input data of a transaction.

        Args:
            input_data (str): The input data of the transaction.
            contract_address (str): The address of the contract.

        Returns:
            Optional[Dict[str, Any]]: Decoded transaction details, or None if failed.
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
            print(f"ContractLogicError in decode_transaction_input: {e}")
            return None
        except Exception as e:
            print(f"Error decoding transaction input: {e}")
            return None

    async def stop(self) -> None:
        """
        Stops the TransactionCore by performing any necessary cleanup.
        """
        try:
            # Currently, no ongoing tasks to stop
            pass
        except Exception as e:
            print(f"Error stopping TransactionCore: {e}")
            raise

class StrategyPerformanceMetrics:
    avg_execution_time: float = 0.0
    success_rate: float = 0.0
    total_executions: int = 0
    successes: int = 0
    failures: int = 0
    profit: Decimal = Decimal("0.0")

@dataclass
class StrategyConfiguration:
    decay_factor: float = 0.95
    min_profit_threshold: Decimal = Decimal("0.01")
    learning_rate: float = 0.01
    exploration_rate: float = 0.1

@dataclass
class StrategyExecutionError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message

class StrategyNet:
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
        if strategy_type not in self.strategy_types:
            logging.warning(f"Strategy type '{strategy_type}' is not recognized.")
            return
        self._strategy_registry[strategy_type].append(strategy_func)
        self.reinforcement_weights[strategy_type].append(1.0)
        logging.info(f"Registered new strategy '{strategy_func.__name__}' for type '{strategy_type}'.")

    def get_strategies(
        self,
        strategy_type: str
    ) -> List[Callable[[Dict[str, Any]], asyncio.Future]]:
        return self._strategy_registry.get(strategy_type, [])

    async def execute_best_strategy(
        self,
        target_tx: Dict[str, Any],
        strategy_type: str
    ) -> bool:
        strategies = self.get_strategies(strategy_type)
        if not strategies:
            logging.warning(f"No strategies available for type '{strategy_type}'.")
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

            logging.info(f"Executed strategy '{selected_strategy.__name__}' for tx '{target_tx.get('tx_hash', 'Unknown')}'. Success: {success}, Profit: {profit_made}, Time: {execution_time:.2f}s")
            return success

        except StrategyExecutionError as e:
            logging.error(f"Strategy execution error: {e}")
            return False
        except Exception as e:
            logging.error(f"Unexpected error during strategy execution: {e}")
            return False

    async def _select_best_strategy(
        self,
        strategies: List[Callable[[Dict[str, Any]], asyncio.Future]],
        strategy_type: str
    ) -> Callable[[Dict[str, Any]], asyncio.Future]:
        weights = self.reinforcement_weights[strategy_type]
        if not weights:
            logging.debug(f"No reinforcement weights found for strategy type '{strategy_type}'. Selecting random strategy.")
            return random.choice(strategies)

        if random.random() < self.configuration.exploration_rate:
            selected_strategy = random.choice(strategies)
            logging.debug(f"Exploration: Selected random strategy '{selected_strategy.__name__}'.")
            return selected_strategy

        # Apply softmax to weights for probability distribution
        max_weight = max(weights)
        exp_weights = [np.exp(w - max_weight) for w in weights]
        sum_exp = sum(exp_weights)
        probabilities = [w / sum_exp for w in exp_weights]

        selected_index = np.random.choice(len(strategies), p=probabilities)
        selected_strategy = strategies[selected_index]
        logging.debug(f"Selected strategy '{selected_strategy.__name__}' with probability {probabilities[selected_index]:.4f}.")
        return selected_strategy

    async def _update_strategy_metrics(
        self,
        strategy_name: str,
        strategy_type: str,
        success: bool,
        profit: Decimal,
        execution_time: float
    ) -> None:
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

        metrics.success_rate = metrics.successes / metrics.total_executions if metrics.total_executions > 0 else 0.0

        strategy_index = self.get_strategy_index(strategy_name, strategy_type)
        if strategy_index >= 0:
            reward = self._calculate_reward(success, profit, execution_time)
            self._update_reinforcement_weight(strategy_type, strategy_index, reward)
            logging.debug(f"Updated reinforcement weight for strategy '{strategy_name}': New weight = {self.reinforcement_weights[strategy_type][strategy_index]:.4f}")

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
        logging.debug(f"Strategy '{strategy_name}' metrics updated: {metrics}")

    def get_strategy_index(self, strategy_name: str, strategy_type: str) -> int:
        strategies = self.get_strategies(strategy_type)
        for index, strategy in enumerate(strategies):
            if strategy.__name__ == strategy_name:
                return index
        logging.warning(f"Strategy '{strategy_name}' not found in registry for type '{strategy_type}'.")
        return -1

    def _calculate_reward(
        self,
        success: bool,
        profit: Decimal,
        execution_time: float
    ) -> float:
        if success:
            base_reward = float(profit)
        else:
            base_reward = -0.1
        time_penalty = -0.01 * execution_time
        total_reward = base_reward + time_penalty
        return total_reward

    def _update_reinforcement_weight(
        self,
        strategy_type: str,
        index: int,
        reward: float
    ) -> None:
        lr = self.configuration.learning_rate
        current_weight = self.reinforcement_weights[strategy_type][index]
        new_weight = current_weight * (1 - lr) + reward * lr
        self.reinforcement_weights[strategy_type][index] = max(0.1, new_weight)

    # ========================= Strategy Implementations =========================

    async def high_value_eth_transfer(self, target_tx: Dict[str, Any]) -> bool:
        try:
            if not self._is_valid_transaction(target_tx):
                logging.debug("Transaction validation failed.")
                return False

            eth_value_in_wei, gas_price, to_address = self._extract_transaction_details(target_tx)
            eth_value = self.transactioncore.web3.from_wei(eth_value_in_wei, "ether")
            gas_price_gwei = self.transactioncore.web3.from_wei(gas_price, "gwei")
            threshold = self._calculate_thresholds(gas_price_gwei)

            if not await self._additional_validation_checks(eth_value_in_wei, to_address):
                logging.debug("Additional validation checks failed.")
                return False

            if eth_value_in_wei > threshold:
                logging.info(f"High-value ETH transfer detected: {eth_value} ETH. Executing handler.")
                return await self.transactioncore.handle_eth_transaction(target_tx)

            return False

        except Exception as e:
            logging.error(f"Error in high_value_eth_transfer strategy: {e}")
            return False

    def _is_valid_transaction(self, target_tx: Dict[str, Any]) -> bool:
        if not isinstance(target_tx, dict) or not target_tx:
            return False
        return True

    def _extract_transaction_details(self, target_tx: Dict[str, Any]) -> Tuple[int, int, str]:
        eth_value_in_wei = int(target_tx.get("value", 0))
        gas_price = int(target_tx.get("gasPrice", 0))
        to_address = target_tx.get("to", "")
        return eth_value_in_wei, gas_price, to_address

    def _calculate_thresholds(self, gas_price_gwei: float) -> int:
        base_threshold = self.transactioncore.web3.to_wei(10, "ether")
        if gas_price_gwei > 200:
            threshold = base_threshold * 2
        elif gas_price_gwei > 100:
            threshold = base_threshold * 1.5
        else:
            threshold = base_threshold
        return int(threshold)

    async def _additional_validation_checks(self, eth_value_in_wei: int, to_address: str) -> bool:
        if eth_value_in_wei <= 0:
            return False

        if not self.transactioncore.web3.is_address(to_address):
            return False

        is_contract = await self._is_contract_address(to_address)
        if is_contract:
            if not await self._validate_contract_interaction(to_address):
                return False

        return True

    async def _is_contract_address(self, address: str) -> bool:
        try:
            code = await self.transactioncore.web3.eth.get_code(address)
            is_contract = len(code) > 0
            return is_contract
        except Exception as e:
            logging.error(f"Error checking if address is contract: {e}")
            return False

    async def _validate_contract_interaction(self, contract_address: str) -> bool:
        try:
            token_symbols = self.configuration.TOKEN_SYMBOLS
            is_valid = contract_address in token_symbols
            return is_valid
        except Exception as e:
            logging.error(f"Error validating contract interaction: {e}")
            return False

    async def aggressive_front_run(self, target_tx: Dict[str, Any]) -> bool:
        try:
            if not self._is_valid_transaction(target_tx):
                logging.debug("Transaction validation failed.")
                return False

            tx_value = int(target_tx.get("value", 0))
            gas_price = int(target_tx.get("gasPrice", 0))

            value_eth = self.transactioncore.web3.from_wei(tx_value, "ether")
            threshold = self._calculate_dynamic_threshold(gas_price)

            risk_score = await self._assess_front_run_risk(target_tx)
            if risk_score < 0.5:
                logging.debug(f"Risk score too low: {risk_score}. Aborting front run.")
                return False

            if value_eth >= threshold:
                if value_eth > 10:
                    if not await self._validate_high_value_transaction(target_tx):
                        logging.debug("High-value transaction validation failed.")
                        return False

                logging.info(f"Aggressive front run triggered for transaction: {target_tx.get('tx_hash', 'Unknown')}")
                return await self.transactioncore.front_run(target_tx)

            return False

        except Exception as e:
            logging.error(f"Error in aggressive_front_run strategy: {e}")
            return False

    def _calculate_dynamic_threshold(self, gas_price: int) -> float:
        gas_price_gwei = float(self.transactioncore.web3.from_wei(gas_price, "gwei"))

        if gas_price_gwei > 200:
            threshold = 2.0
        elif gas_price_gwei > 100:
            threshold = 1.5
        elif gas_price_gwei > 50:
            threshold = 1.0
        else:
            threshold = 0.5

        return threshold

    async def _assess_front_run_risk(self, tx: Dict[str, Any]) -> float:
        try:
            risk_score = 1.0

            gas_price = int(tx.get("gasPrice", 0))
            gas_price_gwei = float(self.transactioncore.web3.from_wei(gas_price, "gwei"))
            if gas_price_gwei > 300:
                risk_score *= 0.7

            input_data = tx.get("input", "0x")
            if len(input_data) > 10:
                risk_score *= 0.8

            market_conditions = await self.marketmonitor.check_market_conditions(tx.get("to", ""))
            if market_conditions.get("high_volatility", False):
                risk_score *= 0.7
            if market_conditions.get("low_liquidity", False):
                risk_score *= 0.6

            risk_score = max(risk_score, 0.0)
            return round(risk_score, 2)
        except Exception as e:
            logging.error(f"Error assessing front run risk: {e}")
            return 0.0

    async def _validate_high_value_transaction(self, tx: Dict[str, Any]) -> bool:
        try:
            to_address = tx.get("to", "")
            if not to_address:
                return False

            code = await self.transactioncore.web3.eth.get_code(to_address)
            if not code:
                return False

            token_symbols = self.configuration.TOKEN_SYMBOLS
            if to_address not in token_symbols:
                return False

            return True
        except Exception as e:
            logging.error(f"Error validating high-value transaction: {e}")
            return False

    async def predictive_front_run(self, target_tx: Dict[str, Any]) -> bool:
        try:
            decoded_tx = await self._decode_transaction(target_tx)
            if not decoded_tx:
                logging.debug("Transaction decoding failed.")
                return False

            path = decoded_tx.get("params", {}).get("path", [])
            if not path or len(path) < 2:
                logging.debug("Invalid token swap path.")
                return False

            token_address = path[0]
            token_symbol = await self._get_token_symbol(token_address)
            if not token_symbol:
                logging.debug(f"Token symbol not found for address: {token_address}")
                return False

            try:
                predicted_price, current_price, market_conditions, historical_prices = await asyncio.gather(
                    self.marketmonitor.predict_price_movement(token_symbol),
                    self.apiconfig.get_real_time_price(token_symbol, vs_currency="eth"),
                    self.marketmonitor.check_market_conditions(target_tx.get("to", "")),
                    self.marketmonitor.fetch_historical_prices(token_symbol, days=1),
                    return_exceptions=True
                )

                if any(isinstance(result, Exception) for result in [predicted_price, current_price, market_conditions, historical_prices]):
                    logging.debug("Error fetching price data or market conditions.")
                    return False

                if current_price is None or predicted_price is None:
                    logging.debug("Current price or predicted price is None.")
                    return False

            except Exception as e:
                logging.error(f"Error fetching price data: {e}")
                return False

            price_change = (predicted_price / float(current_price) - 1) * 100
            volatility = np.std(historical_prices) / np.mean(historical_prices) if historical_prices else 0

            opportunity_score = await self._calculate_opportunity_score(
                price_change=price_change,
                volatility=volatility,
                market_conditions=market_conditions,
                current_price=current_price,
                historical_prices=historical_prices
            )

            if opportunity_score >= 75:
                logging.info(f"Predictive front run opportunity detected with score {opportunity_score}. Executing front run.")
                return await self.transactioncore.front_run(target_tx)

            logging.debug(f"Opportunity score {opportunity_score} is below threshold.")
            return False

        except Exception as e:
            logging.error(f"Error in predictive_front_run strategy: {e}")
            return False

    async def _calculate_opportunity_score(
        self,
        price_change: float,
        volatility: float,
        market_conditions: Dict[str, bool],
        current_price: float,
        historical_prices: List[float]
    ) -> float:
        score = 0.0

        if price_change > 5.0:
            score += 40
        elif price_change > 3.0:
            score += 30
        elif price_change > 1.0:
            score += 20
        elif price_change > 0.5:
            score += 10

        if volatility < 0.02:
            score += 20
        elif volatility < 0.05:
            score += 15
        elif volatility < 0.08:
            score += 10

        if market_conditions.get("bullish_trend", False):
            score += 10
        if not market_conditions.get("high_volatility", True):
            score += 5
        if not market_conditions.get("low_liquidity", True):
            score += 5

        if historical_prices and len(historical_prices) > 1:
            recent_trend = (historical_prices[-1] / historical_prices[0] - 1) * 100
            if recent_trend > 0:
                score += 20
            elif recent_trend > -1:
                score += 10

        return score

    async def volatility_front_run(self, target_tx: Dict[str, Any]) -> bool:
        try:
            decoded_tx = await self._decode_transaction(target_tx)
            if not decoded_tx:
                logging.debug("Transaction decoding failed.")
                return False

            path = decoded_tx.get("params", {}).get("path", [])
            if not path or len(path) < 2:
                logging.debug("Invalid token swap path.")
                return False

            token_symbol = await self._get_token_symbol(path[0])
            if not token_symbol:
                logging.debug(f"Token symbol not found for address: {path[0]}")
                return False

            results = await asyncio.gather(
                self.marketmonitor.check_market_conditions(target_tx.get("to", "")),
                self.apiconfig.get_real_time_price(token_symbol, vs_currency="eth"),
                self.marketmonitor.fetch_historical_prices(token_symbol, days=1),
                return_exceptions=True
            )

            market_conditions, current_price, historical_prices = results

            if any(isinstance(result, Exception) for result in [market_conditions, current_price, historical_prices]):
                logging.debug("Error fetching data for volatility front run.")
                return False

            volatility_score = await self._calculate_volatility_score(
                historical_prices=historical_prices,
                current_price=current_price,
                market_conditions=market_conditions
            )

            if volatility_score >= 75:
                logging.info(f"Volatility front run opportunity detected with score {volatility_score}. Executing front run.")
                return await self.transactioncore.front_run(target_tx)

            logging.debug(f"Volatility score {volatility_score} is below threshold.")
            return False

        except Exception as e:
            logging.error(f"Error in volatility_front_run strategy: {e}")
            return False

    async def _calculate_volatility_score(
        self,
        historical_prices: List[float],
        current_price: float,
        market_conditions: Dict[str, bool]
    ) -> float:
        score = 0.0

        if len(historical_prices) > 1:
            price_changes = np.diff(historical_prices) / np.array(historical_prices[:-1])
            volatility = np.std(price_changes)
            price_range = (max(historical_prices) - min(historical_prices)) / np.mean(historical_prices)

            if volatility > 0.1:
                score += 40
            elif volatility > 0.05:
                score += 30
            elif volatility > 0.02:
                score += 20

            if price_range > 0.15:
                score += 30
            elif price_range > 0.08:
                score += 20
            elif price_range > 0.03:
                score += 10

        if market_conditions.get("high_volatility", False):
            score += 15
        if market_conditions.get("bullish_trend", False):
            score += 10
        if not market_conditions.get("low_liquidity", True):
            score += 5

        return score

    async def advanced_front_run(self, target_tx: Dict[str, Any]) -> bool:
        # Placeholder for advanced front run strategy implementation
        logging.debug("Advanced front run strategy not implemented.")
        return False

    async def price_dip_back_run(self, target_tx: Dict[str, Any]) -> bool:
        decoded_tx = await self._decode_transaction(target_tx)
        if not decoded_tx:
            logging.debug("Transaction decoding failed.")
            return False

        path = decoded_tx.get("params", {}).get("path", [])
        if not path:
            logging.debug("Invalid token swap path.")
            return False

        token_symbol = await self._get_token_symbol(path[-1])
        if not token_symbol:
            logging.debug(f"Token symbol not found for address: {path[-1]}")
            return False

        current_price = await self.apiconfig.get_real_time_price(token_symbol, vs_currency="eth")
        if current_price is None:
            logging.debug("Current price not available.")
            return False

        predicted_price = await self.marketmonitor.predict_price_movement(token_symbol)
        if predicted_price < float(current_price) * 0.99:
            logging.info(f"Price dip detected for token '{token_symbol}'. Executing back run.")
            return await self.transactioncore.back_run(target_tx)

        return False

    async def flashloan_back_run(self, target_tx: Dict[str, Any]) -> bool:
        estimated_amount = await self.transactioncore.calculate_flashloan_amount(target_tx)
        estimated_profit = estimated_amount * Decimal("0.02")
        if estimated_profit > self.configuration.min_profit_threshold:
            logging.info(f"Flashloan back run triggered for tx '{target_tx.get('tx_hash', 'Unknown')}'. Estimated profit: {estimated_profit} ETH.")
            return await self.transactioncore.back_run(target_tx)
        return False

    async def high_volume_back_run(self, target_tx: Dict[str, Any]) -> bool:
        token_address = target_tx.get("to")
        token_symbol = await self._get_token_symbol(token_address)
        if not token_symbol:
            logging.debug(f"Token symbol not found for address: {token_address}")
            return False

        volume_24h = await self.apiconfig.get_token_volume(token_symbol, vs_currency="eth")
        volume_threshold = self._get_volume_threshold(token_symbol)
        if volume_24h > volume_threshold:
            logging.info(f"High volume detected for token '{token_symbol}'. Executing back run.")
            return await self.transactioncore.back_run(target_tx)

        return False

    def _get_volume_threshold(self, token_symbol: str) -> float:
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

        if token_symbol in tier1_tokens:
            threshold = tier1_tokens[token_symbol]
        elif token_symbol in tier2_tokens:
            threshold = tier2_tokens[token_symbol]
        elif token_symbol in tier3_tokens:
            threshold = tier3_tokens[token_symbol]
        elif token_symbol in volatile_tokens:
            threshold = volatile_tokens[token_symbol]
        else:
            threshold = 500_000

        return threshold

    async def advanced_back_run(self, target_tx: Dict[str, Any]) -> bool:
        decoded_tx = await self._decode_transaction(target_tx)
        if not decoded_tx:
            logging.debug("Transaction decoding failed.")
            return False

        market_conditions = await self.marketmonitor.check_market_conditions(target_tx.get("to", ""))
        if market_conditions.get("high_volatility", False) and market_conditions.get("bullish_trend", False):
            logging.info(f"Advanced back run triggered for tx '{target_tx.get('tx_hash', 'Unknown')}'.")
            return await self.transactioncore.back_run(target_tx)

        return False

    async def flash_profit_sandwich(self, target_tx: Dict[str, Any]) -> bool:
        estimated_amount = await self.transactioncore.calculate_flashloan_amount(target_tx)
        estimated_profit = estimated_amount * Decimal("0.02")
        if estimated_profit > self.configuration.min_profit_threshold:
            gas_price = await self.transactioncore.safetynet.get_dynamic_gas_price()
            if gas_price > Decimal("200"):
                logging.debug("Gas price too high for flash profit sandwich.")
                return False
            logging.info(f"Flash profit sandwich executed for tx '{target_tx.get('tx_hash', 'Unknown')}'. Estimated profit: {estimated_profit} ETH.")
            return await self.transactioncore.execute_sandwich_attack(target_tx)
        return False

    async def price_boost_sandwich(self, target_tx: Dict[str, Any]) -> bool:
        decoded_tx = await self._decode_transaction(target_tx)
        if not decoded_tx:
            logging.debug("Transaction decoding failed.")
            return False

        path = decoded_tx.get("params", {}).get("path", [])
        if not path:
            logging.debug("Invalid token swap path.")
            return False

        token_symbol = await self._get_token_symbol(path[0])
        if not token_symbol:
            logging.debug(f"Token symbol not found for address: {path[0]}")
            return False

        historical_prices = await self.marketmonitor.fetch_historical_prices(token_symbol, days=1)
        if not historical_prices:
            logging.debug(f"No historical prices fetched for token '{token_symbol}'.")
            return False

        momentum = await self._analyze_price_momentum(historical_prices)
        if momentum > 0.02:
            logging.info(f"Price boost sandwich triggered for token '{token_symbol}'. Momentum: {momentum}")
            return await self.transactioncore.execute_sandwich_attack(target_tx)

        return False

    async def _analyze_price_momentum(self, prices: List[float]) -> float:
        if not prices or len(prices) < 2:
            return 0.0
        price_changes = [prices[i] / prices[i - 1] - 1 for i in range(1, len(prices))]
        momentum = sum(price_changes) / len(price_changes)
        return momentum

    async def arbitrage_sandwich(self, target_tx: Dict[str, Any]) -> bool:
        decoded_tx = await self._decode_transaction(target_tx)
        if not decoded_tx:
            logging.debug("Transaction decoding failed.")
            return False

        path = decoded_tx.get("params", {}).get("path", [])
        if not path:
            logging.debug("Invalid token swap path.")
            return False

        token_symbol = await self._get_token_symbol(path[-1])
        if not token_symbol:
            logging.debug(f"Token symbol not found for address: {path[-1]}")
            return False

        is_arbitrage = await self.marketmonitor.is_arbitrage_opportunity(target_tx)
        if is_arbitrage:
            logging.info(f"Arbitrage opportunity detected for tx '{target_tx.get('tx_hash', 'Unknown')}'. Executing sandwich attack.")
            return await self.transactioncore.execute_sandwich_attack(target_tx)

        return False

    async def advanced_sandwich_attack(self, target_tx: Dict[str, Any]) -> bool:
        decoded_tx = await self._decode_transaction(target_tx)
        if not decoded_tx:
            logging.debug("Transaction decoding failed.")
            return False

        market_conditions = await self.marketmonitor.check_market_conditions(target_tx.get("to", ""))
        if market_conditions.get("high_volatility", False) and market_conditions.get("bullish_trend", False):
            logging.info(f"Advanced sandwich attack triggered for tx '{target_tx.get('tx_hash', 'Unknown')}'.")
            return await self.transactioncore.execute_sandwich_attack(target_tx)

        return False

    async def _decode_transaction(self, target_tx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            decoded = await self.transactioncore.decode_transaction_input(
                target_tx.get("input", ""), target_tx.get("to", "")
            )
            return decoded
        except Exception as e:
            logging.error(f"Error decoding transaction: {e}")
            return None

    async def _get_token_symbol(self, token_address: str) -> Optional[str]:
        try:
            symbol = await self.apiconfig.get_token_symbol(
                self.transactioncore.web3, token_address
            )
            return symbol
        except Exception as e:
            logging.error(f"Error fetching token symbol: {e}")
            return None

class MarketMonitor:
    MODEL_UPDATE_INTERVAL: int = 3600  # seconds
    VOLATILITY_THRESHOLD: float = 0.05
    LIQUIDITY_THRESHOLD: float = 100000.0

    def __init__(
        self,
        web3: 'AsyncWeb3',
        configuration: Optional['Configuration'] = None,
        apiconfig: Optional['APIConfig'] = None,
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
        self.is_running = False
        logging.debug("MarketMonitor initialized.")

    async def initialize(self) -> None:
        await self.load_model()
        self.is_running = True
        # Start periodic model training tasks for all monitored tokens
        tokens_to_monitor = await self.apiconfig.get_token_addresses()
        for token_address in tokens_to_monitor:
            token_symbol = await self.apiconfig.get_token_symbol(self.web3, token_address)
            if token_symbol:
                asyncio.create_task(self.periodic_model_training(token_symbol))
                logging.info(f"Started periodic model training for token '{token_symbol}'.")

    async def load_model(self) -> None:
        async with self.model_lock:
            if os.path.exists(self.model_path) and os.path.exists(self.training_data_path):
                try:
                    data = joblib.load(self.model_path)
                    self.price_model = data['model']
                    self.model_last_updated = data.get('model_last_updated', 0.0)
                    logging.info(f"Loaded price model from '{self.model_path}'.")
                except Exception as e:
                    logging.error(f"Error loading model: {e}")
            else:
                logging.warning("Model or training data path does not exist. Starting with a new model.")

    async def save_model(self) -> None:
        async with self.model_lock:
            try:
                data = {
                    'model': self.price_model,
                    'model_last_updated': self.model_last_updated
                }
                joblib.dump(data, self.model_path)
                logging.info(f"Saved price model to '{self.model_path}'.")
            except Exception as e:
                logging.error(f"Error saving model: {e}")

    async def _update_price_model(self, token_symbol: str) -> None:
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
                    logging.info(f"Price model updated for token '{token_symbol}'.")
                else:
                    logging.warning(f"Not enough historical prices to update model for token '{token_symbol}'.")
            except Exception as e:
                logging.error(f"Error updating price model for '{token_symbol}': {e}")

    async def periodic_model_training(self, token_symbol: str) -> None:
        while self.is_running:
            try:
                current_time = time.time()
                if current_time - self.model_last_updated > self.MODEL_UPDATE_INTERVAL:
                    await self._update_price_model(token_symbol)
                await asyncio.sleep(60)  # Check every minute
            except asyncio.CancelledError:
                logging.info(f"Periodic model training for '{token_symbol}' cancelled.")
                break
            except Exception as e:
                logging.error(f"Error in periodic model training for '{token_symbol}': {e}")
                await asyncio.sleep(60)

    async def check_market_conditions(self, token_address: str) -> Dict[str, Any]:
        market_conditions: Dict[str, Any] = {
            "high_volatility": False,
            "bullish_trend": False,
            "bearish_trend": False,
            "low_liquidity": False,
        }
        token_symbol = await self.apiconfig.get_token_symbol(self.web3, token_address)
        if not token_symbol:
            logging.debug(f"Token symbol not found for address: {token_address}")
            return market_conditions
        prices = await self.fetch_historical_prices(token_symbol, days=1)
        if len(prices) < 2:
            logging.debug(f"Not enough historical prices to assess market conditions for '{token_symbol}'.")
            return market_conditions
        volatility = self._calculate_volatility(prices)
        if volatility > self.VOLATILITY_THRESHOLD:
            market_conditions["high_volatility"] = True
        moving_average = np.mean(prices)
        if prices[-1] > moving_average:
            market_conditions["bullish_trend"] = True
        elif prices[-1] < moving_average:
            market_conditions["bearish_trend"] = True
        volume = await self.apiconfig.get_token_volume(token_symbol, vs_currency="eth")
        if volume < self.LIQUIDITY_THRESHOLD:
            market_conditions["low_liquidity"] = True
        logging.debug(f"Market conditions for '{token_symbol}': {market_conditions}")
        return market_conditions

    def _calculate_volatility(self, prices: List[float]) -> float:
        prices_array = np.array(prices)
        returns = np.diff(prices_array) / prices_array[:-1]
        return np.std(returns)

    async def fetch_historical_prices(self, token_symbol: str, days: int = 30) -> List[float]:
        cache_key = f"historical_prices_{token_symbol}_{days}"
        if cache_key in self.price_cache:
            logging.debug(f"Fetching historical prices for '{token_symbol}' from cache.")
            return self.price_cache[cache_key]
        prices = await self.apiconfig.fetch_historical_prices(token_symbol, days=days)
        if prices:
            self.price_cache[cache_key] = prices
            logging.debug(f"Fetched and cached historical prices for '{token_symbol}'.")
        else:
            logging.debug(f"No historical prices fetched for '{token_symbol}'.")
        return prices or []

    async def get_token_volume(self, token_symbol: str) -> float:
        cache_key = f"token_volume_{token_symbol}"
        if cache_key in self.price_cache:
            logging.debug(f"Fetching token volume for '{token_symbol}' from cache.")
            return self.price_cache[cache_key]
        volume = await self.apiconfig.get_token_volume(token_symbol, vs_currency="eth")
        if volume is not None:
            self.price_cache[cache_key] = volume
            logging.debug(f"Fetched and cached token volume for '{token_symbol}': {volume}")
        else:
            logging.debug(f"No token volume fetched for '{token_symbol}'.")
        return volume or 0.0

    async def predict_price_movement(self, token_symbol: str) -> float:
        try:
            current_time = time.time()
            if current_time - self.model_last_updated > self.MODEL_UPDATE_INTERVAL:
                await self._update_price_model(token_symbol)
            prices = await self.fetch_historical_prices(token_symbol, days=1)
            if not prices:
                logging.debug(f"No historical prices available for price prediction for '{token_symbol}'.")
                return 0.0
            X_pred = np.array([[len(prices)]])
            predicted_price = self.price_model.predict(X_pred)[0]
            logging.debug(f"Predicted price for '{token_symbol}': {predicted_price}")
            return float(predicted_price)
        except Exception as e:
            logging.error(f"Error predicting price movement for '{token_symbol}': {e}")
            return 0.0

    async def is_arbitrage_opportunity(self, target_tx: Dict[str, Any]) -> bool:
        decoded_tx = await self.decode_transaction_input(target_tx.get("input", ""), target_tx.get("to", ""))
        if not decoded_tx:
            logging.debug("Transaction decoding failed.")
            return False
        path = decoded_tx.get("params", {}).get("path", [])
        if len(path) < 2:
            logging.debug("Invalid token swap path for arbitrage opportunity.")
            return False
        token_address = path[-1]
        token_symbol = await self.apiconfig.get_token_symbol(self.web3, token_address)
        if not token_symbol:
            logging.debug(f"Token symbol not found for address: {token_address}")
            return False
        prices = await self._get_prices_from_services(token_symbol)
        if len(prices) < 2:
            logging.debug(f"Not enough price data to assess arbitrage for '{token_symbol}'.")
            return False
        price_difference = abs(prices[0] - prices[1])
        average_price = sum(prices) / len(prices)
        if average_price == 0:
            logging.debug(f"Average price is zero for '{token_symbol}'.")
            return False
        price_difference_percentage = price_difference / average_price
        if price_difference_percentage > 0.01:
            logging.info(f"Arbitrage opportunity detected for '{token_symbol}' with price difference {price_difference_percentage*100:.2f}%.")
            return True
        return False

    async def _get_prices_from_services(self, token_symbol: str) -> List[float]:
        prices: List[float] = []
        for service in self.apiconfig.apiconfig.keys():
            try:
                price = await self.apiconfig.get_real_time_price(token_symbol, vs_currency="eth", service=service)
                if price is not None:
                    prices.append(float(price))
            except Exception as e:
                logging.error(f"Error fetching price from service '{service}': {e}")
        return prices

    async def decode_transaction_input(self, input_data: str, contract_address: str) -> Optional[Dict[str, Any]]:
        try:
            decoded = await self.transactioncore.decode_transaction_input(input_data, contract_address)
            return decoded
        except Exception as e:
            logging.error(f"Error decoding transaction input: {e}")
            return None

class MainCore:
    def __init__(self, configuration: Optional[Configuration] = None) -> None:
        self.configuration: Configuration = configuration or Configuration()
        self.apiconfig: Optional['APIConfig'] = None
        self.web3: Optional[AsyncWeb3] = None
        self.mempoolmonitor: Optional['MempoolMonitor'] = None
        self.account: Optional[Account] = None
        self.marketmonitor: Optional[MarketMonitor] = None
        self.transactioncore: Optional['TransactionCore'] = None
        self.strategynet: Optional['StrategyNet'] = None
        self.safetynet: Optional['SafetyNet'] = None
        self.noncecore: Optional['NonceCore'] = None
        self.is_running = False
        logging.debug("MainCore initialized.")

    async def initialize(self) -> None:
        try:
            await self.configuration.load()

            wallet_key = self.configuration.WALLET_KEY
            if not wallet_key:
                raise ValueError("Wallet key is not set in configuration.")

            try:
                cleaned_key = wallet_key[2:] if wallet_key.startswith('0x') else wallet_key
                if not all(c in '0123456789abcdefABCDEF' for c in cleaned_key) or len(cleaned_key) != 64:
                    raise ValueError("Invalid wallet key format - must be a 64-character hexadecimal string")

                full_key = f"0x{cleaned_key}" if not wallet_key.startswith('0x') else wallet_key
                self.account = Account.from_key(full_key)
                logging.info("Account successfully initialized.")
            except Exception as e:
                raise ValueError(f"Invalid wallet key format: {e}")

            self.web3 = await self._initialize_web3()
            if not self.web3:
                raise RuntimeError("Failed to initialize Web3 connection")

            await self._check_account_balance()
            await self._initialize_components()
        except Exception as e:
            logging.error(f"Initialization failed: {e}")
            await self.stop()

    async def _initialize_web3(self) -> Optional[AsyncWeb3]:
        providers = self._get_providers()
        if not providers:
            logging.error("No valid providers found in configuration.")
            return None

        for provider_name, provider in providers:
            try:
                web3 = AsyncWeb3(provider, modules={"eth": (AsyncEth,)})
                if await self._test_connection(web3, provider_name):
                    await self._add_middleware(web3)
                    logging.info(f"Connected to Web3 provider '{provider_name}'.")
                    return web3
            except Exception as e:
                logging.error(f"Error connecting to provider '{provider_name}': {e}")
                continue

        logging.error("All Web3 providers failed to connect.")
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
                    logging.debug(f"Connected to chain ID {chain_id} via '{name}'.")
                    return True
            except Exception as e:
                logging.warning(f"Attempt {attempt + 1} failed to connect to '{name}': {e}")
                await asyncio.sleep(1)
        return False

    async def _add_middleware(self, web3: AsyncWeb3) -> None:
        try:
            chain_id = await web3.eth.chain_id
            if chain_id in {99, 100, 77, 7766, 56}:
                web3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
                logging.debug(f"Injected POA middleware for chain ID {chain_id}.")
            elif chain_id in {1, 61, 97, 42, 80001}:
                web3.middleware_onion.add(SignAndSendRawMiddlewareBuilder.build(self.account.key))
                logging.debug(f"Injected signing middleware for chain ID {chain_id}.")
            else:
                logging.debug(f"No middleware injected for chain ID {chain_id}.")
        except Exception as e:
            logging.error(f"Error adding middleware: {e}")
            raise

    async def _check_account_balance(self) -> None:
        try:
            if not self.account:
                raise ValueError("Account not initialized")

            balance = await self.web3.eth.get_balance(self.account.address)
            balance_eth = self.web3.from_wei(balance, 'ether')

            if balance_eth < 0.001:
                logging.warning(f"Account balance low: {balance_eth} ETH.")
            else:
                logging.info(f"Account balance: {balance_eth} ETH.")
        except Exception as e:
            logging.error(f"Error checking account balance: {e}")
            raise

    async def _initialize_components(self) -> None:
        try:
            self.apiconfig = APIConfig(self.configuration)
            await self.apiconfig.__aenter__()

            self.noncecore = NonceCore(
                web3=self.web3,
                address=self.account.address,
                configuration=self.configuration
            )
            await self.noncecore.initialize()

            self.safetynet = SafetyNet(
                web3=self.web3,
                configuration=self.configuration,
                address=self.account.address,
                account=self.account,
                apiconfig=self.apiconfig
            )

            erc20_abi = await self._load_abi(self.configuration.ERC20_ABI)
            aave_flashloan_abi = await self._load_abi(self.configuration.AAVE_FLASHLOAN_ABI)
            aave_lending_pool_abi = await self._load_abi(self.configuration.AAVE_LENDING_POOL_ABI)

            self.marketmonitor = MarketMonitor(
                web3=self.web3,
                configuration=self.configuration,
                apiconfig=self.apiconfig
            )
            await self.marketmonitor.initialize()

            tokens_to_monitor = await self.apiconfig.get_token_addresses()
            for token_address in tokens_to_monitor:
                token_symbol = await self.apiconfig.get_token_symbol(self.web3, token_address)
                if token_symbol:
                    await self.marketmonitor.periodic_model_training(token_symbol)
                    logging.info(f"Started periodic model training for token '{token_symbol}'.")

            self.mempoolmonitor = MempoolMonitor(
                web3=self.web3,
                safetynet=self.safetynet,
                noncecore=self.noncecore,
                apiconfig=self.apiconfig,
                monitored_tokens=tokens_to_monitor,
                erc20_abi=erc20_abi,
                configuration=self.configuration
            )
            await self.mempoolmonitor.initialize()

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
            logging.info("All components initialized successfully.")
        except Exception as e:
            logging.error(f"Error initializing components: {e}")
            await self.stop()

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

            self.is_running = True
            logging.info("Starting mempool monitoring.")
            asyncio.create_task(self.mempoolmonitor.start_monitoring())

            while self.is_running:
                try:
                    await self._process_profitable_transactions()
                    await asyncio.sleep(1)
                except asyncio.CancelledError:
                    logging.info("MainCore run loop cancelled.")
                    break
                except Exception as e:
                    logging.error(f"Error in main run loop: {e}")
                    await asyncio.sleep(5)

        except KeyboardInterrupt:
            logging.info("KeyboardInterrupt received. Shutting down.")
        except Exception as e:
            logging.error(f"Error in run method: {e}")
        finally:
            await self.stop()

    async def stop(self) -> None:
        if not self.is_running:
            return
        self.is_running = False
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
                self.marketmonitor.is_running = False  # Stop periodic tasks

            if self.apiconfig:
                await self.apiconfig.__aexit__(None, None, None)

            if self.web3:
                await self.web3.provider.disconnect()
                logging.info("Web3 provider disconnected.")

        except Exception as e:
            logging.error(f"Error during shutdown: {e}")
        finally:
            logging.info("MainCore shutdown complete.")
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
                    logging.debug("Profitable transaction validation failed.")
                    continue

                tx_hash = tx.get('tx_hash', 'Unknown')[:10]
                strategy_type = self._determine_strategy_type(tx)

                if not await self._is_tx_still_valid(tx):
                    logging.debug(f"Transaction '{tx_hash}' is no longer pending.")
                    continue

                success = await asyncio.wait_for(
                    strategy.execute_best_strategy(tx, strategy_type),
                    timeout=30.0
                )

                execution_time = time.time() - start_time
                self._log_execution_metrics(tx_hash, success, execution_time)

                if success:
                    logging.info(f"Strategy executed successfully for transaction '{tx_hash}'.")
                else:
                    logging.info(f"Strategy execution failed for transaction '{tx_hash}'.")

            except asyncio.TimeoutError:
                logging.warning("Timeout while processing profitable transaction.")
                continue

            except Exception as e:
                tx_hash = tx.get('tx_hash', 'Unknown')[:10] if tx else 'Unknown'
                logging.error(f"Error processing transaction '{tx_hash}': {e}")

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
        if not is_valid:
            logging.debug(f"Transaction missing required fields: {tx}")
        return is_valid

    def _determine_strategy_type(self, tx: Dict[str, Any]) -> str:
        if tx.get('value', 0) > 0:
            return 'eth_transaction'
        elif self._is_token_swap(tx):
            gas_price_gwei = self.web3.from_wei(tx.get('gasPrice', 0), 'gwei')
            if gas_price_gwei > 200:
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
        except TransactionNotFound:
            logging.debug(f"Transaction '{tx.get('tx_hash', 'Unknown')}' not found.")
            return False
        except Exception as e:
            logging.error(f"Error checking transaction status: {e}")
            return False

    def _is_token_swap(self, tx: Dict[str, Any]) -> bool:
        return (
            len(tx.get('input', '0x')) > 10
            and tx.get('value', 0) == 0
        )

    def _log_execution_metrics(self, tx_hash: str, success: bool, execution_time: float) -> None:
        try:
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            logging.info(f"Transaction '{tx_hash}' executed. Success: {success}, Execution Time: {execution_time:.2f}s, Memory Usage: {memory_usage:.2f} MB")
        except ImportError:
            logging.info(f"Transaction '{tx_hash}' executed. Success: {success}, Execution Time: {execution_time:.2f}s")


