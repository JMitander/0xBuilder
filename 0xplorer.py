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
import numpy as np
import tracemalloc
import hexbytes
import pandas as pd

from cachetools import *
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from typing import *
from decimal import Decimal

from eth_account.messages import *
from web3.exceptions import TransactionNotFound, ContractLogicError
from web3 import AsyncWeb3, AsyncIPCProvider, AsyncHTTPProvider, Web3
from web3.middleware import *
from web3.geth import *
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


class NonceManager:
    """
    Manages the nonce for an Ethereum account to prevent transaction nonce collisions.
    """

    def __init__(
        self,
        web3: AsyncWeb3,
        address: str,
        logger: Optional[logging.Logger] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self.web3 = web3
        self.address = address
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.max_retries = max(1, max_retries)  # Ensure at least one retry
        self.retry_delay = max(0.1, retry_delay)  # Ensure delay is positive
        self.lock = asyncio.Lock()
        self.current_nonce = None  # Will be initialized in the async method

    async def initialize(self):
        self.current_nonce = await self._fetch_current_nonce_with_retries()

    async def _fetch_current_nonce_with_retries(self) -> int:

        await loading_bar("Fetching Current Nonce", 0)
        attempt = 0
        while attempt < self.max_retries:
            try:
                nonce = await self.web3.eth.get_transaction_count(
                    self.address, block_identifier="pending"
                )
                self.logger.debug(
                    f"Initialized NonceManager for {self.address} with starting nonce {nonce}. ✅"
                )
                return nonce
            except Exception as e:
                attempt += 1
                self.logger.error(
                    f"Attempt {attempt} - Failed to fetch nonce for {self.address}: {e}. Retrying... ⚠️🔄"
                )
                await asyncio.sleep(self.retry_delay)
        self.logger.error(
            f"Failed to fetch nonce for {self.address} after {self.max_retries} attempts. ❌"
        )
        raise RuntimeError(
            f"Could not fetch nonce for {self.address} after multiple attempts. ❌"
        )

    async def get_nonce(self) -> int:

        async with self.lock:
            nonce = self.current_nonce
            self.current_nonce += 1
            self.logger.debug(
                f"Allocated nonce {nonce} for {self.address}. Next nonce will be {self.current_nonce}."
            )
            return nonce

    async def refresh_nonce(self):

        async with self.lock:
            latest_nonce = await self._fetch_current_nonce_with_retries()
            if latest_nonce > self.current_nonce:
                self.logger.debug(
                    f"Refreshing nonce. Updated from {self.current_nonce} to {latest_nonce}. 🔄"
                )
                self.current_nonce = latest_nonce
            else:
                self.logger.debug(
                    f"No refresh needed. Current nonce {self.current_nonce} is already in sync. ✨✅"
                )

    async def sync_nonce_with_chain(self):

        async with self.lock:
            await loading_bar("Synchronizing Nonce", 0)
            try:
                self.current_nonce = await self._fetch_current_nonce_with_retries()
                self.logger.debug(
                    f"Nonce synchronized successfully to {self.current_nonce}. ✨"
                )
            except Exception as e:
                self.logger.error(f"Failed to sync nonce for {self.address}: {e} ❌")
                raise RuntimeError(f"Failed to synchronize nonce: {e} ❌")

    async def handle_nonce_discrepancy(self, external_nonce: int):

        async with self.lock:
            if external_nonce > self.current_nonce:
                self.logger.warning(
                    f"Discrepancy detected: External nonce {external_nonce} is higher than internal nonce {self.current_nonce}. Adjusting. ⚠️"
                )
                self.current_nonce = (
                    external_nonce + 1
                )  # Move to the next available nonce
                self.logger.debug(f"Nonce adjusted to {self.current_nonce}.")
            else:
                self.logger.debug(
                    f"No discrepancy. External nonce {external_nonce} is not higher than the internal nonce. ✨"
                )

    async def reset_nonce(self):

        async with self.lock:
            await loading_bar("Resetting Nonce", 0)
            try:
                self.current_nonce = await self._fetch_current_nonce_with_retries()
                self.logger.debug(
                    f"Nonce reset successfully to {self.current_nonce}. ✨"
                )
            except Exception as e:
                self.logger.error(f"Failed to reset nonce for {self.address}: {e} ❌")
                raise RuntimeError(f"Failed to reset nonce: {e} ❌")


class SafetyNet:
    def __init__(
        self,
        web3: AsyncWeb3,
        config: Config,
        account: Account,
        logger: Optional[logging.Logger] = None,
    ):
        self.web3 = web3
        self.config = config
        self.account = account
        self.token_symbols = self.config.TOKEN_SYMBOLS
        self.symbol_mapping = self._load_token_symbols()
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.logger.info("SafetyNet initialized. 🛡️✅")
        self.api_success_rate = {
            "binance": 1.0,
            "coingecko": 1.0,
            "coinmarketcap": 1.0,
            "cryptocompare": 1.0,
        }
        self.api_success_rate_lock = asyncio.Lock()

    async def get_balance(self, account: Account) -> Decimal:

        try:
            balance_wei = await self.web3.eth.get_balance(account.address)
            balance_eth = self.web3.from_wei(balance_wei, "ether")
            self.logger.debug(
                f"Balance for account {account.address}: {balance_eth} ETH 💰"
            )
            return Decimal(balance_eth)
        except Exception as e:
            self.logger.error(
                f"Failed to retrieve balance for account {account.address}: {e} ❌"
            )
            return Decimal(0)

    async def ensure_profit(
        self,
        transaction_data: Dict[str, Any],
        minimum_profit_eth: Optional[float] = None,
    ) -> bool:

        if minimum_profit_eth is None:
            if await self.get_balance(self.account) < Decimal("0.5"):
                minimum_profit_eth = 0.003  # Lower threshold for small balances
            else:
                minimum_profit_eth = 0.01

        try:
            # Fetch dynamic gas price
            gas_price_gwei = await self.get_dynamic_gas_price()
            gas_price_gwei = Decimal(gas_price_gwei)

            # Estimate gas used
            gas_used = await self.estimate_gas(transaction_data)
            if gas_used == 0:
                self.logger.error(
                    "Gas used for the transaction is not defined or is zero. ⚠️⛽"
                )
                return False

            # Calculate gas cost in ETH
            gas_cost_eth = gas_price_gwei * gas_used * Decimal("1e-9")
            self.logger.debug(
                f"Gas Cost Calculation:\n"
                f" - Gas Price: {gas_price_gwei} Gwei\n 💸"
                f" - Gas Used: {gas_used}\n ⛽"
                f" - Gas Cost: {gas_cost_eth:.6f} ETH 💰"
            )

            # Adjust slippage tolerance based on market conditions
            slippage_tolerance = await self.adjust_slippage_tolerance()

            # Fetch and calculate expected output based on current real-time price
            output_token = transaction_data["output_token"]
            real_time_price = await self.get_real_time_price(output_token)
            if real_time_price == 0:
                self.logger.error(
                    f"Real-time price for token {output_token} could not be determined. Aborting profit estimation. ⚠️"
                )
                return False

            expected_output = Decimal(real_time_price) * Decimal(
                transaction_data["amountOut"]
            )
            input_amount = Decimal(transaction_data["amountIn"])

            # Adjust expected output based on slippage tolerance
            slippage_adjusted_output = expected_output * (
                1 - Decimal(slippage_tolerance)
            )
            profit = slippage_adjusted_output - input_amount - gas_cost_eth

            # Log all critical values involved in the profit calculation
            self.logger.debug(
                f"Profit Calculation:\n"
                f" - Real-time Price: {real_time_price} ETH per token\n 💎🦄"
                f" - Expected Output: {expected_output:.6f} ETH\n 📈"
                f" - Slippage Adjusted Output: {slippage_adjusted_output:.6f} ETH\n 🔄"
                f" - Input Amount: {input_amount:.6f} ETH\n 📥"
                f" - Gas Cost: {gas_cost_eth:.6f} ETH\n ⛽"
                f" - Calculated Profit: {profit:.6f} ETH 💹"
            )

            # Ensure profit exceeds minimum profit threshold
            is_profitable = profit > Decimal(minimum_profit_eth)
            self.logger.debug(
                "Transaction is profitable."
                if is_profitable
                else "Transaction is not profitable. ⚠️"
            )
            return is_profitable

        except KeyError as e:
            self.logger.error(
                f"Missing key in transaction data: {e}. Data: {transaction_data} ⚠️"
            )
        except Exception as e:
            self.logger.exception(f"Error ensuring transaction profitability: {e} ⚠️")

        return False

    async def estimate_gas(self, transaction_data: Dict[str, Any]) -> int:

        try:
            tx = {
                "from": self.account.address,
                "to": transaction_data.get("to"),
                "value": transaction_data.get("value", 0),
                "data": transaction_data.get("input", ""),
            }
            gas_estimate = await self.web3.eth.estimate_gas(tx)
            return gas_estimate
        except Exception as e:
            self.logger.error(f"Gas estimation failed: {e} ⚠️")
            return 0

    async def get_dynamic_gas_price(self) -> float:

        try:
            # Try Etherscan API
            async with aiohttp.ClientSession() as session:
                response = await session.get(
                    f"https://api.etherscan.io/api",
                    params={
                        "module": "gastracker",
                        "action": "gasoracle",
                        "apikey": self.config.ETHERSCAN_API_KEY,
                    },
                    timeout=10,
                )
                data = await response.json()
                return float(data["result"]["ProposeGasPrice"])
        except Exception as e:
            self.logger.warning(f"Etherscan gas price fetch failed: {e} ⛽⚠️")
            # Fallback to AsyncWeb3
            try:
                gas_price = await self.web3.eth.gas_price
                return self.web3.from_wei(gas_price, "gwei")
            except Exception as e:
                self.logger.error(f"AsyncWeb3 gas price fetch failed: {e} ⛽⚠️")
                # Default value
                return 100.0

    async def adjust_slippage_tolerance(self) -> float:

        network_congestion = await self.get_network_congestion()
        if network_congestion > 0.8:
            self.logger.debug(
                "High network congestion detected. Tightening slippage tolerance. 📉"
            )
            return 0.05  # Tighten slippage tolerance
        elif network_congestion < 0.5:
            self.logger.debug(
                "Low network congestion detected. Relaxing slippage tolerance. 📊"
            )
            return 0.2  # Relax slippage tolerance
        else:
            self.logger.debug(
                "Moderate network congestion. Using default slippage tolerance. 📈"
            )
            return 0.1  # Default slippage tolerance

    async def get_network_congestion(self) -> float:

        try:
            pending_block = await self.web3.eth.get_block("pending", full_transactions=False)
            pending_tx = len(pending_block["transactions"])
            congestion_level = min(pending_tx / 10000, 1.0)  # Assuming 10,000 pending txs as highly congested
            self.logger.debug(f"Network congestion level: {congestion_level} 📡")
            return congestion_level
        except Exception as e:
            self.logger.error(f"Failed to get network congestion: {e} ⚠️")
            return 1.0  # Assume high congestion if failed

    async def get_real_time_price(self, token: str) -> Decimal:

        try:
            await loading_bar(f"Fetching Real-Time Price for {token}", 0)
            price_sources = {
                "binance": await self._fetch_price_from_binance(token),
                "coingecko": await self._fetch_price_from_coingecko(token),
                "coinmarketcap": await self._fetch_price_from_coinmarketcap(token),
                "cryptocompare": await self._fetch_price_from_cryptocompare(token),
            }

            # Prioritize sources with higher historical success rate
            async with self.api_success_rate_lock:
                sorted_sources = sorted(
                    price_sources.items(),
                    key=lambda x: self.api_success_rate.get(x[0], 1.0),
                    reverse=True,
                )

            for source, price in sorted_sources:
                if price is not None:
                    return price

        except Exception as e:
            self.logger.error(f"Error fetching real-time price for {token}: {e} ⚠️")

        # If no price could be fetched, log and return 0
        self.logger.error(f"Failed to retrieve price for {token}. Returning 0. ⚠️")
        return Decimal(0)

    async def _fetch_price_from_binance(self, token: str) -> Optional[Decimal]:

        try:
            symbol = self._convert_token_id_to_binance_symbol(token)
            if not symbol:
                return None
            url = f"https://api.binance.com/api/v3/ticker/price"
            params = {"symbol": symbol}
            response = await self.make_request(url, params=params)
            data = await response.json()
            price_usdt = Decimal(data["price"])
            self.logger.debug(
                f"Real-time price for {token} from Binance: {price_usdt} USDT 💹"
            )
            # Fetch ETHUSDT price to convert USDT to ETH
            eth_price_usdt = await self.get_eth_price_from_binance()
            if eth_price_usdt:
                price_in_eth = price_usdt / eth_price_usdt
                return price_in_eth
            else:
                return None
        except Exception as e:
            self.logger.error(f"Binance price fetch failed for {token}: {e} ⚠️")
            async with self.api_success_rate_lock:
                self.api_success_rate["binance"] = (
                    self.api_success_rate.get("binance", 1.0) * 0.9
                )  # Lower success rate
            return None

    async def get_eth_price_from_binance(self) -> Optional[Decimal]:
        try:
            url = f"https://api.binance.com/api/v3/ticker/price"
            params = {"symbol": "ETHUSDT"}
            response = await self.make_request(url, params=params)
            data = await response.json()
            eth_price_usdt = Decimal(data["price"])
            return eth_price_usdt
        except Exception as e:
            self.logger.error(f"Failed to fetch ETH price from Binance: {e} ⚠️")
            return None

    def _load_token_symbols(self) -> dict:

        try:
            with open(self.token_symbols, "r") as file:
                return json.load(file)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.logger.error(f"Error loading token symbols: {e}")
            return {}

    def _convert_token_id_to_binance_symbol(self, token_id: str) -> Optional[str]:

        return self.symbol_mapping.get(token_id.lower())

    async def _fetch_price_from_coingecko(self, token: str) -> Optional[Decimal]:
        try:
            url = f"https://api.coingecko.com/api/v3/simple/price"
            params = {"ids": token, "vs_currencies": "eth"}
            response = await self.make_request(url, params=params)
            price_data = await response.json()
            price = Decimal(str(price_data[token]["eth"]))
            self.logger.debug(
                f"Real-time price for {token} from CoinGecko: {price} ETH"
            )
            return price
        except Exception as e:
            self.logger.error(f"CoinGecko price fetch failed for {token}: {e} ⚠️")
            async with self.api_success_rate_lock:
                self.api_success_rate["coingecko"] *= 0.9  # Lower success rate
            return None

    async def _fetch_price_from_coinmarketcap(self, token: str) -> Optional[Decimal]:

        try:
            url = f"https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"
            params = {"symbol": token}
            headers = {"X-CMC_PRO_API_KEY": self.config.COINMARKETCAP_API_KEY}
            response = await self.make_request(url, params=params, headers=headers)
            data = await response.json()
            price = Decimal(str(data["data"][token]["quote"]["ETH"]["price"]))
            self.logger.debug(
                f"Real-time price for {token} from CoinMarketCap: {price} ETH 💹"
            )
            return price
        except Exception as e:
            self.logger.error(f"CoinMarketCap price fetch failed for {token}: {e} ⚠️")
            async with self.api_success_rate_lock:
                self.api_success_rate["coinmarketcap"] *= 0.9  # Lower success rate
            return None

    async def _fetch_price_from_cryptocompare(self, token: str) -> Optional[Decimal]:

        try:
            url = f"https://min-api.cryptocompare.com/data/price"
            params = {"fsym": token, "tsyms": "ETH"}
            headers = {"Apikey": self.config.CRYPTOCOMPARE_API_KEY}
            response = await self.make_request(url, params=params, headers=headers)
            data = await response.json()
            price = Decimal(str(data["ETH"]))
            self.logger.debug(
                f"Real-time price for {token} from CryptoCompare: {price} ETH 💹"
            )
            return price
        except Exception as e:
            self.logger.error(f"CryptoCompare price fetch failed for {token}: {e} ⚠️")
            async with self.api_success_rate_lock:
                self.api_success_rate["cryptocompare"] *= 0.9  # Lower success rate
            return None

    async def make_request(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> aiohttp.ClientResponse:

        max_attempts = 5
        backoff_time = 1  # Initial backoff time in seconds

        for attempt in range(1, max_attempts + 1):
            try:
                await loading_bar("Making Request with Exponential Backoff", 1)
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        url, params=params, headers=headers, timeout=10
                    ) as response:
                        if response.status == 200:
                            return response
                        elif response.status == 429:
                            self.logger.warning(
                                f"Rate limit hit on attempt {attempt}. Backing off for {backoff_time} seconds."
                            )
                            await asyncio.sleep(backoff_time)
                            backoff_time *= 2  # Exponential backoff
                        else:
                            self.logger.error(f"HTTP error occurred: {response.status} ❌")
                            break
            except Exception as e:
                self.logger.error(f"Request error on attempt {attempt}: {e} ❌")
                if attempt < max_attempts:
                    await asyncio.sleep(backoff_time)
                    backoff_time *= 2  # Exponential backoff

        raise Exception("Failed to make request after several attempts.")


class MonitorArray:
    """
    MonitorArray class monitors the mempool for profitable transactions.
    """
    def __init__(
        self,
        web3: AsyncWeb3,
        safety_net: SafetyNet,
        nonce_manager: NonceManager,
        logger: Optional[logging.Logger] = None,
        monitored_tokens: Optional[List[str]] = None,
        erc20_ABI: List[Dict[str, Any]] = None,
        config: Config = None,
    ):
        self.web3 = web3
        self.config = config
        self.safety_net = safety_net
        self.nonce_manager = nonce_manager
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.profitable_transactions = (
            asyncio.Queue()
        )  # Async queue to store identified profitable transactions
        self.running = False
        self.monitored_tokens = monitored_tokens or []
        self.erc20_ABI = erc20_ABI or []
        self.token_symbol_cache = TTLCache(
            maxsize=1000, ttl=86400
        )  # Cache for token symbols (24 hours)
        self.minimum_profit_threshold = Decimal(
            "0.001"
        )  # Minimum profit threshold in ETH
        self.processed_transactions: Set[str] = set()
        self.logger.info("MonitorArray initialized and ready for monitoring. 📡✅")

    async def start_monitoring(self):
        if self.running:
            self.logger.warning("Monitoring is already running.")
            return

        self.running = True
        asyncio.create_task(self._run_monitoring())
        self.logger.info("Mempool monitoring started. 📡 ✅")

    async def stop_monitoring(self):
        if not self.running:
            self.logger.warning("Monitoring is not running.")
            return

        self.running = False
        self.logger.info("Mempool monitoring has been stopped. 🛑")

    async def _run_monitoring(self):
        await self.mempool_monitor()

    async def mempool_monitor(self):
        self.logger.info("Starting mempool monitoring... 📡")
        
        if not isinstance(self.web3.provider, (AsyncHTTPProvider, AsyncIPCProvider)):
            self.logger.error("Provider is not an HTTP, WebSocket, or IPC provider. ❌")
        else:
            self.logger.info(
                f"Connected to Ethereum network via {self.web3.provider.__class__.__name__}. ✨"
            )
            return
        try:
            pending_filter = await self.web3.eth.filter("pending")
        except Exception as e:
            self.logger.error(f"Error setting up pending transaction filter: {e} ❌")
            return

        while self.running:
            try:
                # Get new entries from the pending transaction filter
                tx_hashes = await pending_filter.get_new_entries()

                for tx_hash in tx_hashes:
                    await self.process_transaction(tx_hash)

            except Exception as e:
                self.logger.exception(f"Error in mempool monitoring: {str(e)} ⚠️")
                # Reinitialize the filter in case of disconnection or errors
                try:
                    pending_filter = await self.web3.eth.filter("pending")
                except Exception as e:
                    self.logger.error(
                        f"Error resetting pending transaction filter: {e} ❌"
                    )
                    await asyncio.sleep(5)  # Wait before retrying

            await asyncio.sleep(0.1)

    async def process_transaction(self, tx_hash):
        tx_hash_hex = tx_hash.hex()

        # Check if the transaction has already been processed
        if tx_hash_hex in self.processed_transactions:
            return

        # Mark the transaction as processed
        self.processed_transactions.add(tx_hash_hex)

        try:
            # Fetch the transaction details
            tx = await self.web3.eth.get_transaction(tx_hash)

            # Analyze the transaction
            analysis = await self.analyze_transaction(tx)
            if analysis.get("is_profitable"):
                await self.profitable_transactions.put(analysis)
                self.logger.info(
                    f"Identified profitable transaction {tx_hash_hex} in the mempool. 📡"
                )

        except TransactionNotFound:
            # Transaction details not yet available; may need to wait
            self.logger.debug(
                f"Transaction {tx_hash_hex} details not available yet. Will retry. ⏳"
            )
        except Exception as e:
            self.logger.exception(f"Error handling transaction {tx_hash_hex}: {e} ⚠️")

    async def analyze_transaction(self, tx) -> Dict[str, Any]:
        """Analyze a transaction to determine if it's profitable."""
        if not tx.hash or not tx.input:
            self.logger.debug(
                f"Transaction {tx.hash.hex()} is missing essential fields. Skipping."
            )
            return {"is_profitable": False}

        try:
            # Handle ETH transactions
            if tx.value > 0:
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

            # Handle token transactions
            return await self._analyze_token_transaction(tx)
        except Exception as e:
            self.logger.exception(f"Error analyzing transaction {tx.hash.hex()}: {e} ⚠️")
            return {"is_profitable": False}

    async def _analyze_token_transaction(self, tx) -> Dict[str, Any]:

        try:
            # Create a contract instance using the transaction's destination address and ERC20 ABI
            contract = self.web3.eth.contract(address=tx.to, abi=self.erc20_ABI)
            # Decode the transaction input to extract the function ABI and parameters
            function_ABI, function_params = contract.decode_function_input(tx.input)
            function_name = function_ABI["name"]

            # Check if the function name is in the list of ERC20 function signatures
            if function_name in self.config.ERC20_SIGNATURES:
                # Estimate the profit of the transaction
                estimated_profit = await self._estimate_profit(tx, function_params)

                # Check if the estimated profit exceeds the minimum profit threshold
                if estimated_profit > self.minimum_profit_threshold:
                    self.logger.info(
                        f"Identified profitable transaction {tx.hash.hex()} with estimated profit: {estimated_profit:.4f} ETH 💰"
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
                    self.logger.debug(
                        f"Transaction {tx.hash.hex()} is below threshold. Skipping... ⚠️"
                    )
                    return {"is_profitable": False}
            else:
                self.logger.debug(
                    f"Function {function_name} not in ERC20_SIGNATURES. Skipping."
                )
                return {"is_profitable": False}
        except Exception as e:
            self.logger.exception(
                f"Error decoding function input for transaction {tx.hash.hex()}: {e} ❌"
            )
            return {"is_profitable": False}

    async def _is_profitable_eth_transaction(self, tx) -> bool:
        try:
            # Estimate the potential profit of the ETH transaction
            potential_profit = await self._estimate_eth_transaction_profit(tx)
            # Return True if the potential profit exceeds the minimum profit threshold, otherwise False
            return potential_profit > self.minimum_profit_threshold
        except Exception as e:
            self.logger.exception(
                f"Error estimating ETH transaction profit for transaction {tx.hash.hex()}: {e} ❌"
            )
            return False

    async def _estimate_eth_transaction_profit(self, tx: Any) -> Decimal:

        try:
            # Retrieve the current dynamic gas price (assumed to be in Gwei)
            gas_price_gwei = await self.safety_net.get_dynamic_gas_price()
            # Retrieve the gas used for the transaction
            gas_used = tx.gas  # Note: tx.gas is the gas limit, actual gas used is unknown at this point

            # Calculate the gas cost in ETH
            gas_cost_eth = (
                Decimal(gas_price_gwei) * Decimal(gas_used) * Decimal("1e-9")
            )  # Convert Gwei to ETH

            # Convert the transaction value from Wei to ETH
            eth_value = Decimal(self.web3.from_wei(tx.value, "ether"))

            # Calculate the potential profit
            potential_profit = eth_value - gas_cost_eth

            # Return the potential profit if it's positive, otherwise return zero
            return potential_profit if potential_profit > 0 else Decimal(0)
        except Exception as e:
            self.logger.error(f"Error estimating ETH transaction profit: {e} ❌")
            return Decimal(0)

    async def _estimate_profit(self, tx, function_params: Dict[str, Any]) -> Decimal:

        try:
            # Convert gas price from Wei to Gwei
            gas_price_gwei = self.web3.from_wei(tx.gasPrice, "gwei")
            gas_used = tx.gas
            # Calculate gas cost in ETH
            gas_cost_eth = Decimal(gas_price_gwei) * Decimal(gas_used) * Decimal("1e-9")

            # Retrieve input and output amounts from function parameters
            input_amount_wei = Decimal(function_params.get("amountIn", 0))
            output_amount_min_wei = Decimal(function_params.get("amountOutMin", 0))
            path = function_params.get("path", [])

            # Validate the transaction path
            if len(path) < 2:
                self.logger.debug(
                    f"Transaction {tx.hash.hex()} has an invalid path for swapping. Skipping. ⚠️"
                )
                return Decimal(0)

            # Get the output token address and symbol
            output_token_address = path[-1]
            output_token_symbol = await self.get_token_symbol(output_token_address)
            if not output_token_symbol:
                self.logger.debug(
                    f"Output token symbol not found for address {output_token_address}. Skipping. ⚠️"
                )
                return Decimal(0)

            # Get the real-time market price of the output token
            market_price = await self.safety_net.get_real_time_price(
                output_token_symbol.lower()
            )
            if market_price is None or market_price == 0:
                self.logger.debug(
                    f"Market price not available for token {output_token_symbol}. Skipping. ⚠️"
                )
                return Decimal(0)

            # Convert input amount from Wei to ETH
            input_amount_eth = Decimal(self.web3.from_wei(input_amount_wei, "ether"))
            # Calculate the profit
            profit = (
                Decimal(market_price) * output_amount_min_wei
                - input_amount_eth
                - gas_cost_eth
            )

            # Return the profit if it's positive, otherwise return zero
            return profit if profit > 0 else Decimal(0)
        except Exception as e:
            self.logger.exception(
                f"Error estimating profit for transaction {tx.hash.hex()}: {e} ⚠️"
            )
            return Decimal(0)

    @cached(cache=TTLCache(maxsize=1000, ttl=86400))
    async def get_token_symbol(self, token_address: str) -> Optional[str]:

        try:
            # First check token symbols from environment variables
            if token_address in self.config.TOKEN_SYMBOLS:
                return self.config.TOKEN_SYMBOLS[token_address]
            # If not found, fetch from the blockchain
            contract = self.web3.eth.contract(address=token_address, abi=self.erc20_ABI)
            symbol = await contract.functions.symbol().call()
            return symbol
        except Exception as e:
            self.logger.error(f"Error getting symbol for token {token_address}: {e} ❌")
            return None

    async def _log_transaction_details(self, tx, is_eth=False):

        try:
            # Log the transaction details
            transaction_info = {
                "transaction hash": tx.hash.hex(),
                "value": self.web3.from_wei(tx.value, "ether") if is_eth else tx.value,
                "from": tx["from"],
                "to": (tx.to[:10] + "..." + tx.to[-10:]) if tx.to else None,
                "input": tx.input,
                "gas price": self.web3.from_wei(tx.gasPrice, "gwei"),
            }
            if is_eth:
                self.logger.info(
                    f"Pending ETH Transaction Details: {transaction_info} 📜"
                )
            else:
                self.logger.info(
                    f"Pending Token Transaction Details: {transaction_info} 📜"
                )
        except Exception as e:
            self.logger.exception(
                f"Error logging transaction details for {tx.hash.hex()}: {e} ⚠️"
            )

class TransactionArray:
    """
    TransactionArray class builds and executes transactions, including front-run,
    back-run, and sandwich attack strategies. It interacts with smart contracts,
    manages transaction signing, gas price estimation, and handles flashloans.
    """
    def __init__(
        self,
        web3: AsyncWeb3,
        account: Account,
        flashloan_contract_address: str,
        flashloan_contract_ABI: List[Dict[str, Any]],
        lending_pool_contract_address: str,
        lending_pool_contract_ABI: List[Dict[str, Any]],
        monitor: MonitorArray,
        nonce_manager: NonceManager,
        safety_net: SafetyNet,
        config: Config,
        logger: Optional[logging.Logger] = None,
        gas_price_multiplier: float = 1.1,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
        erc20_ABI: Optional[List[Dict[str, Any]]] = None,
    ):
        # since web3 is shared among all instances, we need to make sure it's an async web3 instance
        self.web3 = web3
        self.account = account
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.monitor = monitor
        self.nonce_manager = nonce_manager
        self.safety_net = safety_net
        self.gas_price_multiplier = gas_price_multiplier
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.erc20_ABI = erc20_ABI or []
        self.current_profit = Decimal("0")

        # Store contract details for async initialization
        self.flashloan_contract_address = flashloan_contract_address
        self.flashloan_contract_ABI = flashloan_contract_ABI
        self.lending_pool_contract_address = lending_pool_contract_address
        self.lending_pool_contract_ABI = lending_pool_contract_ABI

        self.logger.info("TransactionArray initialized successfully. ✅")

    # Initialize contracts asynchronously
    async def initialize(self):
        """Async initialization of contracts."""
        self.flashloan_contract = await self._initialize_contract(
            self.flashloan_contract_address,
            self.flashloan_contract_ABI,
            "Flashloan Contract",
        )
        self.lending_pool_contract = await self._initialize_contract(
            self.lending_pool_contract_address,
            self.lending_pool_contract_ABI,
            "Lending Pool Contract",
        )
        self.uniswap_router_contract = await self._initialize_contract(
            self.config.UNISWAP_V2_ROUTER_ADDRESS,
            self.config.UNISWAP_V2_ROUTER_ABI,
            "Uniswap Router Contract",
        )
        self.sushiswap_router_contract = await self._initialize_contract(
            self.config.SUSHISWAP_ROUTER_ADDRESS,
            self.config.SUSHISWAP_ROUTER_ABI,
            "Sushiswap Router Contract",
        )
        self.pancakeswap_router_contract = await self._initialize_contract(
            self.config.PANCAKESWAP_ROUTER_ADDRESS,
            self.config.PANCAKESWAP_ROUTER_ABI,
            "Pancakeswap Router Contract",
        )
        self.balancer_router_contract = await self._initialize_contract(
            self.config.BALANCER_ROUTER_ADDRESS,
            self.config.BALANCER_ROUTER_ABI,
            "Balancer Router Contract",
        )
        # Load ERC20 ABI if not provided
        self.erc20_ABI = self.erc20_ABI or await self._load_erc20_ABI()

    # initialize contract instance
    async def _initialize_contract(
        self,
        contract_address: str,
        contract_ABI: List[Dict[str, Any]],
        contract_name: str,
    ) -> Contract:

        try:
            contract_instance = self.web3.eth.contract(
                address=self.web3.to_checksum_address(contract_address),
                abi=contract_ABI,
            )
            self.logger.info(
                f"Loaded {contract_name} at {contract_address} successfully. ✅"
            )
            return contract_instance
        except Exception as e:
            self.logger.error(
                f"Failed to load {contract_name} at {contract_address}: {e} ❌"
            )
            raise ValueError(
                f"Contract initialization failed for {contract_name}"
            ) from e
    # Builds a transaction with the given function call and additional parameters
    async def build_transaction(
        self, function_call: Any, additional_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:

        additional_params = additional_params or {}
        try:
            tx_details = {
                "data": function_call.encode_ABI(),
                "to": function_call.address,
                "chainId": await self.web3.eth.chain_id,
                "nonce": await self.nonce_manager.get_nonce(),
                "from": self.account.address,
            }
            tx_details.update(additional_params)

            # Estimate gas after building the transaction
            tx = tx_details.copy()
            tx["gas"] = await self.estimate_gas_smart(tx)
            tx.update(await self.get_dynamic_gas_price())

            self.logger.debug(f"Built transaction: {tx}")
            return tx
        except Exception as e:
            self.logger.exception(f"Error building transaction: {e} ⚠️")
            raise

    async def get_dynamic_gas_price(self) -> Dict[str, int]:

        try:
            gas_price_gwei = await self.safety_net.get_dynamic_gas_price()
            self.logger.info(f"Fetched gas price: {gas_price_gwei} Gwei ⛽")
        except Exception as e:
            self.logger.error(
                f"Error fetching dynamic gas price: {e}. Using default gas price. ⛽⚠️"
            )
            gas_price_gwei = 100.0  # Default gas price in Gwei

        gas_price = int(
            self.web3.to_wei(gas_price_gwei * self.gas_price_multiplier, "gwei")
        )
        return {"gasPrice": gas_price}

    async def estimate_gas_smart(self, tx: Dict[str, Any]) -> int:

        try:
            gas_estimate = await self.web3.eth.estimate_gas(tx)
            self.logger.debug(f"Estimated gas: {gas_estimate} ⛽")
            return gas_estimate
        except Exception as e:
            self.logger.warning(
                f"Gas estimation failed: {e}. Using default gas limit of 100000 ⛽⚠️"
            )
            return 100_000  # Default gas limit

    async def execute_transaction(self, tx: Dict[str, Any]) -> Optional[str]:

        for attempt in range(1, self.retry_attempts + 1):
            try:
                # Sign the transaction
                signed_tx = await self.sign_transaction(tx)
                # Send the signed transaction
                tx_hash = await self.web3.eth.send_raw_transaction(signed_tx)
                tx_hash_hex = (
                    tx_hash.hex()
                    if isinstance(tx_hash, hexbytes.HexBytes)
                    else tx_hash
                )
                self.logger.info(
                    f"Transaction sent successfully with hash: {tx_hash_hex} 🚀✅"
                )
                # Refresh the nonce after a successful transaction
                await self.nonce_manager.refresh_nonce()
                return tx_hash_hex
            except Exception as e:
                self.logger.error(
                    f"Error executing transaction: {e}. Attempt {attempt} of {self.retry_attempts} 🔄"
                )
                if attempt < self.retry_attempts:
                    sleep_time = self.retry_delay * attempt
                    self.logger.info(f"Retrying in {sleep_time} seconds...")
                    await asyncio.sleep(sleep_time)  # Exponential backoff

        self.logger.error("Failed to execute transaction after multiple attempts. ❌")
        return None

    async def sign_transaction(self, transaction: Dict[str, Any]) -> bytes:

        try:
            signed_tx = self.web3.eth.account.sign_transaction(
                transaction,
                private_key=self.account.key,
            )
            self.logger.debug(
                f"Transaction signed successfully: Nonce {transaction['nonce']}. 📋"
            )
            return signed_tx.rawTransaction
        except Exception as e:
            self.logger.exception(f"Error signing transaction: {e} ⚠️")
            raise

    async def handle_eth_transaction(self, target_tx: Dict[str, Any]) -> bool:

        tx_hash = target_tx.get("tx_hash", "Unknown")
        self.logger.info(f"Handling ETH transaction {tx_hash} 🚀")

        try:
            # Extract the value of ETH to be transferred
            eth_value = target_tx.get("value", 0)

            # Build the transaction details
            tx_details = {
                "data": target_tx.get("input", "0x"),
                "chainId": await self.web3.eth.chain_id,
                "to": target_tx.get("to", ""),
                "value": eth_value,
                "gas": 21_000,  # Standard gas limit for ETH transfers
                "nonce": await self.nonce_manager.get_nonce(),
                "from": self.account.address,
            }

            # Use a gas price slightly higher than the original transaction
            original_gas_price = int(target_tx.get("gasPrice", 0))
            tx_details["gasPrice"] = int(
                original_gas_price * 1.1
            )  # 10% higher gas price

            # Log transaction details
            eth_value_ether = self.web3.from_wei(eth_value, "ether")
            self.logger.info(
                f"Building ETH front-run transaction for {eth_value_ether} ETH to {tx_details['to']}"
            )

            # Sign and execute the transaction
            tx_hash_executed = await self.execute_transaction(tx_details)
            if tx_hash_executed:
                self.logger.info(
                    f"Successfully executed ETH transaction with hash: {tx_hash_executed} ✅"
                )
                return True
            else:
                self.logger.error("Failed to execute ETH transaction. ❌")
                return False
        except Exception as e:
            self.logger.exception(f"Error handling ETH transaction: {e} ❌")
            return False

    def calculate_flashloan_amount(self, target_tx: Dict[str, Any]) -> int:
        # Extract the estimated profit from the target transaction
        estimated_profit = target_tx.get("profit", 0)
        if estimated_profit > 0:
            flashloan_amount = int(
                Decimal(estimated_profit) * Decimal("0.8")
            )  # Take 80% of estimated profit
            self.logger.info(
                f"Calculated flashloan amount: {flashloan_amount} Wei based on estimated profit. ⚡🏦"
            )
            return flashloan_amount
        else:
            self.logger.info("No estimated profit. Setting flashloan amount to 0. ⚡⚠️")
            return 0

    async def simulate_transaction(self, transaction: Dict[str, Any]) -> bool:
        # Log the transaction details
        self.logger.info(
            f"Simulating transaction with nonce {transaction.get('nonce', 'Unknown')}. 🔍📊"
        )
        try:
            # Use eth_call to simulate the transaction
            await self.web3.eth.call(transaction, block_identifier="pending")
            self.logger.info("Transaction simulation succeeded. 📊✅")
            return True
        except Exception as e:
            self.logger.error(f"Transaction simulation failed: {e} ❌")
            return False

    async def prepare_flashloan_transaction(
        self, flashloan_asset: str, flashloan_amount: int
    ) -> Optional[Dict[str, Any]]:
        # Check if the flashloan amount is valid
        if flashloan_amount <= 0:
            self.logger.warning(
                "Flashloan amount is 0 or less, skipping flashloan transaction preparation. 📉"
            )
            return None

        try:
            flashloan_function = self.flashloan_contract.functions.fn_RequestFlashLoan(
                self.web3.to_checksum_address(flashloan_asset), flashloan_amount
            )
            self.logger.info(
                f"Preparing flashloan transaction for {flashloan_amount} of {flashloan_asset}. ⚡🏦"
            )
            return await self.build_transaction(flashloan_function)
        except ContractLogicError as e:
            self.logger.error(
                f"Contract logic error preparing flashloan transaction: {e} ❌"
            )
            return None
        except Exception as e:
            self.logger.exception(f"Error preparing flashloan transaction: {e} ❌")
            return None

    async def send_bundle(self, transactions: List[Dict[str, Any]]) -> bool:
            # Send a bundle of transactions using Flashbots
        try:
            # Sign each transaction
            signed_txs = [await self.sign_transaction(tx) for tx in transactions]
            # Prepare the bundle payload
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

            # Sign the payload using the private key
            message = encode_defunct(text=json.dumps(bundle_payload["params"][0]))
            signed_message = self.web3.eth.account.sign_message(
                message, private_key=self.account.from_key
            )
            headers = {
                "Content-Type": "application/json",
                "X-Flashbots-Signature": f"{self.account.address}:{signed_message.signature()}",
            }

            for attempt in range(1, self.retry_attempts + 1):
                try:
                    self.logger.info(f"Attempt {attempt} to send bundle. 📦💨")
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            "https://relay.flashbots.net",
                            "https://rpc.beaverbuild.io",
                            "https://rpc.flashbots.net",
                            json=bundle_payload,
                            headers=headers,
                            timeout=30,
                        ) as response:
                            response.raise_for_status()
                            response_data = await response.json()
                            if "error" in response_data:
                                self.logger.error(
                                    f"Bundle submission error: {response_data['error']} ⚠️📦"
                                )
                                raise ValueError(response_data["error"])
                            self.logger.info("Bundle sent successfully. 📦✅")
                            await self.nonce_manager.refresh_nonce()
                            return True
                except aiohttp.ClientResponseError as e:
                    self.logger.error(
                        f"Error sending bundle: {e}. Retrying... 🔄📦"
                    )
                    if attempt < self.retry_attempts:
                        sleep_time = self.retry_delay * attempt
                        self.logger.info(f"Retrying in {sleep_time} seconds...")
                        await asyncio.sleep(sleep_time)
                except ValueError as e:
                    self.logger.error(f"Bundle submission error: {e} ⚠️📦")
                    break

            self.logger.error("Failed to send bundle after multiple attempts. ⚠️📦")
            return False
        except Exception as e:
            self.logger.exception(f"Unexpected error in send_bundle: {e} ❌")
            return False

    async def front_run(self, target_tx: Dict[str, Any]) -> bool:

        tx_hash = target_tx.get("tx_hash", "Unknown")
        self.logger.info(
            f"Attempting front-run on target transaction: {tx_hash} 🏃💨📈"
        )

        decoded_tx = await self.decode_transaction_input(
            target_tx.get("input", "0x"), target_tx.get("to", "")
        )
        if not decoded_tx:
            self.logger.error(
                "Failed to decode target transaction input for front-run. ⚠️"
            )
            return False

        try:
            # Get the parameters for the front-run
            flashloan_asset = decoded_tx["params"].get("path", [])[0]
            flashloan_amount = self.calculate_flashloan_amount(target_tx)

            # Prepare the flashloan transaction
            flashloan_tx = await self.prepare_flashloan_transaction(
                flashloan_asset, flashloan_amount
            )
            if not flashloan_tx:
                self.logger.info(
                    "Failed to prepare flashloan transaction for front-run. Aborting. ⚠️"
                )
                return False

            # Prepare the front-run transaction
            front_run_tx_details = await self._prepare_front_run_transaction(target_tx)
            if not front_run_tx_details:
                self.logger.info(
                    "Failed to prepare front-run transaction. Aborting. ⚠️"
                )
                return False

            # Simulate transactions
            if not (
                await self.simulate_transaction(flashloan_tx)
                and await self.simulate_transaction(front_run_tx_details)
            ):
                self.logger.info(
                    "Simulation of front-run or flashloan failed. Aborting. ⚠️"
                )
                return False

            # Execute as a bundle
            if await self.send_bundle([flashloan_tx, front_run_tx_details]):
                self.logger.info(
                    "Front-run transaction bundle sent successfully. 🏃💨📈✅"
                )
                return True
            else:
                self.logger.error("Failed to send front-run transaction bundle. ⚠️")
                return False

        except Exception as e:
            self.logger.exception(f"Error executing front-run: {e} ⚠️")
            return False

    async def back_run(self, target_tx: Dict[str, Any]) -> bool:

        tx_hash = target_tx.get("tx_hash", "Unknown")
        self.logger.info(f"Attempting back-run on target transaction: {tx_hash} 🔙🏃📉")

        decoded_tx = await self.decode_transaction_input(
            target_tx.get("input", "0x"), target_tx.get("to", "")
        )
        if not decoded_tx:
            self.logger.error(
                "Failed to decode target transaction input for back-run. ⚠️"
            )
            return False

        try:
            # Get the parameters for the back-run
            flashloan_asset = decoded_tx["params"].get("path", [])[-1]
            flashloan_amount = self.calculate_flashloan_amount(target_tx)

            # Prepare the flashloan transaction
            flashloan_tx = await self.prepare_flashloan_transaction(
                flashloan_asset, flashloan_amount
            )
            if not flashloan_tx:
                self.logger.info(
                    "Failed to prepare flashloan transaction for back-run. Aborting. ⚠️"
                )
                return False

            # Prepare the back-run transaction
            back_run_tx_details = await self._prepare_back_run_transaction(target_tx)
            if not back_run_tx_details:
                self.logger.info(
                    "Failed to prepare back-run transaction. Aborting. ⚠️"
                )
                return False

            # Simulate transactions
            if not (
                await self.simulate_transaction(flashloan_tx)
                and await self.simulate_transaction(back_run_tx_details)
            ):
                self.logger.info(
                    "Simulation of back-run or flashloan failed. Aborting. ⚠️"
                )
                return False

            # Execute as a bundle
            if await self.send_bundle([flashloan_tx, back_run_tx_details]):
                self.logger.info(
                    "Back-run transaction bundle sent successfully. 🔙🏃📉✅"
                )
                return True
            else:
                self.logger.error("Failed to send back-run transaction bundle. ⚠️")
                return False

        except Exception as e:
            self.logger.exception(f"Error executing back-run: {e} ⚠️")
            return False

    async def execute_sandwich_attack(self, target_tx: Dict[str, Any]) -> bool:

        tx_hash = target_tx.get("tx_hash", "Unknown")
        self.logger.info(
            f"Attempting sandwich attack on target transaction: {tx_hash} 🥪🏃📈"
        )

        decoded_tx = await self.decode_transaction_input(
            target_tx.get("input", "0x"), target_tx.get("to", "")
        )
        if not decoded_tx:
            self.logger.error(
                "Failed to decode target transaction input for sandwich attack. ⚠️"
            )
            return False

        try:
            # Get the parameters for the sandwich attack
            flashloan_asset = decoded_tx["params"].get("path", [])[0]
            flashloan_amount = self.calculate_flashloan_amount(target_tx)

            # Prepare the flashloan transaction
            flashloan_tx = await self.prepare_flashloan_transaction(
                flashloan_asset, flashloan_amount
            )
            if not flashloan_tx:
                self.logger.info(
                    "Failed to prepare flashloan transaction for sandwich attack. Aborting. ⚠️"
                )
                return False

            # Prepare the front-run transaction
            front_run_tx_details = await self._prepare_front_run_transaction(target_tx)
            if not front_run_tx_details:
                self.logger.info(
                    "Failed to prepare front-run transaction for sandwich attack. Aborting. ⚠️"
                )
                return False

            # Prepare the back-run transaction
            back_run_tx_details = await self._prepare_back_run_transaction(target_tx)
            if not back_run_tx_details:
                self.logger.info(
                    "Failed to prepare back-run transaction for sandwich attack. Aborting. ⚠️"
                )
                return False

            # Simulate transactions
            if not (
                await self.simulate_transaction(flashloan_tx)
                and await self.simulate_transaction(front_run_tx_details)
                and await self.simulate_transaction(back_run_tx_details)
            ):
                self.logger.info(
                    "Simulation of one or more transactions failed during sandwich attack. Aborting. ⚠️"
                )
                return False

            # Execute all three transactions as a bundle
            if await self.send_bundle(
                [flashloan_tx, front_run_tx_details, back_run_tx_details]
            ):
                self.logger.info(
                    "Sandwich attack transaction bundle sent successfully. 🥪🏃📈✅"
                )
                return True
            else:
                self.logger.error(
                    "Failed to send sandwich attack transaction bundle. ⚠️"
                )
                return False

        except Exception as e:
            self.logger.exception(f"Error executing sandwich attack: {e} ❌")
            return False

    async def _prepare_front_run_transaction(
        self, target_tx: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:

        decoded_tx = await self.decode_transaction_input(
            target_tx.get("input", "0x"), target_tx.get("to", "")
        )
        if not decoded_tx:
            self.logger.error(
                "Failed to decode target transaction input for front-run preparation. ⚠️"
            )
            return None

        function_name = decoded_tx.get("function_name")
        function_params = decoded_tx.get("params", {})

        try: 
            # Determine which router to use based on the target address
            to_address = self.web3.to_checksum_address(target_tx.get("to", ""))
            if to_address == self.config.UNISWAP_V2_ROUTER_ADDRESS:
                router_contract = self.uniswap_router_contract
                exchange_name = "Uniswap"
            elif to_address == self.config.SUSHISWAP_ROUTER_ADDRESS:
                router_contract = self.sushiswap_router_contract
                exchange_name = "Sushiswap"
            elif to_address == self.config.PANCAKESWAP_ROUTER_ADDRESS:
                router_contract = self.pancakeswap_router_contract
                exchange_name = "Pancakeswap"
            elif to_address == self.config.BALANCER_ROUTER_ADDRESS:
                router_contract = self.balancer_router_contract
                exchange_name = "Balancer"
            
            else:
                self.logger.error("Unknown router address. Cannot determine exchange. ❌")
                return None

            # Get the function object by name
            front_run_function = getattr(router_contract.functions, function_name)(
                **function_params
            )
            # Build the transaction
            front_run_tx = await self.build_transaction(front_run_function)
            self.logger.info(
                f"Prepared front-run transaction on {exchange_name} successfully. ⚔️🏃"
            )
            return front_run_tx
        except AttributeError:
            self.logger.error(
                f"Function {function_name} not found in {exchange_name} router ABI. ❌"
            )
            return None
        except Exception as e:
            self.logger.exception(f"Error preparing front-run transaction: {e} ❌")
            return None
        

    async def _prepare_back_run_transaction(
        self, target_tx: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:

        decoded_tx = await self.decode_transaction_input(
            target_tx.get("input", "0x"), target_tx.get("to", "")
        )
        if not decoded_tx:
            self.logger.error(
                "Failed to decode target transaction input for back-run preparation. ⚠️"
            )
            return None

        function_name = decoded_tx.get("function_name")
        function_params = decoded_tx.get("params", {})

        # Reverse the path parameter for back-run
        path = function_params.get("path", [])
        if path:
            function_params["path"] = path[::-1]
        else:
            self.logger.warning(
                "Transaction has no path parameter for back-run preparation. ❗"
            )

        try:
            # Determine which router to use based on the target address
            to_address = self.web3.to_checksum_address(target_tx.get("to", ""))
            if to_address == self.config.UNISWAP_V2_ROUTER_ADDRESS:
                router_contract = self.uniswap_router_contract
                exchange_name = "Uniswap"
            elif to_address == self.config.SUSHISWAP_ROUTER_ADDRESS:
                router_contract = self.sushiswap_router_contract
                exchange_name = "Sushiswap"
            elif to_address == self.config.PANCAKESWAP_ROUTER_ADDRESS:
                router_contract = self.pancakeswap_router_contract
                exchange_name = "Pancakeswap"
            elif to_address == self.config.BALANCER_ROUTER_ADDRESS:
                router_contract = self.balancer_router_contract
                exchange_name = "Balancer"
            # if normal eth transaction
            elif to_address == ("0x0000000000000000000000000000000000000000"):
                router_contract = self.web3.eth.contract(address=to_address, abi=self.erc20_ABI)
                exchange_name = "ETH"
            else:
                self.logger.error("Unknown router address. Cannot determine exchange. ❌")
                return None

            # Get the function object by name
            back_run_function = getattr(router_contract.functions, function_name)(
                **function_params
            )
            # Build the transaction
            back_run_tx = await self.build_transaction(back_run_function)
            self.logger.info(
                f"Prepared back-run transaction on {exchange_name} successfully. 🔙🏃"
            )
            return back_run_tx
        except AttributeError:
            self.logger.error(
                f"Function {function_name} not found in {exchange_name} router ABI. ❌"
            )
            return None
        except Exception as e:
            self.logger.exception(f"Error preparing back-run transaction: {e} ❌")
            return None

    async def decode_transaction_input(
        self, input_data: str, to_address: str
    ) -> Optional[Dict[str, Any]]:

        try:
            to_address = self.web3.to_checksum_address(to_address)
            if to_address == self.config.UNISWAP_V2_ROUTER_ADDRESS:
                abi = self.config.UNISWAP_V2_ROUTER_ABI
                exchange_name = "Uniswap"
            elif to_address == self.config.SUSHISWAP_ROUTER_ADDRESS:
                abi = self.config.SUSHISWAP_ROUTER_ABI
                exchange_name = "Sushiswap"
            elif to_address == self.config.PANCAKESWAP_ROUTER_ADDRESS:
                abi = self.config.PANCAKESWAP_ROUTER_ABI
                exchange_name = "Pancakeswap"
            else:
                self.logger.error(
                    "Unknown router address. Cannot determine ABI for decoding. ❌"
                )
                return None

            contract = self.web3.eth.contract(address=to_address, abi=abi)
            function_obj, function_params = contract.decode_function_input(input_data)
            decoded_data = {
                "function_name": function_obj.function_identifier,
                "params": function_params,
            }
            self.logger.debug(
                f"Decoded transaction input using {exchange_name} ABI: {decoded_data}"
            )
            return decoded_data
        except Exception as e:
            self.logger.error(f"Error decoding transaction input: {e} ❌")
            return None

    async def cancel_transaction(self, nonce: NonceManager) -> bool:

        cancel_tx = {
            "data": "0x",
            "chainId": await self.web3.eth.chain_id,
            "nonce": nonce,
            "to": self.account.address,
            "value": 0,
            "gas": 21_000,
            "gasPrice": self.web3.to_wei("150", "gwei"),  # Higher than the stuck transaction
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
            self.logger.info(
                f"Cancellation transaction sent successfully: {tx_hash_hex} 🚀✅"
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to cancel transaction: {e} ❌")
            return False

    async def estimate_gas_limit(self, tx: Dict[str, Any]) -> int:

        try:
            gas_estimate = await self.web3.eth.estimate_gas(tx)
            self.logger.debug(f"Estimated gas: {gas_estimate} ⛽")
            return gas_estimate
        except Exception as e:
            self.logger.warning(
                f"Gas estimation failed: {e}. Using default gas limit of 100000 ⛽⚠️"
            )
            return 100_000  # Default gas limit

    async def get_current_profit(self) -> Decimal:

        try:
            current_profit = await self.safety_net.get_current_profit()
            self.current_profit = Decimal(current_profit)
            self.logger.info(f"Current profit: {self.current_profit} ETH 💰")
            return self.current_profit
        except Exception as e:
            self.logger.error(f"Error fetching current profit: {e} ❌")
            return Decimal("0")

    async def withdraw_eth(self) -> bool:

        try:
            withdraw_function = self.flashloan_contract.functions.withdrawETH()
            tx = await self.build_transaction(withdraw_function)
            tx_hash = await self.execute_transaction(tx)
            if tx_hash:
                self.logger.info(
                    f"ETH withdrawal transaction sent with hash: {tx_hash} ✅"
                )
                return True
            else:
                self.logger.error("Failed to send ETH withdrawal transaction. ❌")
                return False
        except Exception as e:
            self.logger.exception(f"Error withdrawing ETH: {e} ❌")
            return False

    async def withdraw_token(self, token_address: str) -> bool:

        try:
            withdraw_function = self.flashloan_contract.functions.withdrawToken(
                self.web3.to_checksum_address(token_address)
            )
            tx = await self.build_transaction(withdraw_function)
            tx_hash = await self.execute_transaction(tx)
            if tx_hash:
                self.logger.info(
                    f"Token withdrawal transaction sent with hash: {tx_hash} ✅"
                )
                return True
            else:
                self.logger.error("Failed to send token withdrawal transaction. ❌")
                return False
        except Exception as e:
            self.logger.exception(f"Error withdrawing token: {e} ❌")
            return False



class StrategyManager:
    """
    Manages and executes various trading strategies such as ETH transactions, front-running,
    back-running, and sandwich attacks. It tracks strategy performance, predicts market movements,
    and selects the best strategy based on historical performance and reinforcement learning.
    """

    def __init__(
        self,
        transaction_array: TransactionArray,
        market_analyzer: "MarketAnalyzer",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.transaction_array = transaction_array
        self.market_analyzer = market_analyzer
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.logger.info("StrategyManager initialized successfully. ✅")

        # Track strategy performance and profitability
        self.strategy_performance: Dict[str, Dict[str, Any]] = {
            "eth_transaction": {"successes": 0, "failures": 0, "profit": Decimal("0")},
            "front_run": {"successes": 0, "failures": 0, "profit": Decimal("0")},
            "back_run": {"successes": 0, "failures": 0, "profit": Decimal("0")},
            "sandwich_attack": {"successes": 0, "failures": 0, "profit": Decimal("0")},
        }

        # Maintain historical data to identify trends and optimize strategy
        self.history_data: List[Dict[str, Any]] = []
        self.price_model = LinearRegression()

        # Reinforcement weights with decaying factors for continuous learning
        self.reinforcement_weights: Dict[str, np.ndarray] = {
            "eth_transaction": np.ones(1),
            "front_run": np.ones(4),
            "back_run": np.ones(4),
            "sandwich_attack": np.ones(4),
        }
        self.decay_factor: float = 0.9  # Decay factor for past performances
        self.min_profit_threshold: Decimal = Decimal(
            "0.01"
        )  # Minimum profit margin in ETH

    async def execute_best_strategy(
        self, target_tx: Dict[str, Any], strategy_type: str
    ) -> bool:

        strategies = self.get_strategies(strategy_type)
        if not strategies:
            self.logger.warning(f"No strategies available for type: {strategy_type} ❗")
            return False

        selected_strategy = self._select_strategy(strategies, strategy_type)
        self.logger.info(f"Executing strategy: {selected_strategy.__name__} ⚔️🏃")

        try:
            profit_before = await self.transaction_array.get_current_profit()  # Track profit before execution
            success = await selected_strategy(target_tx)
            profit_after = await self.transaction_array.get_current_profit()  # Profit after execution

            # Calculate profit/loss from the strategy execution
            profit_made = Decimal(profit_after) - Decimal(profit_before)
            await self.update_history(
                selected_strategy.__name__, success, strategy_type, profit_made
            )

            return success
        except Exception as e:
            self.logger.error(
                f"Error executing strategy {selected_strategy.__name__}: {e} ❌"
            )
            return False

    def get_strategies(self, strategy_type: str) -> List[Any]:

        strategies = {
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

        if strategy_type not in strategies:
            self.logger.error(f"Invalid strategy type provided: {strategy_type} ❌")
            return []
        return strategies[strategy_type]

    def _select_strategy(self, strategies: List[Any], strategy_type: str) -> Any:

        weights = self.reinforcement_weights[strategy_type]
        # Apply decay to focus on recent performance
        weights *= self.decay_factor
        # Ensure weights are not below a threshold to maintain exploration
        weights = np.maximum(weights, 0.1)
        strategy_indices = np.arange(len(strategies))
        try:
            selected_strategy_index = np.random.choice(
                strategy_indices, p=weights / weights.sum()
            )
            self.logger.debug(
                f"Selected strategy index {selected_strategy_index} for type {strategy_type}."
            )
            return strategies[selected_strategy_index]
        except ValueError as e:
            self.logger.error(
                f"Error selecting strategy: {e}. Falling back to random selection."
            )
            return np.random.choice(strategies)

    async def update_history(
        self, strategy_name: str, success: bool, strategy_type: str, profit: Decimal
    ) -> None:

        self.logger.info(
            f"Updating history for strategy: {strategy_name}, Success: {success}, Profit: {profit} ✅"
        )

        # Update performance metrics
        if success:
            self.strategy_performance[strategy_type]["successes"] += 1
            self.strategy_performance[strategy_type]["profit"] += profit
        else:
            self.strategy_performance[strategy_type]["failures"] += 1
            self.strategy_performance[strategy_type][
                "profit"
            ] += profit  # Note: profit may be negative

        # Append to history data
        self.history_data.append(
            {
                "strategy_name": strategy_name,
                "success": success,
                "profit": profit,
                "strategy_type": strategy_type,
                "total_profit": self.strategy_performance[strategy_type]["profit"],
            }
        )

        # Update reinforcement weights based on profit and success
        strategy_index = self.get_strategy_index(strategy_name, strategy_type)
        if strategy_index >= 0:
            reward_factor = (
                float(profit) if profit > Decimal("0") else -1
            )  # Reward based on profit
            self.reinforcement_weights[strategy_type][strategy_index] += reward_factor
            # Ensure weights remain positive
            self.reinforcement_weights[strategy_type][strategy_index] = max(
                self.reinforcement_weights[strategy_type][strategy_index], 0.1
            )
            self.logger.debug(
                f"Updated reinforcement weight for {strategy_name}: {self.reinforcement_weights[strategy_type][strategy_index]}"
            )

    def get_strategy_index(self, strategy_name: str, strategy_type: str) -> int:

        strategy_mapping = {
            "eth_transaction": {
                "high_value_eth_transfer": 0,
            },
            "front_run": {
                "aggressive_front_run": 0,
                "predictive_front_run": 1,
                "volatility_front_run": 2,
                "advanced_front_run": 3,
            },
            "back_run": {
                "price_dip_back_run": 0,
                "flashloan_back_run": 1,
                "high_volume_back_run": 2,
                "advanced_back_run": 3,
            },
            "sandwich_attack": {
                "flash_profit_sandwich": 0,
                "price_boost_sandwich": 1,
                "arbitrage_sandwich": 2,
                "advanced_sandwich_attack": 3,
            },
        }
        return strategy_mapping.get(strategy_type, {}).get(strategy_name, -1)

    async def predict_price_movement(self, token_symbol: str) -> float:

        self.logger.info(f"Predicting price movement for {token_symbol} 🔮")
        try:
            prices = await self.market_analyzer.fetch_historical_prices(token_symbol)
            if not prices:
                self.logger.warning(
                    f"No historical prices available for {token_symbol}. Cannot predict movement."
                )
                return 0.0
            X = np.arange(len(prices)).reshape(-1, 1)
            y = np.array(prices)
            self.price_model.fit(X, y)
            next_time = np.array([[len(prices)]])
            predicted_price = self.price_model.predict(next_time)[0]
            self.logger.info(
                f"Predicted price for {token_symbol}: {predicted_price} ETH 📈"
            )
            return float(predicted_price)
        except NotFittedError:
            self.logger.error(
                "Price model is not fitted yet. Cannot predict price movement. ❌"
            )
            return 0.0
        except Exception as e:
            self.logger.exception(
                f"Error predicting price movement for {token_symbol}: {e} ❌"
            )
            return 0.0

    # -------------------- Strategy Methods --------------------

    async def high_value_eth_transfer(self, target_tx: Dict[str, Any]) -> bool:

        self.logger.info("Initiating High-Value ETH Transfer Strategy... 🏃💨")
        try:
            # Check if it's a high-value ETH transfer
            eth_value_in_wei = target_tx.get("value", 0)
            if eth_value_in_wei > self.transaction_array.web3.to_wei(10, "ether"):
                eth_value_in_eth = self.transaction_array.web3.from_wei(
                    eth_value_in_wei, "ether"
                )
                self.logger.info(
                    f"High-value ETH transfer detected: {eth_value_in_eth} ETH 🏃"
                )
                # Proceed with handling the ETH transaction (e.g., front-running)
                return await self.transaction_array.handle_eth_transaction(target_tx)
            self.logger.info(
                "ETH transaction does not meet the high-value criteria. Skipping... ⚠️"
            )
            return False
        except Exception as e:
            self.logger.error(
                f"Error executing High-Value ETH Transfer Strategy: {e} ❌"
            )
            return False

    async def aggressive_front_run(self, target_tx: Dict[str, Any]) -> bool:

        self.logger.info("Initiating Aggressive Front-Run Strategy... 🏃")
        try:
            if target_tx.get("value", 0) > self.transaction_array.web3.to_wei(
                1, "ether"
            ):
                self.logger.info(
                    "Transaction value above threshold, proceeding with aggressive front-run."
                )
                return await self.transaction_array.front_run(target_tx)
            self.logger.info(
                "Transaction below threshold. Skipping aggressive front-run."
            )
            return False
        except Exception as e:
            self.logger.error(f"Error executing Aggressive Front-Run Strategy: {e} ❌")
            return False

    async def predictive_front_run(self, target_tx: Dict[str, Any]) -> bool:

        self.logger.info("Initiating Predictive Front-Run Strategy... 🏃")
        try:
            decoded_tx = await self.transaction_array.decode_transaction_input(
                target_tx["input"], target_tx["to"]
            )
            if not decoded_tx:
                self.logger.warning(
                    "Failed to decode transaction input for Predictive Front-Run Strategy. ❗"
                )
                return False
            params = decoded_tx.get("params", {})
            path = params.get("path", [])
            if not path:
                self.logger.warning(
                    "Transaction has no path parameter for Predictive Front-Run Strategy. ❗"
                )
                return False
            token_address = path[0]
            token_symbol = await self.market_analyzer.get_token_symbol(token_address)
            if not token_symbol:
                self.logger.warning(
                    f"Token symbol not found for address {token_address} in Predictive Front-Run Strategy. ❗"
                )
                return False
            predicted_price = await self.predict_price_movement(token_symbol)
            current_price = await self.market_analyzer.get_current_price(token_symbol)
            if current_price is None:
                self.logger.warning(
                    f"Current price not available for {token_symbol} in Predictive Front-Run Strategy. ❗"
                )
                return False
            if predicted_price > float(current_price) * 1.01:  # 1% profit margin
                self.logger.info(
                    "Predicted price increase exceeds threshold, proceeding with predictive front-run."
                )
                return await self.transaction_array.front_run(target_tx)
            self.logger.info(
                "Predicted price increase does not meet threshold. Skipping predictive front-run."
            )
            return False
        except Exception as e:
            self.logger.error(f"Error executing Predictive Front-Run Strategy: {e} ❌")
            return False

    async def volatility_front_run(self, target_tx: Dict[str, Any]) -> bool:

        self.logger.info("Initiating Volatility Front-Run Strategy... 🏃")
        try:
            market_conditions = await self.market_analyzer.check_market_conditions(
                target_tx["to"]
            )
            if market_conditions.get("high_volatility", False):
                self.logger.info(
                    "High volatility detected, proceeding with volatility front-run."
                )
                return await self.transaction_array.front_run(target_tx)
            self.logger.info(
                "Market volatility not high enough. Skipping volatility front-run."
            )
            return False
        except Exception as e:
            self.logger.error(f"Error executing Volatility Front-Run Strategy: {e} ❌")
            return False

    async def advanced_front_run(self, target_tx: Dict[str, Any]) -> bool:

        self.logger.info("Initiating Advanced Front-Run Strategy... 🏃💨")
        try:
            decoded_tx = await self.transaction_array.decode_transaction_input(
                target_tx["input"], target_tx["to"]
            )
            if not decoded_tx:
                self.logger.warning(
                    "Failed to decode transaction input for Advanced Front-Run Strategy. ❗"
                )
                return False
            params = decoded_tx.get("params", {})
            path = params.get("path", [])
            if not path:
                self.logger.warning(
                    "Transaction has no path parameter for Advanced Front-Run Strategy. ❗"
                )
                return False
            token_symbol = await self.market_analyzer.get_token_symbol(path[0])
            if not token_symbol:
                self.logger.warning(
                    f"Token symbol not found for address {path[0]} in Advanced Front-Run Strategy. ❗"
                )
                return False
            predicted_price = await self.predict_price_movement(token_symbol)
            market_conditions = await self.market_analyzer.check_market_conditions(
                target_tx["to"]
            )
            current_price = await self.market_analyzer.get_current_price(token_symbol)
            if current_price is None:
                self.logger.warning(
                    f"Current price not available for {token_symbol} in Advanced Front-Run Strategy. ❗"
                )
                return False
            if (
                predicted_price > float(current_price) * 1.02
            ) and market_conditions.get("bullish_trend", False):
                self.logger.info(
                    "Favorable price and bullish trend detected, proceeding with advanced front-run."
                )
                return await self.transaction_array.front_run(target_tx)
            self.logger.info(
                "Conditions not favorable for advanced front-run. Skipping."
            )
            return False
        except Exception as e:
            self.logger.error(f"Error executing Advanced Front-Run Strategy: {e} ❌")
            return False

    async def price_dip_back_run(self, target_tx: Dict[str, Any]) -> bool:

        self.logger.info("Initiating Price Dip Back-Run Strategy... 🔙🏃")
        try:
            decoded_tx = await self.transaction_array.decode_transaction_input(
                target_tx["input"], target_tx["to"]
            )
            if not decoded_tx:
                self.logger.warning(
                    "Failed to decode transaction input for Price Dip Back-Run Strategy. ❗"
                )
                return False
            params = decoded_tx.get("params", {})
            path = params.get("path", [])
            if not path:
                self.logger.warning(
                    "Transaction has no path parameter for Price Dip Back-Run Strategy. ❗"
                )
                return False
            token_address = path[-1]
            token_symbol = await self.market_analyzer.get_token_symbol(token_address)
            if not token_symbol:
                self.logger.warning(
                    f"Token symbol not found for address {token_address} in Price Dip Back-Run Strategy. ❗"
                )
                return False
            current_price = await self.market_analyzer.get_current_price(token_symbol)
            if current_price is None:
                self.logger.warning(
                    f"Current price not available for {token_symbol} in Price Dip Back-Run Strategy. ❗"
                )
                return False
            predicted_price = await self.predict_price_movement(token_symbol)
            if predicted_price < float(current_price) * 0.99:  # 1% profit margin
                self.logger.info(
                    "Predicted price decrease exceeds threshold, proceeding with price dip back-run."
                )
                return await self.transaction_array.back_run(target_tx)
            self.logger.info(
                "Predicted price decrease does not meet threshold. Skipping price dip back-run."
            )
            return False
        except Exception as e:
            self.logger.error(f"Error executing Price Dip Back-Run Strategy: {e} ❌")
            return False

    async def flashloan_back_run(self, target_tx: Dict[str, Any]) -> bool:

        self.logger.info("Initiating Flashloan Back-Run Strategy... 🔙🏃")
        try:
            estimated_profit = self.transaction_array.calculate_flashloan_amount(
                target_tx
            ) * Decimal(
                "0.02"
            )  # Assume 2% profit margin
            if estimated_profit > self.min_profit_threshold:
                self.logger.info(
                    "Estimated profit meets threshold, proceeding with flashloan back-run."
                )
                return await self.transaction_array.back_run(target_tx)
            self.logger.info("Profit is insufficient for flashloan back-run. Skipping.")
            return False
        except Exception as e:
            self.logger.error(f"Error executing Flashloan Back-Run Strategy: {e} ❌")
            return False

    async def high_volume_back_run(self, target_tx: Dict[str, Any]) -> bool:

        self.logger.info("Initiating High Volume Back-Run Strategy... 🔙🏃")
        try:
            token_volume = await self.market_analyzer.get_token_volume(target_tx["to"])
            if token_volume > 1_000_000:  # Check if volume is high
                self.logger.info(
                    "High volume detected, proceeding with high volume back-run."
                )
                return await self.transaction_array.back_run(target_tx)
            self.logger.info("Volume not favorable for high volume back-run. Skipping.")
            return False
        except Exception as e:
            self.logger.error(f"Error executing High Volume Back-Run Strategy: {e} ❌")
            return False

    async def advanced_back_run(self, target_tx: Dict[str, Any]) -> bool:

        self.logger.info("Initiating Advanced Back-Run Strategy... 🔙🏃💨")
        try:
            decoded_tx = await self.transaction_array.decode_transaction_input(
                target_tx["input"], target_tx["to"]
            )
            if not decoded_tx:
                self.logger.warning(
                    "Failed to decode transaction input for Advanced Back-Run Strategy. ❗"
                )
                return False
            params = decoded_tx.get("params", {})
            path = params.get("path", [])
            if not path:
                self.logger.warning(
                    "Transaction has no path parameter for Advanced Back-Run Strategy. ❗"
                )
                return False
            token_address = path[-1]
            token_symbol = await self.market_analyzer.get_token_symbol(token_address)
            if not token_symbol:
                self.logger.warning(
                    f"Token symbol not found for address {token_address} in Advanced Back-Run Strategy. ❗"
                )
                return False
            current_price = await self.market_analyzer.get_current_price(token_symbol)
            if current_price is None:
                self.logger.warning(
                    f"Current price not available for {token_symbol} in Advanced Back-Run Strategy. ❗"
                )
                return False
            predicted_price = await self.predict_price_movement(token_symbol)
            market_conditions = await self.market_analyzer.check_market_conditions(
                target_tx["to"]
            )
            if (
                predicted_price < float(current_price) * 0.98
            ) and market_conditions.get("bearish_trend", False):
                self.logger.info(
                    "Favorable price and bearish trend detected, proceeding with advanced back-run."
                )
                return await self.transaction_array.back_run(target_tx)
            self.logger.info(
                "Conditions not favorable for advanced back-run. Skipping."
            )
            return False
        except Exception as e:
            self.logger.error(f"Error executing Advanced Back-Run Strategy: {e} ❌")
            return False

    async def flash_profit_sandwich(self, target_tx: Dict[str, Any]) -> bool:

        self.logger.info("Initiating Flash Profit Sandwich Strategy... 🥪🏃")
        try:
            potential_profit = self.transaction_array.calculate_flashloan_amount(
                target_tx
            )
            if potential_profit > self.min_profit_threshold:
                self.logger.info(
                    "Potential profit meets threshold, proceeding with flash profit sandwich attack."
                )
                return await self.transaction_array.execute_sandwich_attack(target_tx)
            self.logger.info(
                "Conditions not met for flash profit sandwich attack. Skipping."
            )
            return False
        except Exception as e:
            self.logger.error(f"Error executing Flash Profit Sandwich Strategy: {e} ❌")
            return False

    async def price_boost_sandwich(self, target_tx: Dict[str, Any]) -> bool:

        self.logger.info("Initiating Price Boost Sandwich Strategy... 🥪🏃")
        try:
            token_symbol = await self.market_analyzer.get_token_symbol(target_tx["to"])
            current_price = await self.market_analyzer.get_current_price(token_symbol)
            if current_price is None:
                self.logger.warning(
                    f"Current price not available for {token_symbol} in Price Boost Sandwich Strategy. ❗"
                )
                return False
            predicted_price = await self.predict_price_movement(token_symbol)
            if predicted_price > float(current_price) * 1.02:  # 2% profit margin
                self.logger.info(
                    "Favorable price detected, proceeding with price boost sandwich attack."
                )
                return await self.transaction_array.execute_sandwich_attack(target_tx)
            self.logger.info(
                "Price conditions not favorable for price boost sandwich attack. Skipping."
            )
            return False
        except Exception as e:
            self.logger.error(f"Error executing Price Boost Sandwich Strategy: {e} ❌")
            return False

    async def arbitrage_sandwich(self, target_tx: Dict[str, Any]) -> bool:

        self.logger.info("Initiating Arbitrage Sandwich Strategy... 🥪🏃")
        try:
            if await self.market_analyzer.is_arbitrage_opportunity(target_tx):
                self.logger.info(
                    "Arbitrage opportunity detected, proceeding with arbitrage sandwich attack."
                )
                return await self.transaction_array.execute_sandwich_attack(target_tx)
            self.logger.info(
                "No arbitrage opportunity detected. Skipping arbitrage sandwich attack."
            )
            return False
        except Exception as e:
            self.logger.error(f"Error executing Arbitrage Sandwich Strategy: {e} ❌")
            return False

    async def advanced_sandwich_attack(self, target_tx: Dict[str, Any]) -> bool:

        self.logger.info("Initiating Advanced Sandwich Attack Strategy... 🥪🏃💨")
        try:
            potential_profit = self.transaction_array.calculate_flashloan_amount(
                target_tx
            )
            market_conditions = await self.market_analyzer.check_market_conditions(
                target_tx["to"]
            )
            if (potential_profit > Decimal("0.02")) and market_conditions.get(
                "high_volatility", False
            ):
                self.logger.info(
                    "Conditions favorable for advanced sandwich attack, executing."
                )
                return await self.transaction_array.execute_sandwich_attack(target_tx)
            self.logger.info(
                "Conditions not favorable for advanced sandwich attack. Skipping."
            )
            return False
        except Exception as e:
            self.logger.error(
                f"Error executing Advanced Sandwich Attack Strategy: {e} ❌"
            )
            return False

    async def _determine_strategy_type(self, target_tx: Dict[str, Any]) -> Optional[str]:

        try:
            # Check for high-value ETH transactions
            if target_tx.get("value", 0) > self.transaction_array.web3.to_wei(
                10, "ether"
            ):
                return "eth_transaction"

            # Analyze market conditions
            market_conditions = await self.market_analyzer.check_market_conditions(
                target_tx["to"]
            )
            self.logger.debug(
                f"Market conditions for {target_tx['to']}: {market_conditions}"
            )

            # Determine strategy based on market conditions and transaction details
            if market_conditions.get("high_volatility", False):
                return "sandwich_attack"
            elif target_tx.get("value", 0) > self.transaction_array.web3.to_wei(
                1, "ether"
            ):
                return "front_run"
            elif await self.market_analyzer.is_arbitrage_opportunity(target_tx):
                return "back_run"
            else:
                self.logger.debug(
                    "No suitable strategy type determined for the transaction."
                )
                return None
        except Exception as e:
            self.logger.error(
                f"Failed to determine strategy type for transaction {target_tx.get('tx_hash', '')}: {e} ❌"
            )
            return None

    async def execute_strategy_for_transaction(self, target_tx: Dict[str, Any]) -> bool:

        strategy_type = await self._determine_strategy_type(target_tx)
        if strategy_type:
            success = await self.execute_best_strategy(target_tx, strategy_type)
            tx_hash = target_tx.get("tx_hash", "Unknown")
            if success:
                self.logger.info(
                    f"Successfully executed {strategy_type} strategy for transaction {tx_hash}. ✅"
                )
            else:
                self.logger.warning(
                    f"Failed to execute {strategy_type} strategy for transaction {tx_hash}. ⚠️"
                )
            return success
        self.logger.debug(
            f"No suitable strategy found for transaction {target_tx.get('tx_hash', '')}."
        )
        return False

class MarketAnalyzer:

    def __init__(
        self,
        web3: AsyncWeb3,
        erc20_ABI: List[Dict[str, Any]],
        config: Config,
        logger: Optional[logging.Logger] = None,
    ):
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.web3 = web3
        self.erc20_ABI = erc20_ABI
        self.config = config
        self.price_cache = TTLCache(maxsize=1000, ttl=300)  # Cache for 5 minutes
        self.volume_cache = TTLCache(maxsize=1000, ttl=300)  # Cache for 5 minutes
        self.token_symbols = self.config.TOKEN_SYMBOLS
        self.symbol_mapping = self._load_token_symbols()
        self.token_symbol_cache = {}
        self.cache_duration = 60 * 5  # Cache duration in seconds (5 minutes)

        # Fallback API keys and services
        self.api_keys = {
            "BINANCE": None,  # Binance Public API does not require an API key
            "COINGECKO": self.config.COINGECKO_API_KEY,
            "COINMARKETCAP": self.config.COINMARKETCAP_API_KEY,
            "CRYPTOCOMPARE": self.config.CRYPTOCOMPARE_API_KEY,
        }

    def _load_token_symbols(self) -> dict:

        try:
            if not self.token_symbols:
                self.logger.error("TOKEN_SYMBOLS path not set in configuration. ❌")
                return {}
            with open(self.token_symbols, "r") as file:
                return json.load(file)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.logger.error(f"Error loading token symbols: {e} ❌")
            return {}

    async def check_market_conditions(self, token_address: str) -> Dict[str, Any]:

        market_conditions = {
            "high_volatility": False,
            "bullish_trend": False,
            "bearish_trend": False,
            "low_liquidity": False,
        }

        token_symbol = await self.get_token_symbol(token_address)
        if not token_symbol:
            self.logger.error(f"Cannot get token symbol for address {token_address} ❌")
            return market_conditions

        # Fetch recent price data (e.g., last 1 day)
        prices = await self.fetch_historical_prices(token_symbol, days=1)
        if len(prices) < 2:
            self.logger.error(
                f"Not enough price data to analyze market conditions for {token_symbol} 📊"
            )
            return market_conditions

        # Calculate volatility
        prices_array = np.array(prices)
        returns = np.diff(prices_array) / prices_array[:-1]
        volatility = np.std(returns)
        self.logger.debug(f"Calculated volatility for {token_symbol}: {volatility} 📊")

        # Define thresholds
        VOLATILITY_THRESHOLD = 0.05  # 5% standard deviation
        LIQUIDITY_THRESHOLD = 100000  # $100,000 in 24h volume

        if volatility > VOLATILITY_THRESHOLD:
            market_conditions["high_volatility"] = True

        # Calculate trend
        moving_average = np.mean(prices_array)
        if prices_array[-1] > moving_average:
            market_conditions["bullish_trend"] = True
        elif prices_array[-1] < moving_average:
            market_conditions["bearish_trend"] = True

        # Check liquidity
        volume = await self.get_token_volume(token_symbol)
        if volume < LIQUIDITY_THRESHOLD:
            market_conditions["low_liquidity"] = True

        return market_conditions

    async def get_token_symbol(self, token_address: str) -> Optional[str]:
        if token_address in self.token_symbol_cache:
            return self.token_symbol_cache[token_address]
        elif token_address in self.config.TOKEN_SYMBOLS:
            return self.config.TOKEN_SYMBOLS[token_address]
        try:
            # Create contract instance
            contract = self.web3.eth.contract(
                address=token_address, abi=self.erc20_ABI
            )
            symbol = await contract.functions.symbol().call()
            self.token_symbol_cache[token_address] = symbol  # Cache the result
            return symbol
        except Exception as e:
            self.logger.error(
                f"We do not have the token symbol for address {token_address}: {e}"
            )
            return None
        
        

    async def decode_transaction_input(
        self, input_data: str, contract_address: str
    ) -> Optional[Dict[str, Any]]:
        """ Decodes the input data of a transaction. """
        try:
            contract = self.web3.eth.contract(
                address=contract_address, abi=self.erc20_ABI
            )
            function_ABI, params = contract.decode_function_input(input_data)
            return {"function_name": function_ABI["name"], "params": params}
        except Exception as e:
            self.logger.error(f"Failed in decoding transaction input: {e} ❌")
            return None

    async def is_arbitrage_opportunity(self, target_tx: Dict[str, Any]) -> bool:
        """ Checks for arbitrage opportunities in a transaction. """
        try:
            # Decode transaction input
            decoded_tx = await self.decode_transaction_input(
                target_tx["input"], target_tx["to"]
            )
            if not decoded_tx:
                return False

            function_params = decoded_tx["params"]
            path = function_params.get("path", [])
            if len(path) < 2:
                return False

            token_address = path[-1]  # The token being bought
            token_symbol = await self.get_token_symbol(token_address)

            if not token_symbol:
                return False

            # Get prices from different services
            price_binance = await self.get_current_price(token_symbol)
            price_coingecko = await self.get_current_price(token_symbol)

            if price_binance is None or price_coingecko is None:
                return False

            # Check for arbitrage opportunity
            price_difference = abs(price_binance - price_coingecko)
            average_price = (price_binance + price_coingecko) / 2
            if average_price == 0:
                return False

            price_difference_percentage = price_difference / average_price
            if price_difference_percentage > 0.01:
                self.logger.debug(
                    f"Arbitrage opportunity detected for {token_symbol} 📈"
                )
                return True
            else:
                return False
        except Exception as e:
            self.logger.error(f"Failed in checking arbitrage opportunity: {e} ❌")
            return False

    async def fetch_historical_prices(self, token_id: str, days: int = 30) -> List[float]:
        cache_key = f"{token_id}_{days}"
        if cache_key in self.price_cache:
            self.logger.debug(
                f"Returning cached historical prices for {token_id}. 📊⏳"
            )
            return self.price_cache[cache_key]

        for service in self.api_keys.keys():
            try:
                self.logger.debug(
                    f"Fetching historical prices for {token_id} using {service}... 📊⏳"
                )
                headers = {}
                if service == "BINANCE":
                    symbol = self._convert_token_id_to_binance_symbol(token_id)
                    if not symbol:
                        continue
                    url = f"https://api.binance.com/api/v3/klines"
                    params = {"symbol": symbol, "interval": "1d", "limit": int(days)}
                elif service == "COINGECKO":
                    url = f"https://api.coingecko.com/api/v3/coins/{token_id}/market_chart"
                    params = {"vs_currency": "usd", "days": days}
                elif service == "COINMARKETCAP":
                    url = f"https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/historical"
                    params = {"symbol": token_id, "time_period": f"{days}d"}
                    headers = {"X-CMC_PRO_API_KEY": self.api_keys[service]}
                elif service == "CRYPTOCOMPARE":
                    url = f"https://min-api.cryptocompare.com/data/v2/histoday"
                    params = {"fsym": token_id, "tsym": "USD", "limit": int(days)}
                    headers = {"Apikey": self.api_keys[service]}
                else:
                    continue

                response = await self.make_request(url, params=params, headers=headers)
                data = await response.json()

                if service == "BINANCE":
                    prices = [float(entry[4]) for entry in data]  # Close prices
                elif service == "COINGECKO":
                    prices = [price[1] for price in data["prices"]]
                elif service == "COINMARKETCAP":
                    prices = [quote["close"] for quote in data["data"]["quotes"]]
                elif service == "CRYPTOCOMPARE":
                    prices = [day["close"] for day in data["Data"]["Data"]]
                else:
                    continue

                self.price_cache[cache_key] = prices
                self.logger.debug(
                    f"Fetched historical prices for {token_id} using {service} successfully. 📊"
                )
                return prices

            except Exception as e:
                self.logger.error(
                    f"Failed to fetch historical prices using {service}: {e} ⚠️"
                )

        self.logger.error(f"Failed to fetch historical prices for {token_id}. ❌")
        return []

    async def get_token_volume(self, token_id: str) -> float:

        if token_id in self.volume_cache:
            self.logger.debug(
                f"Returning cached trading volume for {token_id}. 📊⏳"
            )
            return self.volume_cache[token_id]

        for service in self.api_keys.keys():
            try:
                self.logger.debug(
                    f"Fetching volume for {token_id} using {service}. 📊⏳"
                )
                headers = {}
                if service == "BINANCE":
                    symbol = self._convert_token_id_to_binance_symbol(token_id)
                    if not symbol:
                        continue
                    url = f"https://api.binance.com/api/v3/ticker/24hr"
                    params = {"symbol": symbol}
                elif service == "COINGECKO":
                    url = f"https://api.coingecko.com/api/v3/coins/{token_id}"
                    params = {}
                elif service == "COINMARKETCAP":
                    url = f"https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"
                    params = {"symbol": token_id}
                    headers = {"X-CMC_PRO_API_KEY": self.api_keys[service]}
                elif service == "CRYPTOCOMPARE":
                    url = f"https://min-api.cryptocompare.com/data/pricemultifull"
                    params = {"fsyms": token_id, "tsyms": "USD"}
                    headers = {"Apikey": self.api_keys[service]}
                else:
                    continue

                response = await self.make_request(url, params=params, headers=headers)
                data = await response.json()

                if service == "BINANCE":
                    volume = float(data["quoteVolume"])
                elif service == "COINGECKO":
                    volume = data["market_data"]["total_volume"]["usd"]
                elif service == "COINMARKETCAP":
                    volume = data["data"][token_id]["quote"]["USD"]["volume_24h"]
                elif service == "CRYPTOCOMPARE":
                    volume = data["RAW"][token_id]["USD"]["VOLUME24HOUR"]
                else:
                    continue

                self.volume_cache[token_id] = volume
                self.logger.debug(
                    f"Fetched trading volume for {token_id} using {service} successfully. 📊"
                )
                return volume

            except Exception as e:
                self.logger.error(
                    f"Failed to fetch trading volume using {service}: {e} ⚠️"
                )

        self.logger.error(f"Failed to fetch trading volume for {token_id}. ❌")
        return 0.0

    async def get_current_price(self, token_id: str) -> Optional[float]:

        for service in self.api_keys.keys():
            try:
                self.logger.debug(
                    f"Fetching current price for {token_id} using {service}. 📊⏳"
                )
                headers = {}
                if service == "BINANCE":
                    symbol = self._convert_token_id_to_binance_symbol(token_id)
                    if not symbol:
                        continue
                    url = f"https://api.binance.com/api/v3/ticker/price"
                    params = {"symbol": symbol}
                elif service == "COINGECKO":
                    url = f"https://api.coingecko.com/api/v3/simple/price"
                    params = {"ids": token_id, "vs_currencies": "usd"}
                elif service == "COINMARKETCAP":
                    url = f"https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"
                    params = {"symbol": token_id}
                    headers = {"X-CMC_PRO_API_KEY": self.api_keys[service]}
                elif service == "CRYPTOCOMPARE":
                    url = f"https://min-api.cryptocompare.com/data/price"
                    params = {"fsym": token_id, "tsyms": "USD"}
                    headers = {"Apikey": self.api_keys[service]}
                else:
                    continue

                response = await self.make_request(url, params=params, headers=headers)
                data = await response.json()

                if service == "BINANCE":
                    price = float(data["price"])
                elif service == "COINGECKO":
                    price = data.get(token_id, {}).get("usd", 0.0)
                elif service == "COINMARKETCAP":
                    price = data["data"][token_id]["quote"]["USD"]["price"]
                elif service == "CRYPTOCOMPARE":
                    price = data["USD"]
                else:
                    continue

                self.logger.debug(
                    f"Fetched current price for {token_id} using {service} successfully. 📊"
                )
                return price

            except Exception as e:
                self.logger.error(
                    f"Failed to fetch current price using {service}: {e} ⚠️"
                )

        self.logger.error(
            f"Failed on all services to fetch current price for {token_id}. ❌"
        )
        return None

    async def make_request(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> aiohttp.ClientResponse:
        max_attempts = 5
        backoff_time = 1  # Initial backoff time in seconds

        for attempt in range(1, max_attempts + 1):
            try:
                await asyncio.sleep(1)
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params, headers=headers) as response:
                        response.raise_for_status()
                        return response
            except aiohttp.ClientResponseError as e:
                if e.status == 429:  # Rate limit error
                    self.logger.warning(
                        f"Rate limited. Retrying in {backoff_time} seconds... ⏳"
                    )
                    await asyncio.sleep(backoff_time)
                    backoff_time *= 2  # Exponential backoff
                else:
                    self.logger.error(f"Failed Making HTTP Request: {e} ❌")
                    break
            except Exception as e:
                self.logger.error(f"Failed Making HTTP Request: {e} ❌")
                if attempt < max_attempts:
                    self.logger.warning(
                        f"Retrying in {backoff_time} seconds... ⏳"
                    )
                    await asyncio.sleep(backoff_time)
                    backoff_time *= 2  # Exponential backoff

        raise Exception("Failed HTTP request after multiple attempts. ❌ ")

    def _convert_token_id_to_binance_symbol(self, token_id: str) -> Optional[str]:
        return self.symbol_mapping.get(token_id.lower())


class Xplorer:
    """
    Builds and manages the entire bot, initializing all components,
    managing connections, and orchestrating the main execution loop.
    """

    def __init__(
        self, config: Config, logger: Optional[logging.Logger] = None
    ) -> None:
      
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.config = config
        self.web3: Optional[AsyncWeb3] = None  # Will be initialized asynchronously
        self.account: Optional[Account] = None  # Will be initialized asynchronously

        # Initialize core components (will be set after async initialization)
        self.nonce_manager: Optional[NonceManager] = None
        self.safety_net: Optional[SafetyNet] = None
        self.market_analyzer: Optional[MarketAnalyzer] = None
        self.monitor_array: Optional[MonitorArray] = None
        self.transaction_array: Optional[TransactionArray] = None
        self.strategy_manager: Optional[StrategyManager] = None

        self.logger.debug("Xplorer initialized successfully. 🌐✅")

    async def initialize(self):
     
        try:
            self.web3 = await self._initialize_web3()
            self.account = await self._initialize_account()

            # Initialize NonceManager
            self.nonce_manager = NonceManager(self.web3, self.account.address, self.logger)
            await self.nonce_manager.initialize()

            # Initialize SafetyNet
            self.safety_net = SafetyNet(self.web3, self.config, self.account, self.logger)

            # Load ERC20 ABI
            erc20_ABI = await self._load_contract_ABI(self.config.ERC20_ABI)

            # Initialize MarketAnalyzer
            self.market_analyzer = MarketAnalyzer(
                self.web3, erc20_ABI, self.config, self.logger
            )

            # Initialize MonitorArray
            monitored_tokens = await self.config.get_token_addresses()
            self.monitor_array = MonitorArray(
                web3=self.web3,
                safety_net=self.safety_net,
                nonce_manager=self.nonce_manager,
                logger=self.logger,
                monitored_tokens=monitored_tokens,
                erc20_ABI=erc20_ABI,
                config=self.config,
            )

            # Load Flashloan and Lending Pool ABIs
            flashloan_ABI = await self._load_contract_ABI(self.config.AAVE_V3_FLASHLOAN_ABI)
            lending_pool_ABI = await self._load_contract_ABI(self.config.AAVE_V3_LENDING_POOL_ABI)

            # Initialize TransactionArray
            self.transaction_array = TransactionArray(
                web3=self.web3,
                account=self.account,
                flashloan_contract_address=self.config.AAVE_V3_FLASHLOAN_CONTRACT_ADDRESS,
                flashloan_contract_ABI=flashloan_ABI,
                lending_pool_contract_address=self.config.AAVE_V3_LENDING_POOL_ADDRESS,
                lending_pool_contract_ABI=lending_pool_ABI,
                monitor=self.monitor_array,
                nonce_manager=self.nonce_manager,
                safety_net=self.safety_net,
                config=self.config,
                logger=self.logger,
                erc20_ABI=erc20_ABI,
            )
            await self.transaction_array.initialize()

            # Initialize StrategyManager
            self.strategy_manager = StrategyManager(
                transaction_array=self.transaction_array,
                market_analyzer=self.market_analyzer,
                logger=self.logger,
            )

            self.logger.debug("All components initialized successfully. 🌐✅")
        except Exception as e:
            self.logger.exception(f"Error during initialization: {e} ❌")
            sys.exit(1)

    async def _initialize_web3(self) -> AsyncWeb3:
       
        providers = []    

        # Attempt to connect via IPC if IPC_ENDPOINT is provided
        if self.config.IPC_ENDPOINT:
            providers.append(("IPC", AsyncIPCProvider(self.config.IPC_ENDPOINT)))

        # Attempt to connect via HTTP if HTTP_ENDPOINT is provided
        if self.config.HTTP_ENDPOINT:
            providers.append(("HTTP", AsyncHTTPProvider(self.config.HTTP_ENDPOINT)))

        if not providers:
            self.logger.error(
                "No valid endpoints provided in configuration. Exiting... ❌"
            )
            sys.exit(1)

        for name, provider in providers:
            self.logger.info(f"Connecting to Ethereum Network with {name}...")
            web3 = AsyncWeb3(provider, modules={"eth": (AsyncEth,)}, middleware=[])
            try:
                is_connected = await web3.is_connected()
                if is_connected:
                    client_version = await web3.client_version
                    self.logger.debug(
                        f"{name} Provider connected: {client_version} ✅"
                    )
                    # Add POA middleware if necessary
                    await self._add_middleware(web3)
                    return web3
                else:
                    self.logger.warning(f"Connection failed with {name}. Retrying...")
                    await loading_bar(f"Connection failed. Retrying with {name}...", 3)
            except Exception as e:
                self.logger.error(f"Error connecting with {name} provider: {e} ❌")

        self.logger.error("Failed to connect on all providers. Exiting... ❌")
        sys.exit(1)

    async def _add_middleware(self, web3: AsyncWeb3) -> None:

        try:
            chain_id = await web3.eth.chain_id
            if chain_id in (99, 100, 77, 7766, 56):
                # Inject the POA middleware at layer 0
                web3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
                self.logger.debug(f"POA Middleware added for POA Network at chain {chain_id} ✅")
            elif chain_id in (1, 3, 4, 5, 42, 420):
                # Add SignAndSendRawMiddleware for Mainnet
                web3.middleware_onion.add(SignAndSendRawMiddlewareBuilder.build(Account.from_key(self.config.WALLET_KEY)))
                self.logger.debug(f"SignAndSendRawMiddleware added for Mainnet at chain {chain_id} ✅")
            else:
                self.logger.warning(f"Unsupported chain ID: {chain_id}. No middleware added.")
        except Exception as e:
            self.logger.error(f"Error adding middleware: {e} ❌")
            raise

    async def _initialize_account(self) -> Account:
       
        try:
            account = Account.from_key(self.config.WALLET_KEY)
            self.web3.eth.default_account = account.address
            balance = await self.web3.eth.get_balance(account.address)
            self.logger.debug(f"Ethereum account initialized: {account.address} with balance {self.web3.from_wei(balance, 'ether')} ETH ✅")
            return account
        except ValueError as e:
            self.logger.error(f"Invalid private key: {e} ❌")
            sys.exit(1)
        except Exception as e:
            self.logger.error(f"Failed to initialize Ethereum account: {e} ❌")
            sys.exit(1)

    async def _load_contract_ABI(self, abi_path: str) -> List[Dict[str, Any]]:
      
        try:
            async with aiofiles.open(abi_path, "r") as abi_file:
                content = await abi_file.read()
                abi = json.loads(content)
                self.logger.debug(f"Loaded ABI from {abi_path} successfully. ✅")
                return abi
        except Exception as e:
            self.logger.error(f"Failed to load ABI from {abi_path}: {e} ❌")
            sys.exit(1)

    async def run(self) -> None:
        self.logger.info(f"Starting 0xplorer on {Web3.client_version}.. 🚀")
        await self.monitor_array.start_monitoring()

        try:
            while True:
                # Process profitable transactions
                while not self.monitor_array.profitable_transactions.empty():
                    target_tx = await self.monitor_array.profitable_transactions.get()
                    success = await self.strategy_manager.execute_strategy_for_transaction(
                        target_tx
                    )
                    tx_hash = target_tx.get("tx_hash", "Unknown")
                    if success:
                        self.logger.info(
                            f"Successfully executed strategy for transaction {tx_hash} -> etherscan.io/tx/{tx_hash}. ✅"
                        )
                    else:
                        self.logger.warning(
                            f"Failed to execute strategy for transaction {tx_hash}. ⚠️"
                        )

                await asyncio.sleep(1)

        except KeyboardInterrupt:
            self.logger.warning(
                "0xplorer interrupted by user. Stopping monitoring... 🛑⏳"
            )
            await self.monitor_array.stop_monitoring()
            self.logger.debug("Goodbye! 👋")
            sys.exit(0)
        except Exception as e:
            self.logger.exception(f"Unexpected error in 0xplorer's main loop: {e} ❌")
            await self.monitor_array.stop_monitoring()
            sys.exit(1)

    async def save_linearregression_session_data(self, token_symbol: str, model: LinearRegression) -> None:
        try:
            model_path = f"{self.config.MODEL_DIR}/{token_symbol}_linear_regression.joblib"
            joblib.dump(model, model_path)
            self.logger.debug(f"Saved Linear Regression model for {token_symbol} to {model_path} ✅")
        except Exception as e:
            self.logger.error(f"Failed to save Linear Regression model for {token_symbol}: {e} ❌")
            sys.exit(1)

    async def linearregression_save_market_data(self, token_symbol: str, prices: List[float], volumes: List[float]) -> None:

        try:
            data = pd.DataFrame({"Price": prices, "Volume": volumes})
            data_path = f"{self.config.DATA_DIR}/{token_symbol}_market_data.csv"
            data.to_csv(data_path, index=False)
            self.logger.debug(f"Saved market data for {token_symbol} to {data_path} ✅")
        except Exception as e:
            self.logger.error(f"Failed to save market data for {token_symbol}: {e} ❌")
            sys.exit(1) 

    async def stop(self) -> None:
        self.logger.debug("Stopping 0xplorer... 🛑⏳")
        await self.monitor_array.stop_monitoring()
        self.logger.debug("0xplorer wishes you a great day! 👋")
        sys.exit(0)


async def main():
    # Set up logging
    await setup_logging() 

    # Create a logger instance for the "Xplorer" module
    logger = logging.getLogger("0xplorer")

    # Load configuration
    config = Config(logger)
    await config.load()

    # Initialize and run the bot
    bot = Xplorer(config, logger)
    await bot.initialize()
    await bot.run()

if __name__ == "__main__":
    asyncio.run(main())