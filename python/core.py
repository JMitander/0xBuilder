from ast import AsyncWith
import asyncio
import json
import logging
import os
import signal
import time
import tracemalloc
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from web3.eth import AsyncEth
import aiofiles
import async_timeout
import aiohttp
from eth_account import Account
import hexbytes
from web3 import AsyncHTTPProvider, AsyncIPCProvider, AsyncWeb3, WebSocketProvider
from cachetools import TTLCache
from eth_typing import Address
from web3.exceptions import ContractLogicError, TransactionNotFound
from web3.types import RPCResponse, TxParams, Wei
from web3.middleware import ExtraDataToPOAMiddleware

from .configuration import ABI_Manager, API_Config, Configuration
from .monitor import Market_Monitor, Mempool_Monitor
from .net import Safety_Net, Strategy_Net

logger = logging.getLogger(__name__)

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
        raise RuntimeError("Failed to fetch current nonce after retries")

    async def _get_pending_nonce(self) -> int:
        """Get highest nonce from pending transactions."""
        try:
            pending = await self.web3.eth.get_transaction_count(self.address, 'pending')
            logger.info(f"NonceCore Reports pending nonce: {pending}")
            return pending
        except Exception as e:
            logger.error(f"Error fetching pending nonce: {e}")
            return await self._fetch_current_nonce_with_retries()

    async def track_transaction(self, tx_hash: str, nonce: int) -> None:
        """Track pending transaction for nonce management."""
        self.pending_transactions.add(nonce)
        try:
            receipt = await self.web3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
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
        AAVE_FLASHLOAN_ADDRESS: str,
        AAVE_FLASHLOAN_ABI: List[Dict[str, Any]],
        AAVE_LENDING_POOL_ADDRESS: str,
        AAVE_LENDING_POOL_ABI: List[Dict[str, Any]],
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
        self.AAVE_FLASHLOAN_ADDRESS = AAVE_FLASHLOAN_ADDRESS
        self.AAVE_FLASHLOAN_ABI = AAVE_FLASHLOAN_ABI
        self.AAVE_LENDING_POOL_ADDRESS = AAVE_LENDING_POOL_ADDRESS
        self.AAVE_LENDING_POOL_ABI = AAVE_LENDING_POOL_ABI
        self.abi_manager = ABI_Manager()

    async def initialize(self):
        """Initialize with proper ABI loading."""
        try:
            # Load all required ABIs
            dex_abis = await asyncio.gather(
                self.abi_manager.load_abi('uniswap'),
                self.abi_manager.load_abi('sushiswap'),
                self.abi_manager.load_abi('pancakeswap'),
                self.abi_manager.load_abi('balancer'),
                self.abi_manager.load_abi('erc20')
            )

            # Initialize contracts with loaded ABIs
            router_addresses = [
                (self.configuration.UNISWAP_ROUTER_ADDRESS, dex_abis[0], 'Uniswap'),
                (self.configuration.SUSHISWAP_ROUTER_ADDRESS, dex_abis[1], 'Sushiswap'),
                (self.configuration.PANCAKESWAP_ROUTER_ADDRESS, dex_abis[2], 'Pancakeswap'),
                (self.configuration.BALANCER_ROUTER_ADDRESS, dex_abis[3], 'Balancer')
            ]

            for address, abi, name in router_addresses:
                if not abi:
                    raise ValueError(f"Failed to load {name} ABI")
                contract = await self._initialize_contract(address, abi, f"{name} Router")
                setattr(self, f"{name.lower()}_router_contract", contract)

            
            # Initialize ERC20 ABI
            self.erc20_abi = await self._load_erc20_abi()
            await self._validate_signatures()

            # Initialize Aave contracts
            self.aave_flashloan_contract = await self._initialize_contract(
                self.AAVE_FLASHLOAN_ADDRESS,
                self.AAVE_FLASHLOAN_ABI,
                "Aave Flashloan"
            )
            self.aave_lending_pool_contract = await self._initialize_contract(
                self.AAVE_LENDING_POOL_ADDRESS,
                self.AAVE_LENDING_POOL_ABI,
                "Aave Lending Pool"
            )

            logger.info("Transaction Core initialized successfully")

        except Exception as e:
            logger.error(f"Transaction Core initialization failed: {e}")
            raise

    async def _initialize_contract(self, address: str, abi: List[Dict[str, Any]], name: str):
        """Enhanced contract initialization with validation."""
        try:
            if not address or not abi:
                raise ValueError(f"Missing address or ABI for {name}")

            contract = self.web3.eth.contract(
                address=self.web3.to_checksum_address(address),
                abi=abi
            )
            
            # Validate basic contract functionality
            try:
                await contract.functions.WETH().call()  # Common DEX router method
                logger.debug(f"{name} contract validated successfully")
            except Exception as e:
                logger.warning(f"Contract validation warning for {name}: {e}")

            return contract

        except Exception as e:
            logger.error(f"Failed to initialize {name} contract: {e}")
            raise

    async def _load_erc20_abi(self) -> List[Dict[str, Any]]:
        """Load the ERC20 ABI with better path handling."""
        try:
            # Use pathlib for better path handling
            from pathlib import Path
            base_path = Path(__file__).parent.parent.parent
            abi_path = base_path / "abi" / "erc20_abi.json"
            
            async with aiofiles.open(str(abi_path), 'r') as f:
                content = await f.read()
                abi = json.loads(content)
                
            # Validate ABI structure
            required_methods = {'transfer', 'approve', 'transferFrom', 'balanceOf'}
            found_methods = {func['name'] for func in abi if 'name' in func}
            
            if not required_methods.issubset(found_methods):
                missing = required_methods - found_methods
                raise ValueError(f"ERC20 ABI missing required methods: {missing}")
                
            logger.debug(f"Loaded ERC20 ABI with {len(abi)} functions")
            return abi
            
        except FileNotFoundError:
            logger.error(f"ERC20 ABI file not found at {abi_path}")
            raise
        except json.JSONDecodeError:
            logger.error("Invalid JSON in ERC20 ABI file")
            raise
        except Exception as e:
            logger.error(f"Failed to load ERC20 ABI: {e}")
            raise

    async def _validate_signatures(self) -> None:
        """Validate loaded ERC20 signatures."""
        try:
            from pathlib import Path
            base_path = Path(__file__).parent.parent.parent
            sig_path = base_path / "utils" / "erc20_signatures.json"
            
            async with aiofiles.open(str(sig_path), 'r') as f:
                content = await f.read()
                signatures = json.loads(content)
                
            # Check for duplicate signatures
            sig_values = list(signatures.values())
            duplicates = {sig for sig in sig_values if sig_values.count(sig) > 1}
            if duplicates:
                logger.warning(f"Found duplicate signatures: {duplicates}")
                
            # Validate signature format
            invalid = [sig for sig in sig_values if not sig.startswith('0x') or len(sig) != 10]
            if invalid:
                raise ValueError(f"Invalid signature format: {invalid}")
                
            self.function_signatures = signatures
            logger.debug(f"Loaded {len(signatures)} ERC20 signatures")
            
        except Exception as e:
            logger.error(f"Error validating signatures: {e}")
            raise

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
                    'maxFeePerGas': base_fee * 2,  # Double the base fee
                    'maxPriorityFeePerGas': priority_fee
                })
            else:
                # Legacy gas price
                tx_params.update(await self.get_dynamic_gas_price())
            
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
                return False
        except KeyError as e:
            logger.error(f"Missing required transaction parameter: {e} ⚠️ ")
            return False
        except Exception as e:
            logger.error(f"Error handling ETH transaction: {e} ⚠️ ")
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
                            sleep_time = self.retry_delay * attempt
                            logger.warning(f"Retrying in {sleep_time} seconds...")
                            await asyncio.sleep(sleep_time)
                    except ValueError as e:
                        logger.error(f"Bundle submission error via {builder['name']}: {e} ⚠️ ")
                        break  # Move to next builder
                    except Exception as e:
                        logger.error(f"Unexpected error with {builder['name']}: {e}. Attempt {attempt} of {self.retry_attempts} ⚠️ ")
                        if attempt < self.retry_attempts:
                            sleep_time = self.retry_delay * attempt
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
        """Centralized transaction decoding for all components."""
        try:
            contract = self.web3.eth.contract(
                address=self.web3.to_checksum_address(contract_address),
                abi=self.erc20_abi
            )
            
            function_signature = input_data[:10]
            
            try:
                func_obj, decoded_params = contract.decode_function_input(input_data)
                function_name = (
                    getattr(func_obj, '_name', None) or 
                    getattr(func_obj, 'fn_name', None) or
                    getattr(func_obj, 'function_identifier', None)
                )
                
                return {
                    "function_name": function_name,
                    "params": decoded_params,
                    "signature": function_signature
                }
                
            except ContractLogicError:
                # Fallback to signature lookup
                if hasattr(self.configuration, 'ERC20_SIGNATURES'):
                    for name, sig in self.configuration.ERC20_SIGNATURES.items():
                        if sig == function_signature:
                            return {
                                "function_name": name,
                                "params": {},
                                "signature": function_signature
                            }
                raise
                
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

    async def calculate_gas_parameters(
        self,
        tx: Dict[str, Any],
        gas_limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """Centralized gas parameter calculation."""
        try:
            gas_price = await self.get_dynamic_gas_price()
            estimated_gas = gas_limit or await self.estimate_gas_smart(tx)
            return {
                'gasPrice': gas_price['gasPrice'],
                'gas': int(estimated_gas * 1.1)  # Add 10% buffer
            }
        except Exception as e:
            logger.error(f"Error calculating gas parameters: {e}")
            raise

        
class Main_Core:
    """
    Builds and manages the entire MEV bot, initializing all components,
    managing connections, and orchestrating the main execution loop.
    """

    def __init__(self, configuration: Configuration) -> None:
        # Take first memory snapshot after initialization
        self.memory_snapshot = tracemalloc.take_snapshot()
        self.configuration = configuration
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
                self.components['api_config']
            )
            await self.components['safety_net'].initialize()

            # 5. Initialize transaction core
            self.components['transaction_core'] = Transaction_Core(
                self.web3,
                self.account,
                self.configuration.AAVE_FLASHLOAN_ADDRESS,
                self.configuration.AAVE_FLASHLOAN_ABI,
                self.configuration.AAVE_LENDING_POOL_ADDRESS,
                self.configuration.AAVE_LENDING_POOL_ABI,
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
                erc20_abi=erc20_abi  # Pass the loaded ABI
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
        MAX_RETRIES = 3
        RETRY_DELAY = 2

        providers = await self._get_providers()
        if not providers:
            logger.error("No valid endpoints provided!")
            return None

        for provider_name, provider in providers:
            for attempt in range(MAX_RETRIES):
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
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(RETRY_DELAY * (attempt + 1))
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
        return False

    async def _add_middleware(self, web3: AsyncWeb3) -> None:
        """Add appropriate middleware based on network."""
        try:
            chain_id = await web3.eth.chain_id
            if chain_id in {99, 100, 77, 7766, 56}:  # POA networks
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
                market_monitor=self.components['market_monitor']
            ))

            # 6. Initialize mempool monitor last as it needs all components
            await self._initialize_component('mempool_monitor', Mempool_Monitor(
                web3=self.web3,
                safety_net=self.components['safety_net'],
                nonce_core=self.components['nonce_core'],
                api_config=self.components['api_config'],
                monitored_tokens=await self.configuration.get_token_addresses(),
                market_monitor=self.components['market_monitor']
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

    async def _monitor_memory(self, initial_snapshot) -> None:
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
        
        while not monitor.profitable_transactions.empty():
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
        """Load contract ABI from a file with better error handling."""
        try:
            # Handle both absolute and relative paths
            if not abi_path.startswith('/'):
                abi_path = os.path.join(os.path.dirname(__file__), '..', '..', abi_path)
            
            async with aiofiles.open(abi_path, 'r') as file:
                content = await file.read()
                abi = json.loads(content)
                logger.debug(f"Successfully loaded ABI from {abi_path}")
                return abi
        except FileNotFoundError:
            logger.error(f"ABI file not found at {abi_path}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in ABI file {abi_path}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error loading ABI from {abi_path}: {e}")
            return []

    async def _validate_abis(self) -> None:
        """Validate all required ABIs are present and properly formatted."""
        required_abis = [
            ('ERC20', self.configuration.ERC20_ABI),
            ('AAVE_FLASHLOAN', self.configuration.AAVE_FLASHLOAN_ABI),
            ('AAVE_LENDING_POOL', self.configuration.AAVE_LENDING_POOL_ABI),
            ('UNISWAP_ROUTER', self.configuration.UNISWAP_ROUTER_ABI),
            # ... other required ABIs
        ]
        
        for name, path in required_abis:
            try:
                if not await self._validate_abi(path):
                    raise ValueError(f"Invalid {name} ABI at {path}")
                    
                logger.debug(f"Validated {name} ABI")
                
            except Exception as e:
                logger.error(f"Error validating {name} ABI: {e}")
                raise

    async def _validate_abi(self, path: str) -> bool:
        """Validate individual ABI file."""
        try:
            async with aiofiles.open(path, 'r') as f:
                content = await f.read()
                abi = json.loads(content)
                
            if not isinstance(abi, list):
                logger.error(f"ABI at {path} is not a list")
                return False
                
            for item in abi:
                if not isinstance(item, dict):
                    logger.error(f"Invalid ABI item format in {path}")
                    return False
                    
                if 'type' not in item:
                    logger.error(f"ABI item missing 'type' in {path}")
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"ABI validation error: {e}")
            return False


# Modify the main function for better signal handling
async def main():
    """Main entry point with comprehensive setup and error handling."""
    # Log initial memory statistics
    logger.debug(f"Tracemalloc status: {tracemalloc.is_tracing()}")
    logger.debug(f"Initial traced memory: {tracemalloc.get_traced_memory()}")
    
    configuration = Configuration()
    core = Main_Core(configuration)
    
    def signal_handler():
        logger.debug("Shutdown signal received")
        if not core.running:
            return
        asyncio.create_task(core.stop())

    try:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, signal_handler)

        await core.initialize()
        await core.run()
    except Exception as e:
        logger.critical(f"An unexpected error occurred: {str(e)}")
    finally:
        # Remove signal handlers
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(sig)
        await core.stop()
        
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass  # Handle KeyboardInterrupt silently as it's already handled by signal handlers
    except Exception as e:
        # Get current memory snapshot on error
        snapshot = tracemalloc.take_snapshot()
        logger.critical(f"Program terminated with an error: {e}")
        logger.debug("Top 10 memory allocations at error:")
        top_stats = snapshot.statistics('lineno')
        for stat in top_stats[:10]:            logger.debug(str(stat))