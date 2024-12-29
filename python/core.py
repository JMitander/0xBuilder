# /home/mitander/0xBuilder/core.py
from pathlib import Path
import sys
import aiofiles
import async_timeout
import aiohttp
import hexbytes
import asyncio
import json
import logging
import os
import signal
import time
import tracemalloc

from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple, Union
from web3.eth import AsyncEth
from eth_account import Account
from web3 import AsyncHTTPProvider, AsyncIPCProvider, AsyncWeb3, WebSocketProvider
from web3.exceptions import ContractLogicError, TransactionNotFound
from web3.middleware import ExtraDataToPOAMiddleware
from abi_registry import ABI_Registry
from configuration import API_Config, Configuration
from monitor import Market_Monitor, Mempool_Monitor
from nonce import Nonce_Core
from net import Safety_Net, Strategy_Net

logger = logging.getLogger(__name__)

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
                (self.configuration.SUSHISWAP_ADDRESS, 'sushiswap', 'Sushiswap'),
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
                self.configuration.UNISWAP_ADDRESS: (self.uniswap_contract, "Uniswap"),
                self.configuration.SUSHISWAP_ADDRESS: (self.sushiswap_contract, "Sushiswap"),
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
                self.configuration.UNISWAP_ADDRESS: (self.uniswap_contract, "Uniswap"),
                self.configuration.SUSHISWAP_ADDRESS: (self.sushiswap_contract, "Sushiswap"),
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
        """Check the Ethereum account balancer_abi."""
        try:
            if not self.account:
                raise ValueError("Account not initialized")

            balancer_abi = await self.web3.eth.get_balance(self.account.address)
            balance_eth = self.web3.from_wei(balancer_abi, 'ether')

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
        ''' check Account balance '''
        try:
            
            balance = await self.web3.eth.get_balance(self.account.address)
            balance_eth = self.web3.from_wei(balance, 'ether')
        
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