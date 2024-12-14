import asyncio
import time
from decimal import Decimal
from typing import Any, Dict, List, Optional
from web3 import AsyncWeb3
from eth_account import Account
from web3.exceptions import TransactionNotFound, ContractLogicError, Web3ValueError
import logging

from analysis.safety_net import Safety_Net
from configuration.api_config import API_Config
from configuration.configuration import Configuration
from core.nonce_core import Nonce_Core

logger = logging.getLogger(__name__)

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
        erc20_abi: List[Dict[str, Any]] = None,
        configuration: Optional[Configuration] = None,
    ):
        # Core components
        self.web3 = web3
        self.configuration = configuration
        self.safety_net = safety_net
        self.nonce_core = nonce_core
        self.api_config = api_config

        # Monitoring state
        self.running = False
        self.pending_transactions = asyncio.Queue()
        self.monitored_tokens = set(monitored_tokens or [])
        self.profitable_transactions = asyncio.Queue()
        self.processed_transactions = set()

       
        
        # Configuration
        self.erc20_abi = erc20_abi or []
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
        """Fallback method to poll for pending transactions when filters aren't available."""
        last_block = await self.web3.eth.block_number
        
        while self.running:
            try:
                # Get current block
                current_block = await self.web3.eth.block_number
                
                # Process new blocks
                for block_num in range(last_block + 1, current_block + 1):
                    block = await self.web3.eth.get_block(block_num, full_transactions=True)
                    if block and block.transactions:
                        # Convert transactions to hash list format
                        tx_hashes = [tx.hash.hex() if hasattr(tx, 'hash') else tx['hash'].hex() 
                                   for tx in block.transactions]
                        await self._handle_new_transactions(tx_hashes)
                
                # Try to get pending transactions from txpool
                try:
                    # Use txpool.content() if available
                    if hasattr(self.web3, 'txpool'):
                        txpool_content = await self.web3.txpool.content()
                        pending_txs = []
                        
                        # Extract pending transactions from txpool
                        if 'pending' in txpool_content:
                            for account in txpool_content['pending'].values():
                                for nonce in account.values():
                                    pending_txs.append(nonce)
                        
                        if pending_txs:
                            tx_hashes = [tx['hash'] for tx in pending_txs if 'hash' in tx]
                            await self._handle_new_transactions(tx_hashes)
                    else:
                        # Fallback to getting pending transaction hashes
                        block = await self.web3.eth.get_block('pending', full_transactions=False)
                        if block and block.transactions:
                            await self._handle_new_transactions(block.transactions)
                            
                except Exception as e:
                    logger.debug(f"Could not get pending transactions: {e}")
                
                last_block = current_block
                await asyncio.sleep(1)  # Poll interval

            except Exception as e:
                logger.error(f"Error in polling loop: {e}")
                await asyncio.sleep(2)  # Error cooldown

    async def _setup_pending_filter(self) -> Optional[Any]:
        """Set up pending transaction filter with validation.""" 
        try:
            # Try to create a filter
            pending_filter = await self.web3.eth.filter("pending")
            
            # Validate the filter works
            try:
                await pending_filter.get_new_entries()
                logger.debug("Successfully set up pending transaction filter")
                return pending_filter
            except Exception as e:
                logger.warning(f"Filter validation failed: {e}")
                return None
                
        except Exception as e:
            logger.warning(f"Failed to setup pending filter: {e}")
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
        """Process and queue profitable transactions.""" 
        try:
            await self.profitable_transactions.put(analysis)
            logger.info(
                f"Profitable transaction identified: {analysis['tx_hash']}"
                f"(Estimated profit: {analysis.get('profit', 'Unknown')} ETH)"
            )
        except Exception as e:
            logger.debug(f"Error handling profitable transaction: {e}")

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
        """Analyze token transactions with improved function signature handling."""
        try:
            if not self.erc20_abi or not tx.input or len(tx.input) < 10:
                return {"is_profitable": False}

            # Get function selector from input (first 4 bytes / 8 chars after '0x')
            function_selector = tx.input[:10]  # includes '0x'
            selector_no_prefix = function_selector[2:]  # remove '0x'

            # Try multiple methods to identify the function
            function_name = None
            function_params = {}

            # Method 1: Try direct function signature lookup
            if selector_no_prefix in self.function_signatures:
                function_name = self.function_signatures[selector_no_prefix]
                # Basic parameter extraction for known functions
                if len(tx.input) >= 138:  # Minimum length for standard ERC20 calls
                    try:
                        # Extract parameters from input data
                        params_data = tx.input[10:]  # Remove function selector
                        if function_name == 'transfer':
                            # Parse transfer parameters
                            to_address = '0x' + params_data[:64][-40:]  # last 20 bytes
                            amount = int(params_data[64:128], 16)  # next 32 bytes
                            function_params = {'to': to_address, 'amount': amount}
                    except Exception as e:
                        logger.debug(f"Error parsing raw input parameters: {e}")

            # Method 2: Try contract decode if Method 1 failed
            if not function_name:
                try:
                    contract = self.web3.eth.contract(address=tx.to, abi=self.erc20_abi)
                    func_obj, decoded_params = contract.decode_function_input(tx.input)
                    
                    if hasattr(func_obj, 'fn_name'):
                        function_name = func_obj.fn_name
                    elif hasattr(func_obj, 'function_identifier'):
                        function_name = func_obj.function_identifier
                    
                    if decoded_params:
                        function_params = decoded_params
                except Web3ValueError:
                    logger.debug(f"Could not find any function with matching selector for transaction {tx.hash.hex()}")
                    return {"is_profitable": False}

            # Method 3: Check configuration signatures
            if not function_name and hasattr(self.configuration, 'ERC20_SIGNATURES'):
                function_name = self.configuration.ERC20_SIGNATURES.get(function_selector)

            if not function_name:
                logger.debug(f"Could not identify function for selector {function_selector}")
                return {"is_profitable": False}

            # Process identified transaction
            if function_name in ('transfer', 'transferFrom', 'swap'):
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

            return {"is_profitable": False}

        except Exception as e:
            logger.debug(f"Error analyzing token transaction {tx.hash.hex()}: {e}")
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
            output_token_symbol = await self.api_config.get_token_symbol(self.web3, output_token_address)
            if not output_token_symbol:
                logger.debug(
                    f"Output token symbol not found for address {output_token_address}. Skipping."
                )
                return Decimal(0)
            market_price = await self.api_config.get_real_time_price(
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

    async def initialize(self) -> None:
        """Initialize mempool monitor.""" 
        try:
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
        """Gracefully stop the Mempool Monitor.""" 
        try:
            self.running = False
            self.stopping = True
            await self.task_queue.join()
            logger.debug("Mempool Monitor stopped gracefully.")
        except Exception as e:
            logger.error(f"Error stopping Mempool Monitor: {e}")
            raise
#//////////////////////////////////////////////////////////////////////////////