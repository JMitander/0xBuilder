class Mempool_Monitor:
    """
    Advanced mempool monitoring system that identifies and analyzes profitable transactions.
    Includes sophisticated profit estimation, caching, and parallel processing capabilities.
    """

    def __init__(
        self,
        web3: AsyncWeb3,
        safety_net: Safety_Net,
        nonce_core: Nonce_Core,
        api_config: API_Config,
        
        monitored_tokens: Optional[List[str]] = None,
        erc20_abi: List[Dict[str, Any]] = None,
        configuration: Configuration = None,
    ):
        # Core components
        self.web3 = web3
        self.configuration = configuration
        self.safety_net = safety_net
        self.nonce_core = nonce_core
        self.api_config = api_config
        

        # Monitoring state
        self.running = False
        self.monitored_tokens = set(monitored_tokens or [])
        self.profitable_transactions = asyncio.Queue()
        self.processed_transactions = set()

        # Configurationsuration
        self.erc20_abi = erc20_abi or []
        self.minimum_profit_threshold = Decimal("0.001")
        self.max_parallel_tasks = 50
        self.retry_attempts = 3
        self.backoff_factor = 1.5

        # Concurrency control
        self.semaphore = asyncio.Semaphore(self.max_parallel_tasks)
        self.task_queue = asyncio.Queue()

        logger.debug(f"Mempool_Monitor initialized with enhanced configuration ")

    async def start_monitoring(self) -> None:
        """Start monitoring the mempool with improved error handling."""
        if self.running:
            logger.debug(f"Monitoring is already active ")
            return

        try:
            self.running = True
            monitoring_task = asyncio.create_task(self._run_monitoring())
            processor_task = asyncio.create_task(self._process_task_queue())

            logger.info(f"Mempool monitoring started....  ")
            await asyncio.gather(monitoring_task, processor_task)

        except Exception as e:
            self.running = False
            logger.warning(f"failed to start monitoring: {e} !")
            raise

    async def stop_monitoring(self) -> None:
        """Gracefully stop monitoring activities."""
        if not self.running:
            return

        self.running = False
        try:
            # Wait for remaining tasks to complete
            while not self.task_queue.empty():
                await asyncio.sleep(0.1)
            logger.info(f"Mempool monitoring stopped gracefully ")
        except Exception as e:
            logger.error(f"error during monitoring shutdown: {e} !")

    async def _run_monitoring(self) -> None:
        """Enhanced mempool monitoring with automatic recovery."""
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

            except Exception as e:
                retry_count += 1
                wait_time = min(self.backoff_factor ** retry_count, 30)
                logger.debug(
                     f"Monitoring error (attempt {retry_count}): {e} "
                )
                await asyncio.sleep(wait_time)

    async def _setup_pending_filter(self) -> Optional[Any]:
        """Set up pending transaction filter with validation."""
        try:
            if not isinstance(
                self.web3.provider, (AsyncHTTPProvider, AsyncIPCProvider)
            ):
                raise ValueError("Invalid provider type")

            pending_filter = await self.web3.eth.filter("pending")
            logger.debug(
                f"Connected to network via {self.web3.provider.__class__.__name__} "
            )
            return pending_filter

        except Exception as e:
            logger.warning(f"failed to setup pending filter: {e} !")
            return None

    async def _handle_new_transactions(self, tx_hashes: List[str]) -> None:
        """Process new transactions in parallel with rate limiting."""

        async def process_batch(batch):
            await asyncio.gather(
                *(self._queue_transaction(tx_hash) for tx_hash in batch)
            )

        try:
            # Process transactions in batches
            batch_size = 10
            for i in range(0, len(tx_hashes), batch_size):
                batch = tx_hashes[i : i + batch_size]
                await process_batch(batch)

        except Exception as e:
            logger.error(f"error processing transaction batch: {e} !")

    async def _queue_transaction(self, tx_hash: str) -> None:
        """Queue transaction for processing with deduplication."""
        tx_hash_hex = tx_hash.hex()
        if tx_hash_hex not in self.processed_transactions:
            self.processed_transactions.add(tx_hash_hex)
            await self.task_queue.put(tx_hash)

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
                logger.debug(f"Task processing error: {e} !")

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
            logger.error(f"error processing transaction {tx_hash}: {e} !")

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
                logger.error(f"error fetching transaction {tx_hash}: {e} !")
                return None

    async def _handle_profitable_transaction(self, analysis: Dict[str, Any]) -> None:
        """Process and queue profitable transactions."""
        try:
            await self.profitable_transactions.put(analysis)
            logger.debug(
                f"Profitable transaction identified: {analysis['tx_hash']} "
                f" (Estimated profit: {analysis.get('profit', 'Unknown')} ETH)"
            )
        except Exception as e:
            logger.error(f"error handling profitable transaction: {e} !")

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
            logger.debug(
                f"Error analyzing transaction {tx.hash.hex()}: {e} "
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
            logger.debug(
                f"Error analyzing ETH transaction {tx.hash.hex()}: {e} "
            )
            return {"is_profitable": False}

    async def _analyze_token_transaction(self, tx) -> Dict[str, Any]:
        try:
            contract = self.web3.eth.contract(address=tx.to, abi=self.erc20_abi)
            function_ABI, function_params = contract.decode_function_input(tx.input)
            function_name = function_ABI["name"]
            if function_name in self.configuration.ERC20_SIGNATURES:
                estimated_profit = await self._estimate_profit(tx, function_params)
                if estimated_profit > self.minimum_profit_threshold:
                    logger.debug(
                        f"Identified profitable transaction {tx.hash.hex()} with estimated profit: {estimated_profit:.4f} ETH "
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
                        f"Transaction {tx.hash.hex()} is below threshold. Skipping... "
                    )
                    return {"is_profitable": False}
            else:
                logger.debug(
                     f"Function {function_name} not in ERC20_SIGNATURES. Skipping."
                )
                return {"is_profitable": False}
        except Exception as e:
            logger.debug(
                f"Error decoding function input for transaction {tx.hash.hex()}: {e} !"
            )
            return {"is_profitable": False}

    async def _is_profitable_eth_transaction(self, tx) -> bool:
        try:
            potential_profit = await self._estimate_eth_transaction_profit(tx)
            return potential_profit > self.minimum_profit_threshold
        except Exception as e:
            logger.debug(
                f"Error estimating ETH transaction profit for transaction {tx.hash.hex()}: {e} !"
            )
            return False

    async def _estimate_eth_transaction_profit(self, tx: Any) -> Decimal:
        try:
            gas_price_gwei = await self.safety_net.get_dynamic_gas_price()
            gas_used = tx.gas
            gas_cost_eth = Decimal(gas_price_gwei) * Decimal(gas_used) * Decimal("1e-9")
            eth_value = Decimal(self.web3.from_wei(tx.value, "ether"))
            potential_profit = eth_value - gas_cost_eth
            return potential_profit if potential_profit > 0 else Decimal(0)
        except Exception as e:
            logger.error(f"error estimating ETH transaction profit: {e} !")
            return Decimal(0)

    async def _estimate_profit(self, tx, function_params: Dict[str, Any]) -> Decimal:
        try:
            gas_price_gwei = self.web3.from_wei(tx.gasPrice, "gwei")
            gas_used = tx.gas
            gas_cost_eth = Decimal(gas_price_gwei) * Decimal(gas_used) * Decimal("1e-9")
            input_amount_wei = Decimal(function_params.get("amountIn", 0))
            output_amount_min_wei = Decimal(function_params.get("amountOutMin", 0))
            path = function_params.get("path", [])
            if len(path) < 2:
                logger.debug(
                     f"Transaction {tx.hash.hex()} has an invalid path for swapping. Skipping. "
                )
                return Decimal(0)
            output_token_address = path[-1]
            output_token_symbol = await self.api_config.get_token_symbol(self.web3, output_token_address)
            if not output_token_symbol:
                logger.debug(
                     f"Output token symbol not found for address {output_token_address}. Skipping. "
                )
                return Decimal(0)
            market_price = await self.api_config.get_real_time_price(
                output_token_symbol.lower()
            )
            if market_price is None or market_price == 0:
                logger.debug(
                     f"Market price not available for token {output_token_symbol}. Skipping. "
                )
                return Decimal(0)
            input_amount_eth = Decimal(self.web3.from_wei(input_amount_wei, "ether"))
            output_amount_eth = Decimal(self.web3.from_wei(output_amount_min_wei, "ether"))
            expected_output_value = output_amount_eth * market_price
            profit = expected_output_value - input_amount_eth - gas_cost_eth
            return profit if profit > 0 else Decimal(0)
        except Exception as e:
            logger.debug(
                f"Error estimating profit for transaction {tx.hash.hex()}: {e} "
            )
            return Decimal(0)

    async def _log_transaction_details(self, tx, is_eth=False) -> None:
        try:
            transaction_info = {
                "transaction hash": tx.hash.hex(),
                "value": self.web3.from_wei(tx.value, "ether")
                if is_eth
                else tx.value,
                "from": tx["from"],
                "to": (tx.to[:10] + "..." + tx.to[-10:]) if tx.to else None,
                "input": tx.input,
                "gas price": self.web3.from_wei(tx.gasPrice, "gwei"),
            }
            if is_eth:
                logger.debug(f"Pending ETH Transaction Details: {transaction_info} ")
            else:
                logger.debug(
                     f"Pending Token Transaction Details: {transaction_info} "
                )
        except Exception as e:
            logger.debug(
                f"Error logging transaction details for {tx.hash.hex()}: {e} "
            )