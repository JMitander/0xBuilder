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
            self.logger.error("Provider is not an HTTP or IPC provider. ❌")
            return
        else:
            self.logger.info(
                f"Connected to Ethereum network via {self.web3.provider.__class__.__name__}. ✨"
            )
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