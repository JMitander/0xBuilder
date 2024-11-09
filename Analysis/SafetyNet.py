class SafetyNet:
    """
    SafetyNet provides risk management and price verification functionality
    with multiple data sources, automatic failover, and dynamic adjustments.
    """

    def __init__(
        self,
        web3: AsyncWeb3,
        config: Config,
        account: Account,
        api_client: ApiClient,
        logger: Optional[logging.Logger] = None,
        cache_ttl: int = 300,  # Cache TTL in seconds
    ):
        self.web3 = web3
        self.config = config
        self.account = account
        self.api_client = api_client
        self.logger = logger or logging.getLogger(self.__class__.__name__)

        # Price data caching
        self.price_cache = TTLCache(maxsize=1000, ttl=cache_ttl)
        self.gas_price_cache = TTLCache(maxsize=1, ttl=15)  # 15 sec cache for gas prices

        # Thread-safe primitives
        self.price_lock = asyncio.Lock()

        # Configuration parameters
        self.slippage_config = {
            "default": 0.1,
            "min": 0.01,
            "max": 0.5,
            "high_congestion": 0.05,
            "low_congestion": 0.2,
        }

        self.gas_config = {
            "max_gas_price_gwei": 500,
            "min_profit_multiplier": 2.0,
            "base_gas_limit": 21000,
        }

        self.logger.info("SafetyNet initialized with enhanced configuration 🛡️✅")

    async def get_balance(self, account: Account) -> Decimal:
        """Get account balance with retries and caching."""
        cache_key = f"balance_{account.address}"
        if cache_key in self.price_cache:
            return self.price_cache[cache_key]

        for attempt in range(3):
            try:
                balance_wei = await self.web3.eth.get_balance(account.address)
                balance_eth = Decimal(self.web3.from_wei(balance_wei, "ether"))
                self.price_cache[cache_key] = balance_eth

                self.logger.debug(
                    f"Balance for {account.address[:10]}...: {balance_eth:.4f} ETH 💰"
                )
                return balance_eth
            except Exception as e:
                if attempt == 2:
                    self.logger.exception(f"Failed to get balance after 3 attempts: {e} ❌")
                    return Decimal(0)
                await asyncio.sleep(1 * (attempt + 1))

    async def ensure_profit(
        self,
        transaction_data: Dict[str, Any],
        minimum_profit_eth: Optional[float] = None,
    ) -> bool:
        """Enhanced profit verification with dynamic thresholds and risk assessment."""
        try:
            # Dynamic minimum profit threshold based on account balance
            if minimum_profit_eth is None:
                account_balance = await self.get_balance(self.account)
                minimum_profit_eth = (
                    0.003 if account_balance < Decimal("0.5") else 0.01
                )

            # Get gas costs with dynamic pricing
            gas_price_gwei = Decimal(await self.get_dynamic_gas_price())
            gas_used = await self.estimate_gas(transaction_data)

            if not self._validate_gas_parameters(gas_price_gwei, gas_used):
                return False

            # Calculate costs and expected output
            gas_cost_eth = self._calculate_gas_cost(gas_price_gwei, gas_used)
            slippage = await self.adjust_slippage_tolerance()

            # Get real-time price with weighted average
            output_token = transaction_data.get("output_token")
            real_time_price = await self.api_client.get_real_time_price(output_token)

            if not real_time_price:
                return False

            # Calculate profit with slippage consideration
            profit = await self._calculate_profit(
                transaction_data, real_time_price, slippage, gas_cost_eth
            )

            self._log_profit_calculation(
                transaction_data,
                real_time_price,
                gas_cost_eth,
                profit,
                minimum_profit_eth,
            )

            return profit > Decimal(minimum_profit_eth)

        except KeyError as e:
            self.logger.exception(f"Missing required transaction data key: {e} ❌")
        except Exception as e:
            self.logger.exception(f"Error in profit calculation: {e} ❌")
        return False

    def _validate_gas_parameters(self, gas_price_gwei: Decimal, gas_used: int) -> bool:
        """Validate gas parameters against safety thresholds."""
        if gas_used == 0:
            self.logger.error("Gas estimation returned zero ⚠️")
            return False

        if gas_price_gwei > self.gas_config["max_gas_price_gwei"]:
            self.logger.warning(
                f"Gas price {gas_price_gwei} gwei exceeds maximum threshold ⚠️"
            )
            return False

        return True

    def _calculate_gas_cost(self, gas_price_gwei: Decimal, gas_used: int) -> Decimal:
        """Calculate total gas cost in ETH."""
        return gas_price_gwei * Decimal(gas_used) * Decimal("1e-9")

    async def _calculate_profit(
        self,
        transaction_data: Dict[str, Any],
        real_time_price: Decimal,
        slippage: float,
        gas_cost_eth: Decimal,
    ) -> Decimal:
        """Calculate expected profit considering slippage and gas costs."""
        expected_output = real_time_price * Decimal(transaction_data["amountOut"])
        input_amount = Decimal(transaction_data["amountIn"])
        slippage_adjusted_output = expected_output * (1 - Decimal(slippage))

        return slippage_adjusted_output - input_amount - gas_cost_eth

    def _log_profit_calculation(
        self,
        transaction_data: Dict[str, Any],
        real_time_price: Decimal,
        gas_cost_eth: Decimal,
        profit: Decimal,
        minimum_profit_eth: float,
    ) -> None:
        """Log detailed profit calculation metrics."""
        self.logger.debug(
            f"Profit Calculation Summary:\n"
            f"Token: {transaction_data['output_token']}\n"
            f"Real-time Price: {real_time_price:.6f} ETH\n"
            f"Input Amount: {transaction_data['amountIn']:.6f} ETH\n"
            f"Expected Output: {transaction_data['amountOut']:.6f} tokens\n"
            f"Gas Cost: {gas_cost_eth:.6f} ETH\n"
            f"Calculated Profit: {profit:.6f} ETH\n"
            f"Minimum Required: {minimum_profit_eth} ETH\n"
            f"Profitable: {'Yes ✅' if profit > Decimal(minimum_profit_eth) else 'No ❌'}"
        )

    async def get_dynamic_gas_price(self) -> Decimal:
        """Get the current gas price dynamically."""
        if "gas_price" in self.gas_price_cache:
            return self.gas_price_cache["gas_price"]

        try:
            gas_price = await self.web3.eth.generate_gas_price()
            if gas_price is None:
                gas_price = await self.web3.eth.gas_price
            gas_price_gwei = self.web3.from_wei(gas_price, "gwei")
            self.gas_price_cache["gas_price"] = gas_price_gwei
            return gas_price_gwei
        except Exception as e:
            self.logger.exception(f"Error fetching dynamic gas price: {e} ❌")
            return Decimal(0)

    async def estimate_gas(self, transaction_data: Dict[str, Any]) -> int:
        """Estimate the gas required for a transaction."""
        try:
            gas_estimate = await self.web3.eth.estimate_gas(transaction_data)
            return gas_estimate
        except Exception as e:
            self.logger.exception(f"Gas estimation failed: {e} ❌")
            return 0

    async def adjust_slippage_tolerance(self) -> float:
        """Adjust slippage tolerance based on network conditions."""
        try:
            congestion_level = await self.get_network_congestion()
            if congestion_level > 0.8:
                slippage = self.slippage_config["high_congestion"]
            elif congestion_level < 0.2:
                slippage = self.slippage_config["low_congestion"]
            else:
                slippage = self.slippage_config["default"]
            slippage = min(max(slippage, self.slippage_config["min"]), self.slippage_config["max"])
            self.logger.debug(f"Adjusted slippage tolerance to {slippage * 100}%")
            return slippage
        except Exception as e:
            self.logger.exception(f"Error adjusting slippage tolerance: {e} ❌")
            return self.slippage_config["default"]

    async def get_network_congestion(self) -> float:
        """Estimate the current network congestion level."""
        try:
            latest_block = await self.web3.eth.get_block('latest')
            gas_used = latest_block['gasUsed']
            gas_limit = latest_block['gasLimit']
            congestion_level = gas_used / gas_limit
            self.logger.debug(f"Network congestion level: {congestion_level * 100}%")
            return congestion_level
        except Exception as e:
            self.logger.exception(f"Error fetching network congestion: {e} ❌")
            return 0.5  # Assume medium congestion if unknown
