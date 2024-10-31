class SafetyNet:
    def __init__(
        self,
        web3: AsyncWeb3,
        config: Config,
        account: Account,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Provides safety checks and utility functions for transactions.

        Args:
            web3 (AsyncWeb3): AsyncWeb3 instance connected to the Ethereum network.
            config (Config): Configuration object containing API keys and settings.
            account (Account): The Ethereum account.
            logger (Optional[logging.Logger]): Logger instance.
        """
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
        """
        Returns the balance of an account in ETH.

        Args:
            account (Account): The Ethereum account.

        Returns:
            Decimal: The balance in ETH.
        """
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
        """
        Ensures that a transaction is profitable after accounting for gas costs and slippage.

        Args:
            transaction_data (Dict[str, Any]): Data related to the transaction.
            minimum_profit_eth (float): The minimum acceptable profit in ETH.

        Returns:
            bool: True if the transaction is profitable, False otherwise.
        """
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
        """
        Estimates the gas required for a transaction.

        Args:
            transaction_data (Dict[str, Any]): Data related to the transaction.

        Returns:
            int: The estimated gas required.
        """
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
        """
        Fetch the current gas price using gas oracles with fallback options.

        Returns:
            float: The gas price in Gwei.
        """
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
        """
        Adjust slippage tolerance based on network congestion and market volatility.

        Returns:
            float: The adjusted slippage tolerance.
        """
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
        """
        Estimate network congestion level (0 to 1).

        Returns:
            float: The network congestion level.
        """
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
        """
        Fetches the real-time price of a token in terms of ETH from multiple sources.

        Args:
            token (str): The token symbol.

        Returns:
            Decimal: The price of the token in ETH.
        """
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
        """Fetches the real-time price of a token in terms of ETH from Binance API."""
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
        """Load token symbols from the JSON file."""
        try:
            with open(self.token_symbols, "r") as file:
                return json.load(file)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.logger.error(f"Error loading token symbols: {e}")
            return {}

    def _convert_token_id_to_binance_symbol(self, token_id: str) -> Optional[str]:
        """
        Converts a token ID to a Binance symbol using the loaded mappings.
        """
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
        """Fetches the real-time price of a token in terms of ETH from CoinMarketCap."""
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
        """Fetches the real-time price of a token in terms of ETH from CryptoCompare."""
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
        """
        Make a request to an external API with retry mechanism and exponential backoff.

        Args:
            url (str): The API endpoint.
            params (Optional[Dict[str, Any]]): Query parameters.
            headers (Optional[Dict[str, str]]): Request headers.

        Returns:
            aiohttp.ClientResponse: The API response.

        Raises:
            Exception: If the request fails after maximum retries.
        """
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