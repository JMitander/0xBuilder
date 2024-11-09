class MarketAnalyzer:
    def __init__(
        self,
        web3: AsyncWeb3,
        config: Config,
        api_client: ApiClient,
        logger: Optional[logging.Logger] = None,
    ):
        self.web3 = web3
        self.config = config
        self.api_client = api_client
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.price_model = LinearRegression()
        self.model_last_updated = 0
        self.MODEL_UPDATE_INTERVAL = 3600  # Update model every hour
        self.price_cache = TTLCache(maxsize=1000, ttl=300)  # Cache for 5 minutes

    async def check_market_conditions(self, token_address: str) -> Dict[str, Any]:
        """Check various market conditions for a given token."""
        market_conditions = {
            "high_volatility": False,
            "bullish_trend": False,
            "bearish_trend": False,
            "low_liquidity": False,
        }
        token_symbol = await self.api_client.get_token_symbol(self.web3, token_address)
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

    async def fetch_historical_prices(self, token_symbol: str, days: int = 30) -> List[float]:
        """Fetch historical price data for a given token symbol."""
        cache_key = f"historical_prices_{token_symbol}_{days}"
        if cache_key in self.price_cache:
            self.logger.debug(
                f"Returning cached historical prices for {token_symbol}. 📊⏳"
            )
            return self.price_cache[cache_key]

        for service in self.api_client.api_configs.keys():
            try:
                self.logger.debug(
                    f"Fetching historical prices for {token_symbol} using {service}... 📊⏳"
                )
                prices = await self.api_client.fetch_historical_prices(token_symbol, days=days)
                if prices:
                    self.price_cache[cache_key] = prices
                    return prices
            except Exception as e:
                self.logger.error(
                    f"Failed to fetch historical prices using {service}: {e} ⚠️"
                )

        self.logger.error(f"Failed to fetch historical prices for {token_symbol}. ❌")
        return []

    async def get_token_volume(self, token_symbol: str) -> float:
        """Get the 24-hour trading volume for a given token symbol."""
        cache_key = f"token_volume_{token_symbol}"
        if cache_key in self.price_cache:
            self.logger.debug(
                f"Returning cached trading volume for {token_symbol}. 📊⏳"
            )
            return self.price_cache[cache_key]

        for service in self.api_client.api_configs.keys():
            try:
                self.logger.debug(
                    f"Fetching volume for {token_symbol} using {service}. 📊⏳"
                )
                volume = await self.api_client.get_token_volume(token_symbol)
                if volume:
                    self.price_cache[cache_key] = volume
                    return volume
            except Exception as e:
                self.logger.error(
                    f"Failed to fetch trading volume using {service}: {e} ⚠️"
                )

        self.logger.error(f"Failed to fetch trading volume for {token_symbol}. ❌")
        return 0.0

    async def predict_price_movement(self, token_symbol: str) -> float:
        """Predict the next price movement for a given token symbol."""
        try:
            current_time = time.time()

            if current_time - self.model_last_updated > self.MODEL_UPDATE_INTERVAL:
                prices = await self.fetch_historical_prices(token_symbol)
                if len(prices) > 10:
                    X = np.arange(len(prices)).reshape(-1, 1)
                    y = np.array(prices)
                    self.price_model.fit(X, y)
                    self.model_last_updated = current_time

            prices = await self.fetch_historical_prices(token_symbol, days=1)
            if not prices:
                self.logger.warning(f"No recent prices available for {token_symbol}.")
                return 0.0

            next_time = np.array([[len(prices)]])
            predicted_price = self.price_model.predict(next_time)[0]

            self.logger.debug(f"Price prediction for {token_symbol}: {predicted_price}")
            return float(predicted_price)

        except Exception as e:
            self.logger.error(f"Price prediction failed: {str(e)}", exc_info=True)
            return 0.0

    async def is_arbitrage_opportunity(self, target_tx: Dict[str, Any]) -> bool:
        """Check if there's an arbitrage opportunity based on the target transaction."""
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
            token_symbol = await self.api_client.get_token_symbol(self.web3, token_address)
            if not token_symbol:
                return False
            # Get prices from different services
            price_binance = await self.api_client.get_real_time_price(token_symbol)
            price_coingecko = await self.api_client.get_real_time_price(token_symbol)
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

    async def decode_transaction_input(
        self, input_data: str, contract_address: str
    ) -> Optional[Dict[str, Any]]:
        """Decode the input data of a transaction."""
        try:
            erc20_ABI = await self.api_client._load_contract_ABI(self.config.ERC20_ABI)
            contract = self.web3.eth.contract(
                address=contract_address, abi=erc20_ABI
            )
            function_ABI, params = contract.decode_function_input(input_data)
            return {"function_name": function_ABI["name"], "params": params}
        except Exception as e:
            self.logger.error(f"Failed in decoding transaction input: {e} ❌")
            return None
