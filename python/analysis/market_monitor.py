class Market_Monitor:
    MODEL_UPDATE_INTERVAL = 3600  # Update model every hour
    VOLATILITY_THRESHOLD = 0.05  # 5% standard deviation
    LIQUIDITY_THRESHOLD = 100000  # $100,000 in 24h volume

    def __init__(
        self,
        web3: AsyncWeb3,
        configuration: Optional[Configuration],
        api_config: Optional[API_Config],
    ):
        self.web3 = web3
        self.configuration = configuration
        self.api_config = api_config
        self.price_model = LinearRegression()
        self.model_last_updated = 0
        self.price_cache = TTLCache(maxsize=1000, ttl=300)  # Cache for 5 minutes


    async def check_market_conditions(self, token_address: str) -> Dict[str, Any]:
        """
        Check various market conditions for a given token

        :param token_address: Address of the token to check
        :return: Dictionary of market conditions
        """
        market_conditions = {
            "high_volatility": False,
            "bullish_trend": False,
            "bearish_trend": False,
            "low_liquidity": False,
        }
        token_symbol = await self.api_config.get_token_symbol(self.web3, token_address)
        if not token_symbol:
            logger.debug(f"Cannot get token symbol for address {token_address}!")
            return market_conditions

        prices = await self.fetch_historical_prices(token_symbol, days=1)
        if len(prices) < 2:
            logger.debug(f"Not enough price data to analyze market conditions for {token_symbol}")
            return market_conditions

        volatility = self._calculate_volatility(prices)
        if volatility > self.VOLATILITY_THRESHOLD:
            market_conditions["high_volatility"] = True
        logger.debug(f"Calculated volatility for {token_symbol}: {volatility}")

        moving_average = np.mean(prices)
        if prices[-1] > moving_average:
            market_conditions["bullish_trend"] = True
        elif prices[-1] < moving_average:
            market_conditions["bearish_trend"] = True

        volume = await self.get_token_volume(token_symbol)
        if volume < self.LIQUIDITY_THRESHOLD:
            market_conditions["low_liquidity"] = True

        return market_conditions

    def _calculate_volatility(self, prices: List[float]) -> float:
        """
        Calculate the volatility of a list of prices.

        :param prices: List of prices
        :return: Volatility as standard deviation of returns
        """
        prices_array = np.array(prices)
        returns = np.diff(prices_array) / prices_array[:-1]
        return np.std(returns)

    async def fetch_historical_prices(self, token_symbol: str, days: int = 30) -> List[float]:
        """
        Fetch historical price data for a given token symbol.

        :param token_symbol: Token symbol to fetch prices for
        :param days: Number of days to fetch prices for
        :return: List of historical prices
        """
        cache_key = f"historical_prices_{token_symbol}_{days}"
        if cache_key in self.price_cache:
            logger.debug(f"Returning cached historical prices for {token_symbol}.")
            return self.price_cache[cache_key]

        prices = await self._fetch_from_services(
            lambda _: self.api_config.fetch_historical_prices(token_symbol, days=days),
            f"historical prices for {token_symbol}"
        )
        if prices:
            self.price_cache[cache_key] = prices
        return prices or []

    async def get_token_volume(self, token_symbol: str) -> float:
        """
        Get the 24-hour trading volume for a given token symbol.

        :param token_symbol: Token symbol to fetch volume for
        :return: 24-hour trading volume
        """
        cache_key = f"token_volume_{token_symbol}"
        if cache_key in self.price_cache:
            logger.debug(f"Returning cached trading volume for {token_symbol}.")
            return self.price_cache[cache_key]

        volume = await self._fetch_from_services(
            lambda _: self.api_config.get_token_volume(token_symbol),
            f"trading volume for {token_symbol}"
        )
        if volume is not None:
            self.price_cache[cache_key] = volume
        return volume or 0.0

    async def _fetch_from_services(self, fetch_func, description: str):
        """
        Helper method to fetch data from multiple services.

        :param fetch_func: Function to fetch data from a service
        :param description: Description of the data being fetched
        :return: Fetched data or None
        """
        for service in self.api_config.api_configs.keys():
            try:
                logger.debug(f"Fetching {description} using {service}...")
                result = await fetch_func(service)
                if result:
                    return result
            except Exception as e:
                logger.warning(f"failed to fetch {description} using {service}: {e}")
        logger.warning(f"failed to fetch {description}.")
        return None

    async def predict_price_movement(self, token_symbol: str) -> float:
        """Predict the next price movement for a given token symbol.""" 
        current_time = time.time()
        if current_time - self.model_last_updated > self.MODEL_UPDATE_INTERVAL:
            await self._update_price_model(token_symbol)
        prices = await self.fetch_historical_prices(token_symbol, days=1)
        if not prices:
            logger.debug(f"No recent prices available for {token_symbol}.")
            return 0.0
        next_time = np.array([[len(prices)]])
        predicted_price = self.price_model.predict(next_time)[0]
        logger.debug(f"Price prediction for {token_symbol}: {predicted_price}")
        return float(predicted_price)

    async def _update_price_model(self, token_symbol: str):
        """
        Update the price prediction model.

        :param token_symbol: Token symbol to update the model for
        """
        prices = await self.fetch_historical_prices(token_symbol)
        if len(prices) > 10:
            X = np.arange(len(prices)).reshape(-1, 1)
            y = np.array(prices)
            self.price_model.fit(X, y)
            self.model_last_updated = time.time()

    async def is_arbitrage_opportunity(self, target_tx: Dict[str, Any]) -> bool:
        """
        Check if there's an arbitrage opportunity based on the target transaction.

        :param target_tx: Target transaction dictionary
        :return: True if arbitrage opportunity detected, else False
        """

        decoded_tx = await self.decode_transaction_input(target_tx["input"], target_tx["to"])
        if not decoded_tx:
            return False
        path = decoded_tx["params"].get("path", [])
        if len(path) < 2:
            return False
        token_address = path[-1]  # The token being bought
        token_symbol = await self.api_config.get_token_symbol(self.web3, token_address)
        if not token_symbol:
            return False

        prices = await self._get_prices_from_services(token_symbol)
        if len(prices) < 2:
            return False

        price_difference = abs(prices[0] - prices[1])
        average_price = sum(prices) / len(prices)
        if average_price == 0:
            return False
        price_difference_percentage = price_difference / average_price
        if price_difference_percentage > 0.01:
            logger.debug(f"Arbitrage opportunity detected for {token_symbol}")
            return True
        return False

    async def _get_prices_from_services(self, token_symbol: str) -> List[float]:
        """
        Get real-time prices from different services.

        :param token_symbol: Token symbol to get prices for
        :return: List of real-time prices
        """
        prices = []
        for service in self.api_config.api_configs.keys():
            try:
                price = await self.api_config.get_real_time_price(token_symbol)
                if price is not None:
                    prices.append(price)
            except Exception as e:
                logger.warning(f"failed to get price from {service}: {e}")
        return prices

    async def decode_transaction_input(
        self, input_data: str, contract_address: str
    ) -> Optional[Dict[str, Any]]:
        """
        Decode the input data of a transaction.

        :param input_data: Hexadecimal input data of the transaction.
        :param contract_address: Address of the contract being interacted with.
        :return: Dictionary containing function name and parameters if successful, else None.
        """
        try:
            erc20_abi = await self.api_config._load_abi(self.configuration.ERC20_ABI)
            contract = self.web3.eth.contract(address=contract_address, abi=erc20_abi)
            function_abi, params = contract.decode_function_input(input_data)
            return {"function_name": function_abi["name"], "params": params}
        except Exception as e:
            logger.warning(f"failed in decoding transaction input: {e}")
            return None

    async def initialize(self) -> None:
        """Initialize market monitor.""" 
        try:
            self.price_model = LinearRegression()
            self.model_last_updated = 0
            logger.debug("Market Monitor initialized ✅")
        except Exception as e:
            logger.critical(f"Market Monitor initialization failed: {e}")
            raise

    async def stop(self) -> None:
        """Stop market monitor operations.""" 
        try:
            # Clear caches and clean up resources
            self.price_cache.clear()
            logger.debug("Market Monitor stopped.")
        except Exception as e:
            logger.error(f"Error stopping Market Monitor: {e}")

#//////////////////////////////////////////////////////////////////////////////

@dataclass