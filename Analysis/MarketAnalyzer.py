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
                    url = f"https://min-api.ryptocompare.com/data/price"
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