class API_Config:
    def __init__(self, configuration):
        self.configuration = configuration
        
        self.session = aiohttp.ClientSession()
        self.price_cache = TTLCache(maxsize=1000, ttl=300)  # Cache for 5 minutes
        self.token_symbol_cache = TTLCache(maxsize=1000, ttl=86400)  # Cache for 1 day

        # API configuration
        self.api_configs = {
            "binance": {
                "base_url": "https://api.binance.com/api/v3",
                "success_rate": 1.0,
                "weight": 1.0,
            },
            "coingecko": {
                "base_url": "https://api.coingecko.com/api/v3",
                "api_key": self.configuration.COINGECKO_API_KEY,
                "success_rate": 1.0,
                "weight": 0.8,
            },
            "coinmarketcap": {
                "base_url": "https://pro-api.coinmarketcap.com/v1",
                "api_key": self.configuration.COINMARKETCAP_API_KEY,
                "success_rate": 1.0,
                "weight": 0.7,
            },
            "cryptocompare": {
                "base_url": "https://min-api.cryptocompare.com/data",
                "api_key": self.configuration.CRYPTOCOMPARE_API_KEY,
                "success_rate": 1.0,
                "weight": 0.6,
            },
        }

        # Thread-safe primitives
        self.api_lock = asyncio.Lock()

    async def get_token_symbol(self, web3, token_address: str) -> Optional[str]:
        if token_address in self.token_symbol_cache:
            return self.token_symbol_cache[token_address]
        elif token_address in self.configuration.TOKEN_SYMBOLS:
            symbol = self.configuration.TOKEN_SYMBOLS[token_address]
            self.token_symbol_cache[token_address] = symbol
            return symbol
        try:
            # Create contract instance
            erc20_abi = await self._load_abi(self.configuration.ERC20_ABI)
            contract = web3.eth.contract(address=token_address, abi=erc20_abi)
            symbol = await contract.functions.symbol().call()
            self.token_symbol_cache[token_address] = symbol  # Cache the result
            return symbol
        except Exception as e:
            logger.error(f"error getting symbol for token {token_address}: {e}")
            return None

    async def get_real_time_price(self, token: str, vs_currency: str = 'eth') -> Optional[Decimal]:
        """Get real-time price using weighted average from multiple sources."""
        cache_key = f"price_{token}_{vs_currency}"
        if cache_key in self.price_cache:
            return self.price_cache[cache_key]

        try:
            prices = []
            weights = []

            async with self.api_lock:
                for source, configuration in self.api_configs.items():
                    try:
                        price = await self._fetch_price(source, token, vs_currency)
                        if price:
                            prices.append(price)
                            weights.append(configuration["weight"] * configuration["success_rate"])
                    except Exception as e:
                        logger.error(f"error fetching price from {source}: {e}")
                        configuration["success_rate"] *= 0.9

            if not prices:
                logger.info(f"No valid prices found for {token} !")
                return None

            # Calculate weighted average price
            weighted_price = sum(p * w for p, w in zip(prices, weights)) / sum(weights)
            self.price_cache[cache_key] = Decimal(str(weighted_price))

            return self.price_cache[cache_key]

        except Exception as e:
            logger.error(f"error calculating weighted price for {token}: {e} !")
            return None

    async def _fetch_price(self, source: str, token: str, vs_currency: str) -> Optional[Decimal]:
        """Fetch the price of a token from a specified source."""
        configuration = self.api_configs.get(source)
        if not configuration:
            logger.info(f"API configuration for {source} not found.")
            return None

        if source == "coingecko":
            url = f"{configuration['base_url']}/simple/price"
            params = {"ids": token, "vs_currencies": vs_currency}
            try:
                response = await self.make_request(url, params=params)
                price = Decimal(str(response[token][vs_currency]))
                return price
            except Exception as e:
                logger.error(f"error fetching price from Coingecko: {e}")
                return None

        elif source == "coinmarketcap":
            url = f"{configuration['base_url']}/cryptocurrency/quotes/latest"
            params = {"symbol": token.upper(), "convert": vs_currency.upper()}
            headers = {"X-CMC_PRO_API_KEY": configuration["api_key"]}
            try:
                response = await self.make_request(url, params=params, headers=headers)
                data = response["data"][token.upper()]["quote"][vs_currency.upper()]["price"]
                price = Decimal(str(data))
                return price
            except Exception as e:
                logger.error(f"error fetching price from CoinMarketCap: {e}")
                return None

        elif source == "cryptocompare":
            url = f"{configuration['base_url']}/price"
            params = {"fsym": token.upper(), "tsyms": vs_currency.upper(), "api_key": configuration["api_key"]}
            try:
                response = await self.make_request(url, params=params)
                price = Decimal(str(response[vs_currency.upper()]))
                return price
            except Exception as e:
                logger.error(f"error fetching price from CryptoCompare: {e}")
                return None

        elif source == "binance":
            url = f"{configuration['base_url']}/ticker/price"
            symbol = f"{token.upper()}{vs_currency.upper()}"
            params = {"symbol": symbol}
            try:
                response = await self.make_request(url, params=params)
                price = Decimal(str(response["price"]))
                return price
            except Exception as e:
                logger.error(f"error fetching price from Binance: {e}")
                return None

        else:
            logger.info(f"Unsupported price source: {source}")
            return None

    async def make_request(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        max_attempts: int = 5,
        backoff_factor: float = 1.5,
    ) -> Any:
        """Make HTTP request with exponential backoff and circuit breaker."""

        for attempt in range(max_attempts):
            try:
                timeout = aiohttp.ClientTimeout(total=10 * (attempt + 1))
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(url, params=params, headers=headers) as response:
                        response.raise_for_status()
                        return await response.json()
            except Exception as e:
                if attempt == max_attempts - 1:
                    logger.info(
                        f"Request failed after {max_attempts} attempts: {e} !"
                    )
                    raise Exception(
                        f"Request failed after {max_attempts} attempts: {e}"
                    )
                wait_time = backoff_factor ** attempt
                await asyncio.sleep(wait_time)

    async def _load_abi(self, abi_path: str) -> List[Dict[str, Any]]:
        """Load contract abi from a file."""
        try:
            async with aiofiles.open(abi_path, 'r') as file:
                content = await file.read()
                abi = json.loads(content)
            logger.info(f"Loaded abi from {abi_path} successfully. ")
            return abi
        except Exception as e:
            logger.warning(f"failed to load abi from {abi_path}: {e} !")
            raise
    