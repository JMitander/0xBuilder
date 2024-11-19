class API_Config:
    def __init__(self, Configuration: Any):
        self.configuration = Configuration
        self.session = None
        self.price_cache = TTLCache(maxsize=1000, ttl=300)  # Cache for 5 minutes
        self.token_symbol_cache = TTLCache(maxsize=1000, ttl=86400)  # Cache for 1 day

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self):
        if self.session:
            await self.session.close()

        # API configuration
        self.api_config = {
            "binance": {
                "base_url": "https://api.binance.com/api/v3",
                "success_rate": 1.0,
                "weight": 1.0,
                "rate_limit": 1200,  # Max requests per minute
            },
            "coingecko": {
                "base_url": "https://api.coingecko.com/api/v3",
                "api_key": self.configuration.COINGECKO_API_KEY,
                "success_rate": 1.0,
                "weight": 0.8,
                "rate_limit": 50,  # Max requests per minute
            },
            "coinmarketcap": {
                "base_url": "https://pro-api.coinmarketcap.com/v1",
                "api_key": self.configuration.COINMARKETCAP_API_KEY,
                "success_rate": 1.0,
                "weight": 0.7,
                "rate_limit": 333,  # Max requests per minute
            },
            "cryptocompare": {
                "base_url": "https://min-api.cryptocompare.com/data",
                "api_key": self.configuration.CRYPTOCOMPARE_API_KEY,
                "success_rate": 1.0,
                "weight": 0.6,
                "rate_limit": 80,  # Max requests per minute
            },
        }

        self.api_lock = asyncio.Lock()
        self.rate_limiters = {
            provider: asyncio.Semaphore(config.get("rate_limit", 10))
            for provider, config in self.api_config.items()
        }

    async def get_token_symbol(self, web3: AsyncWeb3, token_address: str) -> Optional[str]:
        """Get the token symbol for a given token address."""
        if token_address in self.token_symbol_cache:
            return self.token_symbol_cache[token_address]
        if token_address in self.configuration.TOKEN_SYMBOLS:
            symbol = self.configuration.TOKEN_SYMBOLS[token_address]
            self.token_symbol_cache[token_address] = symbol
            return symbol
        try:
            erc20_abi = await self._load_abi(self.configuration.ERC20_ABI)
            contract = web3.eth.contract(address=token_address, abi=erc20_abi)
            symbol = await contract.functions.symbol().call()
            self.token_symbol_cache[token_address] = symbol
            return symbol
        except Exception as e:
            logger.error(f"Error getting symbol for token {token_address}: {e}")
            return None

    async def get_real_time_price(self, token: str, vs_currency: str = "eth") -> Optional[Decimal]:
        """Get real-time price using weighted average from multiple sources."""
        cache_key = f"price_{token}_{vs_currency}"
        if cache_key in self.price_cache:
            return self.price_cache[cache_key]
        prices = []
        weights = []
        async with self.api_lock:
            for source, config in self.api_config.items():
                try:
                    price = await self._fetch_price(source, token, vs_currency)
                    if price:
                        prices.append(price)
                        weights.append(config["weight"] * config["success_rate"])
                except Exception as e:
                    logger.error(f"Error fetching price from {source}: {e}")
                    config["success_rate"] *= 0.9
        if not prices:
            logger.warning(f"No valid prices found for {token}!")
            return None
        weighted_price = sum(p * w for p, w in zip(prices, weights)) / sum(weights)
        self.price_cache[cache_key] = Decimal(str(weighted_price))
        return self.price_cache[cache_key]

    async def _fetch_price(self, source: str, token: str, vs_currency: str) -> Optional[Decimal]:
        """Fetch the price of a token from a specified source."""
        config = self.api_config.get(source)
        if not config:
            logger.debug(f"API configuration for {source} not found.")
            return None
        if source == "coingecko":
            url = f"{config['base_url']}/simple/price"
            params = {"ids": token, "vs_currencies": vs_currency}
            response = await self.make_request(source, url, params=params)
            return Decimal(str(response[token][vs_currency]))
        elif source == "coinmarketcap":
            url = f"{config['base_url']}/cryptocurrency/quotes/latest"
            params = {"symbol": token.upper(), "convert": vs_currency.upper()}
            headers = {"X-CMC_PRO_API_KEY": config["api_key"]}
            response = await self.make_request(source, url, params=params, headers=headers)
            data = response["data"][token.upper()]["quote"][vs_currency.upper()]["price"]
            return Decimal(str(data))
        elif source == "cryptocompare":
            url = f"{config['base_url']}/price"
            params = {"fsym": token.upper(), "tsyms": vs_currency.upper(), "api_key": config["api_key"]}
            response = await self.make_request(source, url, params=params)
            return Decimal(str(response[vs_currency.upper()]))
        elif source == "binance":
            url = f"{config['base_url']}/ticker/price"
            symbol = f"{token.upper()}{vs_currency.upper()}"
            params = {"symbol": symbol}
            response = await self.make_request(source, url, params=params)
            return Decimal(str(response["price"]))
        else:
            logger.warning(f"Unsupported price source: {source}")
            return None

    async def make_request(
        self,
        provider_name: str,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        max_attempts: int = 5,
        backoff_factor: float = 1.5,
    ) -> Any:
        """Make HTTP request with exponential backoff and rate limit per provider."""
        rate_limiter = self.rate_limiters.get(provider_name)
        
        if rate_limiter is None:
            rate_limiter = asyncio.Semaphore(10)
            self.rate_limiters[provider_name] = rate_limiter
        
        async with rate_limiter:
            for attempt in range(max_attempts):
                try:
                    timeout = aiohttp.ClientTimeout(total=10 * (attempt + 1))
                    async with self.session.get(url, params=params, headers=headers, timeout=timeout) as response:
                        if response.status == 429:
                            wait_time = backoff_factor ** attempt
                            logger.warning(f"Rate limit exceeded for {provider_name}, retrying in {wait_time}s...")
                            await asyncio.sleep(wait_time)
                            continue
                        response.raise_for_status()
                        return await response.json()

                except aiohttp.ClientResponseError as e:
                    if attempt == max_attempts - 1:
                        logger.error(f"Request failed after {max_attempts} attempts: {e}")
                        raise
                    wait_time = backoff_factor ** attempt
                    logger.warning(f"Request attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                except aiohttp.ClientConnectionError as e:
                    if attempt == max_attempts - 1:
                        logger.error(f"Connection error after {max_attempts} attempts: {e}")
                        raise
                    wait_time = backoff_factor ** attempt
                    logger.warning(f"Connection error on attempt {attempt + 1}: {e}. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                except Exception as e:
                    logger.error(f"Unexpected error: {e}")
                    raise

    async def fetch_historical_prices(self, token: str, days: int = 30) -> List[float]:
        """Fetch historical price data for a given token symbol."""
        cache_key = f"historical_prices_{token}_{days}"
        if cache_key in self.price_cache:
            logger.debug(f"Returning cached historical prices for {token}.")
            return self.price_cache[cache_key]
        prices = await self._fetch_from_services(
            lambda service: self._fetch_historical_prices(service, token, days),
            f"historical prices for {token}",
        )
        if prices:
            self.price_cache[cache_key] = prices
        return prices or []

    async def _fetch_historical_prices(self, source: str, token: str, days: int) -> Optional[List[float]]:
        """Fetch historical prices from a specified source."""
        config = self.api_config.get(source)
        if not config:
            logger.debug(f"API configuration for {source} not found.")
            return None
        if source == "coingecko":
            url = f"{config['base_url']}/coins/{token}/market_chart"
            params = {"vs_currency": "usd", "days": days}
            response = await self.make_request(source, url, params=params)
            return [price[1] for price in response["prices"]]
        else:
            logger.debug(f"Unsupported historical price source: {source}")
            return None

    async def get_token_volume(self, token: str) -> float:
        """Get the 24-hour trading volume for a given token symbol."""
        cache_key = f"token_volume_{token}"
        if cache_key in self.price_cache:
            logger.debug(f"Returning cached trading volume for {token}.")
            return self.price_cache[cache_key]
        volume = await self._fetch_from_services(
            lambda service: self._fetch_token_volume(service, token),
            f"trading volume for {token}",
        )
        if volume is not None:
            self.price_cache[cache_key] = volume
        return volume or 0.0

    async def _fetch_token_volume(self, source: str, token: str) -> Optional[float]:
        """Fetch token volume from a specified source."""
        config = self.api_config.get(source)
        if not config:
            logger.debug(f"API configuration for {source} not found.")
            return None
        if source == "coingecko":
            url = f"{config['base_url']}/coins/markets"
            params = {"vs_currency": "usd", "ids": token}
            response = await self.make_request(source, url, params=params)
            return response[0]["total_volume"] if response else None
        else:
            logger.debug(f"Unsupported volume source: {source}")
            return None

    async def _fetch_from_services(self, fetch_func, description: str):
        """Helper method to fetch data from multiple services."""
        for service in self.api_config.keys():
            try:
                logger.debug(f"Fetching {description} using {service}...")
                result = await fetch_func(service)
                if result:
                    return result
            except Exception as e:
                logger.warning(f"Failed to fetch {description} using {service}: {e}")
        logger.warning(f"Failed to fetch {description}.")
        return None

    async def _load_abi(self, abi_path: str) -> List[Dict[str, Any]]:
        """Load contract abi from a file."""
        try:
            async with aiofiles.open(abi_path, 'r') as file:
                content = await file.read()
                abi = json.loads(content)
            logger.debug(f"Loaded abi from {abi_path} successfully.")
            return abi
        except Exception as e:
            logger.error(f"Failed to load abi from {abi_path}: {e}")
            raise

    async def close(self):
        """Close the aiohttp session."""
        await self.session.close()
