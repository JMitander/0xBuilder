class Configuration:
    """
    Loads configuration from environment variables and monitored tokens from a JSON file.
    """

    async def load(self) -> None:
        """Loads the configuration."""
        await self._load_configuration()

    async def _load_configuration(self) -> None:
        try:
            await loading_bar("Loading Environment Variables", 2)
            self._load_api_keys()
            self._load_providers_and_account()
            await self._load_json_elements()
            logger.info("Configuration loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration: {e}") from e
        
    def _load_api_keys(self) -> None:
        self.ETHERSCAN_API_KEY = self._get_env_variable("ETHERSCAN_API_KEY")
        self.INFURA_PROJECT_ID = self._get_env_variable("INFURA_PROJECT_ID")
        self.COINGECKO_API_KEY = self._get_env_variable("COINGECKO_API_KEY")
        self.COINMARKETCAP_API_KEY = self._get_env_variable("COINMARKETCAP_API_KEY")
        self.CRYPTOCOMPARE_API_KEY = self._get_env_variable("CRYPTOCOMPARE_API_KEY")

    def _load_providers_and_account(self) -> None:
        self.HTTP_ENDPOINT = self._get_env_variable("HTTP_ENDPOINT")
        self.IPC_ENDPOINT = self._get_env_variable("IPC_ENDPOINT")
        self.WEBSOCKET_ENDPOINT = self._get_env_variable("WEBSOCKET_ENDPOINT")
        self.WALLET_KEY = self._get_env_variable("WALLET_KEY")
        self.WALLET_ADDRESS = self._get_env_variable("WALLET_ADDRESS")

    async def _load_json_elements(self) -> None:
        self.AAVE_LENDING_POOL_ADDRESS = self._get_env_variable("AAVE_LENDING_POOL_ADDRESS")
        self.TOKEN_ADDRESSES = await self._load_json_file(
            self._get_env_variable("TOKEN_ADDRESSES"), "monitored tokens"
        )
        self.TOKEN_SYMBOLS = await self._load_json_file(
            self._get_env_variable("TOKEN_SYMBOLS"), "token symbols"
        )
        self.ERC20_ABI = await self._construct_abi_path("abi", "erc20_abi.json")
        self.ERC20_SIGNATURES = await self._load_json_file(
            self._get_env_variable("ERC20_SIGNATURES"), "ERC20 function signatures"
        )
        self.SUSHISWAP_ROUTER_ABI = await self._construct_abi_path("abi", "sushiswap_router_abi.json")
        self.SUSHISWAP_ROUTER_ADDRESS = self._get_env_variable("SUSHISWAP_ROUTER_ADDRESS")
        self.UNISWAP_ROUTER_ABI = await self._construct_abi_path("abi", "uniswap_router_abi.json")
        self.UNISWAP_ROUTER_ADDRESS = self._get_env_variable("UNISWAP_ROUTER_ADDRESS")
        self.AAVE_FLASHLOAN_ABI = await self._construct_abi_path("abi", "aave_flashloan_abi.json")
        self.AAVE_LENDING_POOL_ABI = await self._construct_abi_path("abi", "aave_lending_pool_abi.json")
        self.AAVE_FLASHLOAN_ADDRESS = self._get_env_variable("AAVE_FLASHLOAN_ADDRESS")
        self.PANCAKESWAP_ROUTER_ABI = await self._construct_abi_path("abi", "pancakeswap_router_abi.json")
        self.PANCAKESWAP_ROUTER_ADDRESS = self._get_env_variable("PANCAKESWAP_ROUTER_ADDRESS")
        self.BALANCER_ROUTER_ABI = await self._construct_abi_path("abi", "balancer_router_abi.json")
        self.BALANCER_ROUTER_ADDRESS = self._get_env_variable("BALANCER_ROUTER_ADDRESS")

    def _get_env_variable(self, var_name: str, default: Optional[str] = None) -> str:
        value = os.getenv(var_name, default)
        if value is None:
            raise EnvironmentError(f"Missing environment variable: {var_name}")
        return value

    async def _load_json_file(self, file_path: str, description: str) -> Any:
        try:
            async with aiofiles.open(file_path, "r") as f:
                content = await f.read()
                data = json.loads(content)
                await loading_bar(f"Loading {len(data)} {description} from {file_path}", 3)
                return data
        except FileNotFoundError as e:
            logger.error(f"{description.capitalize()} file not found: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding {description} JSON: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load {description} from {file_path}: {e}")
            raise

    async def _construct_abi_path(self, base_path: str, abi_filename: str) -> str:
        abi_path = os.path.join(base_path, abi_filename)
        if not os.path.exists(abi_path):
            logger.error(f"abi file not found at path: {abi_path}")
            raise FileNotFoundError(f"abi file '{abi_filename}' not found in path '{base_path}'")
        return abi_path

    async def get_token_addresses(self) -> List[str]:
        return self.TOKEN_ADDRESSES

    async def get_token_symbols(self) -> Dict[str, str]:
        return self.TOKEN_SYMBOLS

    def get_abi_path(self, abi_name: str) -> str:
        abi_paths = {
            "erc20_abi": self.ERC20_ABI,
            "sushiswap_router_abi": self.SUSHISWAP_ROUTER_ABI,
            "uniswap_router_abi": self.UNISWAP_ROUTER_ABI,
            "aave_flashloan_abi": self.AAVE_FLASHLOAN_ABI,
            "aave_lending_pool_abi": self.AAVE_LENDING_POOL_ABI,
            "pancakeswap_router_abi": self.PANCAKESWAP_ROUTER_ABI,
            "balancer_router_abi": self.BALANCER_ROUTER_ABI,
        }
        return abi_paths.get(abi_name.lower(), "")
