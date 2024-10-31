class Xplorer:
    """
    Builds and manages the entire bot, initializing all components,
    managing connections, and orchestrating the main execution loop.

    Attributes:

        config (Config): Configuration object containing settings and API keys.
        logger (Optional[logging.Logger]): Logger instance for logging.
        web3 (Optional[AsyncWeb3]): Initialized AsyncWeb3 instance.
        account (Optional[Account]): Account object for bot operations.
        nonce_manager (Optional[NonceManager]): NonceManager instance for managing transaction nonces.
        safety_net (Optional[SafetyNet]): SafetyNet instance for handling failed transactions.
        market_analyzer (Optional[MarketAnalyzer]): MarketAnalyzer instance for analyzing market conditions.
        monitor_array (Optional[MonitorArray]): MonitorArray instance for monitoring transactions.
        transaction_array (Optional[TransactionArray]): TransactionArray instance for managing transactions.
        strategy_manager (Optional[StrategyManager]): StrategyManager instance for executing strategies.
    """

    def __init__(
        self, config: Config, logger: Optional[logging.Logger] = None
    ) -> None:
        """
        Initializes the Xplorer, setting up all necessary components.

        Args:
            config (Config): Configuration object containing settings and API keys.
            logger (Optional[logging.Logger]): Logger instance for logging.

        Raises:
            SystemExit: If any component fails to initialize
        """
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.config = config
        self.web3: Optional[AsyncWeb3] = None  # Will be initialized asynchronously
        self.account: Optional[Account] = None  # Will be initialized asynchronously

        # Initialize core components (will be set after async initialization)
        self.nonce_manager: Optional[NonceManager] = None
        self.safety_net: Optional[SafetyNet] = None
        self.market_analyzer: Optional[MarketAnalyzer] = None
        self.monitor_array: Optional[MonitorArray] = None
        self.transaction_array: Optional[TransactionArray] = None
        self.strategy_manager: Optional[StrategyManager] = None

        self.logger.debug("Xplorer initialized successfully. 🌐✅")

    async def initialize(self):
        """
        Asynchronous initialization of components that require async calls.
        
        Raises:
            SystemExit: If any component fails to initialize.
        """
        try:
            self.web3 = await self._initialize_web3()
            self.account = await self._initialize_account()

            # Initialize NonceManager
            self.nonce_manager = NonceManager(self.web3, self.account.address, self.logger)
            await self.nonce_manager.initialize()

            # Initialize SafetyNet
            self.safety_net = SafetyNet(self.web3, self.config, self.account, self.logger)

            # Load ERC20 ABI
            erc20_ABI = await self._load_contract_ABI(self.config.ERC20_ABI)

            # Initialize MarketAnalyzer
            self.market_analyzer = MarketAnalyzer(
                self.web3, erc20_ABI, self.config, self.logger
            )

            # Initialize MonitorArray
            monitored_tokens = await self.config.get_token_addresses()
            self.monitor_array = MonitorArray(
                web3=self.web3,
                safety_net=self.safety_net,
                nonce_manager=self.nonce_manager,
                logger=self.logger,
                monitored_tokens=monitored_tokens,
                erc20_ABI=erc20_ABI,
                config=self.config,
            )

            # Load Flashloan and Lending Pool ABIs
            flashloan_ABI = await self._load_contract_ABI(self.config.AAVE_V3_FLASHLOAN_ABI)
            lending_pool_ABI = await self._load_contract_ABI(self.config.AAVE_V3_LENDING_POOL_ABI)

            # Initialize TransactionArray
            self.transaction_array = TransactionArray(
                web3=self.web3,
                account=self.account,
                flashloan_contract_address=self.config.AAVE_V3_FLASHLOAN_CONTRACT_ADDRESS,
                flashloan_contract_ABI=flashloan_ABI,
                lending_pool_contract_address=self.config.AAVE_V3_LENDING_POOL_ADDRESS,
                lending_pool_contract_ABI=lending_pool_ABI,
                monitor=self.monitor_array,
                nonce_manager=self.nonce_manager,
                safety_net=self.safety_net,
                config=self.config,
                logger=self.logger,
                erc20_ABI=erc20_ABI,
            )
            await self.transaction_array.initialize()

            # Initialize StrategyManager
            self.strategy_manager = StrategyManager(
                transaction_array=self.transaction_array,
                market_analyzer=self.market_analyzer,
                logger=self.logger,
            )

            self.logger.debug("All components initialized successfully. 🌐✅")
        except Exception as e:
            self.logger.exception(f"Error during initialization: {e} ❌")
            sys.exit(1)

    async def _initialize_web3(self) -> AsyncWeb3:
        """
        Initializes the AsyncWeb3 connection using IPC, WebSocket, or HTTP providers.

        Returns:
            AsyncWeb3: Initialized AsyncWeb3 instance.

        Raises:
            SystemExit: If all connection methods fail.
        """
        providers = []    

        # Attempt to connect via IPC if IPC_ENDPOINT is provided
        if self.config.IPC_ENDPOINT:
            providers.append(("IPC", AsyncIPCProvider(self.config.IPC_ENDPOINT)))

        # Attempt to connect via HTTP if HTTP_ENDPOINT is provided
        if self.config.HTTP_ENDPOINT:
            providers.append(("HTTP", AsyncHTTPProvider(self.config.HTTP_ENDPOINT)))

        if not providers:
            self.logger.error(
                "No valid endpoints provided in configuration. Exiting... ❌"
            )
            sys.exit(1)

        for name, provider in providers:
            self.logger.info(f"Connecting to Ethereum Network with {name}...")
            web3 = AsyncWeb3(provider, modules={"eth": (AsyncEth,)}, middleware=[])
            try:
                is_connected = await web3.is_connected()
                if is_connected:
                    client_version = await web3.client_version
                    self.logger.debug(
                        f"{name} Provider connected: {client_version} ✅"
                    )
                    # Add POA middleware if necessary
                    await self._add_middleware(web3)
                    return web3
                else:
                    self.logger.warning(f"Connection failed with {name}. Retrying...")
                    await loading_bar(f"Connection failed. Retrying with {name}...", 3)
            except Exception as e:
                self.logger.error(f"Error connecting with {name} provider: {e} ❌")

        self.logger.error("Failed to connect on all providers. Exiting... ❌")
        sys.exit(1)

    async def _add_middleware(self, web3: AsyncWeb3) -> None:
        """ 
        Injects the appropriate middleware based on the chain ID.

        Args:
            web3 (AsyncWeb3): Initialized AsyncWeb3 instance.

        Raises:
            SystemExit: If middleware injection fails.
        """
        try:
            chain_id = await web3.eth.chain_id
            if chain_id in (99, 100, 77, 7766, 56):
                # Inject the POA middleware at layer 0
                web3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
                self.logger.debug(f"POA Middleware added for POA Network at chain {chain_id} ✅")
            elif chain_id in (1, 3, 4, 5, 42, 420):
                # Add SignAndSendRawMiddleware for Mainnet
                web3.middleware_onion.add(SignAndSendRawMiddlewareBuilder.build(Account.from_key(self.config.WALLET_KEY)))
                self.logger.debug(f"SignAndSendRawMiddleware added for Mainnet at chain {chain_id} ✅")
            else:
                self.logger.warning(f"Unsupported chain ID: {chain_id}. No middleware added.")
        except Exception as e:
            self.logger.error(f"Error adding middleware: {e} ❌")
            raise

    async def _initialize_account(self) -> Account:
        """
        Initializes the Ethereum account using the private key provided in the configuration.

        Args:
            config (Config): Configuration object containing settings and API keys.

        Returns:
            Account: Initialized Account instance.

        Raises:
            SystemExit: If account initialization fails.
        """
        try:
            account = Account.from_key(self.config.WALLET_KEY)
            self.web3.eth.default_account = account.address
            balance = await self.web3.eth.get_balance(account.address)
            self.logger.debug(f"Ethereum account initialized: {account.address} with balance {self.web3.from_wei(balance, 'ether')} ETH ✅")
            return account
        except ValueError as e:
            self.logger.error(f"Invalid private key: {e} ❌")
            sys.exit(1)
        except Exception as e:
            self.logger.error(f"Failed to initialize Ethereum account: {e} ❌")
            sys.exit(1)

    async def _load_contract_ABI(self, abi_path: str) -> List[Dict[str, Any]]:
        """
        Loads the ABI from a given file path.

        Args:
            abi_path (str): Path to the ABI JSON file.

        Returns:
            List[Dict[str, Any]]: ABI as a list of dictionaries.

        Raises:
            SystemExit: If ABI loading fails.
        """
        try:
            async with aiofiles.open(abi_path, "r") as abi_file:
                content = await abi_file.read()
                abi = json.loads(content)
                self.logger.debug(f"Loaded ABI from {abi_path} successfully. ✅")
                return abi
        except Exception as e:
            self.logger.error(f"Failed to load ABI from {abi_path}: {e} ❌")
            sys.exit(1)

    async def run(self) -> None:
        """
        Main execution loop for the bot. Continuously monitors the mempool,
        identifies profitable transactions, and executes appropriate strategies.

        Args:
            monitor_array (MonitorArray): The MonitorArray instance for monitoring transactions
            strategy_manager (StrategyManager): The StrategyManager instance for executing strategies

        Returns:
            None

        Raises:
            SystemExit: If the bot is interrupted or an unexpected error occurs.
        """
        self.logger.info(f"Starting 0xplorer on {Web3.client_version}.. 🚀")
        await self.monitor_array.start_monitoring()

        try:
            while True:
                # Process profitable transactions
                while not self.monitor_array.profitable_transactions.empty():
                    target_tx = await self.monitor_array.profitable_transactions.get()
                    success = await self.strategy_manager.execute_strategy_for_transaction(
                        target_tx
                    )
                    tx_hash = target_tx.get("tx_hash", "Unknown")
                    if success:
                        self.logger.info(
                            f"Successfully executed strategy for transaction {tx_hash} -> etherscan.io/tx/{tx_hash}. ✅"
                        )
                    else:
                        self.logger.warning(
                            f"Failed to execute strategy for transaction {tx_hash}. ⚠️"
                        )

                await asyncio.sleep(1)

        except KeyboardInterrupt:
            self.logger.warning(
                "0xplorer interrupted by user. Stopping monitoring... 🛑⏳"
            )
            await self.monitor_array.stop_monitoring()
            self.logger.debug("Goodbye! 👋")
            sys.exit(0)
        except Exception as e:
            self.logger.exception(f"Unexpected error in 0xplorer's main loop: {e} ❌")
            await self.monitor_array.stop_monitoring()
            sys.exit(1)

    async def stop(self) -> None:
        """
        Stops the bot's execution gracefully.
        """
        self.logger.debug("Stopping 0xplorer... 🛑⏳")
        await self.monitor_array.stop_monitoring()
        self.logger.debug("0xplorer wishes you a great day! 👋")
        sys.exit(0)


async def main():
    """ Main entry point for the bot. """
    # Set up logging
    await setup_logging() 

    # Create a logger instance for the "Xplorer" module
    logger = logging.getLogger("0xplorer")

    # Load configuration
    config = Config(logger)
    await config.load()

    # Initialize and run the bot
    bot = Xplorer(config, logger)
    await bot.initialize()
    await bot.run()

if __name__ == "__main__":
    asyncio.run(main())


# ----------------- End of 0xplorer.py ----------------- #