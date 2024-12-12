class MainCore:
    """
    Builds and manages the entire MEV bot, initializing all components,
    managing connections, and orchestrating the main execution loop.
    """

    def __init__(self, configuration: Configuration) -> None:
        
        self.configuration = configuration
        self.web3: Optional[AsyncWeb3] = None
        self.account: Optional[Account] = None
        self.components: Dict[str, Any] = {
            'apiconfig': None,
            'noncecore': None,
            'safetynet': None,
            'marketmonitor': None,
            'mempoolmonitor': None,
            'transactioncore': None,
            'strategynet': None
        }
        logger.debug(f"MainCore core initialized successfully. ")

    async def initialize(self) -> None:
        """Initialize all components with  error handling."""
        try:
            # Initialize account first
            wallet_key = self.configuration.WALLET_KEY
            if not wallet_key:
                raise ValueError("WALLET_KEY environment variable is not set or empty")

            try:
                # Remove '0x' prefix if present and ensure the key is valid hex
                cleaned_key = wallet_key[2:] if wallet_key.startswith('0x') else wallet_key
                if not all(c in '0123456789abcdefABCDEF' for c in cleaned_key):
                    raise ValueError("Invalid wallet key format - must be hexadecimal")
                # Add '0x' prefix back if it was removed
                full_key = f"0x{cleaned_key}" if not wallet_key.startswith('0x') else wallet_key
                self.account = Account.from_key(full_key)
            except Exception as e:
                raise ValueError(f"Invalid wallet key format: {e}")

            # Initialize web3 after account is set up
            self.web3 = await self._initialize_web3()
            if not self.web3:
                raise RuntimeError("Failed to initialize Web3 connection")

            if not self.account:
                raise RuntimeError("Failed to initialize account")

            await self._check_account_balance()
            await self._initialize_components()
            logger.debug(f"All components initialized successfully. ")
        except Exception as e:
            logger.debug(f"Fatal error during initialization: {e} !")
            await self.stop()

    async def _initialize_web3(self) -> Optional[AsyncWeb3]:
        """Initialize Web3 connection with multiple provider fallback."""
        providers = self._get_providers()
        if not providers:
            logger.debug(f"No valid endpoints provided. !")
            return None

        for provider_name, provider in providers:
            try:
                logger.debug(f"Attempting connection with {provider_name}...")
                web3 = AsyncWeb3(provider, modules={"eth": (AsyncEth,)})

                if await self._test_connection(web3, provider_name):
                    await self._add_middleware(web3)
                    return web3

            except Exception as e:
                logger.debug(f"{provider_name} connection failed: {e}")
                continue

        return None

    def _get_providers(self) -> List[Tuple[str, Union[AsyncIPCProvider, AsyncHTTPProvider, WebSocketProvider]]]:
        """Get list of available providers with validation."""
        providers = []
        if self.configuration.IPC_ENDPOINT and os.path.exists(self.configuration.IPC_ENDPOINT):
            providers.append(("IPC", AsyncIPCProvider(self.configuration.IPC_ENDPOINT)))
        if self.configuration.HTTP_ENDPOINT:
            providers.append(("HTTP", AsyncHTTPProvider(self.configuration.HTTP_ENDPOINT)))
        if self.configuration.WEBSOCKET_ENDPOINT:
            providers.append(("WebSocket", WebSocketProvider(self.configuration.WEBSOCKET_ENDPOINT)))
        return providers


    async def _test_connection(self, web3: AsyncWeb3, name: str) -> bool:
        """Test Web3 connection with retries."""
        for attempt in range(3):
            try:
                if await web3.is_connected():
                    chain_id = await web3.eth.chain_id
                    logger.info(f"Connected to network {name} (Chain ID: {chain_id}) ")
                    return True
            except Exception as e:
                logger.debug(f"Connection attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(1)
        return False

    async def _add_middleware(self, web3: AsyncWeb3) -> None:
        """Add appropriate middleware based on network."""
        try:
            chain_id = await web3.eth.chain_id
            if chain_id in {99, 100, 77, 7766, 56}:  # POA networks
                web3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
                logger.debug(f"Injected POA middleware.")
            elif chain_id in {1, 3, 4, 5, 42, 420}:  # ETH networks
                logger.debug(f"No additional middleware required for ETH network.")
                pass
            else:
                logger.debug(f"Unknown network; no middleware injected.")
        except Exception as e:
            logger.debug(f"Middleware configuration failed: {e}")
            raise

    async def _check_account_balance(self) -> None:
        """Check the Ethereum account balancer_router_abi."""
        try:
            if not self.account:
                raise ValueError("Account not initialized")

            balancer_router_abi = await self.web3.eth.get_balance(self.account.address)
            balance_eth = self.web3.from_wei(balancer_router_abi, 'ether')

            logger.info(f"Account {self.account.address} initialized")
            logger.debug(f"Balance: {balance_eth:.4f} ETH")

            if balance_eth < 0.1:
                logger.debug(f"Low account balancer_router_abi! (<0.1 ETH)")

        except Exception as e:
            logger.debug(f"Balance check failed: {e}")
            raise

    async def _initialize_components(self) -> None:
        """Initialize all bot components with  error handling."""
        try:
            # Initialize core components
            self.components['noncecore'] = NonceCore(
                self.web3, self.account.address
            )
            await self.components['noncecore'].initialize()

            apiconfig = APIConfig(self.configuration)

            self.components['safetynet'] = SafetyNet(
                self.web3, self.configuration, self.account, apiconfig
            )

            # Load contract ABIs
            erc20_abi = await self._load_abi(self.configuration.ERC20_ABI)
            AAVE_FLASHLOAN_ABI = await self._load_abi(self.configuration.AAVE_FLASHLOAN_ABI)
            AAVE_LENDING_POOL_ABI = await self._load_abi(self.configuration.AAVE_LENDING_POOL_ABI)

            # Initialize analysis components
            self.components['marketmonitor'] = MarketMonitor(
                self.web3, self.configuration, apiconfig
            )

            # Initialize monitoring components
            self.components['mempoolmonitor'] = MempoolMonitor(
                web3=self.web3,
                safetynet=self.components['safetynet'],
                noncecore=self.components['noncecore'],
                apiconfig=apiconfig,
                
                monitored_tokens=await self.configuration.get_token_addresses(),
                erc20_abi=erc20_abi,
                configuration=self.configuration
            )

            # Initialize transaction components
            self.components['transactioncore'] = TransactionCore(
                web3=self.web3,
                account=self.account,
                AAVE_FLASHLOAN_ADDRESS=self.configuration.AAVE_FLASHLOAN_ADDRESS,
                AAVE_FLASHLOAN_ABI=AAVE_FLASHLOAN_ABI,
                AAVE_LENDING_POOL_ADDRESS=self.configuration.AAVE_LENDING_POOL_ADDRESS,
                AAVE_LENDING_POOL_ABI=AAVE_LENDING_POOL_ABI,
                monitor=self.components['mempoolmonitor'],
                noncecore=self.components['noncecore'],
                safetynet=self.components['safetynet'],
                apiconfig=apiconfig,
                configuration=self.configuration,
                
                erc20_abi=erc20_abi
            )
            await self.components['transactioncore'].initialize()

            # Initialize strategy components
            self.components['strategynet'] = StrategyNet(
                transactioncore=self.components['transactioncore'],
                marketmonitor=self.components['marketmonitor'],
                safetynet=self.components['safetynet'],
                apiconfig=apiconfig,
                
            )

        except Exception as e:
            logger.debug(f"Component initialization failed: {e}")
            raise

    async def run(self) -> None:
        """Main execution loop with improved error handling."""
        logger.debug(f"Starting MainCore... ")

        try:
            await self.components['mempoolmonitor'].start_monitoring()

            while True:
                try:
                    await self._process_profitable_transactions()
                    await asyncio.sleep(1)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"error in main loop: {e}")
                    await asyncio.sleep(5)  # Back off on error

        except KeyboardInterrupt:
            logger.debug(f"Received shutdown signal...")
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Graceful shutdown of all components."""
        logger.debug(f"Shutting down MainCore...")

        try:
            if self.components['mempoolmonitor']:
                await self.components['mempoolmonitor'].stop_monitoring()

            # Close the aiohttp session in APIConfig
            apiconfig: APIConfig = self.components['safetynet'].apiconfig
            await apiconfig.session.close()

            logger.debug(f"Shutdown complete ")
        except Exception as e:
            logger.error(f"error during shutdown: {e}")
        finally:
            sys.exit(0)

    async def _process_profitable_transactions(self) -> None:
        """Process profitable transactions from the queue."""
        monitor = self.components['mempoolmonitor']
        strategy = self.components['strategynet']

        while not monitor.profitable_transactions.empty():
            try:
                tx = await monitor.profitable_transactions.get()
                tx_hash = tx.get('tx_hash', 'Unknown')[:10]
                strategy_type = tx.get('strategy_type', 'Unknown')
                logger.debug(f"Processing transaction {tx_hash} with strategy type {strategy_type}")

                success = await strategy.execute_strategy_for_transaction(tx)

                if success:
                    logger.debug(f"Strategy execution successful for tx: {tx_hash} ")
                else:
                    logger.debug(f"Strategy execution failed for tx: {tx_hash} !")

            except Exception as e:
                logger.error(f"error processing transaction: {e}")

    async def _load_abi(self, abi_path: str) -> List[Dict[str, Any]]:
        """Load contract abi from a file."""
        try:
            with open(abi_path, 'r') as file:
                abi = json.load(file)
            logger.info(f"Loaded abi from {abi_path} successfully. ")
            return abi
        except Exception as e:
            logger.warning(f"failed to load abi from {abi_path}: {e} !")
            raise

#//////////////////////////////////////////////////////////////////////////////

async def main():
    """Main entry point with  setup and error handling."""
    logger = None
    try:
        # Setup logging
        await setup_logging()
        logger = logging.getLogger("MainCore")
        logger.debug("Starting MainCore initialization...")

        # Initialize configuration
        configuration = Configuration(logger)
        await configuration.load()

        # Initialize and run the bot
        main_cn_cre = MainCore(configuration, logger)
        await main_cn_cre.initialize()
        await main_cn_cre.run()

    except KeyboardInterrupt:
        logger.debug("Received keyboard interrupt, initiating shutdown...")
        if 'main_cn_cre' in locals():
            await main_cn_cre.stop()
    except Exception as e:
        if logger:
            logger.critical(f"Fatal error: {e}", exc_info=True)
        else:
            logger.debug(f"Fatal error before logger initialization: {e}")
        sys.exit(1)
    finally:
        if logger:
            logger.debug("MainCore shutdown complete.")
        sys.exit(0)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.debug(f"nShutdown complete.")
    except Exception as e:
        logger.debug(f"Fatal error in asyncio.run: {e}")
        sys.exit(1)

#//////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////