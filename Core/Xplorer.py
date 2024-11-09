class Xplorer:
    """
    Builds and manages the entire MEV bot, initializing all components,
    managing connections, and orchestrating the main execution loop.
    """

    def __init__(self, config: Config, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.config = config
        self.web3: Optional[AsyncWeb3] = None
        self.account: Optional[Account] = None
        self.components: Dict[str, Any] = {
            'api_client': None,
            'nonce_manager': None,
            'safety_net': None,
            'market_analyzer': None,
            'monitor_array': None,
            'transaction_array': None,
            'strategy_manager': None
        }
        self.logger.debug("Xplorer core initialized successfully. 🌐✅")

    async def initialize(self) -> None:
        """Initialize all components with proper error handling."""
        try:
            # Initialize account first
            wallet_key = self.config.WALLET_KEY
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
            self.logger.info("All components initialized successfully. 🌐✅")
        except Exception as e:
            self.logger.critical(f"Fatal error during initialization: {e} ❌")
            await self.stop()

    async def _initialize_web3(self) -> Optional[AsyncWeb3]:
        """Initialize Web3 connection with multiple provider fallback."""
        providers = self._get_providers()
        if not providers:
            self.logger.critical("No valid endpoints provided. ❌")
            return None

        for provider_name, provider in providers:
            try:
                self.logger.info(f"Attempting connection with {provider_name}...")
                web3 = AsyncWeb3(provider, modules={"eth": (AsyncEth,)})

                if await self._test_connection(web3, provider_name):
                    await self._add_middleware(web3)
                    return web3

            except Exception as e:
                self.logger.error(f"{provider_name} connection failed: {e}")
                continue

        return None

    def _get_providers(self) -> List[Tuple[str, Union[AsyncIPCProvider, AsyncHTTPProvider, WebSocketProvider]]]:
        """Get list of available providers with validation."""
        providers = []
        if self.config.IPC_ENDPOINT and os.path.exists(self.config.IPC_ENDPOINT):
            providers.append(("IPC", AsyncIPCProvider(self.config.IPC_ENDPOINT)))
        if self.config.HTTP_ENDPOINT:
            providers.append(("HTTP", AsyncHTTPProvider(self.config.HTTP_ENDPOINT)))
        if self.config.WEBSOCKET_ENDPOINT:
            providers.append(("WebSocket", WebSocketProvider(self.config.WEBSOCKET_ENDPOINT)))
        return providers


    async def _test_connection(self, web3: AsyncWeb3, name: str) -> bool:
        """Test Web3 connection with retries."""
        for attempt in range(3):
            try:
                if await web3.is_connected():
                    chain_id = await web3.eth.chain_id
                    self.logger.info(f"Connected to network {name} (Chain ID: {chain_id}) ✅")
                    return True
            except Exception as e:
                self.logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(1)
        return False

    async def _add_middleware(self, web3: AsyncWeb3) -> None:
        """Add appropriate middleware based on network."""
        try:
            chain_id = await web3.eth.chain_id
            if chain_id in {99, 100, 77, 7766, 56}:  # POA networks
                web3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
                self.logger.info("Injected POA middleware.")
            elif chain_id in {1, 3, 4, 5, 42, 420}:  # ETH networks
                self.logger.info("No additional middleware required for ETH network.")
                pass
            else:
                self.logger.warning("Unknown network; no middleware injected.")
        except Exception as e:
            self.logger.error(f"Middleware configuration failed: {e}")
            raise

    async def _check_account_balance(self) -> None:
        """Check the Ethereum account balance."""
        try:
            if not self.account:
                raise ValueError("Account not initialized")

            balance = await self.web3.eth.get_balance(self.account.address)
            balance_eth = self.web3.from_wei(balance, 'ether')

            self.logger.info(f"Account {self.account.address} initialized")
            self.logger.info(f"Balance: {balance_eth:.4f} ETH")

            if balance_eth < 0.1:
                self.logger.warning("Low account balance! (<0.1 ETH)")

        except Exception as e:
            self.logger.error(f"Balance check failed: {e}")
            raise

    async def _initialize_components(self) -> None:
        """Initialize all bot components with proper error handling."""
        try:
            # Initialize core components
            self.components['nonce_manager'] = NonceManager(
                self.web3, self.account.address, self.logger
            )
            await self.components['nonce_manager'].initialize()

            api_client = ApiClient(self.config, self.logger)

            self.components['safety_net'] = SafetyNet(
                self.web3, self.config, self.account, api_client, self.logger
            )

            # Load contract ABIs
            erc20_abi = await self._load_contract_ABI(self.config.ERC20_ABI)
            flashloan_abi = await self._load_contract_ABI(self.config.AAVE_V3_FLASHLOAN_ABI)
            lending_pool_abi = await self._load_contract_ABI(self.config.AAVE_V3_LENDING_POOL_ABI)

            # Initialize analysis components
            self.components['market_analyzer'] = MarketAnalyzer(
                self.web3, self.config, api_client, self.logger
            )

            # Initialize monitoring components
            self.components['monitor_array'] = MonitorArray(
                web3=self.web3,
                safety_net=self.components['safety_net'],
                nonce_manager=self.components['nonce_manager'],
                api_client=api_client,
                logger=self.logger,
                monitored_tokens=await self.config.get_token_addresses(),
                erc20_ABI=erc20_abi,
                config=self.config
            )

            # Initialize transaction components
            self.components['transaction_array'] = TransactionArray(
                web3=self.web3,
                account=self.account,
                flashloan_contract_address=self.config.AAVE_V3_FLASHLOAN_CONTRACT_ADDRESS,
                flashloan_contract_ABI=flashloan_abi,
                lending_pool_contract_address=self.config.AAVE_V3_LENDING_POOL_ADDRESS,
                lending_pool_contract_ABI=lending_pool_abi,
                monitor=self.components['monitor_array'],
                nonce_manager=self.components['nonce_manager'],
                safety_net=self.components['safety_net'],
                api_client=api_client,
                config=self.config,
                logger=self.logger,
                erc20_ABI=erc20_abi
            )
            await self.components['transaction_array'].initialize()

            # Initialize strategy components
            self.components['strategy_manager'] = StrategyManager(
                transaction_array=self.components['transaction_array'],
                market_analyzer=self.components['market_analyzer'],
                safety_net=self.components['safety_net'],
                api_client=api_client,
                logger=self.logger
            )

        except Exception as e:
            self.logger.error(f"Component initialization failed: {e}")
            raise

    async def run(self) -> None:
        """Main execution loop with improved error handling."""
        self.logger.info("Starting Xplorer... 🚀")

        try:
            await self.components['monitor_array'].start_monitoring()

            while True:
                try:
                    await self._process_profitable_transactions()
                    await asyncio.sleep(1)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Error in main loop: {e}")
                    await asyncio.sleep(5)  # Back off on error

        except KeyboardInterrupt:
            self.logger.info("Received shutdown signal...")
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Graceful shutdown of all components."""
        self.logger.info("Shutting down Xplorer...")

        try:
            if self.components['monitor_array']:
                await self.components['monitor_array'].stop_monitoring()

            # Close the aiohttp session in ApiClient
            api_client: ApiClient = self.components['safety_net'].api_client
            await api_client.session.close()

            self.logger.info("Shutdown complete 👋")
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
        finally:
            sys.exit(0)

    async def _process_profitable_transactions(self) -> None:
        """Process profitable transactions from the queue."""
        monitor = self.components['monitor_array']
        strategy = self.components['strategy_manager']

        while not monitor.profitable_transactions.empty():
            try:
                tx = await monitor.profitable_transactions.get()
                tx_hash = tx.get('tx_hash', 'Unknown')[:10]
                strategy_type = tx.get('strategy_type', 'Unknown')
                self.logger.info(f"Processing transaction {tx_hash} with strategy type {strategy_type}")

                success = await strategy.execute_strategy_for_transaction(tx)

                if success:
                    self.logger.info(f"Strategy execution successful for tx: {tx_hash} ✅")
                else:
                    self.logger.warning(f"Strategy execution failed for tx: {tx_hash} ❌")

            except Exception as e:
                self.logger.error(f"Error processing transaction: {e}")

    async def _load_contract_ABI(self, abi_path: str) -> List[Dict[str, Any]]:
        """Load contract ABI from a file."""
        try:
            with open(abi_path, 'r') as file:
                abi = json.load(file)
            self.logger.info(f"Loaded ABI from {abi_path} successfully. ✅")
            return abi
        except Exception as e:
            self.logger.error(f"Failed to load ABI from {abi_path}: {e} ❌")
            raise

#//////////////////////////////////////////////////////////////////////////////

async def main():
    """Main entry point with proper setup and error handling."""
    logger = None
    try:
        # Setup logging
        await setup_logging()
        logger = logging.getLogger("Xplorer")
        logger.info("Starting Xplorer initialization...")

        # Initialize configuration
        config = Config(logger)
        await config.load()

        # Initialize and run the bot
        xplorer = Xplorer(config, logger)
        await xplorer.initialize()
        await xplorer.run()

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, initiating shutdown...")
        if 'xplorer' in locals():
            await xplorer.stop()
    except Exception as e:
        if logger:
            logger.critical(f"Fatal error: {e}", exc_info=True)
        else:
            print(f"Fatal error before logger initialization: {e}")
        sys.exit(1)
    finally:
        if logger:
            logger.info("Xplorer shutdown complete.")
        sys.exit(0)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown complete.")
    except Exception as e:
        print(f"Fatal error in asyncio.run: {e}")
        sys.exit(1)

#//////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////