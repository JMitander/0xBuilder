class Xplorer_UI:
    """Simple terminal-based UI for the 0xplorer bot."""
    def __init__(self, xplorer: Xplorer, logger: Optional[logging.Logger] = None):
        self.xplorer = xplorer
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.running = False
        self.bot_started = False
        self.start_event = asyncio.Event()

    async def start(self):
        """Start the UI and display bot status."""
        self.running = True
        try:
            while self.running:
                await self._display_header()
                await self._display_status()
                await self._display_menu()
                await self._handle_input()
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            self.running = False
            if self.bot_started:
                await self.xplorer.stop()

    async def _display_header(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        print("=" * 50)
        print("🚀 0xplorer MEV Bot")
        print("=" * 50)

    async def _display_status(self):
        try:
            if not self.bot_started:
                print("\n⚠️ Bot not initialized. Select 'Start Bot' to begin.")
                return
                
            balance = await self.xplorer.web3.eth.get_balance(self.xplorer.account.address)
            balance_eth = self.xplorer.web3.from_wei(balance, 'ether')
            print(f"\n📊 Status:")
            print(f"   Wallet: {self.xplorer.account.address[:10]}...{self.xplorer.account.address[-8:]}")
            print(f"   Balance: {balance_eth:.4f} ETH")
            print(f"   Connection: {'✅ Connected' if await self.xplorer.web3.is_connected() else '❌ Disconnected'}")
            print(f"   Monitoring: {'✅ Active' if self.xplorer.components['monitor_array'].running else '❌ Inactive'}")
        except Exception as e:
            self.logger.error(f"Error displaying status: {e}")

    async def _display_menu(self):
        print("\n🔧 Commands:")
        if not self.bot_started:
            print("1. Start Bot")
            print("2. Exit")
        else:
            print("1. Start/Stop Monitoring")
            print("2. View Performance")
            print("3. View Current Settings")
            print("4. Exit")
        print("\nEnter command number: ", end='', flush=True)

    async def _handle_input(self):
        try:
            choice = input()
            if not self.bot_started:
                if choice == "1":
                    self.start_event.set()
                    self.bot_started = True
                    print("Initializing bot...")
                    await asyncio.sleep(2)
                elif choice == "2":
                    self.running = False
                else:
                    print("Invalid choice. Please try again.")
                    await asyncio.sleep(2)
            else:
                if choice == "1":
                    await self._toggle_monitoring()
                elif choice == "2":
                    await self._display_performance()
                elif choice == "3":
                    await self._display_settings()
                elif choice == "4":
                    self.running = False
                    await self.xplorer.stop()
                else:
                    print("Invalid choice. Please try again.")
                    await asyncio.sleep(2)
        except Exception as e:
            self.logger.error(f"Error handling input: {e}")

    async def _toggle_monitoring(self):
        try:
            monitor = self.xplorer.components['monitor_array']
            if monitor.running:
                await monitor.stop_monitoring()
                print("Monitoring stopped.")
            else:
                await monitor.start_monitoring()
                print("Monitoring started.")
            await asyncio.sleep(2)
        except Exception as e:
            self.logger.error(f"Error toggling monitoring: {e}")

    async def _display_performance(self):
        try:
            strategy = self.xplorer.components['strategy_manager']
            performance = strategy.strategy_performance
            
            print("\n📈 Performance Statistics:")
            for strategy_type, metrics in performance.items():
                print(f"\n{strategy_type.upper()}:")
                print(f"   Success Rate: {metrics['success_rate']:.2%}")
                print(f"   Total Profit: {metrics['profit']:.6f} ETH")
                print(f"   Total Executions: {metrics['total_executions']}")
            
            input("\nPress Enter to continue...")
        except Exception as e:
            self.logger.error(f"Error displaying performance: {e}")

    async def _display_settings(self):
        try:
            print("\n⚙️ Current Settings:")
            print(f"HTTP Endpoint: {self.xplorer.config.HTTP_ENDPOINT}")
            print(f"Wallet Address: {self.xplorer.config.WALLET_ADDRESS}")
            print(f"Number of Monitored Tokens: {len(await self.xplorer.config.get_token_addresses())}")
            
            strategy = self.xplorer.components['strategy_manager']
            print(f"\nStrategy Settings:")
            print(f"Min Profit Threshold: {strategy.config['min_profit_threshold']} ETH")
            print(f"Learning Rate: {strategy.config['learning_rate']}")
            print(f"Exploration Rate: {strategy.config['exploration_rate']}")
            
            input("\nPress Enter to continue...")
        except Exception as e:
            self.logger.error(f"Error displaying settings: {e}")
