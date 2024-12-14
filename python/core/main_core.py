
import asyncio
import json
import signal
import sys
import time
import tracemalloc
from typing import Any, Dict, List, Optional, Union, Tuple
from eth_account import Account
from web3 import AsyncWeb3
from web3.eth import AsyncEth
from web3.middleware import ExtraDataToPOAMiddleware
from web3.providers import AsyncHTTPProvider, AsyncIPCProvider, WebSocketProvider
import async_timeout
import logging

from analysis.market_monitor import Market_Monitor
from analysis.mempool_monitor import Mempool_Monitor
from analysis.safety_net import Safety_Net
from analysis.strategy_net import Strategy_Net
from configuration.api_config import API_Config
from configuration.configuration import Configuration
from core.nonce_core import Nonce_Core
from core.transaction_core import Transaction_Core

logger = logging.getLogger(__name__)

class Main_Core:
    """
    Builds and manages the entire MEV bot, initializing all components,
    managing connections, and orchestrating the main execution loop.
    """

    def __init__(self, configuration: Configuration) -> None:
        # Take first memory snapshot after initialization
        self.memory_snapshot = tracemalloc.take_snapshot()
        self.configuration = configuration
        self.web3: Optional[AsyncWeb3] = None
        self.account: Optional[Account] = None
        self.running: bool = False
        self.components: Dict[str, Any] = {
            'api_config': None,
            'nonce_core': None, 
            'safety_net': None,
            'market_monitor': None,
            'mempool_monitor': None,
            'transaction_core': None,
            'strategy_net': None,
        }
        logger.info("Starting 0xBuilder...")

    async def initialize(self) -> None:
        """Initialize all components with error handling."""
        try:
            # Take memory snapshot before initialization
            before_snapshot = tracemalloc.take_snapshot()
            
            await self._load_configuration()
            self.web3 = await self._initialize_web3()
            if not self.web3:
                raise RuntimeError("Failed to initialize Web3 connection")

            self.account = Account.from_key(self.configuration.WALLET_KEY)
            await self._check_account_balance()
            await self._initialize_components()

            # Take memory snapshot after initialization and compare
            after_snapshot = tracemalloc.take_snapshot()
            top_stats = after_snapshot.compare_to(before_snapshot, 'lineno')
            
            logger.debug("Memory allocation during initialization:")
            for stat in top_stats[:3]:  # Show top 3 memory changes
                logger.debug(str(stat))

            logger.debug("Main Core initialization successful.")
            
        except Exception as e:
            logger.critical(f"Main Core initialization failed: {e}")
            raise

    async def _load_configuration(self) -> None:
        """Load all configuration elements in the correct order."""
        try:
            # First load the configuration itself
            await self.configuration.load()
            
            logger.debug("Configuration loaded ✅ ")
        except Exception as e:
            logger.critical(f"Failed to load configuration: {e}")
            raise

    async def _initialize_web3(self) -> Optional[AsyncWeb3]:
        """Initialize Web3 connection with error handling and retries."""
        MAX_RETRIES = 3
        RETRY_DELAY = 2

        providers = await self._get_providers()
        if not providers:
            logger.error("No valid endpoints provided!")
            return None

        for provider_name, provider in providers:
            for attempt in range(MAX_RETRIES):
                try:
                    logger.debug(f"Attempting connection with {provider_name} (attempt {attempt + 1})...")
                    web3 = AsyncWeb3(provider, modules={"eth": (AsyncEth,)})
                    
                    # Test connection with timeout
                    try:
                        async with async_timeout.timeout(10):
                            if await web3.is_connected():
                                chain_id = await web3.eth.chain_id
                                logger.debug(f"Connected to network via {provider_name} (Chain ID: {chain_id})")
                                await self._add_middleware(web3)
                                return web3
                    except asyncio.TimeoutError:
                        logger.warning(f"Connection timeout with {provider_name}")
                        continue
                        
                except Exception as e:
                    logger.warning(f"{provider_name} connection attempt {attempt + 1} failed: {e}")
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                    continue

            logger.error(f"All attempts failed for {provider_name}")
            
        logger.error("Failed to initialize Web3 with any provider")
        return None

    async def _get_providers(self) -> List[Tuple[str, Union[AsyncIPCProvider, AsyncHTTPProvider, WebSocketProvider]]]:
        """Get list of available providers with validation."""
        providers = []
    
        if self.configuration.HTTP_ENDPOINT:
            try:
                http_provider = AsyncHTTPProvider(self.configuration.HTTP_ENDPOINT)
                await http_provider.make_request('eth_blockNumber', [])
                providers.append(("HTTP Provider", http_provider))
                logger.info("Linked to Ethereum network via HTTP Provider. ✅")
                return providers
            except Exception as e:
                logger.warning(f"HTTP Provider failed. {e} ❌ - Attempting WebSocket... ")
    
        if self.configuration.WEBSOCKET_ENDPOINT:
            try:
                ws_provider = WebSocketProvider(self.configuration.WEBSOCKET_ENDPOINT)
                await ws_provider.connect()
                try:
                    await ws_provider.make_request('eth_blockNumber', [])
                    providers.append(("WebSocket Provider", ws_provider))
                    logger.info("Linked to Ethereum network via WebSocket Provider. ✅")
                    return providers
                except Exception as e:
                    await ws_provider.disconnect()
                    raise
            except Exception as e:
                logger.warning(f"WebSocket Provider failed {e} ❌ - Attempting IPC... ")
    
        if self.configuration.IPC_ENDPOINT:
            try:
                ipc_provider = AsyncIPCProvider(self.configuration.IPC_ENDPOINT)
                await ipc_provider.make_request('eth_blockNumber', [])
                providers.append(("IPC Provider", ipc_provider))
                logger.info("Linked to Ethereum network via IPC Provider. ✅")
                return providers
            except Exception as e:
                logger.warning(f"IPC Provider failed: {e} ❌ All providers failed.")
        logger.critical("No more providers are available! ❌")
        return providers
    

    async def _test_connection(self, web3: AsyncWeb3, name: str) -> bool:
        """Test Web3 connection with retries."""
        for attempt in range(3):
            try:
                if await web3.is_connected():
                    chain_id = await web3.eth.chain_id
                    logger.debug(f"Connected to network {name} (Chain ID: {chain_id}) ")
                    return True
            except Exception as e:
                logger.error(f"Connection attempt {attempt + 1} failed: {e}")
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
                logger.debug(f"ETH network.")
                pass
            else:
                logger.warning(f"Unknown network; no middleware injected.")
        except Exception as e:
            logger.error(f"Middleware configuration failed: {e}")
            raise

    async def _check_account_balance(self) -> None:
        """Check the Ethereum account balancer_router_abi."""
        try:
            if not self.account:
                raise ValueError("Account not initialized")

            balancer_router_abi = await self.web3.eth.get_balance(self.account.address)
            balance_eth = self.web3.from_wei(balancer_router_abi, 'ether')

            logger.debug(f"Account {self.account.address} initialized ")
            logger.debug(f"Balance: {balance_eth:.4f} ETH")

            if balance_eth < 0.01:
                logger.warning(f"Low account balance (<0.01 ETH)")

        except Exception as e:
            logger.error(f"Balance check failed: {e}")
            raise

    async def _initialize_components(self) -> None:
        """Initialize all bot components with  error handling."""
        try:
            # Initialize core components
            try:
                self.components['nonce_core'] = Nonce_Core(
                    self.web3, self.account.address, self.configuration
                )
                await self.components['nonce_core'].initialize()
            except Exception as e:
                logger.error(f"Failed to initialize nonce core: {e}")
                raise

            api_config = API_Config(self.configuration)

            self.components['safety_net'] = Safety_Net(
                self.web3, self.configuration, self.account, api_config
            )

            # Load contract ABIs
            erc20_abi = await self._load_abi(self.configuration.ERC20_ABI)
            AAVE_FLASHLOAN_ABI = await self._load_abi(self.configuration.AAVE_FLASHLOAN_ABI)
            AAVE_LENDING_POOL_ABI = await self._load_abi(self.configuration.AAVE_LENDING_POOL_ABI)

            # Initialize analysis components
            self.components['market_monitor'] = Market_Monitor(
                self.web3, self.configuration, api_config
            )

            # Initialize monitoring components
            self.components['mempool_monitor'] = Mempool_Monitor( 
                web3=self.web3,
                safety_net=self.components['safety_net'],
                nonce_core=self.components['nonce_core'],
                api_config=api_config,
                monitored_tokens=await self.configuration.get_token_addresses(),
                erc20_abi=erc20_abi,
                configuration=self.configuration
                
            )

            # Initialize transaction components
            self.components['transaction_core'] = Transaction_Core(
                web3=self.web3,
                account=self.account,
                AAVE_FLASHLOAN_ADDRESS=self.configuration.AAVE_FLASHLOAN_ADDRESS,
                AAVE_FLASHLOAN_ABI=AAVE_FLASHLOAN_ABI,
                AAVE_LENDING_POOL_ADDRESS=self.configuration.AAVE_LENDING_POOL_ADDRESS,
                AAVE_LENDING_POOL_ABI=AAVE_LENDING_POOL_ABI,
                monitor=self.components['mempool_monitor'],
                nonce_core=self.components['nonce_core'],
                safety_net=self.components['safety_net'],
                api_config=api_config,
                configuration=self.configuration,
                
                erc20_abi=erc20_abi
            )
            await self.components['transaction_core'].initialize()

            # Initialize strategy components
            self.components['strategy_net'] = Strategy_Net(
                transaction_core=self.components['transaction_core'],
                market_monitor=self.components['market_monitor'],
                safety_net=self.components['safety_net'],
                api_config=api_config,
                
            )

        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            raise

    async def run(self) -> None:
        """Main execution loop with improved error handling and memory tracking."""
        logger.debug("Starting 0xBuilder...")
        self.running = True

        try:
            if not self.components['mempool_monitor']:
                raise RuntimeError("Mempool monitor not properly initialized")

            # Take initial memory snapshot
            initial_snapshot = tracemalloc.take_snapshot()
            last_memory_check = time.time()
            MEMORY_CHECK_INTERVAL = 300  # Check memory every 5 minutes

            monitoring_task = asyncio.create_task(
                self.components['mempool_monitor'].start_monitoring()
            )

            while self.running:
                try:
                    await self._process_profitable_transactions()
                    
                    # Periodic memory checks
                    current_time = time.time()
                    if current_time - last_memory_check > MEMORY_CHECK_INTERVAL:
                        current_snapshot = tracemalloc.take_snapshot()
                        top_stats = current_snapshot.compare_to(initial_snapshot, 'lineno')
                        
                        logger.debug("Memory allocation changes:")
                        for stat in top_stats[:3]:  # Show top 3 memory changes
                            logger.debug(str(stat))
                            
                        last_memory_check = current_time

                    await asyncio.sleep(1)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    await asyncio.sleep(5)

        except Exception as e:
            logger.error(f"Fatal error in run loop: {e}")
        finally:
            if 'monitoring_task' in locals():
                monitoring_task.cancel()
                try:
                    await monitoring_task
                except asyncio.CancelledError:
                    pass

    async def stop(self) -> None:
        """Gracefully stop all components with final memory snapshot."""
        logger.warning("Shutting down Core...")
        self.running = False

        try:
            # Take final memory snapshot and compare with initial
            final_snapshot = tracemalloc.take_snapshot()
            top_stats = final_snapshot.compare_to(self.memory_snapshot, 'lineno')
            
            logger.debug("Final memory allocation changes:")
            for stat in top_stats[:5]:  # Show top 5 memory changes
                logger.debug(str(stat))

            # Define component shutdown order
            shutdown_order = [
                'mempool_monitor',  # Stop monitoring first
                'strategy_net',     # Stop strategies
                'transaction_core', # Stop transactions
                'market_monitor',   # Stop market monitoring
                'safety_net',      # Stop safety checks
                'nonce_core',      # Stop nonce management
                'api_config'       # Stop API connections last
            ]

            try:
                for component_name in shutdown_order:
                    component = self.components.get(component_name)
                    if component:
                        try:
                            await component.stop()
                            logger.debug(f"Stopped {component_name}")
                        except Exception as e:
                            logger.error(f"Error stopping {component_name}: {e}")

                # Clean up web3 connection
                if self.web3:
                    try:
                        await self.web3.provider.disconnect()
                        self.web3 = None
                    except Exception as e:
                        logger.error(f"Error disconnecting web3: {e}")

                logger.debug("Core shutdown complete.")
                sys.exit(0)
            except Exception as e:
                logger.error(f"Error during shutdown: {e}")

            # Stop tracemalloc
            tracemalloc.stop()
            logger.debug("Core shutdown complete.")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            sys.exit("forced exit")

    async def _process_profitable_transactions(self) -> None:
        """Process profitable transactions from the queue."""
        strategy = self.components['strategy_net']
        monitor = self.components['mempool_monitor']
        
        while not monitor.profitable_transactions.empty():
            try:
                try:
                    tx = await asyncio.wait_for(monitor.profitable_transactions.get(), timeout=1.0)
                    tx_hash = tx.get('tx_hash', 'Unknown')
                    strategy_type = tx.get('strategy_type', 'Unknown')
                except asyncio.TimeoutError:
                    continue
                
                logger.debug(f"Processing transaction {tx_hash} with strategy type {strategy_type}")
                success = await strategy.execute_best_strategy(tx, strategy_type)

                if success:
                    logger.debug(f"Strategy execution successful for tx: {tx_hash}")
                else:
                    logger.warning(f"Strategy execution failed for tx: {tx_hash}")

                # Mark task as done
                monitor.profitable_transactions.task_done()

            except Exception as e:
                logger.error(f"Error processing transaction: {e}")

    async def _load_abi(self, abi_path: str) -> List[Dict[str, Any]]:
        """Load contract abi from a file."""
        try:
            with open(abi_path, 'r') as file:
                return json.load(file)
        except Exception as e:
            logger.error(f"Failed to load ABI from {abi_path}: {e}")
        except Exception as e:
            return []


# Modify the main function for better signal handling
async def main():
    """Main entry point with comprehensive setup and error handling."""
    # Log initial memory statistics
    logger.debug(f"Tracemalloc status: {tracemalloc.is_tracing()}")
    logger.debug(f"Initial traced memory: {tracemalloc.get_traced_memory()}")
    
    configuration = Configuration()
    core = Main_Core(configuration)
    
    def signal_handler():
        logger.debug("Shutdown signal received")
        if not core.running:
            return
        asyncio.create_task(core.stop())

    try:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, signal_handler)

        await core.initialize()
        await core.run()
    except Exception as e:
        logger.critical(f"An unexpected error occurred: {str(e)}")
    finally:
        # Remove signal handlers
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(sig)
        await core.stop()
        
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass  # Handle KeyboardInterrupt silently as it's already handled by signal handlers
    except Exception as e:
        # Get current memory snapshot on error
        snapshot = tracemalloc.take_snapshot()
        logger.critical(f"Program terminated with an error: {e}")
        logger.debug("Top 10 memory allocations at error:")
        top_stats = snapshot.statistics('lineno')
        for stat in top_stats[:10]:
            logger.debug(str(stat))