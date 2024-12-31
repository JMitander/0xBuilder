import asyncio
import logging
import time
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple, Union
from cachetools import TTLCache
from web3 import AsyncWeb3
from eth_account import Account

from api_config import API_Config
from configuration import Configuration

logger = logging.getLogger("0xBuilder")

class Safety_Net:
    """
    Enhanced safety system for risk management and transaction validation.
    """

    CACHE_TTL: int = 300  # Cache TTL in seconds
    GAS_PRICE_CACHE_TTL: int = 15  # 15 sec cache for gas prices

    def __init__(
        self,
        web3: AsyncWeb3,
        configuration: Optional["Configuration"] = None,
        address: Optional[str] = None,
        account: Optional[Account] = None,
        api_config: Optional["API_Config"] = None,
        market_monitor: Optional[Any] = None,  # Add this parameter
    ):
        """
        Initialize Safety Net components.

        Args:
            web3: An AsyncWeb3 instance.
            configuration: Configuration object containing settings.
            address: Address for the transactions.
            account: Account associated with the address.
            api_config: API configuration instance.
            market_monitor: Market monitor instance.
        """
        self.web3: AsyncWeb3 = web3
        self.address: Optional[str] = address
        self.configuration: Optional["Configuration"] = configuration
        self.account: Optional[Account] = account
        self.api_config: Optional["API_Config"] = api_config
        self.price_cache: TTLCache = TTLCache(maxsize=1000, ttl=self.CACHE_TTL)
        self.gas_price_cache: TTLCache = TTLCache(maxsize=1, ttl=self.GAS_PRICE_CACHE_TTL)
        self.market_monitor: Optional[Any] = market_monitor  # Store market_monitor reference

        self.price_lock: asyncio.Lock = asyncio.Lock()
        logger.info("SafetyNet is reporting for duty 🛡️")
        time.sleep(1) # ensuring proper initialization

        # Add safety checks cache
        self.safety_cache: TTLCache = TTLCache(maxsize=100, ttl=60)  # 1 minute cache

        # Load settings from config object
        if self.configuration:
            self.SLIPPAGE_CONFIG: Dict[str, float] = {
                "default": self.configuration.get_config_value("SLIPPAGE_DEFAULT", 0.1),
                "min": self.configuration.get_config_value("SLIPPAGE_MIN", 0.01),
                "max": self.configuration.get_config_value("SLIPPAGE_MAX", 0.5),
                "high_congestion": self.configuration.get_config_value("SLIPPAGE_HIGH_CONGESTION", 0.05),
                "low_congestion": self.configuration.get_config_value("SLIPPAGE_LOW_CONGESTION", 0.2),
            }
            self.GAS_CONFIG: Dict[str, Union[int, float]] = {
                "max_gas_price_gwei": self.configuration.get_config_value("MAX_GAS_PRICE_GWEI", 500),
                "min_profit_multiplier": self.configuration.get_config_value("MIN_PROFIT_MULTIPLIER", 2.0),
                "base_gas_limit": self.configuration.get_config_value("BASE_GAS_LIMIT", 21000)
            }
        else: #Defaults for testing when config class is not available.
             self.SLIPPAGE_CONFIG: Dict[str, float] = {
                "default":  0.1,
                "min": 0.01,
                "max": 0.5,
                "high_congestion":  0.05,
                "low_congestion": 0.2,
            }
             self.GAS_CONFIG: Dict[str, Union[int, float]] = {
                "max_gas_price_gwei":  500,
                "min_profit_multiplier": 2.0,
                "base_gas_limit":  21000
            }
        

    async def initialize(self) -> None:
        """Initialize Safety Net components."""
        try:
            # Initialize price cache
            self.price_cache = TTLCache(maxsize=1000, ttl=self.CACHE_TTL)
            
            # Initialize gas price cache
            self.gas_price_cache = TTLCache(maxsize=1, ttl=self.GAS_PRICE_CACHE_TTL)
            
            # Initialize safety checks cache
            self.safety_cache = TTLCache(maxsize=100, ttl=60)
            
            # Verify web3 connection
            if not self.web3:
                raise RuntimeError("Web3 not initialized in Safety_Net")
                
            # Test connection
            if not await self.web3.is_connected():
                raise RuntimeError("Web3 connection failed in Safety_Net")

            logger.info("SafetyNet initialized successfully ✅")
        except Exception as e:
            logger.critical(f"Safety Net initialization failed: {e}")
            raise

    async def get_balance(self, account: Account) -> Decimal:
        """Get account balance with retries and caching."""
        cache_key = f"balance_{account.address}"
        if cache_key in self.price_cache:
            logger.debug("Balance fetched from cache.")
            return self.price_cache[cache_key]

        for attempt in range(3):
            try:
                balance = Decimal(await self.web3.eth.get_balance(account.address)) / Decimal("1e18")
                self.price_cache[cache_key] = balance
                logger.debug(f"Fetched balance: {balance} ETH")
                return balance
            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed to fetch balance: {e}")
                await asyncio.sleep(2 ** attempt)
        logger.error("Failed to fetch account balance after retries.")
        return Decimal("0")

    async def ensure_profit(
        self,
        transaction_data: Dict[str, Any],
        minimum_profit_eth: Optional[float] = None,
    ) -> bool:
        """Enhanced profit verification with dynamic thresholds and risk assessment."""
        try:
            real_time_price = await self.api_config.get_real_time_price(transaction_data['output_token'])
            if real_time_price is None:
                logger.error("Real-time price unavailable.")
                return False

            gas_cost_eth = self._calculate_gas_cost(
                Decimal(transaction_data["gas_price"]),
                transaction_data["gas_used"]
            )

            slippage = await self.adjust_slippage_tolerance()
            profit = await self._calculate_profit(
                transaction_data, real_time_price, slippage, gas_cost_eth
            )

            self._log_profit_calculation(transaction_data, real_time_price, gas_cost_eth, profit, minimum_profit_eth or 0.001)

            return profit > Decimal(minimum_profit_eth or 0.001)
        except KeyError as e:
            logger.error(f"Missing key in transaction data: {e}")
            return False
        except Exception as e:
            logger.error(f"Error in ensure_profit: {e}")
            return False

    def _validate_gas_parameters(self, gas_price_gwei: Decimal, gas_used: int) -> bool:
        """Validate gas parameters against safety thresholds."""
        if gas_used == 0:
            logger.error("Gas used cannot be zero.")
            return False
        if gas_price_gwei > self.GAS_CONFIG["max_gas_price_gwei"]:
            logger.warning(f"Gas price {gas_price_gwei} Gwei exceeds maximum threshold.")
            return False
        return True

    def _calculate_gas_cost(self, gas_price_gwei: Decimal, gas_used: int) -> Decimal:
        """Calculate total gas cost in ETH."""
        return gas_price_gwei * Decimal(gas_used) * Decimal("1e-9")

    async def _calculate_profit(
        self,
        transaction_data: Dict[str, Any],
        real_time_price: Decimal,
        slippage: float,
        gas_cost_eth: Decimal,
    ) -> Decimal:
        """Calculate expected profit considering slippage and gas costs."""
        expected_output = real_time_price * Decimal(transaction_data["amountOut"])
        input_amount = Decimal(transaction_data["amountIn"])
        slippage_adjusted_output = expected_output * (1 - Decimal(slippage))
        return slippage_adjusted_output - input_amount - gas_cost_eth

    def _log_profit_calculation(
        self,
        transaction_data: Dict[str, Any],
        real_time_price: Decimal,
        gas_cost_eth: Decimal,
        profit: Decimal,
        minimum_profit_eth: float,
    ) -> None:
        """Log detailed profit calculation metrics."""
        profitable = "Yes" if profit > Decimal(minimum_profit_eth) else "No"
        logger.debug(
            f"Profit Calculation Summary:\n"
            f"Token: {transaction_data['output_token']}\n"
            f"Real-time Price: {real_time_price:.6f} ETH\n"
            f"Input Amount: {transaction_data['amountIn']:.6f} ETH\n"
            f"Expected Output: {transaction_data['amountOut']:.6f} tokens\n"
            f"Gas Cost: {gas_cost_eth:.6f} ETH\n"
            f"Calculated Profit: {profit:.6f} ETH\n"
            f"Minimum Required: {minimum_profit_eth} ETH\n"
            f"Profitable: {profitable}"
        )

    async def estimate_gas(self, transaction_data: Dict[str, Any]) -> int:
        """Estimate the gas required for a transaction."""
        try:
            gas_estimate = await self.web3.eth.estimate_gas(transaction_data)
            return gas_estimate
        except Exception as e:
            logger.error(f"Gas estimation failed: {e}")
            return 0

    async def adjust_slippage_tolerance(self) -> float:
        """Adjust slippage tolerance based on network conditions."""
        try:
            congestion_level = await self.get_network_congestion()
            if congestion_level > 0.8:
                slippage = self.SLIPPAGE_CONFIG["high_congestion"]
            elif congestion_level < 0.2:
                slippage = self.SLIPPAGE_CONFIG["low_congestion"]
            else:
                slippage = self.SLIPPAGE_CONFIG["default"]
            slippage = min(
                max(slippage, self.SLIPPAGE_CONFIG["min"]), self.SLIPPAGE_CONFIG["max"]
            )
            logger.debug(f"Adjusted slippage tolerance to {slippage * 100}%")
            return slippage
        except Exception as e:
            logger.error(f"Error adjusting slippage tolerance: {e}")
            return self.SLIPPAGE_CONFIG["default"]

    async def get_network_congestion(self) -> float:
        """Estimate the current network congestion level."""
        try:
            latest_block = await self.web3.eth.get_block("latest")
            gas_used = latest_block["gasUsed"]
            gas_limit = latest_block["gasLimit"]
            congestion_level = gas_used / gas_limit
            logger.debug(f"Network congestion level: {congestion_level * 100}%")
            return congestion_level
        except Exception as e:
            logger.error(f"Error fetching network congestion: {e}")

    async def check_transaction_safety(
        self,
        tx_data: Dict[str, Any],
        check_type: str = 'all'
    ) -> Tuple[bool, Dict[str, Any]]:
        """Unified safety check method for transactions."""
        try:
             is_safe = True
             messages = []
             
             if check_type in ['all', 'gas']:
                gas_price = await self.get_dynamic_gas_price()
                if gas_price > self.configuration.MAX_GAS_PRICE_GWEI:
                     is_safe = False
                     messages.append(f"Gas price too high: {gas_price} Gwei")

             # Check profit potential
             if check_type in ['all', 'profit']:
                profit = await self._calculate_profit(
                    tx_data,
                    await self.api_config.get_real_time_price(tx_data['output_token']),
                    await self.adjust_slippage_tolerance(),
                    self._calculate_gas_cost(
                        Decimal(tx_data['gas_price']),
                        tx_data['gas_used']
                    )
                )
                if profit < self.configuration.MIN_PROFIT:
                     is_safe = False
                     messages.append(f"Insufficient profit: {profit} ETH")

             # Check network congestion
             if check_type in ['all', 'network']:
                congestion = await self.get_network_congestion()
                if congestion > 0.8:
                     is_safe = False
                     messages.append(f"High network congestion: {congestion:.1%}")

             return is_safe, {
                'is_safe': is_safe,
                'gas_ok': is_safe if check_type not in ['all', 'gas'] else gas_price <= self.configuration.MAX_GAS_PRICE_GWEI,
                'profit_ok': is_safe if check_type not in ['all', 'profit'] else profit >= self.configuration.MIN_PROFIT,
                'slippage_ok': True, # Not yet implemented slippage checks
                'congestion_ok': is_safe if check_type not in ['all', 'network'] else congestion <= 0.8,
                'messages': messages
            }

        except Exception as e:
            logger.error(f"Safety check error: {e}")
            return False, {'is_safe': False, 'messages': [str(e)]}

    async def stop(self) -> None:
         """Stops the 0xBuilder gracefully."""
         try:
            if self.api_config:
                await self.api_config.close()
            logger.debug("Safety Net stopped successfully.")
         except Exception as e:
             logger.error(f"Error stopping safety net: {e}")
             raise

    async def assess_transaction_risk(
        self,
        tx: Dict[str, Any],
        token_symbol: str,
        market_conditions: Optional[Dict[str, bool]] = None,
        price_change: float = 0,
        volume: float = 0
    ) -> Tuple[float, Dict[str, Any]]:
        """Centralized risk assessment with proper error handling."""
        try:
            risk_score = 1.0
            
            # Get market conditions if not provided and market_monitor exists
            if not market_conditions and self.market_monitor:
                market_conditions = await self.market_monitor.check_market_conditions(tx.get("to", ""))
            elif not market_conditions:
                market_conditions = {}  # Default empty if no market_monitor
                
            # Gas price impact
            gas_price = int(tx.get("gasPrice", 0))
            gas_price_gwei = float(self.web3.from_wei(gas_price, "gwei"))
            if gas_price_gwei > self.configuration.MAX_GAS_PRICE_GWEI:
                risk_score *= 0.7
                
            # Market conditions impact
            if market_conditions.get("high_volatility", False):
                risk_score *= 0.7
            if market_conditions.get("low_liquidity", False):
                risk_score *= 0.6
            if market_conditions.get("bullish_trend", False):
                risk_score *= 1.2
                
            # Price change impact    
            if price_change > 0:
                risk_score *= min(1.3, 1 + (price_change / 100))
                
            # Volume impact
            if volume >= 1_000_000:
                risk_score *= 1.2
            elif volume <= 100_000:
                risk_score *= 0.8
                
            return risk_score, market_conditions                    
        
        except Exception as e:
            logger.error(f"Error in risk assessment: {e}")
            return 0.0, {}
        
    async def get_dynamic_gas_price(self) -> Decimal:
        """
        Fetch dynamic gas price with caching.

        Returns:
            Decimal: Dynamic gas price in Gwei.
        """
        if self.gas_price_cache and self.gas_price_cache.get("gas_price"):
            return self.gas_price_cache["gas_price"]

        try:
            #Fetch gas price from the latest block
             latest_block = await self.web3.eth.get_block('latest')
             base_fee = latest_block.get("baseFeePerGas")
             
             if base_fee:
                  gas_price_wei =  base_fee * 2
                  gas_price_gwei = Decimal(self.web3.from_wei(gas_price_wei, 'gwei'))
             else:
                  gas_price_gwei =  Decimal(self.web3.from_wei(await self.web3.eth.gas_price, 'gwei'))


             #Cache
             self.gas_price_cache["gas_price"] = gas_price_gwei
             logger.debug(f"Fetched dynamic gas price: {gas_price_gwei} Gwei")
             return gas_price_gwei

        except Exception as e:
            logger.error(f"Error fetching dynamic gas price: {e}")
            return Decimal(str(self.configuration.get_config_value("MAX_GAS_PRICE_GWEI", 50))) # Default gas price