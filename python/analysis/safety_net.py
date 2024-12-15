import asyncio
import time
from decimal import Decimal
from cachetools import TTLCache
from typing import Any, Dict, Optional, Tuple
from web3 import AsyncWeb3
from eth_account import Account
import logging

from configuration.api_config import API_Config
from configuration.configuration import Configuration

logger = logging.getLogger(__name__)

# Add risk thresholds
RISK_THRESHOLDS = {
    'gas_price': 500,  # Gwei
    'min_profit': 0.01,  # ETH
    'max_slippage': 0.05,  # 5%
    'congestion': 0.8  # 80%
}

class Safety_Net:
    """
    Enhanced safety system for risk management and transaction validation.
    """

    CACHE_TTL = 300  # Cache TTL in seconds
    GAS_PRICE_CACHE_TTL = 15  # 15 sec cache for gas prices

    SLIPPAGE_CONFIG = {
        "default": 0.1,
        "min": 0.01,
        "max": 0.5,
        "high_congestion": 0.05,
        "low_congestion": 0.2,
    }

    GAS_CONFIG = {
        "max_gas_price_gwei": 500,
        "min_profit_multiplier": 2.0,
        "base_gas_limit": 21000,
    }

    def __init__(
        self,
        web3: AsyncWeb3,
        configuration: Optional[Configuration] = None,
        address: Optional[str] = None,
        account: Optional[Account] = None,
        api_config: Optional[API_Config] = None,
    ):
        self.web3 = web3
        self.address = address
        self.configuration = configuration
        self.account = account
        self.api_config = api_config
        self.price_cache = TTLCache(maxsize=1000, ttl=self.CACHE_TTL)
        self.gas_price_cache = TTLCache(maxsize=1, ttl=self.GAS_PRICE_CACHE_TTL)

        self.price_lock = asyncio.Lock()
        logger.info("SafetyNet is reporting for duty 🛡️")
        time.sleep(3) # ensuring proper initialization

        # Add safety checks cache
        self.safety_cache = TTLCache(maxsize=100, ttl=60)  # 1 minute cache

    async def get_balance(self, account: Any) -> Decimal:
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

    async def get_dynamic_gas_price(self) -> Decimal:
        """Get the current gas price dynamically with fallback."""
        if "gas_price" in self.gas_price_cache:
            return self.gas_price_cache["gas_price"]
        try:
            if not self.web3:
                logger.error("Web3 not initialized in Safety_Net")
                return Decimal("50")  # Default fallback gas price in Gwei
                
            gas_price = await self.web3.eth.gas_price
            if gas_price is None:
                logger.warning("Failed to get gas price, using fallback")
                return Decimal("50")  # Fallback gas price
                
            gas_price_decimal = Decimal(gas_price) / Decimal(10**9)  # Convert to Gwei
            self.gas_price_cache["gas_price"] = gas_price_decimal
            return gas_price_decimal
        except Exception as e:
            logger.error(f"Failed to get dynamic gas price: {e}")
            return Decimal("50")  # Fallback gas price on error

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
            results = {
                'is_safe': True,
                'gas_ok': True,
                'profit_ok': True,
                'slippage_ok': True,
                'congestion_ok': True,
                'messages': []
            }

            # Check gas price
            if check_type in ['all', 'gas']:
                gas_price = await self.get_dynamic_gas_price()
                if gas_price > RISK_THRESHOLDS['gas_price']:
                    results['gas_ok'] = False
                    results['messages'].append(f"Gas price too high: {gas_price} Gwei")

            # Check profit potential
            if check_type in ['all', 'profit']:
                profit = await self._calculate_profit(tx_data)
                if profit < RISK_THRESHOLDS['min_profit']:
                    results['profit_ok'] = False
                    results['messages'].append(f"Insufficient profit: {profit} ETH")

            # Check network congestion
            if check_type in ['all', 'network']:
                congestion = await self.get_network_congestion()
                if congestion > RISK_THRESHOLDS['congestion']:
                    results['congestion_ok'] = False
                    results['messages'].append(f"High network congestion: {congestion:.1%}")

            results['is_safe'] = all([
                results['gas_ok'],
                results['profit_ok'],
                results['slippage_ok'],
                results['congestion_ok']
            ])

            return results['is_safe'], results

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
             logger.error(f"Error stopping safety net: {e}")
             raise

#//////////////////////////////////////////////////////////////////////////////