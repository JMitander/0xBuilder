import asyncio
from asyncio.log import logger
from decimal import Decimal
import random
import time
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import numpy as np

from utils.Javascript.strategyexecutionerror import StrategyExecutionError

# Constants
MIN_PROFIT_THRESHOLD = Decimal("0.01")
DEFAULT_GAS_THRESHOLD = 200  # Gwei
STRATEGY_SCORE_THRESHOLD = 75
BULLISH_THRESHOLD = 0.02
HIGH_VOLUME_DEFAULT = 500_000  # USD

class Strategy_Net:
    """Advanced strategy network for MEV operations including front-running, back-running, and sandwich attacks."""

    def __init__(
        self,
        transaction_core: Optional[Any],
        market_monitor: Optional[Any],
        safety_net: Optional[Any],
        api_config: Optional[Any],
    ) -> None:
        self.transaction_core = transaction_core
        self.market_monitor = market_monitor
        self.safety_net = safety_net
        self.api_config = api_config

        # Initialize strategy types
        self.strategy_types = [
            "eth_transaction",
            "front_run",
            "back_run",
            "sandwich_attack"
        ]

        # Initialize strategy registry before using it
        self._strategy_registry = {
            "eth_transaction": [self.high_value_eth_transfer],
            "front_run": [
                self.aggressive_front_run,
                self.predictive_front_run,
                self.volatility_front_run,
                self.advanced_front_run,
            ],
            "back_run": [
                self.price_dip_back_run,
                self.flashloan_back_run,
                self.high_volume_back_run,
                self.advanced_back_run,
            ],
            "sandwich_attack": [
                self.flash_profit_sandwich,
                self.price_boost_sandwich,
                self.arbitrage_sandwich,
                self.advanced_sandwich_attack,
            ],
        }

        # Initialize performance metrics after strategy registry
        self.strategy_performance = {
            strategy_type: StrategyPerformanceMetrics()
            for strategy_type in self.strategy_types
        }

        # Initialize reinforcement weights after strategy registry
        self.reinforcement_weights = {
            strategy_type: np.ones(len(self._strategy_registry[strategy_type]))
            for strategy_type in self.strategy_types
        }

        self.configuration = StrategyConfiguration()
        self.history_data = []

        logger.debug("StrategyNet initialized with enhanced configuration")

    async def initialize(self) -> None:
        """Initialize strategy network with performance metrics and reinforcement weights."""
        try:
            # Initialize performance metrics
            self.strategy_performance = {
                strategy_type: StrategyPerformanceMetrics()
                for strategy_type in self.strategy_types
            }
            # Initialize reinforcement weights
            self.reinforcement_weights = {
                strategy_type: np.ones(len(self.get_strategies(strategy_type)))
                for strategy_type in self.strategy_types
            }
            logger.info("StrategyNet initialized ✅")
        except Exception as e:
            logger.critical(f"Strategy Net initialization failed: {e}")
            raise

    def register_strategy(self, strategy_type: str, strategy_func: Callable[[Dict[str, Any]], asyncio.Future]) -> None:
        """Register a new strategy dynamically.""" 
        if strategy_type not in self.strategy_types:
            logger.warning(f"Attempted to register unknown strategy type: {strategy_type}")
            return
        self._strategy_registry[strategy_type].append(strategy_func)
        self.reinforcement_weights[strategy_type] = np.ones(len(self._strategy_registry[strategy_type]))
        logger.debug(f"Registered new strategy '{strategy_func.__name__}' under '{strategy_type}'")

    def get_strategies(self, strategy_type: str) -> List[Callable[[Dict[str, Any]], asyncio.Future]]:
        """Retrieve strategies for a given strategy type.""" 
        return self._strategy_registry.get(strategy_type, [])

    async def execute_best_strategy(
        self, 
        target_tx: Dict[str, Any], 
        strategy_type: str
    ) -> bool:
        """
        Execute the optimal strategy based on current market conditions and historical performance.
        
        Args:
            target_tx: Target transaction details
            strategy_type: Type of strategy to execute
            
        Returns:
            bool: True if execution was successful, False otherwise
        """
        strategies = self.get_strategies(strategy_type)
        if not strategies:
            logger.debug(f"No strategies available for type: {strategy_type}")
            return False

        try:
            start_time = time.time()
            selected_strategy = await self._select_best_strategy(strategies, strategy_type)

            profit_before = await self.transaction_core.get_current_profit()
            success = await selected_strategy(target_tx)
            profit_after = await self.transaction_core.get_current_profit()

            execution_time = time.time() - start_time
            profit_made = profit_after - profit_before

            await self._update_strategy_metrics(
                selected_strategy.__name__,
                strategy_type,
                success,
                profit_made,
                execution_time,
            )

            return success

        except StrategyExecutionError as e:
            logger.error(f"Strategy execution failed: {str(e)}", exc_info=True)
            return False
        except Exception as e:
            logger.exception(f"Unexpected error during strategy execution: {e}")
            return False

    async def _select_best_strategy(
        self, strategies: List[Callable[[Dict[str, Any]], asyncio.Future]], strategy_type: str
    ) -> Callable[[Dict[str, Any]], asyncio.Future]:
        """Select the best strategy based on reinforcement learning weights.""" 
        weights = self.reinforcement_weights[strategy_type]

        if random.random() < self.configuration.exploration_rate:
            logger.debug("Using exploration for strategy selection")
            return random.choice(strategies)

        # Numerical stability for softmax
        max_weight = np.max(weights)
        exp_weights = np.exp(weights - max_weight)
        probabilities = exp_weights / exp_weights.sum()

        selected_index = np.random.choice(len(strategies), p=probabilities)
        selected_strategy = strategies[selected_index]
        logger.debug(f"Selected strategy '{selected_strategy.__name__}' with weight {weights[selected_index]:.4f}")
        return selected_strategy

    async def _update_strategy_metrics(
        self,
        strategy_name: str,
        strategy_type: str,
        success: bool,
        profit: Decimal,
        execution_time: float,
    ) -> None:
        """Update metrics for the executed strategy.""" 
        metrics = self.strategy_performance[strategy_type]
        metrics.total_executions += 1

        if success:
            metrics.successes += 1
            metrics.profit += profit
        else:
            metrics.failures += 1

        metrics.avg_execution_time = (
            metrics.avg_execution_time * self.configuration.decay_factor
            + execution_time * (1 - self.configuration.decay_factor)
        )
        metrics.success_rate = metrics.successes / metrics.total_executions

        strategy_index = self.get_strategy_index(strategy_name, strategy_type)
        if strategy_index >= 0:
            reward = self._calculate_reward(success, profit, execution_time)
            self._update_reinforcement_weight(strategy_type, strategy_index, reward)

        self.history_data.append(
            {
                "timestamp": time.time(),
                "strategy_name": strategy_name,
                "success": success,
                "profit": float(profit),
                "execution_time": execution_time,
                "total_profit": float(metrics.profit),
            }
        )

    def get_strategy_index(self, strategy_name: str, strategy_type: str) -> int:
        """Get the index of a strategy in the strategy list.""" 
        strategies = self.get_strategies(strategy_type)
        for index, strategy in enumerate(strategies):
            if strategy.__name__ == strategy_name:
                return index
        logger.warning(f"Strategy '{strategy_name}' not found in type '{strategy_type}'")
        return -1

    def _calculate_reward(
        self, success: bool, profit: Decimal, execution_time: float
    ) -> float:
        """Calculate the reward for a strategy execution.""" 
        base_reward = float(profit) if success else -0.1
        time_penalty = -0.01 * execution_time
        total_reward = base_reward + time_penalty
        logger.debug(f"Calculated reward: {total_reward:.4f} (Base: {base_reward}, Time Penalty: {time_penalty})")
        return total_reward

    def _update_reinforcement_weight(
        self, strategy_type: str, index: int, reward: float
    ) -> None:
        """Update the reinforcement learning weight for a strategy.""" 
        lr = self.configuration.learning_rate
        current_weight = self.reinforcement_weights[strategy_type][index]
        new_weight = current_weight * (1 - lr) + reward * lr
        self.reinforcement_weights[strategy_type][index] = max(0.1, new_weight)
        logger.debug(f"Updated weight for strategy index {index} in '{strategy_type}': {new_weight:.4f}")

    async def _decode_transaction(self, target_tx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Decode transaction input.""" 
        try:
            decoded = await self.transaction_core.decode_transaction_input(
                target_tx["input"], target_tx["to"]
            )
            logger.debug(f"Decoded transaction: {decoded}")
            return decoded
        except Exception as e:
            logger.error(f"Error decoding transaction: {e}")
            return None

    async def _get_token_symbol(self, token_address: str) -> Optional[str]:
        """Get token symbol from address.""" 
        try:
            symbol = await self.api_config.get_token_symbol(
                self.transaction_core.web3, token_address
            )
            logger.debug(f"Retrieved token symbol '{symbol}' for address '{token_address}'")
            return symbol
        except Exception as e:
            logger.error(f"Error fetching token symbol: {e}")
            return None

    # Consolidate duplicate risk assessment methods into one
    async def _assess_risk(
        self,
        tx: Dict[str, Any],
        token_symbol: str,
        price_change: float = 0,
        volume: float = 0
    ) -> Tuple[float, Dict[str, Any]]:
        """Centralized risk assessment for all strategies."""
        try:
            risk_score = 1.0
            market_conditions = await self.market_monitor.check_market_conditions(tx.get("to", ""))
            
            # Gas price impact 
            gas_price = int(tx.get("gasPrice", 0))
            gas_price_gwei = float(self.transaction_core.web3.from_wei(gas_price, "gwei"))
            if gas_price_gwei > 300:
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

            risk_score = max(0.0, min(1.0, risk_score))
            logger.debug(f"Risk assessment for {token_symbol}: {risk_score:.2f}")
            
            return risk_score, market_conditions
            
        except Exception as e:
            logger.error(f"Error in risk assessment: {e}")
            return 0.0, {}

    # Remove duplicate validation methods and consolidate into one
    async def _validate_transaction(
        self,
        tx: Dict[str, Any],
        strategy_type: str,
        min_value: float = 0
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """Centralized transaction validation for all strategies."""
        try:
            if not isinstance(tx, dict) or not tx:
                logger.debug("Invalid transaction format")
                return False, None, None

            decoded_tx = await self._decode_transaction(tx)
            if not decoded_tx:
                return False, None, None

            # Extract and validate path
            path = decoded_tx.get("params", {}).get("path", [])
            if not path or len(path) < 2:
                logger.debug("Invalid transaction path")
                return False, None, None

            # Get token details
            token_address = path[0] if strategy_type in ["front_run", "sandwich_attack"] else path[-1]
            token_symbol = await self._get_token_symbol(token_address)
            if not token_symbol:
                return False, None, None

            # Validate value if required
            if min_value > 0:
                tx_value = self.transaction_core.web3.from_wei(int(tx.get("value", 0)), "ether")
                if float(tx_value) < min_value:
                    logger.debug(f"Transaction value {tx_value} below minimum {min_value}")
                    return False, None, None

            return True, decoded_tx, token_symbol

        except Exception as e:
            logger.error(f"Transaction validation error: {e}")
            return False, None, None

    # ========================= Strategy Implementations =========================

    async def high_value_eth_transfer(self, target_tx: Dict[str, Any]) -> bool:
        """
        Execute high-value ETH transfer strategy with advanced validation and dynamic thresholds.
        
        :param target_tx: Target transaction dictionary
        :return: True if transaction was executed successfully, else False
        """
        logger.info("Initiating High-Value ETH Transfer Strategy...")

        try:
            # Basic transaction validation
            if not isinstance(target_tx, dict) or not target_tx:
                logger.debug("Invalid transaction format provided!")
                return False

            # Extract transaction details
            eth_value_in_wei = int(target_tx.get("value", 0))
            gas_price = int(target_tx.get("gasPrice", 0))
            to_address = target_tx.get("to", "")

            # Convert values to ETH for readability
            eth_value = self.transaction_core.web3.from_wei(eth_value_in_wei, "ether")
            gas_price_gwei = self.transaction_core.web3.from_wei(gas_price, "gwei")

            # Dynamic threshold based on current gas prices
            base_threshold = self.transaction_core.web3.to_wei(10, "ether")
            if gas_price_gwei > 200:  # High gas price scenario
                threshold = base_threshold * 2  # Double threshold when gas is expensive
            elif gas_price_gwei > 100:
                threshold = base_threshold * 1.5
            else:
                threshold = base_threshold

            # Log detailed transaction analysis
            threshold_eth = self.transaction_core.web3.from_wei(threshold, 'ether')
            logger.debug(
                f"Transaction Analysis:\n"
                f"Value: {eth_value:.4f} ETH\n"
                f"Gas Price: {gas_price_gwei:.2f} Gwei\n"
                f"To Address: {to_address[:10]}...\n"
                f"Current Threshold: {threshold_eth} ETH"
            )

            # Additional validation checks
            if eth_value_in_wei <= 0:
                logger.debug("Transaction value is zero or negative. Skipping...")
                return False

            if not self.transaction_core.web3.is_address(to_address):
                logger.debug("Invalid recipient address. Skipping...")
                return False

            # Check contract interaction
            is_contract = await self._is_contract_address(to_address)
            if is_contract:
                logger.debug("Recipient is a contract. Additional validation required...")
                if not await self._validate_contract_interaction(to_address):
                    return False

            # Execute if value exceeds threshold
            if eth_value_in_wei > threshold:
                logger.debug(
                    f"High-value ETH transfer detected:\n"
                    f"Value: {eth_value:.4f} ETH\n"
                    f"Threshold: {threshold_eth} ETH"
                )
                return await self.transaction_core.handle_eth_transaction(target_tx)

            logger.debug(
                f"ETH transaction value ({eth_value:.4f} ETH) below threshold "
                f"({threshold_eth} ETH). Skipping..."
            )
            return False

        except Exception as e:
            logger.error(f"Error in high-value ETH transfer strategy: {e}")
            return False

    async def _is_contract_address(self, address: str) -> bool:
        """Check if address is a contract."""
        try:
            code = await self.transaction_core.web3.eth.get_code(address)
            is_contract = len(code) > 0
            logger.debug(f"Address '{address}' is_contract: {is_contract}")
            return is_contract
        except Exception as e:
            logger.error(f"Error checking if address is contract: {e}")
            return False

    async def _validate_contract_interaction(self, contract_address: str) -> bool:
        """Validate interaction with contract address."""
        try:
            # Example validation: check if it's a known contract
            token_symbols = await self.api_config.get_token_symbols()
            is_valid = contract_address in token_symbols
            logger.debug(f"Contract address '{contract_address}' validation result: {is_valid}")
            return is_valid
        except Exception as e:
            logger.error(f"Error validating contract interaction: {e}")
            return False

    async def aggressive_front_run(self, target_tx: Dict[str, Any]) -> bool:
        """
        Execute aggressive front-running strategy with dynamic gas pricing and risk assessment.
        
        Args:
            target_tx: Target transaction details
            
        Returns:
            bool: True if front-run was successful, False otherwise
        """
        logger.debug("Initiating Aggressive Front-Run Strategy...")

        # Validate transaction
        valid, decoded_tx, token_symbol = await self._validate_transaction(
            target_tx, "front_run", min_value=0.1
        )
        if not valid:
            return False

        # Assess risk
        risk_score, market_conditions = await self._assess_risk(
            target_tx, 
            token_symbol,
            price_change=await self._get_price_change(token_symbol)
        )

        if risk_score >= 0.7:  # High confidence threshold
            logger.debug(f"Executing aggressive front-run (Risk: {risk_score:.2f})")
            return await self.transaction_core.front_run(target_tx)

        return False

    async def _get_price_change(self, token_symbol: str) -> float:
        """Get price change using centralized price fetching."""
        current_price = await self.api_config.get_token_price_data(token_symbol, 'current')
        predicted_price = await self.market_monitor.predict_price_movement(token_symbol)
        
        if current_price and predicted_price:
            return (predicted_price / float(current_price) - 1) * 100
        return 0.0

    async def predictive_front_run(self, target_tx: Dict[str, Any]) -> bool:
        """
        Execute predictive front-run strategy based on advanced price prediction analysis
        and multiple market indicators.
        """
        logger.debug("Initiating Enhanced Predictive Front-Run Strategy...")

        # Validate transaction
        valid, decoded_tx, token_symbol = await self._validate_transaction(
            target_tx, "front_run"
        )
        if not valid:
            return False

        # Gather market data asynchronously
        try:
            data = await asyncio.gather(
                self.market_monitor.predict_price_movement(token_symbol),
                self.api_config.get_real_time_price(token_symbol),
                self.market_monitor.check_market_conditions(target_tx["to"]),
                self.market_monitor.fetch_historical_prices(token_symbol, days=1),
                return_exceptions=True
            )
            predicted_price, current_price, market_conditions, historical_prices = data

            if any(isinstance(x, Exception) for x in data):
                logger.warning("Failed to gather complete market data.")
                return False

            if current_price is None or predicted_price is None:
                logger.debug("Missing price data for analysis.")
                return False

        except Exception as e:
            logger.error(f"Error gathering market data: {e}")
            return False

        # Calculate price metrics
        price_change = (predicted_price / float(current_price) - 1) * 100
        volatility = np.std(historical_prices) / np.mean(historical_prices) if historical_prices else 0

        # Score the opportunity (0-100)
        opportunity_score = await self._calculate_opportunity_score(
            price_change=price_change,
            volatility=volatility,
            market_conditions=market_conditions,
            current_price=current_price,
            historical_prices=historical_prices
        )

        # Log detailed analysis
        logger.debug(
            f"Predictive Analysis for {token_symbol}:\n"
            f"Current Price: {current_price:.6f}\n"
            f"Predicted Price: {predicted_price:.6f}\n"
            f"Expected Change: {price_change:.2f}%\n"
            f"Volatility: {volatility:.2f}\n"
            f"Opportunity Score: {opportunity_score}/100\n"
            f"Market Conditions: {market_conditions}"
        )

        # Execute if conditions are favorable
        if opportunity_score >= 75:  # High confidence threshold
            logger.debug(
                f"Executing predictive front-run for {token_symbol} "
                f"(Score: {opportunity_score}/100, Expected Change: {price_change:.2f}%)"
            )
            return await self.transaction_core.front_run(target_tx)

        logger.debug(
            f"Opportunity score {opportunity_score}/100 below threshold. Skipping front-run."
        )
        return False

    async def _calculate_opportunity_score(
        self,
        price_change: float,
        volatility: float,
        market_conditions: Dict[str, bool],
        current_price: float,
        historical_prices: List[float]
    ) -> float:
        """
        Calculate comprehensive opportunity score (0-100) based on multiple metrics.
        Higher score indicates more favorable conditions for front-running.
        """
        score = 0

        # Price change component (0-40 points)
        if price_change > 5.0:        # Very strong upward prediction
            score += 40
        elif price_change > 3.0:      # Strong upward prediction
            score += 30
        elif price_change > 1.0:      # Moderate upward prediction
            score += 20
        elif price_change > 0.5:      # Slight upward prediction
            score += 10

        # Volatility component (0-20 points)
        # Lower volatility is better for predictable outcomes
        if volatility < 0.02:         # Very low volatility
            score += 20
        elif volatility < 0.05:       # Low volatility
            score += 15
        elif volatility < 0.08:       # Moderate volatility
            score += 10

        # Market conditions component (0-20 points)
        if market_conditions.get("bullish_trend", False):
            score += 10
        if not market_conditions.get("high_volatility", True):
            score += 5
        if not market_conditions.get("low_liquidity", True):
            score += 5

        # Price trend component (0-20 points)
        if historical_prices and len(historical_prices) > 1:
            recent_trend = (historical_prices[-1] / historical_prices[0] - 1) * 100
            if recent_trend > 0:      # Upward trend
                score += 20
            elif recent_trend > -1:    # Stable trend
                score += 10

        logger.debug(f"Calculated opportunity score: {score}/100")
        return score

    async def volatility_front_run(self, target_tx: Dict[str, Any]) -> bool:
        """
        Execute front-run strategy based on market volatility analysis with 
        advanced risk assessment and dynamic thresholds.
        """
        logger.debug("Initiating Enhanced Volatility Front-Run Strategy...")

        # Validate transaction
        valid, decoded_tx, token_symbol = await self._validate_transaction(
            target_tx, "front_run"
        )
        if not valid:
            return False

        # Gather market data asynchronously
        try:
            results = await asyncio.gather(
                self.market_monitor.check_market_conditions(target_tx["to"]),
                self.api_config.get_real_time_price(token_symbol),
                self.market_monitor.fetch_historical_prices(token_symbol, days=1),
                return_exceptions=True
            )

            market_conditions, current_price, historical_prices = results

            if any(isinstance(result, Exception) for result in results):
                logger.warning("Failed to gather complete market data")
                return False

        except Exception as e:
            logger.error(f"Error gathering market data: {e}")
            return False

        # Calculate volatility metrics
        volatility_score = await self._calculate_volatility_score(
            historical_prices=historical_prices,
            current_price=current_price,
            market_conditions=market_conditions
        )

        # Log detailed analysis
        logger.debug(
            f"Volatility Analysis for {token_symbol}:\n"
            f"Volatility Score: {volatility_score:.2f}/100\n"
            f"Current Price: {current_price}\n"
            f"24h Price Range: {min(historical_prices):.4f} - {max(historical_prices):.4f}\n"
            f"Market Conditions: {market_conditions}"
        )

        # Execute based on volatility thresholds
        if volatility_score >= 75:  # High volatility threshold
            logger.debug(
                f"Executing volatility-based front-run for {token_symbol} "
                f"(Volatility Score: {volatility_score:.2f}/100)"
            )
            return await self.transaction_core.front_run(target_tx)

        logger.debug(
            f"Volatility score {volatility_score:.2f}/100 below threshold. Skipping front-run."
        )
        return False

    async def _calculate_volatility_score(
        self,
        historical_prices: List[float],
        current_price: float,
        market_conditions: Dict[str, bool]
    ) -> float:
        """
        Calculate comprehensive volatility score (0-100) based on multiple metrics.
        Higher score indicates more favorable volatility conditions.
        """
        score = 0

        # Calculate price volatility metrics
        if len(historical_prices) > 1:
            price_changes = np.diff(historical_prices) / np.array(historical_prices[:-1])
            volatility = np.std(price_changes)
            price_range = (max(historical_prices) - min(historical_prices)) / np.mean(historical_prices)

            # Volatility component (0-40 points)
            if volatility > 0.1:  # Very high volatility
                score += 40
            elif volatility > 0.05:  # High volatility
                score += 30
            elif volatility > 0.02:  # Moderate volatility
                score += 20

            # Price range component (0-30 points)
            if price_range > 0.15:  # Wide price range
                score += 30
            elif price_range > 0.08:  # Moderate price range
                score += 20
            elif price_range > 0.03:  # Narrow price range
                score += 10

        # Market conditions component (0-30 points)
        if market_conditions.get("high_volatility", False):
            score += 15
        if market_conditions.get("bullish_trend", False):
            score += 10
        if not market_conditions.get("low_liquidity", True):
            score += 5

        logger.debug(f"Calculated volatility score: {score}/100")
        return score

    async def advanced_front_run(self, target_tx: Dict[str, Any]) -> bool:
        """
        Execute advanced front-run strategy with comprehensive analysis, risk management,
        and multi-factor decision making.
        """
        logger.debug("Initiating Advanced Front-Run Strategy...")

        # Validate transaction
        valid, decoded_tx, token_symbol = await self._validate_transaction(
            target_tx, "front_run"
        )
        if not valid:
            return False

        # Multi-factor analysis
        try:
            analysis_results = await asyncio.gather(
                self.market_monitor.predict_price_movement(token_symbol),
                self.market_monitor.check_market_conditions(target_tx["to"]),
                self.api_config.get_real_time_price(token_symbol),
                self.api_config.get_token_volume(token_symbol),
                return_exceptions=True
            )

            predicted_price, market_conditions, current_price, volume = analysis_results

            if any(isinstance(result, Exception) for result in analysis_results):
                logger.warning("Failed to gather complete market data.")
                return False

            if current_price is None or predicted_price is None:
                logger.debug("Missing price data for analysis. Skipping...")
                return False

        except Exception as e:
            logger.error(f"Error gathering market data: {e}")
            return False

        # Advanced decision making
        price_increase = (predicted_price / float(current_price) - 1) * 100
        is_bullish = market_conditions.get("bullish_trend", False)
        is_volatile = market_conditions.get("high_volatility", False)
        has_liquidity = not market_conditions.get("low_liquidity", True)

        # Calculate risk score (0-100)
        risk_score = self._calculate_risk_score(
            price_increase=price_increase,
            is_bullish=is_bullish,
            is_volatile=is_volatile,
            has_liquidity=has_liquidity,
            volume=volume
        )

        # Log detailed analysis
        logger.debug(
            f"Analysis for {token_symbol}:\n"
            f"Price Increase: {price_increase:.2f}%\n"
            f"Market Trend: {'Bullish' if is_bullish else 'Bearish'}\n"
            f"Volatility: {'High' if is_volatile else 'Low'}\n"
            f"Liquidity: {'Adequate' if has_liquidity else 'Low'}\n"
            f"24h Volume: ${volume:,.2f}\n"
            f"Risk Score: {risk_score}/100"
        )

        # Execute if conditions are favorable
        if risk_score >= 75:  # Minimum risk score threshold
            logger.debug(
                f"Executing advanced front-run for {token_symbol} "
                f"(Risk Score: {risk_score}/100)"
            )
            return await self.transaction_core.front_run(target_tx)

        logger.debug(
            f"Risk score {risk_score}/100 below threshold. Skipping front-run."
        )
        return False

    def _calculate_risk_score(
        self,
        price_increase: float,
        is_bullish: bool,
        is_volatile: bool,
        has_liquidity: bool,
        volume: float
    ) -> int:
        """
        Calculate comprehensive risk score based on multiple market factors.
        
        Args:
            price_increase: Percentage price increase
            is_bullish: Market trend indicator
            is_volatile: Volatility indicator
            has_liquidity: Liquidity indicator
            volume: Trading volume in USD
            
        Returns:
            int: Risk score between 0-100
        """
        score = 0

        # Price momentum (0-30 points)
        if price_increase >= 5.0:
            score += 30
        elif price_increase >= 3.0:
            score += 20
        elif price_increase >= 1.0:
            score += 10

        # Market trend (0-20 points)
        if is_bullish:
            score += 20

        # Volatility (0-15 points)
        if not is_volatile:
            score += 15

        # Liquidity (0-20 points)
        if has_liquidity:
            score += 20

        # Volume-based score (0-15 points)
        if volume >= 1_000_000:  # $1M+ daily volume
            score += 15
        elif volume >= 500_000:   # $500k+ daily volume
            score += 10
        elif volume >= 100_000:   # $100k+ daily volume
            score += 5

        logger.debug(f"Calculated risk score: {score}/100")
        return score

    async def price_dip_back_run(self, target_tx: Dict[str, Any]) -> bool:
        """Execute back-run strategy based on price dip prediction."""
        logger.debug("Initiating Price Dip Back-Run Strategy...")

        # Validate transaction
        valid, decoded_tx, token_symbol = await self._validate_transaction(
            target_tx, "back_run"
        )
        if not valid:
            return False

        current_price = await self.api_config.get_real_time_price(token_symbol)
        if current_price is None:
            return False

        predicted_price = await self.market_monitor.predict_price_movement(token_symbol)
        if predicted_price < float(current_price) * 0.99:
            logger.debug("Predicted price decrease exceeds threshold, proceeding with back-run.")
            return await self.transaction_core.back_run(target_tx)

        logger.debug("Predicted price decrease does not meet threshold. Skipping back-run.")
        return False

    async def flashloan_back_run(self, target_tx: Dict[str, Any]) -> bool:
        """Execute back-run strategy using flash loans."""
        logger.debug("Initiating Flashloan Back-Run Strategy...")
        estimated_amount = await self.transaction_core.calculate_flashloan_amount(target_tx)
        estimated_profit = estimated_amount * Decimal("0.02")
        if estimated_profit > self.configuration.min_profit_threshold:
            logger.debug(f"Estimated profit: {estimated_profit} ETH meets threshold.")
            return await self.transaction_core.back_run(target_tx)
        logger.debug("Profit is insufficient for flashloan back-run. Skipping.")
        return False

    async def high_volume_back_run(self, target_tx: Dict[str, Any]) -> bool:
        """Execute back-run strategy based on high trading volume."""
        logger.debug("Initiating High Volume Back-Run Strategy...")

        # Validate transaction
        valid, decoded_tx, token_symbol = await self._validate_transaction(
            target_tx, "back_run"
        )
        if not valid:
            return False

        volume_24h = await self.api_config.get_token_volume(token_symbol)
        volume_threshold = self._get_volume_threshold(token_symbol)
        if volume_24h > volume_threshold:
            logger.debug(f"High volume detected (${volume_24h:,.2f} USD), proceeding with back-run.")
            return await self.transaction_core.back_run(target_tx)

        logger.debug(f"Volume (${volume_24h:,.2f} USD) below threshold (${volume_threshold:,.2f} USD). Skipping.")
        return False

    def _get_volume_threshold(self, token_symbol: str) -> float:
        """
        Determine the volume threshold for a token based on market cap tiers and liquidity.
        Returns threshold in USD.
        """
        # Define volume thresholds for different token tiers
        tier1_tokens = {
            "WETH": 15_000_000,
            "ETH": 15_000_000,
            "WBTC": 25_000_000,
            "USDT": 50_000_000,
            "USDC": 50_000_000,
            "DAI": 20_000_000,
        }

        tier2_tokens = {
            "UNI": 5_000_000,
            "LINK": 8_000_000,
            "AAVE": 3_000_000,
            "MKR": 2_000_000,
            "CRV": 4_000_000,
            "SUSHI": 2_000_000,
            "SNX": 2_000_000,
            "COMP": 2_000_000,
        }

        tier3_tokens = {
            "1INCH": 1_000_000,
            "YFI": 1_500_000,
            "BAL": 1_000_000,
            "PERP": 800_000,
            "DYDX": 1_200_000,
            "LDO": 1_500_000,
            "RPL": 700_000,
        }

        volatile_tokens = {
            "SHIB": 8_000_000,
            "PEPE": 5_000_000,
            "DOGE": 10_000_000,
            "FLOKI": 3_000_000,
        }

        # Check each tier in order
        if token_symbol in tier1_tokens:
            threshold = tier1_tokens[token_symbol]
        elif token_symbol in tier2_tokens:
            threshold = tier2_tokens[token_symbol]
        elif token_symbol in tier3_tokens:
            threshold = tier3_tokens[token_symbol]
        elif token_symbol in volatile_tokens:
            threshold = volatile_tokens[token_symbol]
        else:
            threshold = 500_000  # Conservative default for unknown tokens

        logger.debug(f"Volume threshold for '{token_symbol}': ${threshold:,.2f} USD")
        return threshold

    async def advanced_back_run(self, target_tx: Dict[str, Any]) -> bool:
        """Execute advanced back-run strategy with comprehensive analysis."""
        logger.debug("Initiating Advanced Back-Run Strategy...")

        # Validate transaction
        valid, decoded_tx, token_symbol = await self._validate_transaction(
            target_tx, "back_run"
        )
        if not valid:
            return False

        market_conditions = await self.market_monitor.check_market_conditions(
            target_tx["to"]
        )
        if market_conditions.get("high_volatility", False) and market_conditions.get(
            "bullish_trend", False
        ):
            logger.debug("Market conditions favorable for advanced back-run.")
            return await self.transaction_core.back_run(target_tx)

        logger.debug("Market conditions unfavorable for advanced back-run. Skipping.")
        return False

    async def flash_profit_sandwich(self, target_tx: Dict[str, Any]) -> bool:
        """Execute sandwich attack strategy using flash loans."""
        logger.debug("Initiating Flash Profit Sandwich Strategy...")
        estimated_amount = await self.transaction_core.calculate_flashloan_amount(target_tx)
        estimated_profit = estimated_amount * Decimal("0.02")
        if estimated_profit > self.configuration.min_profit_threshold:
            gas_price = await self.transaction_core.get_dynamic_gas_price()
            if (gas_price > 200):
                logger.debug(f"Gas price too high for sandwich attack: {gas_price} Gwei")
                return False
            logger.debug(f"Executing sandwich with estimated profit: {estimated_profit:.4f} ETH")
            return await self.transaction_core.execute_sandwich_attack(target_tx)
        logger.debug("Insufficient profit potential for flash sandwich. Skipping.")
        return False

    async def price_boost_sandwich(self, target_tx: Dict[str, Any]) -> bool:
        """Execute sandwich attack strategy based on price momentum."""
        logger.debug("Initiating Price Boost Sandwich Strategy...")

        # Validate transaction
        valid, decoded_tx, token_symbol = await self._validate_transaction(
            target_tx, "sandwich_attack"
        )
        if not valid:
            return False

        historical_prices = await self.market_monitor.fetch_historical_prices(token_symbol)
        if not historical_prices:
            return False

        momentum = await self._analyze_price_momentum(historical_prices)
        if momentum > 0.02:
            logger.debug(f"Strong price momentum detected: {momentum:.2%}")
            return await self.transaction_core.execute_sandwich_attack(target_tx)

        logger.debug(f"Insufficient price momentum: {momentum:.2%}. Skipping.")
        return False

    async def _analyze_price_momentum(self, prices: List[float]) -> float:
        """Analyze price momentum from historical prices."""
        if not prices or len(prices) < 2:
            logger.debug("Insufficient historical prices for momentum analysis.")
            return 0.0
        price_changes = [prices[i] / prices[i - 1] - 1 for i in range(1, len(prices))]
        momentum = sum(price_changes) / len(price_changes)
        logger.debug(f"Calculated price momentum: {momentum:.4f}")
        return momentum

    async def arbitrage_sandwich(self, target_tx: Dict[str, Any]) -> bool:
        """Execute sandwich attack strategy based on arbitrage opportunities."""
        logger.debug("Initiating Arbitrage Sandwich Strategy...")

        # Validate transaction
        valid, decoded_tx, token_symbol = await self._validate_transaction(
            target_tx, "sandwich_attack"
        )
        if not valid:
            return False

        is_arbitrage = await self.market_monitor.is_arbitrage_opportunity(target_tx)
        if is_arbitrage:
            logger.debug(f"Arbitrage opportunity detected for {token_symbol}")
            return await self.transaction_core.execute_sandwich_attack(target_tx)

        logger.debug("No profitable arbitrage opportunity found. Skipping.")
        return False

    async def advanced_sandwich_attack(self, target_tx: Dict[str, Any]) -> bool:
        """Execute advanced sandwich attack strategy with risk management."""
        logger.debug("Initiating Advanced Sandwich Attack...")

        # Validate transaction
        valid, decoded_tx, token_symbol = await self._validate_transaction(
            target_tx, "sandwich_attack"
        )
        if not valid:
            return False

        market_conditions = await self.market_monitor.check_market_conditions(
            target_tx["to"]
        )
        if market_conditions.get("high_volatility", False) and market_conditions.get(
            "bullish_trend", False
        ):
            logger.debug("Conditions favorable for sandwich attack.")
            return await self.transaction_core.execute_sandwich_attack(target_tx)

        logger.debug("Conditions unfavorable for sandwich attack. Skipping.")
        return False

    # ========================= End of Strategy Implementations =========================

    async def stop(self) -> None:
        """Stop strategy network operations.""" 
        try:
            # Clean up any running strategies
            self.strategy_performance.clear()
            self.reinforcement_weights.clear()
            self.history_data.clear()
            logger.info("Strategy Net stopped successfully.")
        except Exception as e:
            logger.error(f"Error stopping Strategy Net: {e}")

#//////////////////////////////////////////////////////////////////////////////
class StrategyConfiguration:
    """Configuration parameters for strategy execution."""
    decay_factor: float = 0.95
    min_profit_threshold: Decimal = Decimal("0.01")
    learning_rate: float = 0.01
    exploration_rate: float = 0.1

class StrategyPerformanceMetrics:
    """Metrics for tracking strategy performance."""
    successes: int = 0
    failures: int = 0
    profit: Decimal = Decimal("0")
    avg_execution_time: float = 0.0
    success_rate: float = 0.0
    total_executions: int = 0

