class StrategyManager:
    """
    Manages and executes various trading strategies such as ETH transactions, front-running,
    back-running, and sandwich attacks. It tracks strategy performance, predicts market movements,
    and selects the best strategy based on historical performance and reinforcement learning.

    Attributes:
        transaction_array (TransactionArray): Instance managing transactions.
        market_analyzer (MarketAnalyzer): Instance analyzing market conditions.
        logger (Optional[logging.Logger]): Logger instance for logging.
        strategy_performance (Dict[str, Dict[str, Any]]): Performance metrics for each strategy.
        history_data (List[Dict[str, Any]]): Historical data for trend analysis.
        price_model (LinearRegression): Model for predicting price trends.
        reinforcement_weights (Dict[str, np.ndarray]): Weights for reinforcement learning.
        decay_factor (float): Decay factor for past performance.
        min_profit_threshold (Decimal): Minimum profit margin in ETH.
    """

    def __init__(
        self,
        transaction_array: TransactionArray,
        market_analyzer: "MarketAnalyzer",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        Initializes the StrategyManager with necessary components.

        Args:
            transaction_array (TransactionArray): Instance managing transactions.
            market_analyzer (MarketAnalyzer): Instance analyzing market conditions.
            logger (Optional[logging.Logger]): Logger instance for logging.
        """
        self.transaction_array = transaction_array
        self.market_analyzer = market_analyzer
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.logger.info("StrategyManager initialized successfully. ✅")

        # Track strategy performance and profitability
        self.strategy_performance: Dict[str, Dict[str, Any]] = {
            "eth_transaction": {"successes": 0, "failures": 0, "profit": Decimal("0")},
            "front_run": {"successes": 0, "failures": 0, "profit": Decimal("0")},
            "back_run": {"successes": 0, "failures": 0, "profit": Decimal("0")},
            "sandwich_attack": {"successes": 0, "failures": 0, "profit": Decimal("0")},
        }

        # Maintain historical data to identify trends and optimize strategy
        self.history_data: List[Dict[str, Any]] = []
        self.price_model = LinearRegression()

        # Reinforcement weights with decaying factors for continuous learning
        self.reinforcement_weights: Dict[str, np.ndarray] = {
            "eth_transaction": np.ones(1),
            "front_run": np.ones(4),
            "back_run": np.ones(4),
            "sandwich_attack": np.ones(4),
        }
        self.decay_factor: float = 0.9  # Decay factor for past performances
        self.min_profit_threshold: Decimal = Decimal(
            "0.01"
        )  # Minimum profit margin in ETH

    async def execute_best_strategy(
        self, target_tx: Dict[str, Any], strategy_type: str
    ) -> bool:
        """
        Executes the most suitable strategy based on historical performance and current market conditions.

        Args:
            target_tx (Dict[str, Any]): The target transaction details.
            strategy_type (str): The type of strategy to execute ('eth_transaction', 'front_run', 'back_run', 'sandwich_attack').

        Returns:
            bool: True if the strategy was executed successfully, False otherwise.
        """
        strategies = self.get_strategies(strategy_type)
        if not strategies:
            self.logger.warning(f"No strategies available for type: {strategy_type} ❗")
            return False

        selected_strategy = self._select_strategy(strategies, strategy_type)
        self.logger.info(f"Executing strategy: {selected_strategy.__name__} ⚔️🏃")

        try:
            profit_before = await self.transaction_array.get_current_profit()  # Track profit before execution
            success = await selected_strategy(target_tx)
            profit_after = await self.transaction_array.get_current_profit()  # Profit after execution

            # Calculate profit/loss from the strategy execution
            profit_made = Decimal(profit_after) - Decimal(profit_before)
            await self.update_history(
                selected_strategy.__name__, success, strategy_type, profit_made
            )

            return success
        except Exception as e:
            self.logger.error(
                f"Error executing strategy {selected_strategy.__name__}: {e} ❌"
            )
            return False

    def get_strategies(self, strategy_type: str) -> List[Any]:
        """
        Retrieves a list of strategies based on the strategy type.

        Args:
            strategy_type (str): The type of strategy.

        Returns:
            List[Any]: List of strategy functions.
        """
        strategies = {
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

        if strategy_type not in strategies:
            self.logger.error(f"Invalid strategy type provided: {strategy_type} ❌")
            return []
        return strategies[strategy_type]

    def _select_strategy(self, strategies: List[Any], strategy_type: str) -> Any:
        """
        Selects a strategy based on reinforcement weights and decaying factors.

        Args:
            strategies (List[Any]): List of strategy functions.
            strategy_type (str): The type of strategy.

        Returns:
            Any: Selected strategy function.
        """
        weights = self.reinforcement_weights[strategy_type]
        # Apply decay to focus on recent performance
        weights *= self.decay_factor
        # Ensure weights are not below a threshold to maintain exploration
        weights = np.maximum(weights, 0.1)
        strategy_indices = np.arange(len(strategies))
        try:
            selected_strategy_index = np.random.choice(
                strategy_indices, p=weights / weights.sum()
            )
            self.logger.debug(
                f"Selected strategy index {selected_strategy_index} for type {strategy_type}."
            )
            return strategies[selected_strategy_index]
        except ValueError as e:
            self.logger.error(
                f"Error selecting strategy: {e}. Falling back to random selection."
            )
            return np.random.choice(strategies)

    async def update_history(
        self, strategy_name: str, success: bool, strategy_type: str, profit: Decimal
    ) -> None:
        """
        Updates the historical data and reinforcement weights based on the strategy execution outcome.

        Args:
            strategy_name (str): Name of the executed strategy.
            success (bool): Whether the strategy execution was successful.
            strategy_type (str): The type of strategy.
            profit (Decimal): Profit made from the strategy execution.
        """
        self.logger.info(
            f"Updating history for strategy: {strategy_name}, Success: {success}, Profit: {profit} ✅"
        )

        # Update performance metrics
        if success:
            self.strategy_performance[strategy_type]["successes"] += 1
            self.strategy_performance[strategy_type]["profit"] += profit
        else:
            self.strategy_performance[strategy_type]["failures"] += 1
            self.strategy_performance[strategy_type][
                "profit"
            ] += profit  # Note: profit may be negative

        # Append to history data
        self.history_data.append(
            {
                "strategy_name": strategy_name,
                "success": success,
                "profit": profit,
                "strategy_type": strategy_type,
                "total_profit": self.strategy_performance[strategy_type]["profit"],
            }
        )

        # Update reinforcement weights based on profit and success
        strategy_index = self.get_strategy_index(strategy_name, strategy_type)
        if strategy_index >= 0:
            reward_factor = (
                float(profit) if profit > Decimal("0") else -1
            )  # Reward based on profit
            self.reinforcement_weights[strategy_type][strategy_index] += reward_factor
            # Ensure weights remain positive
            self.reinforcement_weights[strategy_type][strategy_index] = max(
                self.reinforcement_weights[strategy_type][strategy_index], 0.1
            )
            self.logger.debug(
                f"Updated reinforcement weight for {strategy_name}: {self.reinforcement_weights[strategy_type][strategy_index]}"
            )

    def get_strategy_index(self, strategy_name: str, strategy_type: str) -> int:
        """
        Retrieves the index of a strategy based on its name and type.

        Args:
            strategy_name (str): Name of the strategy.
            strategy_type (str): Type of the strategy.

        Returns:
            int: Index of the strategy in the reinforcement weights array. Returns -1 if not found.
        """
        strategy_mapping = {
            "eth_transaction": {
                "high_value_eth_transfer": 0,
            },
            "front_run": {
                "aggressive_front_run": 0,
                "predictive_front_run": 1,
                "volatility_front_run": 2,
                "advanced_front_run": 3,
            },
            "back_run": {
                "price_dip_back_run": 0,
                "flashloan_back_run": 1,
                "high_volume_back_run": 2,
                "advanced_back_run": 3,
            },
            "sandwich_attack": {
                "flash_profit_sandwich": 0,
                "price_boost_sandwich": 1,
                "arbitrage_sandwich": 2,
                "advanced_sandwich_attack": 3,
            },
        }
        return strategy_mapping.get(strategy_type, {}).get(strategy_name, -1)

    async def predict_price_movement(self, token_symbol: str) -> float:
        """
        Predicts the next price movement of a token using historical price data.

        Args:
            token_symbol (str): The symbol of the token.

        Returns:
            float: Predicted price movement value.
        """
        self.logger.info(f"Predicting price movement for {token_symbol} 🔮")
        try:
            prices = await self.market_analyzer.fetch_historical_prices(token_symbol)
            if not prices:
                self.logger.warning(
                    f"No historical prices available for {token_symbol}. Cannot predict movement."
                )
                return 0.0
            X = np.arange(len(prices)).reshape(-1, 1)
            y = np.array(prices)
            self.price_model.fit(X, y)
            next_time = np.array([[len(prices)]])
            predicted_price = self.price_model.predict(next_time)[0]
            self.logger.info(
                f"Predicted price for {token_symbol}: {predicted_price} ETH 📈"
            )
            return float(predicted_price)
        except NotFittedError:
            self.logger.error(
                "Price model is not fitted yet. Cannot predict price movement. ❌"
            )
            return 0.0
        except Exception as e:
            self.logger.exception(
                f"Error predicting price movement for {token_symbol}: {e} ❌"
            )
            return 0.0

    # -------------------- Strategy Methods --------------------

    async def high_value_eth_transfer(self, target_tx: Dict[str, Any]) -> bool:
        """
        Strategy: Handles high-value ETH transfers.

        Args:
            target_tx (Dict[str, Any]): The target transaction details.

        Returns:
            bool: True if the strategy was executed successfully, False otherwise.
        """
        self.logger.info("Initiating High-Value ETH Transfer Strategy... 🏃💨")
        try:
            # Check if it's a high-value ETH transfer
            eth_value_in_wei = target_tx.get("value", 0)
            if eth_value_in_wei > self.transaction_array.web3.to_wei(10, "ether"):
                eth_value_in_eth = self.transaction_array.web3.from_wei(
                    eth_value_in_wei, "ether"
                )
                self.logger.info(
                    f"High-value ETH transfer detected: {eth_value_in_eth} ETH 🏃"
                )
                # Proceed with handling the ETH transaction (e.g., front-running)
                return await self.transaction_array.handle_eth_transaction(target_tx)
            self.logger.info(
                "ETH transaction does not meet the high-value criteria. Skipping... ⚠️"
            )
            return False
        except Exception as e:
            self.logger.error(
                f"Error executing High-Value ETH Transfer Strategy: {e} ❌"
            )
            return False

    async def aggressive_front_run(self, target_tx: Dict[str, Any]) -> bool:
        """
        Strategy: Aggressively front-runs transactions based on value.

        Args:
            target_tx (Dict[str, Any]): The target transaction details.

        Returns:
            bool: True if the front-run was executed successfully, False otherwise.
        """
        self.logger.info("Initiating Aggressive Front-Run Strategy... 🏃")
        try:
            if target_tx.get("value", 0) > self.transaction_array.web3.to_wei(
                1, "ether"
            ):
                self.logger.info(
                    "Transaction value above threshold, proceeding with aggressive front-run."
                )
                return await self.transaction_array.front_run(target_tx)
            self.logger.info(
                "Transaction below threshold. Skipping aggressive front-run."
            )
            return False
        except Exception as e:
            self.logger.error(f"Error executing Aggressive Front-Run Strategy: {e} ❌")
            return False

    async def predictive_front_run(self, target_tx: Dict[str, Any]) -> bool:
        """
        Strategy: Front-runs transactions based on price predictions.

        Args:
            target_tx (Dict[str, Any]): The target transaction details.

        Returns:
            bool: True if the front-run was executed successfully, False otherwise.
        """
        self.logger.info("Initiating Predictive Front-Run Strategy... 🏃")
        try:
            decoded_tx = await self.transaction_array.decode_transaction_input(
                target_tx["input"], target_tx["to"]
            )
            if not decoded_tx:
                self.logger.warning(
                    "Failed to decode transaction input for Predictive Front-Run Strategy. ❗"
                )
                return False
            params = decoded_tx.get("params", {})
            path = params.get("path", [])
            if not path:
                self.logger.warning(
                    "Transaction has no path parameter for Predictive Front-Run Strategy. ❗"
                )
                return False
            token_address = path[0]
            token_symbol = await self.market_analyzer.get_token_symbol(token_address)
            if not token_symbol:
                self.logger.warning(
                    f"Token symbol not found for address {token_address} in Predictive Front-Run Strategy. ❗"
                )
                return False
            predicted_price = await self.predict_price_movement(token_symbol)
            current_price = await self.market_analyzer.get_current_price(token_symbol)
            if current_price is None:
                self.logger.warning(
                    f"Current price not available for {token_symbol} in Predictive Front-Run Strategy. ❗"
                )
                return False
            if predicted_price > float(current_price) * 1.01:  # 1% profit margin
                self.logger.info(
                    "Predicted price increase exceeds threshold, proceeding with predictive front-run."
                )
                return await self.transaction_array.front_run(target_tx)
            self.logger.info(
                "Predicted price increase does not meet threshold. Skipping predictive front-run."
            )
            return False
        except Exception as e:
            self.logger.error(f"Error executing Predictive Front-Run Strategy: {e} ❌")
            return False

    async def volatility_front_run(self, target_tx: Dict[str, Any]) -> bool:
        """
        Strategy: Front-runs transactions based on market volatility.

        Args:
            target_tx (Dict[str, Any]): The target transaction details.

        Returns:
            bool: True if the front-run was executed successfully, False otherwise.
        """
        self.logger.info("Initiating Volatility Front-Run Strategy... 🏃")
        try:
            market_conditions = await self.market_analyzer.check_market_conditions(
                target_tx["to"]
            )
            if market_conditions.get("high_volatility", False):
                self.logger.info(
                    "High volatility detected, proceeding with volatility front-run."
                )
                return await self.transaction_array.front_run(target_tx)
            self.logger.info(
                "Market volatility not high enough. Skipping volatility front-run."
            )
            return False
        except Exception as e:
            self.logger.error(f"Error executing Volatility Front-Run Strategy: {e} ❌")
            return False

    async def advanced_front_run(self, target_tx: Dict[str, Any]) -> bool:
        """
        Enhanced Strategy: Combines price predictions and market trends for front-running.

        Args:
            target_tx (Dict[str, Any]): The target transaction details.

        Returns:
            bool: True if the enhanced front-run was executed successfully, False otherwise.
        """
        self.logger.info("Initiating Advanced Front-Run Strategy... 🏃💨")
        try:
            decoded_tx = await self.transaction_array.decode_transaction_input(
                target_tx["input"], target_tx["to"]
            )
            if not decoded_tx:
                self.logger.warning(
                    "Failed to decode transaction input for Advanced Front-Run Strategy. ❗"
                )
                return False
            params = decoded_tx.get("params", {})
            path = params.get("path", [])
            if not path:
                self.logger.warning(
                    "Transaction has no path parameter for Advanced Front-Run Strategy. ❗"
                )
                return False
            token_symbol = await self.market_analyzer.get_token_symbol(path[0])
            if not token_symbol:
                self.logger.warning(
                    f"Token symbol not found for address {path[0]} in Advanced Front-Run Strategy. ❗"
                )
                return False
            predicted_price = await self.predict_price_movement(token_symbol)
            market_conditions = await self.market_analyzer.check_market_conditions(
                target_tx["to"]
            )
            current_price = await self.market_analyzer.get_current_price(token_symbol)
            if current_price is None:
                self.logger.warning(
                    f"Current price not available for {token_symbol} in Advanced Front-Run Strategy. ❗"
                )
                return False
            if (
                predicted_price > float(current_price) * 1.02
            ) and market_conditions.get("bullish_trend", False):
                self.logger.info(
                    "Favorable price and bullish trend detected, proceeding with advanced front-run."
                )
                return await self.transaction_array.front_run(target_tx)
            self.logger.info(
                "Conditions not favorable for advanced front-run. Skipping."
            )
            return False
        except Exception as e:
            self.logger.error(f"Error executing Advanced Front-Run Strategy: {e} ❌")
            return False

    async def price_dip_back_run(self, target_tx: Dict[str, Any]) -> bool:
        """
        Strategy: Executes back-run based on significant price dips.

        Args:
            target_tx (Dict[str, Any]): The target transaction details.

        Returns:
            bool: True if the back-run was executed successfully, False otherwise.
        """
        self.logger.info("Initiating Price Dip Back-Run Strategy... 🔙🏃")
        try:
            decoded_tx = await self.transaction_array.decode_transaction_input(
                target_tx["input"], target_tx["to"]
            )
            if not decoded_tx:
                self.logger.warning(
                    "Failed to decode transaction input for Price Dip Back-Run Strategy. ❗"
                )
                return False
            params = decoded_tx.get("params", {})
            path = params.get("path", [])
            if not path:
                self.logger.warning(
                    "Transaction has no path parameter for Price Dip Back-Run Strategy. ❗"
                )
                return False
            token_address = path[-1]
            token_symbol = await self.market_analyzer.get_token_symbol(token_address)
            if not token_symbol:
                self.logger.warning(
                    f"Token symbol not found for address {token_address} in Price Dip Back-Run Strategy. ❗"
                )
                return False
            current_price = await self.market_analyzer.get_current_price(token_symbol)
            if current_price is None:
                self.logger.warning(
                    f"Current price not available for {token_symbol} in Price Dip Back-Run Strategy. ❗"
                )
                return False
            predicted_price = await self.predict_price_movement(token_symbol)
            if predicted_price < float(current_price) * 0.99:  # 1% profit margin
                self.logger.info(
                    "Predicted price decrease exceeds threshold, proceeding with price dip back-run."
                )
                return await self.transaction_array.back_run(target_tx)
            self.logger.info(
                "Predicted price decrease does not meet threshold. Skipping price dip back-run."
            )
            return False
        except Exception as e:
            self.logger.error(f"Error executing Price Dip Back-Run Strategy: {e} ❌")
            return False

    async def flashloan_back_run(self, target_tx: Dict[str, Any]) -> bool:
        """
        Strategy: Utilizes flashloan calculations to determine back-running profitability.

        Args:
            target_tx (Dict[str, Any]): The target transaction details.

        Returns:
            bool: True if the back-run was executed successfully, False otherwise.
        """
        self.logger.info("Initiating Flashloan Back-Run Strategy... 🔙🏃")
        try:
            estimated_profit = self.transaction_array.calculate_flashloan_amount(
                target_tx
            ) * Decimal(
                "0.02"
            )  # Assume 2% profit margin
            if estimated_profit > self.min_profit_threshold:
                self.logger.info(
                    "Estimated profit meets threshold, proceeding with flashloan back-run."
                )
                return await self.transaction_array.back_run(target_tx)
            self.logger.info("Profit is insufficient for flashloan back-run. Skipping.")
            return False
        except Exception as e:
            self.logger.error(f"Error executing Flashloan Back-Run Strategy: {e} ❌")
            return False

    async def high_volume_back_run(self, target_tx: Dict[str, Any]) -> bool:
        """
        Strategy: Executes back-run based on high token trading volume.

        Args:
            target_tx (Dict[str, Any]): The target transaction details.

        Returns:
            bool: True if the back-run was executed successfully, False otherwise.
        """
        self.logger.info("Initiating High Volume Back-Run Strategy... 🔙🏃")
        try:
            token_volume = await self.market_analyzer.get_token_volume(target_tx["to"])
            if token_volume > 1_000_000:  # Check if volume is high
                self.logger.info(
                    "High volume detected, proceeding with high volume back-run."
                )
                return await self.transaction_array.back_run(target_tx)
            self.logger.info("Volume not favorable for high volume back-run. Skipping.")
            return False
        except Exception as e:
            self.logger.error(f"Error executing High Volume Back-Run Strategy: {e} ❌")
            return False

    async def advanced_back_run(self, target_tx: Dict[str, Any]) -> bool:
        """
        Enhanced Strategy: Combines price predictions and market trends for back-running.

        Args:
            target_tx (Dict[str, Any]): The target transaction details.

        Returns:
            bool: True if the enhanced back-run was executed successfully, False otherwise.
        """
        self.logger.info("Initiating Advanced Back-Run Strategy... 🔙🏃💨")
        try:
            decoded_tx = await self.transaction_array.decode_transaction_input(
                target_tx["input"], target_tx["to"]
            )
            if not decoded_tx:
                self.logger.warning(
                    "Failed to decode transaction input for Advanced Back-Run Strategy. ❗"
                )
                return False
            params = decoded_tx.get("params", {})
            path = params.get("path", [])
            if not path:
                self.logger.warning(
                    "Transaction has no path parameter for Advanced Back-Run Strategy. ❗"
                )
                return False
            token_address = path[-1]
            token_symbol = await self.market_analyzer.get_token_symbol(token_address)
            if not token_symbol:
                self.logger.warning(
                    f"Token symbol not found for address {token_address} in Advanced Back-Run Strategy. ❗"
                )
                return False
            current_price = await self.market_analyzer.get_current_price(token_symbol)
            if current_price is None:
                self.logger.warning(
                    f"Current price not available for {token_symbol} in Advanced Back-Run Strategy. ❗"
                )
                return False
            predicted_price = await self.predict_price_movement(token_symbol)
            market_conditions = await self.market_analyzer.check_market_conditions(
                target_tx["to"]
            )
            if (
                predicted_price < float(current_price) * 0.98
            ) and market_conditions.get("bearish_trend", False):
                self.logger.info(
                    "Favorable price and bearish trend detected, proceeding with advanced back-run."
                )
                return await self.transaction_array.back_run(target_tx)
            self.logger.info(
                "Conditions not favorable for advanced back-run. Skipping."
            )
            return False
        except Exception as e:
            self.logger.error(f"Error executing Advanced Back-Run Strategy: {e} ❌")
            return False

    async def flash_profit_sandwich(self, target_tx: Dict[str, Any]) -> bool:
        """
        Strategy: Executes sandwich attack based on flashloan profitability.

        Args:
            target_tx (Dict[str, Any]): The target transaction details.

        Returns:
            bool: True if the sandwich attack was executed successfully, False otherwise.
        """
        self.logger.info("Initiating Flash Profit Sandwich Strategy... 🥪🏃")
        try:
            potential_profit = self.transaction_array.calculate_flashloan_amount(
                target_tx
            )
            if potential_profit > self.min_profit_threshold:
                self.logger.info(
                    "Potential profit meets threshold, proceeding with flash profit sandwich attack."
                )
                return await self.transaction_array.execute_sandwich_attack(target_tx)
            self.logger.info(
                "Conditions not met for flash profit sandwich attack. Skipping."
            )
            return False
        except Exception as e:
            self.logger.error(f"Error executing Flash Profit Sandwich Strategy: {e} ❌")
            return False

    async def price_boost_sandwich(self, target_tx: Dict[str, Any]) -> bool:
        """
        Strategy: Executes sandwich attack based on favorable token price conditions.

        Args:
            target_tx (Dict[str, Any]): The target transaction details.

        Returns:
            bool: True if the sandwich attack was executed successfully, False otherwise.
        """
        self.logger.info("Initiating Price Boost Sandwich Strategy... 🥪🏃")
        try:
            token_symbol = await self.market_analyzer.get_token_symbol(target_tx["to"])
            current_price = await self.market_analyzer.get_current_price(token_symbol)
            if current_price is None:
                self.logger.warning(
                    f"Current price not available for {token_symbol} in Price Boost Sandwich Strategy. ❗"
                )
                return False
            predicted_price = await self.predict_price_movement(token_symbol)
            if predicted_price > float(current_price) * 1.02:  # 2% profit margin
                self.logger.info(
                    "Favorable price detected, proceeding with price boost sandwich attack."
                )
                return await self.transaction_array.execute_sandwich_attack(target_tx)
            self.logger.info(
                "Price conditions not favorable for price boost sandwich attack. Skipping."
            )
            return False
        except Exception as e:
            self.logger.error(f"Error executing Price Boost Sandwich Strategy: {e} ❌")
            return False

    async def arbitrage_sandwich(self, target_tx: Dict[str, Any]) -> bool:
        """
        Strategy: Executes sandwich attack based on arbitrage opportunities.

        Args:
            target_tx (Dict[str, Any]): The target transaction details.

        Returns:
            bool: True if the sandwich attack was executed successfully, False otherwise.
        """
        self.logger.info("Initiating Arbitrage Sandwich Strategy... 🥪🏃")
        try:
            if await self.market_analyzer.is_arbitrage_opportunity(target_tx):
                self.logger.info(
                    "Arbitrage opportunity detected, proceeding with arbitrage sandwich attack."
                )
                return await self.transaction_array.execute_sandwich_attack(target_tx)
            self.logger.info(
                "No arbitrage opportunity detected. Skipping arbitrage sandwich attack."
            )
            return False
        except Exception as e:
            self.logger.error(f"Error executing Arbitrage Sandwich Strategy: {e} ❌")
            return False

    async def advanced_sandwich_attack(self, target_tx: Dict[str, Any]) -> bool:
        """
        Enhanced Strategy: Combines flashloan profitability and market volatility for sandwich attacks.

        Args:
            target_tx (Dict[str, Any]): The target transaction details.

        Returns:
            bool: True if the enhanced sandwich attack was executed successfully, False otherwise.
        """
        self.logger.info("Initiating Advanced Sandwich Attack Strategy... 🥪🏃💨")
        try:
            potential_profit = self.transaction_array.calculate_flashloan_amount(
                target_tx
            )
            market_conditions = await self.market_analyzer.check_market_conditions(
                target_tx["to"]
            )
            if (potential_profit > Decimal("0.02")) and market_conditions.get(
                "high_volatility", False
            ):
                self.logger.info(
                    "Conditions favorable for advanced sandwich attack, executing."
                )
                return await self.transaction_array.execute_sandwich_attack(target_tx)
            self.logger.info(
                "Conditions not favorable for advanced sandwich attack. Skipping."
            )
            return False
        except Exception as e:
            self.logger.error(
                f"Error executing Advanced Sandwich Attack Strategy: {e} ❌"
            )
            return False

    # -------------------- Utility Methods --------------------

    async def _determine_strategy_type(self, target_tx: Dict[str, Any]) -> Optional[str]:
        """
        Determines the appropriate strategy type based on the target transaction and market conditions.

        Args:
            target_tx (Dict[str, Any]): The target transaction details.

        Returns:
            Optional[str]: The strategy type if applicable, else None.
        """
        try:
            # Check for high-value ETH transactions
            if target_tx.get("value", 0) > self.transaction_array.web3.to_wei(
                10, "ether"
            ):
                return "eth_transaction"

            # Analyze market conditions
            market_conditions = await self.market_analyzer.check_market_conditions(
                target_tx["to"]
            )
            self.logger.debug(
                f"Market conditions for {target_tx['to']}: {market_conditions}"
            )

            # Determine strategy based on market conditions and transaction details
            if market_conditions.get("high_volatility", False):
                return "sandwich_attack"
            elif target_tx.get("value", 0) > self.transaction_array.web3.to_wei(
                1, "ether"
            ):
                return "front_run"
            elif await self.market_analyzer.is_arbitrage_opportunity(target_tx):
                return "back_run"
            else:
                self.logger.debug(
                    "No suitable strategy type determined for the transaction."
                )
                return None
        except Exception as e:
            self.logger.error(
                f"Failed to determine strategy type for transaction {target_tx.get('tx_hash', '')}: {e} ❌"
            )
            return None

    async def execute_strategy_for_transaction(self, target_tx: Dict[str, Any]) -> bool:
        """
        Determines and executes the best strategy for a given transaction.

        Args:
            target_tx (Dict[str, Any]): The target transaction details.

        Returns:
            bool: True if a strategy was executed successfully, False otherwise.
        """
        strategy_type = await self._determine_strategy_type(target_tx)
        if strategy_type:
            success = await self.execute_best_strategy(target_tx, strategy_type)
            tx_hash = target_tx.get("tx_hash", "Unknown")
            if success:
                self.logger.info(
                    f"Successfully executed {strategy_type} strategy for transaction {tx_hash}. ✅"
                )
            else:
                self.logger.warning(
                    f"Failed to execute {strategy_type} strategy for transaction {tx_hash}. ⚠️"
                )
            return success
        self.logger.debug(
            f"No suitable strategy found for transaction {target_tx.get('tx_hash', '')}."
        )
        return False