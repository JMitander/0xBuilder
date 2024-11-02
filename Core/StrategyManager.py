class StrategyManager:
    def __init__(
        self,
        transaction_array: TransactionArray,
        market_analyzer: MarketAnalyzer,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.transaction_array = transaction_array
        self.market_analyzer = market_analyzer
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        
        # Enhanced performance tracking with more metrics
        self.strategy_performance = {
            strategy_type: {
                "successes": 0,
                "failures": 0,
                "profit": Decimal("0"),
                "avg_execution_time": 0.0,
                "success_rate": 0.0,
                "total_executions": 0
            }
            for strategy_type in ["eth_transaction", "front_run", "back_run", "sandwich_attack"]
        }
        
        # Initialize ML components and performance tracking
        self.price_model = LinearRegression()
        self.model_last_updated = 0
        self.MODEL_UPDATE_INTERVAL = 3600  # Update model hourly

        # Dynamic reinforcement learning weights
        self.reinforcement_weights = {
            strategy_type: np.ones(len(self.get_strategies(strategy_type)))
            for strategy_type in ["eth_transaction", "front_run", "back_run", "sandwich_attack"]
        }

        # Configuration parameters with adaptive thresholds
        self.config = {
            "decay_factor": 0.95,
            "min_profit_threshold": Decimal("0.01"),
            "learning_rate": 0.01,
            "exploration_rate": 0.1
        }

        self.history_data = []
        self.logger.info("StrategyManager initialized with enhanced configuration ✅")

    async def execute_best_strategy(self, target_tx: Dict[str, Any], strategy_type: str) -> bool:
        strategies = self.get_strategies(strategy_type)
        if not strategies:
            self.logger.warning(f"No strategies available for type: {strategy_type}")
            return False

        try:
            # Track execution time and performance
            start_time = time.time()
            selected_strategy = await self._select_best_strategy(strategies, strategy_type)
            
            # Execute strategy with detailed profit tracking
            profit_before = await self.transaction_array.get_current_profit()
            success = await selected_strategy(target_tx)
            profit_after = await self.transaction_array.get_current_profit()
            
            # Calculate performance metrics
            execution_time = time.time() - start_time
            profit_made = profit_after - profit_before

            # Update performance metrics
            await self._update_strategy_metrics(
                selected_strategy.__name__,
                strategy_type,
                success,
                profit_made,
                execution_time
            )

            return success

        except Exception as e:
            self.logger.error(f"Strategy execution failed: {str(e)}", exc_info=True)
            return False

    async def _select_best_strategy(self, strategies: List[Any], strategy_type: str) -> Any:
        """Enhanced strategy selection using performance metrics and exploration"""
        try:
            weights = self.reinforcement_weights[strategy_type]
            
            # Apply exploration vs exploitation
            if random.random() < self.config["exploration_rate"]:
                self.logger.debug("Using exploration for strategy selection")
                return random.choice(strategies)
            
            # Use softmax for better weight normalization
            exp_weights = np.exp(weights - np.max(weights))
            probabilities = exp_weights / exp_weights.sum()
            
            return strategies[np.random.choice(len(strategies), p=probabilities)]

        except Exception as e:
            self.logger.error(f"Strategy selection failed: {str(e)}", exc_info=True)
            return random.choice(strategies)

    async def _update_strategy_metrics(
        self,
        strategy_name: str,
        strategy_type: str,
        success: bool,
        profit: Decimal,
        execution_time: float
    ) -> None:
        """Enhanced performance metrics tracking"""
        try:
            metrics = self.strategy_performance[strategy_type]
            metrics["total_executions"] += 1
            
            if success:
                metrics["successes"] += 1
                metrics["profit"] += profit
            else:
                metrics["failures"] += 1

            # Update moving averages
            metrics["avg_execution_time"] = (
                metrics["avg_execution_time"] * 0.95 + execution_time * 0.05
            )
            metrics["success_rate"] = metrics["successes"] / metrics["total_executions"]

            # Update reinforcement weights with more sophisticated approach
            strategy_index = self.get_strategy_index(strategy_name, strategy_type)
            if strategy_index >= 0:
                reward = self._calculate_reward(success, profit, execution_time)
                self._update_reinforcement_weight(strategy_type, strategy_index, reward)

            # Store historical data
            self.history_data.append({
                "timestamp": time.time(),
                "strategy_name": strategy_name,
                "success": success,
                "profit": float(profit),
                "execution_time": execution_time,
                "total_profit": float(metrics["profit"])
            })

        except Exception as e:
            self.logger.error(f"Error updating metrics: {str(e)}", exc_info=True)

    def _calculate_reward(self, success: bool, profit: Decimal, execution_time: float) -> float:
        """Sophisticated reward calculation considering multiple factors"""
        base_reward = float(profit) if success else -0.1
        time_penalty = -0.01 * execution_time  # Penalize long execution times
        return base_reward + time_penalty

    def _update_reinforcement_weight(self, strategy_type: str, index: int, reward: float) -> None:
        """Update weights using exponential moving average"""
        current_weight = self.reinforcement_weights[strategy_type][index]
        new_weight = current_weight * (1 - self.config["learning_rate"]) + reward * self.config["learning_rate"]
        self.reinforcement_weights[strategy_type][index] = max(0.1, new_weight)

    async def predict_price_movement(self, token_symbol: str) -> float:
        """Enhanced price prediction with model updates and validation"""
        try:
            current_time = time.time()
            
            # Update model periodically
            if current_time - self.model_last_updated > self.MODEL_UPDATE_INTERVAL:
                prices = await self.market_analyzer.fetch_historical_prices(token_symbol)
                if len(prices) > 10:  # Ensure sufficient data
                    X = np.arange(len(prices)).reshape(-1, 1)
                    y = np.array(prices)
                    self.price_model.fit(X, y)
                    self.model_last_updated = current_time
                    
            # Make prediction
            next_time = np.array([[len(prices)]])
            predicted_price = self.price_model.predict(next_time)[0]
            
            self.logger.debug(f"Price prediction for {token_symbol}: {predicted_price}")
            return float(predicted_price)

        except Exception as e:
            self.logger.error(f"Price prediction failed: {str(e)}", exc_info=True)
            return 0.0

        except Exception as e:
            self.logger.error(f"Strategy type determination failed: {str(e)}", exc_info=True)
            return None

    async def high_value_eth_transfer(self, target_tx: Dict[str, Any]) -> bool:
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

    async def _determine_strategy_type(self, target_tx: Dict[str, Any]) -> Optional[str]:
        """Enhanced strategy type determination with market analysis"""
        try:
            # Analyze transaction value
            tx_value = target_tx.get("value", 0)
            if tx_value > self.transaction_array.web3.to_wei(10, "ether"):
                return "eth_transaction"

            # Get market conditions and metrics
            market_conditions = await self.market_analyzer.check_market_conditions(target_tx["to"])
            is_arbitrage = await self.market_analyzer.is_arbitrage_opportunity(target_tx)

            # Make decision based on multiple factors
            if market_conditions.get("high_volatility", False) and tx_value > self.transaction_array.web3.to_wei(1, "ether"):
                return "sandwich_attack"
            elif is_arbitrage:
                return "front_run" if market_conditions.get("bullish_trend", False) else "back_run"
            elif tx_value > self.transaction_array.web3.to_wei(1, "ether"):
                return "front_run"

            return None

        except Exception as e:
            self.logger.error(f"Strategy type determination failed: {str(e)}", exc_info=True)
            return None
        
    async def execute_strategy_for_transaction(self, target_tx: Dict[str, Any]) -> bool:
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