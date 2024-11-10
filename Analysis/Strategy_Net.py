class Strategy_Net:
    def __init__(
        self,
        transaction_core: Transaction_Core,
        market_monitor: Market_Monitor,
        safety_net: Safety_Net,
        api_config: API_Config,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.transaction_core = transaction_core
        self.market_monitor = market_monitor
        self.safety_net = safety_net
        self.api_config = api_config
        self.logger = logger or logging.getLogger(self.__class__.__name__)

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

        self.reinforcement_weights = {
            strategy_type: np.ones(len(self.get_strategies(strategy_type)))
            for strategy_type in ["eth_transaction", "front_run", "back_run", "sandwich_attack"]
        }

        self.configuration = {
            "decay_factor": 0.95,
            "min_profit_threshold": Decimal("0.01"),
            "learning_rate": 0.01,
            "exploration_rate": 0.1
        }

        self.history_data = []
        self.logger.info("Strategy_Net initialized with enhanced configuration ✅")

    async def execute_best_strategy(self, target_tx: Dict[str, Any], strategy_type: str) -> bool:
        """Execute the best strategy for the given strategy type."""
        strategies = self.get_strategies(strategy_type)
        if not strategies:
            self.logger.warning(f"No strategies available for type: {strategy_type}")
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
                execution_time
            )

            return success

        except Exception as e:
            self.logger.error(f"Strategy execution failed: {str(e)}", exc_info=True)
            return False

    def get_strategies(self, strategy_type: str) -> List[Any]:
        """Get the list of strategies for a given strategy type."""
        strategies_mapping = {
            "eth_transaction": [self.high_value_eth_transfer],
            "front_run": [
                self.aggressive_front_run,
                self.predictive_front_run,
                self.volatility_front_run,
                self.advanced_front_run
            ],
            "back_run": [
                self.price_dip_back_run,
                self.flashloan_back_run,
                self.high_volume_back_run,
                self.advanced_back_run
            ],
            "sandwich_attack": [
                self.flash_profit_sandwich,
                self.price_boost_sandwich,
                self.arbitrage_sandwich,
                self.advanced_sandwich_attack
            ]
        }
        return strategies_mapping.get(strategy_type, [])

    async def _select_best_strategy(self, strategies: List[Any], strategy_type: str) -> Any:
        """Select the best strategy based on reinforcement learning weights."""
        try:
            weights = self.reinforcement_weights[strategy_type]

            if random.random() < self.configuration["exploration_rate"]:
                self.logger.debug("Using exploration for strategy selection")
                return random.choice(strategies)

            exp_weights = np.exp(weights - np.max(weights))
            probabilities = exp_weights / exp_weights.sum()

            selected_index = np.random.choice(len(strategies), p=probabilities)
            return strategies[selected_index]

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
        """Update metrics for the executed strategy."""
        try:
            metrics = self.strategy_performance[strategy_type]
            metrics["total_executions"] += 1

            if success:
                metrics["successes"] += 1
                metrics["profit"] += profit
            else:
                metrics["failures"] += 1

            metrics["avg_execution_time"] = (
                metrics["avg_execution_time"] * self.configuration["decay_factor"] + execution_time * (1 - self.configuration["decay_factor"])
            )
            metrics["success_rate"] = metrics["successes"] / metrics["total_executions"]

            strategy_index = self.get_strategy_index(strategy_name, strategy_type)
            if strategy_index >= 0:
                reward = self._calculate_reward(success, profit, execution_time)
                self._update_reinforcement_weight(strategy_type, strategy_index, reward)

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

    def get_strategy_index(self, strategy_name: str, strategy_type: str) -> int:
        """Get the index of a strategy in the strategy list."""
        strategies = self.get_strategies(strategy_type)
        for index, strategy in enumerate(strategies):
            if strategy.__name__ == strategy_name:
                return index
        return -1

    def _calculate_reward(self, success: bool, profit: Decimal, execution_time: float) -> float:
        """Calculate the reward for a strategy execution."""
        base_reward = float(profit) if success else -0.1
        time_penalty = -0.01 * execution_time
        return base_reward + time_penalty

    def _update_reinforcement_weight(self, strategy_type: str, index: int, reward: float) -> None:
        """Update the reinforcement learning weight for a strategy."""
        current_weight = self.reinforcement_weights[strategy_type][index]
        new_weight = current_weight * (1 - self.configuration["learning_rate"]) + reward * self.configuration["learning_rate"]
        self.reinforcement_weights[strategy_type][index] = max(0.1, new_weight)

    async def high_value_eth_transfer(self, target_tx: Dict[str, Any]) -> bool:
        """Execute high-value ETH transfer strategy."""
        self.logger.info("Initiating High-Value ETH Transfer Strategy... 🏃💨")
        try:
            eth_value_in_wei = target_tx.get("value", 0)
            if eth_value_in_wei > self.transaction_core.web3.to_wei(10, "ether"):
                eth_value_in_eth = self.transaction_core.web3.from_wei(
                    eth_value_in_wei, "ether"
                )
                self.logger.info(
                    f"High-value ETH transfer detected: {eth_value_in_eth} ETH 🏃"
                )
                return await self.transaction_core.handle_eth_transaction(target_tx)
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
        """Execute aggressive front-run strategy."""
        self.logger.info("Initiating Aggressive Front-Run Strategy... 🏃")
        try:
            if target_tx.get("value", 0) > self.transaction_core.web3.to_wei(
                1, "ether"
            ):
                self.logger.info(
                    "Transaction value above threshold, proceeding with aggressive front-run."
                )
                return await self.transaction_core.front_run(target_tx)
            self.logger.info(
                "Transaction below threshold. Skipping aggressive front-run."
            )
            return False
        except Exception as e:
            self.logger.error(f"Error executing Aggressive Front-Run Strategy: {e} ❌")
            return False

    async def predictive_front_run(self, target_tx: Dict[str, Any]) -> bool:
        """Execute predictive front-run strategy based on price prediction."""
        self.logger.info("Initiating Predictive Front-Run Strategy... 🏃")
        try:
            decoded_tx = await self.transaction_core.decode_transaction_input(
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
            token_symbol = await self.api_config.get_token_symbol(self.transaction_core.web3, token_address)
            if not token_symbol:
                self.logger.warning(
                    f"Token symbol not found for address {token_address} in Predictive Front-Run Strategy. ❗"
                )
                return False
            predicted_price = await self.market_monitor.predict_price_movement(token_symbol)
            current_price = await self.api_config.get_real_time_price(token_symbol)
            if current_price is None:
                self.logger.warning(
                    f"Current price not available for {token_symbol} in Predictive Front-Run Strategy. ❗"
                )
                return False
            if predicted_price > float(current_price) * 1.01:
                self.logger.info(
                    "Predicted price increase exceeds threshold, proceeding with predictive front-run."
                )
                return await self.transaction_core.front_run(target_tx)
            self.logger.info(
                "Predicted price increase does not meet threshold. Skipping predictive front-run."
            )
            return False
        except Exception as e:
            self.logger.error(f"Error executing Predictive Front-Run Strategy: {e} ❌")
            return False

    async def volatility_front_run(self, target_tx: Dict[str, Any]) -> bool:
        """Execute front-run strategy based on market volatility."""
        self.logger.info("Initiating Volatility Front-Run Strategy... 🏃")
        try:
            market_conditions = await self.market_monitor.check_market_conditions(
                target_tx["to"]
            )
            if market_conditions.get("high_volatility", False):
                self.logger.info(
                    "High volatility detected, proceeding with volatility front-run."
                )
                return await self.transaction_core.front_run(target_tx)
            self.logger.info(
                "Market volatility not high enough. Skipping volatility front-run."
            )
            return False
        except Exception as e:
            self.logger.error(f"Error executing Volatility Front-Run Strategy: {e} ❌")
            return False

    async def advanced_front_run(self, target_tx: Dict[str, Any]) -> bool:
        """Execute advanced front-run strategy with comprehensive analysis."""
        self.logger.info("Initiating Advanced Front-Run Strategy... 🏃💨")
        try:
            decoded_tx = await self.transaction_core.decode_transaction_input(
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
            token_symbol = await self.api_config.get_token_symbol(self.transaction_core.web3, path[0])
            if not token_symbol:
                self.logger.warning(
                    f"Token symbol not found for address {path[0]} in Advanced Front-Run Strategy. ❗"
                )
                return False
            predicted_price = await self.market_monitor.predict_price_movement(token_symbol)
            market_conditions = await self.market_monitor.check_market_conditions(
                target_tx["to"]
            )
            current_price = await self.api_config.get_real_time_price(token_symbol)
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
                return await self.transaction_core.front_run(target_tx)
            self.logger.info(
                "Conditions not favorable for advanced front-run. Skipping."
            )
            return False
        except Exception as e:
            self.logger.error(f"Error executing Advanced Front-Run Strategy: {e} ❌")
            return False

    async def price_dip_back_run(self, target_tx: Dict[str, Any]) -> bool:
        """Execute back-run strategy based on price dip prediction."""
        self.logger.info("Initiating Price Dip Back-Run Strategy... 🔙🏃")
        try:
            decoded_tx = await self.transaction_core.decode_transaction_input(
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
            token_symbol = await self.api_config.get_token_symbol(self.transaction_core.web3, token_address)
            if not token_symbol:
                self.logger.warning(
                    f"Token symbol not found for address {token_address} in Price Dip Back-Run Strategy. ❗"
                )
                return False
            current_price = await self.api_config.get_real_time_price(token_symbol)
            if current_price is None:
                self.logger.warning(
                    f"Current price not available for {token_symbol} in Price Dip Back-Run Strategy. ❗"
                )
                return False
            predicted_price = await self.market_monitor.predict_price_movement(token_symbol)
            if predicted_price < float(current_price) * 0.99:
                self.logger.info(
                    "Predicted price decrease exceeds threshold, proceeding with price dip back-run."
                )
                return await self.transaction_core.back_run(target_tx)
            self.logger.info(
                "Predicted price decrease does not meet threshold. Skipping price dip back-run."
            )
            return False
        except Exception as e:
            self.logger.error(f"Error executing Price Dip Back-Run Strategy: {e} ❌")
            return False

    async def flashloan_back_run(self, target_tx: Dict[str, Any]) -> bool:
        """Execute back-run strategy using flash loans."""
        self.logger.info("Initiating Flashloan Back-Run Strategy... 🔙🏃")
        try:
            estimated_profit = await self.transaction_core.calculate_flashloan_amount(
                target_tx
            ) * Decimal(
                "0.02"
            )
            if estimated_profit > self.configuration["min_profit_threshold"]:
                self.logger.info(
                    "Estimated profit meets threshold, proceeding with flashloan back-run."
                )
                return await self.transaction_core.back_run(target_tx)
            self.logger.info("Profit is insufficient for flashloan back-run. Skipping.")
            return False
        except Exception as e:
            self.logger.error(f"Error executing Flashloan Back-Run Strategy: {e} ❌")
            return False

    async def high_volume_back_run(self, target_tx: Dict[str, Any]) -> bool:
        """Execute back-run strategy based on high trading volume."""
        self.logger.info("Initiating High Volume Back-Run Strategy... 🔙🏃")
        try:
            token_address = target_tx.get("to")
            token_symbol = await self.api_config.get_token_symbol(self.transaction_core.web3, token_address)
            if not token_symbol:
                self.logger.warning(f"Could not find token symbol for {token_address}")
                return False

            volume_24h = await self.api_config.get_token_volume(token_symbol)
            volume_threshold = self._get_volume_threshold(token_symbol)

            if volume_24h > volume_threshold:
                self.logger.info(f"High volume detected ({volume_24h:,.2f} USD), proceeding with back-run")
                return await self.transaction_core.back_run(target_tx)

            self.logger.info(f"Volume ({volume_24h:,.2f} USD) below threshold ({volume_threshold:,.2f} USD)")
            return False

        except Exception as e:
            self.logger.error(f"High Volume Back-Run failed: {str(e)}", exc_info=True)
            return False

    def _get_volume_threshold(self, token_symbol: str) -> float:
        """Determine the volume threshold for a token."""
        thresholds = {
            'WETH': 5_000_000,
            'USDT': 10_000_000,
            'USDC': 10_000_000,
            'default': 1_000_000
        }
        return thresholds.get(token_symbol, thresholds['default'])

    async def advanced_back_run(self, target_tx: Dict[str, Any]) -> bool:
        """Execute advanced back-run strategy with comprehensive analysis."""
        self.logger.info("Initiating Advanced Back-Run Strategy... 🔙🏃💨")
        try:
            decoded_tx = await self.transaction_core.decode_transaction_input(
                target_tx["input"], target_tx["to"]
            )
            if not decoded_tx:
                self.logger.warning("Failed to decode transaction input for advanced back-run")
                return False

            market_conditions = await self.market_monitor.check_market_conditions(
                target_tx["to"]
            )
            if market_conditions.get("high_volatility", False) and market_conditions.get("bullish_trend", False):
                self.logger.info("Market conditions favorable for advanced back-run")
                return await self.transaction_core.back_run(target_tx)

            self.logger.info("Market conditions unfavorable for advanced back-run")
            return False

        except Exception as e:
            self.logger.error(f"Advanced Back-Run failed: {str(e)}", exc_info=True)
            return False

    async def flash_profit_sandwich(self, target_tx: Dict[str, Any]) -> bool:
        """Execute sandwich attack strategy using flash loans."""
        self.logger.info("Initiating Flash Profit Sandwich Strategy... 🥪🏃")
        try:
            estimated_profit = await self.transaction_core.calculate_flashloan_amount(
                target_tx
            ) * Decimal(
                "0.02"
            )
            if estimated_profit > self.configuration["min_profit_threshold"]:
                gas_price = await self.transaction_core.get_dynamic_gas_price()
                if gas_price > 200:
                    self.logger.warning(f"Gas price too high for sandwich attack: {gas_price} Gwei")
                    return False

                self.logger.info(f"Executing sandwich with estimated profit: {estimated_profit:.4f} ETH")
                return await self.transaction_core.execute_sandwich_attack(target_tx)
            self.logger.info("Insufficient profit potential for flash sandwich")
            return False
        except Exception as e:
            self.logger.error(f"Flash Profit Sandwich failed: {str(e)}", exc_info=True)
            return False

    async def price_boost_sandwich(self, target_tx: Dict[str, Any]) -> bool:
        """Execute sandwich attack strategy based on price momentum."""
        self.logger.info("Initiating Price Boost Sandwich Strategy... 🥪🏃")
        try:
            decoded_tx = await self.transaction_core.decode_transaction_input(
                target_tx["input"], target_tx["to"]
            )
            if not decoded_tx:
                self.logger.warning("Failed to decode transaction input for price boost sandwich")
                return False

            params = decoded_tx.get("params", {})
            path = params.get("path", [])
            if not path:
                self.logger.warning("Transaction has no path parameter for price boost sandwich")
                return False

            token_symbol = await self.api_config.get_token_symbol(self.transaction_core.web3, path[0])
            if not token_symbol:
                self.logger.warning(f"Token symbol not found for address {path[0]}")
                return False

            historical_prices = await self.market_monitor.fetch_historical_prices(token_symbol)
            if not historical_prices:
                self.logger.warning(f"No historical prices found for {token_symbol}")
                return False

            momentum = await self._analyze_price_momentum(historical_prices)
            if momentum > 0.02:
                self.logger.info(f"Strong price momentum detected: {momentum:.2%}")
                return await self.transaction_core.execute_sandwich_attack(target_tx)

            self.logger.info(f"Insufficient price momentum: {momentum:.2%}")
            return False

        except Exception as e:
            self.logger.error(f"Price Boost Sandwich failed: {str(e)}", exc_info=True)
            return False

    async def _analyze_price_momentum(self, prices: List[float]) -> float:
        """Analyze price momentum from historical prices."""
        try:
            if not prices:
                self.logger.warning("No price data found for momentum analysis")
                return 0.0

            price_changes = [prices[i] / prices[i - 1] - 1 for i in range(1, len(prices))]
            momentum = sum(price_changes) / len(price_changes)

            return momentum

        except Exception as e:
            self.logger.error(f"Price momentum analysis failed: {str(e)}", exc_info=True)
            return 0.0

    async def arbitrage_sandwich(self, target_tx: Dict[str, Any]) -> bool:
        """Execute sandwich attack strategy based on arbitrage opportunities."""
        self.logger.info("Initiating Arbitrage Sandwich Strategy... 🥪🏃")
        try:
            decoded_tx = await self.transaction_core.decode_transaction_input(
                target_tx["input"], target_tx["to"]
            )
            if not decoded_tx:
                self.logger.warning("Failed to decode transaction input for arbitrage sandwich")
                return False

            params = decoded_tx.get("params", {})
            path = params.get("path", [])
            if not path:
                self.logger.warning("Transaction has no path parameter for arbitrage sandwich")
                return False

            token_address = path[-1]
            token_symbol = await self.api_config.get_token_symbol(self.transaction_core.web3, token_address)
            if not token_symbol:
                self.logger.warning(f"Token symbol not found for address {token_address}")
                return False

            is_arbitrage = await self.market_monitor.is_arbitrage_opportunity(target_tx)
            if is_arbitrage:
                self.logger.info(f"Arbitrage opportunity detected for {token_symbol}")
                return await self.transaction_core.execute_sandwich_attack(target_tx)

            self.logger.info("No profitable arbitrage opportunity found")
            return False

        except Exception as e:
            self.logger.error(f"Arbitrage Sandwich failed: {str(e)}", exc_info=True)
            return False

    async def advanced_sandwich_attack(self, target_tx: Dict[str, Any]) -> bool:
        """Execute advanced sandwich attack strategy with risk management."""
        self.logger.info("Initiating Advanced Sandwich Attack Strategy... 🥪🏃💨")
        try:
            decoded_tx = await self.transaction_core.decode_transaction_input(
                target_tx["input"], target_tx["to"]
            )
            if not decoded_tx:
                self.logger.warning("Failed to decode transaction input for advanced sandwich attack")
                return False

            market_conditions = await self.market_monitor.check_market_conditions(
                target_tx["to"]
            )
            if market_conditions.get("high_volatility", False) and market_conditions.get("bullish_trend", False):
                self.logger.info("Conditions favorable for advanced sandwich attack")
                return await self.transaction_core.execute_sandwich_attack(target_tx)

            self.logger.info("Conditions unfavorable for advanced sandwich attack")
            return False

        except Exception as e:
            self.logger.error(f"Advanced Sandwich Attack failed: {str(e)}", exc_info=True)
            return False