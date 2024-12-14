class StrategyConfiguration:
    decay_factor: float = 0.95
    min_profit_threshold: Decimal = Decimal("0.01")
    learning_rate: float = 0.01
    exploration_rate: float = 0.1

@dataclass