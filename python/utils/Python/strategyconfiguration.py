from decimal import Decimal

class StrategyConfiguration:
    """Configuration parameters for strategy execution."""
    
    def __init__(self):
        self.decay_factor = 0.95
        self.min_profit_threshold = Decimal("0.01")
        self.learning_rate = 0.01
        self.exploration_rate = 0.1
