# /home/mitander/0xBuilder/utils/Python/strategyconfiguration.py
from decimal import Decimal

class StrategyConfiguration:
    """Configuration parameters for strategy execution."""
    
    def __init__(self):
        self.decay_factor: float = 0.95
        self.min_profit_threshold: Decimal = Decimal("0.01")
        self.learning_rate: float = 0.01
        self.exploration_rate: float = 0.1