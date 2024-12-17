class StrategyExecutionError(Exception):
    """Custom exception for strategy execution failures."""
    def __init__(self, message: str = "Strategy execution failed"):
        self.message = message
        super().__init__(self.message)
