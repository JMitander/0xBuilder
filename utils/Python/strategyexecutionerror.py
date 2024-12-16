class StrategyExecutionError(Exception):
    """Exception raised when a strategy execution fails.""" 
    message: str

    def __str__(self) -> str:
        return self.message