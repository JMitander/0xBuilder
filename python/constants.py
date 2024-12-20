from typing import Dict

# Error codes
ERROR_MARKET_MONITOR_INIT: int = 1001
ERROR_MODEL_LOAD: int = 1002
ERROR_DATA_LOAD: int = 1003
ERROR_MODEL_TRAIN: int = 1004
ERROR_CORE_INIT: int = 1005
ERROR_WEB3_INIT: int = 1006
ERROR_CONFIG_LOAD: int = 1007

# Error messages with default fallbacks
ERROR_MESSAGES: Dict[int, str] = {
    ERROR_MARKET_MONITOR_INIT: "Market Monitor initialization failed",
    ERROR_MODEL_LOAD: "Failed to load price prediction model",
    ERROR_DATA_LOAD: "Failed to load historical training data",
    ERROR_MODEL_TRAIN: "Failed to train price prediction model",
    ERROR_CORE_INIT: "Core initialization failed",
    ERROR_WEB3_INIT: "Web3 connection failed",
    ERROR_CONFIG_LOAD: "Configuration loading failed"
}

# Add a helper function to get error message with fallback
def get_error_message(code: int, default: str = "Unknown error") -> str:
    """Get error message for error code with fallback to default message."""
    return ERROR_MESSAGES.get(code, default)