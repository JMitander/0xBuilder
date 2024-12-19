import logging
import sys
from typing import Optional, Dict
import dotenv

# Load environment variables
dotenv.load_dotenv()

class ColorFormatter(logging.Formatter):
    """Custom formatter for colored log output."""
    COLORS = {
        "DEBUG": "\033[94m",    # Blue
        "INFO": "\033[92m",     # Green
        "WARNING": "\033[93m",  # Yellow
        "ERROR": "\033[91m",    # Red
        "CRITICAL": "\033[91m\033[1m", # Bold Red
        "RESET": "\033[0m",     # Reset
    }

    def format(self, record: logging.LogRecord) -> str:
        """Formats a log record with colors."""
        color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        reset = self.COLORS["RESET"]
        record.levelname = f"{color}{record.levelname}{reset}"  # Colorize level name
        record.msg = f"{color}{record.msg}{reset}"              # Colorize message
        return super().format(record)

# Configure the logging once
def configure_logging(level: int = logging.DEBUG) -> None:
    """Configures logging with a colored formatter."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColorFormatter("%(asctime)s [%(levelname)s] %(message)s"))

    logging.basicConfig(
        level=level,  # Global logging level
        handlers=[handler]
    )

# Factory function to get a logger instance
def get_logger(name: Optional[str] = None, level: int = logging.DEBUG) -> logging.Logger:
    """Returns a logger instance, configuring logging if it hasn't been yet."""
    if not logging.getLogger().hasHandlers():
        configure_logging(level)
        
    logger = logging.getLogger(name if name else "0xBuilder")
    return logger

# Initialize the logger globally so it can be used throughout the script
logger = get_logger("0xBuilder")