# utils.py
import logging
import sys
from typing import Optional
from colorama import Fore, Style, init

init(autoreset=True)

class CustomFormatter(logging.Formatter):
    """Custom logging formatter with colors."""
    LEVEL_COLORS = {
        logging.DEBUG: f"{Fore.MAGENTA}",
        logging.INFO: f"{Fore.GREEN}",
        logging.WARNING: f"{Fore.YELLOW}",
        logging.ERROR: f"{Fore.RED}",
        logging.CRITICAL: f"{Fore.RED}",
        "RESET": "\033[0m",
    }

    COLORS = {
        "RESET": "\033[0m",
        "RED": "\033[31m",
        "GREEN": "\033[32m",
        "YELLOW": "\033[33m",
        "MAGENTA": "\033[35m",

    }

    def format(self, record: logging.LogRecord) -> str:
        """Formats a log record with colors."""
        color = self.LEVEL_COLORS.get(record.levelname, self.COLORS["RESET"])
        reset = self.COLORS["RESET"]
        record.levelname = f"{color}{record.levelname}{reset}"  # Colorize level name
        record.msg = f"{color}{record.msg}{reset}"              # Colorize message
        return super().format(record)

# Configure the logging once
def configure_logging(level: int = logging.DEBUG) -> None:
    """Configures logging with a colored formatter."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(CustomFormatter("%(asctime)s [%(levelname)s] %(message)s"))
    handler.stream.reconfigure(encoding='utf-8') # explicitly specify UTF-8 encoding
    logging.basicConfig(
        level=level,  # Global logging level
        handlers=[handler]
    )

# Factory function to get a logger instance
def getLogger(name: Optional[str] = None, level: int = logging.DEBUG) -> logging.Logger:
    """Returns a logger instance, configuring logging if it hasn't been yet."""
    if not logging.getLogger().hasHandlers():
        configure_logging(level)
        
    logger = logging.getLogger(name if name else "0xBuilder")
    return logger