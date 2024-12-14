class ColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        color = COLORS.get(record.levelname, COLORS["RESET"])
        reset = COLORS["RESET"]
        record.levelname = f"{color}{record.levelname}{reset}"  # Colorize level name
        record.msg = f"{color}{record.msg}{reset}"              # Colorize message
        return super().format(record)

# Configure the logging once
def configure_logging():
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColorFormatter("%(asctime)s [%(levelname)s] %(message)s"))

    logging.basicConfig(
        level=logging.DEBUG,  # Global logging level
        handlers=[handler]
    )

# Helper function to get the logger
def get_logger(name: Optional[str] = None) -> logging.Logger:
    if not logging.getLogger().hasHandlers():
        configure_logging()
        
    logger = logging.getLogger(name if name else "0xBuilder")
    return logger

# Initialize the logger globally so it can be used throughout the script
logger = get_logger("0xBuilder")

dotenv.load_dotenv()


# ////////////////////////////////////////////////////////////////////////////