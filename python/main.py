import asyncio
import logging
import tracemalloc
from typing import Optional

from main_core import Main_Core
from configuration import Configuration
from utils import configure_logging, getLogger  # Moved logging setup to utils

# Initialize the logger before everything else
logger = getLogger("0xBuilder")

async def main():
    """Main entry point with graceful shutdown handling."""
    loop = asyncio.get_running_loop()
    shutdown_handler_task: Optional[asyncio.Task] = None

    try:
        # Start memory tracking
        tracemalloc.start()
        logger.info("Starting 0xBuilder...")

        # Initialize configuration
        configuration = Configuration()

        # Create and initialize main core
        core = Main_Core(configuration)

        # Initialize and run
        await core.initialize()
        await core.run()

    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        if tracemalloc.is_tracing():
            snapshot = tracemalloc.take_snapshot()
            logger.debug("Top 10 memory allocations:")
            for stat in snapshot.statistics('lineno')[:10]:
                logger.debug(str(stat))
    finally:
        # Stop memory tracking
        tracemalloc.stop()
        logger.info("0xBuilder shutdown complete")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        # Get current memory snapshot on error
        snapshot = tracemalloc.take_snapshot()
        logger.critical(f"Program terminated with an error: {e}")
        logger.debug("Top 10 memory allocations at error:")
        top_stats = snapshot.statistics('lineno')
        for stat in top_stats[:10]:
            logger.debug(str(stat))