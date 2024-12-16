#!/usr/bin/env python3
import os
import sys
import asyncio
import logging
import signal
import tracemalloc

# Add the python directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'python'))

# Import core components
from core.main_core import Main_Core
from configuration import Configuration

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """Main entry point with graceful shutdown handling."""
    try:
        # Start memory tracking
        tracemalloc.start()
        logger.info("Starting 0xBuilder...")

        # Initialize configuration
        configuration = Configuration()
        
        # Create and initialize main core
        core = Main_Core(configuration)
        
        def shutdown_handler():
            logger.info("Shutdown signal received")
            asyncio.create_task(core.stop())

        # Set up signal handlers
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, shutdown_handler)

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
        # Clean shutdown
        if 'core' in locals():
            await core.stop()
        
        # Stop memory tracking
        tracemalloc.stop()
        logger.info("0xBuilder shutdown complete")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
