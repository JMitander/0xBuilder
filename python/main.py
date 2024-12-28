# /home/mitander/0xBuilder/main.py
#!/usr/bin/env python3
import asyncio
import logging
import signal
import tracemalloc
import logging

from configuration import Configuration
from core import Main_Core

# Get the logger
logger = logging.getLogger("Main")

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
         # Remove signal handlers and perform clean shutdown
         try:
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                 loop.remove_signal_handler(sig)
         except Exception as e:
            logger.error(f"Error removing signal handlers during shutdown: {e}")
         finally:
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
    except Exception as e:
        # Get current memory snapshot on error
        snapshot = tracemalloc.take_snapshot()
        logger.critical(f"Program terminated with an error: {e}")
        logger.debug("Top 10 memory allocations at error:")
        top_stats = snapshot.statistics('lineno')
        for stat in top_stats[:10]:
            logger.debug(str(stat))