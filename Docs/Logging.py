import os
import random
import sys
import dotenv
import time
import logging
import json
import asyncio
import aiofiles
import aiohttp
import numpy as np
import tracemalloc
import hexbytes
from cachetools import TTLCache
from sklearn.linear_model import LinearRegression
from decimal import Decimal
from typing import List, Dict, Any, Optional, Tuple, Union

from eth_account.messages import *
from eth_account.signers.local import *
from eth_abi import *
from eth_utils import *
from eth_typing import *
from eth_account import *

from web3.middleware import *
from web3.providers import *
from web3.types import *
from web3.geth import *
from web3.exceptions import *
from web3.utils import *
from web3.contract import *
from web3.eth import *

#//////////////////////////////////////////////////////////////////////////////

dotenv.load_dotenv()

async def loading_bar(
    message: str,
    total_time: int,
    error_message: Optional[str] = None,
    success_message: Optional[str] = None,
) -> None:
    """Displays a loading bar in the console."""
    bar_length = 20
    try:
        for i in range(101):
            await asyncio.sleep(total_time / 100)
            percent = i / 100
            filled_length = int(percent * bar_length)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            sys.stdout.write(f"\r{message} [{bar}] {i}%")
            sys.stdout.flush()
        sys.stdout.write("\n")
        sys.stdout.flush()

        final_message = success_message or "Success"
        sys.stdout.write(f"{message} [{'█' * bar_length}] 100% - ✅ {final_message}\n")
        sys.stdout.flush()
    except Exception as e:
        error_msg = error_message or f"Error: {e}"
        sys.stdout.write(f"\r{message} [{'█' * bar_length}] 100% - ❌ {error_msg}\n")
        sys.stdout.flush()


async def setup_logging() -> None:
    """Sets up logging configuration."""
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    tracemalloc.start()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_formatter)

    file_handler = logging.FileHandler("0xplorer_log.txt", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.info("Warming up peripherals... 🚀")

#//////////////////////////////////////////////////////////////////////////////