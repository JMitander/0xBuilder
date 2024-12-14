class Nonce_Core:
    """
    Advanced nonce management system for Ethereum transactions with caching,
    auto-recovery, and comprehensive error handling.
    """

    MAX_RETRIES = 3
    RETRY_DELAY = 1.0
    CACHE_TTL = 300  # Cache TTL in seconds

    def __init__(
        self,
        web3: AsyncWeb3,
        address: str,
        configuration: Configuration,
    ):
        self.pending_transactions = set()
        self.web3 = web3
        self.configuration = configuration
        self.address = address
        self.lock = asyncio.Lock()
        self.nonce_cache = TTLCache(maxsize=1, ttl=self.CACHE_TTL)
        self.last_sync = time.monotonic()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the nonce manager with error recovery."""
        try:
            await self._init_nonce()
            self._initialized = True
            logger.debug("Noncecore initialized ✅")
        except Exception as e:
            logger.error(f"Failed to initialize Nonce_Core: {e}")
            raise

    async def _init_nonce(self) -> None:
        """Initialize nonce with fallback mechanisms."""
        current_nonce = await self._fetch_current_nonce_with_retries()
        pending_nonce = await self._get_pending_nonce()
        # Use the higher of current or pending nonce
        self.nonce_cache[self.address] = max(current_nonce, pending_nonce)
        self.last_sync = time.monotonic()

    async def get_nonce(self, force_refresh: bool = False) -> int:
        """Get next available nonce with optional force refresh."""
        if not self._initialized:
            await self.initialize()

        if force_refresh or self._should_refresh_cache():
            await self.refresh_nonce()

        return self.nonce_cache.get(self.address, 0)

    async def refresh_nonce(self) -> None:
        """Refresh nonce from chain with conflict resolution."""
        async with self.lock:
            current_nonce = await self.web3.eth.get_transaction_count(self.address)
            self.nonce_cache[self.address] = current_nonce
            self.last_sync = time.monotonic()
            logger.debug(f"Nonce refreshed to {current_nonce}.")

    async def _fetch_current_nonce_with_retries(self) -> int:
        """Fetch current nonce with exponential backoff."""
        backoff = self.RETRY_DELAY
        for attempt in range(self.MAX_RETRIES):
            try:
                nonce = await self.web3.eth.get_transaction_count(self.address)
                logger.debug(f"Fetched nonce: {nonce}")
                return nonce
            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed to fetch nonce: {e}")
                await asyncio.sleep(backoff)
                backoff *= 2
        raise RuntimeError("Failed to fetch current nonce after retries")

    async def _get_pending_nonce(self) -> int:
        """Get highest nonce from pending transactions."""
        try:
            pending = await self.web3.eth.get_transaction_count(self.address, 'pending')
            logger.info(f"NonceCore Reports pending nonce: {pending}")
            return pending
        except Exception as e:
            logger.error(f"Error fetching pending nonce: {e}")
            return await self._fetch_current_nonce_with_retries()

    async def track_transaction(self, tx_hash: str, nonce: int) -> None:
        """Track pending transaction for nonce management."""
        self.pending_transactions.add(nonce)
        try:
            receipt = await self.web3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            if receipt.status == 1:
                logger.info(f"Transaction {tx_hash} succeeded.")
            else:
                logger.error(f"Transaction {tx_hash} failed.")
        except Exception as e:
            logger.error(f"Error tracking transaction {tx_hash}: {e}")
        finally:
            self.pending_transactions.discard(nonce)

    async def _handle_nonce_error(self) -> None:
        """Handle nonce-related errors with recovery attempt."""
        logger.warning("Handling nonce-related error. Refreshing nonce.")
        await self.refresh_nonce()

    async def sync_nonce_with_chain(self) -> None:
        """Force synchronization with chain state."""
        async with self.lock:
            await self.refresh_nonce()

    async def reset(self) -> None:
        """Complete reset of nonce manager state."""
        async with self.lock:
            self.nonce_cache.clear()
            self.pending_transactions.clear()
            await self.refresh_nonce()
            logger.debug("NonceCore reset. OK ✅")

    async def stop(self) -> None:
        """Stop nonce manager operations."""
        if not self._initialized:
            return
        try:
            await self.reset()
            logger.debug("Nonce Core stopped successfully.")
        except Exception as e:
            logger.error(f"Error stopping Nonce Core: {e}")

#//////////////////////////////////////////////////////////////////////////////