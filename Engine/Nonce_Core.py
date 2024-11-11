class Nonce_Core:
    """
    Advanced nonce management system for Ethereum transactions with caching,
    auto-recovery, and comprehensive error handling.
    """

    def __init__(
        self,
        web3: AsyncWeb3,
        address: str,
        
        max_retries: int = 3,
        retry_delay: float = 1.0,
        cache_ttl: int = 300,  # Cache TTL in seconds
    ):
        self.web3 = web3
        self.address = self.web3.to_checksum_address(address)
        
        self.max_retries = max(1, max_retries)
        self.retry_delay = max(0.1, retry_delay)
        self.cache_ttl = cache_ttl

        # Thread-safe primitives
        self.lock = asyncio.Lock()
        self.nonce_cache = TTLCache(maxsize=1, ttl=cache_ttl)
        self.last_sync = 0.0
        self.pending_transactions = set()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the nonce manager with error recovery."""
        try:
            async with self.lock:
                if not self._initialized:
                    await self._init_nonce()
                    self._initialized = True
                    logger.debug(
                        f"Nonce_Core initialized for {self.address[:10]}... "
                    )
        except Exception as e:
            logger.debug(f"Initialization failed: {e} !")
            raise RuntimeError("Nonce_Core initialization failed") from e

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

        async with self.lock:
            try:
                if force_refresh or self._should_refresh_cache():
                    await self.refresh_nonce()

                current_nonce = self.nonce_cache.get(self.address, 0)
                next_nonce = current_nonce + 1
                self.nonce_cache[self.address] = next_nonce

                logger.debug(
                     f"Allocated nonce {current_nonce} for {self.address[:10]}... "
                )
                return current_nonce

            except Exception as e:
                logger.error(f"error getting nonce: {e} !")
                await self._handle_nonce_error()
                raise

    async def refresh_nonce(self) -> None:
        """Refresh nonce from chain with conflict resolution."""
        async with self.lock:
            try:
                chain_nonce = await self._fetch_current_nonce_with_retries()
                cached_nonce = self.nonce_cache.get(self.address, 0)
                pending_nonce = await self._get_pending_nonce()

                # Take the highest nonce to avoid conflicts
                new_nonce = max(chain_nonce, cached_nonce, pending_nonce)
                self.nonce_cache[self.address] = new_nonce
                self.last_sync = time.monotonic()

                logger.debug(f"Nonce refreshed to {new_nonce} ")

            except Exception as e:
                logger.debug(f"Nonce refresh failed: {e} !")
                raise

    async def _fetch_current_nonce_with_retries(self) -> int:
        """Fetch current nonce with exponential backoff."""
        backoff = self.retry_delay

        for attempt in range(self.max_retries):
            try:
                return await self.web3.eth.get_transaction_count(
                    self.address, block_identifier="pending"
                )
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.debug(f"Nonce fetch failed after retries: {e} !")
                    raise
                logger.debug(
                     f"Nonce fetch attempt {attempt + 1} failed: {e}. Retrying in {backoff}s... "
                )
                await asyncio.sleep(backoff)
                backoff *= 2

    async def _get_pending_nonce(self) -> int:
        """Get highest nonce from pending transactions."""
        try:
            pending_nonces = [int(nonce) for nonce in self.pending_transactions]
            return max(pending_nonces) + 1 if pending_nonces else 0
        except Exception as e:
            logger.error(f"error getting pending nonce: {e} !")
            return 0

    async def track_transaction(self, tx_hash: str, nonce: int) -> None:
        """Track pending transaction for nonce management."""
        self.pending_transactions.add(nonce)
        try:
            # Wait for transaction confirmation
            await self.web3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            self.pending_transactions.discard(nonce)
        except Exception as e:
            logger.debug(f"Transaction tracking failed: {e} !")
        finally:
            self.pending_transactions.discard(nonce)

    async def _handle_nonce_error(self) -> None:
        """Handle nonce-related errors with recovery attempt."""
        try:
            await self.sync_nonce_with_chain()
        except Exception as e:
            logger.debug(f"Nonce error recovery failed: {e} !")
            raise

    async def sync_nonce_with_chain(self) -> None:
        """Force synchronization with chain state."""
        async with self.lock:
            try:
                await loading_bar("Synchronizing Nonce", 1)
                new_nonce = await self._fetch_current_nonce_with_retries()
                self.nonce_cache[self.address] = new_nonce
                self.last_sync = time.monotonic()
                self.pending_transactions.clear()
                logger.debug(f"Nonce synchronized to {new_nonce} ")
            except Exception as e:
                logger.debug(f"Nonce synchronization failed: {e} !")
                raise

    def _should_refresh_cache(self) -> bool:
        """Determine if cache refresh is needed."""
        return time.monotonic() - self.last_sync > (self.cache_ttl / 2)

    async def reset(self) -> None:
        """Complete reset of nonce manager state."""
        async with self.lock:
            try:
                self.nonce_cache.clear()
                self.pending_transactions.clear()
                self.last_sync = 0.0
                self._initialized = False
                await self.initialize()
                logger.debug(f"Nonce_Core reset complete ")
            except Exception as e:
                logger.debug(f"Reset failed: {e} !")
                raise

#//////////////////////////////////////////////////////////////////////////////