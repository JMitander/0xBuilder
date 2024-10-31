class NonceManager:
    """
    Manages the nonce for an Ethereum account to prevent transaction nonce collisions.

    Args:
        web3 (AsyncWeb3): AsyncWeb3 instance connected to the Ethereum network.
        address (str): The Ethereum address for which the nonce is managed.
        logger (Optional[logging.Logger]): Logger instance.
        max_retries (int): Maximum number of retries for nonce fetch.
        retry_delay (float): Delay between retries in seconds.
    
    Attributes:
        web3 (AsyncWeb3): AsyncWeb3 instance connected to the Ethereum network.
        address (str): The Ethereum address for which the nonce is managed.
        logger (logging.Logger): Logger instance.
        max_retries (int): Maximum number of retries for nonce fetch.
        retry_delay (float): Delay between retries in seconds.
        lock (asyncio.Lock): Lock to ensure atomic nonce updates.
        current_nonce (Optional[int]): The current nonce value.
    """

    def __init__(
        self,
        web3: AsyncWeb3,
        address: str,
        logger: Optional[logging.Logger] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self.web3 = web3
        self.address = address
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.max_retries = max(1, max_retries)  # Ensure at least one retry
        self.retry_delay = max(0.1, retry_delay)  # Ensure delay is positive
        self.lock = asyncio.Lock()
        self.current_nonce = None  # Will be initialized in the async method

    async def initialize(self):
        self.current_nonce = await self._fetch_current_nonce_with_retries()

    async def _fetch_current_nonce_with_retries(self) -> int:
        """ Fetches the current nonce with retry mechanism. """
        await loading_bar("Fetching Current Nonce", 0)
        attempt = 0
        while attempt < self.max_retries:
            try:
                nonce = await self.web3.eth.get_transaction_count(
                    self.address, block_identifier="pending"
                )
                self.logger.debug(
                    f"Initialized NonceManager for {self.address} with starting nonce {nonce}. ✅"
                )
                return nonce
            except Exception as e:
                attempt += 1
                self.logger.error(
                    f"Attempt {attempt} - Failed to fetch nonce for {self.address}: {e}. Retrying... ⚠️🔄"
                )
                await asyncio.sleep(self.retry_delay)
        self.logger.error(
            f"Failed to fetch nonce for {self.address} after {self.max_retries} attempts. ❌"
        )
        raise RuntimeError(
            f"Could not fetch nonce for {self.address} after multiple attempts. ❌"
        )

    async def get_nonce(self) -> int:
        """
        Retrieves the next nonce safely for the address. This method locks the nonce manager to avoid
        concurrent modifications, ensuring each transaction has a unique nonce.

        Returns:
            int: The next nonce for the address.
        
        Raises:
            RuntimeError: If the nonce fetch fails after multiple attempts.
        """
        async with self.lock:
            nonce = self.current_nonce
            self.current_nonce += 1
            self.logger.debug(
                f"Allocated nonce {nonce} for {self.address}. Next nonce will be {self.current_nonce}."
            )
            return nonce

    async def refresh_nonce(self):
        """
        Updates the current nonce value by fetching the latest nonce from the blockchain.
        This is a soft refresh and only updates if the on-chain nonce is higher than the internal one.

        Raises:
            RuntimeError: If the nonce refresh fails after multiple attempts.
        """
        async with self.lock:
            latest_nonce = await self._fetch_current_nonce_with_retries()
            if latest_nonce > self.current_nonce:
                self.logger.debug(
                    f"Refreshing nonce. Updated from {self.current_nonce} to {latest_nonce}. 🔄"
                )
                self.current_nonce = latest_nonce
            else:
                self.logger.debug(
                    f"No refresh needed. Current nonce {self.current_nonce} is already in sync. ✨✅"
                )

    async def sync_nonce_with_chain(self):
        """
        This method is a more aggressive sync that forces the nonce to synchronize with the blockchain.
        Useful in case of transaction reverts or nonce conflicts.

        Raises:
            RuntimeError: If the nonce sync fails after multiple attempts.
        """
        async with self.lock:
            await loading_bar("Synchronizing Nonce", 0)
            try:
                self.current_nonce = await self._fetch_current_nonce_with_retries()
                self.logger.debug(
                    f"Nonce synchronized successfully to {self.current_nonce}. ✨"
                )
            except Exception as e:
                self.logger.error(f"Failed to sync nonce for {self.address}: {e} ❌")
                raise RuntimeError(f"Failed to synchronize nonce: {e} ❌")

    async def handle_nonce_discrepancy(self, external_nonce: int):
        """
        Adjusts the current nonce if an external nonce (e.g., from a failed transaction) is higher
        than the internally managed nonce. This helps resolve nonce conflicts.

        Args:
            external_nonce (int): The externally detected nonce (e.g., from a failed or pending transaction).
        """
        async with self.lock:
            if external_nonce > self.current_nonce:
                self.logger.warning(
                    f"Discrepancy detected: External nonce {external_nonce} is higher than internal nonce {self.current_nonce}. Adjusting. ⚠️"
                )
                self.current_nonce = (
                    external_nonce + 1
                )  # Move to the next available nonce
                self.logger.debug(f"Nonce adjusted to {self.current_nonce}.")
            else:
                self.logger.debug(
                    f"No discrepancy. External nonce {external_nonce} is not higher than the internal nonce. ✨"
                )

    async def reset_nonce(self):
        """
        Resets the current nonce to the on-chain value. Useful in case of major nonce conflicts or after
        manually handling stuck transactions.

        Raises:
            RuntimeError: If the nonce reset fails after multiple attempts.
        """
        async with self.lock:
            await loading_bar("Resetting Nonce", 0)
            try:
                self.current_nonce = await self._fetch_current_nonce_with_retries()
                self.logger.debug(
                    f"Nonce reset successfully to {self.current_nonce}. ✨"
                )
            except Exception as e:
                self.logger.error(f"Failed to reset nonce for {self.address}: {e} ❌")
                raise RuntimeError(f"Failed to reset nonce: {e} ❌")