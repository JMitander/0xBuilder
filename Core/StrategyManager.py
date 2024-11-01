class TransactionArray:
    """
    TransactionArray class builds and executes transactions, including front-run,
    back-run, and sandwich attack strategies. It interacts with smart contracts,
    manages transaction signing, gas price estimation, and handles flashloans.
    """
    def __init__(
        self,
        web3: AsyncWeb3,
        account: Account,
        flashloan_contract_address: str,
        flashloan_contract_ABI: List[Dict[str, Any]],
        lending_pool_contract_address: str,
        lending_pool_contract_ABI: List[Dict[str, Any]],
        monitor: MonitorArray,
        nonce_manager: NonceManager,
        safety_net: SafetyNet,
        config: Config,
        logger: Optional[logging.Logger] = None,
        gas_price_multiplier: float = 1.1,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
        erc20_ABI: Optional[List[Dict[str, Any]]] = None,
    ):
        self.web3 = web3
        self.account = account
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.monitor = monitor
        self.nonce_manager = nonce_manager
        self.safety_net = safety_net
        self.gas_price_multiplier = gas_price_multiplier
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.erc20_ABI = erc20_ABI or []
        self.current_profit = Decimal("0")
        self.flashloan_contract_address = flashloan_contract_address
        self.flashloan_contract_ABI = flashloan_contract_ABI
        self.lending_pool_contract_address = lending_pool_contract_address
        self.lending_pool_contract_ABI = lending_pool_contract_ABI
        self.logger.info("TransactionArray initialized successfully. ✅")

    async def initialize(self):
        self.flashloan_contract = await self._initialize_contract(
            self.flashloan_contract_address,
            self.flashloan_contract_ABI,
            "Flashloan Contract",
        )
        self.lending_pool_contract = await self._initialize_contract(
            self.lending_pool_contract_address,
            self.lending_pool_contract_ABI,
            "Lending Pool Contract",
        )
        self.uniswap_router_contract = await self._initialize_contract(
            self.config.UNISWAP_V2_ROUTER_ADDRESS,
            self.config.UNISWAP_V2_ROUTER_ABI,
            "Uniswap Router Contract",
        )
        self.sushiswap_router_contract = await self._initialize_contract(
            self.config.SUSHISWAP_ROUTER_ADDRESS,
            self.config.SUSHISWAP_ROUTER_ABI,
            "Sushiswap Router Contract",
        )
        self.pancakeswap_router_contract = await self._initialize_contract(
            self.config.PANCAKESWAP_ROUTER_ADDRESS,
            self.config.PANCAKESWAP_ROUTER_ABI,
            "Pancakeswap Router Contract",
        )
        self.balancer_router_contract = await self._initialize_contract(
            self.config.BALANCER_ROUTER_ADDRESS,
            self.config.BALANCER_ROUTER_ABI,
            "Balancer Router Contract",
        )
        self.erc20_ABI = self.erc20_ABI or await self._load_erc20_ABI()

    async def _initialize_contract(
        self,
        contract_address: str,
        contract_ABI: List[Dict[str, Any]],
        contract_name: str,
    ) -> Contract:
        try:
            contract_instance = self.web3.eth.contract(
                address=self.web3.to_checksum_address(contract_address),
                abi=contract_ABI,
            )
            self.logger.info(
                f"Loaded {contract_name} at {contract_address} successfully. ✅"
            )
            return contract_instance
        except Exception as e:
            self.logger.error(
                f"Failed to load {contract_name} at {contract_address}: {e} ❌"
            )
            raise ValueError(
                f"Contract initialization failed for {contract_name}"
            ) from e

    async def build_transaction(
        self, function_call: Any, additional_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        additional_params = additional_params or {}
        try:
            tx_details = {
                "data": function_call.encode_ABI(),
                "to": function_call.address,
                "chainId": await self.web3.eth.chain_id,
                "nonce": await self.nonce_manager.get_nonce(),
                "from": self.account.address,
            }
            tx_details.update(additional_params)
            tx = tx_details.copy()
            tx["gas"] = await self.estimate_gas_smart(tx)
            tx.update(await self.get_dynamic_gas_price())
            self.logger.debug(f"Built transaction: {tx}")
            return tx
        except Exception as e:
            self.logger.exception(f"Error building transaction: {e} ⚠️")
            raise

    async def get_dynamic_gas_price(self) -> Dict[str, int]:
        try:
            gas_price_gwei = await self.safety_net.get_dynamic_gas_price()
            self.logger.info(f"Fetched gas price: {gas_price_gwei} Gwei ⛽")
        except Exception as e:
            self.logger.error(
                f"Error fetching dynamic gas price: {e}. Using default gas price. ⛽⚠️"
            )
            gas_price_gwei = 100.0  # Default gas price in Gwei

        gas_price = int(
            self.web3.to_wei(gas_price_gwei * self.gas_price_multiplier, "gwei")
        )
        return {"gasPrice": gas_price}

    async def estimate_gas_smart(self, tx: Dict[str, Any]) -> int:
        try:
            gas_estimate = await self.web3.eth.estimate_gas(tx)
            self.logger.debug(f"Estimated gas: {gas_estimate} ⛽")
            return gas_estimate
        except Exception as e:
            self.logger.warning(
                f"Gas estimation failed: {e}. Using default gas limit of 100000 ⛽⚠️"
            )
            return 100_000  # Default gas limit

    async def execute_transaction(self, tx: Dict[str, Any]) -> Optional[str]:
        for attempt in range(1, self.retry_attempts + 1):
            try:
                signed_tx = await self.sign_transaction(tx)
                tx_hash = await self.web3.eth.send_raw_transaction(signed_tx)
                tx_hash_hex = (
                    tx_hash.hex()
                    if isinstance(tx_hash, hexbytes.HexBytes)
                    else tx_hash
                )
                self.logger.info(
                    f"Transaction sent successfully with hash: {tx_hash_hex} 🚀✅"
                )
                await self.nonce_manager.refresh_nonce()
                return tx_hash_hex
            except Exception as e:
                self.logger.error(
                    f"Error executing transaction: {e}. Attempt {attempt} of {self.retry_attempts} 🔄"
                )
                if attempt < self.retry_attempts:
                    sleep_time = self.retry_delay * attempt
                    self.logger.info(f"Retrying in {sleep_time} seconds...")
                    await asyncio.sleep(sleep_time)
        self.logger.error("Failed to execute transaction after multiple attempts. ❌")
        return None

    async def sign_transaction(self, transaction: Dict[str, Any]) -> bytes:
        try:
            signed_tx = self.web3.eth.account.sign_transaction(
                transaction,
                private_key=self.account.key,
            )
            self.logger.debug(
                f"Transaction signed successfully: Nonce {transaction['nonce']}. 📋"
            )
            return signed_tx.rawTransaction
        except Exception as e:
            self.logger.exception(f"Error signing transaction: {e} ⚠️")
            raise

    async def handle_eth_transaction(self, target_tx: Dict[str, Any]) -> bool:
        tx_hash = target_tx.get("tx_hash", "Unknown")
        self.logger.info(f"Handling ETH transaction {tx_hash} 🚀")
        try:
            eth_value = target_tx.get("value", 0)
            tx_details = {
                "data": target_tx.get("input", "0x"),
                "chainId": await self.web3.eth.chain_id,
                "to": target_tx.get("to", ""),
                "value": eth_value,
                "gas": 21_000,
                "nonce": await self.nonce_manager.get_nonce(),
                "from": self.account.address,
            }
            original_gas_price = int(target_tx.get("gasPrice", 0))
            tx_details["gasPrice"] = int(
                original_gas_price * 1.1
            )
            eth_value_ether = self.web3.from_wei(eth_value, "ether")
            self.logger.info(
                f"Building ETH front-run transaction for {eth_value_ether} ETH to {tx_details['to']}"
            )
            tx_hash_executed = await self.execute_transaction(tx_details)
            if tx_hash_executed:
                self.logger.info(
                    f"Successfully executed ETH transaction with hash: {tx_hash_executed} ✅"
                )
                return True
            else:
                self.logger.error("Failed to execute ETH transaction. ❌")
                return False
        except Exception as e:
            self.logger.exception(f"Error handling ETH transaction: {e} ❌")
            return False

    def calculate_flashloan_amount(self, target_tx: Dict[str, Any]) -> int:
        estimated_profit = target_tx.get("profit", 0)
        if estimated_profit > 0:
            flashloan_amount = int(
                Decimal(estimated_profit) * Decimal("0.8")
            )
            self.logger.info(
                f"Calculated flashloan amount: {flashloan_amount} Wei based on estimated profit. ⚡🏦"
            )
            return flashloan_amount
        else:
            self.logger.info("No estimated profit. Setting flashloan amount to 0. ⚡⚠️")
            return 0

    async def simulate_transaction(self, transaction: Dict[str, Any]) -> bool:
        self.logger.info(
            f"Simulating transaction with nonce {transaction.get('nonce', 'Unknown')}. 🔍📊"
        )
        try:
            await self.web3.eth.call(transaction, block_identifier="pending")
            self.logger.info("Transaction simulation succeeded. 📊✅")
            return True
        except Exception as e:
            self.logger.error(f"Transaction simulation failed: {e} ❌")
            return False

    async def prepare_flashloan_transaction(
        self, flashloan_asset: str, flashloan_amount: int
    ) -> Optional[Dict[str, Any]]:
        if flashloan_amount <= 0:
            self.logger.warning(
                "Flashloan amount is 0 or less, skipping flashloan transaction preparation. 📉"
            )
            return None
        try:
            flashloan_function = self.flashloan_contract.functions.fn_RequestFlashLoan(
                self.web3.to_checksum_address(flashloan_asset), flashloan_amount
            )
            self.logger.info(
                f"Preparing flashloan transaction for {flashloan_amount} of {flashloan_asset}. ⚡🏦"
            )
            return await self.build_transaction(flashloan_function)
        except ContractLogicError as e:
            self.logger.error(
                f"Contract logic error preparing flashloan transaction: {e} ❌"
            )
            return None
        except Exception as e:
            self.logger.exception(f"Error preparing flashloan transaction: {e} ❌")
            return None

    async def send_bundle(self, transactions: List[Dict[str, Any]]) -> bool:
        try:
            signed_txs = [await self.sign_transaction(tx) for tx in transactions]
            bundle_payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "eth_sendBundle",
                "params": [
                    {
                        "txs": [signed_tx.hex() for signed_tx in signed_txs],
                        "blockNumber": hex(await self.web3.eth.block_number + 1),
                    }
                ],
            }
            message = encode_defunct(text=json.dumps(bundle_payload["params"][0]))
            signed_message = self.web3.eth.account.sign_message(
                message, private_key=self.account.key
            )
            headers = {
                "Content-Type": "application/json",
                "X-Flashbots-Signature": f"{self.account.address}:{signed_message.signature.hex()}",
            }
            for attempt in range(1, self.retry_attempts + 1):
                try:
                    self.logger.info(f"Attempt {attempt} to send bundle. 📦💨")
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            "https://relay.flashbots.net",
                            json=bundle_payload,
                            headers=headers,
                            timeout=30,
                        ) as response:
                            response.raise_for_status()
                            response_data = await response.json()
                            if "error" in response_data:
                                self.logger.error(
                                    f"Bundle submission error: {response_data['error']} ⚠️📦"
                                )
                                raise ValueError(response_data["error"])
                            self.logger.info("Bundle sent successfully. 📦✅")
                            await self.nonce_manager.refresh_nonce()
                            return True
                except aiohttp.ClientResponseError as e:
                    self.logger.error(
                        f"Error sending bundle: {e}. Retrying... 🔄📦"
                    )
                    if attempt < self.retry_attempts:
                        sleep_time = self.retry_delay * attempt
                        self.logger.info(f"Retrying in {sleep_time} seconds...")
                        await asyncio.sleep(sleep_time)
                except ValueError as e:
                    self.logger.error(f"Bundle submission error: {e} ⚠️📦")
                    break
            self.logger.error("Failed to send bundle after multiple attempts. ⚠️📦")
            return False
        except Exception as e:
            self.logger.exception(f"Unexpected error in send_bundle: {e} ❌")
            return False

    async def front_run(self, target_tx: Dict[str, Any]) -> bool:
        tx_hash = target_tx.get("tx_hash", "Unknown")
        self.logger.info(
            f"Attempting front-run on target transaction: {tx_hash} 🏃💨📈"
        )
        decoded_tx = await self.decode_transaction_input(
            target_tx.get("input", "0x"), target_tx.get("to", "")
        )
        if not decoded_tx:
            self.logger.error(
                "Failed to decode target transaction input for front-run. ⚠️"
            )
            return False
        try:
            flashloan_asset = decoded_tx["params"].get("path", [])[0]
            flashloan_amount = self.calculate_flashloan_amount(target_tx)
            if flashloan_amount > 0:
                flashloan_tx = await self.prepare_flashloan_transaction(
                    flashloan_asset, flashloan_amount
                )
            else:
                self.logger.info("Flashloan amount is zero or less. Skipping flashloan transaction preparation. ⚠️")
                return False
            if not flashloan_tx:
                self.logger.info(
                    "Failed to prepare flashloan transaction for front-run. Aborting. ⚠️"
                )
                return False
            front_run_tx_details = await self._prepare_front_run_transaction(target_tx)
            if not front_run_tx_details:
                self.logger.info(
                    "Failed to prepare front-run transaction. Aborting. ⚠️"
                )
                return False
            if not (
                await self.simulate_transaction(flashloan_tx)
                and await self.simulate_transaction(front_run_tx_details)
            ):
                self.logger.info(
                    "Simulation of front-run or flashloan failed. Aborting. ⚠️"
                )
                return False
            if await self.send_bundle([flashloan_tx, front_run_tx_details]):
                self.logger.info(
                    "Front-run transaction bundle sent successfully. 🏃💨📈✅"
                )
                return True
            else:
                self.logger.error("Failed to send front-run transaction bundle. ⚠️")
                return False
        except Exception as e:
            self.logger.exception(f"Error executing front-run: {e} ⚠️")
            return False


    async def back_run(self, target_tx: Dict[str, Any]) -> bool:
        tx_hash = target_tx.get("tx_hash", "Unknown")
        self.logger.info(f"Attempting back-run on target transaction: {tx_hash} 🔙🏃📉")
        decoded_tx = await self.decode_transaction_input(
            target_tx.get("input", "0x"), target_tx.get("to", "")
        )
        if not decoded_tx:
            self.logger.error(
                "Failed to decode target transaction input for back-run. ⚠️"
            )
            return False
        try:
            # Get the parameters for the back-run
            flashloan_asset = decoded_tx["params"].get("path", [])[-1]
            flashloan_amount = self.calculate_flashloan_amount(target_tx)
            # Prepare the flashloan transaction
            flashloan_tx = await self.prepare_flashloan_transaction(
                flashloan_asset, flashloan_amount
            )
            if not flashloan_tx:
                self.logger.info(
                    "Failed to prepare flashloan transaction for back-run. Aborting. ⚠️"
                )
                return False
            # Prepare the back-run transaction
            back_run_tx_details = await self._prepare_back_run_transaction(target_tx)
            if not back_run_tx_details:
                self.logger.info(
                    "Failed to prepare back-run transaction. Aborting. ⚠️"
                )
                return False
            # Simulate transactions
            if not (
                await self.simulate_transaction(flashloan_tx)
                and await self.simulate_transaction(back_run_tx_details)
            ):
                self.logger.info(
                    "Simulation of back-run or flashloan failed. Aborting. ⚠️"
                )
                return False

            # Execute as a bundle
            if await self.send_bundle([flashloan_tx, back_run_tx_details]):
                self.logger.info(
                    "Back-run transaction bundle sent successfully. 🔙🏃📉✅"
                )
                return True
            else:
                self.logger.error("Failed to send back-run transaction bundle. ⚠️")
                return False

        except Exception as e:
            self.logger.exception(f"Error executing back-run: {e} ⚠️")
            return False
        
    async def execute_sandwich_attack(self, target_tx: Dict[str, Any]) -> bool:
        tx_hash = target_tx.get("tx_hash", "Unknown")
        self.logger.info(
            f"Attempting sandwich attack on target transaction: {tx_hash} 🥪🏃📈"
        )
        decoded_tx = await self.decode_transaction_input(
            target_tx.get("input", "0x"), target_tx.get("to", "")
        )
        if not decoded_tx:
            self.logger.error(
                "Failed to decode target transaction input for sandwich attack. ⚠️"
            )
            return False
        try:
            # Get the parameters for the sandwich attack
            flashloan_asset = decoded_tx["params"].get("path", [])[0]
            flashloan_amount = self.calculate_flashloan_amount(target_tx)
            # Prepare the flashloan transaction
            flashloan_tx = await self.prepare_flashloan_transaction(
                flashloan_asset, flashloan_amount
            )
            if not flashloan_tx:
                self.logger.info(
                    "Failed to prepare flashloan transaction for sandwich attack. Aborting. ⚠️"
                )
                return False
            # Prepare the front-run transaction
            front_run_tx_details = await self._prepare_front_run_transaction(target_tx)
            if not front_run_tx_details:
                self.logger.info(
                    "Failed to prepare front-run transaction for sandwich attack. Aborting. ⚠️"
                )
                return False
            # Prepare the back-run transaction
            back_run_tx_details = await self._prepare_back_run_transaction(target_tx)
            if not back_run_tx_details:
                self.logger.info(
                    "Failed to prepare back-run transaction for sandwich attack. Aborting. ⚠️"
                )
                return False
            # Simulate transactions
            if not (
                await self.simulate_transaction(flashloan_tx)
                and await self.simulate_transaction(front_run_tx_details)
                and await self.simulate_transaction(back_run_tx_details)
            ):
                self.logger.info(
                    "Simulation of one or more transactions failed during sandwich attack. Aborting. ⚠️"
                )
                return False
            # Execute all three transactions as a bundle
            if await self.send_bundle(
                [flashloan_tx, front_run_tx_details, back_run_tx_details]
            ):
                self.logger.info(
                    "Sandwich attack transaction bundle sent successfully. 🥪🏃📈✅"
                )
                return True
            else:
                self.logger.error(
                    "Failed to send sandwich attack transaction bundle. ⚠️"
                )
                return False
        except Exception as e:
            self.logger.exception(f"Error executing sandwich attack: {e} ❌")
            return False

    async def _prepare_front_run_transaction(
        self, target_tx: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        decoded_tx = await self.decode_transaction_input(
            target_tx.get("input", "0x"), target_tx.get("to", "")
        )
        if not decoded_tx:
            self.logger.error(
                "Failed to decode target transaction input for front-run preparation. ⚠️"
            )
            return None
        function_name = decoded_tx.get("function_name")
        function_params = decoded_tx.get("params", {})
        try:
            # Determine which router to use based on the target address
            to_address = self.web3.to_checksum_address(target_tx.get("to", ""))
            if to_address == self.config.UNISWAP_V2_ROUTER_ADDRESS:
                router_contract = self.uniswap_router_contract
                exchange_name = "Uniswap"
            elif to_address == self.config.SUSHISWAP_ROUTER_ADDRESS:
                router_contract = self.sushiswap_router_contract
                exchange_name = "Sushiswap"
            elif to_address == self.config.PANCAKESWAP_ROUTER_ADDRESS:
                router_contract = self.pancakeswap_router_contract
                exchange_name = "Pancakeswap"
            elif to_address == self.config.BALANCER_ROUTER_ADDRESS:
                router_contract = self.balancer_router_contract
                exchange_name = "Balancer"
            else:
                self.logger.error("Unknown router address. Cannot determine exchange. ❌")
                return None
            # Get the function object by name
            front_run_function = getattr(router_contract.functions, function_name)(
                **function_params
            )
            # Build the transaction
            front_run_tx = await self.build_transaction(front_run_function)
            self.logger.info(
                f"Prepared front-run transaction on {exchange_name} successfully. ⚔️🏃"
            )
            return front_run_tx
        except AttributeError:
            self.logger.error(
                f"Function {function_name} not found in {exchange_name} router ABI. ❌"
            )
            return None
        except Exception as e:
            self.logger.exception(f"Error preparing front-run transaction: {e} ❌")
            return None

    async def _prepare_back_run_transaction(
        self, target_tx: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        decoded_tx = await self.decode_transaction_input(
            target_tx.get("input", "0x"), target_tx.get("to", "")
        )
        if not decoded_tx:
            self.logger.error(
                "Failed to decode target transaction input for back-run preparation. ⚠️"
            )
            return None
        function_name = decoded_tx.get("function_name")
        function_params = decoded_tx.get("params", {})
        # Reverse the path parameter for back-run
        path = function_params.get("path", [])
        if path:
            function_params["path"] = path[::-1]
        else:
            self.logger.warning(
                "Transaction has no path parameter for back-run preparation. ❗"
            )
        try:
            # Determine which router to use based on the target address
            to_address = self.web3.to_checksum_address(target_tx.get("to", ""))
            if to_address == self.config.UNISWAP_V2_ROUTER_ADDRESS:
                router_contract = self.uniswap_router_contract
                exchange_name = "Uniswap"
            elif to_address == self.config.SUSHISWAP_ROUTER_ADDRESS:
                router_contract = self.sushiswap_router_contract
                exchange_name = "Sushiswap"
            elif to_address == self.config.PANCAKESWAP_ROUTER_ADDRESS:
                router_contract = self.pancakeswap_router_contract
                exchange_name = "Pancakeswap"
            elif to_address == self.config.BALANCER_ROUTER_ADDRESS:
                router_contract = self.balancer_router_contract
                exchange_name = "Balancer"
            else:
                self.logger.error("Unknown router address. Cannot determine exchange. ❌")
                return None
            # Get the function object by name
            back_run_function = getattr(router_contract.functions, function_name)(
                **function_params
            )
            # Build the transaction
            back_run_tx = await self.build_transaction(back_run_function)
            self.logger.info(
                f"Prepared back-run transaction on {exchange_name} successfully. 🔙🏃"
            )
            return back_run_tx
        except AttributeError:
            self.logger.error(
                f"Function {function_name} not found in {exchange_name} router ABI. ❌"
            )
            return None
        except Exception as e:
            self.logger.exception(f"Error preparing back-run transaction: {e} ❌")
            return None

    async def decode_transaction_input(
        self, input_data: str, to_address: str
    ) -> Optional[Dict[str, Any]]:
        try:
            to_address = self.web3.to_checksum_address(to_address)
            if to_address == self.config.UNISWAP_V2_ROUTER_ADDRESS:
                abi = self.config.UNISWAP_V2_ROUTER_ABI
                exchange_name = "Uniswap"
            elif to_address == self.config.SUSHISWAP_ROUTER_ADDRESS:
                abi = self.config.SUSHISWAP_ROUTER_ABI
                exchange_name = "Sushiswap"
            elif to_address == self.config.PANCAKESWAP_ROUTER_ADDRESS:
                abi = self.config.PANCAKESWAP_ROUTER_ABI
                exchange_name = "Pancakeswap"
            elif to_address == self.config.BALANCER_ROUTER_ADDRESS:
                abi = self.config.BALANCER_ROUTER_ABI
                exchange_name = "Balancer"
            else:
                self.logger.error(
                    "Unknown router address. Cannot determine ABI for decoding. ❌"
                )
                return None
            contract = self.web3.eth.contract(address=to_address, abi=abi)
            function_obj, function_params = contract.decode_function_input(input_data)
            decoded_data = {
                "function_name": function_obj.function_identifier,
                "params": function_params,
            }
            self.logger.debug(
                f"Decoded transaction input using {exchange_name} ABI: {decoded_data}"
            )
            return decoded_data
        except Exception as e:
            self.logger.error(f"Error decoding transaction input: {e} ❌")
            return None

    async def cancel_transaction(self, nonce: int) -> bool:
        cancel_tx = {
            "data": "0x",
            "chainId": await self.web3.eth.chain_id,
            "nonce": nonce,
            "to": self.account.address,
            "value": 0,
            "gas": 21_000,
            "gasPrice": self.web3.to_wei("150", "gwei"),  # Higher than the stuck transaction
            "from": self.account.address,
        }
        try:
            signed_cancel_tx = await self.sign_transaction(cancel_tx)
            tx_hash = await self.web3.eth.send_raw_transaction(signed_cancel_tx)
            tx_hash_hex = (
                tx_hash.hex()
                if isinstance(tx_hash, hexbytes.HexBytes)
                else tx_hash
            )
            self.logger.info(
                f"Cancellation transaction sent successfully: {tx_hash_hex} 🚀✅"
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to cancel transaction: {e} ❌")
            return False

    async def estimate_gas_limit(self, tx: Dict[str, Any]) -> int:
        try:
            gas_estimate = await self.web3.eth.estimate_gas(tx)
            self.logger.debug(f"Estimated gas: {gas_estimate} ⛽")
            return gas_estimate
        except Exception as e:
            self.logger.warning(
                f"Gas estimation failed: {e}. Using default gas limit of 100000 ⛽⚠️"
            )
            return 100_000  # Default gas limit

    async def get_current_profit(self) -> Decimal:
        try:
            current_profit = await self.safety_net.get_balance(self.account)
            self.current_profit = Decimal(current_profit)
            self.logger.info(f"Current profit: {self.current_profit} ETH 💰")
            return self.current_profit
        except Exception as e:
            self.logger.error(f"Error fetching current profit: {e} ❌")
            return Decimal("0")

    async def withdraw_eth(self) -> bool:
        try:
            withdraw_function = self.flashloan_contract.functions.withdrawETH()
            tx = await self.build_transaction(withdraw_function)
            tx_hash = await self.execute_transaction(tx)
            if tx_hash:
                self.logger.info(
                    f"ETH withdrawal transaction sent with hash: {tx_hash} ✅"
                )
                return True
            else:
                self.logger.error("Failed to send ETH withdrawal transaction. ❌")
                return False
        except Exception as e:
            self.logger.exception(f"Error withdrawing ETH: {e} ❌")
            return False

    async def withdraw_token(self, token_address: str) -> bool:
        try:
            withdraw_function = self.flashloan_contract.functions.withdrawToken(
                self.web3.to_checksum_address(token_address)
            )
            tx = await self.build_transaction(withdraw_function)
            tx_hash = await self.execute_transaction(tx)
            if tx_hash:
                self.logger.info(
                    f"Token withdrawal transaction sent with hash: {tx_hash} ✅"
                )
                return True
            else:
                self.logger.error("Failed to send token withdrawal transaction. ❌")
                return False
        except Exception as e:
            self.logger.exception(f"Error withdrawing token: {e} ❌")
            return False
