class Transaction_Core:
    """
    Transaction_Core is the main transaction engine that handles all transaction-related
    Builds and executes transactions, including front-run, back-run, and sandwich attack strategies.
    It interacts with smart contracts, manages transaction signing, gas price estimation, and handles flashloans
    """
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0  # Base delay in seconds for retries

    def __init__(
        self,
        web3: AsyncWeb3,
        account: Account,
        AAVE_FLASHLOAN_ADDRESS: str,
        AAVE_FLASHLOAN_ABI: List[Dict[str, Any]],
        AAVE_LENDING_POOL_ADDRESS: str,
        AAVE_LENDING_POOL_ABI: List[Dict[str, Any]],
        api_config: Optional[API_Config] = None,
        monitor: Optional[Mempool_Monitor] = None,
        nonce_core: Optional[Nonce_Core] = None,
        safety_net: Optional[Safety_Net] = None,
        configuration: Optional[Configuration] = None,
        gas_price_multiplier: float = 1.1,
        erc20_abi: Optional[List[Dict[str, Any]]] = None,
    ):
        self.web3 = web3
        self.account = account
        self.configuration = configuration
        self.monitor = monitor
        self.api_config = api_config
        self.nonce_core = nonce_core
        self.safety_net = safety_net
        self.gas_price_multiplier = gas_price_multiplier
        self.RETRY_ATTEMPTS = self.MAX_RETRIES
        self.erc20_abi = erc20_abi or []
        self.current_profit = Decimal("0")
        self.AAVE_FLASHLOAN_ADDRESS = AAVE_FLASHLOAN_ADDRESS
        self.AAVE_FLASHLOAN_ABI = AAVE_FLASHLOAN_ABI
        self.AAVE_LENDING_POOL_ADDRESS = AAVE_LENDING_POOL_ADDRESS
        self.AAVE_LENDING_POOL_ABI = AAVE_LENDING_POOL_ABI

    async def initialize(self):
        """Initializes all required contracts.""" 
        self.flashloan_contract = await self._initialize_contract(
            self.AAVE_FLASHLOAN_ADDRESS,
            self.AAVE_FLASHLOAN_ABI,
            "Flashloan Contract",
        )
        self.lending_pool_contract = await self._initialize_contract(
            self.AAVE_LENDING_POOL_ADDRESS,
            self.AAVE_LENDING_POOL_ABI,
            "Lending Pool Contract",
        )
        self.uniswap_router_contract = await self._initialize_contract(
            self.configuration.UNISWAP_ROUTER_ADDRESS,
            self.configuration.UNISWAP_ROUTER_ABI,
            "Uniswap Router Contract",
        )
        self.sushiswap_router_contract = await self._initialize_contract(
            self.configuration.SUSHISWAP_ROUTER_ADDRESS,
            self.configuration.SUSHISWAP_ROUTER_ABI,
            "Sushiswap Router Contract",
        )
        self.pancakeswap_router_contract = await self._initialize_contract(
            self.configuration.PANCAKESWAP_ROUTER_ADDRESS,
            self.configuration.PANCAKESWAP_ROUTER_ABI,
            "Pancakeswap Router Contract",
        )
        self.balancer_router_contract = await self._initialize_contract(
            self.configuration.BALANCER_ROUTER_ADDRESS,
            self.configuration.BALANCER_ROUTER_ABI,
            "Balancer Router Contract",
        )

        self.erc20_abi = self.erc20_abi or await self._load_erc20_abi()

        logger.info("Transaction Core initialized with all lights green... Booster ignition.. ✅")
        time.sleep(3)  # ensuring proper initialization

    async def _initialize_contract(
        self,
        contract_address: str,
        contract_abi: List[Dict[str, Any]],
        contract_name: str,
    ) -> Any:
        """Initializes a contract instance with  error handling.""" 
        try:
            # Load ABI from file if it's a string path
            if isinstance(contract_abi, str):
                async with aiofiles.open(contract_abi, 'r') as f:
                    abi_content = await f.read()
                    contract_abi = json.loads(abi_content)

            contract_instance = self.web3.eth.contract(
                address=self.web3.to_checksum_address(contract_address),
                abi=contract_abi,
            )
            logger.debug(f"Loaded {contract_name} successfully.")
            return contract_instance
        except FileNotFoundError:
            logger.error(f"ABI file for {contract_name} not found at {contract_abi}.")
            raise
        except json.JSONDecodeError:
            logger.error(f"ABI file for {contract_name} is not valid JSON.")
            raise
        except Exception as e:
            logger.error(
                f"Failed to load {contract_name} at {contract_address}: {e}"
            )
            raise ValueError(
                f"Contract initialization failed for {contract_name}"
            ) from e

    async def _load_erc20_abi(self) -> List[Dict[str, Any]]:
        """Load the ERC20 ABI.""" 
        try:
            erc20_abi = await self.api_config._load_abi(self.configuration.ERC20_ABI)
            logger.debug("ERC20 ABI loaded successfully.")
            return erc20_abi
        except FileNotFoundError:
            logger.error(f"ERC20 ABI file not found at {self.configuration.ERC20_ABI}.")
            raise
        except json.JSONDecodeError:
            logger.error("ERC20 ABI file is not valid JSON.")
            raise
        except Exception as e:
            logger.error(f"Failed to load ERC20 ABI: {e}")
            raise ValueError("ERC20 ABI loading failed") from e

    async def build_transaction(self, function_call: Any, additional_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Enhanced transaction building with EIP-1559 support and proper gas estimation.""" 
        additional_params = additional_params or {}
        try:
            # Get chain ID
            chain_id = await self.web3.eth.chain_id
            
            # Check if chain supports EIP-1559
            latest_block = await self.web3.eth.get_block('latest')
            supports_eip1559 = 'baseFeePerGas' in latest_block
            
            # Base transaction parameters
            tx_params = {
                'chainId': chain_id,
                'nonce': await self.nonce_core.get_nonce(),
                'from': self.account.address,
            }
            
            # Add EIP-1559 specific parameters if supported
            if supports_eip1559:
                base_fee = latest_block['baseFeePerGas']
                priority_fee = await self.web3.eth.max_priority_fee
                
                tx_params.update({
                    'maxFeePerGas': base_fee * 2,  # Double the base fee
                    'maxPriorityFeePerGas': priority_fee
                })
            else:
                # Legacy gas price
                tx_params.update(await self.get_dynamic_gas_price())
            
            # Build transaction
            tx_details = function_call.buildTransaction(tx_params)
            
            # Add additional parameters
            tx_details.update(additional_params)
            
            # Estimate gas with buffer
            estimated_gas = await self.estimate_gas_smart(tx_details)
            tx_details['gas'] = int(estimated_gas * 1.1)  # Add 10% buffer
            
            return tx_details
            
        except Exception as e:
            logger.error(f"Error building transaction: {e}")
            raise

    async def get_dynamic_gas_price(self) -> Dict[str, int]:
        """
        Gets dynamic gas price adjusted by the multiplier.

        :return: Dictionary containing 'gasPrice'.
        """
        try:
            gas_price_gwei = await self.safety_net.get_dynamic_gas_price()
            logger.debug(f"Fetched gas price: {gas_price_gwei} Gwei")
        except Exception as e:
            logger.error(
                f"Error fetching dynamic gas price: {e}. Using default gas price."
            )
            gas_price_gwei = Decimal(100.0)  # Default gas price in Gwei

        gas_price = int(
            self.web3.to_wei(gas_price_gwei * self.gas_price_multiplier, "gwei")
        )
        return {"gasPrice": gas_price}

    async def estimate_gas_smart(self, tx: Dict[str, Any]) -> int:
        """
        Estimates gas with fallback to a default value.

        :param tx: Transaction dictionary.
        :return: Estimated gas.
        """
        try:
            gas_estimate = await self.web3.eth.estimate_gas(tx)
            logger.debug(f"Estimated gas: {gas_estimate}")
            return gas_estimate
        except ContractLogicError as e:
            logger.warning(f"Contract logic error during gas estimation: {e}. Using default gas limit.")
            return 100_000  # Default gas limit
        except TransactionNotFound:
            logger.warning("Transaction not found during gas estimation. Using default gas limit.")
            return 100_000
        except Exception as e:
            logger.error(f"Gas estimation failed: {e}. Using default gas limit.")
            return 100_000  # Default gas limit

    async def execute_transaction(self, tx: Dict[str, Any]) -> Optional[str]:
        """
        Executes a transaction with retries.

        :param tx: Transaction dictionary.
        :return: Transaction hash if successful, else None.
        """
        try:
            for attempt in range(1, self.MAX_RETRIES + 1):
                signed_tx = await self.sign_transaction(tx)
                tx_hash = await self.web3.eth.send_raw_transaction(signed_tx)
                logger.debug(f"Transaction sent successfully: {tx_hash.hex()}")
                return tx_hash.hex()
        except TransactionNotFound as e:
            logger.error(
                f"Transaction not found: {e}. Attempt {attempt} of {self.retry_attempts}"
            )
        except ContractLogicError as e:
            logger.error(
                f"Contract logic error: {e}. Attempt {attempt} of {self.retry_attempts}"
            )
        except Exception as e:
            logger.warning(f"Attempt {attempt+1}: Failed to execute transaction - {e}")
            await asyncio.sleep(self.RETRY_DELAY * (attempt + 1))
        logger.error("Failed to execute transaction after retries")
        return None

    async def sign_transaction(self, transaction: Dict[str, Any]) -> bytes:
        """
        Signs a transaction with the account's private key.

        :param transaction: Transaction dictionary.
        :return: Signed transaction bytes.
        """
        try:
            signed_tx = self.web3.eth.account.sign_transaction(
                transaction,
                private_key=self.account.key,
            )
            logger.debug(
                f"Transaction signed successfully: Nonce {transaction['nonce']}. ✍️ "
            )
            return signed_tx.rawTransaction
        except KeyError as e:
            logger.error(f"Missing transaction parameter for signing: {e}")
            raise
        except Exception as e:
            logger.error(f"Error signing transaction: {e}")
            raise

    async def handle_eth_transaction(self, target_tx: Dict[str, Any]) -> bool:
        """
        Handles an ETH transfer transaction.

        :param target_tx: Target transaction dictionary.
        :return: True if successful, else False.
        """
        tx_hash = target_tx.get("tx_hash", "Unknown")
        logger.debug(f"Handling ETH transaction {tx_hash}")
        try:
            eth_value = target_tx.get("value", 0)
            if eth_value <= 0:
                logger.debug("Transaction value is zero or negative. Skipping.")
                return False

            tx_details = {
                "to": target_tx.get("to", ""),
                "value": eth_value,
                "gas": 21_000,
                "nonce": await self.nonce_core.get_nonce(),
                "chainId": self.web3.eth.chain_id,
                "from": self.account.address,
            }
            original_gas_price = int(target_tx.get("gasPrice", 0))
            if original_gas_price <= 0:
                logger.warning("Original gas price is zero or negative. Skipping.")
                return False
            tx_details["gasPrice"] = int(
                original_gas_price * 1.1  # Increase gas price by 10%
            )

            eth_value_ether = self.web3.from_wei(eth_value, "ether")
            logger.debug(
                f"Building ETH front-run transaction for {eth_value_ether} ETH to {tx_details['to']}"
            )
            tx_hash_executed = await self.execute_transaction(tx_details)
            if tx_hash_executed:
                logger.debug(
                    f"Successfully executed ETH transaction with hash: {tx_hash_executed} ✅ "
                )
                return True
            else:
                logger.warning("Failed to execute ETH transaction. Retrying... ⚠️ ")
                return False
        except KeyError as e:
            logger.error(f"Missing required transaction parameter: {e} ⚠️ ")
            return False
        except Exception as e:
            logger.error(f"Error handling ETH transaction: {e} ⚠️ ")
            return False

    def calculate_flashloan_amount(self, target_tx: Dict[str, Any]) -> int:
        """
        Calculates the flashloan amount based on estimated profit.

        :param target_tx: Target transaction dictionary.
        :return: Flashloan amount in Wei.
        """
        estimated_profit = target_tx.get("profit", 0)
        if estimated_profit > 0:
            flashloan_amount = int(
                Decimal(estimated_profit) * Decimal("0.8") * Decimal("1e18")
            )  # Convert ETH to Wei
            logger.debug(
                f"Calculated flashloan amount: {flashloan_amount} Wei based on estimated profit."
            )
            return flashloan_amount
        else:
            logger.debug("No estimated profit. Setting flashloan amount to 0.")
            return 0

    async def simulate_transaction(self, transaction: Dict[str, Any]) -> bool:
        """
        Simulates a transaction to check if it will succeed.

        :param transaction: Transaction dictionary.
        :return: True if simulation succeeds, else False.
        """
        logger.debug(
            f"Simulating transaction with nonce {transaction.get('nonce', 'Unknown')}."
        )
        try:
            await self.web3.eth.call(transaction, block_identifier="pending")
            logger.debug("Transaction simulation succeeded.")
            return True
        except ContractLogicError as e:
            logger.debug(f"Transaction simulation failed due to contract logic error: {e}")
            return False
        except Exception as e:
            logger.debug(f"Transaction simulation failed: {e}")
            return False

    async def prepare_flashloan_transaction(
        self, flashloan_asset: str, flashloan_amount: int
    ) -> Optional[Dict[str, Any]]:
        """
        Prepares a flashloan transaction.

        :param flashloan_asset: Asset address to borrow.
        :param flashloan_amount: Amount to borrow in Wei.
        :return: Transaction dictionary if successful, else None.
        """
        if flashloan_amount <= 0:
            logger.debug(
                "Flashloan amount is 0 or less, skipping flashloan transaction preparation."
            )
            return None
        try:
            flashloan_function = self.flashloan_contract.functions.RequestFlashLoan(
                self.web3.to_checksum_address(flashloan_asset), flashloan_amount
            )
            logger.debug(
                f"Preparing flashloan transaction for {flashloan_amount} Wei of {flashloan_asset}."
            )
            flashloan_tx = await self.build_transaction(flashloan_function)
            return flashloan_tx
        except ContractLogicError as e:
            logger.error(
                f"Contract logic error preparing flashloan transaction: {e} ⚠️ "
            )
            return None
        except Exception as e:
            logger.error(f"Error preparing flashloan transaction: {e} ⚠️ ")
            return None

    async def send_bundle(self, transactions: List[Dict[str, Any]]) -> bool:
        """
        Sends a bundle of transactions to MEV relays.

        :param transactions: List of transaction dictionaries.
        :return: True if bundle sent successfully, else False.
        """
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

            # List of MEV builders to try
            mev_builders = [
                {
                    "name": "Flashbots",
                    "url": "https://relay.flashbots.net",
                    "auth_header": "X-Flashbots-Signature"
                },
                # Add other builders as needed...
            ]

            # Track successful submissions
            successes = []

            # Try sending to each builder
            for builder in mev_builders:
                headers = {
                    "Content-Type": "application/json",
                     builder["auth_header"]: f"{self.account.address}:{self.account.key}",
               }

                for attempt in range(1, self.retry_attempts + 1):
                    try:
                        logger.debug(f"Attempt {attempt} to send bundle via {builder['name']}. ℹ️ ")
                        async with aiohttp.ClientSession() as session:
                            async with session.post(
                                builder["url"],
                                json=bundle_payload,
                                headers=headers,
                                timeout=30,
                            ) as response:
                                response.raise_for_status()
                                response_data = await response.json()

                                if "error" in response_data:
                                    logger.error(
                                        f"Bundle submission error via {builder['name']}: {response_data['error']}"
                                    )
                                    raise ValueError(response_data["error"])

                                logger.info(f"Bundle sent successfully via {builder['name']}. ✅ ")
                                successes.append(builder['name'])
                                break  # Success, move to next builder

                    except aiohttp.ClientResponseError as e:
                        logger.error(
                            f"HTTP error sending bundle via {builder['name']}: {e}. Attempt {attempt} of {self.retry_attempts}"
                        )
                        if attempt < self.retry_attempts:
                            sleep_time = self.retry_delay * attempt
                            logger.warning(f"Retrying in {sleep_time} seconds...")
                            await asyncio.sleep(sleep_time)
                    except ValueError as e:
                        logger.error(f"Bundle submission error via {builder['name']}: {e} ⚠️ ")
                        break  # Move to next builder
                    except Exception as e:
                        logger.error(f"Unexpected error with {builder['name']}: {e}. Attempt {attempt} of {self.retry_attempts} ⚠️ ")
                        if attempt < self.retry_attempts:
                            sleep_time = self.retry_delay * attempt
                            logger.warning(f"Retrying in {sleep_time} seconds...")
                            await asyncio.sleep(sleep_time)

            # Update nonce if any submissions succeeded
            if successes:
                await self.nonce_core.refresh_nonce()
                logger.info(f"Bundle successfully sent to builders: {', '.join(successes)} ✅ ")
                return True
            else:
                logger.warning("Failed to send bundle to any MEV builders. ⚠️ ")
                return False

        except Exception as e:
            logger.error(f"Unexpected error in send_bundle: {e} ⚠️ ")
            return False

    async def front_run(self, target_tx: Dict[str, Any]) -> bool:
        """
        Executes a front-run transaction with  validation and error handling.

        :param target_tx: Target transaction dictionary.
        :return: True if successful, else False.
        """
        if not isinstance(target_tx, dict):
            logger.debug("Invalid transaction format provided!")
            return False

        tx_hash = target_tx.get("tx_hash", "Unknown")
        logger.debug(f"Attempting front-run on target transaction: {tx_hash} ✅ ")

        # Validate required transaction parameters
        required_fields = ["input", "to", "value", "gasPrice"]
        if not all(field in target_tx for field in required_fields):
            missing = [field for field in required_fields if field not in target_tx]
            logger.debug(f"Missing required transaction parameters: {missing}. Skipping...")
            return False

        try:
            # Decode transaction input with validation
            decoded_tx = await self.decode_transaction_input(
                target_tx.get("input", "0x"),
                self.web3.to_checksum_address(target_tx.get("to", ""))
            )
            if not decoded_tx or "params" not in decoded_tx:
                logger.debug("Failed to decode transaction input for front-run. Skipping...")
                return False

            # Extract and validate path parameter
            path = decoded_tx["params"].get("path", [])
            if not path or not isinstance(path, list) or len(path) < 2:
                logger.debug("Transaction has invalid or no path parameter. Skipping...")
                return False

            # Prepare flashloan
            try:
                flashloan_asset = self.web3.to_checksum_address(path[0])
                flashloan_amount = self.calculate_flashloan_amount(target_tx)

                if flashloan_amount <= 0:
                    logger.debug("Insufficient flashloan amount calculated. Skipping...")
                    return False

                flashloan_tx = await self.prepare_flashloan_transaction(
                    flashloan_asset, flashloan_amount
                )
                if not flashloan_tx:
                    logger.debug("Failed to prepare flashloan transaction!")
                    return False

            except (ValueError, IndexError) as e:
                logger.error(f"Error preparing flashloan: {str(e)}")
                return False

            # Prepare front-run transaction
            front_run_tx_details = await self._prepare_front_run_transaction(target_tx)
            if not front_run_tx_details:
                logger.warning("Failed to prepare front-run transaction! Skipping...")
                return False

            # Simulate transactions
            simulation_success = await asyncio.gather(
                self.simulate_transaction(flashloan_tx),
                self.simulate_transaction(front_run_tx_details)
            )

            if not all(simulation_success):
                logger.error("Transaction simulation failed! Skipping...")
                return False

            # Send transaction bundle
            if await self.send_bundle([flashloan_tx, front_run_tx_details]):
                logger.info("Front-run transaction bundle sent successfully. ✅ ")
                return True
            else:
                logger.warning("Failed to send front-run transaction bundle! ⚠️ ")
                return False

        except KeyError as e:
            logger.error(f"Missing required transaction parameter: {e} ⚠️ ")
            return False
        except Exception as e:
            logger.error(f"Unexpected error in front-run execution: {str(e)}")
            return False

    async def back_run(self, target_tx: Dict[str, Any]) -> bool:
        """
        Executes a back-run transaction with validation and error handling.

        :param target_tx: Target transaction dictionary.
        :return: True if successful, else False.
        """
        if not isinstance(target_tx, dict):
            logger.debug("Invalid transaction format provided!")
            return False

        tx_hash = target_tx.get("tx_hash", "Unknown")
        logger.debug(f"Attempting back-run on target transaction: {tx_hash} ✅ ")

        # Validate required transaction parameters
        required_fields = ["input", "to", "value", "gasPrice"]
        if not all(field in target_tx for field in required_fields):
            missing = [field for field in required_fields if field not in target_tx]
            logger.debug(f"Missing required transaction parameters: {missing}. Skipping... ⚠️ ")
            return False

        try:
            # Decode transaction input with validation
            decoded_tx = await self.decode_transaction_input(
                target_tx.get("input", "0x"),
                self.web3.to_checksum_address(target_tx.get("to", ""))
            )
            if not decoded_tx or "params" not in decoded_tx:
                logger.debug("Failed to decode transaction input for back-run. Skipping...")
                return False

            # Extract and validate path parameter
            path = decoded_tx["params"].get("path", [])
            if not path or not isinstance(path, list) or len(path) < 2:
                logger.debug("Transaction has invalid or no path parameter. Skipping...")
                return False

            # Reverse the path for back-run
            reversed_path = path[::-1]
            decoded_tx["params"]["path"] = reversed_path

            # Prepare back-run transaction
            back_run_tx_details = await self._prepare_back_run_transaction(target_tx, decoded_tx)
            if not back_run_tx_details:
                logger.warning("Failed to prepare back-run transaction! Skipping...")
                return False

            # Simulate back-run transaction
            simulation_success = await self.simulate_transaction(back_run_tx_details)

            if not simulation_success:
                logger.error("Back-run transaction simulation failed! Skipping...")
                return False

            # Send back-run transaction
            if await self.send_bundle([back_run_tx_details]):
                logger.info("Back-run transaction bundle sent successfully. ✅ ")
                return True
            else:
                logger.warning("Failed to send back-run transaction bundle! ⚠️ ")
                return False

        except KeyError as e:
            logger.error(f"Missing required transaction parameter: {e} ⚠️ ")
            return False
        except Exception as e:
            logger.error(f"Unexpected error in back-run execution: {e}  ⚠️ ")
            return False

    async def execute_sandwich_attack(self, target_tx: Dict[str, Any]) -> bool:
        """
        Executes a sandwich attack on the target transaction.

        :param target_tx: Target transaction dictionary.
        :return: True if successful, else False.
        """
        if not isinstance(target_tx, dict):
            logger.critical("Invalid transaction format provided! 🚨 ")
            return False

        tx_hash = target_tx.get("tx_hash", "Unknown")
        logger.debug(f"Attempting sandwich attack on target transaction: {tx_hash} 🥪 ")

        # Validate required transaction parameters
        required_fields = ["input", "to", "value", "gasPrice"]
        if not all(field in target_tx for field in required_fields):
            missing = [field for field in required_fields if field not in target_tx]
            logger.debug(f"Missing required transaction parameters: {missing}. Skipping...")
            return False

        try:
            # Decode transaction input with validation
            decoded_tx = await self.decode_transaction_input(
                target_tx.get("input", "0x"),
                self.web3.to_checksum_address(target_tx.get("to", ""))
            )
            if not decoded_tx or "params" not in decoded_tx:
                logger.debug("Failed to decode transaction input for sandwich attack. Skipping...")
                return False

            # Extract and validate path parameter
            path = decoded_tx["params"].get("path", [])
            if not path or not isinstance(path, list) or len(path) < 2:
                logger.debug("Transaction has invalid or no path parameter. Skipping...")
                return False

            flashloan_asset = self.web3.to_checksum_address(path[0])
            flashloan_amount = self.calculate_flashloan_amount(target_tx)

            if flashloan_amount <= 0:
                logger.debug("Insufficient flashloan amount calculated. Skipping...")
                return False

            # Prepare flashloan transaction
            flashloan_tx = await self.prepare_flashloan_transaction(
                flashloan_asset, flashloan_amount
            )
            if not flashloan_tx:
                logger.debug("Failed to prepare flashloan transaction! Skipping...")
                return False

            # Prepare front-run transaction
            front_run_tx_details = await self._prepare_front_run_transaction(target_tx)
            if not front_run_tx_details:
                logger.warning("Failed to prepare front-run transaction! Skipping...")
                return False

            # Prepare back-run transaction
            back_run_tx_details = await self._prepare_back_run_transaction(target_tx, decoded_tx)
            if not back_run_tx_details:
                logger.warning("Failed to prepare back-run transaction! Skipping...")
                return False

            # Simulate all transactions
            simulation_results = await asyncio.gather(
                self.simulate_transaction(flashloan_tx),
                self.simulate_transaction(front_run_tx_details),
                self.simulate_transaction(back_run_tx_details),
                return_exceptions=True
            )

            if any(isinstance(result, Exception) for result in simulation_results):
                logger.critical("One or more transaction simulations failed! 🚨")
                return False

            if not all(simulation_results):
                logger.warning("Not all transaction simulations were successful!")
                return False

            # Execute transaction bundle
            if await self.send_bundle([flashloan_tx, front_run_tx_details, back_run_tx_details]):
                logger.info("Sandwich attack transaction bundle sent successfully. 🥪✅")
                return True
            else:
                logger.warning("Failed to send sandwich attack transaction bundle! ⚠️ ")
                return False

        except KeyError as e:
            logger.error(f"Missing required transaction parameter: {e} ⚠️ ")
            return False
        except Exception as e:
            logger.error(f"Unexpected error in sandwich attack execution: {e} ⚠️ ")
            return False

    async def _prepare_front_run_transaction(
        self, target_tx: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Prepares the front-run transaction based on the target transaction.

        :param target_tx: Target transaction dictionary.
        :return: Front-run transaction dictionary if successful, else None.
        """
        try:
            decoded_tx = await self.decode_transaction_input(
                target_tx.get("input", "0x"),
                self.web3.to_checksum_address(target_tx.get("to", ""))
            )
            if not decoded_tx:
                logger.debug("Failed to decode target transaction input for front-run. Skipping.")
                return None

            function_name = decoded_tx.get("function_name")
            if not function_name:
                logger.debug("Missing function name in decoded transaction.  🚨")
                return None

            function_params = decoded_tx.get("params", {})
            to_address = self.web3.to_checksum_address(target_tx.get("to", ""))

            # Router address mapping
            routers = {
                self.configuration.UNISWAP_ROUTER_ADDRESS: (self.uniswap_router_contract, "Uniswap"),
                self.configuration.SUSHISWAP_ROUTER_ADDRESS: (self.sushiswap_router_contract, "Sushiswap"),
                self.configuration.PANCAKESWAP_ROUTER_ADDRESS: (self.pancakeswap_router_contract, "Pancakeswap"),
                self.configuration.BALANCER_ROUTER_ADDRESS: (self.balancer_router_contract, "Balancer")
            }

            if to_address not in routers:
                logger.warning(f"Unknown router address {to_address}. Cannot determine exchange.")
                return None

            router_contract, exchange_name = routers[to_address]
            if not router_contract:
                logger.warning(f"Router contract not initialized for {exchange_name}.")
                return None

            # Get the function object by name
            try:
                front_run_function = getattr(router_contract.functions, function_name)(**function_params)
            except AttributeError:
                logger.debug(f"Function {function_name} not found in {exchange_name} router ABI.")
                return None

            # Build the transaction
            front_run_tx = await self.build_transaction(front_run_function)
            logger.info(f"Prepared front-run transaction on {exchange_name} successfully. ✅ ")
            return front_run_tx

        except Exception as e:
            logger.error(f"Error preparing front-run transaction: {e}")
            return None


    async def _prepare_back_run_transaction(
        self, target_tx: Dict[str, Any], decoded_tx: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Prepare the back-run transaction based on the target transaction.

        :param target_tx: Target transaction dictionary.
        :param decoded_tx: Decoded target transaction dictionary.
        :return: Back-run transaction dictionary if successful, else None.
        """
        try:
            function_name = decoded_tx.get("function_name")
            if not function_name:
                logger.debug("Missing function name in decoded transaction.")
                return None

            function_params = decoded_tx.get("params", {})

            # Handle path parameter for back-run
            path = function_params.get("path", [])
            if not path or not isinstance(path, list) or len(path) < 2:
                logger.debug("Transaction has invalid or no path parameter for back-run.")
                return None

            # Reverse the path for back-run
            reversed_path = path[::-1]
            function_params["path"] = reversed_path

            to_address = self.web3.to_checksum_address(target_tx.get("to", ""))

            # Router address mapping
            routers = {
                self.configuration.UNISWAP_ROUTER_ADDRESS: (self.uniswap_router_contract, "Uniswap"),
                self.configuration.SUSHISWAP_ROUTER_ADDRESS: (self.sushiswap_router_contract, "Sushiswap"),
                self.configuration.PANCAKESWAP_ROUTER_ADDRESS: (self.pancakeswap_router_contract, "Pancakeswap"),
                self.configuration.BALANCER_ROUTER_ADDRESS: (self.balancer_router_contract, "Balancer")
            }

            if to_address not in routers:
                logger.debug(f"Unknown router address {to_address}. Cannot determine exchange.")
                return None

            router_contract, exchange_name = routers[to_address]
            if not router_contract:
                logger.debug(f"Router contract not initialized for {exchange_name}.")
                return None

            # Get the function object by name
            try:
                back_run_function = getattr(router_contract.functions, function_name)(**function_params)
            except AttributeError:
                logger.debug(f"Function {function_name} not found in {exchange_name} router ABI.")
                return None

            # Build the transaction
            back_run_tx = await self.build_transaction(back_run_function)
            logger.info(f"Prepared back-run transaction on {exchange_name} successfully.")
            return back_run_tx

        except Exception as e:
            logger.error(f"Error preparing back-run transaction: {e}")
            return None
        
    async def decode_transaction_input(self, input_data: str, contract_address: str) -> Optional[Dict[str, Any]]:
        """
        Decodes the input data of a transaction.

        :param input_data: Hexadecimal input data of the transaction.
        :param contract_address: Address of the contract being interacted with.
        :return: Dictionary containing function name and parameters if successful, else None.
        """
        try:
            # Load the contract ABI
            contract = self.web3.eth.contract(address=contract_address, abi=self.erc20_abi)
            
            # Get function signature (first 4 bytes of input data)
            function_signature = input_data[:10]
            
            # Decode the function call
            try:
                function_obj, function_params = contract.decode_function_input(input_data)
                # Get function name from the function object's FallbackFn or type name
                function_name = getattr(function_obj, '_name', None) or getattr(function_obj, 'fn_name', None)
                if not function_name and hasattr(function_obj, 'function_identifier'):
                    function_name = function_obj.function_identifier
                    
                if not function_name:
                    # Fallback to checking known signatures
                    for name, sig in self.configuration.ERC20_SIGNATURES.items():
                        if sig == function_signature:
                            function_name = name
                            break
                
                decoded_data = {
                    "function_name": function_name,
                    "params": function_params,
                    "signature": function_signature
                }
                
                logger.debug(
                    f"Decoded transaction input: function={function_name}, "
                    f"signature={function_signature}"
                )
                return decoded_data
                
            except Exception as decode_error:
                # Fallback to signature lookup if decoding fails
                for name, sig in self.configuration.ERC20_SIGNATURES.items():
                    if sig == function_signature:
                        # For known signatures, return basic decoded data
                        return {
                            "function_name": name,
                            "params": {},
                            "signature": function_signature
                        }
                raise decode_error
                
        except Exception as e:
            logger.debug(f"Error decoding transaction input: {e}")
            return None

    async def cancel_transaction(self, nonce: int) -> bool:
            """
            Cancels a stuck transaction by sending a zero-value transaction with the same nonce.
    
            :param nonce: Nonce of the transaction to cancel.
            :return: True if cancellation was successful, else False.
            """
            cancel_tx = {
                "to": self.account.address,
                "value": 0,
                "gas": 21_000,
                "gasPrice": self.web3.to_wei("150", "gwei"),  # Higher than the stuck transaction
                "nonce": nonce,
                "chainId": await self.web3.eth.chain_id,
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
                logger.debug(
                    f"Cancellation transaction sent successfully: {tx_hash_hex}"
                )
                return True
            except Exception as e:
                logger.warning(f"Failed to cancel transaction: {e}")
                return False

    async def estimate_gas_limit(self, tx: Dict[str, Any]) -> int:
        """
        Estimates the gas limit for a transaction.

        :param tx: Transaction dictionary.
        :return: Estimated gas limit.
        """
        try:
            gas_estimate = await self.web3.eth.estimate_gas(tx)
            logger.debug(f"Estimated gas: {gas_estimate}")
            return gas_estimate
        except Exception as e:
            logger.debug(
                f"Gas estimation failed: {e}. Using default gas limit of 100000."
            )
            return 100_000  # Default gas limit

    async def get_current_profit(self) -> Decimal:
        """
        Fetches the current profit from the safety net.

        :return: Current profit as Decimal.
        """
        try:
            current_profit = await self.safety_net.get_balance(self.account)
            self.current_profit = Decimal(current_profit)
            logger.debug(f"Current profit: {self.current_profit} ETH")
            return self.current_profit
        except Exception as e:
            logger.error(f"Error fetching current profit: {e}")
            return Decimal("0")

    async def withdraw_eth(self) -> bool:
        """
        Withdraws ETH from the flashloan contract.

        :return: True if successful, else False.
        """
        try:
            withdraw_function = self.flashloan_contract.functions.withdrawETH()
            tx = await self.build_transaction(withdraw_function)
            tx_hash = await self.execute_transaction(tx)
            if tx_hash:
                logger.debug(
                    f"ETH withdrawal transaction sent with hash: {tx_hash}"
                )
                return True
            else:
                logger.warning("Failed to send ETH withdrawal transaction.")
                return False
        except ContractLogicError as e:
            logger.error(f"Contract logic error during ETH withdrawal: {e}")
            return False
        except Exception as e:
            logger.error(f"Error withdrawing ETH: {e}")
            return False

    async def withdraw_token(self, token_address: str) -> bool:
        """
        Withdraws a specific token from the flashloan contract.

        :param token_address: Address of the token to withdraw.
        :return: True if successful, else False.
        """
        try:
            withdraw_function = self.flashloan_contract.functions.withdrawToken(
                self.web3.to_checksum_address(token_address)
            )
            tx = await self.build_transaction(withdraw_function)
            tx_hash = await self.execute_transaction(tx)
            if tx_hash:
                logger.debug(
                    f"Token withdrawal transaction sent with hash: {tx_hash}"
                )
                return True
            else:
                logger.warning("Failed to send token withdrawal transaction.")
                return False
        except ContractLogicError as e:
            logger.error(f"Contract logic error during token withdrawal: {e}")
            return False
        except Exception as e:
            logger.error(f"Error withdrawing token: {e}")
            return False

    async def transfer_profit_to_account(self, amount: Decimal, account: str) -> bool:
        """
        Transfers profit to another account.

        :param amount: Amount of ETH to transfer.
        :param account: Recipient account address.
        :return: True if successful, else False.
        """
        try:
            transfer_function = self.flashloan_contract.functions.transfer(
                self.web3.to_checksum_address(account), int(amount * Decimal("1e18"))
            )
            tx = await self.build_transaction(transfer_function)
            tx_hash = await self.execute_transaction(tx)
            if tx_hash:
                logger.debug(
                    f"Profit transfer transaction sent with hash: {tx_hash}"
                )
                return True
            else:
                logger.warning("Failed to send profit transfer transaction.")
                return False
        except ContractLogicError as e:
            logger.error(f"Contract logic error during profit transfer: {e}")
            return False
        except Exception as e:
            logger.error(f"Error transferring profit: {e}")
            return False

    async def stop(self) -> None:
        try:
            await self.safety_net.stop()
            await self.nonce_core.stop()
            logger.debug("Stopped 0xBuilder. ")
        except Exception as e:
            logger.error(f"Error stopping 0xBuilder: {e} !")
            raise

#//////////////////////////////////////////////////////////////////////////////