// TransactionCore.js

import logger from './Logger.js'; // Assuming Logger.js handles logging
import axios from 'axios';
import { Decimal } from 'decimal.js';
import Semaphore from './Semaphore.js';
import { ContractLogicError, TransactionNotFound } from './Errors.js'; // Custom error classes as needed

class TransactionCore {
    /**
     * TransactionCore is the main transaction engine that Builds and executes transactions,
     * including front-run, back-run, and sandwich attack strategies. It interacts with smart contracts,
     * manages transaction signing, gas price estimation, and handles flashloans.
     *
     * Flashloans are a type of uncollateralized loan that must be borrowed and repaid within the same transaction.
     */

    static MAX_RETRIES = 3;
    static RETRY_DELAY = 1000; // Base delay in milliseconds for retries

    /**
     * Initializes the TransactionCore instance.
     *
     * @param {Web3} web3 - The Web3 instance for blockchain interactions.
     * @param {Object} account - The account object containing address and private key.
     * @param {string} aave_flashloan_address - The address of the Aave Flashloan contract.
     * @param {Array<Object>} aave_flashloan_abi - The ABI of the Aave Flashloan contract.
     * @param {string} aave_lending_pool_address - The address of the Aave Lending Pool contract.
     * @param {Array<Object>} aave_lending_pool_abi - The ABI of the Aave Lending Pool contract.
     * @param {APIConfig} apiconfig - The API configuration instance.
     * @param {MempoolMonitor} monitor - The mempool monitor instance.
     * @param {NonceCore} noncecore - The nonce management instance.
     * @param {SafetyNet} safetynet - The safety net instance.
     * @param {Configuration} configuration - The configuration instance.
     * @param {number} gas_price_multiplier - Multiplier for gas price adjustments.
     * @param {Array<Object>} erc20_abi - The ABI for ERC20 tokens.
     */
    constructor(
        web3,
        account,
        aave_flashloan_address,
        aave_flashloan_abi,
        aave_lending_pool_address,
        aave_lending_pool_abi,
        apiconfig = null,
        monitor = null,
        noncecore = null,
        safetynet = null,
        configuration = null,
        gas_price_multiplier = 1.1,
        erc20_abi = []
    ) {
        this.web3 = web3;
        this.account = account;
        this.configuration = configuration;
        this.monitor = monitor;
        this.apiconfig = apiconfig;
        this.noncecore = noncecore;
        this.safetynet = safetynet;
        this.gas_price_multiplier = gas_price_multiplier;
        this.retry_attempts = TransactionCore.MAX_RETRIES;
        this.erc20_abi = erc20_abi;
        this.current_profit = new Decimal(0);
        this.aave_flashloan_address = aave_flashloan_address;
        this.aave_flashloan_abi = aave_flashloan_abi;
        this.aave_lending_pool_address = aave_lending_pool_address;
        this.aave_lending_pool_abi = aave_lending_pool_abi;

        this._abi_cache = {};
    }

    /**
     * Initializes all required contracts.
     */
    async initialize() {
        try {
            this.flashloan_contract = await this._initialize_contract(
                this.aave_flashloan_address,
                this.aave_flashloan_abi,
                "Flashloan Contract"
            );
            this.lending_pool_contract = await this._initialize_contract(
                this.aave_lending_pool_address,
                this.aave_lending_pool_abi,
                "Lending Pool Contract"
            );
            this.uniswap_router_contract = await this._initialize_contract(
                this.configuration.UNISWAP_ROUTER_ADDRESS,
                this.configuration.UNISWAP_ROUTER_ABI,
                "Uniswap Router Contract"
            );
            this.sushiswap_router_contract = await this._initialize_contract(
                this.configuration.SUSHISWAP_ROUTER_ADDRESS,
                this.configuration.SUSHISWAP_ROUTER_ABI,
                "Sushiswap Router Contract"
            );
            this.pancakeswap_router_contract = await this._initialize_contract(
                this.configuration.PANCAKESWAP_ROUTER_ADDRESS,
                this.configuration.PANCAKESWAP_ROUTER_ABI,
                "Pancakeswap Router Contract"
            );
            this.balancer_router_contract = await this._initialize_contract(
                this.configuration.BALANCER_ROUTER_ADDRESS,
                this.configuration.BALANCER_ROUTER_ABI,
                "Balancer Router Contract"
            );

            if (!this.erc20_abi.length) {
                this.erc20_abi = await this._load_erc20_abi();
            }

            logger.info("All contracts initialized successfully.");
        } catch (error) {
            logger.error(`Initialization failed: ${error.message}`);
            throw error;
        }
    }

    /**
     * Initializes a contract instance with error handling and ABI caching.
     *
     * @param {string} contract_address - The address of the contract.
     * @param {string|Array<Object>} contract_abi - The ABI path or ABI object of the contract.
     * @param {string} contract_name - The name of the contract for logging.
     * @returns {Contract} - The initialized contract instance.
     */
    async _initialize_contract(contract_address, contract_abi, contract_name) {
        try {
            // Load ABI from file if it's a string path and cache it
            if (typeof contract_abi === 'string') {
                if (this._abi_cache[contract_abi]) {
                    var abiContent = this._abi_cache[contract_abi];
                } else {
                    const fs = await import('fs/promises');
                    const abiData = await fs.readFile(contract_abi, 'utf-8');
                    abiContent = JSON.parse(abiData);
                    this._abi_cache[contract_abi] = abiContent;
                }
            } else {
                var abiContent = contract_abi;
            }

            const contractInstance = new this.web3.eth.Contract(
                abiContent,
                this.web3.utils.toChecksumAddress(contract_address)
            );
            logger.info(`Loaded ${contract_name} successfully.`);
            return contractInstance;
        } catch (error) {
            logger.error(`Failed to load ${contract_name} at ${contract_address}: ${error.message}`);
            throw new Error(`Contract initialization failed for ${contract_name}`);
        }
    }

    /**
     * Loads the ERC20 ABI.
     *
     * @returns {Array<Object>} - The ERC20 ABI.
     */
    async _load_erc20_abi() {
        try {
            const abi = await this.apiconfig._load_abi(this.configuration.ERC20_ABI);
            logger.info("ERC20 ABI loaded successfully.");
            return abi;
        } catch (error) {
            logger.error(`Failed to load ERC20 ABI: ${error.message}`);
            throw new Error("ERC20 ABI loading failed");
        }
    }

    /**
     * Builds a transaction dictionary from a contract function call.
     *
     * @param {Object} function_call - The contract function call object.
     * @param {Object} [additional_params={}] - Additional transaction parameters.
     * @returns {Object} - The built transaction dictionary.
     */
    async build_transaction(function_call, additional_params = {}) {
        try {
            const chainId = await this.web3.eth.getChainId();
            const nonce = await this.noncecore.getNonce();
            const txDetails = function_call.buildTransaction({
                chainId: chainId,
                nonce: nonce,
                from: this.account.address,
            });
            Object.assign(txDetails, additional_params);
            txDetails.gas = await this.estimate_gas_smart(txDetails);
            const gasPriceParams = await this.get_dynamic_gas_price();
            Object.assign(txDetails, gasPriceParams);
            logger.debug(`Built transaction: ${JSON.stringify(txDetails)}`);
            return txDetails;
        } catch (error) {
            logger.error(`Error building transaction: ${error.message}`);
            throw error;
        }
    }

    /**
     * Gets dynamic gas price adjusted by the multiplier.
     *
     * @returns {Object} - Object containing 'gasPrice'.
     */
    async get_dynamic_gas_price() {
        try {
            const gasPriceGwei = await this.safetynet.getDynamicGasPrice();
            logger.debug(`Fetched gas price: ${gasPriceGwei} Gwei`);
            const gasPrice = this.web3.utils.toWei((gasPriceGwei * this.gas_price_multiplier).toString(), 'gwei');
            return { gasPrice: gasPrice };
        } catch (error) {
            logger.error(`Error fetching dynamic gas price: ${error.message}. Using default gas price.`);
            const defaultGasPrice = this.web3.utils.toWei("100", 'gwei'); // Default gas price in Wei
            return { gasPrice: defaultGasPrice };
        }
    }

    /**
     * Estimates gas with fallback to a default value.
     *
     * @param {Object} tx - The transaction dictionary.
     * @returns {number} - The estimated gas.
     */
    async estimate_gas_smart(tx) {
        try {
            const gasEstimate = await this.web3.eth.estimateGas(tx);
            logger.debug(`Estimated gas: ${gasEstimate}`);
            return gasEstimate;
        } catch (error) {
            logger.warn(`Gas estimation failed: ${error.message}. Using default gas limit.`);
            return 100000; // Default gas limit
        }
    }

    /**
     * Executes a transaction with retries.
     *
     * @param {Object} tx - The transaction dictionary.
     * @returns {string|null} - The transaction hash if successful, else null.
     */
    async execute_transaction(tx) {
        for (let attempt = 1; attempt <= this.retry_attempts; attempt++) {
            try {
                const signedTx = await this.sign_transaction(tx);
                const receipt = await this.web3.eth.sendSignedTransaction(signedTx);
                logger.info(`Transaction sent successfully with hash: ${receipt.transactionHash}`);
                await this.noncecore.refreshNonce();
                return receipt.transactionHash;
            } catch (error) {
                logger.error(`Error executing transaction: ${error.message}. Attempt ${attempt} of ${TransactionCore.MAX_RETRIES}`);
                if (attempt < TransactionCore.MAX_RETRIES) {
                    const sleepTime = TransactionCore.RETRY_DELAY * attempt;
                    logger.warn(`Retrying in ${sleepTime} ms...`);
                    await this._sleep(sleepTime);
                } else {
                    logger.warn("Failed to execute transaction after multiple attempts.");
                    return null;
                }
            }
        }
        return null;
    }

    /**
     * Signs a transaction with the account's private key.
     *
     * @param {Object} transaction - The transaction dictionary.
     * @returns {string} - The signed transaction data.
     */
    async sign_transaction(transaction) {
        try {
            const signedTx = await this.web3.eth.accounts.signTransaction(
                transaction,
                this.account.privateKey
            );
            logger.info(`Transaction signed successfully: Nonce ${transaction.nonce}.`);
            return signedTx.rawTransaction;
        } catch (error) {
            logger.error(`Error signing transaction: ${error.message}`);
            throw error;
        }
    }

    /**
     * Handles an ETH transfer transaction.
     *
     * @param {Object} target_tx - The target transaction dictionary.
     * @returns {boolean} - True if successful, else False.
     */
    async handle_eth_transaction(target_tx) {
        const tx_hash = target_tx.tx_hash || "Unknown";
        logger.debug(`Handling ETH transaction ${tx_hash}`);
        try {
            const eth_value = target_tx.value || 0;
            if (eth_value <= 0) {
                logger.debug("Transaction value is zero or negative. Skipping.");
                return false;
            }

            const tx_details = {
                to: target_tx.to || "",
                value: eth_value,
                gas: 21000,
                nonce: await this.noncecore.getNonce(),
                chainId: await this.web3.eth.getChainId(),
                from: this.account.address,
            };

            const original_gas_price = parseInt(target_tx.gasPrice || 0);
            if (original_gas_price <= 0) {
                logger.warn("Original gas price is zero or negative. Skipping.");
                return false;
            }

            tx_details.gasPrice = parseInt(original_gas_price * 1.1); // Increase gas price by 10%

            const eth_value_ether = this.web3.utils.fromWei(eth_value.toString(), "ether");
            logger.info(`Building ETH front-run transaction for ${eth_value_ether} ETH to ${tx_details.to}`);
            const tx_hash_executed = await this.execute_transaction(tx_details);
            if (tx_hash_executed) {
                logger.info(`Successfully executed ETH transaction with hash: ${tx_hash_executed}`);
                return true;
            } else {
                logger.warn("Failed to execute ETH transaction.");
                return false;
            }
        } catch (error) {
            logger.error(`Error handling ETH transaction: ${error.message}`);
            return false;
        }
    }

    /**
     * Calculates the flashloan amount based on estimated profit.
     *
     * @param {Object} target_tx - The target transaction dictionary.
     * @returns {number} - The flashloan amount in Wei.
     */
    calculate_flashloan_amount(target_tx) {
        const estimated_profit = target_tx.profit || 0;
        if (estimated_profit > 0) {
            const flashloan_amount = parseInt(new Decimal(estimated_profit).mul(0.8).mul(1e18).toString());
            logger.debug(`Calculated flashloan amount: ${flashloan_amount} Wei based on estimated profit.`);
            return flashloan_amount;
        } else {
            logger.debug("No estimated profit. Setting flashloan amount to 0.");
            return 0;
        }
    }

    /**
     * Simulates a transaction to check if it will succeed.
     *
     * @param {Object} transaction - The transaction dictionary.
     * @returns {boolean} - True if simulation succeeds, else False.
     */
    async simulate_transaction(transaction) {
        logger.debug(`Simulating transaction with nonce ${transaction.nonce}.`);
        try {
            await this.web3.eth.call(transaction, "pending");
            logger.debug("Transaction simulation succeeded.");
            return true;
        } catch (error) {
            if (error instanceof ContractLogicError) {
                logger.debug(`Transaction simulation failed due to contract logic error: ${error.message}`);
            } else {
                logger.debug(`Transaction simulation failed: ${error.message}`);
            }
            return false;
        }
    }

    /**
     * Prepares a flashloan transaction.
     *
     * @param {string} flashloan_asset - The asset address to borrow.
     * @param {number} flashloan_amount - The amount to borrow in Wei.
     * @returns {Object|null} - The transaction dictionary if successful, else null.
     */
    async prepare_flashloan_transaction(flashloan_asset, flashloan_amount) {
        if (flashloan_amount <= 0) {
            logger.debug("Flashloan amount is 0 or less, skipping flashloan transaction preparation.");
            return null;
        }
        try {
            const flashloan_function = this.flashloan_contract.methods.RequestFlashLoan(
                this.web3.utils.toChecksumAddress(flashloan_asset),
                flashloan_amount
            );
            logger.debug(`Preparing flashloan transaction for ${flashloan_amount} Wei of ${flashloan_asset}.`);
            const flashloan_tx = await this.build_transaction(flashloan_function);
            return flashloan_tx;
        } catch (error) {
            logger.error(`Error preparing flashloan transaction: ${error.message}`);
            return null;
        }
    }

    /**
     * Sends a bundle of transactions to MEV relays.
     *
     * @param {Array<Object>} transactions - List of transaction dictionaries.
     * @returns {boolean} - True if bundle sent successfully, else False.
     */
    async send_bundle(transactions) {
        try {
            const signed_txs = await Promise.all(transactions.map(tx => this.sign_transaction(tx)));
            const bundle_payload = {
                jsonrpc: "2.0",
                id: 1,
                method: "eth_sendBundle",
                params: [
                    {
                        txs: signed_txs.map(signedTx => this.web3.utils.toHex(signedTx)),
                        blockNumber: this.web3.utils.toHex(await this.web3.eth.getBlockNumber() + 1),
                    },
                ],
            };

            // List of MEV builders to try
            const mev_builders = [
                {
                    name: "Flashbots",
                    url: "https://relay.flashbots.net",
                    auth_header: "X-Flashbots-Signature",
                },
                // Add other builders as needed...
            ];

            const successes = [];

            for (const builder of mev_builders) {
                const headers = {
                    "Content-Type": "application/json",
                    [builder.auth_header]: `${this.account.address}:${this.account.privateKey}`,
                };

                for (let attempt = 1; attempt <= TransactionCore.MAX_RETRIES; attempt++) {
                    try {
                        logger.debug(`Attempt ${attempt} to send bundle via ${builder.name}.`);
                        const response = await axios.post(builder.url, bundle_payload, { headers, timeout: 30000 });
                        if (response.data.error) {
                            logger.error(`Bundle submission error via ${builder.name}: ${response.data.error}`);
                            throw new Error(response.data.error.message);
                        }
                        logger.info(`Bundle sent successfully via ${builder.name}.`);
                        successes.push(builder.name);
                        break; // Success, move to next builder
                    } catch (error) {
                        logger.error(`Error sending bundle via ${builder.name}: ${error.message}. Attempt ${attempt} of ${TransactionCore.MAX_RETRIES}`);
                        if (attempt < TransactionCore.MAX_RETRIES) {
                            const sleepTime = TransactionCore.RETRY_DELAY * attempt;
                            logger.warn(`Retrying in ${sleepTime} ms...`);
                            await this._sleep(sleepTime);
                        }
                    }
                }
            }

            if (successes.length > 0) {
                await this.noncecore.refreshNonce();
                logger.info(`Bundle successfully sent to builders: ${successes.join(', ')}`);
                return true;
            } else {
                logger.warn("Failed to send bundle to any MEV builders.");
                return false;
            }
        } catch (error) {
            logger.error(`Unexpected error in send_bundle: ${error.message}`);
            return false;
        }
    }

    /**
     * Executes a front-run transaction with validation and error handling.
     *
     * @param {Object} target_tx - The target transaction dictionary.
     * @returns {boolean} - True if successful, else False.
     */
    async front_run(target_tx) {
        /**
         * A front-running attack is a type of transaction ordering dependence (TOD) attack where
         * the attacker attempts to exploit the time delay between the transaction submission and
         * its confirmation on the blockchain. The attacker can insert a transaction in the same
         * block as the target transaction to manipulate the order of execution and gain an advantage.
         */
        if (typeof target_tx !== 'object' || !target_tx) {
            logger.debug("Invalid transaction format provided!");
            return false;
        }

        const tx_hash = target_tx.tx_hash || "Unknown";
        logger.debug(`Attempting front-run on target transaction: ${tx_hash}`);

        // Validate required transaction parameters
        const required_fields = ["input", "to", "value", "gasPrice"];
        const missing_fields = required_fields.filter(field => !(field in target_tx));
        if (missing_fields.length > 0) {
            logger.debug(`Missing required transaction parameters: ${missing_fields.join(', ')}. Skipping...`);
            return false;
        }

        try {
            // Decode transaction input with validation
            const decoded_tx = await this.decode_transaction_input(target_tx.input, this.web3.utils.toChecksumAddress(target_tx.to));
            if (!decoded_tx || !decoded_tx.params) {
                logger.debug("Failed to decode transaction input for front-run.");
                return false;
            }

            // Extract and validate path parameter
            const path = decoded_tx.params.path || [];
            if (!Array.isArray(path) || path.length < 2) {
                logger.debug("Transaction has invalid or no path parameter. Skipping...");
                return false;
            }

            // Prepare flashloan
            try {
                const flashloan_asset = this.web3.utils.toChecksumAddress(path[0]);
                const flashloan_amount = this.calculate_flashloan_amount(target_tx);

                if (flashloan_amount <= 0) {
                    logger.debug("Insufficient flashloan amount calculated.");
                    return false;
                }

                const flashloan_tx = await this.prepare_flashloan_transaction(flashloan_asset, flashloan_amount);
                if (!flashloan_tx) {
                    logger.debug("Failed to prepare flashloan transaction!");
                    return false;
                }

                // Prepare front-run transaction
                const front_run_tx_details = await this._prepare_front_run_transaction(target_tx);
                if (!front_run_tx_details) {
                    logger.warning("Failed to prepare front-run transaction!");
                    return false;
                }

                // Simulate transactions
                const simulation_success = await Promise.all([
                    this.simulate_transaction(flashloan_tx),
                    this.simulate_transaction(front_run_tx_details)
                ]);

                if (!simulation_success.every(success => success)) {
                    logger.error("Transaction simulation failed!");
                    return false;
                }

                // Send transaction bundle
                const bundle_sent = await this.send_bundle([flashloan_tx, front_run_tx_details]);
                if (bundle_sent) {
                    logger.info("Front-run transaction bundle sent successfully.");
                    return true;
                } else {
                    logger.warning("Failed to send front-run transaction bundle!");
                    return false;
                }

            } catch (error) {
                logger.error(`Error preparing flashloan: ${error.message}`);
                return false;
            }

        } catch (error) {
            logger.error(`Unexpected error in front-run execution: ${error.message}`);
            return false;
        }
    }

    /**
     * Executes a back-run transaction with validation and error handling.
     *
     * @param {Object} target_tx - The target transaction dictionary.
     * @returns {boolean} - True if successful, else False.
     */
    async back_run(target_tx) {
        if (typeof target_tx !== 'object' || !target_tx) {
            logger.debug("Invalid transaction format provided!");
            return false;
        }

        const tx_hash = target_tx.tx_hash || "Unknown";
        logger.debug(`Attempting back-run on target transaction: ${tx_hash}`);

        // Validate required transaction parameters
        const required_fields = ["input", "to", "value", "gasPrice"];
        const missing_fields = required_fields.filter(field => !(field in target_tx));
        if (missing_fields.length > 0) {
            logger.debug(`Missing required transaction parameters: ${missing_fields.join(', ')}. Skipping...`);
            return false;
        }

        try {
            // Decode transaction input with validation
            const decoded_tx = await this.decode_transaction_input(target_tx.input, this.web3.utils.toChecksumAddress(target_tx.to));
            if (!decoded_tx || !decoded_tx.params) {
                logger.debug("Failed to decode transaction input for back-run.");
                return false;
            }

            // Extract and validate path parameter
            const path = decoded_tx.params.path || [];
            if (!Array.isArray(path) || path.length < 2) {
                logger.debug("Transaction has invalid or no path parameter. Skipping...");
                return false;
            }

            // Reverse the path for back-run
            const reversed_path = path.slice().reverse();
            decoded_tx.params.path = reversed_path;

            // Prepare back-run transaction
            const back_run_tx_details = await this._prepare_back_run_transaction(target_tx, decoded_tx);
            if (!back_run_tx_details) {
                logger.warning("Failed to prepare back-run transaction!");
                return false;
            }

            // Simulate back-run transaction
            const simulation_success = await this.simulate_transaction(back_run_tx_details);
            if (!simulation_success) {
                logger.error("Back-run transaction simulation failed!");
                return false;
            }

            // Send back-run transaction
            const bundle_sent = await this.send_bundle([back_run_tx_details]);
            if (bundle_sent) {
                logger.info("Back-run transaction bundle sent successfully.");
                return true;
            } else {
                logger.warning("Failed to send back-run transaction bundle!");
                return false;
            }

        } catch (error) {
            logger.error(`Unexpected error in back-run execution: ${error.message}`);
            return false;
        }
    }

    /**
     * Executes a sandwich attack on the target transaction.
     *
     * @param {Object} target_tx - The target transaction dictionary.
     * @returns {boolean} - True if successful, else False.
     */
    async execute_sandwich_attack(target_tx) {
        if (typeof target_tx !== 'object' || !target_tx) {
            logger.debug("Invalid transaction format provided!");
            return false;
        }

        const tx_hash = target_tx.tx_hash || "Unknown";
        logger.debug(`Attempting sandwich attack on target transaction: ${tx_hash}`);

        // Validate required transaction parameters
        const required_fields = ["input", "to", "value", "gasPrice"];
        const missing_fields = required_fields.filter(field => !(field in target_tx));
        if (missing_fields.length > 0) {
            logger.debug(`Missing required transaction parameters: ${missing_fields.join(', ')}. Skipping...`);
            return false;
        }

        try {
            // Decode transaction input with validation
            const decoded_tx = await this.decode_transaction_input(target_tx.input, this.web3.utils.toChecksumAddress(target_tx.to));
            if (!decoded_tx || !decoded_tx.params) {
                logger.debug("Failed to decode transaction input for sandwich attack.");
                return false;
            }

            // Extract and validate path parameter
            const path = decoded_tx.params.path || [];
            if (!Array.isArray(path) || path.length < 2) {
                logger.debug("Transaction has invalid or no path parameter. Skipping...");
                return false;
            }

            const flashloan_asset = this.web3.utils.toChecksumAddress(path[0]);
            const flashloan_amount = this.calculate_flashloan_amount(target_tx);

            if (flashloan_amount <= 0) {
                logger.debug("Insufficient flashloan amount calculated.");
                return false;
            }

            // Prepare flashloan transaction
            const flashloan_tx = await this.prepare_flashloan_transaction(flashloan_asset, flashloan_amount);
            if (!flashloan_tx) {
                logger.debug("Failed to prepare flashloan transaction!");
                return false;
            }

            // Prepare front-run transaction
            const front_run_tx_details = await this._prepare_front_run_transaction(target_tx);
            if (!front_run_tx_details) {
                logger.warning("Failed to prepare front-run transaction!");
                return false;
            }

            // Prepare back-run transaction
            const back_run_tx_details = await this._prepare_back_run_transaction(target_tx, decoded_tx);
            if (!back_run_tx_details) {
                logger.warning("Failed to prepare back-run transaction!");
                return false;
            }

            // Simulate all transactions
            const simulation_results = await Promise.all([
                this.simulate_transaction(flashloan_tx),
                this.simulate_transaction(front_run_tx_details),
                this.simulate_transaction(back_run_tx_details)
            ]);

            if (!simulation_results.every(success => success)) {
                logger.error("One or more transaction simulations failed!");
                return false;
            }

            // Execute transaction bundle
            const bundle_sent = await this.send_bundle([flashloan_tx, front_run_tx_details, back_run_tx_details]);
            if (bundle_sent) {
                logger.info("Sandwich attack transaction bundle sent successfully.");
                return true;
            } else {
                logger.warning("Failed to send sandwich attack transaction bundle!");
                return false;
            }

        } catch (error) {
            logger.error(`Unexpected error in sandwich attack execution: ${error.message}`);
            return false;
        }
    }

    /**
     * Prepares the front-run transaction based on the target transaction.
     *
     * @param {Object} target_tx - The target transaction dictionary.
     * @returns {Object|null} - The front-run transaction dictionary if successful, else null.
     */
    async _prepare_front_run_transaction(target_tx) {
        try {
            const decoded_tx = await this.decode_transaction_input(target_tx.input, this.web3.utils.toChecksumAddress(target_tx.to));
            if (!decoded_tx) {
                logger.debug("Failed to decode target transaction input for front-run.");
                return null;
            }

            const function_name = decoded_tx.function_name;
            if (!function_name) {
                logger.debug("Missing function name in decoded transaction.");
                return null;
            }

            const function_params = decoded_tx.params;
            const to_address = this.web3.utils.toChecksumAddress(target_tx.to);

            // Router address mapping
            const routers = {
                [this.configuration.UNISWAP_ROUTER_ADDRESS]: { contract: this.uniswap_router_contract, name: "Uniswap" },
                [this.configuration.SUSHISWAP_ROUTER_ADDRESS]: { contract: this.sushiswap_router_contract, name: "Sushiswap" },
                [this.configuration.PANCAKESWAP_ROUTER_ADDRESS]: { contract: this.pancakeswap_router_contract, name: "Pancakeswap" },
                [this.configuration.BALANCER_ROUTER_ADDRESS]: { contract: this.balancer_router_contract, name: "Balancer" },
            };

            const router_info = routers[to_address];
            if (!router_info) {
                logger.warning(`Unknown router address ${to_address}. Cannot determine exchange.`);
                return null;
            }

            const { contract, name } = router_info;

            // Get the function object by name
            if (typeof contract.methods[function_name] !== 'function') {
                logger.debug(`Function ${function_name} not found in ${name} router ABI.`);
                return null;
            }

            const front_run_function = contract.methods[function_name](...Object.values(function_params));
            // Build the transaction
            const front_run_tx = await this.build_transaction(front_run_function);
            logger.info(`Prepared front-run transaction on ${name} successfully.`);
            return front_run_tx;
        } catch (error) {
            logger.error(`Error preparing front-run transaction: ${error.message}`);
            return null;
        }
    }

    /**
     * Prepares the back-run transaction based on the target transaction.
     *
     * @param {Object} target_tx - The target transaction dictionary.
     * @param {Object} decoded_tx - The decoded target transaction dictionary.
     * @returns {Object|null} - The back-run transaction dictionary if successful, else null.
     */
    async _prepare_back_run_transaction(target_tx, decoded_tx) {
        try {
            const function_name = decoded_tx.function_name;
            if (!function_name) {
                logger.debug("Missing function name in decoded transaction.");
                return null;
            }

            const function_params = decoded_tx.params;

            // Handle path parameter for back-run
            const path = function_params.path || [];
            if (!Array.isArray(path) || path.length < 2) {
                logger.debug("Transaction has invalid or no path parameter for back-run.");
                return null;
            }

            // Reverse the path for back-run
            const reversed_path = path.slice().reverse();
            function_params.path = reversed_path;

            const to_address = this.web3.utils.toChecksumAddress(target_tx.to);

            // Router address mapping
            const routers = {
                [this.configuration.UNISWAP_ROUTER_ADDRESS]: { contract: this.uniswap_router_contract, name: "Uniswap" },
                [this.configuration.SUSHISWAP_ROUTER_ADDRESS]: { contract: this.sushiswap_router_contract, name: "Sushiswap" },
                [this.configuration.PANCAKESWAP_ROUTER_ADDRESS]: { contract: this.pancakeswap_router_contract, name: "Pancakeswap" },
                [this.configuration.BALANCER_ROUTER_ADDRESS]: { contract: this.balancer_router_contract, name: "Balancer" },
            };

            const router_info = routers[to_address];
            if (!router_info) {
                logger.debug(`Unknown router address ${to_address}. Cannot determine exchange.`);
                return null;
            }

            const { contract, name } = router_info;

            // Get the function object by name
            if (typeof contract.methods[function_name] !== 'function') {
                logger.debug(`Function ${function_name} not found in ${name} router ABI.`);
                return null;
            }

            const back_run_function = contract.methods[function_name](...Object.values(function_params));
            // Build the transaction
            const back_run_tx = await this.build_transaction(back_run_function);
            logger.info(`Prepared back-run transaction on ${name} successfully.`);
            return back_run_tx;
        } catch (error) {
            logger.error(`Error preparing back-run transaction: ${error.message}`);
            return null;
        }
    }

    /**
     * Decodes the input data of a transaction to understand its purpose.
     *
     * @param {string} input_data - Hexadecimal input data of the transaction.
     * @param {string} contract_address - Address of the contract being interacted with.
     * @returns {Object|null} - Object containing function name and parameters if successful, else null.
     */
    async decode_transaction_input(input_data, contract_address) {
        try {
            const contract = new this.web3.eth.Contract(this.erc20_abi, contract_address);
            const decoded = contract.methods[contract.options.jsonInterface.find(method => input_data.startsWith(method.signature)).name].decodeParameters([], input_data);
            const function_obj = contract.methods[contract.options.jsonInterface.find(method => input_data.startsWith(method.signature)).name];
            const function_name = function_obj._method.name;
            const params = decoded;

            const decoded_data = {
                function_name: function_name,
                params: params,
            };
            logger.debug(`Decoded transaction input: ${JSON.stringify(decoded_data)}`);
            return decoded_data;
        } catch (error) {
            logger.warn(`Failed in decoding transaction input: ${error.message}`);
            return null;
        }
    }

    /**
     * Cancels a stuck transaction by sending a zero-value transaction with the same nonce.
     *
     * @param {number} nonce - The nonce of the transaction to cancel.
     * @returns {boolean} - True if cancellation was successful, else False.
     */
    async cancel_transaction(nonce) {
        const cancel_tx = {
            to: this.account.address,
            value: 0,
            gas: 21000,
            gasPrice: this.web3.utils.toWei("150", "gwei"), // Higher than the stuck transaction
            nonce: nonce,
            chainId: await this.web3.eth.getChainId(),
            from: this.account.address,
        };

        try {
            const signed_cancel_tx = await this.sign_transaction(cancel_tx);
            const receipt = await this.web3.eth.sendSignedTransaction(signed_cancel_tx);
            const tx_hash_hex = typeof receipt.transactionHash === 'string' ? receipt.transactionHash : receipt.transactionHash.toString('hex');
            logger.debug(`Cancellation transaction sent successfully: ${tx_hash_hex}`);
            return true;
        } catch (error) {
            logger.warn(`Failed to cancel transaction: ${error.message}`);
            return false;
        }
    }

    /**
     * Estimates the gas limit for a transaction.
     *
     * @param {Object} tx - The transaction dictionary.
     * @returns {number} - The estimated gas limit.
     */
    async estimate_gas_limit(tx) {
        try {
            const gas_estimate = await this.web3.eth.estimateGas(tx);
            logger.debug(`Estimated gas: ${gas_estimate}`);
            return gas_estimate;
        } catch (error) {
            logger.debug(`Gas estimation failed: ${error.message}. Using default gas limit of 100000.`);
            return 100000; // Default gas limit
        }
    }

    /**
     * Fetches the current profit from the safety net.
     *
     * @returns {Decimal} - The current profit as Decimal.
     */
    async get_current_profit() {
        try {
            const current_profit = await this.safetynet.getBalance(this.account);
            this.current_profit = new Decimal(current_profit);
            logger.debug(`Current profit: ${this.current_profit} ETH`);
            return this.current_profit;
        } catch (error) {
            logger.error(`Error fetching current profit: ${error.message}`);
            return new Decimal(0);
        }
    }

    /**
     * Withdraws ETH from the flashloan contract.
     *
     * @returns {boolean} - True if successful, else False.
     */
    async withdraw_eth() {
        try {
            const withdraw_function = this.flashloan_contract.methods.withdrawETH();
            const tx = await this.build_transaction(withdraw_function);
            const tx_hash = await this.execute_transaction(tx);
            if (tx_hash) {
                logger.debug(`ETH withdrawal transaction sent with hash: ${tx_hash}`);
                return true;
            } else {
                logger.warn("Failed to send ETH withdrawal transaction.");
                return false;
            }
        } catch (error) {
            logger.error(`Error withdrawing ETH: ${error.message}`);
            return false;
        }
    }

    /**
     * Withdraws a specific token from the flashloan contract.
     *
     * @param {string} token_address - Address of the token to withdraw.
     * @returns {boolean} - True if successful, else False.
     */
    async withdraw_token(token_address) {
        try {
            const withdraw_function = this.flashloan_contract.methods.withdrawToken(
                this.web3.utils.toChecksumAddress(token_address)
            );
            const tx = await this.build_transaction(withdraw_function);
            const tx_hash = await this.execute_transaction(tx);
            if (tx_hash) {
                logger.debug(`Token withdrawal transaction sent with hash: ${tx_hash}`);
                return true;
            } else {
                logger.warn("Failed to send token withdrawal transaction.");
                return false;
            }
        } catch (error) {
            logger.error(`Error withdrawing token: ${error.message}`);
            return false;
        }
    }

    /**
     * Transfers profit to another account.
     *
     * @param {Decimal} amount - Amount of ETH to transfer.
     * @param {string} account - Recipient account address.
     * @returns {boolean} - True if successful, else False.
     */
    async transfer_profit_to_account(amount, account) {
        try {
            const transfer_function = this.flashloan_contract.methods.transfer(
                this.web3.utils.toChecksumAddress(account),
                this.web3.utils.toWei(amount.toString(), "ether")
            );
            const tx = await this.build_transaction(transfer_function);
            const tx_hash = await this.execute_transaction(tx);
            if (tx_hash) {
                logger.debug(`Profit transfer transaction sent with hash: ${tx_hash}`);
                return true;
            } else {
                logger.warn("Failed to send profit transfer transaction.");
                return false;
            }
        } catch (error) {
            logger.error(`Error transferring profit: ${error.message}`);
            return false;
        }
    }

    /**
     * Sleeps for the specified duration.
     *
     * @param {number} ms - Duration to sleep in milliseconds.
     * @returns {Promise<void>}
     */
    _sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    /**
     * Stops the TransactionCore instance gracefully.
     */
    async stop() {
        try {
            await this.safetynet.stop();
            await this.noncecore.stop();
            logger.debug("Stopped TransactionCore successfully.");
        } catch (error) {
            logger.error(`Error stopping TransactionCore: ${error.message}`);
            throw error;
        }
    }
}

export default TransactionCore;
