const { ethers } = require("ethers");
const axios = require("axios");
const { ABI_Registry } = require("./abi_registry");
const { API_Config, Configuration } = require("./configuration");
const { Market_Monitor, Mempool_Monitor } = require("./monitor");
const { Nonce_Core } = require("./nonce");
const { Safety_Net, Strategy_Net } = require("./net");
const hexBytes = require("hexbytes");
const tracemalloc = require("tracemalloc");
const logger = require("./logger");

class Transaction_Core {
    static MAX_RETRIES = 3;
    static RETRY_DELAY = 1.0; // Base delay in seconds for retries
    static DEFAULT_GAS_LIMIT = 100000;
    static DEFAULT_CANCEL_GAS_PRICE_GWEI = 150;
    static DEFAULT_PROFIT_TRANSFER_MULTIPLIER = Math.pow(10, 18);
    static DEFAULT_GAS_PRICE_GWEI = 50;

    constructor(
        web3, account, AAVE_FLASHLOAN_ADDRESS, AAVE_FLASHLOAN_ABI,
        AAVE_POOL_ADDRESS, AAVE_POOL_ABI, api_config = null, market_monitor = null,
        mempool_monitor = null, nonce_core = null, safety_net = null, configuration = null,
        gas_price_multiplier = 1.1, erc20_abi = null, uniswap_address = null, uniswap_abi = null
    ) {
        this.web3 = web3;
        this.account = account;
        this.configuration = configuration;
        this.market_monitor = market_monitor;
        this.mempool_monitor = mempool_monitor;
        this.api_config = api_config;
        this.nonce_core = nonce_core;
        this.safety_net = safety_net;
        this.gas_price_multiplier = gas_price_multiplier;
        this.RETRY_ATTEMPTS = Transaction_Core.MAX_RETRIES;
        this.erc20_abi = erc20_abi || [];
        this.current_profit = ethers.BigNumber.from(0);
        this.AAVE_FLASHLOAN_ADDRESS = AAVE_FLASHLOAN_ADDRESS;
        this.AAVE_FLASHLOAN_ABI = AAVE_FLASHLOAN_ABI;
        this.AAVE_POOL_ADDRESS = AAVE_POOL_ADDRESS;
        this.AAVE_POOL_ABI = AAVE_POOL_ABI;
        this.abi_registry = new ABI_Registry();
        this.uniswap_address = uniswap_address;
        this.uniswap_abi = uniswap_abi || [];
    }

    normalize_address(address) {
        try {
            return ethers.utils.getAddress(address); // Normalize Ethereum address to checksum format
        } catch (e) {
            logger.error(`Error normalizing address ${address}: ${e}`);
            throw e;
        }
    }

    async initialize() {
        try {
            // Initialize contracts using ABIs from registry
            const router_configs = [
                { address: this.configuration.UNISWAP_ADDRESS, abi_type: 'uniswap', name: 'Uniswap' },
                { address: this.configuration.SUSHISWAP_ADDRESS, abi_type: 'sushiswap', name: 'Sushiswap' },
            ];

            for (const { address, abi_type, name } of router_configs) {
                try {
                    const normalized_address = this.normalize_address(address);
                    const abi = this.abi_registry.get_abi(abi_type);
                    if (!abi) throw new Error(`Failed to load ${name} ABI`);
                    const contract = new ethers.Contract(normalized_address, abi, this.web3);

                    // Validate contract
                    await this._validate_contract(contract, name, abi_type);
                    this[`${name.toLowerCase()}_router_contract`] = contract;
                } catch (e) {
                    logger.error(`Failed to initialize ${name} router: ${e}`);
                    throw e;
                }
            }

            // Initialize Aave contracts
            this.aave_flashloan = new ethers.Contract(this.normalize_address(this.configuration.AAVE_FLASHLOAN_ADDRESS), this.configuration.AAVE_FLASHLOAN_ABI, this.web3);
            await this._validate_contract(this.aave_flashloan, "Aave Flashloan", 'aave_flashloan');

            this.aave_pool = new ethers.Contract(this.normalize_address(this.AAVE_POOL_ADDRESS), this.AAVE_POOL_ABI, this.web3);
            await this._validate_contract(this.aave_pool, "Aave Lending Pool", 'aave');

            logger.info("Transaction Core initialized successfully");
        } catch (e) {
            logger.error(`Transaction Core initialization failed: ${e}`);
            throw e;
        }
    }

    async send_bundle(transactions) {
        try {
            const signed_txs = await Promise.all(
                transactions.map(tx => this.sign_transaction(tx))
            );
            const bundle_payload = {
                jsonrpc: "2.0",
                id: 1,
                method: "eth_sendBundle",
                params: [{
                    txs: signed_txs.map(signed_tx => signed_tx.hex()),
                    blockNumber: (await this.web3.getBlockNumber()) + 1
                }]
            };

            // List of MEV builders to try
            const mev_builders = [
                { name: "Flashbots", url: "https://relay.flashbots.net", auth_header: "X-Flashbots-Signature" },
            ];

            let successes = [];

            for (let builder of mev_builders) {
                const headers = {
                    "Content-Type": "application/json",
                    [builder.auth_header]: `${this.account.address}:${this.account.privateKey}`,
                };

                for (let attempt = 1; attempt <= this.RETRY_ATTEMPTS; attempt++) {
                    try {
                        logger.debug(`Attempt ${attempt} to send bundle via ${builder.name}...`);
                        const response = await axios.post(builder.url, bundle_payload, { headers });

                        if (response.data.error) {
                            logger.error(`Bundle submission error via ${builder.name}: ${response.data.error}`);
                            throw new Error(response.data.error);
                        }

                        logger.info(`Bundle sent successfully via ${builder.name}.`);
                        successes.push(builder.name);
                        break; // Move to next builder if successful
                    } catch (e) {
                        logger.error(`Error with ${builder.name} (Attempt ${attempt}): ${e}`);
                        if (attempt < this.RETRY_ATTEMPTS) {
                            await new Promise(resolve => setTimeout(resolve, this.RETRY_DELAY * 1000));
                        }
                    }
                }
            }

            if (successes.length > 0) {
                await this.nonce_core.refresh_nonce();
                logger.info(`Bundle successfully sent to builders: ${successes.join(', ')}`);
                return true;
            } else {
                logger.warning("Failed to send bundle to any MEV builders.");
                return false;
            }
        } catch (e) {
            logger.error(`Unexpected error in send_bundle: ${e}`);
            return false;
        }
    }

    async front_run(target_tx) {
        const decoded_tx = await this._validate_transaction(target_tx, "front-run");
        if (!decoded_tx) return false;

        try {
            const path = decoded_tx.params.path;
            const flashloan_tx = await this._prepare_flashloan(path[0], target_tx);
            const front_run_tx = await this._prepare_front_run_transaction(target_tx);

            if (!flashloan_tx || !front_run_tx) return false;

            if (await this._validate_and_send_bundle([flashloan_tx, front_run_tx])) {
                logger.info("Front-run executed successfully.");
                return true;
            }

            return false;
        } catch (e) {
            logger.error(`Front-run execution failed: ${e}`);
            return false;
        }
    }

    async back_run(target_tx) {
        const decoded_tx = await this._validate_transaction(target_tx, "back-run");
        if (!decoded_tx) return false;

        try {
            const back_run_tx = await this._prepare_back_run_transaction(target_tx, decoded_tx);
            if (!back_run_tx) return false;

            if (await this._validate_and_send_bundle([back_run_tx])) {
                logger.info("Back-run executed successfully.");
                return true;
            }

            return false;
        } catch (e) {
            logger.error(`Back-run execution failed: ${e}`);
            return false;
        }
    }

    async execute_sandwich_attack(target_tx) {
        const decoded_tx = await this._validate_transaction(target_tx, "sandwich");
        if (!decoded_tx) return false;

        try {
            const path = decoded_tx.params.path;
            const flashloan_tx = await this._prepare_flashloan(path[0], target_tx);
            const front_tx = await this._prepare_front_run_transaction(target_tx);
            const back_tx = await this._prepare_back_run_transaction(target_tx, decoded_tx);

            if (!flashloan_tx || !front_tx || !back_tx) return false;

            if (await this._validate_and_send_bundle([flashloan_tx, front_tx, back_tx])) {
                logger.info("Sandwich attack executed successfully.");
                return true;
            }

            return false;
        } catch (e) {
            logger.error(`Sandwich attack execution failed: ${e}`);
            return false;
        }
    }

    async _prepare_flashloan(asset, target_tx) {
        const flashloan_amount = this.calculate_flashloan_amount(target_tx);
        if (flashloan_amount <= 0) return null;
        return this.prepare_flashloan_transaction(ethers.utils.getAddress(asset), flashloan_amount);
    }

    async _validate_and_send_bundle(transactions) {
        const simulations = await Promise.all(transactions.map(tx => this.simulate_transaction(tx)));

        if (simulations.includes(false)) {
            logger.warning("Transaction simulation failed.");
            return false;
        }

        return await this.send_bundle(transactions);
    }

    async _prepare_front_run_transaction(target_tx) {
        // Prepare the front-run transaction
        const decoded_tx = await this.decode_transaction_input(target_tx.input, target_tx.to);
        if (!decoded_tx) return null;

        const { function_name, params } = decoded_tx;
        const router_contract = this.uniswap_contract;

        try {
            const front_run_function = router_contract[function_name](params);
            return await this.build_transaction(front_run_function);
        } catch (e) {
            logger.error(`Error preparing front-run transaction: ${e}`);
            return null;
        }
    }

    async _prepare_back_run_transaction(target_tx, decoded_tx) {
        // Prepare the back-run transaction
        const { function_name, params } = decoded_tx;
        const path = params.path;
        if (!path || path.length < 2) return null;

        const reversed_path = path.reverse();
        params.path = reversed_path;

        try {
            const router_contract = this.uniswap_contract;
            const back_run_function = router_contract[function_name](params);
            return await this.build_transaction(back_run_function);
        } catch (e) {
            logger.error(`Error preparing back-run transaction: ${e}`);
            return null;
        }
    }

    async decode_transaction_input(input_data, contract_address) {
        try {
            const selector = input_data.slice(0, 10).substring(2);
            const method_name = this.abi_registry.get_method_selector(selector);

            if (!method_name) {
                logger.debug(`Unknown method selector: ${selector}`);
                return null;
            }

            const abi = await this.abi_registry.get_abi('erc20');
            const contract = new ethers.Contract(contract_address, abi, this.web3);
            const decoded_params = contract.interface.decodeFunctionData(input_data);

            return { function_name: method_name, params: decoded_params };
        } catch (e) {
            logger.error(`Error decoding transaction input: ${e}`);
            return null;
        }
    }

    async simulate_transaction(transaction) {
        try {
            await this.web3.call(transaction, { blockTag: 'pending' });
            logger.debug("Transaction simulation succeeded.");
            return true;
        } catch (e) {
            logger.debug(`Transaction simulation failed: ${e}`);
            return false;
        }
    }

    async _validate_contract(contract, name, abi_type) {
        try {
            if (name.includes("Lending Pool")) {
                await contract.getReservesList();
                logger.debug(`${name} contract validated successfully via getReservesList()`);
            } else if (name.includes("Flashloan")) {
                await contract.ADDRESSES_PROVIDER();
                logger.debug(`${name} contract validated successfully via ADDRESSES_PROVIDER()`);
            } else if (['uniswap', 'sushiswap'].includes(abi_type)) {
                const path = [this.configuration.WETH_ADDRESS, this.configuration.USDC_ADDRESS];
                await contract.getAmountsOut(1000000, path);
                logger.debug(`${name} contract validated successfully`);
            } else {
                logger.debug(`No specific validation for ${name}, but initialized`);
            }
        } catch (e) {
            logger.warning(`Contract validation warning for ${name}: ${e}`);
        }
    }

    async build_transaction(function_call, additional_params = {}) {
        try {
            const chain_id = await this.web3.getChainId();
            const latest_block = await this.web3.getBlock('latest');
            const supports_eip1559 = latest_block.baseFeePerGas !== undefined;

            const tx_params = {
                chainId: chain_id,
                nonce: await this.nonce_core.get_nonce(),
                from: this.account.address,
            };

            if (supports_eip1559) {
                const base_fee = latest_block.baseFeePerGas;
                const priority_fee = await this.web3.getMaxPriorityFeePerGas();
                tx_params.maxFeePerGas = base_fee.mul(2);
                tx_params.maxPriorityFeePerGas = priority_fee;
            } else {
                tx_params.gasPrice = await this._get_dynamic_gas_parameters();
            }

            const tx_details = await function_call.buildTransaction(tx_params);
            tx_details.gas = await this.estimate_gas_smart(tx_details) * 1.1;

            return { ...tx_details, ...additional_params };
        } catch (e) {
            logger.error(`Error building transaction: ${e}`);
            throw e;
        }
    }
    
    

    async _get_dynamic_gas_parameters() {
        try {
            const gas_price_gwei = await this.safety_net.get_dynamic_gas_price();
            logger.debug(`Fetched gas price: ${gas_price_gwei} Gwei`);
            return ethers.utils.parseUnits(gas_price_gwei.toString(), "gwei");
        } catch (e) {
            logger.error(`Error fetching dynamic gas price: ${e}`);
            return ethers.utils.parseUnits(Transaction_Core.DEFAULT_GAS_PRICE_GWEI.toString(), "gwei");
        }
    }

    async estimate_gas_smart(tx) {
        try {
            const gas_estimate = await this.web3.estimateGas(tx);
            logger.debug(`Estimated gas: ${gas_estimate}`);
            return gas_estimate;
        } catch (e) {
            logger.warning(`Gas estimation failed: ${e}. Using default gas limit.`);
            return Transaction_Core.DEFAULT_GAS_LIMIT;
        }
    }

    async execute_transaction(tx) {
        try {
            for (let attempt = 1; attempt <= this.MAX_RETRIES; attempt++) {
                const signed_tx = await this.sign_transaction(tx);
                const tx_hash = await this.web3.sendTransaction(signed_tx);
                logger.debug(`Transaction sent successfully: ${tx_hash}`);
                return tx_hash;
            }
        } catch (e) {
            logger.warning(`Transaction failed, retrying: ${e}`);
            await new Promise(resolve => setTimeout(resolve, this.RETRY_DELAY * 1000));
        }
        logger.error("Failed to execute transaction after retries");
        return null;
    }

    async sign_transaction(transaction) {
        try {
            const signed_tx = await this.web3.signTransaction(transaction, this.account.privateKey);
            logger.debug(`Transaction signed successfully: Nonce ${transaction.nonce}`);
            return signed_tx;
        } catch (e) {
            logger.error(`Error signing transaction: ${e}`);
            throw e;
        }
    }

    async stop() {
        try {
            await this.safety_net.stop();
            await this.nonce_core.stop();
            logger.debug("Stopped 0xBuilder.");
        } catch (e) {
            logger.error(`Error stopping 0xBuilder: ${e}`);
            throw e;
        }
    }
}

module.exports = { Transaction_Core };



const logger = require('./logger');  // Assuming you have a logger configured

class Main_Core {
    constructor(configuration) {
        this.configuration = configuration;
        this.web3 = null;
        this.account = null;
        this.running = false;
        this.components = {};
        logger.info("Starting 0xBuilder...");
    }

    async initialize() {
        try {
            await this._initialize_components();
            logger.info("0xBuilder initialized successfully.");
        } catch (e) {
            logger.critical(`Error during initialization: ${e}`);
            throw e;
        }
    }

    async _initialize_components() {
        try {
            // 1. Initialize configuration and load ABIs
            await this._load_configuration();
            await this.configuration.abi_registry.initialize();
            
            // Load and validate ERC20 ABI
            const erc20_abi = await this._load_abi(this.configuration.ERC20_ABI);
            if (!erc20_abi) {
                throw new Error("Failed to load ERC20 ABI");
            }
            
            this.web3 = await this._initialize_web3();
            if (!this.web3) {
                throw new Error("Failed to initialize Web3 connection");
            }

            // Set up the account
            this.account = new Account(this.configuration.WALLET_KEY);
            await this._check_account_balance();

            // Initialize the components one by one
            this.components['api_config'] = new API_Config(this.configuration);
            await this.components['api_config'].initialize();

            this.components['nonce_core'] = new Nonce_Core(this.web3, this.account.address, this.configuration);
            await this.components['nonce_core'].initialize();

            this.components['safety_net'] = new Safety_Net(
                this.web3, this.configuration, this.account, this.components['api_config']
            );
            await this.components['safety_net'].initialize();

            this.components['transaction_core'] = new Transaction_Core(
                this.web3,
                this.account,
                this.configuration.AAVE_FLASHLOAN_ADDRESS,
                this.configuration.AAVE_FLASHLOAN_ABI,
                this.configuration.AAVE_POOL_ADDRESS,
                this.configuration.AAVE_POOL_ABI,
                this.components['api_config'],
                this.components['nonce_core'],
                this.components['safety_net'],
                this.configuration
            );
            await this.components['transaction_core'].initialize();

            this.components['market_monitor'] = new Market_Monitor(
                this.web3,
                this.configuration,
                this.components['api_config'],
                this.components['transaction_core']
            );
            await this.components['market_monitor'].initialize();

            this.components['mempool_monitor'] = new Mempool_Monitor(
                this.web3,
                this.components['safety_net'],
                this.components['nonce_core'],
                this.components['api_config'],
                await this.configuration.get_token_addresses(),
                this.configuration,
                erc20_abi,
                this.components['market_monitor']
            );
            await this.components['mempool_monitor'].initialize();

            this.components['strategy_net'] = new Strategy_Net(
                this.components['transaction_core'],
                this.components['market_monitor'],
                this.components['safety_net'],
                this.components['api_config']
            );
            await this.components['strategy_net'].initialize();

            logger.info("All components initialized successfully.");
        } catch (e) {
            logger.critical(`Component initialization failed: ${e}`);
            throw e;
        }
    }

    async run() {
        logger.debug("Starting main execution loop...");
        this.running = true;

        try {
            if (!this.components['mempool_monitor']) {
                throw new Error("Mempool monitor not properly initialized");
            }

            const initial_snapshot = tracemalloc.take_snapshot();
            let last_memory_check = Date.now();
            const MEMORY_CHECK_INTERVAL = 300;

            // Create task groups for different operations
            await Promise.all([
                this._start_monitoring(),
                this._process_profitable_transactions(),
                this._monitor_memory(initial_snapshot)
            ]);

        } catch (e) {
            logger.error(`Fatal error in execution loop: ${e}`);
            await this.stop();
        }
    }

    async stop() {
        logger.warning("Shutting down 0xBuilder...");
        this.running = false;

        try {
            const shutdown_order = [
                'mempool_monitor',  // Stop monitoring first
                'strategy_net',     // Stop strategies
                'transaction_core', // Stop transactions
                'market_monitor',   // Stop market monitoring
                'safety_net',       // Stop safety checks
                'nonce_core',       // Stop nonce management
                'api_config'        // Stop API connections last
            ];

            // Stop all components in parallel
            const stop_tasks = shutdown_order.map(component_name => {
                const component = this.components[component_name];
                return component && component.stop ? component.stop() : Promise.resolve();
            });

            await Promise.all(stop_tasks);

            // Clean up Web3 connection
            if (this.web3 && this.web3.provider && this.web3.provider.disconnect) {
                await this.web3.provider.disconnect();
            }

            // Final memory snapshot
            const final_snapshot = tracemalloc.take_snapshot();
            const top_stats = final_snapshot.compare_to(initial_snapshot, 'lineno');
            
            logger.debug("Final memory allocation changes:");
            for (const stat of top_stats.slice(0, 5)) {
                logger.debug(stat);
            }

            logger.debug("Core shutdown complete.");
        } catch (e) {
            logger.error(`Error during shutdown: ${e}`);
        } finally {
            tracemalloc.stop();
        }
    }

    async _start_monitoring() {
        // Start the mempool monitoring task
        await this.components['mempool_monitor'].start_monitoring();
    }

    async _process_profitable_transactions() {
        // Processing profitable transactions
        while (this.running) {
            try {
                const tx = await this.components['mempool_monitor'].get_profitable_transaction();
                logger.debug(`Processing profitable transaction: ${tx.tx_hash}`);
                await this.components['strategy_net'].execute_best_strategy(tx);
            } catch (e) {
                logger.error(`Error processing profitable transaction: ${e}`);
            }
        }
    }

    async _monitor_memory(initial_snapshot) {
        // Monitoring memory usage
        while (this.running) {
            const current_snapshot = tracemalloc.take_snapshot();
            const top_stats = current_snapshot.compare_to(initial_snapshot, 'lineno');
            
            logger.debug("Memory allocation changes:");
            for (const stat of top_stats.slice(0, 3)) {
                logger.debug(stat);
            }

            await new Promise(resolve => setTimeout(resolve, 300000));  // Sleep for 5 minutes
        }
    }

    async _load_configuration() {
        // Loading configuration settings
        await this.configuration.load();
        logger.debug("Configuration loaded.");
    }

    async _initialize_web3() {
        // Initialize Web3 connection
        const providers = await this._get_providers();

        if (providers.length === 0) {
            logger.error("No valid providers found.");
            return null;
        }

        // Try each provider and connect
        for (const [provider_name, provider] of providers) {
            try {
                logger.debug(`Trying to connect to ${provider_name}...`);
                const web3 = new Web3(provider);
                await web3.eth.net.isListening();
                logger.debug(`Connected to ${provider_name}`);
                return web3;
            } catch (e) {
                logger.error(`Error connecting to ${provider_name}: ${e}`);
            }
        }

        logger.error("Failed to connect to any provider.");
        return null;
    }

    async _get_providers() {
        // Get the available Ethereum providers
        const providers = [];

        if (this.configuration.HTTP_ENDPOINT) {
            try {
                const http_provider = new Web3.providers.HttpProvider(this.configuration.HTTP_ENDPOINT);
                providers.push(['HTTP', http_provider]);
                logger.debug("HTTP provider added.");
            } catch (e) {
                logger.error(`Error with HTTP provider: ${e}`);
            }
        }

        if (this.configuration.WEBSOCKET_ENDPOINT) {
            try {
                const ws_provider = new Web3.providers.WebsocketProvider(this.configuration.WEBSOCKET_ENDPOINT);
                providers.push(['WebSocket', ws_provider]);
                logger.debug("WebSocket provider added.");
            } catch (e) {
                logger.error(`Error with WebSocket provider: ${e}`);
            }
        }

        return providers;
    }

    async _check_account_balance() {
        // Check if the account has sufficient balance
        const balance = await this.web3.eth.getBalance(this.account.address);
        const balanceInEth = Web3.utils.fromWei(balance, 'ether');
        
        logger.debug(`Account balance: ${balanceInEth} ETH`);
        
        if (parseFloat(balanceInEth) < 0.01) {
            logger.warning("Account balance is low.");
        }
    }

    // Other helper methods for ABI loading, transaction construction, etc. will be complete next commit -J
}

module.exports = Main_Core;
