// index.js

import Web3 from 'web3';
import Configuration from './Configuration.js';
import Nonce_Core from './Nonce_Core.js';
import API_Config from './API_Config.js';
import Safety_Net from './Safety_Net.js';
import Mempool_Monitor from './Mempool_Monitor.js';
import Transaction_Core from './Transaction_Core.js';
import Market_Monitor from './Market_Monitor.js';
import Strategy_Net from './Strategy_Net.js';
import logger from './Logger.js'; // Assuming Logger.js handles logging

async function main() {
    try {
        // Initialize Configuration
        const configuration = new Configuration();
        await configuration.load();

        // Initialize Web3
        const web3 = new Web3(new Web3.providers.HttpProvider(configuration.HTTP_ENDPOINT));

        // Initialize API Config
        const apiConfig = new API_Config(configuration);
        await apiConfig.initialize();

        // Initialize Safety Net
        const safetyNet = new Safety_Net(web3, configuration, configuration.WALLET_ADDRESS, null, apiConfig);
        await safetyNet.initialize();

        // Initialize Nonce Core
        const nonceCore = new Nonce_Core(web3, configuration.WALLET_ADDRESS, configuration);
        await nonceCore.initialize();

        // Initialize Transaction Core
        const transactionCore = new Transaction_Core(
            web3,
            configuration.account,
            configuration.AAVE_FLASHLOAN_ADDRESS,
            configuration.AAVE_FLASHLOAN_ABI,
            configuration.AAVE_LENDING_POOL_ADDRESS,
            configuration.AAVE_LENDING_POOL_ABI,
            apiConfig,
            null,
            nonceCore,
            safetyNet,
            configuration
        );
        await transactionCore.initialize();

        // Initialize Market Monitor
        const marketMonitor = new Market_Monitor(web3, configuration, apiConfig);
        await marketMonitor.start_periodic_training("ETH"); // Example for ETH

        // Initialize Strategy Net
        const strategyNet = new Strategy_Net(transactionCore, marketMonitor, safetyNet, apiConfig, configuration);

        // Example: Start monitoring and executing strategies
        // This part would typically involve event listeners or periodic checks
        // For brevity, it's left as an exercise

    } catch (error) {
        logger.error(`Application failed to start: ${error.message}`);
        process.exit(1);
    }
}

main();
