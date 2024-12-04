// index.js

import Web3 from 'web3';
import Configuration from './Configuration.js';
import NonceCore from './NonceCore.js';
import APIConfig from './APIConfig.js';
import SafetyNet from './SafetyNet.js';
import MempoolMonitor from './MempoolMonitor.js';
import TransactionCore from './TransactionCore.js';
import MarketMonitor from './MarketMonitor.js';
import StrategyNet from './StrategyNet.js';
import logger from './Logger.js'; //  Logger.js handles logging

async function main() {
    try {
        // Initialize Configuration
        const configuration = new Configuration();
        await configuration.load();

        // Initialize Web3
        const web3 = new Web3(new Web3.providers.HttpProvider(configuration.HTTP_ENDPOINT));

        // Initialize API Config
        const apiConfig = new APIConfig(configuration);
        await apiConfig.initialize();

        // Initialize Safety Net
        const safetyNet = new SafetyNet(web3, configuration, configuration.WALLET_ADDRESS, null, apiConfig);
        await safetyNet.initialize();

        // Initialize Nonce Core
        const nonceCore = new NonceCore(web3, configuration.WALLET_ADDRESS, configuration);
        await nonceCore.initialize();

        // Initialize Transaction Core
        const transactionCore = new TransactionCore(
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
        const marketMonitor = new MarketMonitor(web3, configuration, apiConfig);
        await marketMonitor.start_periodic_training("ETH"); //  for ETH

        // Initialize Strategy Net
        const strategyNet = new StrategyNet(transactionCore, marketMonitor, safetyNet, apiConfig, configuration);

        // : Start monitoring and executing strategies
        // This part would typically involve event listeners or periodic checks
        // For brevity, it's left as an exercise

    } catch (error) {
        logger.error(`Application failed to start: ${error.message}`);
        process.exit(1);
    }
}

main();
