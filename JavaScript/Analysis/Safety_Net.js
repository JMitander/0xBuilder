// SafetyNet.js
import NodeCache from 'node-cache';
import logger from './logger.js';
import Decimal from 'decimal.js';

class SafetyNet {
    constructor(web3, configuration = null, address = null, account = null, apiconfig = null) {
        this.web3 = web3; // Web3 instance for blockchain interactions
        this.address = address; // Ethereum address
        this.configuration = configuration; // Configuration settings
        this.account = account; // Ethereum account instance
        this.apiconfig = apiconfig; // API configuration for data retrieval
        this.priceCache = new NodeCache({ stdTTL: 300, checkperiod: 60 }); // Cache for various price data
        this.gasPriceCache = new NodeCache({ stdTTL: 15, checkperiod: 5 }); // Cache for gas prices

        this.priceLock = Promise.resolve(); // Simple lock implementation
        logger.debug("Safety Net initialized with enhanced configuration");
    }

    async getBalance(account) {
        const cacheKey = `balance_${account.address}`;
        if (this.priceCache.has(cacheKey)) {
            return this.priceCache.get(cacheKey);
        }

        for (let attempt = 0; attempt < 3; attempt++) {
            try {
                const balanceWei = await this.web3.eth.getBalance(account.address);
                const balanceEth = new Decimal(this.web3.utils.fromWei(balanceWei, 'ether'));
                this.priceCache.set(cacheKey, balanceEth);
                logger.debug(`Balance for ${account.address.substring(0, 10)}...: ${balanceEth.toFixed(4)} ETH`);
                return balanceEth;
            } catch (e) {
                if (attempt === 2) {
                    logger.error(`Failed to get balance after 3 attempts: ${e.message}`);
                    return new Decimal(0);
                }
                await new Promise(res => setTimeout(res, 1000 * (attempt + 1))); // Exponential backoff
            }
        }
    }

    async ensureProfit(transactionData, minimumProfitEth = null) {
        try {
            if (minimumProfitEth === null) {
                const accountBalance = await this.getBalance(this.account);
                // Set minimum profit based on account balance to ensure sustainability
                minimumProfitEth = accountBalance.lessThan(new Decimal("0.5")) ? 0.003 : 0.01;
            }

            const gasPriceGwei = await this.getDynamicGasPrice();
            const gasUsed = await this.estimateGas(transactionData);

            if (!this._validateGasParameters(gasPriceGwei, gasUsed)) {
                return false;
            }

            const gasCostEth = this._calculateGasCost(gasPriceGwei, gasUsed);
            const slippage = await this.adjustSlippageTolerance();

            const outputToken = transactionData.output_token;
            const realTimePrice = await this.apiconfig.getRealTimePrice(outputToken);
            if (!realTimePrice) {
                return false;
            }

            const profit = await this._calculateProfit(transactionData, realTimePrice, slippage, gasCostEth);

            this._logProfitCalculation(transactionData, realTimePrice, gasCostEth, profit, minimumProfitEth);

            return profit.greaterThan(new Decimal(minimumProfitEth));
        } catch (e) {
            logger.error(`Error in profit calculation: ${e.message}`);
            return false;
        }
    }

    _validateGasParameters(gasPriceGwei, gasUsed) {
        const GAS_CONFIG = {
            "max_gas_price_gwei": 500, // Maximum allowable gas price in Gwei
            "min_profit_multiplier": 2.0, // Minimum profit multiplier to consider a transaction
            "base_gas_limit": 21000, // Base gas limit for standard transactions
        };
        if (gasUsed === 0) {
            logger.error("Gas estimation returned zero");
            return false;
        }
        if (gasPriceGwei.greaterThan(GAS_CONFIG.max_gas_price_gwei)) {
            logger.warn(`Gas price ${gasPriceGwei.toFixed()} gwei exceeds maximum threshold`);
            return false;
        }
        return true;
    }

    _calculateGasCost(gasPriceGwei, gasUsed) {
        return gasPriceGwei.mul(new Decimal(gasUsed)).mul(new Decimal("1e-9")); // Convert Gwei to ETH
    }

    async _calculateProfit(transactionData, realTimePrice, slippage, gasCostEth) {
        const expectedOutput = realTimePrice.mul(new Decimal(transactionData.amountOut));
        const inputAmount = new Decimal(transactionData.amountIn);
        const slippageAdjustedOutput = expectedOutput.mul(new Decimal(1 - slippage));
        return slippageAdjustedOutput.sub(inputAmount).sub(gasCostEth);
    }

    _logProfitCalculation(transactionData, realTimePrice, gasCostEth, profit, minimumProfitEth) {
        const profitable = profit.greaterThan(new Decimal(minimumProfitEth)) ? "Yes" : "No";
        logger.debug(
            `Profit Calculation Summary:\n` +
            `Token: ${transactionData.output_token}\n` +
            `Real-time Price: ${realTimePrice.toFixed(6)} ETH\n` +
            `Input Amount: ${transactionData.amountIn.toFixed(6)} ETH\n` +
            `Expected Output: ${transactionData.amountOut.toFixed(6)} tokens\n` +
            `Gas Cost: ${gasCostEth.toFixed(6)} ETH\n` +
            `Calculated Profit: ${profit.toFixed(6)} ETH\n` +
            `Minimum Required: ${minimumProfitEth} ETH\n` +
            `Profitable: ${profitable}`
        );
    }

    async getDynamicGasPrice() {
        if (this.gasPriceCache.has("gas_price")) {
            return this.gasPriceCache.get("gas_price");
        }

        try {
            let gasPrice = await this.web3.eth.getGasPrice();
            if (!gasPrice) {
                gasPrice = await this.web3.eth.getGasPrice(); // Fallback
            }
            const gasPriceGwei = new Decimal(this.web3.utils.fromWei(gasPrice, 'gwei'));
            this.gasPriceCache.set("gas_price", gasPriceGwei);
            return gasPriceGwei;
        } catch (e) {
            logger.error(`Error fetching dynamic gas price: ${e.message}`);
            return new Decimal(0);
        }
    }

    async estimateGas(transactionData) {
        try {
            const gasEstimate = await this.web3.eth.estimateGas(transactionData);
            return gasEstimate;
        } catch (e) {
            logger.error(`Gas estimation failed: ${e.message}`);
            return 0;
        }
    }

    async adjustSlippageTolerance() {
        try {
            const congestionLevel = await this.getNetworkCongestion();
            let slippage = 0.1; // Default slippage tolerance (10%)
            const SLIPPAGE_CONFIG = {
                "default": 0.1, // Default slippage tolerance (10%)
                "min": 0.01, // Minimum slippage tolerance (1%)
                "max": 0.5, // Maximum slippage tolerance (50%)
                "high_congestion": 0.05, // Slippage during high network congestion (5%)
                "low_congestion": 0.2, // Slippage during low network congestion (20%)
            };
            if (congestionLevel > 0.8) {
                slippage = SLIPPAGE_CONFIG.high_congestion;
            } else if (congestionLevel < 0.2) {
                slippage = SLIPPAGE_CONFIG.low_congestion;
            } else {
                slippage = SLIPPAGE_CONFIG.default;
            }
            // Ensure slippage is within defined min and max bounds
            slippage = Math.min(Math.max(slippage, SLIPPAGE_CONFIG.min), SLIPPAGE_CONFIG.max);
            logger.debug(`Adjusted slippage tolerance to ${slippage * 100}%`);
            return slippage;
        } catch (e) {
            logger.error(`Error adjusting slippage tolerance: ${e.message}`);
            return 0.1; // Default slippage
        }
    }

    async getNetworkCongestion() {
        try {
            const latestBlock = await this.web3.eth.getBlock('latest');
            const gasUsed = latestBlock.gasUsed;
            const gasLimit = latestBlock.gasLimit;
            const congestionLevel = gasUsed / gasLimit;
            logger.debug(`Network congestion level: ${congestionLevel * 100}%`);
            return congestionLevel;
        } catch (e) {
            logger.error(`Error fetching network congestion: ${e.message}`);
            return 0.5; //  medium congestion if unknown
        }
    }

    async _calculateProfit(transactionData, realTimePrice, slippage, gasCostEth) {
        const expectedOutput = realTimePrice.mul(new Decimal(transactionData.amountOut));
        const inputAmount = new Decimal(transactionData.amountIn);
        const slippageAdjustedOutput = expectedOutput.mul(new Decimal(1 - slippage));
        return slippageAdjustedOutput.sub(inputAmount).sub(gasCostEth);
    }

    async stop() {
        try {
            await this.apiconfig.close();
            logger.debug("Safety Net stopped successfully.");
        } catch (e) {
            logger.error(`Error stopping Safety Net: ${e.message}`);
            throw e;
        }
    }
}

export default SafetyNet;
