// MempoolMonitor.js
import logger from './logger.js';
import { Semaphore } from 'async-mutex';
import Decimal from 'decimal.js';

class MempoolMonitor {
    constructor(web3, safetyNet, nonceCore, apiConfig, monitoredTokens = [], erc20Abi = [], configuration = null) {
        this.web3 = web3;
        this.configuration = configuration;
        this.safetyNet = safetyNet;
        this.nonceCore = nonceCore;
        this.apiConfig = apiConfig;

        this.running = false;
        this.pendingTransactions = new Set();
        this.monitoredTokens = new Set(monitoredTokens);
        this.profitableTransactions = [];
        this.processedTransactions = new Set();

        this.erc20Abi = erc20Abi;
        this.minimumProfitThreshold = new Decimal("0.001");
        this.maxParallelTasks = 50;
        this.retryAttempts = 3;
        this.backoffFactor = 1.5;

        this.semaphore = new Semaphore(this.maxParallelTasks);
        this.taskQueue = [];

        logger.debug("MempoolMonitor initialized with enhanced configuration.");
    }

    async startMonitoring() {
        if (this.running) {
            logger.debug("Monitoring is already active.");
            return;
        }

        this.running = true;
        try {
            this.subscription = this.web3.eth.subscribe('pendingTransactions', (error, result) => {
                if (error) {
                    logger.error(`Subscription error: ${error.message}`);
                }
            })
            .on('data', async (txHash) => {
                if (!this.processedTransactions.has(txHash)) {
                    this.processedTransactions.add(txHash);
                    this.taskQueue.push(txHash);
                    this._processTaskQueue();
                }
            });

            logger.debug("Mempool monitoring started.");
        } catch (e) {
            logger.error(`Failed to start monitoring: ${e.message}`);
            throw e;
        }
    }

    async stopMonitoring() {
        if (!this.running) return;

        this.running = false;
        try {
            if (this.subscription) {
                this.subscription.unsubscribe((error, success) => {
                    if (success) {
                        logger.debug("Successfully unsubscribed from pending transactions.");
                    }
                });
            }

            // Wait for the task queue to be empty
            while (this.taskQueue.length > 0) {
                await new Promise(res => setTimeout(res, 100));
            }

            logger.debug("Mempool monitoring stopped gracefully.");
            process.exit(0);
        } catch (e) {
            logger.error(`Error during monitoring shutdown: ${e.message}`);
        }
    }

    async _processTaskQueue() {
        while (this.taskQueue.length > 0 && this.running) {
            const txHash = this.taskQueue.shift();
            this.semaphore.acquire().then(async (release) => {
                try {
                    await this.processTransaction(txHash);
                } catch (e) {
                    logger.debug(`Error processing transaction ${txHash}: ${e.message}`);
                } finally {
                    release();
                }
            });
        }
    }

    async processTransaction(txHash) {
        try {
            const tx = await this._getTransactionWithRetry(txHash);
            if (!tx) return;

            const analysis = await this.analyzeTransaction(tx);
            if (analysis.is_profitable) {
                await this._handleProfitableTransaction(analysis);
            }
        } catch (e) {
            logger.debug(`Error processing transaction ${txHash}: ${e.message}`);
        }
    }

    async _getTransactionWithRetry(txHash) {
        for (let attempt = 0; attempt < this.retryAttempts; attempt++) {
            try {
                const tx = await this.web3.eth.getTransaction(txHash);
                return tx;
            } catch (e) {
                if (e.message.includes('not found') && attempt < this.retryAttempts - 1) {
                    await new Promise(res => setTimeout(res, Math.pow(this.backoffFactor, attempt) * 1000));
                } else {
                    logger.error(`Error fetching transaction ${txHash}: ${e.message}`);
                    return null;
                }
            }
        }
        return null;
    }

    async _handleProfitableTransaction(analysis) {
        try {
            this.profitableTransactions.push(analysis);
            logger.debug(`Profitable transaction identified: ${analysis.tx_hash} (Estimated profit: ${analysis.profit.toFixed()} ETH)`);
        } catch (e) {
            logger.debug(`Error handling profitable transaction: ${e.message}`);
        }
    }

    async analyzeTransaction(tx) {
        if (!tx.hash || !tx.input) {
            logger.debug(`Transaction ${tx.hash} is missing essential fields. Skipping.`);
            return { is_profitable: false };
        }

        try {
            if (tx.value > 0) {
                return await this._analyzeEthTransaction(tx);
            }
            return await this._analyzeTokenTransaction(tx);
        } catch (e) {
            logger.error(`Error analyzing transaction ${tx.hash}: ${e.message}`);
            return { is_profitable: false };
        }
    }

    async _analyzeEthTransaction(tx) {
        try {
            const isProfitable = await this._isProfitableEthTransaction(tx);
            if (isProfitable) {
                await this._logTransactionDetails(tx, true);
                return {
                    is_profitable: true,
                    tx_hash: tx.hash,
                    value: tx.value,
                    to: tx.to,
                    from: tx.from,
                    input: tx.input,
                    gasPrice: tx.gasPrice,
                };
            }
            return { is_profitable: false };
        } catch (e) {
            logger.error(`Error analyzing ETH transaction ${tx.hash}: ${e.message}`);
            return { is_profitable: false };
        }
    }

    async _analyzeTokenTransaction(tx) {
        try {
            if (this.erc20Abi.length === 0) {
                logger.warn("ERC20 ABI not loaded. Cannot analyze token transaction.");
                return { is_profitable: false };
            }

            const contract = new this.web3.eth.Contract(this.erc20Abi, tx.to);
            const decodedData = await this._decodeFunctionInput(tx.input, contract);
            if (!decodedData) {
                logger.debug(`Failed to decode function input for transaction ${tx.hash}. Skipping.`);
                return { is_profitable: false };
            }

            const { functionName, params } = decodedData;

            if (this.configuration.ERC20_SIGNATURES[functionName]) {
                const estimatedProfit = await this._estimateProfit(tx, params);
                if (estimatedProfit.greaterThan(this.minimumProfitThreshold)) {
                    logger.debug(`Identified profitable transaction ${tx.hash} with estimated profit: ${estimatedProfit.toFixed()} ETH`);
                    await this._logTransactionDetails(tx);
                    return {
                        is_profitable: true,
                        profit: estimatedProfit,
                        function_name: functionName,
                        params: params,
                        tx_hash: tx.hash,
                        to: tx.to,
                        input: tx.input,
                        value: tx.value,
                        gasPrice: tx.gasPrice,
                    };
                } else {
                    logger.debug(`Transaction ${tx.hash} is below threshold. Skipping...`);
                    return { is_profitable: false };
                }
            } else {
                logger.debug(`Function ${functionName} not in ERC20_SIGNATURES. Skipping.`);
                return { is_profitable: false };
            }
        } catch (e) {
            logger.debug(`Error decoding function input for transaction ${tx.hash}: ${e.message}`);
            return { is_profitable: false };
        }
    }

    async _decodeFunctionInput(input, contract) {
        try {
            const functionSignature = input.slice(0, 10); // First 4 bytes + '0x'
            const functionAbi = contract.options.jsonInterface.find(fn => {
                return this.web3.eth.abi.encodeFunctionSignature(fn) === functionSignature;
            });

            if (!functionAbi) {
                logger.debug(`Function ABI not found for signature ${functionSignature}`);
                return null;
            }

            const decoded = this.web3.eth.abi.decodeParameters(functionAbi.inputs, input.slice(10));
            const params = {};
            functionAbi.inputs.forEach((inputParam, index) => {
                params[inputParam.name] = decoded[index];
            });

            return { functionName: functionAbi.name, params };
        } catch (e) {
            logger.debug(`Error decoding function input: ${e.message}`);
            return null;
        }
    }

    async _isProfitableEthTransaction(tx) {
        try {
            const potentialProfit = await this.safetyNet._calculateProfit(tx, null, null, null); // Adjust as needed
            return potentialProfit.greaterThan(this.minimumProfitThreshold);
        } catch (e) {
            logger.debug(`Error estimating ETH transaction profit for transaction ${tx.hash}: ${e.message}`);
            return false;
        }
    }

    async _estimateEthTransactionProfit(tx) {
        try {
            const gasPriceGwei = await this.safetyNet.getDynamicGasPrice();
            const gasUsed = tx.gas || await this.web3.eth.estimateGas(tx);
            const gasCostEth = gasPriceGwei.mul(new Decimal(gasUsed)).mul(new Decimal("1e-9"));
            const ethValue = new Decimal(this.web3.utils.fromWei(tx.value, 'ether'));
            const potentialProfit = ethValue.sub(gasCostEth);
            return potentialProfit.gt(0) ? potentialProfit : new Decimal(0);
        } catch (e) {
            logger.error(`Error estimating ETH transaction profit: ${e.message}`);
            return new Decimal(0);
        }
    }

    async _estimateProfit(tx, params) {
        try {
            const gasPriceGwei = new Decimal(this.web3.utils.fromWei(tx.gasPrice, 'gwei'));
            const gasUsed = tx.gas || await this.web3.eth.estimateGas(tx);
            const gasCostEth = gasPriceGwei.mul(new Decimal(gasUsed)).mul(new Decimal("1e-9"));
            const inputAmountWei = new Decimal(params.amountIn || 0);
            const outputAmountMinWei = new Decimal(params.amountOutMin || 0);
            const path = params.path || [];
            if (path.length < 2) {
                logger.debug(`Transaction ${tx.hash} has an invalid path for swapping. Skipping.`);
                return new Decimal(0);
            }
            const outputTokenAddress = path[path.length - 1];
            const outputTokenSymbol = await this.apiConfig.getTokenSymbol(this.web3, outputTokenAddress);
            if (!outputTokenSymbol) {
                logger.debug(`Output token symbol not found for address ${outputTokenAddress}. Skipping.`);
                return new Decimal(0);
            }
            const marketPrice = await this.apiConfig.getRealTimePrice(outputTokenSymbol.toLowerCase());
            if (!marketPrice || marketPrice.equals(0)) {
                logger.debug(`Market price not available for token ${outputTokenSymbol}. Skipping.`);
                return new Decimal(0);
            }
            const inputAmountEth = new Decimal(this.web3.utils.fromWei(inputAmountWei, 'ether'));
            const outputAmountEth = new Decimal(this.web3.utils.fromWei(outputAmountMinWei, 'ether'));
            const expectedOutputValue = outputAmountEth.mul(marketPrice);
            const profit = expectedOutputValue.sub(inputAmountEth).sub(gasCostEth);
            return profit.gt(0) ? profit : new Decimal(0);
        } catch (e) {
            logger.debug(`Error estimating profit for transaction ${tx.hash}: ${e.message}`);
            return new Decimal(0);
        }
    }

    async _logTransactionDetails(tx, isEth = false) {
        if (isEth) {
            // Log ETH transaction details
            logger.debug(
                `ETH Transaction Details:\n` +
                `Hash: ${tx.hash}\n` +
                `From: ${tx.from}\n` +
                `To: ${tx.to}\n` +
                `Value: ${this.web3.utils.fromWei(tx.value, 'ether')} ETH\n` +
                `Gas Price: ${this.web3.utils.fromWei(tx.gasPrice, 'gwei')} Gwei\n` +
                `Input: ${tx.input}`
            );
        } else {
            // Log token transaction details
            logger.debug(
                `Token Transaction Details:\n` +
                `Hash: ${tx.hash}\n` +
                `From: ${tx.from}\n` +
                `To: ${tx.to}\n` +
                `Value: ${this.web3.utils.fromWei(tx.value, 'ether')} ETH\n` +
                `Gas Price: ${this.web3.utils.fromWei(tx.gasPrice, 'gwei')} Gwei\n` +
                `Input: ${tx.input}`
            );
        }
    }
}

export default MempoolMonitor;
