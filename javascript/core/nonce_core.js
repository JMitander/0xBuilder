// NonceCore.js
import NodeCache from 'node-cache';
import logger from './logger.js';

class NonceCore {
    constructor(web3, address, configuration) {
        this.web3 = web3; // Web3 instance for blockchain interactions
        this.address = address; // Ethereum address
        this.configuration = configuration; // Configuration settings
        this.pendingTransactions = new Set(); // Set of nonces for pending transactions
        this.nonceCache = new NodeCache({ stdTTL: 300, checkperiod: 60 }); // Cache with TTL of 5 minutes
        this.lastSync = Date.now(); // Timestamp of the last nonce synchronization
        this.initialized = false; // Flag indicating if the nonce manager is initialized
    }

    async initialize() {
        try {
            if (!this.initialized) {
                await this._initNonce();
                this.initialized = true;
                logger.debug(`NonceCore initialized for ${this.address.substring(0, 10)}...`);
            }
        } catch (e) {
            logger.error(`Initialization failed: ${e.message}`);
            throw new Error("NonceCore initialization failed");
        }
    }

    async _initNonce() {
        const currentNonce = await this._fetchCurrentNonceWithRetries();
        const pendingNonce = await this._getPendingNonce();
        const maxNonce = Math.max(currentNonce, pendingNonce);
        this.nonceCache.set(this.address, maxNonce);
        this.lastSync = Date.now();
    }

    async getNonce(forceRefresh = false) {
        if (!this.initialized) {
            await this.initialize();
        }

        if (forceRefresh || this._shouldRefreshCache()) {
            await this.refreshNonce();
        }

        let currentNonce = this.nonceCache.get(this.address) || 0;
        const nextNonce = currentNonce;
        this.nonceCache.set(this.address, currentNonce + 1); // Increment nonce
        logger.debug(`Allocated nonce ${nextNonce} for ${this.address.substring(0, 10)}...`);
        return nextNonce;
    }

    async refreshNonce() {
        try {
            const chainNonce = await this._fetchCurrentNonceWithRetries();
            const cachedNonce = this.nonceCache.get(this.address) || 0;
            const pendingNonce = await this._getPendingNonce();
            const newNonce = Math.max(chainNonce, cachedNonce, pendingNonce);
            this.nonceCache.set(this.address, newNonce);
            this.lastSync = Date.now();
            logger.debug(`Nonce refreshed to ${newNonce}`);
        } catch (e) {
            logger.error(`Nonce refresh failed: ${e.message}`);
            throw e;
        }
    }

    async _fetchCurrentNonceWithRetries() {
        const MAX_RETRIES = 3;
        const RETRY_DELAY = 1000; // in ms
        let backoff = RETRY_DELAY;

        for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
            try {
                const nonce = await this.web3.eth.getTransactionCount(this.address, 'pending');
                return nonce;
            } catch (e) {
                if (attempt === MAX_RETRIES - 1) {
                    logger.error(`Nonce fetch failed after retries: ${e.message}`);
                    throw e;
                }
                logger.warn(`Nonce fetch attempt ${attempt + 1} failed: ${e.message}. Retrying in ${backoff}ms...`);
                await new Promise(resolve => setTimeout(resolve, backoff));
                backoff *= 2; // Exponential backoff
            }
        }
    }

    async _getPendingNonce() {
        try {
            const pendingNounces = Array.from(this.pendingTransactions);
            return pendingNounces.length > 0 ? Math.max(...pendingNounces) + 1 : 0;
        } catch (e) {
            logger.error(`Error getting pending nonce: ${e.message}`);
            return 0;
        }
    }

    async trackTransaction(txHash, nonce) {
        this.pendingTransactions.add(nonce);
        try {
            // Wait for the transaction to be mined with a timeout of 120 seconds
            const receipt = await this.web3.eth.waitForTransactionReceipt(txHash, 120000);
            this.pendingTransactions.delete(nonce);
        } catch (e) {
            logger.error(`Transaction tracking failed: ${e.message}`);
            this.pendingTransactions.delete(nonce);
        }
    }

    async _handleNonceError() {
        try {
            await this.syncNonceWithChain();
        } catch (e) {
            logger.error(`Nonce error recovery failed: ${e.message}`);
            throw e;
        }
    }

    async syncNonceWithChain() {
        try {
            const newNonce = await this._fetchCurrentNonceWithRetries();
            this.nonceCache.set(this.address, newNonce);
            this.lastSync = Date.now();
            this.pendingTransactions.clear();
            logger.debug(`Nonce synchronized to ${newNonce}`);
        } catch (e) {
            logger.error(`Nonce synchronization failed: ${e.message}`);
            throw e;
        }
    }

    _shouldRefreshCache() {
        const CACHE_TTL = 300; // 5 minutes
        return (Date.now() - this.lastSync) > (CACHE_TTL / 2) * 1000; // Half TTL
    }

    async reset() {
        try {
            this.nonceCache.del(this.address);
            this.pendingTransactions.clear();
            this.lastSync = Date.now();
            this.initialized = false;
            await this.initialize();
            logger.debug("NonceCore reset complete");
        } catch (e) {
            logger.error(`Reset failed: ${e.message}`);
            throw e;
        }
    }

    async stop() {
        try {
            await this.reset();
            logger.debug("NonceCore stopped successfully.");
        } catch (e) {
            logger.error(`Error stopping nonce core: ${e.message}`);
            throw e;
        }
    }
}

export default NonceCore;
