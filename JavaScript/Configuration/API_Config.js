// API_Config.js
import axios from 'axios';
import NodeCache from 'node-cache';
import logger from './logger.js';
import Decimal from 'decimal.js';
import fs from 'fs/promises';

class API_Config {
    constructor(configuration = null) {
        this.configuration = configuration; // Configuration settings
        this.session = axios.create();
        this.priceCache = new NodeCache({ stdTTL: 300, checkperiod: 60 }); // Real-time prices cache
        this.tokenSymbolCache = new NodeCache({ stdTTL: 86400, checkperiod: 3600 }); // Token symbols cache

        // API configuration detailing different API providers and their settings
        this.apiConfigs = {
            "binance": {
                "base_url": "https://api.binance.com/api/v3",
                "success_rate": 1.0,
                "weight": 1.0,
                "rate_limit": 1200, // Max requests per minute
            },
            "coingecko": {
                "base_url": "https://api.coingecko.com/api/v3",
                "api_key": this.configuration.COINGECKO_API_KEY,
                "success_rate": 1.0,
                "weight": 0.8,
                "rate_limit": 50, // Max requests per minute
            },
            "coinmarketcap": {
                "base_url": "https://pro-api.coinmarketcap.com/v1",
                "api_key": this.configuration.COINMARKETCAP_API_KEY,
                "success_rate": 1.0,
                "weight": 0.7,
                "rate_limit": 333, // Max requests per minute
            },
            "cryptocompare": {
                "base_url": "https://min-api.cryptocompare.com/data",
                "api_key": this.configuration.CRYPTOCOMPARE_API_KEY,
                "success_rate": 1.0,
                "weight": 0.6,
                "rate_limit": 80, // Max requests per minute
            },
        };

        // Initialize rate limiters for each API provider based on their rate limits
        this.rateLimiters = {};
        for (let provider in this.apiConfigs) {
            const config = this.apiConfigs[provider];
            this.rateLimiters[provider] = {
                tokens: config.rate_limit,
                lastRefill: Date.now(),
                tokensPerInterval: config.rate_limit,
                interval: 60000, // 60 seconds
            };
        }
    }

    async getTokenSymbol(web3, tokenAddress) {
        if (this.tokenSymbolCache.has(tokenAddress)) {
            return this.tokenSymbolCache.get(tokenAddress);
        }

        if (this.configuration.TOKEN_SYMBOLS[tokenAddress]) {
            const symbol = this.configuration.TOKEN_SYMBOLS[tokenAddress];
            this.tokenSymbolCache.set(tokenAddress, symbol);
            return symbol;
        }

        try {
            const abiPath = this.configuration.ERC20_ABI;
            const content = await fs.readFile(abiPath, 'utf-8');
            const abi = JSON.parse(content);
            const contract = new web3.eth.Contract(abi, tokenAddress);
            const symbol = await contract.methods.symbol().call();
            this.tokenSymbolCache.set(tokenAddress, symbol);
            return symbol;
        } catch (e) {
            logger.error(`Error getting symbol for token ${tokenAddress}: ${e.message}`);
            return null;
        }
    }

    async getRealTimePrice(token, vsCurrency = "eth") {
        const cacheKey = `price_${token}_${vsCurrency}`;
        if (this.priceCache.has(cacheKey)) {
            return this.priceCache.get(cacheKey);
        }

        let prices = [];
        let weights = [];

        for (let source in this.apiConfigs) {
            try {
                const price = await this._fetchPrice(source, token, vsCurrency);
                if (price) {
                    const config = this.apiConfigs[source];
                    prices.push(price);
                    weights.push(config.weight * config.success_rate);
                }
            } catch (e) {
                logger.error(`Error fetching price from ${source}: ${e.message}`);
                this.apiConfigs[source].success_rate *= 0.9; // Penalize failed source
            }
        }

        if (prices.length === 0) {
            logger.warn(`No valid prices found for ${token}!`);
            return null;
        }

        // Calculate weighted average price
        const weightedSum = prices.reduce((acc, p, idx) => acc + p * weights[idx], 0);
        const totalWeight = weights.reduce((acc, w) => acc + w, 0);
        const weightedPrice = weightedSum / totalWeight;

        this.priceCache.set(cacheKey, new Decimal(weightedPrice));
        return this.priceCache.get(cacheKey);
    }

    async _fetchPrice(source, token, vsCurrency) {
        const config = this.apiConfigs[source];
        if (!config) {
            logger.debug(`API configuration for ${source} not found.`);
            return null;
        }

        let url = '';
        let params = {};
        let headers = {};

        if (source === "coingecko") {
            url = `${config.base_url}/simple/price`;
            params = { ids: token, vs_currencies: vsCurrency };
        } else if (source === "coinmarketcap") {
            url = `${config.base_url}/cryptocurrency/quotes/latest`;
            params = { symbol: token.toUpperCase(), convert: vsCurrency.toUpperCase() };
            headers = { 'X-CMC_PRO_API_KEY': config.api_key };
        } else if (source === "cryptocompare") {
            url = `${config.base_url}/price`;
            params = { fsym: token.toUpperCase(), tsyms: vsCurrency.toUpperCase(), api_key: config.api_key };
        } else if (source === "binance") {
            url = `${config.base_url}/ticker/price`;
            const symbol = `${token.toUpperCase()}${vsCurrency.toUpperCase()}`;
            params = { symbol: symbol };
        } else {
            logger.warn(`Unsupported price source: ${source}`);
            return null;
        }

        const response = await this.makeRequest(source, url, { params, headers });
        if (source === "coingecko") {
            return new Decimal(response[token][vsCurrency]);
        } else if (source === "coinmarketcap") {
            const data = response.data[token.toUpperCase()].quote[vsCurrency.toUpperCase()].price;
            return new Decimal(data);
        } else if (source === "cryptocompare") {
            return new Decimal(response[vsCurrency.toUpperCase()]);
        } else if (source === "binance") {
            return new Decimal(response.price);
        }

        return null;
    }

    async makeRequest(providerName, url, { params = {}, headers = {} } = {}, maxAttempts = 5, backoffFactor = 1.5) {
        await this._acquireRateLimit(providerName);

        for (let attempt = 0; attempt < maxAttempts; attempt++) {
            try {
                const response = await this.session.get(url, { params, headers, timeout: 10000 * (attempt + 1) });
                if (response.status === 429) {
                    const waitTime = Math.pow(backoffFactor, attempt) * 1000;
                    logger.warn(`Rate limit exceeded for ${providerName}, retrying in ${waitTime}ms...`);
                    await new Promise(res => setTimeout(res, waitTime));
                    continue;
                }
                response.status !== 200 && response.raiseForStatus();
                return response.data;
            } catch (e) {
                if (attempt === maxAttempts - 1) {
                    logger.error(`Request failed after ${maxAttempts} attempts: ${e.message}`);
                    throw e;
                }
                const waitTime = Math.pow(backoffFactor, attempt) * 1000;
                logger.warn(`Request attempt ${attempt + 1} failed: ${e.message}. Retrying in ${waitTime}ms...`);
                await new Promise(res => setTimeout(res, waitTime));
            }
        }
    }

    async _acquireRateLimit(providerName) {
        const limiter = this.rateLimiters[providerName];
        if (!limiter) return;

        const now = Date.now();
        if (now - limiter.lastRefill > limiter.interval) {
            limiter.tokens = limiter.tokensPerInterval;
            limiter.lastRefill = now;
        }

        if (limiter.tokens > 0) {
            limiter.tokens -= 1;
            return;
        }

        const waitTime = limiter.interval - (now - limiter.lastRefill);
        logger.warn(`Rate limit reached for ${providerName}. Waiting for ${waitTime}ms...`);
        await new Promise(res => setTimeout(res, waitTime));
        limiter.tokens = limiter.tokensPerInterval - 1;
        limiter.lastRefill = Date.now();
    }

    async fetchHistoricalPrices(token, days = 30) {
        const cacheKey = `historical_prices_${token}_${days}`;
        if (this.priceCache.has(cacheKey)) {
            logger.debug(`Returning cached historical prices for ${token}.`);
            return this.priceCache.get(cacheKey);
        }

        const prices = await this._fetchFromServices(
            (service) => this._fetchHistoricalPrices(service, token, days),
            `historical prices for ${token}`
        );

        if (prices) {
            this.priceCache.set(cacheKey, prices);
        }

        return prices || [];
    }

    async _fetchHistoricalPrices(source, token, days) {
        const config = this.apiConfigs[source];
        if (!config) {
            logger.debug(`API configuration for ${source} not found.`);
            return null;
        }

        if (source === "coingecko") {
            const url = `${config.base_url}/coins/${token}/market_chart`;
            const params = { vs_currency: 'usd', days: days };
            const response = await this.makeRequest(source, url, { params });
            return response.prices.map(price => price[1]);
        } else {
            logger.debug(`Unsupported historical price source: ${source}`);
            return null;
        }
    }

    async getTokenVolume(token) {
        const cacheKey = `token_volume_${token}`;
        if (this.priceCache.has(cacheKey)) {
            logger.debug(`Returning cached trading volume for ${token}.`);
            return this.priceCache.get(cacheKey);
        }

        const volume = await this._fetchFromServices(
            (service) => this._fetchTokenVolume(service, token),
            `trading volume for ${token}`
        );

        if (volume !== null) {
            this.priceCache.set(cacheKey, volume);
        }

        return volume || 0.0;
    }

    async _fetchTokenVolume(source, token) {
        const config = this.apiConfigs[source];
        if (!config) {
            logger.debug(`API configuration for ${source} not found.`);
            return null;
        }

        if (source === "coingecko") {
            const url = `${config.base_url}/coins/markets`;
            const params = { vs_currency: 'usd', ids: token };
            const response = await this.makeRequest(source, url, { params });
            return response.length > 0 ? response[0].total_volume : null;
        } else {
            logger.debug(`Unsupported volume source: ${source}`);
            return null;
        }
    }

    async _fetchFromServices(fetchFunc, description) {
        for (let service in this.apiConfigs) {
            try {
                logger.debug(`Fetching ${description} using ${service}...`);
                const result = await fetchFunc(service);
                if (result) {
                    return result;
                }
            } catch (e) {
                logger.warn(`Failed to fetch ${description} using ${service}: ${e.message}`);
            }
        }
        logger.warn(`Failed to fetch ${description}.`);
        return null;
    }

    async _loadAbi(abiPath) {
        try {
            const content = await fs.readFile(abiPath, 'utf-8');
            const abi = JSON.parse(content);
            logger.debug(`Loaded ABI from ${abiPath} successfully.`);
            return abi;
        } catch (e) {
            logger.error(`Failed to load ABI from ${abiPath}: ${e.message}`);
            throw e;
        }
    }

    async close() {
        try {
            await this.session.close();
        } catch (e) {
            logger.error(`Error closing API session: ${e.message}`);
        }
    }
}

export default API_Config;
