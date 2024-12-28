const fs = require('fs');
const path = require('path');
const axios = require('axios');
const dotenv = require('dotenv');
const { TTLCache } = require('cachetools');
const { AsyncWeb3 } = require('web3');
const { ABI_Registry } = require('./abi_registry');
const { Decimal } = require('decimal.js');
const { setTimeout } = require('timers/promises');
const { ClientSession } = require('aiohttp');
const { StringIO } = require('stringio');

dotenv.config();

class Configuration {
    constructor() {
        this.IPC_ENDPOINT = null;
        this.HTTP_ENDPOINT = null;
        this.WEBSOCKET_ENDPOINT = null;
        this.WALLET_KEY = null;
        this.WALLET_ADDRESS = null;
        this.ETHERSCAN_API_KEY = null;
        this.INFURA_PROJECT_ID = null;
        this.COINGECKO_API_KEY = null;
        this.COINMARKETCAP_API_KEY = null;
        this.CRYPTOCOMPARE_API_KEY = null;
        this.AAVE_POOL_ADDRESS = null;
        this.TOKEN_ADDRESSES = null;
        this.TOKEN_SYMBOLS = null;
        this.ERC20_ABI = null;
        this.ERC20_SIGNATURES = null;
        this.SUSHISWAP_ABI = null;
        this.SUSHISWAP_ADDRESS = null;
        this.UNISWAP_ABI = null;
        this.UNISWAP_ADDRESS = null;
        this.AAVE_FLASHLOAN_ADDRESS = null;
        this.AAVE_FLASHLOAN_ABI = null;
        this.AAVE_POOL_ABI = null;
        this.AAVE_POOL_ADDRESS = null;
        
        this.MODEL_RETRAINING_INTERVAL = 3600;
        this.MIN_TRAINING_SAMPLES = 100;
        this.MODEL_ACCURACY_THRESHOLD = 0.7;
        this.PREDICTION_CACHE_TTL = 300;

        this.SLIPPAGE_DEFAULT = 0.1;
        this.SLIPPAGE_MIN = 0.01;
        this.SLIPPAGE_MAX = 0.5;
        this.SLIPPAGE_HIGH_CONGESTION = 0.05;
        this.SLIPPAGE_LOW_CONGESTION = 0.2;
        this.MAX_GAS_PRICE_GWEI = 500;
        this.MIN_PROFIT_MULTIPLIER = 2.0;
        this.BASE_GAS_LIMIT = 21000;
        this.LINEAR_REGRESSION_PATH = "/linear_regression";
        this.MODEL_PATH = "/linear_regression/price_model.joblib";
        this.TRAINING_DATA_PATH = "/linear_regression/training_data.csv";

        this.abi_registry = new ABI_Registry();

        this.WETH_ADDRESS = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2";
        this.USDC_ADDRESS = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48";
        this.USDT_ADDRESS = "0xdAC17F958D2ee523a2206206994597C13D831ec7";
    }

    async load() {
        try {
            console.info("Loading configuration... ⏳");
            await setTimeout(1000);

            await this.abi_registry.initialize();
            await this._loadConfiguration();
            console.info("System reporting go for launch ✅...");
            await setTimeout(3000);

            console.debug("All Configurations and Environment Variables Loaded Successfully ✅");
        } catch (e) {
            console.error(`Error loading configuration: ${e}`);
            throw e;
        }
    }

    async _loadConfiguration() {
        try {
            if (!this.abi_registry.abis) {
                throw new Error("Failed to load ABIs");
            }

            this._loadProvidersAndAccount();
            this._loadApiKeys();
            await this._loadJsonElements();
        } catch (e) {
            console.error(`Error loading configuration: ${e}`);
            throw e;
        }
    }

    _loadApiKeys() {
        this.ETHERSCAN_API_KEY = this._getEnvVariable("ETHERSCAN_API_KEY");
        this.INFURA_PROJECT_ID = this._getEnvVariable("INFURA_PROJECT_ID");
        this.COINGECKO_API_KEY = this._getEnvVariable("COINGECKO_API_KEY");
        this.COINMARKETCAP_API_KEY = this._getEnvVariable("COINMARKETCAP_API_KEY");
        this.CRYPTOCOMPARE_API_KEY = this._getEnvVariable("CRYPTOCOMPARE_API_KEY");
    }

    _loadProvidersAndAccount() {
        try {
            this.IPC_ENDPOINT = process.env.IPC_ENDPOINT;
            this.HTTP_ENDPOINT = process.env.HTTP_ENDPOINT;
            this.WEBSOCKET_ENDPOINT = process.env.WEBSOCKET_ENDPOINT;

            const activeEndpoints = [this.IPC_ENDPOINT, this.HTTP_ENDPOINT, this.WEBSOCKET_ENDPOINT].filter(endpoint => endpoint != null).length;

            if (activeEndpoints !== 1) {
                const active = [];
                if (this.IPC_ENDPOINT) active.push("IPC");
                if (this.HTTP_ENDPOINT) active.push("HTTP");
                if (this.WEBSOCKET_ENDPOINT) active.push("WebSocket");

                throw new Error(`Exactly one endpoint (IPC, HTTP, or WebSocket) must be configured. Found ${activeEndpoints} active endpoints: ${active.join(', ')}`);
            }

            this.WALLET_KEY = this._getEnvVariable("WALLET_KEY");
            this.WALLET_ADDRESS = this._getEnvVariable("WALLET_ADDRESS");

            console.info("Providers and Account loaded ✅");
        } catch (e) {
            console.error(`Error loading providers and account: ${e}`);
            throw e;
        }
    }

    _getEnvVariable(varName, defaultValue = null) {
        const value = process.env[varName] || defaultValue;
        if (value === null) {
            throw new Error(`Missing environment variable: ${varName}`);
        }
        return value;
    }

    async _loadJsonElements() {
        try {
            this.AAVE_POOL_ADDRESS = this._getEnvVariable("AAVE_POOL_ADDRESS");
            this.TOKEN_ADDRESSES = await this._loadJsonFile(this._getEnvVariable("TOKEN_ADDRESSES"), "monitored tokens");
            this.TOKEN_SYMBOLS = await this._loadJsonFile(this._getEnvVariable("TOKEN_SYMBOLS"), "token symbols");
            this.ERC20_ABI = await this._constructAbiPath("abi", "erc20_abi.json");
            this.ERC20_SIGNATURES = await this._loadJsonFile(this._getEnvVariable("ERC20_SIGNATURES"), "ERC20 function signatures");
            this.SUSHISWAP_ABI = await this._constructAbiPath("abi", "sushiswap_abi.json");
            this.SUSHISWAP_ADDRESS = this._getEnvVariable("SUSHISWAP_ADDRESS");
            this.UNISWAP_ABI = await this._constructAbiPath("abi", "uniswap_abi.json");
            this.UNISWAP_ADDRESS = this._getEnvVariable("UNISWAP_ADDRESS");
            this.AAVE_FLASHLOAN_ABI = await this._loadJsonFile(await this._constructAbiPath("abi", "aave_flashloan_abi.json"), "Aave Flashloan ABI");
            this.AAVE_POOL_ABI = await this._loadJsonFile(await this._constructAbiPath("abi", "aave_pool_abi.json"), "Aave Lending Pool ABI");
            this.AAVE_FLASHLOAN_ADDRESS = this._getEnvVariable("AAVE_FLASHLOAN_ADDRESS");
            console.debug("JSON elements loaded successfully.");
        } catch (e) {
            console.error(`Error loading JSON elements: ${e}`);
            throw e;
        }
    }

    async _loadJsonFile(filePath, description) {
        try {
            const data = JSON.parse(await fs.promises.readFile(filePath, 'utf-8'));
            console.debug(`Successfully loaded ${description} from ${filePath}`);
            return data;
        } catch (e) {
            console.error(`Error loading file ${description}: ${e}`);
            throw e;
        }
    }

    async _constructAbiPath(basePath, abiFilename) {
        const abiPath = path.join(basePath, abiFilename);
        if (!fs.existsSync(abiPath)) {
            console.error(`ABI file does not exist: ${abiPath}`);
            throw new Error(`ABI file not found: ${abiPath}`);
        }
        console.debug(`ABI path constructed: ${abiPath}`);
        return abiPath;
    }
}

class API_Config {
    constructor(configuration = null) {
        this.configuration = configuration;
        this.session = null;
        this.price_cache = new TTLCache({ maxsize: 2000, ttl: 300 });  // 5 min cache for prices
        this.volume_cache = new TTLCache({ maxsize: 1000, ttl: 900 }); // 15 min cache for volumes
        this.market_data_cache = new TTLCache({ maxsize: 1000, ttl: 1800 });  // 30 min cache for market data
        this.token_metadata_cache = new TTLCache({ maxsize: 500, ttl: 86400 });  // 24h cache for metadata

        this.api_configs = {
            "binance": {
                "base_url": "https://api.binance.com/api/v3",
                "market_url": "/ticker/24hr",
                "rate_limit": 1200,
            },
            "coingecko": {
                "base_url": "https://api.coingecko.com/api/v3",
                "market_url": "/coins/{id}/market_chart",
                "api_key": configuration.COINGECKO_API_KEY,
                "rate_limit": 50,
            },
            "coinmarketcap": {
                "base_url": "https://pro-api.coinmarketcap.com/v1",
                "ticker_url": "/cryptocurrency/quotes/latest",
                "api_key": configuration.COINMARKETCAP_API_KEY,
                "rate_limit": 333,
            },
            "cryptocompare": {
                "base_url": "https://min-api.cryptocompare.com/data",
                "price_url": "/price",
                "api_key": configuration.CRYPTOCOMPARE_API_KEY,
                "rate_limit": 80,
            },
        };

        this.rate_limiters = Object.fromEntries(
            Object.entries(this.api_configs).map(([provider, config]) => [
                provider,
                new AsyncSemaphore(config.rate_limit)
            ])
        );
    }

    async get_token_symbol(web3, token_address) {
        if (this.token_metadata_cache.has(token_address)) {
            return this.token_metadata_cache.get(token_address).symbol;
        }
        try {
            const erc20_abi = await this._load_abi(this.configuration.ERC20_ABI);
            const contract = web3.eth.contract({ address: token_address, abi: erc20_abi });
            const symbol = await contract.methods.symbol().call();
            this.token_metadata_cache.set(token_address, { symbol });
            return symbol;
        } catch (e) {
            console.error(`Error getting symbol for token ${token_address}: ${e}`);
            return null;
        }
    }

    async get_real_time_price(token, vs_currency = 'eth') {
        try {
            const price = await this._fetch_price("coingecko", token, vs_currency);
            if (!price) {
                throw new Error("Price fetching failed");
            }
            return new Decimal(price);
        } catch (e) {
            console.error(`Error fetching real-time price for ${token}: ${e}`);
            return null;
        }
    }

    async _fetch_price(provider, token, vs_currency) {
        const config = this.api_configs[provider];
        try {
            const url = `${config.base_url}/simple/price?ids=${token}&vs_currencies=${vs_currency}`;
            const response = await axios.get(url);
            if (response.status === 200) {
                return response.data[token][vs_currency];
            } else {
                console.error(`Failed to fetch price from ${provider}: Status ${response.status}`);
                return null;
            }
        } catch (e) {
            console.error(`Exception fetching price from ${provider}: ${e}`);
            return null;
        }
    }

    async _load_abi(abiPath) {
        try {
            return await this.configuration.abi_registry.get_abi(abiPath);
        } catch (e) {
            console.error(`Failed to load ABI from ${abiPath}: ${e}`);
            throw e;
        }
    }

    async close() {
        if (this.session) {
            await this.session.close();
        }
    }
}

module.exports = { Configuration, API_Config };
