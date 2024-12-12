// Configuration.js
import fs from 'fs/promises';
import path from 'path';
import dotenv from 'dotenv';
import logger from './logger.js';
import { loadingBar } from './loadingBar.js';

dotenv.config(); // Load environment variables at the start

class Configuration {
    constructor() {
        // Initialize configuration variables
        this.ETHERSCAN_API_KEY = "";
        this.INFURA_PROJECT_ID = "";
        this.COINGECKO_API_KEY = "";
        this.COINMARKETCAP_API_KEY = "";
        this.CRYPTOCOMPARE_API_KEY = "";

        this.HTTP_ENDPOINT = "";
        this.IPC_ENDPOINT = "";
        this.WEBSOCKET_ENDPOINT = "";
        this.WALLET_KEY = "";
        this.WALLET_ADDRESS = "";

        this.AAVE_LENDING_POOL_ADDRESS = "";
        this.TOKEN_ADDRESSES = {};
        this.TOKEN_SYMBOLS = {};

        this.ERC20_ABI = "";
        this.ERC20_SIGNATURES = {};
        this.SUSHISWAP_ROUTER_ABI = "";
        this.SUSHISWAP_ROUTER_ADDRESS = "";
        this.UNISWAP_ROUTER_ABI = "";
        this.UNISWAP_ROUTER_ADDRESS = "";
        this.AAVE_FLASHLOAN_ABI = "";
        this.AAVE_LENDING_POOL_ABI = "";
        this.AAVE_FLASHLOAN_ADDRESS = "";
        this.PANCAKESWAP_ROUTER_ABI = "";
        this.PANCAKESWAP_ROUTER_ADDRESS = "";
        this.BALANCER_ROUTER_ABI = "";
        this.BALANCER_ROUTER_ADDRESS = "";
    }

    async load() {
        await this._load_configuration();
    }

    async _load_configuration() {
        try {
            // Display a loading bar while loading environment variables
            await loadingBar("Loading Environment Variables", 2);
            this._load_api_keys();
            this._load_providers_and_account();
            this._load_ML_models();
            await this._load_json_elements();
            logger.info("Configuration loaded successfully.");
        } catch (e) {
            throw new Error(`Failed to load configuration: ${e.message}`);
        }
    }

    _load_api_keys() {
        // Load API keys from environment variables
        this.ETHERSCAN_API_KEY = this._get_env_variable("ETHERSCAN_API_KEY");
        this.INFURA_PROJECT_ID = this._get_env_variable("INFURA_PROJECT_ID");
        this.COINGECKO_API_KEY = this._get_env_variable("COINGECKO_API_KEY");
        this.COINMARKETCAP_API_KEY = this._get_env_variable("COINMARKETCAP_API_KEY");
        this.CRYPTOCOMPARE_API_KEY = this._get_env_variable("CRYPTOCOMPARE_API_KEY");
    }

    _load_providers_and_account() {
        // Load blockchain provider endpoints and account details from environment variables
        this.HTTP_ENDPOINT = this._get_env_variable("HTTP_ENDPOINT");
        this.IPC_ENDPOINT = this._get_env_variable("IPC_ENDPOINT");
        this.WEBSOCKET_ENDPOINT = this._get_env_variable("WEBSOCKET_ENDPOINT");
        this.WALLET_KEY = this._get_env_variable("WALLET_KEY");
        this.WALLET_ADDRESS = this._get_env_variable("WALLET_ADDRESS");
    }

    _load_ML_models() {
        // Load machine learning model paths
        this.ML_MODEL_PATH = "models/price_model.joblib"; // Path to the trained price prediction model
        this.ML_TRAINING_DATA_PATH = "data/training_data.csv"; // Path to the training data CSV file
    }

    async _load_json_elements() {
        // Load monitored tokens and contract ABIs from JSON files
        this.AAVE_LENDING_POOL_ADDRESS = this._get_env_variable("AAVE_LENDING_POOL_ADDRESS");
        this.TOKEN_ADDRESSES = await this._load_json_file(
            this._get_env_variable("TOKEN_ADDRESSES"), "monitored tokens"
        );
        this.TOKEN_SYMBOLS = await this._load_json_file(
            this._get_env_variable("TOKEN_SYMBOLS"), "token symbols"
        );
        this.ERC20_ABI = await this._construct_abi_path("abi", "erc20_abi.json");
        this.ERC20_SIGNATURES = await this._load_json_file(
            this._get_env_variable("ERC20_SIGNATURES"), "ERC20 function signatures"
        );
        this.SUSHISWAP_ROUTER_ABI = await this._construct_abi_path("abi", "sushiswap_router_abi.json");
        this.SUSHISWAP_ROUTER_ADDRESS = this._get_env_variable("SUSHISWAP_ROUTER_ADDRESS");
        this.UNISWAP_ROUTER_ABI = await this._construct_abi_path("abi", "uniswap_router_abi.json");
        this.UNISWAP_ROUTER_ADDRESS = this._get_env_variable("UNISWAP_ROUTER_ADDRESS");
        this.AAVE_FLASHLOAN_ABI = await this._construct_abi_path("abi", "AAVE_FLASHLOAN_ABI.json");
        this.AAVE_LENDING_POOL_ABI = await this._construct_abi_path("abi", "AAVE_LENDING_POOL_ABI.json");
        this.AAVE_FLASHLOAN_ADDRESS = this._get_env_variable("AAVE_FLASHLOAN_ADDRESS");
        this.PANCAKESWAP_ROUTER_ABI = await this._construct_abi_path("abi", "pancakeswap_router_abi.json");
        this.PANCAKESWAP_ROUTER_ADDRESS = this._get_env_variable("PANCAKESWAP_ROUTER_ADDRESS");
        this.BALANCER_ROUTER_ABI = await this._construct_abi_path("abi", "balancer_router_abi.json");
        this.BALANCER_ROUTER_ADDRESS = this._get_env_variable("BALANCER_ROUTER_ADDRESS");
    }

    _get_env_variable(varName, defaultVal = null) {
        const value = process.env[varName] || defaultVal;
        if (value === null) {
            throw new Error(`Missing environment variable: ${varName}`);
        }
        return value;
    }

    async _load_json_file(filePath, description) {
        try {
            const content = await fs.readFile(filePath, 'utf-8');
            const data = JSON.parse(content);
            const itemCount = Array.isArray(data) ? data.length : Object.keys(data).length;
            await loadingBar(`Loading ${itemCount} ${description} from ${filePath}`, 3);
            return data;
        } catch (e) {
            logger.error(`${description.charAt(0).toUpperCase() + description.slice(1)} file not found: ${e.message}`);
            throw e;
        }
    }

    async _construct_abi_path(basePath, abiFilename) {
        const abiPath = path.join(basePath, abiFilename);
        try {
            await fs.access(abiPath);
            return abiPath;
        } catch (e) {
            logger.error(`ABI file not found at path: ${abiPath}`);
            throw new Error(`ABI file '${abiFilename}' not found in path '${basePath}'`);
        }
    }

    async getTokenAddresses() {
        return this.TOKEN_ADDRESSES;
    }

    async getTokenSymbols() {
        return this.TOKEN_SYMBOLS;
    }

    getAbiPath(abiName) {
        const abiPaths = {
            "erc20_abi": this.ERC20_ABI,
            "sushiswap_router_abi": this.SUSHISWAP_ROUTER_ABI,
            "uniswap_router_abi": this.UNISWAP_ROUTER_ABI,
            "AAVE_FLASHLOAN_ABI": this.AAVE_FLASHLOAN_ABI,
            "AAVE_LENDING_POOL_ABI": this.AAVE_LENDING_POOL_ABI,
            "pancakeswap_router_abi": this.PANCAKESWAP_ROUTER_ABI,
            "balancer_router_abi": this.BALANCER_ROUTER_ABI,
        };
        return abiPaths[abiName.toLowerCase()] || "";
    }
}

export default Configuration;
