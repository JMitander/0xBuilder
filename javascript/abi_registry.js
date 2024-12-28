const fs = require('fs');
const path = require('path');
const util = require('util');

const readFile = util.promisify(fs.readFile);

class ABIRegistry {
    // Define required methods for various protocols
    static REQUIRED_METHODS = {
        'erc20': ['transfer', 'approve', 'transferFrom', 'balanceOf'],
        'uniswap': ['swapExactTokensForTokens', 'swapTokensForExactTokens', 'addLiquidity', 'getAmountsOut'],
        'sushiswap': ['swapExactTokensForTokens', 'swapTokensForExactTokens', 'addLiquidity', 'getAmountsOut'],
        'pancakeswap': ['swapExactTokensForTokens', 'swapTokensForExactTokens', 'addLiquidity', 'getAmountsOut'],
        'balancer': ['swap', 'addLiquidity'],
        'aave_flashloan': ['fn_RequestFlashLoan', 'executeOperation', 'ADDRESSES_PROVIDER', 'POOL'],
        'aave': ['ADDRESSES_PROVIDER', 'getReservesList', 'getReserveData']
    };

    constructor() {
        this.abis = {};
        this.signatures = {};
        this.methodSelectors = {};
        this._initialized = false;
    }

    async initialize() {
        if (this._initialized) {
            console.debug("ABI_Registry already initialized.");
            return;
        }
        await this._loadAllAbis();
        this._initialized = true;
        console.debug("ABI_Registry initialization complete.");
    }

    async _loadAllAbis() {
        const abiDir = path.join(__dirname, '..', 'abi');

        const abiFiles = {
            'erc20': 'erc20_abi.json',
            'uniswap': 'uniswap_abi.json',
            'sushiswap': 'sushiswap_router_abi.json',
            'pancakeswap': 'pancakeswap_router_abi.json',
            'balancer': 'balancer_router_abi.json',
            'aave_flashloan': 'aave_flashloan.abi.json',
            'aave': 'aave_pool_abi.json'
        };

        // Define critical ABIs that are essential for the application
        const criticalAbis = ['erc20', 'uniswap'];

        for (let [abiType, filename] of Object.entries(abiFiles)) {
            const abiPath = path.join(abiDir, filename);
            try {
                const abi = await this._loadAbiFromPath(abiPath, abiType);
                this.abis[abiType] = abi;
                this._extractSignatures(abi, abiType);
                console.debug(`Loaded and validated ${abiType} ABI from ${abiPath}`);
            } catch (err) {
                console.error(`Error loading ${abiType} ABI: ${err.message}`);
                if (criticalAbis.includes(abiType)) {
                    throw err;
                } else {
                    console.warn(`Skipping non-critical ABI: ${abiType}`);
                }
            }
        }
    }

    async _loadAbiFromPath(abiPath, abiType) {
        try {
            const abiContent = await readFile(abiPath, 'utf-8');
            const abi = JSON.parse(abiContent);
            console.debug(`ABI content loaded from ${abiPath}`);

            if (!this._validateAbi(abi, abiType)) {
                throw new Error(`Validation failed for ${abiType} ABI from file ${abiPath}`);
            }

            return abi;
        } catch (err) {
            console.error(`Error loading ABI ${abiType} from ${abiPath}: ${err.message}`);
            throw err;
        }
    }

    async loadAbi(abiType) {
        const abiDir = path.join(__dirname, '..', 'abi');
        const abiFiles = {
            'erc20': 'erc20_abi.json',
            'uniswap': 'uniswap_abi.json',
            'sushiswap': 'sushiswap_router_abi.json',
            'pancakeswap': 'pancakeswap_router_abi.json',
            'balancer': 'balancer_router_abi.json',
            'aave_flashloan': 'aave_flashloan.abi.json',
            'aave': 'aave_pool_abi.json'
        };

        try {
            if (!(abiType in abiFiles)) {
                console.error(`Unknown ABI type: ${abiType}`);
                return null;
            }

            const abiPath = path.join(abiDir, abiFiles[abiType]);
            const abi = await this._loadAbiFromPath(abiPath, abiType);
            this.abis[abiType] = abi;
            this._extractSignatures(abi, abiType);
            console.debug(`Loaded and validated ${abiType} ABI`);
            return abi;
        } catch (err) {
            console.error(`Error loading ABI ${abiType}: ${err.message}`);
            return null;
        }
    }

    _validateAbi(abi, abiType) {
        if (!Array.isArray(abi)) {
            console.error(`Invalid ABI format for ${abiType}`);
            return false;
        }

        const foundMethods = new Set(
            abi.filter(item => item.type === 'function' && item.name).map(item => item.name)
        );

        const required = new Set(ABIRegistry.REQUIRED_METHODS[abiType] || []);
        const missing = [...required].filter(x => !foundMethods.has(x));

        if (missing.length > 0) {
            console.error(`Missing required methods in ${abiType} ABI: ${missing.join(', ')}`);
            return false;
        }

        return true;
    }

    _extractSignatures(abi, abiType) {
        const signatures = {};
        const selectors = {};

        for (let item of abi) {
            if (item.type === 'function') {
                const name = item.name;
                if (name) {
                    const inputs = (item.inputs || []).map(inp => inp.type).join(',');
                    const signature = `${name}(${inputs})`;
                    const selector = this._functionSignatureTo4ByteSelector(signature);

                    signatures[name] = signature;
                    selectors[selector] = name;
                }
            }
        }

        this.signatures[abiType] = signatures;
        this.methodSelectors[abiType] = selectors;
    }

    _functionSignatureTo4ByteSelector(signature) {
        const web3 = require('web3');
        return web3.utils.sha3(signature).slice(0, 10); // Getting the first 4 bytes
    }

    getAbi(abiType) {
        return this.abis[abiType] || null;
    }

    getMethodSelector(selector) {
        for (let [abiType, selectors] of Object.entries(this.methodSelectors)) {
            if (selectors[selector]) {
                return selectors[selector];
            }
        }
        return null;
    }

    getFunctionSignature(abiType, methodName) {
        return this.signatures[abiType]?.[methodName] || null;
    }
}

module.exports = ABIRegistry;
