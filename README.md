
# 0xBuilder MEV Bot

[![License](https://img.shields.io/badge/license-MIT-white.svg)](LICENSE)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-white.svg)](CONTRIBUTING.md)
##
[![Python Version](https://img.shields.io/badge/Python-3.12.*-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![geth](https://img.shields.io/badge/Geth-v1.14.*-blue.svg)](https://geth.ethereum.org/)
[![Remix](https://img.shields.io/badge/Remix-IDE-blue.svg)](https://remix.ethereum.org/) 
##
[![AAVE](https://img.shields.io/badge/Aave-v3-orange.svg)](https://aave.com/)
[![Uniswap](https://img.shields.io/badge/Uniswap-v2.0-orange.svg)](https://uniswap.org/)
[![Sushiswap](https://img.shields.io/badge/Sushiswap-v2-orange.svg)](https://sushiswap.fi/)
[![PancakeSwap](https://img.shields.io/badge/PancakeSwap-v2/v3-orange.svg)](https://pancakeswap.finance/)
[![Balancer](https://img.shields.io/badge/Balancer-v3-orange.svg)](https://balancer.finance/)
##
[![Infura](https://img.shields.io/badge/Infura-API-red.svg)](https://infura.io/)
[![CoinGecko](https://img.shields.io/badge/Coingecko-API-red.svg)](https://www.coingecko.com/en/api)
[![CoinMarketCap](https://img.shields.io/badge/CoinMarketcap-API-red.svg)](https://coinmarketcap.com/api/)
[![CryptoCompare](https://img.shields.io/badge/Cryptocompare-API-red.svg)](https://min-api.cryptocompare.com/)
[![Etherscan](https://img.shields.io/badge/Etherscan-API-red.svg)](https://etherscan.io/apis)

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
  - [System Requirements](#system-requirements)
  - [Software Dependencies](#software-dependencies)
  - [Ethereum Node Setup](#ethereum-node-setup)
- [Installation](#installation)
  - [Cloning the Repository](#cloning-the-repository)
  - [Setting up Virtual Environment](#setting-up-virtual-environment)
  - [Installing Dependencies](#installing-dependencies)
- [Configuration](#configuration)
  - [Environment Variables](#environment-variables)
  - [Configuration Files](#configuration-files)
- [Deploying the Flashloan Contract](#deploying-the-flashloan-contract)
- [Obtaining API Keys](#obtaining-api-keys)
- [Running the Bot](#running-the-bot)
- [Strategies](#strategies)
- [Logging](#logging)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Disclaimer](#disclaimer)

## Introduction

**0xBuilder** is an advanced Ethereum trading bot designed for high-frequency trading and MEV (Maximal Extractable Value) opportunities. It implements strategies like front-running, back-running, sandwich attacks, and flashloan executions using Python, Geth, Remix, and AsyncWeb3.py. The bot continuously monitors the Ethereum mempool for profitable transactions and executes trades automatically.

The bot is highly configurable, allowing users to adjust parameters, strategies, and risk levels based on their preferences. It supports multiple wallets, tokens, and trading pairs, with real-time market analysis and safety checks. The bot can be run on any Ethereum-compatible network, with support for various APIs and external data sources.

**Note:** 0xBuilder is a work in progress and is not production-ready. Use it at your own risk and discretion.

## Features

![0xBuilder-flow](https://github.com/user-attachments/assets/29e3da12-d253-4304-acb1-f2d74f407bf1)

- **Mempool Monitoring**: Continuously monitors the Ethereum mempool for potential arbitrage and profit opportunities.
- **Strategy Execution**: Implements various strategies, including front-running, back-running, sandwich attacks, and flashloan executions.
- **Flashloan Integration**: Utilizes flashloans to maximize capital efficiency without initial capital requirements.
- **Market Analysis**: Analyzes market conditions using data from multiple APIs and external data sources.
- **Dynamic Gas Pricing**: Adjusts gas prices based on network conditions to optimize transaction inclusion and costs.
- **Nonce Management**: Manages nonces effectively to prevent transaction failures due to nonce collisions.
- **Safety Mechanisms**: Includes safety checks and validations to manage risks and ensure profitability.
- **Smart Contract Interactions**: Interacts with various DeFi protocols, including Uniswap, Aave, Sushiswap, PancakeSwap, and Balancer.
- **Transaction Bundling**: Groups multiple transactions into a single block for efficiency.
- **API Integration**: Connects to various APIs for blockchain data, pricing, and market data.
- **Configurable Parameters**: Allows users to adjust parameters, strategies, and risk levels based on preferences.
- **Detailed Logging**: Provides detailed logs of bot activities, transactions, and strategies for analysis and debugging.
- **Customizable**: Supports multiple wallets, tokens, and trading pairs, with the ability to add new strategies and features.



## Project Structure

```
/0xBuilder/
├── abi/
│   ├── uniswap_router_abi.json
│   ├── sushiswap_router_abi.json
│   ├── pancakeswap_router_abi.json
│   ├── erc20_abi.json
│   ├── balancer_router_abi.json
│   └── aave_lending_pool_abi.json
├── contracts/
│   ├── SimpleFlashloan.sol
│   └── IERC20.sol
├── javascript/
│   ├── nonce.js
│   ├── net.js
│   ├── monitor.js
│   ├── main.js
│   ├── core.js
│   ├── configuration.js
│   ├── colorformatter.js
│   ├── abi_registry.js
│   ├── __init__.js
│   └── jsutils/
│       ├── strategyperformancemetrics.js
│       ├── strategyexecutionerror.js
│       ├── strategyconfiguration.js
│       └── colorformatter.html
├── linear_regression/
│   ├── training_data.csv
│   └── price_model.joblib
├── python/
│   ├── nonce.py
│   ├── net.py
│   ├── monitor.py
│   ├── main.py
│   ├── core.py
│   ├── constants.py
│   ├── configuration.py
│   ├── abi_registry.py
│   ├── __init__.py
│   └── pyutils/
│       ├── strategyexecutionerror.py
│       ├── strategyconfiguration.py
│       ├── colorformatter.py
│       └── __init__.py
├── shared/
│   └── MITANDER.py
├── utils/
│   ├── token_addresses.json
│   ├── erc20_signatures.json
│   └── token_symbols.json
├── Logs/
│   └── 0xBuilder_log.txt
├── .env.example
├── .gitignore
├── CONTRIBUTING.md
├── LICENSE
├── README.md
└── requirements.txt
```

### Description of Key Directories and Files

- **abi/**: Contains JSON files for various smart contract ABIs used in the project.
  
- **contracts/**: Includes Solidity smart contracts such as `SimpleFlashloan.sol` and `IERC20.sol`.
  
- **javascript/**: Holds all JavaScript files related to the project, including utility scripts in the `jsutils/` subdirectory.
  
- **linear_regression/**: Contains data and models related to linear regression analysis, such as `training_data.csv` and `price_model.joblib`.
  
- **python/**: Contains Python scripts that form the core functionality of the project. The `pyutils/` subdirectory includes utility modules for error handling and configuration.
  
- **shared/**: Includes shared Python scripts like `MITANDER.py` that might be used across different parts of the project.
  
- **utils/**: Stores utility JSON files that hold token addresses, ERC20 signatures, and token symbols.
  
- **Logs/**: Maintains log files such as `0xBuilder_log.txt` to track bot activities and operations.
  
- **.env.example**: Example environment variables file to guide configuration.
  
- **.gitignore**: Specifies files and directories to be ignored by Git.
  
- **CONTRIBUTING.md**: Guidelines for contributing to the project.
  
- **LICENSE**: Contains the licensing information for the project.
  
- **README.md**: Provides an overview and documentation for the project.
  
- **requirements.txt**: Lists the Python dependencies required for the project.

## Prerequisites

Before running 0xBuilder, ensure you have the following:

### System Requirements

- **Operating System**: Ubuntu 22.04 LTS or later (Windows 11 and macOS Ventura also supported)
- **Python**: Version 3.12 or higher
- **Node.js**: Version 18 LTS or higher (required for smart contract deployment)
- **Geth**: Latest stable version for running a full Ethereum node
- **Internet**: High-speed connection with minimum 50Mbps upload/download
- **Hardware**:
   - CPU: 4+ cores, 3.0GHz or faster
   - RAM: 16GB minimum, 32GB recommended
   - Storage: 2TB NVMe SSD recommended
   - Network: Low-latency ethernet connection

### Software Dependencies

Primary Components:
- **Execution Client**: Latest version of Geth, Nethermind, or Besu
- **Consensus Client**: Latest version of Prysm or Lighthouse
- **Development Tools**:
   - solc v0.8.19 or higher
   - web3.py v6.0 or higher
   - ethers.js v6.0 or higher
   - All Python packages from `requirements.txt`

Additional Requirements:
- **Git**: Latest stable version for version control
- **Docker**: Latest stable version (optional, for containerization)
- **Build Tools**: make, gcc, and platform-specific compilers

### Ethereum Node Setup

Choose and set up an execution client (EL) compatible with the Ethereum network:

| Client | Language | OS Support | Networks | Sync Methods |
|--------|----------|------------|----------|--------------|
| [Geth](https://geth.ethereum.org/) | Go | Linux, Windows, macOS | Mainnet, Sepolia, Holesky | Snap, Full |
| [Nethermind](https://www.nethermind.io/) | C#/.NET | Linux, Windows, macOS | Mainnet, Sepolia, Holesky | Snap, Fast, Full |
| [Besu](https://besu.hyperledger.org/) | Java | Linux, Windows, macOS | Mainnet, Sepolia, Holesky | Snap, Fast, Full |
| [Erigon](https://github.com/ledgerwatch/erigon) | Go | Linux, Windows, macOS | Mainnet, Sepolia, Holesky | Full |
| [Reth](https://reth.rs/) | Rust | Linux, Windows, macOS | Mainnet, Sepolia, Holesky | Full |
| [EthereumJS](https://github.com/ethereumjs/ethereumjs-monorepo) | TypeScript | Linux, Windows, macOS | Sepolia, Holesky | Full |

#### Geth Configuration

1. **Installation**:
   Follow the official [Geth installation guide](https://geth.ethereum.org/docs/install-and-build/installing-geth).

2. **Launch Node**:
   ```bash
   geth --mainnet \
     --syncmode "snap" \
     --http \
     --http.api "eth,net,web3,txpool" \
     --ws \
     --ws.api "eth,net,web3,txpool" \
     --maxpeers 100 \
     --cache 8192 \
     --ipcpath "/path/to/geth.ipc"
   ```

3. **Monitor Sync**:
   ```bash
   # Connect to node
   geth attach ipc:/path/to/geth.ipc

   # Check sync status
   > eth.syncing
   ```

#### Beacon Node Setup

For PoS consensus layer, install either:

- [Prysm](https://docs.prylabs.network/docs/getting-started)
- [Lighthouse](https://lighthouse-book.sigmaprime.io/installation.html)

## Installation

### Cloning the Repository

```bash
git clone https://github.com/yourusername/0xBuilder.git
cd 0xBuilder
```

### Setting up Virtual Environment

Using a virtual environment is strongly recommended to manage dependencies and avoid conflicts:

For Linux/MacOS:

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Verify activation
which python
```

For Windows:

```powershell
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate

# Verify activation
where python
```

### Installing Dependencies

Install required packages:

```bash
# Upgrade pip to latest version
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Verify installations
pip list
```

## Configuration

### Environment Variables

1. Create a `.env` file in the project root:

```bash
# Linux/MacOS
cp .env.example .env

# Windows
copy .env.example .env
```

2. Configure the environment variables in `.env`:
   - Add API keys from various services
   - Configure node endpoints
   - Set up wallet details
   - Define smart contract addresses

3. Validate the configuration:

```bash
# Verify .env file exists and permissions
ls -la .env

# Set secure file permissions (Linux/MacOS)
chmod 600 .env
```

Example `.env` configuration:

```ini
# API Configuration
ETHERSCAN_API_KEY=your_etherscan_api_key
INFURA_PROJECT_ID=your_infura_project_id
COINGECKO_API_KEY=your_coingecko_api_key
COINMARKETCAP_API_KEY=your_coinmarketcap_api_key
CRYPTOCOMPARE_API_KEY=your_cryptocompare_api_key

# Ethereum Node Configuration
HTTP_ENDPOINT=http://127.0.0.1:8545
WS_ENDPOINT=wss://127.0.0.1:8546
IPC_ENDPOINT=/path/to/geth.ipc

# Wallet Configuration
PRIVATE_KEY=your_private_key
WALLET_ADDRESS=0xYourWalletAddress
PROFIT_WALLET=0xYourProfitAddress

# Token Configuration
TOKEN_LIST_PATH=utils/token_addresses.json
TOKEN_SYMBOLS_PATH=utils/token_symbols.json

# DEX Router Configurations
UNISWAP_V2_ROUTER=0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D
SUSHISWAP_ROUTER=0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F
PANCAKESWAP_ROUTER=0xEfF92A263d31888d860bD50809A8D171709b7b1c
BALANCER_ROUTER=0x3E66B66Fd1d0b02fDa6C811da9E0547970DB2f21

# ABI Paths
UNISWAP_V2_ABI=abi/uniswap_router_abi.json
SUSHISWAP_ABI=abi/sushiswap_router_abi.json
PANCAKESWAP_ABI=abi/pancakeswap_router_abi.json
BALANCER_ABI=abi/balancer_router_abi.json
ERC20_ABI=abi/erc20_abi.json

# Flashloan Configuration
AAVE_V3_FLASHLOAN_CONTRACT=0xYourFlashloanContractAddress
AAVE_V3_LENDING_POOL=0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2
```

### Configuration Files

Essential JSON configuration files must be present in the `utils` directory:

| File | Description | Format |
|------|-------------|--------|
| `token_addresses.json` | Actively monitored token contracts | `{"symbol": "address"}` |
| `token_symbols.json` | Token address to symbol mapping | `{"address": "symbol"}` |
| `erc20_signatures.json` | Common ERC20 function signatures | `{"name": "signature"}` |

Verify all configuration files are properly formatted and contain valid data before starting the bot.

## Deploying the Flashloan Contract

Deploy a flashloan contract compatible with Aave V3 or your preferred protocol to enable flashloan functionality.

### Deployment Options

#### Using Remix IDE (Recommended)

1. Launch [Remix IDE](https://remix.ethereum.org/)
2. Create `SimpleFlashloan.sol`
3. Implement flashloan logic following Aave's specifications
4. Compile:
   - Select Solidity compiler v0.8.19+
   - Verify successful compilation
5. Deploy:
   - Connect MetaMask via "Injected Web3"
   - Supply constructor arguments
   - Confirm deployment transaction
6. Update `.env` with contract address

#### Using Development Frameworks

1. Install framework:
   ```bash
   # Hardhat
   npm install --save-dev hardhat
   # or Truffle
   npm install -g truffle
   ```
2. Compile contract:
   ```bash
   # Hardhat
   npx hardhat compile
   # or Truffle
   truffle compile
   ```
3. Deploy:
   ```bash
   # Hardhat
   npx hardhat run scripts/deploy.js
   # or Truffle
   truffle migrate
   ```
4. Update `.env` configuration

## Obtaining API Keys

Register and obtain API keys from:

1. [Infura](https://infura.io/) - RPC endpoints
2. [Etherscan](https://etherscan.io/apis) - Transaction data
3. [CoinGecko](https://www.coingecko.com/en/api) - Price feeds
4. [CoinMarketCap](https://coinmarketcap.com/api/) - Market data
5. [CryptoCompare](https://min-api.cryptocompare.com/) - Real-time prices

Ensure that all API keys are stored securely and not shared publicly.

## Installation

### Cloning the Repository

```bash
git clone https://github.com/yourusername/0xBuilder.git
cd 0xBuilder
```

### Setting up Virtual Environment

Using a virtual environment is strongly recommended to manage dependencies and avoid conflicts:

For Linux/MacOS:

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Verify activation
which python
```

For Windows:

```powershell
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate

# Verify activation
where python
```

### Installing Dependencies

Install required packages:

```bash
# Upgrade pip to latest version
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Verify installations
pip list
```

## Configuration

### Environment Variables

1. Create a `.env` file in the project root:

```bash
# Linux/MacOS
cp .env.example .env

# Windows
copy .env.example .env
```

2. Configure the environment variables in `.env`:
   - Add API keys from various services
   - Configure node endpoints
   - Set up wallet details
   - Define smart contract addresses

3. Validate the configuration:

```bash
# Verify .env file exists and permissions
ls -la .env

# Set secure file permissions (Linux/MacOS)
chmod 600 .env
```

Example `.env` configuration:

```ini
# API Configuration
ETHERSCAN_API_KEY=your_etherscan_api_key
INFURA_PROJECT_ID=your_infura_project_id
COINGECKO_API_KEY=your_coingecko_api_key
COINMARKETCAP_API_KEY=your_coinmarketcap_api_key
CRYPTOCOMPARE_API_KEY=your_cryptocompare_api_key

# Ethereum Node Configuration
HTTP_ENDPOINT=http://127.0.0.1:8545
WS_ENDPOINT=wss://127.0.0.1:8546
IPC_ENDPOINT=/path/to/geth.ipc

# Wallet Configuration
PRIVATE_KEY=your_private_key
WALLET_ADDRESS=0xYourWalletAddress
PROFIT_WALLET=0xYourProfitAddress

# Token Configuration
TOKEN_LIST_PATH=utils/token_addresses.json
TOKEN_SYMBOLS_PATH=utils/token_symbols.json

# DEX Router Configurations
UNISWAP_V2_ROUTER=0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D
SUSHISWAP_ROUTER=0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F
PANCAKESWAP_ROUTER=0xEfF92A263d31888d860bD50809A8D171709b7b1c
BALANCER_ROUTER=0x3E66B66Fd1d0b02fDa6C811da9E0547970DB2f21

# ABI Paths
UNISWAP_V2_ABI=abi/uniswap_router_abi.json
SUSHISWAP_ABI=abi/sushiswap_router_abi.json
PANCAKESWAP_ABI=abi/pancakeswap_router_abi.json
BALANCER_ABI=abi/balancer_router_abi.json
ERC20_ABI=abi/erc20_abi.json

# Flashloan Configuration
AAVE_V3_FLASHLOAN_CONTRACT=0xYourFlashloanContractAddress
AAVE_V3_LENDING_POOL=0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2
```

### Configuration Files

Essential JSON configuration files must be present in the `utils` directory:

| File | Description | Format |
|------|-------------|--------|
| `token_addresses.json` | Actively monitored token contracts | `{"symbol": "address"}` |
| `token_symbols.json` | Token address to symbol mapping | `{"address": "symbol"}` |
| `erc20_signatures.json` | Common ERC20 function signatures | `{"name": "signature"}` |

Verify all configuration files are properly formatted and contain valid data before starting the bot.

## Deploying the Flashloan Contract

Deploy a flashloan contract compatible with Aave V3 or your preferred protocol to enable flashloan functionality.

### Deployment Options

#### Using Remix IDE (Recommended)

1. Launch [Remix IDE](https://remix.ethereum.org/)
2. Create `SimpleFlashloan.sol`
3. Implement flashloan logic following Aave's specifications
4. Compile:
   - Select Solidity compiler v0.8.19+
   - Verify successful compilation
5. Deploy:
   - Connect MetaMask via "Injected Web3"
   - Supply constructor arguments
   - Confirm deployment transaction
6. Update `.env` with contract address

#### Using Development Frameworks

1. Install framework:
   ```bash
   # Hardhat
   npm install --save-dev hardhat
   # or Truffle
   npm install -g truffle
   ```
2. Compile contract:
   ```bash
   # Hardhat
   npx hardhat compile
   # or Truffle
   truffle compile
   ```
3. Deploy:
   ```bash
   # Hardhat
   npx hardhat run scripts/deploy.js
   # or Truffle
   truffle migrate
   ```
4. Update `.env` configuration

## Obtaining API Keys

Register and obtain API keys from:

1. [Infura](https://infura.io/)
2. [Etherscan](https://etherscan.io/apis)
3. [CoinGecko](https://www.coingecko.com/en/api)
4. [CoinMarketCap](https://coinmarketcap.com/api/)
5. [CryptoCompare](https://min-api.cryptocompare.com/)

Ensure that all API keys are stored securely and not shared publicly.

## Running the Bot

### Prerequisites

- Synchronized Ethereum node
- Active beacon node
- Configured environment variables
- Valid API keys

### Launch Sequence

1. Activate environment:
   ```bash
   source venv/bin/activate
   ```

2. Start bot:
   ```bash
   python python/main.py
   ```

   *Note:* Adjust the script path if necessary based on your project structure.

### Monitoring

- Check `Logs/0xBuilder_log.txt` for detailed operation logs
- Monitor console output for real-time status
- Use `Ctrl+C` for graceful shutdown

### Performance Optimization

- Keep node fully synced
- Monitor API rate limits
- Maintain sufficient ETH balance
- Regularly check log files
- Update dependencies as needed

## Strategies

0xBuilder implements several sophisticated trading strategies to capitalize on profitable opportunities within the Ethereum network:

### Core Strategies

- **Front-Running**: Executes higher-priority transactions ahead of detected profitable transactions
- **Back-Running**: Places transactions immediately after identified profitable transactions
- **Sandwich Attacks**: Employs coordinated front-running and back-running around target transactions
- **Flashloan Arbitrage**: Leverages borrowed assets for zero-capital arbitrage opportunities

### Technical Components

- **Nonce Management System**: Maintains precise transaction ordering while preventing nonce collisions
- **Dynamic Gas Optimization**: Automatically adjusts gas prices based on network conditions
- **Real-time Market Analysis**: Processes market data to identify profitable trading opportunities
- **Multi-layer Safety Protocol**: Implements comprehensive transaction validation and risk assessment
- **Transaction Bundling Engine**: Optimizes efficiency by grouping multiple transactions per block

## Logging

The bot maintains detailed logs in `Logs/0xBuilder_log.txt`, including:

- Profitable transaction detection events
- Strategy execution metrics
- System errors and exceptions
- Detailed transaction results

Logging configuration can be customized in `python/main.py` through the `setup_logging()` function.

## Troubleshooting

### Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| Node Connection Failures | Verify Ethereum node status and endpoint configuration |
| API Rate Limit Exceeded | Implement request throttling or upgrade API tier |
| Insufficient Gas Balance | Maintain adequate ETH for transaction fees |
| Nonce Synchronization | Reset nonce manager or manually synchronize |
| Node Sync Status | Ensure full node synchronization before operation |

### Debug Tips

1. Enable verbose logging for detailed debugging
2. Maintain updated dependencies
3. Verify smart contract deployment on block explorers
4. Test thoroughly on testnets before mainnet deployment

## Contributing

We welcome contributions! Please review [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Contribution Process

1. Fork the repository
2. Create a feature branch
3. Follow PEP 8 style guidelines
4. Include unit tests
5. Submit pull request

## License

Licensed under the [MIT License](LICENSE). See LICENSE file for details.

## Disclaimer

**IMPORTANT**: This software is provided for educational and research purposes only. Use at your own risk.

### Risk Factors

- Trading strategies may be considered aggressive or unethical
- Cryptocurrency trading carries significant financial risk
- Smart contract interactions may contain unforeseen vulnerabilities

### Security Notice

- Protect private keys. Share them only with your dog. but never your cat! Cats cannot be trusted. 🐕✅ 🐱❌ 
- Test thoroughly with small amounts first
- Consider regulatory compliance in your jurisdiction

[logo]: 0xBuilder.png
