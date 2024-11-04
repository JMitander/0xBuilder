# 0xplorer MEV Bot

![0xplorer Logo](https://your-image-link.com/logo.png)

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

**0xplorer** is an advanced Ethereum trading bot designed for high-frequency trading and MEV (Maximal Extractable Value) opportunities. It implements strategies like front-running, back-running, sandwich attacks, and flashloan executions using Python, Geth, Remix, and AsyncWeb3.py. The bot continuously monitors the Ethereum mempool for profitable transactions and executes trades automatically.

The bot is highly configurable, allowing users to adjust parameters, strategies, and risk levels based on their preferences. It supports multiple wallets, tokens, and trading pairs, with real-time market analysis and safety checks. The bot can be run on any Ethereum-compatible network, with support for various APIs and external data sources.

**Note:** 0xplorer is a work in progress and is not production-ready. Use it at your own risk and discretion.

## Features

- **Mempool Monitoring**: Continuously monitors the Ethereum mempool for potential arbitrage and profit opportunities.
- **Strategy Execution**: Implements various strategies, including front-running, back-running, sandwich attacks, and flashloan executions.
- **Flashloan Integration**: Utilizes flashloans to maximize capital efficiency without initial capital requirements.
- **Market Analysis**: Analyzes market conditions using data from multiple APIs and external data sources.
- **Dynamic Gas Pricing**: Adjusts gas prices based on network conditions to optimize transaction inclusion and costs.
- **Nonce Management**: Manages nonces effectively to prevent transaction failures due to nonce collisions.
- **Safety Mechanisms**: Includes safety checks and validations to manage risks and ensure profitability.
- **Transaction Bundling**: Groups multiple transactions into a single block for efficiency.
- **API Integration**: Connects to various APIs for blockchain data, pricing, and market information.
- **Configurable Parameters**: Allows users to adjust parameters, strategies, and risk levels based on preferences.
- **Detailed Logging**: Provides detailed logs of bot activities, transactions, and strategies for analysis and debugging.
- **Customizable**: Supports multiple wallets, tokens, and trading pairs, with the ability to add new strategies and features.

## Project Structure

```
/0xplorer/
├── Config/
│   └── Config.py               # Configuration management
├── Core/
│   ├── 0xplorer.py             # Main bot script
│   ├── NonceManager.py         # Manages Ethereum nonces
│   ├── StrategyManager.py      # Handles trading strategies
│   └── TransactionArray.py     # Builds and sends transaction bundles
├── Utils/
│   ├── token_addresses.json    # List of monitored token addresses
│   ├── token_symbols.json      # Mapping of token addresses to symbols
│   └── erc20_signatures.json   # ERC20 function signatures
├── ABI/
│   ├── erc20_ABI.json
│   ├── aave_v3_flashloan_ABI.json
│   ├── aave_v3_lending_pool_ABI.json
│   ├── uniswap_v2_router_ABI.json
│   ├── sushiswap_router_ABI.json
│   ├── pancakeswap_router_ABI.json
│   └── balancer_router_ABI.json
├── Contracts/
│   └── SimpleFlashLoan.sol     # Flashloan smart contract
├── Analysis/
│   ├── MarketAnalyzer.py       # Analyzes market data
│   ├── MonitorArray.py         # Monitors mempool for transactions
│   └── SafetyNet.py            # Safety checks and validations
├── Contracts/
│   └── SimpleFlashLoan.sol     # Flashloan smart contract
├── Logs/
│   └── 0xplorer_log.txt        # Logs bot activities
├── .env                        # Environment variables
├── 0xplorer.py                 # All-in-one 
├── requirements.txt            # Python dependencies
├── LICENSE                     # License information
├── CONTRIBUTING.md             # Contribution guidelines
└── README.md                   # Project documentation
```

## Prerequisites

Before running 0xplorer, ensure you have the following:

### System Requirements

- **Operating System**: Ubuntu 20.04 LTS or later recommended (Windows and macOS are also supported)
- **Python Version**: Python 3.8 or higher
- **Node.js**: Required for deploying smart contracts via Truffle or Hardhat (optional)
- **Geth**: Go Ethereum client for running a full Ethereum node
- **Internet Connection**: Stable and fast internet connection

### Software Dependencies

- **Ethereum Node**: A fully synchronized execution client (e.g., Geth, Nethermind)
- **Beacon Node (for Ethereum 2.0)**: Prysm or Lighthouse
- **Python Packages**: Listed in `requirements.txt`
- **Solidity Compiler**: For compiling smart contracts (solc)

### Ethereum Node Setup

Set up an Ethereum node using one of the following clients:

| Client                                                                   | Language   | Operating Systems     | Networks                  | Sync Strategies   |
| ------------------------------------------------------------------------ | ---------- | --------------------- | ------------------------- | ----------------- |
| [Geth](https://geth.ethereum.org/)                                       | Go         | Linux, Windows, macOS | Mainnet, Sepolia, Holesky | Snap, Full        |
| [Nethermind](https://www.nethermind.io/)                                 | C#, .NET   | Linux, Windows, macOS | Mainnet, Sepolia, Holesky | Snap, Fast, Full  |
| [Besu](https://besu.hyperledger.org/en/stable/)                          | Java       | Linux, Windows, macOS | Mainnet, Sepolia, Holesky | Snap, Fast, Full  |
| [Erigon](https://github.com/ledgerwatch/erigon)                          | Go         | Linux, Windows, macOS | Mainnet, Sepolia, Holesky | Full              |
| [Reth](https://reth.rs/)                                                 | Rust       | Linux, Windows, macOS | Mainnet, Sepolia, Holesky | Full              |
| [EthereumJS](https://github.com/ethereumjs/ethereumjs-monorepo) _(beta)_ | TypeScript | Linux, Windows, macOS | Sepolia, Holesky          | Full              |

#### Setting Up Geth

1. **Install Geth**:

   For detailed instructions, refer to the [Geth installation guide](https://geth.ethereum.org/docs/install-and-build/installing-geth).

2. **Start the Geth Node**:

   ```bash
   geth --mainnet --syncmode "snap" --http --http.api eth,net,web3,txpool --ws --ws.api eth,net,web3,txpool --ipcpath /path/to/geth.ipc --maxpeers 100 --http.corsdomain "*" --cache 4096
   ```

3. **Verify Synchronization**:

   Attach to Geth:

   ```bash
   geth attach ipc:/path/to/geth.ipc
   ```

   Check sync status:

   ```javascript
   eth.syncing
   ```

   Wait until synchronization is complete before running the bot.

#### Setting Up a Beacon Node

For Ethereum 2.0 interactions, set up a beacon node using Prysm or Lighthouse.

- **Prysm**: [Prysm Installation Guide](https://docs.prylabs.network/docs/getting-started)
- **Lighthouse**: [Lighthouse Installation Guide](https://lighthouse-book.sigmaprime.io/installation.html)

### API Keys and Providers

- **Wallet Address**: An Ethereum wallet with sufficient funds for trading and gas fees.
- **Private Key**: For signing transactions and interacting with the Ethereum network.
- **API Providers**: Register and obtain API keys from:

  - [Infura](https://infura.io/register)
  - [Etherscan](https://etherscan.io/register)
  - [CoinGecko](https://www.coingecko.com/en/api)
  - [CoinMarketCap](https://coinmarketcap.com/api/)
  - [CryptoCompare](https://min-api.cryptocompare.com/)

## Installation

### Cloning the Repository

```bash
git clone https://github.com/yourusername/0xplorer.git
cd 0xplorer
```

### Setting up Virtual Environment

It's recommended to use a virtual environment to manage dependencies.

For Linux/MacOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

For Windows:

```cmd
python -m venv venv
venv\Scripts\activate
```

### Installing Dependencies

```bash
pip install -r requirements.txt
```

Ensure that all packages are installed successfully.

## Configuration

### Environment Variables

Create a `.env` file in the root directory of the project to store your environment variables securely.

```bash
cp .env-example .env
```

Edit the `.env` file and provide the required values:

```ini
# ================================ API Configuration ================================ #
ETHERSCAN_API_KEY=your_etherscan_api_key
INFURA_PROJECT_ID=your_infura_project_id
COINGECKO_API_KEY=your_coingecko_api_key
COINMARKETCAP_API_KEY=your_coinmarketcap_api_key
CRYPTOCOMPARE_API_KEY=your_cryptocompare_api_key

# ================================ Ethereum Node Configuration ================================ #
HTTP_ENDPOINT=http://127.0.0.1:8545
WEB3_ENDPOINT=wss://127.0.0.1:8546
WEBSOCKET_ENDPOINT=wss://127.0.0.1:8546
IPC_ENDPOINT=/path/to/geth.ipc

# ================================ Wallet Configuration ================================ #
WALLET_KEY=your_private_key
WALLET_ADDRESS=0xYourWalletAddress
PROFIT_ADDRESS=0xYourProfitAddress

# ================================ Token Configuration ================================ #
TOKEN_ADDRESSES=Utils/token_addresses.json
TOKEN_SYMBOLS=Utils/token_symbols.json

# ============================ Uniswap V2 ============================== #
UNISWAP_V2_ROUTER_ADDRESS=0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D
UNISWAP_V2_ROUTER_ABI=ABI/uniswap_v2_router_ABI.json

# ============================ Sushiswap ============================== #
SUSHISWAP_ROUTER_ADDRESS=0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F
SUSHISWAP_ROUTER_ABI=ABI/sushiswap_router_ABI.json

# ============================ PancakeSwap ============================== #
PANCAKESWAP_ROUTER_ADDRESS=0xEfF92A263d31888d860bD50809A8D171709b7b1c
PANCAKESWAP_ROUTER_ABI=ABI/pancakeswap_router_ABI.json

# ============================ Balancer ============================== #
BALANCER_ROUTER_ADDRESS=0x3E66B66Fd1d0b02fDa6C811da9E0547970DB2f21
BALANCER_ROUTER_ABI=ABI/balancer_router_ABI.json

# ============================ ERC20 ================================ #
ERC20_ABI=ABI/erc20_ABI.json
ERC20_SIGNATURES=Utils/erc20_signatures.json

# ================================ Flashloan Configuration ================================ #
AAVE_V3_FLASHLOAN_CONTRACT_ADDRESS=0xYourFlashloanContractAddress
AAVE_V3_FLASHLOAN_ABI=ABI/aave_v3_flashloan_ABI.json
AAVE_V3_LENDING_POOL_ADDRESS=0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2
AAVE_V3_LENDING_POOL_ABI=ABI/aave_v3_lending_pool_ABI.json
```

### Configuration Files

Ensure that the following JSON configuration files are present in the `Utils` directory:

- `token_addresses.json`: List of token contract addresses to monitor.
- `token_symbols.json`: Mapping of token addresses to their symbols.
- `erc20_signatures.json`: List of ERC20 function signatures to monitor.

## Deploying the Flashloan Contract

To utilize flashloans, you need to deploy a flashloan contract compatible with Aave V3 or another protocol of your choice.

### Steps to Deploy the Contract

#### Option 1: Using Remix IDE

1. **Open Remix IDE**: [Remix IDE](https://remix.ethereum.org/)
2. **Create a New File**: Name it `SimpleFlashLoan.sol`
3. **Implement Your Flashloan Logic**: Refer to Aave's documentation and examples to implement your custom flashloan contract.
4. **Compile the Contract**:
   - Select the appropriate Solidity compiler version.
   - Click on the "Compile" button.
   - Ensure there are no errors during compilation.
5. **Deploy the Contract**:
   - Choose "Injected Web3" as the environment to connect via MetaMask.
   - Provide any required constructor arguments.
   - Deploy the contract and confirm the transaction in MetaMask.
6. **Save Contract Address**:
   - After deployment, note the contract address.
   - Update the `AAVE_V3_FLASHLOAN_CONTRACT_ADDRESS` in your `.env` file.

#### Option 2: Using Truffle or Hardhat

1. **Install Truffle or Hardhat**:
   - Truffle: [Installation Guide](https://www.trufflesuite.com/docs/truffle/getting-started/installation)
   - Hardhat: [Installation Guide](https://hardhat.org/getting-started/)
2. **Compile the Smart Contract**:
   - Place your flashloan smart contract in the `Contracts` directory.
   - Compile the contract using `truffle compile` or `npx hardhat compile`.
3. **Deploy the Contract**:
   - Update the deployment script with your private key and network details.
   - Deploy the contract using `truffle migrate` or `npx hardhat run scripts/deploy.js`.
4. **Update Configuration**:
   - After deployment, update the `AAVE_V3_FLASHLOAN_CONTRACT_ADDRESS` in your `.env` file.

### Integrate with 0xplorer

Ensure that the ABI files are correctly placed in the `ABI` directory and that the paths in your `.env` file are accurate.

## Obtaining API Keys

To access data from various APIs, you need to register and obtain API keys.

### Etherscan API Key

Register at [Etherscan](https://etherscan.io/apis) and obtain an API key.

### Infura Project ID

Register at [Infura](https://infura.io/register) and create a project to get the Project ID.

### CoinGecko API Key

Visit [CoinGecko API](https://www.coingecko.com/en/api) to obtain an API key.

### CoinMarketCap API Key

Register at [CoinMarketCap](https://pro.coinmarketcap.com/signup) for an API key.

### CryptoCompare API Key

Register at [CryptoCompare](https://www.cryptocompare.com/cryptopian/api-keys) to get an API key.

### Updating the `.env` File

Add all your API keys to the `.env` file:

```ini
ETHERSCAN_API_KEY=your_etherscan_api_key
INFURA_PROJECT_ID=your_infura_project_id
COINGECKO_API_KEY=your_coingecko_api_key
COINMARKETCAP_API_KEY=your_coinmarketcap_api_key
CRYPTOCOMPARE_API_KEY=your_cryptocompare_api_key
```

## Running the Bot

### Start Your Ethereum Node

Ensure that your Geth node and beacon node are running and fully synchronized.

### Activate Virtual Environment

```bash
source venv/bin/activate
```

### Run the Bot

```bash
python Core/0xplorer.py
```

Replace `Core/0xplorer.py` with the path to your main bot script if different.

### Monitoring

- **Logs**: Check `Logs/0xplorer_log.txt` for detailed logs.
- **Console Output**: The bot will output important information and statuses to the console.

### Stopping the Bot

Press `Ctrl+C` to safely stop the bot. It will finish the current operation and shut down gracefully.

## Strategies

0xplorer employs several advanced trading strategies to exploit profitable opportunities on the Ethereum network:

- **Front-Running**: Places a higher-priority transaction before a detected one.
- **Back-Running**: Executes a transaction immediately after a profitable one.
- **Sandwich Attacks**: Combines front-running and back-running around a target transaction.
- **Flashloans**: Utilizes borrowed assets for arbitrage without initial capital.
- **Nonce Management**: Ensures nonces are correctly ordered to avoid collisions.
- **Dynamic Gas Pricing**: Adjusts gas prices based on real-time network conditions.
- **Market Analysis**: Analyzes market data for profitable opportunities.
- **Safety Checks**: Validates transactions and ensures they meet predefined criteria.
- **Transaction Bundling**: Groups multiple transactions into a single block for efficiency.

## Logging

Logs are stored in `Logs/0xplorer_log.txt`. They include:

- Detected profitable transactions
- Strategy execution details
- Errors and exceptions
- Transaction details and results

Configure logging in `Core/0xplorer.py` within the `setup_logging()` function.

## Troubleshooting

### Common Issues

- **Connection Errors**: Ensure your Ethereum node is running and accessible via the endpoints specified.
- **API Rate Limits**: Be mindful of API rate limits; consider implementing rate limiting in your code.
- **Insufficient Funds**: Ensure your wallet has enough ETH to cover gas fees.
- **Invalid Nonce**: If you encounter nonce errors, reset the nonce manager or synchronize nonces.
- **Synchronization Issues**: Wait for your Ethereum node to be fully synchronized before running the bot.

### Tips

- **Verbose Logging**: Increase logging verbosity for debugging.
- **Check Dependencies**: Ensure all Python packages are up-to-date.
- **Smart Contract Verification**: Verify your flashloan contract on Etherscan for transparency.
- **Testnet Testing**: Test the bot on a testnet (e.g., Ropsten, Rinkeby) before deploying to mainnet.

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

- **Report Issues**: Use the GitHub issue tracker for bugs and feature requests.
- **Pull Requests**: Submit pull requests for improvements or fixes.
- **Code Style**: Follow PEP 8 guidelines and ensure code passes linting.
- **Testing**: Write unit tests for new features or bug fixes.

## License

This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute the code with proper attribution.

## Disclaimer

**Warning**: This software is provided "as is" and is intended for educational and research purposes only. Use it at your own risk. The developers are not responsible for any financial losses or legal issues that may arise from using this software.

- **Ethical Considerations**: strategies used by this bot may be considered unethical. Use at your own discretion.
- **Security**: Safeguard your keys and do not share them, especially with your girlfriend. Sharing with your dog might be fine, but not your cat. cats are sneaky and can't be trusted 🐱


- **Risk of Loss**: Trading cryptocurrencies involves significant risk. Only trade with funds you can afford to lose.
