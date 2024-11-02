# 0xplorer Bot

0xplorer is an advanced Ethereum trading bot designed for high-frequency trading and MEV (Maximal Extractable Value) opportunities. It implements strategies like front-running, back-running, sandwich attacks, and flashloan executions using Python, Geth, Remix, and AsyncWeb3.py. The bot continuously monitors the Ethereum mempool for profitable transactions and executes trades automatically.

The bot is highly configurable, allowing users to adjust parameters, strategies, and risk levels based on their preferences. It supports multiple wallets, tokens, and trading pairs, with real-time market analysis and safety checks. The bot can be run on any Ethereum-compatible network, with support for various APIs and external data sources.

**Note:** 0xplorer is a work in progress and is not production-ready.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [API Setup](#api-setup)
- [Ethereum Node Setup](#ethereum-node-setup)
- [Configuration](#configuration)
- [Creating and Deploying the Flash Loan Contract](#creating-and-deploying-the-flash-loan-contract)
- [Usage](#usage)
- [Strategies](#strategies)
- [Logging](#logging)
- [License](#license)

## Overview

0xplorer leverages various advanced trading strategies to exploit profitable opportunities on the Ethereum network:

- **Flashloans**: Borrow assets temporarily to execute arbitrage opportunities without initial capital.
- **Front-running**: Place trades before detected profitable transactions.
- **Back-running**: Execute trades immediately after profitable transactions.
- **Sandwich Attacks**: Surround a trade with your own transactions to capture profits.
- **Nonce Management**: Avoid nonce collisions when sending multiple transactions.
- **Dynamic Gas Pricing**: Optimize gas prices in real-time using external APIs.
- **Market Analysis**: Analyze market data and trends for profitable opportunities.
- **Safety Checks**: Validate transactions and ensure they meet predefined criteria.
- **Transaction Bundling**: Group multiple transactions into a single block for efficiency.
- **API Integration**: Connect to various APIs for blockchain data, pricing, and market information.

## Project Structure

![Project Structure](image.png)

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
│   └── token_symbols.json      # Mapping of token addresses to symbols
├── ABI/
│   ├── erc20_ABI.json
│   ├── aave_v3_flashloan_ABI.json
│   ├── aave_v3_lending_pool_ABI.json
│   ├── uniswap_v2_router_ABI.json
│   ├── sushiswap_router_ABI.json
│   ├── pancakeswap_router_ABI.json
│   └── balancer_router_ABI.json
├── Analysis/
│   ├── MarketAnalyzer.py       # Analyzes market data
│   ├── MonitorArray.py         # Monitors mempool for transactions
│   └── SafetyNet.py            # Safety checks and validations
├── Contracts/
│   └── SimpleFlashLoan.sol     # Flash loan smart contract
├── Logs/
│   └── 0xplorer.log            # Logs bot activities
├── .env                        # Environment variables
├── requirements.txt            # Python dependencies
├── LICENSE.md                  # License information
└── README.md                   # Project documentation
```

## Prerequisites

Before running 0xplorer, ensure you have the following:

- **Python 3.x**: Programming language used for the bot.
- **Ethereum Node**: A fully synchronized execution client.
- **API keys**: For Infura, Etherscan, CoinGecko, CoinMarketCap, and CryptoCompare.
- **Wallet Address**: With sufficient funds for trading and gas fees.
- **Private Key**: For signing transactions and interacting with the Ethereum network.

| Client                                                                   | Language   | Operating Systems     | Networks                  | Sync Strategies                                                |
| ------------------------------------------------------------------------ | ---------- | --------------------- | ------------------------- | -------------------------------------------------------------- |
| [Geth](https://geth.ethereum.org/)                                       | Go         | Linux, Windows, macOS | Mainnet, Sepolia, Holesky | Snap, Full                                                     |
| [Nethermind](https://www.nethermind.io/)                                 | C#, .NET   | Linux, Windows, macOS | Mainnet, Sepolia, Holesky | Snap (without serving), Fast, Full                             |
| [Besu](https://besu.hyperledger.org/en/stable/)                          | Java       | Linux, Windows, macOS | Mainnet, Sepolia, Holesky | Snap, Fast, Full                                               |
| [Erigon](https://github.com/ledgerwatch/erigon)                          | Go         | Linux, Windows, macOS | Mainnet, Sepolia, Holesky | Full                                                           |
| [Reth](https://reth.rs/)                                                 | Rust       | Linux, Windows, macOS | Mainnet, Sepolia, Holesky | Full                                                           |
| [EthereumJS](https://github.com/ethereumjs/ethereumjs-monorepo) _(beta)_ | TypeScript | Linux, Windows, macOS | Sepolia, Holesky          | Full                                                           |

- **API Providers**: Register and obtain API keys from:
   - [Infura](https://infura.io/register)
   - [Etherscan](https://etherscan.io/register)
   - [CoinGecko](https://www.coingecko.com/en/api)
   - [CoinMarketCap](https://coinmarketcap.com/api/)
   - [CryptoCompare](https://min-api.cryptocompare.com/)

- **Remix IDE**: Browser-based IDE for Solidity smart contracts (optional)
- **Node.js**: Open-source JavaScript runtime environment (optional)

## Installation

1. **Clone the Repository**

    ```sh
    git clone https://github.com/JMitander/0xplorer.git
    cd 0xplorer
    ```

2. **Create a Virtual Environment**

    For Linux/MacOS:

    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```

    For Windows:

    ```cmd
    python -m venv venv
    venv\Scripts\activate
    ```

3. **Install Dependencies**

    ```sh
    pip install -r requirements.txt
    ```

4. **Copy and Configure `.env` File**

    ```sh
    cp .env-example .env
    ```

    Edit the `.env` file with your configuration:
    - Ethereum node connection details
    - Wallet address and private key
    - API keys for Infura, Etherscan, CoinGecko, etc.

## API Setup

Follow the instructions provided by each API provider to obtain your API keys.

## Ethereum Node Setup

Set up an Ethereum node using one of the following clients: Geth, Nethermind, Besu, Erigon, Reth, or EthereumJS. Refer to the [Ethereum Nodes and Clients](https://ethereum.org/en/developers/docs/nodes-and-clients/) documentation for detailed instructions.

In this guide, we'll use Geth as the execution client.

1. **Install Geth**

    Follow the [Geth installation guide](https://geth.ethereum.org/docs/install-and-build/installing-geth) for your operating system.

2. **Start the Geth Node**

    ```sh
    geth --mainnet --syncmode "snap" --http --http.api eth,net,web3,txpool --ws --ws.api eth,net,web3,txpool --ipcpath /path/to/geth.ipc --maxpeers 100 --http.corsdomain "*" --cache 4096
    ```

3. **Verify Synchronization**

    Attach to Geth:

    ```sh
    geth attach ipc:/path/to/geth.ipc
    ```

    Check sync status:

    ```javascript
    eth.syncing
    ```

    Wait until synchronization is complete before running the bot.

## Configuration

Ensure all environment variables in the `.env` file are correctly set.

### Example `.env` File

```sh
# ================================ API Configuration ================================ #
ETHERSCAN_API_KEY=YourEtherscanAPIKey
INFURA_PROJECT_ID=YourInfuraProjectID
COINGECKO_API_KEY=YourCoinGeckoAPIKey
COINMARKETCAP_API_KEY=YourCoinMarketCapAPIKey
CRYPTOCOMPARE_API_KEY=YourCryptoCompareAPIKey

# ================================ Ethereum Node Configuration ================================ #
HTTP_ENDPOINT=http://127.0.0.1:8545
WEB3_ENDPOINT=wss://127.0.0.1:8546
WEBSOCKET_ENDPOINT=wss://127.0.0.1:8546
IPC_ENDPOINT=/path/to/geth.ipc

# ================================ Wallet Configuration ================================ #

WALLET_KEY=YourWalletPrivateKey
WALLET_ADDRESS=0xYourWalletAddress
PROFIT_ADDRESS=0xYourProfitAddress

# ================================ Token Configuration ================================ #

TOKEN_ADDRESSES=/Your/directory/0xplorer/Utils/token_addresses.json
TOKEN_SYMBOLS=/Your/directory/0xplorer/Utils/token_symbols.json

# ============================ UNISWAP V2 ============================== #

UNISWAP_V2_ROUTER_ADDRESS=0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D
UNISWAP_V2_ROUTER_ABI=/Your/directory/0xplorer/ABI/uniswap_v2_router_ABI.json

# ============================ SUSHISWAP ==============================

SUSHISWAP_ROUTER_ADDRESS=0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F 
SUSHISWAP_ROUTER_ABI=/Your/directory/0xplorer/ABI/sushiswap_router_ABI.json

# ============================ PANCAKESWAP ============================== #

PANCAKESWAP_ROUTER_ADDRESS=0xEfF92A263d31888d860bD50809A8D171709b7b1c
PANCAKESWAP_ROUTER_ABI=/Your/directory/0xplorer/ABI/pancakeswap_router_ABI.json

# ============================ BALANCER ============================== #

BALANCER_ROUTER_ADDRESS=0x3E66B66Fd1d0b02fDa6C811da9E0547970DB2f21
BALANCER_ROUTER_ABI=/Your/directory/0xplorer/ABI/balancer_router_ABI.json

# ============================ ERC20 ================================ #

ERC20_ABI=/Your/directory/0xplorer/ABI/erc20_ABI.json
ERC20_SIGNATURES=/Your/directory/0xplorer/Utils/erc20_signatures.json

# ================================ FLASHLOAN Configuration ================================ #

AAVE_V3_FLASHLOAN_CONTRACT_ADDRESS=YourDeployedFlashloanContractAddress
AAVE_V3_FLASHLOAN_CONTRACT_ABI=/Your/directory/0xplorer/ABI/aave_v3_flashloan_contract_ABI.json
AAVE_V3_LENDING_POOL_ADDRESS=0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2 
AAVE_V3_LENDING_POOL_ABI=/Your/directory/0xplorer/ABI/aave_v3_lending_pool_ABI.json
```

## Creating and Deploying the Flash Loan Contract

To execute flashloan strategies, you need to create and deploy a flashloan smart contract. Below are the steps to create and deploy a flashloan contract using Aave V3 on the Ethereum network.

For the latest implementation of flashloans, please refer to the [QuickNode Flashloan Guide](https://www.quicknode.com/guides/defi/lending-protocols/how-to-make-a-flash-loan-using-aave).

### Step 1: Create the Flash Loan Contract

1. **Open Remix IDE**: [Remix IDE](https://remix.ethereum.org/)
2. **Create a New File**: Name it `SimpleFlashLoan.sol`
3. **Implement Your Flashloan Logic**: Refer to Aave's documentation and examples to implement your custom flashloan contract.

### Step 2: Compile the Contract

1. **Select Compiler Version**: Choose the appropriate Solidity compiler version.
2. **Compile**: Click on the "Compile" button.
3. **Resolve Warnings/Errors**: Ensure there are no errors during compilation.

### Step 3: Deploy the Contract

1. **Set Environment**: Choose "Injected Web3" to connect via MetaMask.
2. **Provide Constructor Arguments**: Input required parameters.
3. **Deploy**: Confirm the deployment in MetaMask.
4. **Save Contract Address**: Note the contract address for configuration.

### Step 4: Integrate with 0xplorer

Configure 0xplorer to interact with your deployed flashloan contract by updating the `.env` file with the contract address and ABI path.

## Usage

### Running the Bot

Start 0xplorer:

```sh
python3 Core/0xplorer.py
```

### Stopping the Bot

Press `Ctrl+C` to safely stop the bot. It will finish the current operation and shut down gracefully.

## Strategies

0xplorer employs several strategies:

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

Logs are stored in `Logs/0xplorer.log`. They include:

- Detected profitable transactions
- Strategy execution details
- Errors and exceptions
- Transaction details and results

Configure logging in `Core/0xplorer.py` within the `setup_logging()` function.

### Warning

0xplorer is a work in progress and should be used with caution. It is recommended to test the bot on a testnet before running it on the mainnet. It's not production-ready.

We welcome contributions to this project. Feel free to fork and submit a pull request. Open an issue if you have any questions or suggestions.

## License

This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute the code with proper attribution.

