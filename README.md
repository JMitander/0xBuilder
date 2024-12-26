# 0xBuilder




[![License](https://img.shields.io/badge/license-MIT-white.svg)](LICENSE)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-white.svg)](CONTRIBUTING.md)

[![Python Version](https://img.shields.io/badge/Python-3.12.*-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Node.js Version](https://img.shields.io/badge/Node.js-18.*-blue.svg)](https://nodejs.org/en/download/)
[![Solidity Version](https://img.shields.io/badge/Solidity-v0.8.*-blue.svg)](https://docs.soliditylang.org/en/v0.8.19/)
[![Geth](https://img.shields.io/badge/Geth-v1.14.*-blue.svg)](https://geth.ethereum.org/)
[![Remix](https://img.shields.io/badge/Remix-IDE-blue.svg)](https://remix.ethereum.org/) 

[![AAVE](https://img.shields.io/badge/Aave-v3-orange.svg)](https://aave.com/)
[![Uniswap](https://img.shields.io/badge/Uniswap-v2.0-orange.svg)](https://uniswap.org/)
[![Sushiswap](https://img.shields.io/badge/Sushiswap-v2-orange.svg)](https://sushiswap_router_abi.fi/)


[![CoinGecko](https://img.shields.io/badge/CoinGecko-API-red.svg)](https://www.coingecko.com/en/api)
[![CoinMarketCap](https://img.shields.io/badge/CoinMarketCap-API-red.svg)](https://coinmarketcap.com/api/)
[![CryptoCompare](https://img.shields.io/badge/CryptoCompare-API-red.svg)](https://min-api.cryptocompare.com/)
[![Etherscan](https://img.shields.io/badge/Etherscan-API-red.svg)](https://etherscan.io/apis)

## Overview

0xBuilder is a sophisticated MEV (Miner Extractable Value) bot designed to automatically identify and exploit profitable opportunities on the Ethereum blockchain. It's engineered for high performance, utilizing asynchronous programming and robust error handling. The bot implements advanced trading strategies such as front-running, back-running, and sandwich attacks, while incorporating a sophisticated risk management system to protect against potential losses. Additionally, it uses linear regression for price predictions and integrates with various decentralized exchanges.

## Key Features

*   **Mempool Monitoring:** Real-time scanning of the Ethereum mempool for pending transactions.
*   **DeFi Protocol Integration:** Seamless interaction with Aave V3 for flash loans, and Uniswap/Sushiswap for DEX trading.
*   **Advanced Strategies:** Implementation of front-running, back-running, and sandwich attack strategies.
*   **Dynamic Gas Optimization:** Automatically adjusts gas prices for timely inclusion and maximal profit.
*   **Risk Management:** Utilizes a "safety net" to prevent excessive losses, including transaction simulation and dynamic slippage adjustments.
*  **Machine Learning:** Uses a linear regression model for price prediction.
*   **Asynchronous Programming:** Leverages `asyncio` for high performance and concurrent operation.
*   **Modular Design:** Well-organized code into core components for maintainability and scalability.
*   **Comprehensive Logging:** Detailed logging for all operations and transactions.

## Project Structure

```
0xBuilder/
├── abi/
│   ├── aave_flashloan_abi.json
│   ├── aave_lending_pool_abi.json
│   ├── balancer_router_abi.json
│   ├── erc20_abi.json
│   ├── pancakeswap_router_abi.json
│   ├── sushiswap_router_abi.json
│   └── uniswap_router_abi.json
├── python/
│   ├── abi_registry.py
│   ├── configuration.py
│   ├── constants.py
│   ├── core.py
│   ├── main.py
│   ├── monitor.py
│   ├── net.py
│   ├── nonce.py
│   └── __init__.py
├── utils/Python/
│   ├── strategyexecutionerror.py
│   ├── strategyconfiguration.py
│   ├── colorformatter.py
│   └── __init__.py
├── linear_regression/
│   ├── training_data.csv
│   └── price_model.joblib
└── README.md
```

*   **`/abi/`:** Contains JSON files defining the Application Binary Interface (ABI) for interacting with various smart contracts.
*   **`/python/`:** Holds the main Python source code, including modules for ABI handling, configuration, core transaction logic, mempool monitoring, safety checks, and strategy implementation.
*   **`/utils/Python/`:** Contains utility classes, exceptions, and configurations.
*   **`/linear_regression/`:** Includes files related to the price prediction model, including training data and model file.
*   **`README.md`:** The current documentation.

## Setup Instructions

### Prerequisites

*   **Python 3.9+:** Ensure you have Python 3.9 or higher installed.
*   **pip:** Python package installer.
*   **Ethereum Node:** You need an Ethereum node client (Geth, Nethermind, etc.). We will provide a guide for Geth.
*   **Git:** Version control system.

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/0xBuilder.git
cd 0xBuilder
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # For Linux/macOS
venv\Scripts\activate  # For Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment Configuration

Create a `.env` file in the root directory and populate it with your environment variables. Here is a template:

```env
# Ethereum Node Endpoints (Choose One)
IPC_ENDPOINT= # Path to your geth.ipc file
HTTP_ENDPOINT= # Your http endpoint
WEBSOCKET_ENDPOINT= # Your Websocket endpoint

# Wallet Configuration
WALLET_KEY= # Your Private Key
WALLET_ADDRESS= # Your public address

# API Keys for Various Services
ETHERSCAN_API_KEY= # Your Etherscan API Key
INFURA_PROJECT_ID= # Your Infura API KEY
COINGECKO_API_KEY= # Your Coingecko API Key
COINMARKETCAP_API_KEY= # Your CoinMarketCap API key
CRYPTOCOMPARE_API_KEY= # Your CryptoCompare API key
BINANCE_API_KEY= # Your Binance API Key

# Aave Addresses
AAVE_FLASHLOAN_ADDRESS= # Aave V3 Flashloan contract Address
AAVE_LENDING_POOL_ADDRESS= # Aave V3 Lending Pool contract Address

# DEX Router Addresses
UNISWAP_ROUTER_ADDRESS= # Uniswap v3 router address
SUSHISWAP_ROUTER_ADDRESS= # Sushiswap router address

# JSON File Paths
TOKEN_ADDRESSES= # Path to your token_addresses.json file
TOKEN_SYMBOLS= # Path to your token_symbols.json
ERC20_SIGNATURES= # Path to your erc20_signatures.json
```

**Note:**

*   Replace the `#` comments with your actual values.
*   You need exactly one endpoint (IPC, HTTP or WebSocket) configured to connect to the Ethereum network.
*   Make sure the keys and addresses are valid.
*   See below for instructions on how to create the necessary accounts and API keys.

### 5. Run the Bot

```bash
python python/main.py
```

## Additional Documentation

### 1. Setting Up a Geth Node

1.  **Download Geth:** Get the latest version of Geth from the [official Ethereum website](https://geth.ethereum.org/downloads/).
2.  **Initialize the Node:**

    ```bash
    geth --datadir ~/.ethereum init <genesis-file> # replace genesis-file with the path to your genesis.json
    ```

    *(If you connect to the Mainnet or a testnet you won't need the `--datadir` and `init` commands)*
3.  **Run Geth:**

    *   **Mainnet:**
        ```bash
        geth --mainnet --http --http.api eth,net,admin,engine,txpool,web3 --ws --ws.api eth,net,admin,engine,txpool,web3 --ws.origins="*" --syncmode=snap --ipcpath= /PATH/TO/geth.ipc
        ```
    *   **Testnet (Goerli):**
        ```bash
         geth --goerli --http --http.api eth,net,admin,engine,txpool,web3 --ws --ws.api eth,net,admin,engine,txpool,web3 --ws.origins="*" --syncmode=snap --ipcpath= /PATH/TO/geth.ipc
        ```
4.  **IPC Endpoint:** Find your IPC endpoint path in the data directory (e.g., `~/.ethereum/geth.ipc`). This path will be needed in your `.env` file.

### 2. Setting Up a Prysm Beacon Node

If you're planning on using a PoS network, or you'd like to run a full node including a beacon node please follow the following instructions:

1.  **Download Prysm:** Get the latest Prysm release from the [official Github repository](https://github.com/prysmaticlabs/prysm/releases).
2.  **Run the Beacon Node:**

### 3. Deploying Aave V3 Flashloan Contract with Remix

1.  **Open Remix:** Go to [Remix IDE](https://remix.ethereum.org).
2.  **Create a New File:** Create a new file (e.g., `AaveFlashloan.sol`).
3.  **Paste the Contract Code:** Copy and paste the Aave V3 Flashloan contract code into the editor.
    *(You can use the Aave v3 Flashloan Example implementation, or you can copy the `aave_flashloan.sol` from the `test` directory of the repository)*
4.  **Compile:** Compile the contract in the Solidity compiler tab.
5.  **Deploy:** In the Deploy & Run Transactions tab, select `Injected Provider` as the environment and choose the account, and deploy the contract.
   *   You will need to specify the address of the `IPoolAddressesProvider` as the contract constructor argument. You can get this address from Aave documentation for each network.
6.  **Get the Contract Address:** Once deployed, copy the contract address. You'll need this for your `.env` file.

**Note:** This is a general guide; always refer to Aave's official documentation for the latest information.

### 4. Obtaining API Keys

1.  **Etherscan API Key:**
    *   Go to [Etherscan](https://etherscan.io/) and register an account.
    *   Find the API Key section in your account settings and create one.
2.  **Infura Project ID:**
    *   Go to [Infura](https://infura.io/) and create an account.
    *   Create a new project and copy the Project ID.
3. **CoinGecko API Key:**
    *   Go to [CoinGecko](https://www.coingecko.com/en/api) and obtain an API key for their service.
    *   Follow the instructions to get the free API plan and its Key.
4.  **CoinMarketCap API Key:**
    *   Go to [CoinMarketCap](https://coinmarketcap.com/api/) and register for an API key.
5.  **CryptoCompare API Key:**
    *   Go to [CryptoCompare](https://min-api.cryptocompare.com/documentation) and create an account. Get your API key.
6.  **Binance API Key:**
    *   Go to [Binance](https://www.binance.com/en/my/settings/api-management) and create an API key.

**Note:**
*    Treat your API keys as passwords. Do not commit or share them with anyone.
*   API keys are often rate-limited, pay attention to their usage.

## Running The Bot

1.  **Set up environment variables**.
2.  **Activate the venv**.
3.  **Execute** `python python/main.py`.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss potential improvements and bug fixes.

## License

This project is licensed under the MIT License.
