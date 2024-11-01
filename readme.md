# 0xplorer Bot

0xplorer is an advanced Ethereum trading bot designed for high-frequency trading and MEV (Miner Extractable Value) opportunities. It implements strategies like front-running, back-running, sandwich attacks, and flashloan executions using Python, Geth, Remix, and AsyncWeb3.py. The bot continuously monitors the Ethereum mempool for profitable transactions and executes trades automatically.

the bot is highly configurable, allowing users to adjust parameters, strategies, and risk levels based on their preferences. It supports multiple wallets, tokens, and trading pairs, with real-time market analysis and safety checks. The bot can be run on any Ethereum-compatible network, with support for various APIs and external data sources.

Note that 0xplorer is a work in progress and nowhere near production-ready.


## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [API Setup](#api-setup)
- [Geth Node Setup](#geth-node-setup)
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

The bot is designed to be highly configurable, allowing users to adjust parameters, strategies, and risk levels based on their preferences. It can be run on any Ethereum-compatible network, with support for multiple wallets, tokens, and trading pairs.

## Project Structure

![alt text](image.png)

```
/0xplorer/
├── Config/
│   └── Config.py               # Configuration management
├── Core/
│   ├── 0xplorer.py             # Main
│   ├── NonceManager.py         # Manages Ethereum nonces
│   ├── StrategyManager.py      # Handles trading strategies
│   └── TransactionArray.py     # Builds and sends transaction bundles
├── Utils/
|   ├── token_adresses.json     # List of monitored token addresses
|   └── token_symbols.json      # List of monitored token addresses 
├── ABI
|   ├── erc20_ABI.json
|   ├── aave_v3_flashloan_ABI.json
|   ├── aave_v3_lending_pool_ABI.sjon
|   ├── sushiswap_router_ABI.json
|   ├── pancakeswap_router_ABI.json
|   └── balancer_router_ABI.json
├── Analysis/
│   ├── MarketAnalyzer.py       # Analyzes market data
│   ├── MonitorArray.py         # Monitors mempool for transactions
│   └── SafetyNet.py            # Safety checks and validations
├── Contracts/
│   └── SimpleFlashLoan.sol     # Flash loan smart contract
├── Logs/
│   └── 0xplorer.log            # Logs bot activities
├── .env                        # environment variables
├── requirements.txt            # Python dependencies
├── License.md                  # License information
└── README.md                   # Project documentation
```

## Prerequisites

Before running 0xplorer, ensure you have the following:

- **Python 3.x**: Programming language used for the bot.
- **Ethereum Node**: A fully synchronized execution client.
- **API keys**: For Infura, Etherscan, CoinGecko, CoinMarketCap, and CryptoCompare.
- **Wallet Address**: With sufficient funds for trading and gas fees.
- **Private Key**: For signing transactions and interacting with the Ethereum network.

| Client                                                                   | Language   | Operating systems     | Networks                  | Sync strategies                                                |
| ------------------------------------------------------------------------ | ---------- | --------------------- | ------------------------- | -------------------------------------------------------------- |
| [Geth](https://geth.ethereum.org/)                                       | Go         | Linux, Windows, macOS | Mainnet, Sepolia, Holesky | [Snap](#snap-sync), [Full](#full-sync)                         |
| [Nethermind](https://www.nethermind.io/)                                 | C#, .NET   | Linux, Windows, macOS | Mainnet, Sepolia, Holesky | [Snap](#snap-sync) (without serving), Fast, [Full](#full-sync) |
| [Besu](https://besu.hyperledger.org/en/stable/)                          | Java       | Linux, Windows, macOS | Mainnet, Sepolia, Holesky | [Snap](#snap-sync), [Fast](#fast-sync), [Full](#full-sync)     |
| [Erigon](https://github.com/ledgerwatch/erigon)                          | Go         | Linux, Windows, macOS | Mainnet, Sepolia, Holesky | [Full](#full-sync)                                             |
| [Reth](https://reth.rs/)                                                 | Rust       | Linux, Windows, macOS | Mainnet, Sepolia, Holesky | [Full](#full-sync)                                             |
| [EthereumJS](https://github.com/ethereumjs/ethereumjs-monorepo) _(beta)_ | TypeScript | Linux, Windows, macOS | Sepolia, Holesky          | [Full](#full-sync)                                             |

- **API Providers**: Register and obtain API keys from:
  - [Infura](https://infura.io/register)
  - [Etherscan](https://etherscan.io/register)
  - [CoinGecko](https://www.coingecko.com/en/api)
  - [CoinMarketCap](https://coinmarketcap.com/api/)
  - [CryptoCompare](https://min-api.cryptocompare.com/)
  - [Binance](https://www.binance.com/en/binance-api)

- **Remix IDE**: Browser-based IDE for Solidity smart contracts - OPTIONAL
- **Node.js**: Open-source JavaScript runtime environment - OPTIONAL

## Installation

1. **Clone the Repository**

   ```sh
   git clone https://github.com/JMitander/0xplorer.git
   cd 0xplorer
   ```

2. **Create a Virtual Environment** 

   ```sh
   python3 -m venv venv
   source venv/bin/activate  # For Linux/MacOS
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

### Infura
- **Sign Up**: [Infura Registration](https://infura.io/register)
- **Create a Project**: Obtain your **Project ID**.

### Etherscan
- **Sign Up**: [Etherscan Registration](https://etherscan.io/register)
- **API Key**: Navigate to the API section to get your key.

### CoinGecko
- **Sign Up**: [CoinGecko API](https://www.coingecko.com/en/api)
- **API Key**: Obtain from your account dashboard.

### CoinMarketCap
- **Sign Up**: [CoinMarketCap API](https://coinmarketcap.com/api/)
- **API Key**: Generate a new key in the API section.

### CryptoCompare
- **Sign Up**: [CryptoCompare API](https://min-api.cryptocompare.com/)
- **API Key**: Obtain from your account settings.

## Geth Node Setup

- Set up a Ethereum node using either Geth, Nethermind, Besu, Erigon, Reth, or EthereumJS. refer to the [Ethereum Node](https://ethereum.org/en/developers/docs/nodes-and-clients/) documentation for more information.
- We will use Geth for the execution client and prysm for the beacon chain in this guide.
  You will also need to set up a beacon chain. We recommend using [Prysm](https://docs.prylabs.network/docs/install/install-with-docker) or [Lighthouse](https://lighthouse-book.sigmaprime.io/installation.html).

1. **Install Geth**

   Follow the [Geth installation guide](https://geth.ethereum.org/docs/install-and-build/installing-geth) for your operating system.

  **Install Prysm**

   Follow the [Prysm installation guide](https://docs.prylabs.network/docs/install/install-with-docker) for your operating system.

2. **Start the execution client and beacon chain**

   ```sh
   ./prysm.sh beacon-chain --execution-endpoint=<PATH_GETH_IPC> --mainnet --checkpoint-sync-url=https://beaconstate.info --genesis-beacon-api-url=https://beaconstate.info --suggested-fee-recipient=YourWalletAddress --accept-terms-of-use
   ```

   ```sh
   ./geth --mainnet --syncmode "snap" --http --http.api eth,net,admin,engine,web3,txpool --ipcpath /path/to/geth.ipc --maxpeers 100 --http.corsdomain "*" --cache 4096 
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
   This may take from 12 hours to several days, depending on your hardware and network speed.

## Configuration

Ensure all environment variables in the `.env` file are correctly set.

### Example `.env` File

```sh
# ================================ API Configuration ================================ #
# All API keys can be obtained free of charge by registering on the respective platforms (limitied usage may apply)
# Etherscan API Key for accessing Etherscan services
ETHERSCAN_API_KEY=YourEtherscanAPIKey
# Infura Project ID for connecting to Ethereum nodes via Infura
INFURA_PROJECT_ID=YourInfuraProjectID
# CoinGecko API Key for fetching cryptocurrency prices and market data
COINGECKO_API_KEY=YourCoinGeckoAPIKey
# CoinMarketCap API Key for accessing market capitalization and pricing data
COINMARKETCAP_API_KEY=YourCoinMarketCapAPIKey
# CryptoCompare API Key for obtaining cryptocurrency price information
CRYPTOCOMPARE_API_KEY=YourCryptoCompareAPIKey

# ================================ Ethereum Node Configuration ================================ #
# At least one of the following Ethereum node configurations is required to connect to the Ethereum network

# HTTP Provider URL for connecting to the Ethereum network via Geth
HTTP_ENDPOINT=http://127.0.0.1:8545
# AsyncWeb3 Provider URL (WebSocket) for subscribing to blockchain events via Geth
WEB3_ENDPOINT=wss://127.0.0.1:8545
# WebSocket Provider URL for subscribing to blockchain events via Geth
WEBSOCKET_ENDPOINT=wss://127.0.0.1:8545
# IPC Provider Path for inter-process communication trough pipe socket via Geth
IPC_ENDPOINT=path/to/geth.ipc

# ================================ Wallet Configuration ================================ #

# Private Key (**Ensure this is kept secret! USE ENCRYPTION AND NEVER SHARE, NOT EVEN WITH YOUR DOG!**) 
WALLET_KEY=YourWalletPrivateKey
# BOT Wallet Address associated with the private key above 
WALLET_ADDRESS=0xYourWalletAddress
# Wallet for receiving potential profits from operations.
PROFIT_ADDRESS=0xYourProfitAddress

# ================================ Token Configuration ================================ #

# Path to the JSON file containing monitored token addresses # REQUIRED
TOKEN_ADDRESSES=/Your/directory/0xplorer/monitored_tokens.json
# Path to the JSON file containing token symbols mapping (address to symbol)
TOKEN_SYMBOLS=/Your/directory/0xplorer/token_symbols.json

# ============================ UNISWAP V2 ============================== #

# Uniswap v2 router contract address for executing trades
UNISWAP_V2_ROUTER_ADDRESS=0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D
# Uniswap v2 router contract ABI for interacting with the router contract
UNISWAP_V2_ROUTER_ABI=Y/our/directory/0xplorer/ABI/uniswap_v2_router_ABI.json

# ============================ SUSHISWAP ==============================

# Sushiswap router contract address for executing trades # OPTIONAL
SUSHISWAP_ROUTER_ADDRESS=0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F 
# Sushiswap router contract ABI for interacting with the router contract
SUSHISWAP_ROUTER_ABI=Your/directory/0xplorer/ABI/sushiswap_router_ABI.json

# ============================ PANCAKESWAP ============================== #

# Pancakeswap router contract address for executing trades # OPTIONAL
PANCAKESWAP_ROUTER_ADDRESS=0xEfF92A263d31888d860bD50809A8D171709b7b1c
# Pancakeswap router contract ABI for interacting with the router contract
PANCAKESWAP_ROUTER_ABI=Your/directory/0xplorer/ABI/pancakeswap_router_ABI.json

# ============================ BALANCER ============================== #

# Balancer exchange contract address for executing trades
BALANCER_ROUTER_ADDRESS=0x3E66B66Fd1d0b02fDa6C811da9E0547970DB2f21
# Balancer exchange contract ABI for interacting with the exchange contract
BALANCER_ROUTER_ABI=Your/directory/0xplorer/ABI/balancer_router_ABI.json

# ============================ ERC20 ================================ #

# ERC20 ABI for interacting with ERC20 tokens # REQUIRED
ERC20_ABI=Your/directory/0xplorer/ABI/erc20_ABI.json
# ERC20 function signatures # REQUIRED
ERC20_SIGNATURES=/Your/directory/0xplorer/erc20_signatures.json

# ================================ FLASHLOAN Configuration ================================ #

# Aave V3 Flashloan Contract Address for executing flashloan operations
AAVE_V3_FLASHLOAN_CONTRACT_ADDRESS=YourDeployedFlashloanContractAddress

# Aave V3 Flashloan Contract ABI for interacting with the flashloan contract
AAVE_V3_FLASHLOAN_CONTRACT_ABI=Your/directory/0xplorer/ABI/aave_v3_flashloan_contract_ABI.json

# Aave V3 Lending Pool Address for interacting with Aave's lending protocols
AAVE_V3_LENDING_POOL_ADDRESS=0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2 

# Aave V3 Pool Abi for interacting with Aave's lending protocols
AAVE_V3_LENDING_POOL_ABI=Your/directory/0xplorer/ABI/aave_v3_lending_pool_ABI.json


```

### Monitored Tokens

You can replace the default token addresses in `Utils/token_addresses.json` with your own list of tokens to monitor. The bot will track these tokens for profitable opportunities. Make sure the token addresses are valid ERC20 tokens.

```json
[
  "0xTokenAddress1",
  "0xTokenAddress2",
  "0xTokenAddress3"
]
```


## Creating and Deploying the Flash Loan Contract

To execute flashloan strategies, you need to create and deploy a flashloan smart contract. Below are the steps to create and deploy a flashloan contract using Aave V3 on the Ethereum network.

[Quicknode Flashloan Guide](https://www.quicknode.com/guides/defi/lending-protocols/how-to-make-a-flash-loan-using-aave)

In this guide, we will use the Remix IDE to create and deploy the flashloan contract. Aave V3 is used for the flashloan operations.
Its recommended to read the guide provided above for the latest implementation of flashloans.

### Step 1: Create the Flash Loan Contract

1. **Open Remix IDE**: [Remix IDE](https://remix.ethereum.org/)
2. **Create a New File**: Name it `SimpleFlashLoan.sol`
3. **Copy the Following Code**:

   ```solidity
   // SPDX-License-Identifier: MIT
   pragma solidity 0.8.10;

   import "https://github.com/aave/aave-v3-core/blob/master/contracts/flashloan/base/FlashLoanSimpleReceiverBase.sol";
   import "https://github.com/aave/aave-v3-core/blob/master/contracts/interfaces/IPoolAddressesProvider.sol";
   import "https://github.com/aave/aave-v3-core/blob/master/contracts/dependencies/openzeppelin/contracts/IERC20.sol";

   contract SimpleFlashLoan is FlashLoanSimpleReceiverBase {
       address payable owner;

       constructor(address _addressProvider)
           FlashLoanSimpleReceiverBase(IPoolAddressesProvider(_addressProvider))
       {
           owner = payable(msg.sender);
       }

       function fn_RequestFlashLoan(address _token, uint256 _amount) public {
           address receiverAddress = address(this);
           address asset = _token;
           uint256 amount = _amount;
           bytes memory params = "";
           uint16 referralCode = 0;

           POOL.flashLoanSimple(
               receiverAddress,
               asset,
               amount,
               params,
               referralCode
           );
       }

       // This function is called after your contract has received the flash loaned amount
       function executeOperation(
           address asset,
           uint256 amount,
           uint256 premium,
           address initiator,
           bytes calldata params
       ) external override returns (bool) {
           // Implement your custom logic here

           uint256 totalAmount = amount + premium;
           IERC20(asset).approve(address(POOL), totalAmount);

           return true;
       }

       receive() external payable {}
   }
   ```

### Step 2: Compile the Contract

1. **Select Compiler Version**: In the Remix sidebar, click on the "Solidity Compiler" tab and select version `0.8.10`.
2. **Compile**: Click on the "Compile SimpleFlashLoan.sol" button.
3. **Ignore Warnings**: You may see some warnings; ensure there are no errors.

### Step 3: Deploy the Contract

1. **Set Environment**: In the "Deploy & Run Transactions" tab, set the environment to "Injected Web3" to use MetaMask.
2. **Provide Constructor Argument**:
   - **_addressProvider**: Obtain the `PoolAddressesProvider` address from [Aave V3 Deployed Addresses](https://docs.aave.com/developers/deployed-contracts/v3-mainnet).
3. **Deploy**: Click the "Deploy" button and confirm the transaction in MetaMask.
4. **Save Contract Address**: After deployment, note down the contract address for future use.

### Step 4: Execute Flashloan Function

The flashloan execution is handled within the `fn_RequestFlashLoan`. 0xplorer is automatically configured to interact with the flashloan contract using the provided ABI and address.
## Usage

### Running the Bot

Start 0xplorer:

```sh
python3 core/0xplorer.py
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

Logs are stored in `logs/0xplorer.log`. They include:

- Detected profitable transactions
- Strategy execution details
- Errors and exceptions
- Transaction details and results

###

Configure logging in `core/0xplorer.py` within the `setup_logging()` function.

### Warning

0xplorer is a work in progress and should be used with caution. It is recommended to test the bot on a testnet before running it on the mainnet. Its nowhere near production ready and should be used with caution.

We love to see contributions to this project. Feel free to fork and submit a pull request.
Together we can make this project better. Open an issue if you have any questions or suggestions.

#

## License

This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute the code with proper attribution.
