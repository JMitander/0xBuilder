# 0xplorer Bot

0xplorer is an advanced Ethereum trading bot designed for high-frequency trading and MEV (Miner Extractable Value) opportunities. It implements strategies like front-running, back-running, sandwich attacks, and flashloan executions using Python, Geth, Remix, and AsyncWeb3.py. The bot continuously monitors the Ethereum mempool for profitable transactions and executes trades automatically.

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

## Project Structure

```
/0xplorer/
├── Config/
│   ├── Config.py               # Configuration management
│   ├── token.adresses.json     # List of monitored token addresses
│   └── token.symbols.json      # List of monitored token addresses combined with the relevant symbol
├── Core/
│   ├── 0xplorer.py             # Main
│   ├── NonceManager.py         # Manages Ethereum nonces
│   ├── StrategyManager.py      # Handles trading strategies
│   └── TransactionArray.py     # Builds and sends transaction bundles
├── Analysis/
│   ├── MarketAnalyzer.py       # Analyzes market data
│   ├── MonitorArray.py         # Monitors mempool for transactions
│   └── SafetyNet.py            # Safety checks and validations
├── Contracts/
│   └── SimpleFlashLoan.sol     # Flash loan smart contract
├── Logs/
│   └── 0xplorer.log            # Logs bot activities
├── .env-template               # Example environment variables
├── requirements.txt            # Python dependencies
├── License.md                  # License information
└── README.md                   # Project documentation
```

## Prerequisites

Before running 0xplorer, ensure you have the following:

- **Python 3.x**: Programming language used for the bot.
- **Ethereum Node**: A fully synchronized execution client.

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

2. **Create a Virtual Environment** (recommended)

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

- Set up an Ethereum node/Execution Client using Geth as an example.

1. **Install Geth**

   Follow the [official installation guide](https://geth.ethereum.org/docs)

2. **Start Geth Node**

   ```sh
   geth --mainnet --networkid "1" --syncmode "snap" --http --http.api eth,net,admin,engine,txpool,web3 --ipcpath /path/to/geth.ipc --maxpeers 100
   ```

   You will also need to set up a beacon-chain running alongside Geth (required after the merge). Follow the [Prylabs docs](https://docs.prylabs.network/docs/install/install-with-script).

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
# Ethereum Node Connection
HTTP_PROVIDER=http://127.0.0.1:8545
WEBSOCKET_PROVIDER=ws://127.0.0.1:8546
IPC_PROVIDER=/path/to/geth.ipc

# Wallet Details
WALLET_ADDRESS=0xYourWalletAddressHere
WALLET_PRIVATE_KEY=YourPrivateKeyHere

# Flashloan Configuration
AAVE_V3_LENDING_POOL_ADDRESS=0xYourLendingPoolAddress
AAVE_V3_FLASHLOAN_CONTRACT_ADDRESS=0xYourFlashloanContractAddress

# API Keys
INFURA_PROJECT_ID=YourInfuraProjectID
ETHERSCAN_API_KEY=YourEtherscanAPIKey
COINGECKO_API_KEY=YourCoinGeckoAPIKey
COINMARKETCAP_API_KEY=YourCoinMarketCapAPIKey
CRYPTOCOMPARE_API_KEY=YourCryptoCompareAPIKey

# ABI Paths
ERC20_ABI_PATH=ABI/erc20.json
AAVE_V3_FLASHLOAN_ABI_PATH=ABI/aave_v3_flashloan.json
AAVE_V3_LENDING_POOL_ABI_PATH=ABI/lending_pool.json

# Monitored Tokens
MONITORED_TOKENS=config/Monitored_tokens.json
TOKEN_SYMBOLS=tokens/token_symbols.json
```

### Monitored Tokens

Optional

Update `config/Monitored_tokens.json` with the token addresses you want the bot to monitor:

```json
[
  "0xTokenAddress1",
  "0xTokenAddress2"
]
```

## Creating and Deploying the Flash Loan Contract

To execute flashloan strategies, you need to create and deploy a flashloan smart contract. Below are the steps to create and deploy a flashloan contract using Aave V3 on the Ethereum network.

It's recommended to follow this guide: [Flashloan Guide](https://www.quicknode.com/guides/defi/lending-protocols/how-to-make-a-flash-loan-using-aave)

### Prerequisites

- **MetaMask**: Browser extension wallet (or any other wallet provider: Coinbase, Trustwallet, Exodus, etc.)
- **Remix IDE**: [Remix IDE](https://remix.ethereum.org/) (Recommended)

### Step 1: Setup MetaMask

1. **Install MetaMask**: [Download MetaMask](https://metamask.io/)
2. **Create a Wallet**: Set up a new wallet and secure your seed phrase.
3. **Add Custom RPC**: Add the Ethereum testnet or mainnet RPC URL from QuickNode.
   - **Network Name**: Ethereum Mainnet (or desired testnet)
   - **New RPC URL**: Your Geth HTTP Provider URL
   - **Chain ID**: 1 for Mainnet (or corresponding testnet ID)
   - **Currency Symbol**: ETH
4. **Fund Your Wallet**: Obtain test ETH from a faucet if deploying on a testnet.

### Step 2: Create the Flash Loan Contract

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

### Step 3: Compile the Contract

1. **Select Compiler Version**: In the Remix sidebar, click on the "Solidity Compiler" tab and select version `0.8.10`.
2. **Compile**: Click on the "Compile SimpleFlashLoan.sol" button.
3. **Ignore Warnings**: You may see some warnings; ensure there are no errors.

### Step 4: Deploy the Contract

1. **Set Environment**: In the "Deploy & Run Transactions" tab, set the environment to "Injected Web3" to use MetaMask.
2. **Provide Constructor Argument**:
   - **_addressProvider**: Obtain the `PoolAddressesProvider` address from [Aave V3 Deployed Addresses](https://docs.aave.com/developers/deployed-contracts/v3-mainnet).
3. **Deploy**: Click the "Deploy" button and confirm the transaction in MetaMask.
4. **Save Contract Address**: After deployment, note down the contract address for future use.

### Step 5: Fund the Contract (if needed)

If your strategy requires initial funds (e.g., to cover flashloan fees), send tokens or ETH to the contract address.

### Step 6: Execute Flashloan Function

1. **Interact with the Contract**:
   - Expand the deployed contract in Remix.
   - Locate the `fn_RequestFlashLoan` function.
2. **Provide Parameters**:
   - **_token**: The address of the token you want to borrow.
   - **_amount**: The amount you want to borrow (consider token decimals).
3. **Execute**: Click on "transact" and confirm the transaction in MetaMask.

### Step 7: Verify Execution

1. **Check Transaction**: Use Etherscan to view the transaction details.
2. **Logs**: Ensure the flashloan was successful and the funds were returned.

### Notes

- **Custom Logic**: Implement your trading logic within the `executeOperation` function.
- **Fees**: Remember that Aave charges a fee (e.g., 0.09%) on the flashloan amount.
- **Permissions**: Ensure your contract has the necessary approvals to interact with tokens.

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

## Logging

Logs are stored in `logs/0xplorer.log`. They include:

- Detected profitable transactions
- Strategy execution details
- Errors and exceptions

Configure logging in `core/0xplorer.py` within the `setup_logging()` function.

## License

This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute the code with proper attribution.
