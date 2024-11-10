<!-- markdownlint-disable -->

# <kbd>module</kbd> `0xplorer_main.py`





---

## <kbd>function</kbd> `loading_bar`

```python
loading_bar(message: str, total_time: int)
```






---

## <kbd>function</kbd> `setup_logging`

```python
setup_logging()
```






---

## <kbd>function</kbd> `main`

```python
main()
```






---

## <kbd>class</kbd> `Configuration`
Loads configuration from environment variables and monitored tokens from a JSON file. 

### <kbd>function</kbd> `__init__`

```python
__init__(logger: Optional[Logger] = None)
```








---

### <kbd>function</kbd> `get_ABI_path`

```python
get_ABI_path(ABI_name: str) → str
```

Retrieves the abi path for a given contract name. 

---

### <kbd>function</kbd> `get_token_addresses`

```python
get_token_addresses() → List[str]
```

Returns the list of monitored token addresses loaded from the JSON file, path specified in the environment variable. 

---

### <kbd>function</kbd> `get_token_symbols`

```python
get_token_symbols() → str
```

Returns the path to the token symbols JSON file. 

---

### <kbd>function</kbd> `load`

```python
load()
```






---

## <kbd>class</kbd> `Market_Monitor`
Market_Monitor class analyzes market conditions, fetches price data, and checks for arbitrage opportunities. 

### <kbd>function</kbd> `__init__`

```python
__init__(
    web3: AsyncWeb3,
    erc20_ABI: List[Dict[str, Any]],
    configuration: Configuration,
    logger: Optional[Logger] = None
)
```








---

### <kbd>function</kbd> `check_market_conditions`

```python
check_market_conditions(token_address: str) → Dict[str, Any]
```





---

### <kbd>function</kbd> `decode_transaction_input`

```python
decode_transaction_input(
    input_data: str,
    contract_address: str
) → Optional[Dict[str, Any]]
```





---

### <kbd>function</kbd> `fetch_historical_prices`

```python
fetch_historical_prices(token_id: str, days: int = 30) → List[float]
```





---

### <kbd>function</kbd> `get_current_price`

```python
get_current_price(token_id: str) → Optional[float]
```





---

### <kbd>function</kbd> `get_token_symbol`

```python
get_token_symbol(token_address: str) → Optional[str]
```





---

### <kbd>function</kbd> `get_token_volume`

```python
get_token_volume(token_id: str) → float
```





---

### <kbd>function</kbd> `is_arbitrage_opportunity`

```python
is_arbitrage_opportunity(target_tx: Dict[str, Any]) → bool
```





---

### <kbd>function</kbd> `make_request`

```python
make_request(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None
) → ClientResponse
```






---

## <kbd>class</kbd> `Mempool_Monitor`
Mempool_Monitor class monitors the mempool for profitable transactions. 

### <kbd>function</kbd> `__init__`

```python
__init__(
    web3: AsyncWeb3,
    safety_net: Safety_Net,
    nonce_core: Nonce_Core,
    logger: Optional[Logger] = None,
    monitored_tokens: Optional[List[str]] = None,
    erc20_ABI: List[Dict[str, Any]] = None,
    configuration: Configuration = None
)
```








---

### <kbd>function</kbd> `analyze_transaction`

```python
analyze_transaction(tx) → Dict[str, Any]
```

Analyze a transaction to determine if it's profitable. 

---

### <kbd>function</kbd> `get_token_symbol`

```python
get_token_symbol(token_address: str) → Optional[str]
```

Get the symbol of a token using its address. 

---

### <kbd>function</kbd> `mempool_monitor`

```python
mempool_monitor()
```

Monitor the mempool for profitable transactions. 

---

### <kbd>function</kbd> `process_transaction`

```python
process_transaction(tx_hash)
```

Process a single transaction hash. 

---

### <kbd>function</kbd> `start_monitoring`

```python
start_monitoring()
```

Start monitoring the mempool. 

---

### <kbd>function</kbd> `stop_monitoring`

```python
stop_monitoring()
```

Stop monitoring the mempool. 


---

## <kbd>class</kbd> `Nonce_Core`
Manages the nonce for an Ethereum account to prevent transaction nonce collisions. 

### <kbd>function</kbd> `__init__`

```python
__init__(
    web3: AsyncWeb3,
    address: str,
    logger: Optional[Logger] = None,
    max_retries: int = 3,
    retry_delay: float = 1.0
)
```








---

### <kbd>function</kbd> `get_nonce`

```python
get_nonce() → int
```

Retrieves the next nonce safely for the address. This method locks the nonce manager to avoid concurrent modifications, ensuring each transaction has a unique nonce. 

---

### <kbd>function</kbd> `handle_nonce_discrepancy`

```python
handle_nonce_discrepancy(external_nonce: int)
```

Adjusts the current nonce if an external nonce (e.g., from a failed transaction) is higher than the internally managed nonce. This helps resolve nonce conflicts. 



**Args:**
 
 - <b>`external_nonce`</b> (int):  The externally detected nonce (e.g., from a failed or pending transaction). 

---

### <kbd>function</kbd> `initialize`

```python
initialize()
```





---

### <kbd>function</kbd> `refresh_nonce`

```python
refresh_nonce()
```

Updates the current nonce value by fetching the latest nonce from the blockchain. This is a soft refresh and only updates if the on-chain nonce is higher than the internal one. 

---

### <kbd>function</kbd> `reset_nonce`

```python
reset_nonce()
```

Resets the current nonce to the on-chain value. Useful in case of major nonce conflicts or after manually handling stuck transactions. 

---

### <kbd>function</kbd> `sync_nonce_with_chain`

```python
sync_nonce_with_chain()
```

This method is a more aggressive sync that forces the nonce to synchronize with the blockchain. Useful in case of transaction reverts or nonce conflicts. 


---

## <kbd>class</kbd> `Safety_Net`




### <kbd>function</kbd> `__init__`

```python
__init__(
    web3: AsyncWeb3,
    configuration: Configuration,
    account: Account,
    logger: Optional[Logger] = None
)
```

Provides safety checks and utility functions for transactions. 



**Args:**
 
 - <b>`web3`</b> (AsyncWeb3):  AsyncWeb3 instance connected to the Ethereum network. 
 - <b>`configuration`</b> (Configuration):  Configurationsuration object containing API keys and settings. 
 - <b>`account`</b> (Account):  The Ethereum account. 
 - <b>`logger`</b> (Optional[logging.Logger]):  Logger instance. 




---

### <kbd>function</kbd> `adjust_slippage_tolerance`

```python
adjust_slippage_tolerance() → float
```

Adjust slippage tolerance based on network congestion and market volatility. 



**Returns:**
 
 - <b>`float`</b>:  The adjusted slippage tolerance. 

---

### <kbd>function</kbd> `ensure_profit`

```python
ensure_profit(
    transaction_data: Dict[str, Any],
    minimum_profit_eth: Optional[float] = None
) → bool
```

Ensures that a transaction is profitable after accounting for gas costs and slippage. 



**Args:**
 
 - <b>`transaction_data`</b> (Dict[str, Any]):  Data related to the transaction. 
 - <b>`minimum_profit_eth`</b> (float):  The minimum acceptable profit in ETH. 



**Returns:**
 
 - <b>`bool`</b>:  True if the transaction is profitable, False otherwise. 

---

### <kbd>function</kbd> `estimate_gas`

```python
estimate_gas(transaction_data: Dict[str, Any]) → int
```

Estimates the gas required for a transaction. 



**Args:**
 
 - <b>`transaction_data`</b> (Dict[str, Any]):  Data related to the transaction. 



**Returns:**
 
 - <b>`int`</b>:  The estimated gas required. 

---

### <kbd>function</kbd> `get_balance`

```python
get_balance(account: Account) → Decimal
```

Returns the balance of an account in ETH. 



**Args:**
 
 - <b>`account`</b> (Account):  The Ethereum account. 



**Returns:**
 
 - <b>`Decimal`</b>:  The balance in ETH. 

---

### <kbd>function</kbd> `get_dynamic_gas_price`

```python
get_dynamic_gas_price() → float
```

Fetch the current gas price using gas oracles with fallback options. 



**Returns:**
 
 - <b>`float`</b>:  The gas price in Gwei. 

---

### <kbd>function</kbd> `get_eth_price_from_binance`

```python
get_eth_price_from_binance() → Optional[Decimal]
```





---

### <kbd>function</kbd> `get_network_congestion`

```python
get_network_congestion() → float
```

Estimate network congestion level (0 to 1). 



**Returns:**
 
 - <b>`float`</b>:  The network congestion level. 

---

### <kbd>function</kbd> `get_real_time_price`

```python
get_real_time_price(token: str) → Decimal
```

Fetches the real-time price of a token in terms of ETH from multiple sources. 



**Args:**
 
 - <b>`token`</b> (str):  The token symbol. 



**Returns:**
 
 - <b>`Decimal`</b>:  The price of the token in ETH. 

---

### <kbd>function</kbd> `make_request`

```python
make_request(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None
) → ClientResponse
```

Make a request to an external API with retry mechanism and exponential backoff. 



**Args:**
 
 - <b>`url`</b> (str):  The API endpoint. 
 - <b>`params`</b> (Optional[Dict[str, Any]]):  Query parameters. 
 - <b>`headers`</b> (Optional[Dict[str, str]]):  Request headers. 



**Returns:**
 
 - <b>`aiohttp.ClientResponse`</b>:  The API response. 



**Raises:**
 
 - <b>`Exception`</b>:  If the request fails after maximum retries. 


---

## <kbd>class</kbd> `Strategy_Net`
Manages and executes various trading strategies such as ETH transactions, front-running, back-running, and sandwich attacks. It tracks strategy performance, predicts market movements, and selects the best strategy based on historical performance and reinforcement learning. 

### <kbd>function</kbd> `__init__`

```python
__init__(
    transaction_core: Transaction_Core,
    market_monitor: 'Market_Monitor',
    logger: Optional[Logger] = None
) → None
```

Initializes the Strategy_Net with necessary components. 



**Args:**
 
 - <b>`transaction_core`</b> (Transaction_Core):  Instance managing transactions. 
 - <b>`market_monitor`</b> (Market_Monitor):  Instance analyzing market conditions. 
 - <b>`logger`</b> (Optional[logging.Logger]):  Logger instance for logging. 




---

### <kbd>function</kbd> `advanced_back_run`

```python
advanced_back_run(target_tx: Dict[str, Any]) → bool
```

Enhanced Strategy: Combines price predictions and market trends for back-running. 



**Args:**
 
 - <b>`target_tx`</b> (Dict[str, Any]):  The target transaction details. 



**Returns:**
 
 - <b>`bool`</b>:  True if the enhanced back-run was executed successfully, False otherwise. 

---

### <kbd>function</kbd> `advanced_front_run`

```python
advanced_front_run(target_tx: Dict[str, Any]) → bool
```

Enhanced Strategy: Combines price predictions and market trends for front-running. 



**Args:**
 
 - <b>`target_tx`</b> (Dict[str, Any]):  The target transaction details. 



**Returns:**
 
 - <b>`bool`</b>:  True if the enhanced front-run was executed successfully, False otherwise. 

---

### <kbd>function</kbd> `advanced_sandwich_attack`

```python
advanced_sandwich_attack(target_tx: Dict[str, Any]) → bool
```

Enhanced Strategy: Combines flashloan profitability and market volatility for sandwich attacks. 



**Args:**
 
 - <b>`target_tx`</b> (Dict[str, Any]):  The target transaction details. 



**Returns:**
 
 - <b>`bool`</b>:  True if the enhanced sandwich attack was executed successfully, False otherwise. 

---

### <kbd>function</kbd> `aggressive_front_run`

```python
aggressive_front_run(target_tx: Dict[str, Any]) → bool
```

Strategy: Aggressively front-runs transactions based on value. 



**Args:**
 
 - <b>`target_tx`</b> (Dict[str, Any]):  The target transaction details. 



**Returns:**
 
 - <b>`bool`</b>:  True if the front-run was executed successfully, False otherwise. 

---

### <kbd>function</kbd> `arbitrage_sandwich`

```python
arbitrage_sandwich(target_tx: Dict[str, Any]) → bool
```

Strategy: Executes sandwich attack based on arbitrage opportunities. 



**Args:**
 
 - <b>`target_tx`</b> (Dict[str, Any]):  The target transaction details. 



**Returns:**
 
 - <b>`bool`</b>:  True if the sandwich attack was executed successfully, False otherwise. 

---

### <kbd>function</kbd> `execute_best_strategy`

```python
execute_best_strategy(target_tx: Dict[str, Any], strategy_type: str) → bool
```

Executes the most suitable strategy based on historical performance and current market conditions. 



**Args:**
 
 - <b>`target_tx`</b> (Dict[str, Any]):  The target transaction details. 
 - <b>`strategy_type`</b> (str):  The type of strategy to execute ('eth_transaction', 'front_run', 'back_run', 'sandwich_attack'). 



**Returns:**
 
 - <b>`bool`</b>:  True if the strategy was executed successfully, False otherwise. 

---

### <kbd>function</kbd> `execute_strategy_for_transaction`

```python
execute_strategy_for_transaction(target_tx: Dict[str, Any]) → bool
```

Determines and executes the best strategy for a given transaction. 



**Args:**
 
 - <b>`target_tx`</b> (Dict[str, Any]):  The target transaction details. 



**Returns:**
 
 - <b>`bool`</b>:  True if a strategy was executed successfully, False otherwise. 

---

### <kbd>function</kbd> `flash_profit_sandwich`

```python
flash_profit_sandwich(target_tx: Dict[str, Any]) → bool
```

Strategy: Executes sandwich attack based on flashloan profitability. 



**Args:**
 
 - <b>`target_tx`</b> (Dict[str, Any]):  The target transaction details. 



**Returns:**
 
 - <b>`bool`</b>:  True if the sandwich attack was executed successfully, False otherwise. 

---

### <kbd>function</kbd> `flashloan_back_run`

```python
flashloan_back_run(target_tx: Dict[str, Any]) → bool
```

Strategy: Utilizes flashloan calculations to determine back-running profitability. 



**Args:**
 
 - <b>`target_tx`</b> (Dict[str, Any]):  The target transaction details. 



**Returns:**
 
 - <b>`bool`</b>:  True if the back-run was executed successfully, False otherwise. 

---

### <kbd>function</kbd> `get_strategies`

```python
get_strategies(strategy_type: str) → List[Any]
```

Retrieves a list of strategies based on the strategy type. 



**Args:**
 
 - <b>`strategy_type`</b> (str):  The type of strategy. 



**Returns:**
 
 - <b>`List[Any]`</b>:  List of strategy functions. 

---

### <kbd>function</kbd> `get_strategy_index`

```python
get_strategy_index(strategy_name: str, strategy_type: str) → int
```

Retrieves the index of a strategy based on its name and type. 



**Args:**
 
 - <b>`strategy_name`</b> (str):  Name of the strategy. 
 - <b>`strategy_type`</b> (str):  Type of the strategy. 



**Returns:**
 
 - <b>`int`</b>:  Index of the strategy in the reinforcement weights array. Returns -1 if not found. 

---

### <kbd>function</kbd> `high_value_eth_transfer`

```python
high_value_eth_transfer(target_tx: Dict[str, Any]) → bool
```

Strategy: Handles high-value ETH transfers. 



**Args:**
 
 - <b>`target_tx`</b> (Dict[str, Any]):  The target transaction details. 



**Returns:**
 
 - <b>`bool`</b>:  True if the strategy was executed successfully, False otherwise. 

---

### <kbd>function</kbd> `high_volume_back_run`

```python
high_volume_back_run(target_tx: Dict[str, Any]) → bool
```

Strategy: Executes back-run based on high token trading volume. 



**Args:**
 
 - <b>`target_tx`</b> (Dict[str, Any]):  The target transaction details. 



**Returns:**
 
 - <b>`bool`</b>:  True if the back-run was executed successfully, False otherwise. 

---

### <kbd>function</kbd> `predict_price_movement`

```python
predict_price_movement(token_symbol: str) → float
```

Predicts the next price movement of a token using historical price data. 



**Args:**
 
 - <b>`token_symbol`</b> (str):  The symbol of the token. 



**Returns:**
 
 - <b>`float`</b>:  Predicted price movement value. 

---

### <kbd>function</kbd> `predictive_front_run`

```python
predictive_front_run(target_tx: Dict[str, Any]) → bool
```

Strategy: Front-runs transactions based on price predictions. 



**Args:**
 
 - <b>`target_tx`</b> (Dict[str, Any]):  The target transaction details. 



**Returns:**
 
 - <b>`bool`</b>:  True if the front-run was executed successfully, False otherwise. 

---

### <kbd>function</kbd> `price_boost_sandwich`

```python
price_boost_sandwich(target_tx: Dict[str, Any]) → bool
```

Strategy: Executes sandwich attack based on favorable token price conditions. 



**Args:**
 
 - <b>`target_tx`</b> (Dict[str, Any]):  The target transaction details. 



**Returns:**
 
 - <b>`bool`</b>:  True if the sandwich attack was executed successfully, False otherwise. 

---

### <kbd>function</kbd> `price_dip_back_run`

```python
price_dip_back_run(target_tx: Dict[str, Any]) → bool
```

Strategy: Executes back-run based on significant price dips. 



**Args:**
 
 - <b>`target_tx`</b> (Dict[str, Any]):  The target transaction details. 



**Returns:**
 
 - <b>`bool`</b>:  True if the back-run was executed successfully, False otherwise. 

---

### <kbd>function</kbd> `update_history`

```python
update_history(
    strategy_name: str,
    success: bool,
    strategy_type: str,
    profit: Decimal
) → None
```

Updates the historical data and reinforcement weights based on the strategy execution outcome. 



**Args:**
 
 - <b>`strategy_name`</b> (str):  Name of the executed strategy. 
 - <b>`success`</b> (bool):  Whether the strategy execution was successful. 
 - <b>`strategy_type`</b> (str):  The type of strategy. 
 - <b>`profit`</b> (Decimal):  Profit made from the strategy execution. 

---

### <kbd>function</kbd> `volatility_front_run`

```python
volatility_front_run(target_tx: Dict[str, Any]) → bool
```

Strategy: Front-runs transactions based on market volatility. 



**Args:**
 
 - <b>`target_tx`</b> (Dict[str, Any]):  The target transaction details. 



**Returns:**
 
 - <b>`bool`</b>:  True if the front-run was executed successfully, False otherwise. 


---

## <kbd>class</kbd> `Transaction_Core`
Transaction_Core class builds and executes transactions, including front-run, back-run, and sandwich attack strategies. It interacts with smart contracts, manages transaction signing, gas price estimation, and handles flashloans. 

### <kbd>function</kbd> `__init__`

```python
__init__(
    web3: AsyncWeb3,
    account: Account,
    flashloan_contract_address: str,
    flashloan_contract_ABI: List[Dict[str, Any]],
    lending_pool_contract_address: str,
    lending_pool_contract_ABI: List[Dict[str, Any]],
    monitor: Mempool_Monitor,
    nonce_core: Nonce_Core,
    safety_net: Safety_Net,
    configuration: Configuration,
    logger: Optional[Logger] = None,
    gas_price_multiplier: float = 1.1,
    retry_attempts: int = 3,
    retry_delay: float = 1.0,
    erc20_ABI: Optional[List[Dict[str, Any]]] = None
)
```

Initializes the Transaction_Core with necessary components. 

Note: Since __init__ cannot be async, any async initializations are moved to an async `initialize` method. 




---

### <kbd>function</kbd> `back_run`

```python
back_run(target_tx: Dict[str, Any]) → bool
```

Attempts to back-run a target transaction by preparing and executing a flashloan and back-run transaction bundle. 



**Args:**
 
 - <b>`target_tx`</b> (Dict[str, Any]):  The target transaction details. 



**Returns:**
 
 - <b>`bool`</b>:  True if the back-run was successfully executed, False otherwise. 

---

### <kbd>function</kbd> `build_transaction`

```python
build_transaction(
    function_call: Any,
    additional_params: Optional[Dict[str, Any]] = None
) → Dict[str, Any]
```

Builds a transaction dictionary with necessary parameters. 



**Args:**
 
 - <b>`function_call`</b> (Any):  The contract function to call. 
 - <b>`additional_params`</b> (Optional[Dict[str, Any]], optional):  Additional transaction parameters. Defaults to None. 



**Returns:**
 
 - <b>`Dict[str, Any]`</b>:  The built transaction dictionary. 



**Raises:**
 
 - <b>`Exception`</b>:  If transaction building fails. 

---

### <kbd>function</kbd> `calculate_flashloan_amount`

```python
calculate_flashloan_amount(target_tx: Dict[str, Any]) → int
```

Calculates the flashloan amount based on the estimated profit from the target transaction. 



**Args:**
 
 - <b>`target_tx`</b> (Dict[str, Any]):  The target transaction details. 



**Returns:**
 
 - <b>`int`</b>:  The calculated flashloan amount in Wei. 

---

### <kbd>function</kbd> `cancel_transaction`

```python
cancel_transaction(nonce: int) → bool
```

Cancels a pending transaction by sending a zero-value transaction with the same nonce. 



**Args:**
 
 - <b>`nonce`</b> (int):  The nonce of the transaction to cancel. 



**Returns:**
 
 - <b>`bool`</b>:  True if the cancellation was successful, False otherwise. 

---

### <kbd>function</kbd> `decode_transaction_input`

```python
decode_transaction_input(
    input_data: str,
    to_address: str
) → Optional[Dict[str, Any]]
```

Decodes the input data of a transaction to extract the function name and parameters. 



**Args:**
 
 - <b>`input_data`</b> (str):  The input data of the transaction. 
 - <b>`to_address`</b> (str):  The address to which the transaction is sent. 



**Returns:**
 
 - <b>`Optional[Dict[str, Any]]`</b>:  A dictionary containing the function name and parameters, or None if decoding fails. 

---

### <kbd>function</kbd> `estimate_gas_limit`

```python
estimate_gas_limit(tx: Dict[str, Any]) → int
```

Estimates the gas limit for a given transaction. 



**Args:**
 
 - <b>`tx`</b> (Dict[str, Any]):  The transaction details. 



**Returns:**
 
 - <b>`int`</b>:  Estimated gas limit. 

---

### <kbd>function</kbd> `estimate_gas_smart`

```python
estimate_gas_smart(tx: Dict[str, Any]) → int
```

Estimates gas for a transaction using a smart estimation method. 



**Args:**
 
 - <b>`tx`</b> (Dict[str, Any]):  The transaction details. 



**Returns:**
 
 - <b>`int`</b>:  Estimated gas amount. 

---

### <kbd>function</kbd> `execute_sandwich_attack`

```python
execute_sandwich_attack(target_tx: Dict[str, Any]) → bool
```

Attempts a sandwich attack on a target transaction by preparing and executing a flashloan, front-run, and back-run transaction bundle. 



**Args:**
 
 - <b>`target_tx`</b> (Dict[str, Any]):  The target transaction details. 



**Returns:**
 
 - <b>`bool`</b>:  True if the sandwich attack was successfully executed, False otherwise. 

---

### <kbd>function</kbd> `execute_transaction`

```python
execute_transaction(tx: Dict[str, Any]) → Optional[str]
```

Attempts to execute a transaction with retries in case of failure. 



**Args:**
 
 - <b>`tx`</b> (Dict[str, Any]):  The transaction details. 



**Returns:**
 
 - <b>`Optional[str]`</b>:  The transaction hash if successful, None otherwise. 

---

### <kbd>function</kbd> `front_run`

```python
front_run(target_tx: Dict[str, Any]) → bool
```

Attempts to front-run a target transaction by preparing and executing a flashloan and front-run transaction bundle. 



**Args:**
 
 - <b>`target_tx`</b> (Dict[str, Any]):  The target transaction details. 



**Returns:**
 
 - <b>`bool`</b>:  True if the front-run was successfully executed, False otherwise. 

---

### <kbd>function</kbd> `get_current_profit`

```python
get_current_profit() → Decimal
```

Placeholder method to retrieve current profit. Should be implemented based on actual profit tracking. 



**Returns:**
 
 - <b>`Decimal`</b>:  Current profit. 

---

### <kbd>function</kbd> `get_dynamic_gas_price`

```python
get_dynamic_gas_price() → Dict[str, int]
```

Retrieves the dynamic gas price, applying a multiplier. 



**Returns:**
 
 - <b>`Dict[str, int]`</b>:  Dictionary containing the gas price in Wei. 

---

### <kbd>function</kbd> `handle_eth_transaction`

```python
handle_eth_transaction(target_tx: Dict[str, Any]) → bool
```

Handles an ETH transaction by building and executing a front-run transaction. 



**Args:**
 
 - <b>`target_tx`</b> (Dict[str, Any]):  The target transaction details. 



**Returns:**
 
 - <b>`bool`</b>:  True if the transaction was successfully executed, False otherwise. 

---

### <kbd>function</kbd> `initialize`

```python
initialize()
```

Async initialization of contracts. 

---

### <kbd>function</kbd> `prepare_flashloan_transaction`

```python
prepare_flashloan_transaction(
    flashloan_asset: str,
    flashloan_amount: int
) → Optional[Dict[str, Any]]
```

Prepares a flashloan transaction. 



**Args:**
 
 - <b>`flashloan_asset`</b> (str):  The asset to be flashloaned. 
 - <b>`flashloan_amount`</b> (int):  The amount of the asset to be flashloaned. 



**Returns:**
 
 - <b>`Optional[Dict[str, Any]]`</b>:  The prepared transaction details or None if preparation fails. 

---

### <kbd>function</kbd> `send_bundle`

```python
send_bundle(transactions: List[Dict[str, Any]]) → bool
```

Sends a bundle of transactions to the Flashbots relay. 



**Args:**
 
 - <b>`transactions`</b> (List[Dict[str, Any]]):  The list of transaction details. 



**Returns:**
 
 - <b>`bool`</b>:  True if the bundle was successfully sent, False otherwise. 

---

### <kbd>function</kbd> `sign_transaction`

```python
sign_transaction(transaction: Dict[str, Any]) → bytes
```

Signs a transaction using the account's private key. 



**Args:**
 
 - <b>`transaction`</b> (Dict[str, Any]):  The transaction details. 



**Returns:**
 
 - <b>`bytes`</b>:  The signed transaction in raw bytes. 



**Raises:**
 
 - <b>`Exception`</b>:  If signing fails. 

---

### <kbd>function</kbd> `simulate_transaction`

```python
simulate_transaction(transaction: Dict[str, Any]) → bool
```

Simulates a transaction using eth_call to ensure it will succeed. 



**Args:**
 
 - <b>`transaction`</b> (Dict[str, Any]):  The transaction details. 



**Returns:**
 
 - <b>`bool`</b>:  True if the simulation succeeds, False otherwise. 


---

## <kbd>class</kbd> `Main_Core`
Builds and manages the entire bot, initializing all components, managing connections, and orchestrating the main execution loop. 

### <kbd>function</kbd> `__init__`

```python
__init__(configuration: Configuration, logger: Optional[Logger] = None) → None
```

Initializes the Main_Core, setting up all necessary components. 



**Args:**
 
 - <b>`configuration`</b> (Configuration):  Configurationsuration object containing settings and API keys. 
 - <b>`logger`</b> (Optional[logging.Logger]):  Logger instance for logging. 




---

### <kbd>function</kbd> `initialize`

```python
initialize()
```

Asynchronous initialization of components that require async calls. 

---

### <kbd>function</kbd> `run`

```python
run() → None
```

Main execution loop for the bot. Continuously monitors the mempool, identifies profitable transactions, and executes appropriate strategies. 

---

### <kbd>function</kbd> `stop`

```python
stop() → None
```

Stops the bot's execution gracefully. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
