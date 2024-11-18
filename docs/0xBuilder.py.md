<!-- markdownlint-disable -->

<a href="../0xBuilder.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `0xBuilder.py`




**Global Variables**
---------------
- **COLORS**

---

<a href="../0xBuilder.py#L57"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `configure_logging`

```python
configure_logging()
```






---

<a href="../0xBuilder.py#L67"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_logger`

```python
get_logger(name: Optional[str] = None) → Logger
```






---

<a href="../0xBuilder.py#L2643"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `loading_bar`

```python
loading_bar(message: str, duration: float)
```






---

<a href="../0xBuilder.py#L4068"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `main`

```python
main()
```

Main entry point with setup and error handling. 


---

## <kbd>class</kbd> `API_Config`




<a href="../0xBuilder.py#L402"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `__init__`

```python
__init__(configuration: Optional[Configuration] = None)
```








---

<a href="../0xBuilder.py#L658"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `close`

```python
close()
```

Close the aiohttp session. 

---

<a href="../0xBuilder.py#L575"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `fetch_historical_prices`

```python
fetch_historical_prices(token: str, days: int = 30) → List[float]
```

Fetch historical price data for a given token symbol. 

---

<a href="../0xBuilder.py#L471"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_real_time_price`

```python
get_real_time_price(token: str, vs_currency: str = 'eth') → Optional[Decimal]
```

Get real-time price using weighted average from multiple sources. 

---

<a href="../0xBuilder.py#L453"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_token_symbol`

```python
get_token_symbol(web3: AsyncWeb3, token_address: str) → Optional[str]
```

Get the token symbol for a given token address. 

---

<a href="../0xBuilder.py#L604"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_token_volume`

```python
get_token_volume(token: str) → float
```

Get the 24-hour trading volume for a given token symbol. 

---

<a href="../0xBuilder.py#L528"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `make_request`

```python
make_request(
    provider_name: str,
    url: str,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    max_attempts: int = 5,
    backoff_factor: float = 1.5
) → Any
```

Make HTTP request with exponential backoff and rate limit per provider. 


---

## <kbd>class</kbd> `ColorFormatter`







---

<a href="../0xBuilder.py#L49"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `format`

```python
format(record: LogRecord) → str
```






---

## <kbd>class</kbd> `Configuration`
Loads configuration from environment variables and monitored tokens from a JSON file. 




---

<a href="../0xBuilder.py#L217"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_abi_path`

```python
get_abi_path(abi_name: str) → str
```





---

<a href="../0xBuilder.py#L211"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_token_addresses`

```python
get_token_addresses() → List[str]
```





---

<a href="../0xBuilder.py#L214"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_token_symbols`

```python
get_token_symbols() → Dict[str, str]
```





---

<a href="../0xBuilder.py#L124"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `load`

```python
load() → None
```

Loads the configuration. 


---

## <kbd>class</kbd> `Main_Core`
Builds and manages the entire MEV bot, initializing all components, managing connections, and orchestrating the main execution loop. 

<a href="../0xBuilder.py#L3636"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `__init__`

```python
__init__(configuration: Optional[Configuration] = None) → None
```








---

<a href="../0xBuilder.py#L3650"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `initialize`

```python
initialize() → None
```

Initialize all components with error handling. 

---

<a href="../0xBuilder.py#L3840"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `run`

```python
run() → None
```

Main execution loop with improved error handling. 

---

<a href="../0xBuilder.py#L3878"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `stop`

```python
stop() → None
```

Graceful shutdown of all components. 


---

## <kbd>class</kbd> `Market_Monitor`




<a href="../0xBuilder.py#L2283"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `__init__`

```python
__init__(
    web3: AsyncWeb3,
    configuration: Optional[Configuration],
    api_config: Optional[API_Config]
)
```








---

<a href="../0xBuilder.py#L2395"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `check_market_conditions`

```python
check_market_conditions(token_address: str) → Dict[str, Any]
```

Check various market conditions for a given token 

:param token_address: Address of the token to check :return: Dictionary of market conditions 

---

<a href="../0xBuilder.py#L2577"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `decode_transaction_input`

```python
decode_transaction_input(
    input_data: str,
    contract_address: str
) → Optional[Dict[str, Any]]
```

Decode the input data of a transaction. 

:param input_data: Hexadecimal input data of the transaction. :param contract_address: Address of the contract being interacted with. :return: Dictionary containing function name and parameters if successful, else None. 

---

<a href="../0xBuilder.py#L2446"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `fetch_historical_prices`

```python
fetch_historical_prices(token_symbol: str, days: int = 30) → List[float]
```

Fetch historical price data for a given token symbol. 

:param token_symbol: Token symbol to fetch prices for :param days: Number of days to fetch prices for :return: List of historical prices 

---

<a href="../0xBuilder.py#L2467"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_token_volume`

```python
get_token_volume(token_symbol: str) → float
```

Get the 24-hour trading volume for a given token symbol. 

:param token_symbol: Token symbol to fetch volume for :return: 24-hour trading volume 

---

<a href="../0xBuilder.py#L2527"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `is_arbitrage_opportunity`

```python
is_arbitrage_opportunity(target_tx: Dict[str, Any]) → bool
```

Check if there's an arbitrage opportunity based on the target transaction. 

:param target_tx: Target transaction dictionary :return: True if arbitrage opportunity detected, else False 

---

<a href="../0xBuilder.py#L2310"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `load_model`

```python
load_model()
```

Load the ML model and training data from disk. 

---

<a href="../0xBuilder.py#L2376"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `periodic_model_training`

```python
periodic_model_training(token_symbol: str)
```

Periodically train the ML model based on the defined interval. 

---

<a href="../0xBuilder.py#L2506"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `predict_price_movement`

```python
predict_price_movement(token_symbol: str) → float
```

Predict the next price movement for a given token symbol. 

---

<a href="../0xBuilder.py#L2324"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `save_model`

```python
save_model()
```

Save the ML model and training data to disk. 

---

<a href="../0xBuilder.py#L2391"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `start_periodic_training`

```python
start_periodic_training(token_symbol: str)
```

Start the background task for periodic model training. 


---

## <kbd>class</kbd> `Mempool_Monitor`
Advanced mempool monitoring system that identifies and analyzes profitable transactions. Includes sophisticated profit estimation, caching, and parallel processing capabilities. 

<a href="../0xBuilder.py#L894"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `__init__`

```python
__init__(
    web3: AsyncWeb3,
    safety_net: Safety_Net,
    nonce_core: Nonce_Core,
    api_config: API_Config,
    monitored_tokens: Optional[List[str]] = None,
    erc20_abi: List[Dict[str, Any]] = None,
    configuration: Optional[Configuration] = None
)
```








---

<a href="../0xBuilder.py#L1076"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `analyze_transaction`

```python
analyze_transaction(tx) → Dict[str, Any]
```





---

<a href="../0xBuilder.py#L1038"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `process_transaction`

```python
process_transaction(tx_hash: str) → None
```

Process individual transactions with enhanced error handling. 

---

<a href="../0xBuilder.py#L931"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `start_monitoring`

```python
start_monitoring() → None
```

Start monitoring the mempool with improved error handling. 

---

<a href="../0xBuilder.py#L950"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `stop_monitoring`

```python
stop_monitoring() → None
```

Gracefully stop monitoring activities. 


---

## <kbd>class</kbd> `Nonce_Core`




<a href="../0xBuilder.py#L2633"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `__init__`

```python
__init__(web3: AsyncWeb3, address: str, configuration: Configuration)
```








---

<a href="../0xBuilder.py#L2636"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `initialize`

```python
initialize()
```





---

<a href="../0xBuilder.py#L2639"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `stop`

```python
stop()
```






---

## <kbd>class</kbd> `Safety_Net`




<a href="../0xBuilder.py#L2625"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `__init__`

```python
__init__(
    web3: AsyncWeb3,
    configuration: Configuration,
    account: Account,
    api_config: API_Config
)
```








---

<a href="../0xBuilder.py#L2628"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `stop`

```python
stop()
```






---

## <kbd>class</kbd> `StrategyConfiguration`
StrategyConfiguration(decay_factor: float = 0.95, min_profit_threshold: decimal.Decimal = Decimal('0.01'), learning_rate: float = 0.01, exploration_rate: float = 0.1) 





---

## <kbd>class</kbd> `StrategyExecutionError`
Exception raised when a strategy execution fails. 





---

## <kbd>class</kbd> `StrategyPerformanceMetrics`
StrategyPerformanceMetrics(successes: int = 0, failures: int = 0, profit: decimal.Decimal = Decimal('0'), avg_execution_time: float = 0.0, success_rate: float = 0.0, total_executions: int = 0) 





---

## <kbd>class</kbd> `Strategy_Net`




<a href="../0xBuilder.py#L2676"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `__init__`

```python
__init__(
    transaction_core: Optional[ForwardRef('Transaction_Core')] = None,
    market_monitor: Optional[Market_Monitor] = None,
    safety_net: Optional[ForwardRef('Safety_Net')] = None,
    api_config: Optional[API_Config] = None,
    configuration: Optional[Configuration] = None
) → None
```








---

<a href="../0xBuilder.py#L3524"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `advanced_back_run`

```python
advanced_back_run(target_tx: Dict[str, Any]) → bool
```

Execute advanced back-run strategy with comprehensive analysis. 

---

<a href="../0xBuilder.py#L3405"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `advanced_front_run`

```python
advanced_front_run(target_tx: Dict[str, Any]) → bool
```

Execute advanced front-run strategy with comprehensive analysis, risk management, and multi-factor decision making. 

---

<a href="../0xBuilder.py#L3609"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `advanced_sandwich_attack`

```python
advanced_sandwich_attack(target_tx: Dict[str, Any]) → bool
```

Execute advanced sandwich attack strategy with risk management. 

---

<a href="../0xBuilder.py#L3008"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `aggressive_front_run`

```python
aggressive_front_run(target_tx: Dict[str, Any]) → bool
```

Execute aggressive front-run strategy with comprehensive validation, dynamic thresholds, and risk assessment. 

---

<a href="../0xBuilder.py#L3589"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `arbitrage_sandwich`

```python
arbitrage_sandwich(target_tx: Dict[str, Any]) → bool
```

Execute sandwich attack strategy based on arbitrage opportunities. 

---

<a href="../0xBuilder.py#L2749"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `execute_best_strategy`

```python
execute_best_strategy(target_tx: Dict[str, Any], strategy_type: str) → bool
```

Execute the best strategy for the given strategy type. 

:param target_tx: Target transaction dictionary. :param strategy_type: Type of strategy to execute :return: True if successful, else False. 

---

<a href="../0xBuilder.py#L3541"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `flash_profit_sandwich`

```python
flash_profit_sandwich(target_tx: Dict[str, Any]) → bool
```

Execute sandwich attack strategy using flash loans. 

---

<a href="../0xBuilder.py#L3440"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `flashloan_back_run`

```python
flashloan_back_run(target_tx: Dict[str, Any]) → bool
```

Execute back-run strategy using flash loans. 

---

<a href="../0xBuilder.py#L2745"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_strategies`

```python
get_strategies(strategy_type: str) → List[Callable[[Dict[str, Any]], Future]]
```

Retrieve strategies for a given strategy type. 

---

<a href="../0xBuilder.py#L2852"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_strategy_index`

```python
get_strategy_index(strategy_name: str, strategy_type: str) → int
```

Get the index of a strategy in the strategy list. 

---

<a href="../0xBuilder.py#L2907"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `high_value_eth_transfer`

```python
high_value_eth_transfer(target_tx: Dict[str, Any]) → bool
```

Execute high-value ETH transfer strategy with advanced validation and dynamic thresholds. 

:param target_tx: Target transaction dictionary :return: True if transaction was executed successfully, else False 

---

<a href="../0xBuilder.py#L3451"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `high_volume_back_run`

```python
high_volume_back_run(target_tx: Dict[str, Any]) → bool
```

Execute back-run strategy based on high trading volume. 

---

<a href="../0xBuilder.py#L3148"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `predictive_front_run`

```python
predictive_front_run(target_tx: Dict[str, Any]) → bool
```

Execute predictive front-run strategy based on advanced price prediction analysis and multiple market indicators. 

---

<a href="../0xBuilder.py#L3556"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `price_boost_sandwich`

```python
price_boost_sandwich(target_tx: Dict[str, Any]) → bool
```

Execute sandwich attack strategy based on price momentum. 

---

<a href="../0xBuilder.py#L3417"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `price_dip_back_run`

```python
price_dip_back_run(target_tx: Dict[str, Any]) → bool
```

Execute back-run strategy based on price dip prediction. 

---

<a href="../0xBuilder.py#L2736"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `register_strategy`

```python
register_strategy(
    strategy_type: str,
    strategy_func: Callable[[Dict[str, Any]], Future]
) → None
```

Register a new strategy dynamically. 

---

<a href="../0xBuilder.py#L3290"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `volatility_front_run`

```python
volatility_front_run(target_tx: Dict[str, Any]) → bool
```

Execute front-run strategy based on market volatility analysis with  advanced risk assessment and dynamic thresholds. 


---

## <kbd>class</kbd> `Transaction_Core`







---

<a href="../0xBuilder.py#L2611"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `back_run`

```python
back_run(target_tx: Dict[str, Any]) → bool
```





---

<a href="../0xBuilder.py#L2614"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `calculate_flashloan_amount`

```python
calculate_flashloan_amount(target_tx: Dict[str, Any]) → Decimal
```





---

<a href="../0xBuilder.py#L2617"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `execute_sandwich_attack`

```python
execute_sandwich_attack(target_tx: Dict[str, Any]) → bool
```





---

<a href="../0xBuilder.py#L2608"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `front_run`

```python
front_run(target_tx: Dict[str, Any]) → bool
```





---

<a href="../0xBuilder.py#L2602"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_current_profit`

```python
get_current_profit() → Decimal
```





---

<a href="../0xBuilder.py#L2620"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_dynamic_gas_price`

```python
get_dynamic_gas_price() → float
```





---

<a href="../0xBuilder.py#L2605"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `handle_eth_transaction`

```python
handle_eth_transaction(target_tx: Dict[str, Any]) → bool
```








---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
