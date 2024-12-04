// MarketMonitor.js

import logger from './Logger.js'; //  Logger.js handles logging
import { LinearRegression } from 'ml-regression'; // You may need to install ml-regression
import { TTLCache } from 'ttl-cache'; // You may need to install ttl-cache
import joblib from 'joblib'; // Note: joblib is Python-specific. Use a similar library or implement serialization.
import { Decimal } from 'decimal.js';

class MarketMonitor {
    /**
     * The MarketMonitor class is responsible for monitoring and analyzing market data,
     * updating predictive models, and assessing market conditions such as volatility
     * and liquidity for specified tokens.
     */

    // Class-level constants defining intervals and thresholds
    MODEL_UPDATE_INTERVAL = 3600; // Update model every hour (in seconds)
    VOLATILITY_THRESHOLD = 0.05; // 5% standard deviation as volatility threshold
    LIQUIDITY_THRESHOLD = 100000; // $100,000 in 24h volume as liquidity threshold

    /**
     * Initializes the MarketMonitor instance.
     *
     * @param {Web3} web3 - The Web3 instance for blockchain interactions.
     * @param {Configuration} configuration - The configuration instance.
     * @param {APIConfig} apiconfig - The API configuration instance.
     */
    constructor(web3, configuration, apiconfig) {
        this.web3 = web3;
        this.configuration = configuration;
        this.apiconfig = apiconfig;

        // Initialize the Linear Regression model for price prediction
        this.price_model = new LinearRegression();
        this.model_last_updated = 0; // Timestamp of the last model update

        // Initialize a cache for price data with a TTL of 5 minutes and max size of 1000 entries
        this.price_cache = new TTLCache({ ttl: 300, max: 1000 });

        // Paths for saving/loading the ML model and training data
        this.model_path = this.configuration.ML_MODEL_PATH;
        this.training_data_path = this.configuration.ML_TRAINING_DATA_PATH;

        // Lock to ensure that model updates are thread-safe and prevent concurrent updates
        this.model_lock = false;

        // Asynchronously load existing model and training data if available
        this.load_model();

        logger.debug("MarketMonitor initialized with enhanced configuration.");
    }

    /**
     * Asynchronously load the machine learning model and training data from disk.
     * Utilizes a lock to prevent concurrent access during loading.
     */
    async load_model() {
        if (this.model_lock) return;
        this.model_lock = true;
        try {
            const fs = await import('fs/promises');
            if (await this._file_exists(this.model_path) && await this._file_exists(this.training_data_path)) {
                const data = await joblib.load(this.model_path);
                this.price_model = data.model;
                this.model_last_updated = data.model_last_updated || 0;
                logger.info("ML model and training data loaded successfully.");
            } else {
                logger.info("No existing ML model found. Starting fresh.");
            }
        } catch (error) {
            logger.error(`Failed to load ML model: ${error.message}`);
        } finally {
            this.model_lock = false;
        }
    }

    /**
     * Saves the machine learning model and training data to disk.
     * Utilizes a lock to ensure thread-safe operations during saving.
     */
    async save_model() {
        if (this.model_lock) return;
        this.model_lock = true;
        try {
            const fs = await import('fs/promises');
            const data = {
                model: this.price_model,
                model_last_updated: this.model_last_updated,
            };
            await joblib.dump(data, this.model_path);
            logger.info(`ML model saved to ${this.model_path}.`);
        } catch (error) {
            logger.error(`Failed to save ML model: ${error.message}`);
        } finally {
            this.model_lock = false;
        }
    }

    /**
     * Checks if a file exists.
     *
     * @param {string} path - The file path.
     * @returns {Promise<boolean>} - True if exists, else False.
     */
    async _file_exists(path) {
        const fs = await import('fs/promises');
        try {
            await fs.access(path);
            return true;
        } catch {
            return false;
        }
    }

    /**
     * Periodically train the ML model based on the defined interval.
     * This coroutine runs indefinitely, checking if the model needs an update
     * and triggering the update process accordingly.
     *
     * @param {string} token_symbol - The symbol of the token to update the model for.
     */
    async periodic_model_training(token_symbol) {
        while (true) {
            try {
                const current_time = Date.now() / 1000; // Convert to seconds
                if (current_time - this.model_last_updated > this.MODEL_UPDATE_INTERVAL) {
                    logger.debug(`Time to update ML model for ${token_symbol}.`);
                    await this._update_price_model(token_symbol);
                }
                await this._sleep(60000); // Check every minute
            } catch (error) {
                logger.error(`Error in periodic model training: ${error.message}`);
                await this._sleep(60000); // Wait a minute before retrying
            }
        }
    }

    /**
     * Starts the periodic model training for a specific token.
     *
     * @param {string} token_symbol - The symbol of the token to train the model for.
     */
    async start_periodic_training(token_symbol) {
        this.periodic_model_training(token_symbol);
    }

    /**
     * Updates the price prediction model with new historical data for a specific token.
     * Fetches historical prices, prepares training data, fits the model, and saves it.
     *
     * @param {string} token_symbol - The symbol of the token to update the model for.
     */
    async _update_price_model(token_symbol) {
        if (this.model_lock) return;
        this.model_lock = true;
        try {
            const prices = await this.fetch_historical_prices(token_symbol);
            if (prices.length > 10) {
                // Prepare training data with time steps as features
                const X = prices.map((_, index) => [index]);
                const y = prices;

                // If training data exists, load and append new data
                let X_combined = X;
                let y_combined = y;
                if (await this._file_exists(this.training_data_path)) {
                    const csv = await this.web3.utils.toHex(await this.apiconfig._load_abi(this.training_data_path));
                    // Implement CSV parsing as needed
                    // icity,  existing_data is loaded as arrays
                    const existing_data = []; // 
                    X_combined = existing_data.X.concat(X);
                    y_combined = existing_data.y.concat(y);
                }

                // Fit the Linear Regression model with the new data
                this.price_model = new LinearRegression(X_combined, y_combined);
                this.model_last_updated = Date.now() / 1000;
                logger.info(`ML model updated and trained for ${token_symbol}.`);

                // Persist the updated model to disk
                await this.save_model();
            } else {
                logger.debug(`Not enough price data to train the model for ${token_symbol}.`);
            }
        } catch (error) {
            logger.error(`Error updating price model: ${error.message}`);
        } finally {
            this.model_lock = false;
        }
    }

    /**
     * Assesses various market conditions for a given token, such as volatility,
     * trend direction, and liquidity.
     *
     * @param {string} token_address - The blockchain address of the token to check.
     * @returns {Object} - A dictionary containing the evaluated market conditions.
     */
    async check_market_conditions(token_address) {
        const market_conditions = {
            high_volatility: false,
            bullish_trend: false,
            bearish_trend: false,
            low_liquidity: false,
        };
        const token_symbol = await this.apiconfig.getTokenSymbol(this.web3, token_address);
        if (!token_symbol) {
            logger.debug(`Cannot get token symbol for address ${token_address}!`);
            return market_conditions;
        }

        const prices = await this.fetch_historical_prices(token_symbol, 1);
        if (prices.length < 2) {
            logger.debug(`Not enough price data to analyze market conditions for ${token_symbol}`);
            return market_conditions;
        }

        // Calculate volatility and assess if it exceeds the threshold
        const volatility = this._calculate_volatility(prices);
        if (volatility > this.VOLATILITY_THRESHOLD) {
            market_conditions.high_volatility = true;
        }
        logger.debug(`Calculated volatility for ${token_symbol}: ${volatility}`);

        // Determine trend based on the moving average
        const moving_average = prices.reduce((a, b) => a + b, 0) / prices.length;
        if (prices[prices.length - 1] > moving_average) {
            market_conditions.bullish_trend = true;
        } else if (prices[prices.length - 1] < moving_average) {
            market_conditions.bearish_trend = true;
        }

        // Assess liquidity based on trading volume
        const volume = await this.get_token_volume(token_symbol);
        if (volume < this.LIQUIDITY_THRESHOLD) {
            market_conditions.low_liquidity = true;
        }

        return market_conditions;
    }

    /**
     * Calculates the volatility of a list of prices as the standard deviation of returns.
     *
     * @param {Array<number>} prices - A list of historical prices.
     * @returns {number} - The calculated volatility.
     */
    _calculate_volatility(prices) {
        const price_changes = [];
        for (let i = 1; i < prices.length; i++) {
            price_changes.push((prices[i] - prices[i - 1]) / prices[i - 1]);
        }
        const mean = price_changes.reduce((a, b) => a + b, 0) / price_changes.length;
        const variance = price_changes.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / price_changes.length;
        const std_dev = Math.sqrt(variance);
        return std_dev;
    }

    /**
     * Fetches historical price data for a given token symbol.
     *
     * @param {string} token_symbol - The symbol of the token to fetch prices for.
     * @param {number} [days=30] - Number of days to fetch prices for.
     * @returns {Array<number>} - A list of historical prices.
     */
    async fetch_historical_prices(token_symbol, days = 30) {
        const cache_key = `historical_prices_${token_symbol}_${days}`;
        if (this.price_cache.has(cache_key)) {
            logger.debug(`Returning cached historical prices for ${token_symbol}.`);
            return this.price_cache.get(cache_key);
        }

        const prices = await this._fetch_from_services(
            async (service) => await this.apiconfig.fetchHistoricalPrices(token_symbol, days),
            `historical prices for ${token_symbol}`
        );

        if (prices && prices.length > 0) {
            this.price_cache.set(cache_key, prices);
            return prices;
        }

        return [];
    }

    /**
     * Retrieves the 24-hour trading volume for a specified token.
     *
     * @param {string} token_symbol - The symbol of the token to fetch volume for.
     * @returns {number} - The 24-hour trading volume in USD.
     */
    async get_token_volume(token_symbol) {
        const cache_key = `token_volume_${token_symbol}`;
        if (this.price_cache.has(cache_key)) {
            logger.debug(`Returning cached trading volume for ${token_symbol}.`);
            return this.price_cache.get(cache_key);
        }

        const volume = await this._fetch_from_services(
            async (service) => await this.apiconfig.getTokenVolume(token_symbol),
            `trading volume for ${token_symbol}`
        );

        if (volume !== null) {
            this.price_cache.set(cache_key, volume);
            return volume;
        }

        return 0.0;
    }

    /**
     * Helper method to fetch data from multiple configured services.
     *
     * @param {Function} fetch_func - The function to fetch data from a service.
     * @param {string} description - Description of the data being fetched.
     * @returns {any} - The fetched data or null if all services fail.
     */
    async _fetch_from_services(fetch_func, description) {
        const services = Object.keys(this.apiconfig.apiConfigs);
        for (const service of services) {
            try {
                logger.debug(`Fetching ${description} using ${service}...`);
                const result = await fetch_func(service);
                if (result) {
                    return result;
                }
            } catch (error) {
                logger.warn(`Failed to fetch ${description} using ${service}: ${error.message}`);
            }
        }
        logger.warn(`Failed to fetch ${description}.`);
        return null;
    }

    /**
     * Predicts the next price movement for a given token symbol using the ML model.
     *
     * @param {string} token_symbol - The symbol of the token to predict price movement for.
     * @returns {number} - The predicted price movement.
     */
    async predict_price_movement(token_symbol) {
        const current_time = Date.now() / 1000; // Convert to seconds
        if (current_time - this.model_last_updated > this.MODEL_UPDATE_INTERVAL) {
            logger.debug(`Model needs updating for ${token_symbol}. Triggering update.`);
            await this._update_price_model(token_symbol);
        }

        const prices = await this.fetch_historical_prices(token_symbol, 1);
        if (!prices.length) {
            logger.debug(`No recent prices available for ${token_symbol}.`);
            return 0.0;
        }

        try {
            // Predict the next price based on the current data
            const X_pred = [[prices.length]];
            const predicted_price = this.price_model.predict(X_pred)[0];
            logger.debug(`Price prediction for ${token_symbol}: ${predicted_price}`);
            return predicted_price;
        } catch (error) {
            logger.error(`Error predicting price movement: ${error.message}`);
            return 0.0;
        }
    }

    /**
     * Determines if there's an arbitrage opportunity based on the target transaction.
     *
     * @param {Object} target_tx - The target transaction dictionary.
     * @returns {boolean} - True if an arbitrage opportunity is detected, else False.
     */
    async is_arbitrage_opportunity(target_tx) {
        const decoded_tx = await this.decode_transaction_input(target_tx.input, this.web3.utils.toChecksumAddress(target_tx.to));
        if (!decoded_tx) {
            return false;
        }

        const path = decoded_tx.params.path || [];
        if (path.length < 2) {
            return false;
        }

        const token_address = path[path.length - 1]; // The token being bought
        const token_symbol = await this.apiconfig.getTokenSymbol(this.web3, token_address);
        if (!token_symbol) {
            return false;
        }

        // Fetch real-time prices from different services
        const prices = await this._get_prices_from_services(token_symbol);
        if (prices.length < 2) {
            return false;
        }

        // Calculate the difference and percentage to assess arbitrage potential
        const price_difference = Math.abs(prices[0] - prices[1]);
        const average_price = prices.reduce((a, b) => a + b, 0) / prices.length;
        if (average_price === 0) {
            return false;
        }

        const price_difference_percentage = price_difference / average_price;
        if (price_difference_percentage > 0.01) { // Arbitrage threshold set at 1%
            logger.debug(`Arbitrage opportunity detected for ${token_symbol}`);
            return true;
        }
        return false;
    }

    /**
     * Retrieves real-time prices for a token from different services.
     *
     * @param {string} token_symbol - The symbol of the token to get prices for.
     * @returns {Array<number>} - A list of real-time prices from various services.
     */
    async _get_prices_from_services(token_symbol) {
        const prices = [];
        const services = Object.keys(this.apiconfig.apiConfigs);
        for (const service of services) {
            try {
                const price = await this.apiconfig.getRealTimePrice(token_symbol.toLowerCase());
                if (price !== null) {
                    prices.push(price);
                }
            } catch (error) {
                logger.warn(`Failed to get price from ${service}: ${error.message}`);
            }
        }
        return prices;
    }
}

export default MarketMonitor;
