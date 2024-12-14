// StrategyConfiguration.js

import { Decimal } from 'decimal.js';

/**
 * Class to hold configuration parameters for strategy execution and learning.
 */
class StrategyConfiguration {
    /**
     * Initializes the StrategyConfiguration instance with default parameters.
     *
     * @param {number} [decay_factor=0.95] - Decay factor for exponential moving averages.
     * @param {Decimal} [min_profit_threshold=new Decimal("0.01")] - Minimum profit threshold in ETH.
     * @param {number} [learning_rate=0.01] - Learning rate for reinforcement learning.
     * @param {number} [exploration_rate=0.1] - Exploration rate for strategy selection.
     */
    constructor(
        decay_factor = 0.95,
        min_profit_threshold = new Decimal("0.01"),
        learning_rate = 0.01,
        exploration_rate = 0.1
    ) {
        this.decay_factor = decay_factor;
        this.min_profit_threshold = min_profit_threshold;
        this.learning_rate = learning_rate;
        this.exploration_rate = exploration_rate;
    }
}

export default StrategyConfiguration;
