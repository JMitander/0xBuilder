// StrategyPerformanceMetrics.js

import { Decimal } from 'decimal.js';

/**
 * Class to track performance metrics of different strategies.
 */
class StrategyPerformanceMetrics {
    /**
     * Initializes the StrategyPerformanceMetrics instance.
     */
    constructor() {
        this.successes = 0;
        this.failures = 0;
        this.profit = new Decimal(0);
        this.avg_execution_time = 0.0;
        this.success_rate = 0.0;
        this.total_executions = 0;
    }
}

export default StrategyPerformanceMetrics;
