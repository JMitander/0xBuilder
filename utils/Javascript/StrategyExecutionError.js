// StrategyExecutionError.js

/**
 * Custom exception raised when a strategy execution fails.
 */
class StrategyExecutionError extends Error {
    /**
     * Initializes the StrategyExecutionError instance.
     *
     * @param {string} message - The error message.
     */
    constructor(message) {
        super(message);
        this.name = "StrategyExecutionError";
    }
}

export default StrategyExecutionError;
