// Errors.js

/**
 * Custom error class for contract logic errors.
 */
class ContractLogicError extends Error {
    constructor(message) {
        super(message);
        this.name = "ContractLogicError";
    }
}

/**
 * Custom error class for transaction not found errors.
 */
class TransactionNotFound extends Error {
    constructor(message) {
        super(message);
        this.name = "TransactionNotFound";
    }
}

export { ContractLogicError, TransactionNotFound };
