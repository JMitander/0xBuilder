// Semaphore.js

/**
 * Simple Semaphore Implementation
 */
class Semaphore {
    /**
     * Initializes the Semaphore with a maximum number of concurrent tasks.
     *
     * @param {number} max - The maximum number of concurrent tasks.
     */
    constructor(max) {
        this.max = max;
        this.current = 0;
        this.queue = [];
    }

    /**
     * Acquires a semaphore slot.
     *
     * @returns {Promise<void>}
     */
    acquire() {
        return new Promise(resolve => {
            if (this.current < this.max) {
                this.current++;
                resolve();
            } else {
                this.queue.push(resolve);
            }
        });
    }

    /**
     * Releases a semaphore slot.
     */
    release() {
        this.current--;
        if (this.queue.length > 0) {
            this.current++;
            const resolve = this.queue.shift();
            resolve();
        }
    }
}

export default Semaphore;
