// Dataclass.js

/**
 * Simple dataclass equivalent in JavaScript using classes.
 */
class Dataclass {
    /**
     * Initializes the Dataclass instance with given properties.
     *
     * @param {Object} props - The properties to initialize.
     */
    constructor(props = {}) {
        Object.assign(this, props);
    }
}

export default Dataclass;
