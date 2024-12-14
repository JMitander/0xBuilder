// loadingBar.js
import logger from './logger.js';

export async function loadingBar(message, totalTime, successMessage = null) {
    const YELLOW = "\x1b[33m";
    const GREEN = "\x1b[32m";
    const RESET = "\x1b[0m";

    const barLength = 20;

    for (let i = 0; i <= 100; i++) {
        await new Promise(resolve => setTimeout(resolve, (totalTime * 1000) / 100));
        const percent = i / 100;
        const filledLength = Math.floor(percent * barLength);
        const bar = '█'.repeat(filledLength) + '-'.repeat(barLength - filledLength);
        process.stdout.write(`\r${GREEN}${message} [${bar}] ${i}%${RESET}`);
        process.stdout.flush();
    }
    process.stdout.write("\n");
    process.stdout.flush();

    if (successMessage) {
        logger.debug(`${YELLOW}${successMessage}${RESET}`);
    }
}
