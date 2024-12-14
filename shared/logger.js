// logger.js
import winston from 'winston';
import { format, transports } from 'winston';

const { combine, timestamp, printf, colorize } = format;

const customFormat = printf(({ level, message, timestamp }) => {
    return `${timestamp} [${level}] ${message}`;
});

const logger = winston.createLogger({
    level: 'info', // Global logging level
    format: combine(
        colorize(),
        timestamp({ format: 'YYYY-MM-DD HH:mm:ss' }),
        customFormat
    ),
    transports: [new transports.Console()]
});

export default logger;
