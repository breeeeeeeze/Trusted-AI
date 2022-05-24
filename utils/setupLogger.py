import logging

from utils.configReader import readConfig

config = readConfig()


def setupLogger(name: str, level='INFO') -> logging.Logger:
    """
    Set up a logger with the given name and level.
    """
    loggerLevel = getattr(logging, level.upper())
    if name == 'tf':
        from tensorflow import get_logger

        logger = get_logger()
    else:
        logger = logging.getLogger(name)
    logger.setLevel(loggerLevel)
    if not config['log']['logToConsole'] and not config['log']['logToFile']:
        raise Exception('No log destination specified')
    if config['log']['logToFile']:
        handler = logging.FileHandler(config['log']['logFile'])
        handler.setFormatter(logging.Formatter(config['log']['logFormat']))
        logger.addHandler(handler)
    if config['log']['logToConsole']:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(config['log']['logFormat']))
        logger.addHandler(handler)
    return logger
