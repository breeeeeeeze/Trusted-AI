import logging

from utils.configReader import readConfig

config = readConfig()


def setupLogger(name, level='INFO'):
    level = getattr(logging, level.upper())
    if name == 'tf':
        from tensorflow import get_logger

        logger = get_logger()
    else:
        logger = logging.getLogger(name)
    logger.setLevel(level)
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
