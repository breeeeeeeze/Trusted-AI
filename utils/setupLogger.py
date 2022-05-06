import logging


def setupLogger(name, level='INFO'):
    level = getattr(logging, level.upper())
    if name == 'tf':
        from tensorflow import get_logger

        logger = get_logger()
    else:
        logger = logging.getLogger(name)
    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            '\u001b[35;1m[%(asctime)s]\u001b[0m[%(name)s][%(levelname)s] %(message)s'
        )
    )
    logger.addHandler(handler)
    return logger
