import logging

from learn.Trainer import Trainer

logger = logging.getLogger('ai')
logger.setLevel(logging.DEBUG)
consoleHandler = logging.StreamHandler()
consoleHandler.setLevel(logging.DEBUG)
consoleHandler.setFormatter(
    logging.Formatter('\u001b[35;1m[%(asctime)s]\u001b[0m[%(name)s][%(levelname)s] %(message)s'))
logger.addHandler(consoleHandler)

trainer = Trainer()
trainer.run()
