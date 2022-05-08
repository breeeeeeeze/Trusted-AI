from learn.Trainer import Trainer
from utils.setupLogger import setupLogger
from utils.configReader import readConfig

config = readConfig()
setupLogger('ai', config['learn']['run']['logLevel'])

trainer = Trainer()
trainer.run()
