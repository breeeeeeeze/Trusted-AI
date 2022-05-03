from learn.Trainer import Trainer
from utils.setupLogger import setupLogger

setupLogger('ai', 'DEBUG')

trainer = Trainer()
trainer.run()
