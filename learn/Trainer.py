import logging

from learn.DataProcessor import DataProcessor
from learn.TrustedRNN import TrustedRNN
from utils.configReader import readConfig
from utils.colorizer import colorize

config = readConfig()
config = config['training']['run']

logger = logging.getLogger('ai.learn.trainer')


class Trainer:

    def __init__(self):
        self.dp = DataProcessor()
        self.rnn = None

    def run(self):
        try:
            logger.info(colorize('Starting full training sequence', 'BLUE', 'BACKGROUND_WHITE'))
            text, vocab = self.dp.processInputData()
            self.dp.exportVocab(f'{config["vocabPath"]}vocab_{config["runName"]}.txt')
            self.rnn = TrustedRNN(vocab, text)
            self.rnn.makeDataset()
            self.rnn.makeModel()
            self.rnn.trainModel()
            self.rnn.saveModelWeights(f'final_weights_{config["runName"]}')
            if config['training']['pickleHistory']:
                self.rnn.pickleHistory(f'history_{config["runName"]}')
        except Exception:
            logger.error(colorize(f'{Exception}', 'FAIL'))
            raise
