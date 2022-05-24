import logging
from typing import Optional

from learn.DataProcessor import DataProcessor
from learn.TrustedRNN import TrustedRNN
from utils.configReader import readConfig
from utils.colorizer import colorize

config = readConfig()
config = config['learn']['run']

logger = logging.getLogger('ai.learn.trainer')


class Trainer:
    def __init__(self):
        self.dp = DataProcessor()
        self.rnn: Optional[TrustedRNN] = None

    def run(self) -> None:
        try:
            logger.info(colorize('Starting full training sequence', 'BLUE', 'BACKGROUND_WHITE'))
            text, vocab = self.dp.processInputData()
            self.dp.exportVocab(f'{config["vocabPath"]}vocab_{config["runName"]}.txt')
            logger.info(
                colorize(
                    'Data processing completed. Starting training...',
                    'BLUE',
                    'BACKGROUND_WHITE',
                )
            )
            self.rnn = TrustedRNN(vocab, text, runName=config['runName'])
            self.rnn.makeDataset()
            self.rnn.makeModel()
            self.rnn.trainModel()
            logger.info(colorize('Finished training, saving results...', 'BLUE', 'BACKGROUND_WHITE'))
            self.rnn.saveModelWeights(f'final_weights_{config["runName"]}')
            if config['pickleHistory']:
                self.rnn.pickleHistory(f'history_{config["runName"]}')
            self.rnn.makePredictor()
            logger.info(colorize('Making some predictions to check', 'BLUE', 'BACKGROUND_WHITE'))
            for _ in range(20):
                self.rnn.predict('\n')
        except Exception as err:
            logger.error(colorize(f'{err}', 'FAIL'))
            raise
