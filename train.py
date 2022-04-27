import logging

from learn.DataProcessor import DataProcessor
from learn.TrustedRNN import TrustedRNN

logger = logging.getLogger('ai')
logger.setLevel(logging.DEBUG)
consoleHandler = logging.StreamHandler()
consoleHandler.setLevel(logging.DEBUG)
consoleHandler.setFormatter(
    logging.Formatter('\u001b[35;1m[%(asctime)s]\u001b[0m[%(name)s][%(levelname)s] %(message)s'))
logger.addHandler(consoleHandler)

dp = DataProcessor()
text, vocab = dp.processInputData()
rnn = TrustedRNN(vocab, text)
rnn.makeDataset()
rnn.makeModel()
rnn.trainModel()
rnn.makePredictor()
rnn.saveModelWeights('final_weights')
