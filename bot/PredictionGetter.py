import logging

from learn.TrustedRNN import TrustedRNN
from utils.configReader import readConfig
from utils.colorizer import colorize

logger = logging.getLogger('ai.bot.predictiongetter')
config = readConfig()


class PredictionGetter:
    def __init__(self, activate: bool = False):
        self.builtModels = {}
        self.active = activate
        self.modelSettings = config['prediction']['models']
        if self.active:
            logger.info(f'{colorize("Predictor", "OKBLUE")} is {colorize("active", "GREEN")}')
            logger.info(colorize('Initializing models...', 'OKBLUE'))
            self.initializeModels()
        else:
            logger.info(f'{colorize("Predictor", "OKBLUE")} is {colorize("inactive", "RED")}')

    def activate(self):
        if not self.active:
            self.active = True
            self.initializeModels()
            logger.info(f'{colorize("Predictor", "OKBLUE")} is {colorize("active", "GREEN")}')

    def deactivate(self):
        if self.active:
            self.active = False
            self.builtModels = {}
            logger.info(f'{colorize("Predictor", "OKBLUE")} is {colorize("inactive", "RED")}')

    def initializeModels(self):
        self.builtModels = {}
        for model in self.modelSettings:
            builtModel = self.buildModel(model)
            if not builtModel:
                logger.warning(colorize(f'{model["name"]} failed to build, skipping.', 'WARNING'))
            self.builtModels[model['name']] = builtModel
        logger.info(colorize('Models initialized', 'OKBLUE'))

    def buildModel(self, model):
        logger.info(f'{colorize("Building model", "OKGREEN")} {model["name"]}')
        vocab = self.loadVocab(model)
        if not vocab:
            logger.warning(colorize(f'{model["name"]} has no vocab, skipping.', 'WARNING'))
            return
        rnn = TrustedRNN(vocab, runName=model['name'], **model['options'])
        rnn.loadFromWeights()
        return rnn

    def loadVocab(self, model):
        filename = config['prediction']['vocabPath'].replace('{runName}', model['name'])
        try:
            with open(f'{filename}.txt', 'r', encoding='utf-8') as f:
                vocab = list(f.read())
        except FileNotFoundError:
            logger.error(
                f'{colorize("Vocab file not found", "FAIL")} '
                f'{filename}. '
                f'{colorize("Provide vocab file or remove model", "FAIL")}'
            )
            return
        logger.debug('Vocabulary imported.')
        return vocab

    def predict(self, modelName, seed, temperature):
        logger.debug(f'Predicting {modelName} with seed {seed} and temperature {temperature}')
        prediction = self.builtModels[modelName].predict(seed, temperature)
        if prediction.startswith('\n'):
            prediction = prediction[1:]
        return prediction
