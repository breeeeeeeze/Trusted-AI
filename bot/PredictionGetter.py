import logging

from learn.TrustedRNN import TrustedRNN
from utils.configReader import readConfig
from utils.colorizer import colorize

logger = logging.getLogger('ai.bot.predictiongetter')
config = readConfig()


class PredictionGetter:
    models = None

    @staticmethod
    def makeModels():
        models = {}
        for model in config['prediction']['models']:
            models[model['name']] = PredictionGetter.makeModel(model)
        PredictionGetter.models = models
        logger.debug(PredictionGetter.models)

    @staticmethod
    def makeModel(modelDict):
        modelName = modelDict['name']
        modelType = modelDict['model']
        logger.info(f'{colorize("Making model", "OKGREEN")} {modelName}')
        vocabPath = config['prediction']['vocabPath'].replace('{runName}', modelName)
        vocab = PredictionGetter.loadVocab(vocabPath)
        return TrustedRNN(vocab,
                          runName=modelName,
                          modelType=modelType,
                          loadFromWeights=True)

    @staticmethod
    def loadVocab(fileName):
        try:
            with open(f'{fileName}.txt', 'r', encoding='utf-8') as f:
                vocab = list(f.read())
        except FileNotFoundError:
            logger.error(f'{colorize("Vocab file not found", "FAIL")} {fileName}')
            raise FileNotFoundError(f'Vocabulary file {fileName}.txt not found.')
        logger.debug('Vocabulary imported.')
        return vocab

    @staticmethod
    async def predict(modelName, seed, temperature):
        logger.debug(f'Predicting {modelName} with seed {seed} and temperature {temperature}')
        return await PredictionGetter.models[modelName].predict(seed, temperature)
