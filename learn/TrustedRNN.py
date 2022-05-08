import logging
from os import path
from importlib import import_module
from pickle import dump

import tensorflow as tf

from learn.Predictor import Predictor
from utils.configReader import readConfig
from utils.colorizer import colorize

config = readConfig()
config = config['learn']
logger = logging.getLogger('ai.learn.trustedrnn')


class TrustedRNN:
    def __init__(
        self,
        vocab,
        text=None,
        loadFromWeights=False,
        **kwargs,
    ):

        self.runName = config['run']['runName']
        self.modelType = config['model']['modelType']
        self.batchSize = config['training']['batchSize']
        self.nUnits = config['model']['nUnits']
        self.seqLength = config['model']['seqLength']
        self.bufferSize = config['model']['bufferSize']
        self.nEpochs = config['training']['nEpochs']
        self.verbose = config['training']['verbose']
        self.embeddingSize = config['model']['embeddingSize']
        self.optimizer = config['training']['optimizer']
        self.checkpointPath = config['training']['checkpoints']['path']
        self.checkpointPrefix = config['training']['checkpoints']['prefix']
        self.checkpointSaveBestOnly = config['training']['checkpoints']['saveBestOnly']
        self.checkpointMonitor = config['training']['checkpoints']['monitor']
        self.layers = config['model']['layers']
        self.text = text
        self.vocab = vocab
        self.dataset = None
        self.model = None
        self.history = None
        self.predictor = None
        self.useEarlyStopping = config['training']['earlyStopping']['useEarlyStopping']
        self.earlyStoppingPatience = config['training']['earlyStopping']['patience']
        self.earlyStoppingRestoreBestWeights = config['training']['earlyStopping']['restoreBestWeights']
        self.earlyStoppingMonitor = config['training']['earlyStopping']['monitor']
        for option in kwargs.keys():
            if hasattr(self, option) and isinstance(getattr(self, option), type(kwargs[option])):
                setattr(self, option, kwargs[option])
            else:
                logger.warning(colorize(f'{option} is not a valid option, using default.', 'WARNING'))
        self.checkpointPrefix += f'_{self.runName}'

        # dynamically import the RNN model
        self.RNNModel = import_module(f'learn.models.{self.modelType}')
        self.RNNModel = self.RNNModel.RNNModel

        logger.info(f'{colorize("TrustedRNN initialized", "OKGREEN")}')

        self.charToID = tf.keras.layers.StringLookup(vocabulary=self.vocab, mask_token=None)
        self.IDToChar = tf.keras.layers.StringLookup(
            vocabulary=self.charToID.get_vocabulary(), invert=True, mask_token=None
        )

        if loadFromWeights and self.runName:
            self.loadWithWeights()

    def loadWithWeights(self):
        logger.debug('Loading model with weights')
        self.makeModel()
        self.loadModelWeights(self.checkpointPrefix)
        self.makePredictor()

    def makeDataset(self):
        if not self.text:
            logger.error(f'{colorize("No text passed", "FAIL")}')
            return

        def splitInputTarget(sequence):
            input = sequence[:-1]
            target = sequence[1:]
            return input, target

        allIDs = self.charToID(tf.strings.unicode_split(self.text, 'UTF-8'))
        sequences = tf.data.Dataset.from_tensor_slices(allIDs)
        sequences = sequences.batch(self.seqLength + 1, drop_remainder=True)
        self.dataset = sequences.map(splitInputTarget)
        self.dataset = (
            self.dataset.shuffle(self.bufferSize)
            .batch(self.batchSize, drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )
        logger.info(f'{colorize("Dataset created", "OKGREEN")}')

    def makeModel(self):
        layers = None
        if self.modelType == 'LSTM_multilayer' or self.modelType == 'BiLSTM_multilayer':
            layers = self.layers
            logger.debug(f'Calling dynamic models with {layers} layers')
        self.model = self.RNNModel(
            len(self.charToID.get_vocabulary()),
            self.embeddingSize,
            self.nUnits,
            layers=layers,
        )
        self.model.build(input_shape=(self.batchSize, self.seqLength))
        self.model.summary()
        self.model.compile(optimizer=self.optimizer, loss=self.loss(), metrics=['accuracy'])
        logger.info(f'{colorize("Model created", "OKGREEN")}')

    def loss(self):
        logger.debug('Using SparseCategoricalCrossentropy')
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def checkpointCallback(self):
        filepath = path.join(self.checkpointPath, self.checkpointPrefix)
        return tf.keras.callbacks.ModelCheckpoint(
            filepath=filepath,
            monitor=self.checkpointMonitor,
            save_best_only=self.checkpointSaveBestOnly,
            save_weights_only=True,
        )

    def earlyStoppingCallback(self):
        return tf.keras.callbacks.EarlyStopping(
            monitor=config['training']['earlyStopping']['monitor'],
            patience=config['training']['earlyStopping']['patience'],
            restore_best_weights=config['training']['earlyStopping']['restoreBestWeights'],
        )

    def tensorboardCallback(self):
        return tf.keras.callbacks.TensorBoard(
            log_dir=f'tb_logs/{self.runName}',
            histogram_freq=1,
            write_graph=True,
            write_images=False,
            update_freq='epoch',
        )

    def getCallbacks(self):
        callbacks = []
        callbacks.append(self.checkpointCallback())
        if config['training']['earlyStopping']['useEarlyStopping']:
            callbacks.append(self.earlyStoppingCallback())
        callbacks.append(self.tensorboardCallback())
        return callbacks

    def trainModel(self):
        if not self.model or not self.dataset:
            logger.error(f'{colorize("Model or dataset not created", "FAIL")}')
            return
        logger.info(f'{colorize("Training model", "OKBLUE")}')
        self.history = self.model.fit(
            self.dataset,
            epochs=self.nEpochs,
            callbacks=[self.getCallbacks()],
            verbose=self.verbose,
        )

    def saveModelWeights(self, filename):
        if not self.model:
            logger.error(f'{colorize("Model not created", "FAIL")}')
            return
        filepath = path.join(self.checkpointPath, filename)
        self.model.save_weights(filepath)
        logger.info(f'{colorize("Model weights saved", "OKBLUE")}')

    def loadModelWeights(self, filename):
        if not self.model:
            logger.error(f'{colorize("Model not created", "FAIL")}')
            return
        filepath = path.join(self.checkpointPath, filename)
        self.model.load_weights(filepath).expect_partial()
        logger.info(f'{colorize("Model weights loaded", "OKBLUE")}')

    def makePredictor(self):
        if not self.model:
            logger.error(f'{colorize("Model not created", "FAIL")}')
            return
        self.predictor = Predictor(self.model, self.IDToChar, self.charToID)
        logger.info(f'{colorize("Predictor created", "OKGREEN")}')

    def pickleHistory(self, filename):
        if not self.history:
            string = 'History doesn\'t exist, train the model first'
            logger.error(f'{colorize(string, "FAIL")}')
            return
        with open(filename, 'wb') as f:
            dump(self.history, f)

    def predict(self, seed, temperature=None):
        if not self.predictor:
            logger.error(f'{colorize("No predictor found, make predictor first", "FAIL")}')
            return
        seed = seed.lower()
        states = None
        nextChar = tf.constant([seed])
        result = []

        for _ in range(2000):
            nextChar, states = self.predictor.predictNextChar(nextChar, states, temperature=temperature)
            result.append(nextChar)
            if nextChar == '\n':
                if len(result) > 3:
                    break
                else:
                    seed = ''
                    result = []
            # maximum discord message length is 2000
            if len(seed) + len(result) >= 1999:
                break

        result = (seed + tf.strings.join(result))[0].numpy().decode('utf-8').strip()
        logger.debug(f'{colorize("Prediction:", "OKGREEN")} {colorize(result, "OKCYAN")}')
        return result
