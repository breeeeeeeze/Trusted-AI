import logging
from os import path
from importlib import import_module
from pickle import dump

import tensorflow as tf

from learn.Predictor import Predictor
from utils.configReader import readConfig
from utils.colorizer import colorize

config = readConfig()
config = config['training']['learn']
logger = logging.getLogger('ai.learn.trustedrnn')


class TrustedRNN:
    def __init__(self, vocab, text=None, runName='', modelType=None, loadFromWeights=False):
        # janky way to dynamically import the RNN model
        self.modelType = modelType
        if not modelType:
            self.modelType = config['model']
        self.RNNModel = import_module(f'learn.models.{self.modelType}')
        self.RNNModel = self.RNNModel.RNNModel

        self.runName = runName
        self.batchSize = config['batchSize']
        self.nUnits = config['nUnits']
        self.seqLength = config['seqLength']
        self.bufferSize = config['bufferSize']
        self.nEpochs = config['nEpochs']
        self.verbose = config['verbose']
        self.embeddingSize = config['embeddingSize']
        self.optimizer = config['optimizer']
        self.checkpointPath = config['checkpointPath']
        self.checkpointPrefix = config['checkpointPrefix'] + '_' + self.runName
        self.checkpointFreq = config['checkpointFreq']
        self.layers = config['layers']
        self.text = text
        self.vocab = vocab
        self.dataset = None
        self.model = None
        self.history = None
        self.predictor = None
        logger.info(f'{colorize("TrustedRNN initialized", "OKGREEN")}')

        self.charToID = tf.keras.layers.StringLookup(vocabulary=self.vocab, mask_token=None)
        self.IDToChar = tf.keras.layers.StringLookup(
            vocabulary=self.charToID.get_vocabulary(),
            invert=True,
            mask_token=None)

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
        self.dataset = self.dataset \
            .shuffle(self.bufferSize) \
            .batch(self.batchSize, drop_remainder=True) \
            .prefetch(tf.data.experimental.AUTOTUNE)
        logger.info(f'{colorize("Dataset created", "OKGREEN")}')

    def makeModel(self):
        layers = None
        if self.modelType == 'LSTM_dynamiclayer':
            layers = self.layers
            logger.debug(f'Calling dynamic models with {layers} layers')
        self.model = self.RNNModel(
            len(self.charToID.get_vocabulary()),
            self.embeddingSize,
            self.nUnits,
            layers=layers)
        self.model.build(input_shape=(self.batchSize, self.seqLength))
        self.model.summary()
        self.model.compile(optimizer=self.optimizer, loss=self.loss(), metrics=['accuracy'])
        logger.info(f'{colorize("Model created", "OKGREEN")}')

    def loss(self):
        logger.debug('Using SparseCategoricalCrossentropy')
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def checkpointCallback(self):
        filepath = path.join(
            self.checkpointPath,
            self.checkpointPrefix)
        return tf.keras.callbacks.ModelCheckpoint(
            filepath=filepath,
            monitor='accuracy',
            save_best_only=True,
            save_weights_only=True,
        )

    def earlyStoppingCallback(self):
        return tf.keras.callbacks.EarlyStopping(
            monitor=config['earlyStopping']['monitor'],
            patience=config['earlyStopping']['patience'],
            restore_best_weights=config['earlyStopping']['restoreBestWeights']
        )

    def tensorboardCallback(self):
        return tf.keras.callbacks.TensorBoard(
            log_dir=f'tb_logs/{self.runName}',
            histogram_freq=1,
            write_graph=True,
            write_images=False,
            update_freq='epoch'
        )

    def getCallbacks(self):
        callbacks = []
        callbacks.append(self.checkpointCallback())
        if config['earlyStopping']['useEarlyStopping']:
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
            verbose=self.verbose)

    def saveModelWeights(self, filename):
        if not self.model:
            logger.error(f'{colorize("Model not created", "FAIL")}')
            return
        filepath = path.join(
            self.checkpointPath,
            filename)
        self.model.save_weights(filepath)
        logger.info(f'{colorize("Model weights saved", "OKBLUE")}')

    def loadModelWeights(self, filename):
        if not self.model:
            logger.error(f'{colorize("Model not created", "FAIL")}')
            return
        filepath = path.join(
            self.checkpointPath,
            filename)
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

        for _ in range(1000):
            nextChar, states = self.predictor.predictNextChar(nextChar,
                                                              states,
                                                              temperature=temperature)
            result.append(nextChar)
            if len(result) > 0 and nextChar == '\n':
                break

        result = (seed + tf.strings.join(result))[0].numpy().decode('utf-8').strip()
        logger.debug(f'{colorize("Prediction:", "OKGREEN")} {colorize(result, "OKCYAN")}')
        return result
