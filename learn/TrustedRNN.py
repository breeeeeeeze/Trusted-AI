import logging
import os

import tensorflow as tf

from learn.RNNModel import RNNModel
from learn.Predictor import Predictor
from utils.configReader import readConfig
from utils.colorizer import colorize

config = readConfig()
logger = logging.getLogger('ai.learn.trustedrnn')


class TrustedRNN:
    def __init__(self, vocab, text=None):
        self.batchSize = config['training']['learn']['batchSize']
        self.nUnits = config['training']['learn']['nUnits']
        self.seqLength = config['training']['learn']['seqLength']
        self.bufferSize = config['training']['learn']['bufferSize']
        self.nEpochs = config['training']['learn']['nEpochs']
        self.verbose = config['training']['learn']['verbose']
        self.embeddingSize = config['training']['learn']['embeddingSize']
        self.optimizer = config['training']['learn']['optimizer']
        self.checkpointPath = config['training']['learn']['checkpointPath']
        self.checkpointPrefix = config['training']['learn']['checkpointPrefix']
        self.checkpointFreq = config['training']['learn']['checkpointFreq']
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
        self.model = RNNModel(
            len(self.charToID.get_vocabulary()),
            self.embeddingSize,
            self.nUnits)
        for i, _ in self.dataset.take(1):
            _ = self.model(i)
        print(self.model.summary())
        self.model.compile(optimizer=self.optimizer, loss=self.loss(), metrics=['accuracy'])
        logger.info(f'{colorize("Model created", "OKGREEN")}')

    def loss(self):
        logger.debug('Using SparseCategoricalCrossentropy')
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def checkpointCallback(self):
        saveFreq = len(self.dataset) * self.checkpointFreq
        filepath = os.path.join(
            self.checkpointPath,
            self.checkpointPrefix)
        return tf.keras.callbacks.ModelCheckpoint(
            filepath=filepath,
            save_weights_only=True,
            save_freq=saveFreq)

    def trainModel(self):
        if not self.model or not self.dataset:
            logger.error(f'{colorize("Model or dataset not created", "FAIL")}')
            return
        logger.info(f'{colorize("Training model", "OKBLUE")}')
        self.history = self.model.fit(
            self.dataset,
            epochs=self.nEpochs,
            callbacks=[self.checkpointCallback()],
            verbose=self.verbose)

    def saveModelWeights(self, filename):
        if not self.model:
            logger.error(f'{colorize("Model not created", "FAIL")}')
            return
        filepath = os.path.join(
            self.checkpointPath,
            filename)
        self.model.save_weights(filepath)
        logger.info(f'{colorize("Model weights saved", "OKBLUE")}')

    def makePredictor(self):
        if not self.model:
            logger.error(f'{colorize("Model not created", "FAIL")}')
            return
        self.predictor = Predictor(self.model, self.IDToChar, self.charToID)
        logger.info(f'{colorize("Predictor created", "OKGREEN")}')

    def predict(self, seed):
        states = None
        nextChar = tf.constant([seed])
        result = []

        for _ in range(1000):
            nextChar, states = self.predictor.predictNextChar(nextChar, states)
            result.append(nextChar)
            if len(result) > 2 and nextChar == '\n':
                break

        result = (seed + tf.strings.join(result))[0].numpy().decode('utf-8')
        logger.info(f'{colorize("Prediction:", "OKGREEN")} {colorize(result, "OKCYAN")}')
        return result
