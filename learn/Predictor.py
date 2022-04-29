import tensorflow as tf


class Predictor(tf.keras.Model):
    def __init__(self, model, IDToChar, charToID, temperature=1.0):
        super().__init__()
        self.model = model
        self.IDToChar = IDToChar
        self.charToID = charToID
        self.temperature = temperature

        skipIDs = self.charToID(['[UNK]'])[:, None]
        sparseMask = tf.SparseTensor(
            values=[-float('inf')]*len(skipIDs),
            indices=skipIDs,
            dense_shape=[len(self.charToID.get_vocabulary())]
        )
        self.skipMask = tf.sparse.to_dense(sparseMask)

    @tf.function
    def predictNextChar(self, inputs, states=None, temperature=None):
        if not temperature:
            temperature = self.temperature
        inputChars = tf.strings.unicode_split(inputs, 'UTF-8')
        inputIDs = self.charToID(inputChars).to_tensor()

        predictedLogits, states = self.model(inputIDs, states=states, returnState=True)
        predictedLogits = predictedLogits[:, -1, :]
        predictedLogits /= temperature
        predictedLogits += self.skipMask

        predictedIDs = tf.random.categorical(predictedLogits, num_samples=1)
        predictedIDs = tf.squeeze(predictedIDs, axis=-1)

        predictedChars = self.IDToChar(predictedIDs)

        return predictedChars, states
