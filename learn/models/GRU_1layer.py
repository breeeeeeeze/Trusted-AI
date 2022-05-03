from tensorflow.keras.layers import Embedding, GRU, Dense  # pyright: ignore[reportMissingImports]
from tensorflow.keras import Model  # pyright: ignore[reportMissingImports]


class RNNModel(Model):
    def __init__(self, vocabSize, embeddingSize, nUnits, **kwargs):
        super().__init__(self)
        self.embedding = Embedding(vocabSize, embeddingSize)
        self.gru = GRU(nUnits, return_sequences=True, return_state=True)
        self.dense = Dense(vocabSize)

    def call(self, inputs, states=None, returnState=False, training=False):
        x = self.embedding(inputs, training=training)
        if states is None:
            states = self.gru.get_initial_state(inputs=x)
        x, states = self.gru(x, initial_state=states, training=training)
        x = self.dense(x, training=training)

        if returnState:
            return x, states
        else:
            return x
