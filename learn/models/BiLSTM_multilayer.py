from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional  # pyright: ignore[reportMissingImports] # noqa: E501
from tensorflow.keras import Model  # pyright: ignore[reportMissingImports]


class RNNModel(Model):
    def __init__(self, vocabSize, embeddingSize, nUnits, **kwargs):
        if not kwargs['layers']:
            raise ValueError('No layer count passed to RNNModel')
        self.layerCount = kwargs['layers']
        super().__init__(self)
        self.embedding = Embedding(vocabSize, embeddingSize)
        self.lstm = []
        for _ in range(self.layerCount):
            self.lstm.append(Bidirectional(LSTM(nUnits, return_sequences=True, return_state=True)))
        self.dense = Dense(vocabSize)

    def call(self, inputs, states=None, returnState=False, training=False):
        outState = []
        x = self.embedding(inputs, training=training)
        for i in range(self.layerCount):
            if states is None:
                outState.append(self.lstm[i].get_initial_state(inputs=x))
            else:
                outState.append(states[i])
            x, state1, state2 = self.lstm[i](x, initial_state=outState[i], training=training)
            outState[i] = [state1, state2]
        x = self.dense(x, training=training)

        if returnState:
            return x, outState
        else:
            return x
