from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras import Model


class RNNModel(Model):
    def __init__(self, vocabSize, embeddingSize, rnnUnits):
        super().__init__(self)
        self.embedding = Embedding(vocabSize, embeddingSize)
        self.gru = LSTM(rnnUnits, return_sequences=True, return_state=True)
        self.dense = Dense(vocabSize)

    def call(self, inputs, states=None, returnState=False, training=False):
        x = self.embedding(inputs, training=training)
        if states is None:
            states = self.gru.get_initial_state(inputs=x)
        x, state1, state2 = self.gru(x, initial_state=states, training=training)
        states = [state1, state2]
        x = self.dense(x, training=training)

        if returnState:
            return x, states
        else:
            return x
