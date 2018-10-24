from keras.layers import (LSTM, Activation, Bidirectional, Dense, Embedding,
                          InputLayer, TimeDistributed)
from keras.models import Sequential
from keras.optimizers import Adam


class Model:
    def __init__(self):
        self.model = Sequential()
        self.model.add(InputLayer())
