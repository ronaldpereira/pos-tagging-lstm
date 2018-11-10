import numpy as np
from keras import backend as K
from keras.layers import (LSTM, Activation, Bidirectional, Dense, Embedding,
                          InputLayer, TimeDistributed)
from keras.models import Sequential


class Model:
    def __init__(self, data):
        self.model = Sequential()
        self.model.add(InputLayer(input_shape=(data.maxSentenceLength, )))
        self.model.add(Embedding(len(data.word2index), 128))
        self.model.add(Bidirectional(LSTM(256, return_sequences=True)))
        self.model.add(TimeDistributed(Dense(len(data.tag2index))))
        self.model.add(Activation('softmax'))

        self.model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])

        self.model.summary()

    def train(self, x_train, y_train):        
        self.model.fit(x_train, y_train, batch_size=128, epochs=40, validation_split=0.2)
