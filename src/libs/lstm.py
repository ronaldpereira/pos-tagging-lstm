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

        self.model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy', Model.ignore_padding_accuracy(0)])

        self.model.summary()

    @staticmethod
    def ignore_padding_accuracy(ignored_tag_id=0):
        def ignore_accuracy(y_true, y_pred):
            y_true_class = K.argmax(y_true, axis=-1)
            y_pred_class = K.argmax(y_pred, axis=-1)
    
            ignore_mask = K.cast(K.not_equal(y_pred_class, ignored_tag_id), 'int32')
            matches = K.cast(K.equal(y_true_class, y_pred_class), 'int32') * ignore_mask
            accuracy = K.sum(matches) / K.maximum(K.sum(ignore_mask), 1)
            return accuracy
        return ignore_accuracy

    def train(self, x_train, y_train):        
        self.model.fit(x_train, y_train, batch_size=128, epochs=40, validation_split=0.2)
