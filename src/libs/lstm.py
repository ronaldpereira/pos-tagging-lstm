from keras.models import Sequential
from keras.layers import InputLayer, LSTM, Bidirectional, TimeDistributed, Dense, Embedding, Activation
from keras.optimizers import Adam

class Model:
    def __init__(self):
        self.model = Sequential()
        self.model.add(InputLayer())
