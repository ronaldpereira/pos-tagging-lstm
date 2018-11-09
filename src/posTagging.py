import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

import libs.argParseConfig as argParseConfig
import libs.dataPreprocessor as dataPreprocessor
import libs.inputReader as inputReader
import libs.lstm as lstm

def logits_to_tokens(sequences, index):
    token_sequences = []
    for categorical_sequence in sequences:
        token_sequence = []
        for categorical in categorical_sequence:
            token_sequence.append(index[np.argmax(categorical)])
 
        token_sequences.append(token_sequence)
 
    return token_sequences

args = argParseConfig.parser()

data = inputReader.InputReader(args.train, args.validation, args.test)

dataPreprocessor.sentence_encoder(data)

dataPreprocessor.tag_encoder(data)

print('*** LSTM POS-Tagging Model ***')

model = lstm.Model(data)

x_train, x_test, y_train, y_test = train_test_split(data.train['enc_sentence'], data.train['one_hot_enc_tags'], test_size=0.2)
x_train = np.array(x_train.values.tolist())
y_train = np.array(y_train.values.tolist())
x_test = np.array(x_test.values.tolist())
y_test = np.array(y_test.values.tolist())

model.train(x_train, y_train)

predictions = model.model.predict(x_test)
print(logits_to_tokens(predictions, {i: t for t, i in data.tag2index.items()}))
