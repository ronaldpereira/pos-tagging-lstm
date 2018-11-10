import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

import libs.argParseConfig as argParseConfig
import libs.dataPreprocessor as dataPreprocessor
import libs.inputReader as inputReader
import libs.lstm as lstm

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
predictions_tokens = dataPreprocessor.encoded_to_tokens(predictions, {i: t for t, i in data.tag2index.items()})

with open('output/predicted.txt', 'w') as predictedFile, open('output/real.txt', 'w') as realFile:
    predictedFile.write(str(predictions_tokens))
    realFile.write(str(y_test))
