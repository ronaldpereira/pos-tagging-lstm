import pandas as pd
from sklearn.model_selection import train_test_split

import libs.argParseConfig as argParseConfig
import libs.dataPreprocessor as dataPreprocessor
import libs.inputReader as inputReader
import libs.lstm as lstm

args = argParseConfig.parser()

data = inputReader.InputReader(args.train, args.validation, args.test)
print(data.train.head())

dataPreprocessor.sentence_encoder(data)
print(data.train.head())

dataPreprocessor.tag_encoder(data)
print(data.train.head())

print('*** Model ***')

model = lstm.Model(data)

x_train, x_test, y_train, y_test = train_test_split(data.train['enc_sentence'], data.train['one_hot_enc_tags'], test_size=0.2)

model.train(x_train, y_train)
