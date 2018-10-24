import pandas as pd

import libs.argParseConfig as argParseConfig
import libs.inputReader as inputReader
import libs.dataPreprocessor as dataPreprocessor

args = argParseConfig.parser()

if bool(args.data_preproc):
    data = inputReader.InputReader(args.train, args.validation, args.test)

    print(data.train.head())

    dataPreprocessor.sentence_encoder(data)

    print(data.train.head())

    dataPreprocessor.tag_encoder(data)

    print(data.train.head())

    data.train.to_csv('tmp/train.csv')
    data.validation.to_csv('tmp/validation.csv')
    data.test.to_csv('tmp/test.csv')

else:
    data = inputReader.InputReader(args.train, args.validation, args.test)

    data.train = pd.read_csv('tmp/train.csv')
    data.validation = pd.read_csv('tmp/validation.csv')
    data.test = pd.read_csv('tmp/test.csv')
