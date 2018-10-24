import pandas as pd

import libs.argParseConfig as argParseConfig
import libs.dataPreprocessor as dataPreprocessor
import libs.inputReader as inputReader

args = argParseConfig.parser()

data = inputReader.InputReader(args.train, args.validation, args.test)
print(data.validation.head())

dataPreprocessor.sentence_encoder(data)
print(data.validation.head())

dataPreprocessor.tag_encoder(data)
print(data.validation.head())
