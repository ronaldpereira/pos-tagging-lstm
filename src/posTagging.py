import libs.argParseConfig as argParseConfig
import libs.inputReader as inputReader
import libs.dataPreprocessor as dataPreprocessor

args = argParseConfig.parser()

data = inputReader.InputReader(args.train, args.validation, args.test)

print(data.train.head())

dataPreprocessor.label_encoder(data)

print(data.train.head())
