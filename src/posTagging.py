import libs.argParseConfig as argParseConfig
import libs.inputReader as inputReader

args = argParseConfig.parser()

data = inputReader.InputReader(args.train, args.validation, args.test)
