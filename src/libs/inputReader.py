import pandas as pd

class InputReader:
    def __init__(self, trainPath, validationPath, testPath):
        self.train = inputPreprocessor(trainPath)
        self.validation = inputPreprocessor(validationPath)
        self.test = inputPreprocessor(testPath)


def inputPreprocessor(filePath):
    with open(filePath, "r") as inputFile:
        words_tags = []
        for line in inputFile:
            for word_tag in line.split():
                word, tag = word_tag.split('_')
                words_tags.append((word, tag))

    df = pd.DataFrame({'word': list(map(lambda word_tag: word_tag[0], words_tags)), 'tag': list(map(lambda word_tag: word_tag[1], words_tags))})

    return df
