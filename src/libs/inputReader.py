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
            words_line = []
            tags_line = []
            for word_tag in line.split():
                word, tag = word_tag.split('_')
                words_line.append(word.lower())
                tags_line.append(tag)
            words_tags.append([words_line, tags_line])

    df = pd.DataFrame({'sentence': list(map(lambda word_tag: word_tag[0], words_tags)), 'tags': list(map(lambda word_tag: word_tag[1], words_tags))})

    print(df.head())

    return df
