import numpy as np
from sklearn.preprocessing import LabelEncoder
from multiprocessing import Process

def label_encoder(le, data, df, col_name, info):
    dataEncoded = []
    for index, value in enumerate(data):
        if index % 100 == 0 or index == len(data)-1:
            print('%s: %d/%d -> %.1f%%' %(info, index, len(data)-1, index*100/(len(data)-1)))
        dataEncoded.append(le.transform(value))

    df[col_name] = dataEncoded

def sentence_encoder(data):
    print('Sentence encoder')

    sentences = []
    sentences.extend(data.train['sentence'].values)
    sentences.extend(data.validation['sentence'].values)
    sentences.extend(data.test['sentence'].values)

    vocabulary = []
    for sentence in sentences:
        for word in sentence:
            vocabulary.append(word)

    le = LabelEncoder()
    le.fit(vocabulary)

    trainProcess = Process(target=label_encoder, args=(le, data.train['sentence'].values, data.train, 'enc_sentence', 'train'))
    trainProcess.start()

    validationProcess = Process(target=label_encoder, args=(le, data.validation['sentence'].values, data.validation, 'enc_sentence', 'validation'))
    validationProcess.start()

    testProcess = Process(target=label_encoder, args=(le, data.test['sentence'].values, data.test, 'enc_sentence', 'test'))
    testProcess.start()

    trainProcess.join()
    validationProcess.join()
    testProcess.join()


def tag_encoder(data):
    print('Tags encoder')

    tagLine = []
    tagLine.extend(data.train['tags'].values)
    tagLine.extend(data.validation['tags'].values)
    tagLine.extend(data.test['tags'].values)

    allTags = []
    for line in tagLine:
        for tag in line:
            allTags.append(tag)

    le = LabelEncoder()
    le.fit(allTags)

    trainProcess = Process(target=label_encoder, args=(le, data.train['tags'].values, data.train, 'enc_tags', 'train'))
    trainProcess.start()

    validationProcess = Process(target=label_encoder, args=(le, data.validation['tags'].values, data.validation, 'enc_tags', 'validation'))
    validationProcess.start()

    testProcess = Process(target=label_encoder, args=(le, data.test['tags'].values, data.test, 'enc_tags', 'test'))
    testProcess.start()

    trainProcess.join()
    validationProcess.join()
    testProcess.join()
