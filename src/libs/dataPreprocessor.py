from multiprocessing import Manager, Process

import numpy as np
from sklearn.preprocessing import LabelEncoder


def label_encoder(le, data, info, return_dict):
    dataEncoded = []
    for index, value in enumerate(data):
        if index % 100 == 0 or index == len(data)-1:
            print('%s: %d/%d -> %.1f%%' %(info, index, len(data)-1, index*100/(len(data)-1)))
        dataEncoded.append(le.transform(value))

    return_dict[info] = dataEncoded

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

    manager = Manager()
    return_dict = manager.dict()

    trainProcess = Process(target=label_encoder, args=(le, data.train['sentence'].values, 'train', return_dict))
    trainProcess.start()

    validationProcess = Process(target=label_encoder, args=(le, data.validation['sentence'].values, 'validation', return_dict))
    validationProcess.start()

    testProcess = Process(target=label_encoder, args=(le, data.test['sentence'].values, 'test', return_dict))
    testProcess.start()

    trainProcess.join()
    validationProcess.join()
    testProcess.join()

    data.train['enc_sentence'] = return_dict['train']
    data.validation['enc_sentence'] = return_dict['validation']
    data.test['enc_sentence'] = return_dict['test']

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

    manager = Manager()
    return_dict = manager.dict()

    trainProcess = Process(target=label_encoder, args=(le, data.train['tags'].values, 'train', return_dict))
    trainProcess.start()

    validationProcess = Process(target=label_encoder, args=(le, data.validation['tags'].values, 'validation', return_dict))
    validationProcess.start()

    testProcess = Process(target=label_encoder, args=(le, data.test['tags'].values, 'test', return_dict))
    testProcess.start()

    trainProcess.join()
    validationProcess.join()
    testProcess.join()

    data.train['enc_tags'] = return_dict['train']
    data.validation['enc_tags'] = return_dict['validation']
    data.test['enc_tags'] = return_dict['test']
