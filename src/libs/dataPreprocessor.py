from multiprocessing import Manager, Process

import numpy as np
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder


def label_encoder(indexDict, data, info, return_dict):
    dataEncoded = []
    for index in range(len(data)):
        if index % 100 == 0 or index == len(data)-1:
            print('%s: %d/%d -> %.1f%%' %(info, index, len(data)-1, index*100/(len(data)-1)))

        dataEncoded.append([indexDict[w] for w in data[index]])

    return_dict[info] = dataEncoded

def one_hot_encode_tags(data, categoryLength):
    category_sequences = []
    for seq in data:
        categories = []
        for item in seq:
            categories.append(np.zeros(categoryLength))
            categories[-1][item] = 1.0

        category_sequences.append(categories)
    
    return category_sequences


def getSentenceMaxLength(data):
    sentences = []
    sentences.extend(data.train['sentence'].values)
    sentences.extend(data.validation['sentence'].values)
    sentences.extend(data.test['sentence'].values)
    
    maxSentenceLength = len(max(sentences, key=len))

    data.maxSentenceLength = maxSentenceLength

    return maxSentenceLength

def sentence_encoder(data):
    print('Sentence encoder')

    sentences = []
    sentences.extend(data.train['sentence'].values)
    sentences.extend(data.validation['sentence'].values)
    sentences.extend(data.test['sentence'].values)

    vocabulary = set()
    for sentence in sentences:
        for word in sentence:
            vocabulary.add(word)
    
    word2index = {word: i+1 for i, word in enumerate(list(vocabulary))}
    word2index['<pad>'] = 0

    data.word2index = word2index

    manager = Manager()
    return_dict = manager.dict()

    trainProcess = Process(target=label_encoder, args=(word2index, data.train['sentence'].values, 'train', return_dict))
    trainProcess.start()

    validationProcess = Process(target=label_encoder, args=(word2index, data.validation['sentence'].values, 'validation', return_dict))
    validationProcess.start()

    testProcess = Process(target=label_encoder, args=(word2index, data.test['sentence'].values, 'test', return_dict))
    testProcess.start()

    trainProcess.join()
    validationProcess.join()
    testProcess.join()

    maxSentenceLength = getSentenceMaxLength(data)

    data.train['enc_sentence'] = pad_sequences(return_dict['train'], maxlen=maxSentenceLength, padding='post').tolist()
    data.validation['enc_sentence'] = pad_sequences(return_dict['validation'], maxlen=maxSentenceLength, padding='post').tolist()
    data.test['enc_sentence'] = pad_sequences(return_dict['test'], maxlen=maxSentenceLength, padding='post').tolist()

def tag_encoder(data):
    print('Tags encoder')

    tagLine = []
    tagLine.extend(data.train['tags'].values)
    tagLine.extend(data.validation['tags'].values)
    tagLine.extend(data.test['tags'].values)

    allTags = set()
    for line in tagLine:
        for tag in line:
            allTags.add(tag)
    
    tag2index = {tag: i+1 for i, tag in enumerate(list(allTags))}
    tag2index['<pad>'] = 0

    data.tag2index = tag2index

    manager = Manager()
    return_dict = manager.dict()

    trainProcess = Process(target=label_encoder, args=(tag2index, data.train['tags'].values, 'train', return_dict))
    trainProcess.start()

    validationProcess = Process(target=label_encoder, args=(tag2index, data.validation['tags'].values, 'validation', return_dict))
    validationProcess.start()

    testProcess = Process(target=label_encoder, args=(tag2index, data.test['tags'].values, 'test', return_dict))
    testProcess.start()

    trainProcess.join()
    validationProcess.join()
    testProcess.join()

    maxSentenceLength = getSentenceMaxLength(data)

    data.train['enc_tags'] = pad_sequences(return_dict['train'], maxlen=maxSentenceLength, padding='post').tolist()
    data.validation['enc_tags'] = pad_sequences(return_dict['validation'], maxlen=maxSentenceLength, padding='post').tolist()
    data.test['enc_tags'] = pad_sequences(return_dict['test'], maxlen=maxSentenceLength, padding='post').tolist()


    data.train['one_hot_enc_tags'] = one_hot_encode_tags(data.train['enc_tags'], len(tag2index))
    data.validation['one_hot_enc_tags'] = one_hot_encode_tags(data.validation['enc_tags'], len(tag2index))
    data.test['one_hot_enc_tags'] = one_hot_encode_tags(data.test['enc_tags'], len(tag2index))
