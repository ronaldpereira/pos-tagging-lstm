import numpy as np
from sklearn.preprocessing import LabelEncoder

def sentence_encoder(data):
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

    data.train['enc_sentence'] = list(map(lambda sentence: le.transform(sentence), data.train['sentence'].values))
    data.validation['enc_sentence'] = list(map(lambda sentence: le.transform(sentence), data.validation['sentence'].values))
    data.test['enc_sentence'] = list(map(lambda sentence: le.transform(sentence), data.test['sentence'].values))

def tag_encoder(data):
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

    data.train['enc_tags'] = list(map(lambda tag: le.transform(tag), data.train['tags'].values))
    data.validation['enc_tags'] = list(map(lambda tag: le.transform(tag), data.validation['tags'].values))
    data.test['enc_tags'] = list(map(lambda tag: le.transform(tag), data.test['tags'].values))
