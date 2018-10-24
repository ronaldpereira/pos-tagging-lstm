import numpy as np
from sklearn.preprocessing import LabelEncoder

def label_encoder(data):
    allLabels = []
    allLabels.extend(data.train['tag'].values)
    allLabels.extend(data.validation['tag'].values)
    allLabels.extend(data.test['tag'].values)

    le = LabelEncoder()
    le.fit(allLabels)

    trainEncodedLabels = le.transform(data.train['tag'].values)
    validationEncodedLabels = le.transform(data.validation['tag'].values)
    testEncodedLabels = le.transform(data.test['tag'].values)

    data.train['enc_tag'] = trainEncodedLabels
    data.validation['enc_tag'] = validationEncodedLabels
    data.test['enc_tag'] = testEncodedLabels
