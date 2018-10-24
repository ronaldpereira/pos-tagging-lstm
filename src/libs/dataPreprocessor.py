import numpy as np
from sklearn.preprocessing import LabelEncoder

def label_encoder(data, trainLabels, validationLabels, testLabels):
    labels = []
    labels.extend(trainLabels.values)
    labels.extend(validationLabels.values)
    labels.extend(testLabels.values)

    le = LabelEncoder()
    le.fit(labels)

    trainEncodedLabels = le.transform(trainLabels.values)
    validationEncodedLabels = le.transform(validationLabels.values)
    testEncodedLabels = le.transform(testLabels.values)

    data.train['enc_tag'] = trainEncodedLabels
    data.validation['enc_tag'] = validationEncodedLabels
    data.test['enc_tag'] = testEncodedLabels
