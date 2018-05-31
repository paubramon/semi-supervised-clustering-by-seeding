from scipy.io import arff
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


def prepareIrisData():
    filename = "datasets/iris.arff"
    fileObject = open(filename)
    data, meta = arff.loadarff(fileObject)

    headers = meta.names()
    types = meta.types()

    df = pd.DataFrame(data, columns=headers)
    classes = df['class']
    del df['class']
    headers.remove('class')

    # Standardize data
    ss = StandardScaler(with_std=False).fit(df)
    df = ss.transform(df)

    # Encoder classes
    le = LabelEncoder()
    le.fit(classes)
    classes_int = le.transform(classes)

    # Shuffle data
    p = np.random.permutation(len(classes_int))
    df = df[p, :]
    classes_int = classes_int[p]

    return df, classes_int
