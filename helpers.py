import pandas as pd
import numpy as np

def get_data():
    df = pd.read_csv("http://mlr.cs.umass.edu/ml/machine-learning-databases/iris/iris.data", header=None)
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)
    X = df.iloc[0:100, [0, 2]].values
    return (y,X)

def update_check(list1, list2):
    for i,j in zip(list1, list2):
        if i != j:
            return True
    return False