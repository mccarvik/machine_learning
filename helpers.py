import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from matplotlib.colors import ListedColormap

PIC_LOC = '/home/ubuntu/workspace/machine_learning/png/'
PL4 = '/home/ubuntu/workspace/machine_learning/ch4_png/'
PL5 = '/home/ubuntu/workspace/machine_learning/ch5_png/'
PL6 = '/home/ubuntu/workspace/machine_learning/ch6_png/'

def get_data():
    df = pd.read_csv("http://mlr.cs.umass.edu/ml/machine-learning-databases/iris/iris.data", header=None)
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)
    X = df.iloc[0:100, [0, 2]].values
    return (y,X)
    
def get_std_data():
    df = pd.read_csv("http://mlr.cs.umass.edu/ml/machine-learning-databases/iris/iris.data", header=None)
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)
    X = df.iloc[0:100, [0, 2]].values
    
    # standardize features
    X_std = np.copy(X)
    X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
    X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()
    return (y,X_std)

def get_data_sklearn():
    iris = datasets.load_iris()
    X = iris.data[:, [2,3]]
    y = iris.target
    return (X, y)
    
   
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

   # setup marker generator and color map
   markers = ('s', 'x', 'o', '^', 'v')
   colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
   cmap = ListedColormap(colors[:len(np.unique(y))])

   # plot the decision surface
   x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
   x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
   xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                          np.arange(x2_min, x2_max, resolution))
   Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
   Z = Z.reshape(xx1.shape)
   plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
   plt.xlim(xx1.min(), xx1.max())
   plt.ylim(xx2.min(), xx2.max())

   for idx, cl in enumerate(np.unique(y)):
       plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                   alpha=0.8, c=cmap(idx),
                   marker=markers[idx], label=cl)

   # highlight test samples
   if test_idx:
       X_test, y_test = X[test_idx, :], y[test_idx]
       plt.scatter(X_test[:, 0],
                   X_test[:, 1],
                   c='',
                   alpha=1.0,
                   linewidths=1,
                   marker='o',
                   s=55, label='test set')
                   

def update_check(list1, list2):
    for i,j in zip(list1, list2):
        if i != j:
            return True
    return False