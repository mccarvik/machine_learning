import matplotlib as mpl
mpl.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from helpers import PIC_LOC
import pdb

from matplotlib.colors import ListedColormap
from perceptron import Perceptron

def ch2():
    # df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
    df = pd.read_csv("http://mlr.cs.umass.edu/ml/machine-learning-databases/iris/iris.data", header=None)
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)
    X = df.iloc[0:100, [0, 2]].values
    plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
    
    ppn = Perceptron(eta=0.1, n_iter=10)
    pdb.set_trace()
    ppn.fit(X,y)
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')
    plot_decision_regions(X, y, classifier=ppn)
    plt.savefig(PIC_LOC + "iris_ch2.png")
    plt.close()
    # plt.show()
    
    plt.plot(range(1,len(ppn.errors_) + 1), ppn.errors_,marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of misclassifications')
    plt.savefig(PIC_LOC + "iris2_ch2.png")
    plt.close()
    

def plot_decision_regions(X, y, classifier, resolution=0.02):
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

    # plot class samples,
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)



if __name__ == "__main__":
    # import pdb; pdb.set_trace()
    ch2()