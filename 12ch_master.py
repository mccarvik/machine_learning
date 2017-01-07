import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from helpers import PL12, plot_decision_regions
import os, sys
import struct
from neural_net import NeuralNetMLP, MLPGradientCheck


MNIST_PATH = '/home/ubuntu/workspace/machine_learning/mnist/'

def load_mnist(path, kind='train'):
        #  Load MNIST data from `path`
        labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
        images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)
        with open(labels_path, 'rb') as lbpath:
            magic, n = struct.unpack('>II', lbpath.read(8))
            labels = np.fromfile(lbpath, dtype=np.uint8)
        with open(images_path, 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
            images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
        return images, labels


def orig_train():
    X_train, y_train = load_mnist(MNIST_PATH, kind='train')
    print('Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))
    X_test, y_test = load_mnist('mnist', kind='t10k')
    print('Rows: %d, columns: %d' % (X_test.shape[0], X_test.shape[1]))
    
    fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True,)
    ax = ax.flatten()
    for i in range(10):
        img = X_train[y_train == i][0].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.savefig(PL12 + 'mnist_all.png', dpi=300)
    plt.close()
    
    fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True,)
    ax = ax.flatten()
    for i in range(25):
        img = X_train[y_train == 7][i].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.savefig(PL12 + 'mnist_7.png', dpi=300)

def neural():
    X_train, y_train = load_mnist(MNIST_PATH, kind='train')
    print('Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))
    X_test, y_test = load_mnist('mnist', kind='t10k')
    print('Rows: %d, columns: %d' % (X_test.shape[0], X_test.shape[1]))
    
    nn = NeuralNetMLP(n_output=10, 
                      n_features=X_train.shape[1], 
                      n_hidden=50, 
                      l2=0.1, 
                      l1=0.0, 
                      epochs=1000, 
                      eta=0.001,
                      alpha=0.001,
                      decrease_const=0.00001,
                      minibatches=50, 
                      shuffle=True,
                      random_state=1)
    
    nn.fit(X_train, y_train, print_progress=True)
    
    plt.plot(range(len(nn.cost_)), nn.cost_)
    plt.ylim([0, 2000])
    plt.ylabel('Cost')
    plt.xlabel('Epochs * 50')
    plt.tight_layout()
    plt.savefig(PL12 + 'cost.png', dpi=300)
    plt.close()
    
    batches = np.array_split(range(len(nn.cost_)), 1000)
    cost_ary = np.array(nn.cost_)
    cost_avgs = [np.mean(cost_ary[i]) for i in batches]
    
    plt.plot(range(len(cost_avgs)), cost_avgs, color='red')
    plt.ylim([0, 2000])
    plt.ylabel('Cost')
    plt.xlabel('Epochs')
    plt.tight_layout()
    plt.savefig(PL12 + 'cost2.png', dpi=300)
    plt.close()
    
    y_train_pred = nn.predict(X_train)
    if sys.version_info < (3, 0):
        acc = (np.sum(y_train == y_train_pred, axis=0)).astype('float') / X_train.shape[0]
    else:
        acc = np.sum(y_train == y_train_pred, axis=0) / X_train.shape[0]
    print('Training accuracy: %.2f%%' % (acc * 100))
    
    y_test_pred = nn.predict(X_test)    
    if sys.version_info < (3, 0):    
        acc = (np.sum(y_test == y_test_pred, axis=0)).astype('float') / X_test.shape[0]    
    else:    
        acc = np.sum(y_test == y_test_pred, axis=0) / X_test.shape[0]    
    print('Test accuracy: %.2f%%' % (acc * 100))
    
    miscl_img = X_test[y_test != y_test_pred][:25]
    correct_lab = y_test[y_test != y_test_pred][:25]
    miscl_lab= y_test_pred[y_test != y_test_pred][:25]
    fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True,)
    ax = ax.flatten()
    for i in range(25):
        img = miscl_img[i].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
        ax[i].set_title('%d) t: %d p: %d' % (i+1, correct_lab[i], miscl_lab[i]))
    
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.savefig(PL12 + 'mnist_miscl.png', dpi=300)
    plt.close()
    
    
def gradient_check():
    X_train, y_train = load_mnist(MNIST_PATH, kind='train')
    print('Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))
    X_test, y_test = load_mnist('mnist', kind='t10k')
    print('Rows: %d, columns: %d' % (X_test.shape[0], X_test.shape[1]))
    
    nn_check = MLPGradientCheck(n_output=10, 
                                n_features=X_train.shape[1], 
                                n_hidden=10, 
                                l2=0.0, 
                                l1=0.0, 
                                epochs=10, 
                                eta=0.001,
                                alpha=0.0,
                                decrease_const=0.0,
                                minibatches=1, 
                                shuffle=False,
                                random_state=1)
    print(nn_check.fit(X_train[:5], y_train[:5], print_progress=False))

if __name__ == "__main__":
    # import pdb; pdb.set_trace()
    # load_mnist(MNIST_PATH)
    # orig_train()
    # neural()
    gradient_check()