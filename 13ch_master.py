import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from helpers import PL13, plot_decision_regions
import theano, os, struct
from theano import tensor as T
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD


def intro():
     # initialize
    x1 = T.scalar()
    w1 = T.scalar()
    w0 = T.scalar()
    z1 = w1 * x1 + w0
    # compile
    net_input = theano.function(inputs=[w1, x1, w0], outputs=z1)
    # execute
    print('Net input: %.2f' % net_input(2.0, 1.0, 0.5))
    print(theano.config.floatX)

def arrays():
    # initialize
    # if you are running Theano on 64 bit mode, 
    # you need to use dmatrix instead of fmatrix
    theano.config.floatX = 'float32'
    x = T.fmatrix(name='x')
    x_sum = T.sum(x, axis=0)
    
    # compile
    calc_sum = theano.function(inputs=[x], outputs=x_sum)
    
    # execute (Python list)
    ary = [[1, 2, 3], [1, 2, 3]]
    print('Column sum:', calc_sum(ary))
    
    # execute (NumPy array)
    ary = np.array([[1, 2, 3], [1, 2, 3]], dtype=theano.config.floatX)
    print('Column sum:', calc_sum(ary))
    
def shared():
    theano.config.floatX = 'float32'
    
    # initialize
    x = T.fmatrix(name='x')
    w = theano.shared(np.asarray([[0.0, 0.0, 0.0]], 
                                 dtype=theano.config.floatX))
    z = x.dot(w.T)
    update = [[w, w + 1.0]]
    # compile
    net_input = theano.function(inputs=[x], 
                                updates=update, 
                                outputs=z)
    # execute
    data = np.array([[1, 2, 3]], dtype=theano.config.floatX)
    for i in range(5):
        print('z%d:' % i, net_input(data))
        
def givens():
    theano.config.floatX = 'float32'
    # initialize
    data = np.array([[1, 2, 3]], 
                    dtype=theano.config.floatX)
    x = T.fmatrix(name='x')
    w = theano.shared(np.asarray([[0.0, 0.0, 0.0]], 
                                 dtype=theano.config.floatX))
    z = x.dot(w.T)
    update = [[w, w + 1.0]]
    # compile
    net_input = theano.function(inputs=[], 
                                updates=update, 
                                givens={x: data},
                                outputs=z)
    # execute
    for i in range(5):
        print('z:', net_input())

def lin_reg():
    theano.config.floatX = 'float32'
    X_train = np.asarray([[0.0], [1.0], [2.0], [3.0], [4.0],
                          [5.0], [6.0], [7.0], [8.0], [9.0]], 
                         dtype=theano.config.floatX)
    y_train = np.asarray([1.0, 1.3, 3.1, 2.0, 5.0, 
                          6.3, 6.6, 7.4, 8.0, 9.0], 
                         dtype=theano.config.floatX)
                         
    costs, w = train_linreg(X_train, y_train, eta=0.001, epochs=10)
       
    plt.plot(range(1, len(costs)+1), costs)
    plt.tight_layout()
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.tight_layout()
    plt.savefig(PL13 + 'cost_convergence.png', dpi=300)
    plt.close()
    
    plt.scatter(X_train, y_train, marker='s', s=50)
    plt.plot(range(X_train.shape[0]), 
             predict_linreg(X_train, w), 
             color='gray', 
             marker='o', 
             markersize=4, 
             linewidth=3)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.savefig(PL13 + 'linreg.png', dpi=300)

def train_linreg(X_train, y_train, eta, epochs):
    costs = []
    # Initialize arrays
    eta0 = T.fscalar('eta0')
    y = T.fvector(name='y') 
    X = T.fmatrix(name='X')   
    w = theano.shared(np.zeros(
                      shape=(X_train.shape[1] + 1),
                      dtype=theano.config.floatX),
                      name='w')
    
    # calculate cost
    net_input = T.dot(X, w[1:]) + w[0]
    errors = y - net_input
    cost = T.sum(T.pow(errors, 2)) 

    # perform gradient update
    gradient = T.grad(cost, wrt=w)
    update = [(w, w - eta0 * gradient)]

    # compile model
    train = theano.function(inputs=[eta0],
                            outputs=cost,
                            updates=update,
                            givens={X: X_train,
                                    y: y_train,})      
    for _ in range(epochs):
        costs.append(train(eta))
    return costs, w

def predict_linreg(X, w):
    Xt = T.matrix(name='X')
    net_input = T.dot(Xt, w[1:]) + w[0]
    predict = theano.function(inputs=[Xt], givens={w: w}, outputs=net_input)
    return predict(X)

def logistic_func():
    X = np.array([[1, 1.4, 1.5]])
    w = np.array([0.0, 0.2, 0.4])
    print('P(y=1|x) = %.3f' % logistic_activation(X, w)[0])
    
    # W : array, shape = [n_output_units, n_hidden_units+1]
    #          Weight matrix for hidden layer -> output layer.
    # note that first column (A[:][0] = 1) are the bias units
    W = np.array([[1.1, 1.2, 1.3, 0.5],
                  [0.1, 0.2, 0.4, 0.1],
                  [0.2, 0.5, 2.1, 1.9]])
    
    # A : array, shape = [n_hidden+1, n_samples]
    #          Activation of hidden layer.
    # note that first element (A[0][0] = 1) is for the bias units
    A = np.array([[1.0], 
                  [0.1], 
                  [0.3], 
                  [0.7]])
    
    # Z : array, shape = [n_output_units, n_samples]
    #          Net input of output layer.
    Z = W.dot(A) 
    y_probas = logistic(Z)
    print('Probabilities: \n', y_probas)
    y_class = np.argmax(Z, axis=0)
    print('predicted class label: %d' % y_class[0])
    y_probas = softmax(Z)
    print('Probabilities: \n', y_probas)
    y_class = np.argmax(Z, axis=0)
    print('predicted class label: %d' % y_class[0])
    
def softmax(z): 
    return np.exp(z) / np.sum(np.exp(z))
    
def softmax_activation(X, w):
    z = net_input(X, w)
    return softmax(z)
    
def net_input(X, w):
    z = X.dot(w)
    return z

def logistic(z):
    return 1.0 / (1.0 + np.exp(-z))

def logistic_activation(X, w):
    z = net_input(X, w)
    return logistic(z)
    
def tanh(z):
    e_p = np.exp(z)
    e_m = np.exp(-z)
    return (e_p - e_m) / (e_p + e_m)

def hyperbolic_tangent():
    z = np.arange(-5, 5, 0.005)
    log_act = logistic(z)
    tanh_act = tanh(z)
    
    # alternatives:
    # from scipy.special import expit
    # log_act = expit(z)
    # tanh_act = np.tanh(z)
    
    plt.ylim([-1.5, 1.5])
    plt.xlabel('net input $z$')
    plt.ylabel('activation $\\phi(z)$')
    plt.axhline(1, color='black', linestyle='--')
    plt.axhline(0.5, color='black', linestyle='--')
    plt.axhline(0, color='black', linestyle='--')
    plt.axhline(-1, color='black', linestyle='--')
    
    plt.plot(z, tanh_act, 
             linewidth=2, 
             color='black', 
             label='tanh')
    plt.plot(z, log_act, 
             linewidth=2, 
             color='lightgreen', 
             label='logistic')
    
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(PL13 + 'activation.png', dpi=300)

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

def keras():
    X_train, y_train = load_mnist('mnist', kind='train')
    print('Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))
    X_test, y_test = load_mnist('mnist', kind='t10k')
    print('Rows: %d, columns: %d' % (X_test.shape[0], X_test.shape[1]))
    
    theano.config.floatX = 'float32'
    X_train = X_train.astype(theano.config.floatX)
    X_test = X_test.astype(theano.config.floatX)
    print('First 3 labels: ', y_train[:3])
    y_train_ohe = np_utils.to_categorical(y_train)
    print('\nFirst 3 labels (one-hot):\n', y_train_ohe[:3])
    
    np.random.seed(1) 
    model = Sequential()
    model.add(Dense(input_dim=X_train.shape[1], 
                    output_dim=50, 
                    init='uniform', 
                    activation='tanh'))
    
    model.add(Dense(input_dim=50, 
                    output_dim=50, 
                    init='uniform', 
                    activation='tanh'))
    
    model.add(Dense(input_dim=50, 
                    output_dim=y_train_ohe.shape[1], 
                    init='uniform', 
                    activation='softmax'))
    sgd = SGD(lr=0.001, decay=1e-7, momentum=.9)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    model.fit(X_train, y_train_ohe, 
              nb_epoch=50, 
              batch_size=300, 
              verbose=1, 
              validation_split=0.1, 
              show_accuracy=True)
    y_train_pred = model.predict_classes(X_train, verbose=0)
    print('First 3 predictions: ', y_train_pred[:3])
    train_acc = np.sum(y_train == y_train_pred, axis=0) / X_train.shape[0]
    print('Training accuracy: %.2f%%' % (train_acc * 100))
    y_test_pred = model.predict_classes(X_test, verbose=0)
    test_acc = np.sum(y_test == y_test_pred, axis=0) / X_test.shape[0]
    print('Test accuracy: %.2f%%' % (test_acc * 100))
    

if __name__ == "__main__":
    # import pdb; pdb.set_trace()
    # intro()
    # arrays()
    # shared()
    # givens()
    # lin_reg()
    # logistic_func()
    # hyperbolic_tangent()
    keras()