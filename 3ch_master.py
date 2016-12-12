import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
from helpers import get_data_sklearn, PIC_LOC, plot_decision_regions
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
    
    
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
    
def main1():
    z = np.arange(-7, 7, 0.1)
    phi_z = sigmoid(z)
    
    plt.plot(z, phi_z)
    plt.axvline(0.0, color='k')
    plt.ylim(-0.1, 1.1)
    plt.xlabel('z')
    plt.ylabel('$\\phi (z)$')
    
    # y axis ticks and gridline
    plt.yticks([0.0, 0.5, 1.0])
    ax = plt.gca()
    ax.yaxis.grid(True)
    
    plt.tight_layout()
    plt.savefig(PIC_LOC + 'sigmoid.png', dpi=300)
    # plt.show()

def main2():
    X, y = get_data_sklearn()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    
    lr = LogisticRegression(C=1000.0, random_state=0)
    lr.fit(X_train_std, y_train)
    
    plot_decision_regions(X_combined_std, y_combined,
                          classifier=lr, test_idx=range(105, 150))
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(PIC_LOC + 'logistic_regression.png', dpi=300)
    # plt.show()
    
def main3():
    # logistic regretion with regularization
    X, y = get_data_sklearn()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    
    weights, params = [], []
    for c in np.arange(-5, 5):
      lr = LogisticRegression(C=10**c, random_state=0)
      lr.fit(X_train_std, y_train)
      weights.append(lr.coef_[1])
      params.append(10**c)
    
    weights = np.array(weights)
    plt.plot(params, weights[:, 0],
           label='petal length')
    plt.plot(params, weights[:, 1], linestyle='--',
           label='petal width')
    plt.ylabel('weight coefficient')
    plt.xlabel('C')
    plt.legend(loc='upper left')
    plt.xscale('log')
    plt.savefig(PIC_LOC + 'regression_path.png', dpi=300)
    # plt.show()

def support_vector_machines():
    X, y = get_data_sklearn()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    svm = SVC(kernel='linear', C=1.0, random_state=0)
    svm.fit(X_train_std, y_train)
    
    plot_decision_regions(X_combined_std, y_combined,
                        classifier=svm, test_idx=range(105, 150))
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(PIC_LOC + 'support_vector_machine_linear.png', dpi=300)
    # plt.show()

def nonlinear_svm():
    np.random.seed(0)
    X_xor = np.random.randn(200, 2)
    y_xor = np.logical_xor(X_xor[:, 0] > 0,
                         X_xor[:, 1] > 0)
    y_xor = np.where(y_xor, 1, -1)
    
    plt.scatter(X_xor[y_xor == 1, 0],
              X_xor[y_xor == 1, 1],
              c='b', marker='x',
              label='1')
    plt.scatter(X_xor[y_xor == -1, 0],
              X_xor[y_xor == -1, 1],
              c='r',
              marker='s',
              label='-1')
    
    plt.xlim([-3, 3])
    plt.ylim([-3, 3])
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(PIC_LOC + 'xor.png', dpi=300)
    # plt.show()
    plt.close()
    
    # Dealing with nonlinear dataset
    svm = SVC(kernel='rbf', random_state=0, gamma=0.10, C=10.0)
    svm.fit(X_xor, y_xor)
    plot_decision_regions(X_xor, y_xor,
                        classifier=svm)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(PIC_LOC + 'rbf_xor.png', dpi=300)
    # plt.show()

def linear_svm():
    X, y = get_data_sklearn()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    
    svm = SVC(kernel='rbf', random_state=0, gamma=0.10, C=10.0)
    svm.fit(X_train_std, y_train)
    plot_decision_regions(X_combined_std, y_combined,
                        classifier=svm, test_idx=range(105, 150))
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(PIC_LOC + 'support_vector_machine_linear.png', dpi=300)
    # plt.show()
    plt.close()
    
    svm = SVC(kernel='rbf', random_state=0, gamma=100.0, C=10.0)
    svm.fit(X_train_std, y_train)
    plot_decision_regions(X_combined_std, y_combined,
                        classifier=svm, test_idx=range(105, 150))
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(PIC_LOC + 'svm_linear_highgamma.png', dpi=300)
    # plt.show()
    
    
    
    
if __name__ == "__main__":
    # import pdb; pdb.set_trace()
    # support_vector_machines()
    # nonlinear_svm()
    linear_svm()