import sys, pdb
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
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
    
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
    
    pdb.set_trace()
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
    

def gini(p):
    return (p)*(1 - (p)) + (1-p)*(1 - (1-p))
  
  
def entropy(p):
    return - p*np.log2(p) - (1 - p)*np.log2((1 - p))
  
  
def error(p):
    return 1 - np.max([p, 1 - p])
  
def dec_tree_impurity():
    x = np.arange(0.0, 1.0, 0.01)
    
    ent = [entropy(p) if p != 0 else None for p in x]
    sc_ent = [e*0.5 if e else None for e in ent]
    err = [error(i) for i in x]
    fig = plt.figure()
    ax = plt.subplot(111)
    for i, lab, ls, c, in zip([ent, sc_ent, gini(x), err], 
                            ['Entropy', 'Entropy (scaled)', 
                             'Gini Impurity', 'Misclassification Error'],
                            ['-', '-', '--', '-.'],
                            ['black', 'lightgray', 'red', 'green', 'cyan']):
      line = ax.plot(x, i, label=lab, linestyle=ls, lw=2, color=c)
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
            ncol=3, fancybox=True, shadow=False)
    
    ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
    ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
    plt.ylim([0, 1.1])
    plt.xlabel('p(i=1)')
    plt.ylabel('Impurity Index')
    plt.tight_layout()
    plt.savefig(PIC_LOC + '/impurity.png', dpi=300, bbox_inches='tight')

def build_dec_tree():
    X, y = get_data_sklearn()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
    tree.fit(X_train, y_train)
    
    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))
    plot_decision_regions(X_combined, y_combined, 
                        classifier=tree, test_idx=range(105,150))
    
    plt.xlabel('petal length [cm]')
    plt.ylabel('petal width [cm]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(PIC_LOC + 'decision_tree_decision.png', dpi=300)
  
    export_graphviz(tree, 
                  out_file='tree.dot', 
                  feature_names=['petal length', 'petal width'])
    
    # execute "dot -Tpng tree.dot -o tree.png" to turn file into png file

def random_forests():
    X, y = get_data_sklearn()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    forest = RandomForestClassifier(criterion='entropy',
                                  n_estimators=10, 
                                  random_state=1,
                                  n_jobs=2)
    forest.fit(X_train, y_train)
    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))
    plot_decision_regions(X_combined, y_combined, 
                        classifier=forest, test_idx=range(105,150))
    
    plt.xlabel('petal length [cm]')
    plt.ylabel('petal width [cm]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(PIC_LOC + 'random_forest.png', dpi=300)
    # plt.show()"

def KNN_model():
    X, y = get_data_sklearn()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    X_combined = np.vstack((X_train, X_test))
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
    knn.fit(X_train_std, y_train)
    
    plot_decision_regions(X_combined_std, y_combined, 
                        classifier=knn, test_idx=range(105,150))
    
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(PIC_LOC + 'k_nearest_neighbors.png', dpi=300)
    # plt.show()

    
if __name__ == "__main__":
    main2()
    # import pdb; pdb.set_trace()
    # support_vector_machines()
    # nonlinear_svm()
    # linear_svm()
    # dec_tree_impurity()
    # build_dec_tree()
    # random_forests()
    # KNN_model()