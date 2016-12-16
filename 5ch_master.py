import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from helpers import PL5, PL4, plot_decision_regions
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

def tot_vs_explained_var():
    df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
    df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 
    'Alcalinity of ash', 'Magnesium', 'Total phenols', 
    'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 
    'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']
   
    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
    X_train, X_test, y_train, y_test = \
          train_test_split(X, y, test_size=0.3, random_state=0)
  
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)
    
    cov_mat = np.cov(X_train_std.T)
    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
    print('Eigenvalues \n%s' % eigen_vals)
    tot = sum(eigen_vals)
    var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)
    
    plt.bar(range(1, 14), var_exp, alpha=0.5, align='center',
         label='individual explained variance')
    plt.step(range(1, 14), cum_var_exp, where='mid',
          label='cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(PL5 + 'pca1.png', dpi=300)
    plt.close()
    # plt.show()
    
    eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i]) for i in range(len(eigen_vals))]
    eigen_pairs.sort(reverse=True)
    w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
               eigen_pairs[1][1][:, np.newaxis]))
    # print('Matrix W:\n', w)
    
    X_train_pca = X_train_std.dot(w)
    colors = ['r', 'b', 'g']
    markers = ['s', 'x', 'o']
    
    for l, c, m in zip(np.unique(y_train), colors, markers):
        plt.scatter(X_train_pca[y_train==l, 0], 
                    X_train_pca[y_train==l, 1], 
                    c=c, label=l, marker=m)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(PL5 + 'pca2.png', dpi=300)
    
    

def pca_scikit():
    df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
    df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 
    'Alcalinity of ash', 'Magnesium', 'Total phenols', 
    'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 
    'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']
    
    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
    X_train, X_test, y_train, y_test = \
          train_test_split(X, y, test_size=0.3, random_state=0)
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)
    
    pca = PCA(n_components=2)
    lr = LogisticRegression()
    X_train_pca = pca.fit_transform(X_train_std)
    X_test_pca = pca.transform(X_test_std)
    lr = lr.fit(X_train_pca, y_train)
    plot_decision_regions(X_train_pca, y_train, classifier=lr)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(PL5 + 'pca4.png', dpi=300)
    plt.close()
    
    plot_decision_regions(X_test_pca, y_test, classifier=lr)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(PL5 + 'pca5.png', dpi=300)
    # plt.show()
    
    pca = PCA(n_components=None)
    X_train_pca = pca.fit_transform(X_train_std)
    print(pca.explained_variance_ratio_)

def linear_discriminant_analysis():
    df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
    df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 
    'Alcalinity of ash', 'Magnesium', 'Total phenols', 
    'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 
    'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']
    
    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
    X_train, X_test, y_train, y_test = \
          train_test_split(X, y, test_size=0.3, random_state=0)
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)    
    
    np.set_printoptions(precision=4)
    mean_vecs = []
    for label in range(1,4):
        mean_vecs.append(np.mean(X_train_std[y_train==label], axis=0))
        print('MV %s: %s\n' %(label, mean_vecs[label-1]))
    
    d = 13 # number of features
    S_W = np.zeros((d, d))
    for label,mv in zip(range(1, 4), mean_vecs):
        class_scatter = np.zeros((d, d)) # scatter matrix for each class
        for row in X_train_std[y_train == label]:
            row, mv = row.reshape(d, 1), mv.reshape(d, 1) # make column vectors
            class_scatter += (row-mv).dot((row-mv).T)
        S_W += class_scatter                             # sum class scatter matrices
    print('Within-class scatter matrix: %sx%s' % (S_W.shape[0], S_W.shape[1]))
    print('Class label distribution: %s' % np.bincount(y_train)[1:])
    
    d = 13 # number of features
    S_W = np.zeros((d, d))
    for label,mv in zip(range(1, 4), mean_vecs):
        class_scatter = np.cov(X_train_std[y_train==label].T)
        S_W += class_scatter
    print('Scaled within-class scatter matrix: %sx%s' % (S_W.shape[0], S_W.shape[1]))
    
    mean_overall = np.mean(X_train_std, axis=0)
    d = 13 # number of features
    S_B = np.zeros((d, d))
    for i,mean_vec in enumerate(mean_vecs):
        n = X_train[y_train==i+1, :].shape[0]
        mean_vec = mean_vec.reshape(d, 1) # make column vector
        mean_overall = mean_overall.reshape(d, 1) # make column vector
        S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)
    print('Between-class scatter matrix: %sx%s' % (S_B.shape[0], S_B.shape[1]))
    

if __name__ == "__main__":
    # import pdb; pdb.set_trace()
    # tot_vs_explained_var()
    # pca_scikit()
    linear_discriminant_analysis()