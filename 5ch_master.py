import pdb
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from helpers import PL5, PL4, plot_decision_regions
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA, KernelPCA
from sklearn.lda import LDA
from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
from sklearn.datasets import make_moons, make_circles


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
        # print('MV %s: %s\n' %(label, mean_vecs[label-1]))
    
    d = 13 # number of features
    S_W = np.zeros((d, d))
    for label,mv in zip(range(1, 4), mean_vecs):
        class_scatter = np.zeros((d, d)) # scatter matrix for each class
        for row in X_train_std[y_train == label]:
            row, mv = row.reshape(d, 1), mv.reshape(d, 1) # make column vectors
            class_scatter += (row-mv).dot((row-mv).T)
        S_W += class_scatter                             # sum class scatter matrices
    # print('Within-class scatter matrix: %s' % (S_W))
    # print('Class label distribution: %s' % np.bincount(y_train)[1:])
    
    d = 13 # number of features
    S_W = np.zeros((d, d))
    for label,mv in zip(range(1, 4), mean_vecs):
        class_scatter = np.cov(X_train_std[y_train==label].T)
        S_W += class_scatter
    # print('Scaled within-class scatter matrix: %s' % (S_W))
    
    mean_overall = np.mean(X_train_std, axis=0)
    d = 13 # number of features
    S_B = np.zeros((d, d))
    for i,mean_vec in enumerate(mean_vecs):
        n = X_train[y_train==i+1, :].shape[0]
        mean_vec = mean_vec.reshape(d, 1) # make column vector
        mean_overall = mean_overall.reshape(d, 1) # make column vector
        S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)
    # print('Between-class scatter matrix: %s' % (S_B))
    
    eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
    eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i]) for i in range(len(eigen_vals))]
    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)
    # Visually confirm that the list is correctly sorted by decreasing eigenvalues
    # print('Eigenvalues in decreasing order:\\n')
    # for eigen_val in eigen_pairs:
    #     print(eigen_val[0])
    
    tot = sum(eigen_vals.real)
    discr = [(i / tot) for i in sorted(eigen_vals.real, reverse=True)]
    cum_discr = np.cumsum(discr)
    
    plt.bar(range(1, 14), discr, alpha=0.5, align='center',
            label='individual \"discriminability\"')
    plt.step(range(1, 14), cum_discr, where='mid',
             label='cumulative \"discriminability\"')
    plt.ylabel('\"discriminability\" ratio')
    plt.xlabel('Linear Discriminants')
    plt.ylim([-0.1, 1.1])
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(PL5 + 'pca6.png', dpi=300)
    plt.close()
    
    w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real,
                          eigen_pairs[1][1][:, np.newaxis].real))
    print('Matrix W:\\n', w)
    
    X_train_lda = X_train_std.dot(w)
    colors = ['r', 'b', 'g']
    markers = ['s', 'x', 'o']
    
    for l, c, m in zip(np.unique(y_train), colors, markers):
        plt.scatter(X_train_lda[y_train==l, 0] * (-1), 
                    X_train_lda[y_train==l, 1] * (-1), 
                    c=c, label=l, marker=m)
    plt.xlabel('LD 1')
    plt.ylabel('LD 2')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(PL5 + 'lda1.png', dpi=300)

def lda_scikit():
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
    
    pdb.set_trace()
    lda = LDA(n_components=3)
    X_train_lda = lda.fit_transform(X_train_std, y_train)
    lr = LogisticRegression()
    lr = lr.fit(X_train_lda, y_train)
    
    plot_decision_regions(X_train_lda, y_train, classifier=lr)
    plt.xlabel('LD 1')
    plt.ylabel('LD 2')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(PL5 + 'lda_scikit.png', dpi=300)
    plt.close()
    
    X_test_lda = lda.transform(X_test_std)
    
    plot_decision_regions(X_test_lda, y_test, classifier=lr)
    plt.xlabel('LD 1')
    plt.ylabel('LD 2')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(PL5 + 'lda_scikit_test.png', dpi=300)


def rbf_kernel_pca(X, gamma, n_components):
    """
    RBF kernel PCA implementation.

    Parameters
    ------------
    X: {NumPy ndarray}, shape = [n_samples, n_features]
        
    gamma: float
      Tuning parameter of the RBF kernel
        
    n_components: int
      Number of principal components to return

    Returns
    ------------
     X_pc: {NumPy ndarray}, shape = [n_samples, k_features]
       Projected dataset
     
     lambdas: list
        Eigenvalues

    """
    
    # Calculate pairwise squared Euclidean distances
    # in the MxN dimensional dataset.
    sq_dists = pdist(X, 'sqeuclidean')

    # Convert pairwise distances into a square matrix.
    mat_sq_dists = squareform(sq_dists)

    # Compute the symmetric kernel matrix.
    K = exp(-gamma * mat_sq_dists)

    # Center the kernel matrix.
    N = K.shape[0]
    one_n = np.ones((N,N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # Obtaining eigenpairs from the centered kernel matrix
    # numpy.eigh returns them in sorted order
    eigvals, eigvecs = eigh(K)

    # Collect the top k eigenvectors (projected samples)
    alphas = np.column_stack((eigvecs[:,-i] for i in range(1,n_components+1)))
    
    # Collect the corresponding eigenvalues
    lambdas = [eigvals[-i] for i in range(1,n_components+1)]
    
    return alphas, lambdas


def half_moon_kernel_pca():
    X, y = make_moons(n_samples=100, random_state=123)
    
    plt.scatter(X[y==0, 0], X[y==0, 1], color='red', marker='^', alpha=0.5)
    plt.scatter(X[y==1, 0], X[y==1, 1], color='blue', marker='o', alpha=0.5)
    plt.tight_layout()
    plt.savefig(PL5 + 'half_moon_1.png', dpi=300)
    plt.close()

    scikit_pca = PCA(n_components=2)
    X_spca = scikit_pca.fit_transform(X)
    fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(7,3))
    ax[0].scatter(X_spca[y==0, 0], X_spca[y==0, 1], 
                color='red', marker='^', alpha=0.5)
    ax[0].scatter(X_spca[y==1, 0], X_spca[y==1, 1],
                color='blue', marker='o', alpha=0.5)
    ax[1].scatter(X_spca[y==0, 0], np.zeros((50,1))+0.02, 
                color='red', marker='^', alpha=0.5)
    ax[1].scatter(X_spca[y==1, 0], np.zeros((50,1))-0.02,
                color='blue', marker='o', alpha=0.5)
    ax[0].set_xlabel('PC1')
    ax[0].set_ylabel('PC2')
    ax[1].set_ylim([-1, 1])
    ax[1].set_yticks([])
    ax[1].set_xlabel('PC1')
    
    plt.tight_layout()
    plt.savefig(PL5 + 'half_moon_2.png', dpi=300)
    plt.close()

    X_kpca = rbf_kernel_pca(X, gamma=15, n_components=2)[0]
    fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(7,3))
    ax[0].scatter(X_kpca[y==0, 0], X_kpca[y==0, 1], 
                color='red', marker='^', alpha=0.5)
    ax[0].scatter(X_kpca[y==1, 0], X_kpca[y==1, 1],
                color='blue', marker='o', alpha=0.5)
    ax[1].scatter(X_kpca[y==0, 0], np.zeros((50,1))+0.02, 
                color='red', marker='^', alpha=0.5)
    ax[1].scatter(X_kpca[y==1, 0], np.zeros((50,1))-0.02,
                color='blue', marker='o', alpha=0.5)
    ax[0].set_xlabel('PC1')
    ax[0].set_ylabel('PC2')
    ax[1].set_ylim([-1, 1])
    ax[1].set_yticks([])
    ax[1].set_xlabel('PC1')
    ax[0].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
    ax[1].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
    
    plt.tight_layout()
    plt.savefig(PL5 + 'half_moon_3.png', dpi=300)

def circles_kernel_pca():
    X, y = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)
    plt.scatter(X[y==0, 0], X[y==0, 1], color='red', marker='^', alpha=0.5)
    plt.scatter(X[y==1, 0], X[y==1, 1], color='blue', marker='o', alpha=0.5)
    plt.tight_layout()
    plt.savefig(PL5 + 'circles_1.png', dpi=300)
    plt.close()
    
    scikit_pca = PCA(n_components=2)
    X_spca = scikit_pca.fit_transform(X)
    fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(7,3))
    ax[0].scatter(X_spca[y==0, 0], X_spca[y==0, 1], 
                color='red', marker='^', alpha=0.5)
    ax[0].scatter(X_spca[y==1, 0], X_spca[y==1, 1],
                color='blue', marker='o', alpha=0.5)
    
    ax[1].scatter(X_spca[y==0, 0], np.zeros((500,1))+0.02, 
                color='red', marker='^', alpha=0.5)
    ax[1].scatter(X_spca[y==1, 0], np.zeros((500,1))-0.02,
                color='blue', marker='o', alpha=0.5)
    ax[0].set_xlabel('PC1')
    ax[0].set_ylabel('PC2')
    ax[1].set_ylim([-1, 1])
    ax[1].set_yticks([])
    ax[1].set_xlabel('PC1')
    plt.tight_layout()
    plt.savefig(PL5 + 'circles_2.png', dpi=300)
    
    X_kpca = rbf_kernel_pca(X, gamma=15, n_components=2)[0]

    fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(7,3))
    ax[0].scatter(X_kpca[y==0, 0], X_kpca[y==0, 1], 
                color='red', marker='^', alpha=0.5)
    ax[0].scatter(X_kpca[y==1, 0], X_kpca[y==1, 1],
                color='blue', marker='o', alpha=0.5)
    
    ax[1].scatter(X_kpca[y==0, 0], np.zeros((500,1))+0.02, 
                color='red', marker='^', alpha=0.5)
    ax[1].scatter(X_kpca[y==1, 0], np.zeros((500,1))-0.02,
                color='blue', marker='o', alpha=0.5)
    ax[0].set_xlabel('PC1')
    ax[0].set_ylabel('PC2')
    ax[1].set_ylim([-1, 1])
    ax[1].set_yticks([])
    ax[1].set_xlabel('PC1')
    plt.tight_layout()
    plt.savefig(PL5 + 'circles_3.png', dpi=300)
    plt.close()

def project_new_data_kernel_pca():
    X, y = make_moons(n_samples=100, random_state=123)
    alphas, lambdas = rbf_kernel_pca(X, gamma=15, n_components=1)
    x_new = X[25]
    print(x_new)
    x_proj = alphas[25]
    print(x_proj)
    
    # projection of the "new" datapoint
    x_reproj = project_x(x_new, X, gamma=15, alphas=alphas, lambdas=lambdas)
    print(x_reproj)
    
    plt.scatter(alphas[y==0, 0], np.zeros((50)), 
            color='red', marker='^',alpha=0.5)
    plt.scatter(alphas[y==1, 0], np.zeros((50)), 
            color='blue', marker='o', alpha=0.5)
    plt.scatter(x_proj, 0, color='black', label='original projection of point X[25]', marker='^', s=100)
    plt.scatter(x_reproj, 0, color='green', label='remapped point X[25]', marker='x', s=500)
    plt.legend(scatterpoints=1)
    plt.tight_layout()
    plt.savefig(PL5 + 'new_data.png', dpi=300)
    

def project_x(x_new, X, gamma, alphas, lambdas):
    pair_dist = np.array([np.sum((x_new-row)**2) for row in X])
    k = np.exp(-gamma * pair_dist)
    return k.dot(alphas / lambdas)

def scikit_kernel_pca():
    X, y = make_moons(n_samples=100, random_state=123)
    scikit_kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
    X_skernpca = scikit_kpca.fit_transform(X)
    
    plt.scatter(X_skernpca[y==0, 0], X_skernpca[y==0, 1], 
                color='red', marker='^', alpha=0.5)
    plt.scatter(X_skernpca[y==1, 0], X_skernpca[y==1, 1], 
                color='blue', marker='o', alpha=0.5)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()
    plt.savefig(PL5 + 'scikit_kernel_pca.png', dpi=300)


if __name__ == "__main__":
    # import pdb; pdb.set_trace()
    # tot_vs_explained_var()
    # pca_scikit()
    # linear_discriminant_analysis()
    lda_scikit()
    # half_moon_kernel_pca()
    # circles_kernel_pca()
    # project_new_data_kernel_pca()
    # scikit_kernel_pca()