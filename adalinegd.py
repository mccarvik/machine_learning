import matplotlib as mpl
mpl.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from classification_training import plot_decision_regions
from helpers import get_data, update_check, PIC_LOC

class AdalineGD(object):
    """ADAptive LInear NEuron classifier
    
        Parameters
        ------------
        eta : float
            Learning rate (between 0.0 and 1.0)
        n_iter : int
            Passes over the training dataset.
    
        Attributes
        -----------
        w_ : 1d-array
            Weights after fitting.
        errors_ : list
            Number of misclassifications in every epoch.
    
        """
    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """ Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object

        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []
        import pdb; pdb.set_trace()
        for i in range(self.n_iter):
            import pdb; pdb.set_trace()
            output = self.net_input(X)
            errors = (y - output)
            temp = tuple(self.w_)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            if update_check(temp, self.w_):
                print(str(self.w_) + "----" + str(i))
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute linear activation"""
        return self.net_input(X)

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(X) >= 0.0, 1, -1)
    
def main1():
    y,X = get_data()
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    
    ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X, y)
    ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('log(Sum-squared-error)')
    ax[0].set_title('Adaline - Learning rate 0.01')
    
    ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
    ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Sum-squared-error')
    ax[1].set_title('Adaline - Learning rate 0.0001')
    
    plt.tight_layout()
    plt.savefig("/home/ubuntu/workspace/machine_learning/png/ch2/adaline_1.png", dpi=300)
    plt.close()
    # plt.show()

def main2():
    y,X = get_data()
    
    # standardize features
    X_std = np.copy(X)
    X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
    X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()
    
    ada = AdalineGD(n_iter=15, eta=0.01)
    ada.fit(X_std, y)
    
    plot_decision_regions(X_std, y, classifier=ada)
    plt.title('Adaline - Gradient Descent')
    plt.xlabel('sepal length [standardized]')
    plt.ylabel('petal length [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(PIC_LOC + 'adaline_2.png', dpi=300)
    plt.close()
    
    plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Sum-squared-error')
    
    plt.tight_layout()
    plt.savefig(PIC_LOC + 'adaline_3.png', dpi=300)
    # plt.show()
    plt.close()

if __name__ == "__main__":
    # import pdb; pdb.set_trace()
    # main1()
    main2()
    