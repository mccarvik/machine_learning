import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
from helpers import PL10, plot_decision_regions
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Ridge, ElasticNet, Lasso


def exploratory_data_analysis():
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data',     
                     header=None, sep='\\s+')    
    df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS',     
                  'NOX', 'RM', 'AGE', 'DIS', 'RAD',     
                  'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']    
    print(df.head())
    
    sns.set(style='whitegrid', context='notebook')    
    cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']    
    sns.pairplot(df[cols], size=2.5)    
    plt.tight_layout()    
    plt.savefig(PL10 + 'scatter.png', dpi=300)
    plt.close()
    
    cm = np.corrcoef(df[cols].values.T)
    sns.set(font_scale=1.5)
    hm = sns.heatmap(cm, 
                cbar=True,
                annot=True, 
                square=True,
                fmt='.2f',
                annot_kws={'size': 15},
                yticklabels=cols,
                xticklabels=cols)
    plt.tight_layout()
    plt.savefig(PL10 + 'corr_mat.png', dpi=300)
    plt.close()
    
    X = df[['RM']].values
    y = df['MEDV'].values
    sc_x = StandardScaler()
    sc_y = StandardScaler()
    X_std = sc_x.fit_transform(X)
    y_std = sc_y.fit_transform(y)
    lr = LinearRegressionGD()
    lr.fit(X_std, y_std)
    plt.plot(range(1, lr.n_iter+1), lr.cost_)
    plt.ylabel('SSE')
    plt.xlabel('Epoch')
    plt.tight_layout()
    plt.savefig(PL10 + 'cost.png', dpi=300)
    plt.close()
    
    lin_regplot(X_std, y_std, lr)
    plt.xlabel('Average number of rooms [RM] (standardized)')
    plt.ylabel('Price in $1000\'s [MEDV] (standardized)')
    plt.tight_layout()
    plt.savefig(PL10 + 'gradient_fit.png', dpi=300)
    plt.close()
    
    num_rooms_std = sc_x.transform([5.0])
    price_std = lr.predict(num_rooms_std)
    print("Price in $1000's: %.3f" % sc_y.inverse_transform(price_std))
    print('Slope: %.3f' % lr.w_[1])
    print('Intercept: %.3f' % lr.w_[0])
    
    slr = LinearRegression()
    slr.fit(X, y)
    y_pred = slr.predict(X)
    print('Slope: %.3f' % slr.coef_[0])
    print('Intercept: %.3f' % slr.intercept_)
    
    lin_regplot(X, y, slr)
    plt.xlabel('Average number of rooms [RM]')
    plt.ylabel('Price in $1000\'s [MEDV]')
    plt.tight_layout()
    plt.savefig(PL10 + 'scikit_lr_fit.png', dpi=300)
    plt.close()

    # adding a column vector of \"ones\"
    Xb = np.hstack((np.ones((X.shape[0], 1)), X))
    w = np.zeros(X.shape[1])
    z = np.linalg.inv(np.dot(Xb.T, Xb))
    w = np.dot(z, np.dot(Xb.T, y))
    print('Slope: %.3f' % w[1])
    print('Intercept: %.3f' % w[0])
    
def ransac():
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data',     
                     header=None, sep='\\s+')    
    df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS',     
                  'NOX', 'RM', 'AGE', 'DIS', 'RAD',     
                  'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
                  
    ransac = RANSACRegressor(LinearRegression(), 
                             max_trials=100, 
                             min_samples=50, 
                             residual_metric=lambda x: np.sum(np.abs(x), axis=1), 
                             residual_threshold=5.0, 
                             random_state=0)
    X = df[['RM']].values
    y = df['MEDV'].values                         
    ransac.fit(X, y)
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)
    line_X = np.arange(3, 10, 1)
    line_y_ransac = ransac.predict(line_X[:, np.newaxis])
    plt.scatter(X[inlier_mask], y[inlier_mask], c='blue', marker='o', label='Inliers')
    plt.scatter(X[outlier_mask], y[outlier_mask], c='lightgreen', marker='s', label='Outliers')
    plt.plot(line_X, line_y_ransac, color='red')   
    plt.xlabel('Average number of rooms [RM]')
    plt.ylabel('Price in $1000\'s [MEDV]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(PL10 + 'ransac_fit.png', dpi=300)
    plt.close()
    
    print('Slope: %.3f' % ransac.estimator_.coef_[0])
    print('Intercept: %.3f' % ransac.estimator_.intercept_)

def performance():
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data',     
                     header=None, sep='\\s+')    
    df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS',     
                  'NOX', 'RM', 'AGE', 'DIS', 'RAD',     
                  'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    X = df.iloc[:, :-1].values    
    y = df['MEDV'].values    
        
    X_train, X_test, y_train, y_test = train_test_split(    
        X, y, test_size=0.3, random_state=0)
    slr = LinearRegression()
    slr.fit(X_train, y_train)
    y_train_pred = slr.predict(X_train)
    y_test_pred = slr.predict(X_test)
    plt.scatter(y_train_pred,  y_train_pred - y_train, c='blue', marker='o', label='Training data')
    plt.scatter(y_test_pred,  y_test_pred - y_test, c='lightgreen', marker='s', label='Test data')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red')
    plt.xlim([-10, 50])
    plt.tight_layout()
    plt.savefig(PL10 + 'slr_residuals.png', dpi=300)
    plt.close()
    
    print('MSE train: %.3f, test: %.3f' % (    
            mean_squared_error(y_train, y_train_pred),    
            mean_squared_error(y_test, y_test_pred)))    
    print('R^2 train: %.3f, test: %.3f' % (    
            r2_score(y_train, y_train_pred),    
            r2_score(y_test, y_test_pred)))
    
    # Regularized Linear Regression
    lasso = Ridge(alpha=1.0)
    # lasso = ElasticNet(alpha=0.1, l1_ratio=0.5)
    # lasso = Lasso(alpha=0.1)
    lasso.fit(X_train, y_train)
    y_train_pred = lasso.predict(X_train)
    y_test_pred = lasso.predict(X_test)
    print(lasso.coef_)
    

def lin_regplot(X, y, model):
    plt.scatter(X, y, c='lightblue')
    plt.plot(X, model.predict(X), color='red', linewidth=2)
    return None


class LinearRegressionGD(object):
    def __init__(self, eta=0.001, n_iter=20):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return self.net_input(X)

def polynomial_regression():
    X = np.array([258.0, 270.0, 294.0, 
                  320.0, 342.0, 368.0, 
                  396.0, 446.0, 480.0, 586.0])[:, np.newaxis]
    y = np.array([236.4, 234.4, 252.8, 
                  298.6, 314.2, 342.2, 
                  360.8, 368.0, 391.2,
                  390.8])
    lr = LinearRegression()
    pr = LinearRegression()
    quadratic = PolynomialFeatures(degree=2)
    X_quad = quadratic.fit_transform(X)
    # fit linear features
    lr.fit(X, y)
    X_fit = np.arange(250,600,10)[:, np.newaxis]
    y_lin_fit = lr.predict(X_fit)
    
    # fit quadratic features
    pr.fit(X_quad, y)
    y_quad_fit = pr.predict(quadratic.fit_transform(X_fit))
    
    # plot results
    plt.scatter(X, y, label='training points')
    plt.plot(X_fit, y_lin_fit, label='linear fit', linestyle='--')
    plt.plot(X_fit, y_quad_fit, label='quadratic fit')
    plt.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(PL10 + 'poly_example.png', dpi=300)
    
    y_lin_pred = lr.predict(X)
    y_quad_pred = pr.predict(X_quad)
    print('Training MSE linear: %.3f, quadratic: %.3f' % (    
            mean_squared_error(y, y_lin_pred),    
            mean_squared_error(y, y_quad_pred)))    
    print('Training R^2 linear: %.3f, quadratic: %.3f' % (    
            r2_score(y, y_lin_pred),    
            r2_score(y, y_quad_pred)))


if __name__ == "__main__":
    # import pdb; pdb.set_trace()
    # exploratory_data_analysis()
    # ransac()
    # performance()
    polynomial_regression()