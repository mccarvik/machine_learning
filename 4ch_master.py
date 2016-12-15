import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from helpers import PL4
from sklearn.base import clone
from itertools import combinations
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


def csv_data():
    csv_data = '''A,B,C,D
    1.0,2.0,3.0,4.0
    5.0,6.0,,8.0
    10.0,11.0,12.0,'''
    # If you are using Python 2.7, you need
    # to convert the string to unicode:
    # csv_data = unicode(csv_data)
    df = pd.read_csv(StringIO(csv_data))
    print(df)
    
    imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imr = imr.fit(df)
    imputed_data = imr.transform(df.values)
    print(imputed_data)

def cat_data():
    df = pd.DataFrame([
              ['green', 'M', 10.1, 'class1'], 
              ['red', 'L', 13.5, 'class2'], 
              ['blue', 'XL', 15.3, 'class1']])
    
    df.columns = ['color', 'size', 'price', 'classlabel']
    # print(df)
    
    size_mapping = {
               'XL': 3,
               'L': 2,
               'M': 1}
    
    df['size'] = df['size'].map(size_mapping)
    class_mapping = {label:idx for idx,label in enumerate(np.unique(df['classlabel']))}
    # print(class_mapping)
    
    df['classlabel'] = df['classlabel'].map(class_mapping)
    # print(df)
    
    inv_size_mapping = {v: k for k, v in class_mapping.items()}
    df['classlabel'] = df['classlabel'].map(inv_size_mapping)
    # print(df)
    
    
    
    class_le = LabelEncoder()
    y = class_le.fit_transform(df['classlabel'].values)
    # print(y)
    # print(class_le.inverse_transform(y))
    
    X = df[['color', 'size', 'price']].values
    color_le = LabelEncoder()
    X[:, 0] = color_le.fit_transform(X[:, 0])
    print(X)
    
    ohe = OneHotEncoder(categorical_features=[0])
    print(ohe.fit_transform(X).toarray())
    print(pd.get_dummies(df[['price', 'color', 'size']]))
 
def part_dataset():
    df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
    
    df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 
    'Alcalinity of ash', 'Magnesium', 'Total phenols', 
    'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 
    'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']
    # print(df_wine.head())
    
    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
    X_train, X_test, y_train, y_test = \
          train_test_split(X, y, test_size=0.3, random_state=0)
    
    # Normalization of the data
    mms = MinMaxScaler()
    X_train_norm = mms.fit_transform(X_train)
    X_test_norm = mms.transform(X_test)
    
    # Standardization of the data
    stdsc = StandardScaler()
    X_train_std = stdsc.fit_transform(X_train)
    X_test_std = stdsc.transform(X_test)
    
    lr = LogisticRegression(penalty='l1', C=0.1)
    lr.fit(X_train_std, y_train)
    # print('Training accuracy:', lr.score(X_train_std, y_train))
    # print('Test accuracy:', lr.score(X_test_std, y_test))
    # print(lr.intercept_)
    # print(lr.coef_)
    
    fig = plt.figure()
    ax = plt.subplot(111)
      
    colors = ['blue', 'green', 'red', 'cyan', 
           'magenta', 'yellow', 'black', 
            'pink', 'lightgreen', 'lightblue', 
            'gray', 'indigo', 'orange']
    
    weights, params = [], []
    for c in np.arange(-4, 6):
      lr = LogisticRegression(penalty='l1', C=10**c, random_state=0)
      lr.fit(X_train_std, y_train)
      weights.append(lr.coef_[1])
      params.append(10**c)
    
    weights = np.array(weights)
    
    for column, color in zip(range(weights.shape[1]), colors):
        plt.plot(params, weights[:, column],
               label=df_wine.columns[column+1],
               color=color)
    plt.axhline(0, color='black', linestyle='--', linewidth=3)
    plt.xlim([10**(-5), 10**5])
    plt.ylabel('weight coefficient')
    plt.xlabel('C')
    plt.xscale('log')
    plt.legend(loc='upper left')
    ax.legend(loc='upper center', 
            bbox_to_anchor=(1.38, 1.03),
            ncol=1, fancybox=True)
    plt.savefig(PL4 + 'l1_path.png', dpi=300)
    # plt.show()

class SBS():
    def __init__(self, estimator, k_features, scoring=accuracy_score,
               test_size=0.25, random_state=1):
      self.scoring = scoring
      self.estimator = clone(estimator)
      self.k_features = k_features
      self.test_size = test_size
      self.random_state = random_state
    
    def fit(self, X, y):
        X_train, X_test, y_train, y_test = \
              train_test_split(X, y, test_size=self.test_size, 
                               random_state=self.random_state)
    
        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train, 
                               X_test, y_test, self.indices_)
        self.scores_ = [score]
    
        while dim > self.k_features:
            scores = []
            subsets = []
    
            for p in combinations(self.indices_, r=dim-1):
                score = self._calc_score(X_train, y_train, 
                                       X_test, y_test, p)
                scores.append(score)
                subsets.append(p)
    
            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1
    
            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]
        return self
    
    def transform(self, X):
        return X[:, self.indices_]
    
    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score

def knn_sbs():
    df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
    
    df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 
    'Alcalinity of ash', 'Magnesium', 'Total phenols', 
    'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 
    'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']
    
    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
    X_train, X_test, y_train, y_test = \
          train_test_split(X, y, test_size=0.3, random_state=0)
    # Standardization of the data
    stdsc = StandardScaler()
    X_train_std = stdsc.fit_transform(X_train)
    X_test_std = stdsc.transform(X_test)
    
    knn = KNeighborsClassifier(n_neighbors=2)
    # selecting features
    sbs = SBS(knn, k_features=1)
    sbs.fit(X_train_std, y_train)
    
    # plotting performance of feature subsets
    k_feat = [len(k) for k in sbs.subsets_]
    plt.plot(k_feat, sbs.scores_, marker='o')
    plt.ylim([0.7, 1.1])
    plt.ylabel('Accuracy')
    plt.xlabel('Number of features')
    plt.grid()
    plt.tight_layout()
    plt.savefig(PL4 + 'sbs.png', dpi=300)
    
    k5 = list(sbs.subsets_[8])
    print(df_wine.columns[1:][k5])
    
    knn.fit(X_train_std, y_train)
    print('Training accuracy:', knn.score(X_train_std, y_train))
    print('Test accuracy:', knn.score(X_test_std, y_test))
    
    knn.fit(X_train_std[:, k5], y_train)
    print('Training accuracy:', knn.score(X_train_std[:, k5], y_train))
    print('Test accuracy:', knn.score(X_test_std[:, k5], y_test))

def random_forest_classifer():
    df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
    
    df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 
    'Alcalinity of ash', 'Magnesium', 'Total phenols', 
    'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 
    'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']
    
    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
    X_train, X_test, y_train, y_test = \
          train_test_split(X, y, test_size=0.3, random_state=0)
    
    feat_labels = df_wine.columns[1:]
    forest = RandomForestClassifier(n_estimators=10000,
                                  random_state=0,
                                  n_jobs=-1)
    forest.fit(X_train, y_train)
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    for f in range(X_train.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30, 
            feat_labels[indices[f]], 
            importances[indices[f]]))
                    
    plt.title('Feature Importances')
    plt.bar(range(X_train.shape[1]), 
                importances[indices],
                color='lightblue', 
                align='center')
  
    plt.xticks(range(X_train.shape[1]), 
             feat_labels[indices], rotation=90)
    plt.xlim([-1, X_train.shape[1]])
    plt.tight_layout()
    plt.savefig(PL4 + 'random_forest.png', dpi=300)
    
    X_selected = forest.transform(X_train, threshold=0.15)
    print(X_selected.shape)

if __name__ == "__main__":
    # import pdb; pdb.set_trace()
    # csv_data()
    # cat_data()
    # part_dataset()
    # knn_sbs()
    random_forest_classifer()