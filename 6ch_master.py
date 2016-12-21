import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from helpers import PL6, plot_decision_regions
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.learning_curve import learning_curve, validation_curve
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, make_scorer


def hyper_paramater_tuning():
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header=None)
    X = df.loc[:, 2:].values
    y = df.loc[:, 1].values
    le = LabelEncoder()
    y = le.fit_transform(y)
    le.transform(['M', 'B'])
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.20, random_state=1)
        
    pipe_lr = Pipeline([('scl', StandardScaler()),
            ('pca', PCA(n_components=2)),
            ('clf', LogisticRegression(random_state=1))])

    pipe_lr.fit(X_train, y_train)
    print('Test Accuracy: %.3f' % pipe_lr.score(X_test, y_test))
    y_pred = pipe_lr.predict(X_test)

def kfold_cross_validation():
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header=None)
    X = df.loc[:, 2:].values
    y = df.loc[:, 1].values
    le = LabelEncoder()
    y = le.fit_transform(y)
    le.transform(['M', 'B'])
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.20, random_state=1)
    
    pipe_lr = Pipeline([('scl', StandardScaler()),
            ('pca', PCA(n_components=2)),
            ('clf', LogisticRegression(random_state=1))])

    kfold = StratifiedKFold(y=y_train, 
                            n_folds=10,
                            random_state=1)
    
    scores = []
    for k, (train, test) in enumerate(kfold):
        pipe_lr.fit(X_train[train], y_train[train])
        score = pipe_lr.score(X_train[test], y_train[test])
        scores.append(score)
        print('Fold: %s, Class dist.: %s, Acc: %.3f' % (k+1, np.bincount(y_train[train]), score))
    print('\\nCV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

    scores = cross_val_score(estimator=pipe_lr, 
                             X=X_train, 
                             y=y_train, 
                             cv=10,
                             n_jobs=1)
    print('CV accuracy scores: %s' % scores)
    print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

def debug_variance_and_bias():
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header=None)
    X = df.loc[:, 2:].values
    y = df.loc[:, 1].values
    le = LabelEncoder()
    y = le.fit_transform(y)
    le.transform(['M', 'B'])
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.20, random_state=1)
        
    pipe_lr = Pipeline([('scl', StandardScaler()),
            ('clf', LogisticRegression(penalty='l2', random_state=0))])

    train_sizes, train_scores, test_scores = \
                learning_curve(estimator=pipe_lr, 
                X=X_train, 
                y=y_train, 
                train_sizes=np.linspace(0.1, 1.0, 10), 
                cv=10,
                n_jobs=1)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.plot(train_sizes, train_mean, 
             color='blue', marker='o', 
             markersize=5, label='training accuracy')
    plt.fill_between(train_sizes, 
                     train_mean + train_std,
                     train_mean - train_std, 
                     alpha=0.15, color='blue')
    plt.plot(train_sizes, test_mean, 
             color='green', linestyle='--', 
             marker='s', markersize=5, 
             label='validation accuracy')
    plt.fill_between(train_sizes, 
                     test_mean + test_std,
                     test_mean - test_std, 
                     alpha=0.15, color='green')
    plt.grid()
    plt.xlabel('Number of training samples')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.ylim([0.8, 1.0])
    plt.tight_layout()
    plt.savefig(PL6 + 'learning_curve.png', dpi=300)

def addressing_over_under_fitting():
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header=None)
    X = df.loc[:, 2:].values
    y = df.loc[:, 1].values
    le = LabelEncoder()
    y = le.fit_transform(y)
    le.transform(['M', 'B'])
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.20, random_state=1)
    pipe_lr = Pipeline([('scl', StandardScaler()),
            ('clf', LogisticRegression(penalty='l2', random_state=0))])
    
    param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    train_scores, test_scores = validation_curve(
                    estimator=pipe_lr, 
                    X=X_train, 
                    y=y_train, 
                    param_name='clf__C', 
                    param_range=param_range,
                    cv=10)
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.plot(param_range, train_mean, 
             color='blue', marker='o', 
             markersize=5, label='training accuracy')
    plt.fill_between(param_range, train_mean + train_std,
                     train_mean - train_std, alpha=0.15,
                     color='blue')
    plt.plot(param_range, test_mean, 
             color='green', linestyle='--', 
             marker='s', markersize=5, 
             label='validation accuracy')
    plt.fill_between(param_range, 
                     test_mean + test_std,
                     test_mean - test_std, 
                     alpha=0.15, color='green')
    plt.grid()
    plt.xscale('log')
    plt.legend(loc='lower right')
    plt.xlabel('Parameter C')
    plt.ylabel('Accuracy')
    plt.ylim([0.8, 1.0])
    plt.tight_layout()
    plt.savefig(PL6 + 'validation_curve.png', dpi=300)
    
def grid_search():
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header=None)
    X = df.loc[:, 2:].values
    y = df.loc[:, 1].values
    le = LabelEncoder()
    y = le.fit_transform(y)
    le.transform(['M', 'B'])
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.20, random_state=1)
    pipe_svc = Pipeline([('scl', StandardScaler()),
                ('clf', SVC(random_state=1))])
    
    param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    
    param_grid = [{'clf__C': param_range, 
                   'clf__kernel': ['linear']},
                     {'clf__C': param_range, 
                      'clf__gamma': param_range, 
                      'clf__kernel': ['rbf']}]
    
    gs = GridSearchCV(estimator=pipe_svc, 
                      param_grid=param_grid, 
                      scoring='accuracy', 
                      cv=10,
                      n_jobs=-1)
    gs = gs.fit(X_train, y_train)
    print(gs.best_score_)
    print(gs.best_params_)
    clf = gs.best_estimator_
    clf.fit(X_train, y_train)
    print('Test accuracy: %.3f' % clf.score(X_test, y_test))
    
    gs = GridSearchCV(estimator=pipe_svc, 
                                param_grid=param_grid, 
                                scoring='accuracy', 
                                cv=2)
    scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=5)
    print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
    
    gs = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0), 
                                param_grid=[{'max_depth': [1, 2, 3, 4, 5, 6, 7, None]}], 
                                scoring='accuracy', 
                                cv=2)
    scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=5)
    print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

def performance_evaluation():
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header=None)
    X = df.loc[:, 2:].values
    y = df.loc[:, 1].values
    le = LabelEncoder()
    y = le.fit_transform(y)
    le.transform(['M', 'B'])
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.20, random_state=1)
    pipe_svc = Pipeline([('scl', StandardScaler()),
            ('clf', SVC(random_state=1))])
        
    pipe_svc.fit(X_train, y_train)
    y_pred = pipe_svc.predict(X_test)
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print(confmat)

    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
    plt.xlabel('predicted label')
    plt.ylabel('true label')
    plt.tight_layout()
    plt.savefig(PL6 + 'confusion_matrix.png', dpi=300)
    plt.close()
    
    print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred))
    print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred))
    print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred))
    

    scorer = make_scorer(f1_score, pos_label=0)
    c_gamma_range = [0.01, 0.1, 1.0, 10.0]
    param_grid = [{'clf__C': c_gamma_range, 
                   'clf__kernel': ['linear']},
                     {'clf__C': c_gamma_range, 
                      'clf__gamma': c_gamma_range, 
                      'clf__kernel': ['rbf'],}]
    gs = GridSearchCV(estimator=pipe_svc, 
                                    param_grid=param_grid, 
                                    scoring=scorer, 
                                    cv=10,
                                    n_jobs=-1)
    gs = gs.fit(X_train, y_train)
    print(gs.best_score_)
    print(gs.best_params_)

if __name__ == "__main__":
    # hyper_paramater_tuning()
    # kfold_cross_validation()
    # debug_variance_and_bias()
    # addressing_over_under_fitting()
    # grid_search()
    performance_evaluation()