import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from helpers import PL8, plot_decision_regions
import pyprind
import os
import re
from nltk.stem.porter import PorterStemmer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer, HashingVectorizer 


def sentiment_analysis():
    # Only need to do this once
    # change the `basepath` to the directory of the
    # unzipped movie dataset
    #basepath = '/Users/Sebastian/Desktop/aclImdb/'
    # basepath = './aclImdb'
    # labels = {'pos':1, 'neg':0}
    # pbar = pyprind.ProgBar(50000)
    # df = pd.DataFrame()
    # for s in ('test', 'train'):
    #     for l in ('pos', 'neg'):
    #         path = os.path.join(basepath, s, l)
    #         for file in os.listdir(path):
    #             with open(os.path.join(path, file), 'r', encoding='utf-8') as infile:
    #                 txt = infile.read()
    #             df = df.append([[txt, labels[l]]], ignore_index=True)
    #             pbar.update()
    # import pdb; pdb.set_trace()
    # df.columns = ['review', 'sentiment']
    # np.random.seed(0)
    # df = df.reindex(np.random.permutation(df.index))
    # df.to_csv('./movie_data.csv', index=False)
    df = pd.read_csv('./movie_data.csv')

    count = CountVectorizer()    
    docs = np.array([    
            'The sun is shining',    
            'The weather is sweet',    
            'The sun is shining and the weather is sweet'])    
    bag = count.fit_transform(docs)
    
    # tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
    # print(tfidf.fit_transform(count.fit_transform(docs)).toarray())
    
    df['review'] = df['review'].apply(preprocessor)
    stop = stopwords.words('english')
    [w for w in tokenizer_porter('a runner likes running and runs a lot')[-10:] if w not in stop]

def preprocessor(text):    
    text = re.sub('<[^>]*>', '', text)    
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\\)|\\(|D|P)', text)    
    text = re.sub('[\\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')    
    return text

def tokenizer(text):
    return text.split()

def tokenizer_porter(text):
    porter = PorterStemmer()
    return [porter.stem(word) for word in text.split()]
    
def log_regr_doc():
    df = pd.read_csv('./movie_data.csv')
    X_train = df.loc[:25000, 'review'].values
    y_train = df.loc[:25000, 'sentiment'].values
    X_test = df.loc[25000:, 'review'].values
    y_test = df.loc[25000:, 'sentiment'].values
    
    stop = stopwords.words('english')
    tfidf = TfidfVectorizer(strip_accents=None, 
                            lowercase=False, 
                            preprocessor=None)
    param_grid = [{'vect__ngram_range': [(1,1)],
                   'vect__stop_words': [stop, None],
                   'vect__tokenizer': [tokenizer, tokenizer_porter],
                   'clf__penalty': ['l1', 'l2'],
                   'clf__C': [1.0, 10.0, 100.0]},
                 {'vect__ngram_range': [(1,1)],
                   'vect__stop_words': [stop, None],
                   'vect__tokenizer': [tokenizer, tokenizer_porter],
                   'vect__use_idf':[False],
                   'vect__norm':[None],
                   'clf__penalty': ['l1', 'l2'],
                   'clf__C': [1.0, 10.0, 100.0]},
                 ]
    lr_tfidf = Pipeline([('vect', tfidf),
                         ('clf', LogisticRegression(random_state=0))])
    gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid, 
                               scoring='accuracy',
                               cv=5, verbose=1,
                               n_jobs=-1)
    gs_lr_tfidf.fit(X_train, y_train)
    with open('./log_regr_ch8.txt', 'w') as f:
        f.write('Best parameter set: %s ' % gs_lr_tfidf.best_params_)
        f.write('CV Accuracy: %.3f' % gs_lr_tfidf.best_score_)
    
def out_of_core():
    vect = HashingVectorizer(decode_error='ignore',     
                             n_features=2**21,    
                             preprocessor=None,     
                             tokenizer=tokenizer_new)    
    clf = SGDClassifier(loss='log', random_state=1, n_iter=1)    
    doc_stream = stream_docs(path='./movie_data.csv')
    pbar = pyprind.ProgBar(45)    
    classes = np.array([0, 1])    
    for _ in range(45):
        # import pdb; pdb.set_trace()
        X_train, y_train = get_minibatch(doc_stream, size=1000)    
        if not X_train:    
            break    
        X_train = vect.transform(X_train)    
        clf.partial_fit(X_train, y_train, classes=classes)    
        pbar.update()
    
    X_test, y_test = get_minibatch(doc_stream, size=5000)
    X_test = vect.transform(X_test)
    print('\nAccuracy: %.3f' % clf.score(X_test, y_test))
    clf = clf.partial_fit(X_test, y_test)
    
def get_minibatch(doc_stream, size):    
    docs, y = [], []    
    try:    
        for _ in range(size):    
            text, label = next(doc_stream)
            docs.append(text)    
            y.append(label)    
    except StopIteration:    
        return None, None    
    return docs, y
        
def tokenizer_new(text):
    stop = stopwords.words('english')
    text = re.sub('<[^>]*>', '', text)    
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\\)|\\(|D|P)', text.lower())    
    text = re.sub('[\\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')    
    tokenized = [w for w in text.split() if w not in stop]    
    return tokenized    
        
def stream_docs(path):    
    with open(path, 'r', encoding='utf-8') as csv:    
        next(csv) # skip header    
        for line in csv:    
            text, label = line[:-3], int(line[-2])    
            yield text, label
    
    
if __name__ == "__main__":
    # import pdb; pdb.set_trace()
    # sentiment_analysis()
    log_regr_doc()
    # out_of_core()