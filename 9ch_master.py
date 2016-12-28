import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from helpers import PL9, plot_decision_regions
import re, pyprind, pickle, os
from nltk.corpus import stopwords
stop = stopwords.words('english')
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from movieclassifier.vectorizer import vect

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
    dest = os.path.join('movieclassifier', 'pkl_objects')
    if not os.path.exists(dest):
        os.makedirs(dest)
    pickle.dump(stop, open(os.path.join(dest, 'stopwords.pkl'), 'wb'), protocol=4)   
    pickle.dump(clf, open(os.path.join(dest, 'classifier.pkl'), 'wb'), protocol=4)
    
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
    
def pickle_test():
    label = {0:'negative', 1:'positive'}
    
    example = ['I love this movie']
    clf = pickle.load(open(os.path.join('movieclassifier/pkl_objects', 'classifier.pkl'), 'rb'))
    X = vect.transform(example)
    print('Prediction: %s\nProbability: %.2f%%' %\
          (label[clf.predict(X)[0]], clf.predict_proba(X).max()*100))

if __name__ == "__main__":
    # import pdb; pdb.set_trace()
    # out_of_core()
    pickle_test()