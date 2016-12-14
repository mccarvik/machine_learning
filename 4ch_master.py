import pandas as pd
import numpy as np
from io import StringIO
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.cross_validation import train_test_split

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
    print(df_wine.head())
    
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
    
    

if __name__ == "__main__":
    # import pdb; pdb.set_trace()
    # csv_data()
    # cat_data()
    part_dataset()