import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from helpers import PL10, plot_decision_regions


def exploratory_data_analysis():
    import pandas as pd    
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data',     
                     header=None, sep='\\s+')    
    df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS',     
                  'NOX', 'RM', 'AGE', 'DIS', 'RAD',     
                  'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']    
    print(df.head())

if __name__ == "__main__":
    # import pdb; pdb.set_trace()
    exploratory_data_analysis()