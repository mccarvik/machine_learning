import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from helpers import PL11, plot_decision_regions
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans


def k_means():
    X, y = make_blobs(n_samples=150, 
                  n_features=2, 
                  centers=3, 
                  cluster_std=0.5, 
                  shuffle=True, 
                  random_state=0)
    plt.scatter(X[:,0], X[:,1], c='white', marker='o', s=50)
    plt.grid()
    plt.tight_layout()
    plt.savefig(PL11 + 'spheres.png', dpi=300)
    plt.close()
    km = KMeans(n_clusters=3, 
                init='random', 
                n_init=10, 
                max_iter=300,
                tol=1e-04,
                random_state=0)
    y_km = km.fit_predict(X)
    plt.scatter(X[y_km==0,0], 
                X[y_km==0,1], 
                s=50, 
                c='lightgreen', 
                marker='s', 
                label='cluster 1')
    plt.scatter(X[y_km==1,0], 
                X[y_km==1,1], 
                s=50, 
                c='orange', 
                marker='o', 
                label='cluster 2')
    plt.scatter(X[y_km==2,0], 
                X[y_km==2,1], 
                s=50, 
                c='lightblue', 
                marker='v', 
                label='cluster 3')
    plt.scatter(km.cluster_centers_[:,0], 
                km.cluster_centers_[:,1], 
                s=250, 
                marker='*', 
                c='red', 
                label='centroids')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(PL11 + 'centroids.png', dpi=300)


if __name__ == "__main__":
    # import pdb; pdb.set_trace()
    k_means()