import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from helpers import PL11, plot_decision_regions
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples


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
    print('Distortion: %.2f' % km.inertia_)

def elbow():
    X, y = make_blobs(n_samples=150, 
                  n_features=2, 
                  centers=3, 
                  cluster_std=0.5, 
                  shuffle=True, 
                  random_state=0)
    distortions = []
    for i in range(1, 11):
        km = KMeans(n_clusters=i, 
                    init='k-means++', 
                    n_init=10, 
                    max_iter=300, 
                    random_state=0)
        km.fit(X)
        distortions.append(km.inertia_)
    plt.plot(range(1,11), distortions , marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.tight_layout()
    plt.savefig(PL11 + 'elbow.png', dpi=300)

def silhouette():
    X, y = make_blobs(n_samples=150, 
                  n_features=2, 
                  centers=3, 
                  cluster_std=0.5, 
                  shuffle=True, 
                  random_state=0)
     
    km = KMeans(n_clusters=3,     
                init='k-means++',     
                n_init=10,     
                max_iter=300,    
                tol=1e-04,    
                random_state=0)    
    y_km = km.fit_predict(X)    
        
    cluster_labels = np.unique(y_km)    
    n_clusters = cluster_labels.shape[0]    
    silhouette_vals = silhouette_samples(X, y_km, metric='euclidean')    
    y_ax_lower, y_ax_upper = 0, 0    
    yticks = []    
    for i, c in enumerate(cluster_labels):    
        c_silhouette_vals = silhouette_vals[y_km==c]    
        c_silhouette_vals.sort()    
        y_ax_upper += len(c_silhouette_vals)    
        color = cm.jet(i / n_clusters)    
        plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0,     
                edgecolor='none', color=color)    
        
        yticks.append((y_ax_lower + y_ax_upper) / 2)    
        y_ax_lower += len(c_silhouette_vals)    
            
    silhouette_avg = np.mean(silhouette_vals)    
    plt.axvline(silhouette_avg, color="red", linestyle="--")     
        
    plt.yticks(yticks, cluster_labels + 1)    
    plt.ylabel('Cluster')    
    plt.xlabel('Silhouette coefficient')    
    plt.tight_layout()    
    plt.savefig(PL11 + 'silhouette.png', dpi=300)    
    plt.close()
    
    km = KMeans(n_clusters=2, 
                init='k-means++', 
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
    plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], s=250, marker='*', c='red', label='centroids')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(PL11 + 'centroids_bad.png', dpi=300)
    plt.close()
    
    cluster_labels = np.unique(y_km)
    n_clusters = cluster_labels.shape[0]
    silhouette_vals = silhouette_samples(X, y_km, metric='euclidean')
    y_ax_lower, y_ax_upper = 0, 0
    yticks = []
    for i, c in enumerate(cluster_labels):
        c_silhouette_vals = silhouette_vals[y_km==c]
        c_silhouette_vals.sort()
        y_ax_upper += len(c_silhouette_vals)
        color = cm.jet(i / n_clusters)
        plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0, 
                edgecolor='none', color=color)
    
        yticks.append((y_ax_lower + y_ax_upper) / 2)
        y_ax_lower += len(c_silhouette_vals)
        
    silhouette_avg = np.mean(silhouette_vals)
    plt.axvline(silhouette_avg, color="red", linestyle="--") 
    
    plt.yticks(yticks, cluster_labels + 1)
    plt.ylabel('Cluster')
    plt.xlabel('Silhouette coefficient')
    plt.tight_layout()
    plt.savefig(PL11 + 'silhouette_bad.png', dpi=300)



if __name__ == "__main__":
    # import pdb; pdb.set_trace()
    # k_means()
    # elbow()
    silhouette()