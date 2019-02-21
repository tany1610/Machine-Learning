# Hierarchical Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Loading the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

# Creating a dendrogram to find optimal number of clusters
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Cusomers')
plt.ylabel('Euclidean Distance')
plt.show()

# Fit the clusters to the dataset
from sklearn.cluster import AgglomerativeClustering 
hierarchical = AgglomerativeClustering(n_clusters=5, linkage='ward')
y = hierarchical.fit_predict(X)
colors = ['red', 'blue', 'green', 'cyan', 'magenta']

# Visualizing the data
for i in range(0, 5):
    plt.scatter(X[y == i, 0], X[y == i, 1], label="Cluster {}".format(i), c=colors[i], s=100)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()