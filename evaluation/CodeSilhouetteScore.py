from yellowbrick.cluster import SilhouetteVisualizer

from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
import DataLoader
import torch
import Utils
from PIL import Image

from Model import create_model
from Utils import load_model
from main import get_config
from torch import FloatTensor

#
# Load IRIS dataset
#
iris = datasets.load_iris()
X = iris.data
y = iris.target
#
# Instantiate the KMeans models
#

print(X.shape)

# fig, ax = plt.subplots()
# i = 3
# km = KMeans(n_clusters=i, not_grey_init='k-means++', n_init=10, max_iter=100, random_state=42)
# q, mod = divmod(i, 2)
# visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax)
# visualizer.fit(X)


# plt.show()
n_clusters = 25


def get_codes():
    codes = []
    labels = []
    for c in range(2, 2 + n_clusters):
        c_codes = np.load(f"code_data/code_{c:02}.data.npy")
        labels.extend([c - 2] * len(c_codes))
        codes.extend(c_codes)
    return codes, labels


codes, labels = get_codes()
score = silhouette_score(codes, labels, metric='euclidean')
print('Silhouetter Score: %.3f' % score)

km = KMeans(n_clusters=n_clusters, random_state=42)
km.fit_predict(codes)
score = silhouette_score(codes, km.labels_, metric='euclidean')
print('Silhouetter Score: %.3f' % score)


class MyKMeans(KMeans):
    def __init__(self, labels, n_clusters):
        self.labels = np.asarray(labels)
        self.n_clusters = n_clusters

    def predict(self, X):
        return self.labels


codes = np.asarray(codes)
print(codes.shape)
fig, ax = plt.subplots()
visualizer = SilhouetteVisualizer(MyKMeans(labels, n_clusters), colors='yellowbrick', ax=ax)
visualizer.fit(codes)
plt.show()
