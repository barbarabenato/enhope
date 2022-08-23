from sklearn.datasets import make_classification
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

import numpy as np
import enhope
import torch

import matplotlib.pyplot as plt
np.random.seed(42)


def plot(X, y, filename, colormap=plt.cm.Paired):
    plt.figure(figsize=(8, 6))

    # clean the figure
    plt.clf()
    # plot X if dim is equal to 2
    if X.shape[1] == 2:
        X_embedded = X
    else: # compute tSNE otherwise and then plot
        tsne = TSNE()
        X_embedded = tsne.fit_transform(X)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap=colormap)

    plt.xticks(())
    plt.yticks(())

    plt.savefig(filename)


# generate random data
X, y = make_classification(n_samples=300, n_classes=3, n_clusters_per_class=1, n_informative=4, class_sep=4., n_features=5, n_redundant=0, shuffle=True, scale=[1, 1, 20, 20, 20])
plot(X, y, 'original.png')

# define en-HOPE parameters
F = 800
m = 400
epochs = 1000
input_dim = X.shape[1]
output_dim = 2
exemplars_per_class = 2
n_classes = np.unique(y).shape[0]

y = y.reshape(y.shape[0],)
Xnp = X
ynp = y

# run supervised kmeans to choose exemplars
id_e = []
for i in range(n_classes):
    km = KMeans(n_clusters=exemplars_per_class).fit(X[y==i])
    ind, _ = pairwise_distances_argmin_min(km.cluster_centers_, X)
    id_e.append(ind)

id_e = np.array(id_e)
id_e = id_e.reshape(id_e.shape[0]*id_e.shape[1],)
y = y.reshape(y.shape[0],1)

# prepare data
X, y = enhope.convert_to_torch_data(X, y)

# run en-HOPE
model = enhope.enHOPE(input_dim, m, output_dim, F)
# define optimizer
model.optim = torch.optim.SGD(model.parameters(), lr=1)
# train en-HOPE
model.train(X, y, epochs, id_e)

# obtaining features for data
output = model.predict(X)

# ploting data
plot(output, ynp, 'reduced.png')
