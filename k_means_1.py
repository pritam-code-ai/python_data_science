





import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import classification_report, accuracy_score
from sklearn.datasets import load_digits



### first section ic sklearn makeblobs dataset 3d graph
### and
### second part is sklearn dataset on digit recognition with variouc metrics
### and
### third part is k_means classification by 3 and 8 clusters on a numpy array


#
#
#
#
# FIRST PART IS 3D GRAPH OF make_blobs DATASET
#
#
#
#
#

plt.rcParams['figure.figsize'] = (16, 9)

# Creating a sample dataset with 4 clusters
X, y = make_blobs(n_samples=800, n_features=3, centers=4)

# Initializing KMeans
kmeans = KMeans(n_clusters=4)
# Fitting with inputs
kmeans = kmeans.fit(X)
# Predicting the clusters
labels = kmeans.predict(X)
# Getting the cluster centers
C = kmeans.cluster_centers_

fig = plt.figure()
ax = Axes3D(fig)

ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y)
ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c='#050505', s=1000)
plt.show()



#
#
#
#
# SECOND PART IS digit recognition DATASET WITH metrics
#
#
#
#
#
#
#


digits = load_digits()
digits.data.shape

kmeans = KMeans(n_clusters=10, random_state=0)
clusters = kmeans.fit_predict(digits.data)
kmeans.cluster_centers_.shape

fig, ax = plt.subplots(2, 5, figsize=(8, 3))
centers = kmeans.cluster_centers_.reshape(10, 8, 8)
for axi, center in zip(ax.flat, centers):
    axi.set(xticks=[], yticks=[])
    axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)

from scipy.stats import mode
labels = np.zeros_like(clusters)
for i in range(10):
        mask = (clusters == i)
        labels[mask] = mode(digits.target[mask])[0]


n_errors = (labels != (digits.target)).sum()
# Run classification metrics
print('{}: {}'.format("k_means", n_errors))
print(accuracy_score(digits.target, labels))
print(classification_report(digits.target, labels))










#
#
#
#
# THIRD PART IS k_means classification by 3 and 8 clusters on a numpy array
#
#
#
#
#
#
#


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
import seaborn as sns; sns.set()
import pandas as pd





X_feature =np.array([
            [1, 2, 0],
            [0, 4, 3],
            [1, 2, 6],
            [6, 2, 1],
            [0, 6, 3],
            [1, 0, 2],
            [1, 4, 6],
            [1, 10, 3],
            [5, 0, 5],
            [6, 0, 3],
            [2, 3, 1]
            ])



estimators = [
             ('k_means_8_clusters_apply_on_array', KMeans(n_clusters=8)),
             ('k_means_3_clusters_apply_on_array', KMeans(n_clusters=3))
             ]

fignum = 1
titles = ['k_means by 8 clusters', 'k_means by 3 clusters']
for name, est in estimators:
    fig = plt.figure(fignum, figsize=(4, 3))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    est.fit(X_feature)
    labels = est.labels_

    ax.scatter(X_feature[:, 0], X_feature[:, 1], X_feature[:, 2], c=labels.astype(np.float), s = 90)

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('x axis labels')
    ax.set_ylabel('y axis labels')
    ax.set_zlabel('z axis labels')
    ax.set_title(titles[fignum - 1])
    ax.dist = 12
    fignum = fignum + 1

plt.show()




#
#
#
#
#
#
#
#
# convert a numpy array to a dataframe and drop a column in dataframe
# and k_means classify this dataframe to form a 2 dimensional graph




import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd





X_feature =np.array([
            [1, 2, 0],
            [0, 4, 3],
            [1, 2, 6],
            [6, 2, 1],
            [0, 6, 3],
            [1, 0, 2],
            [1, 4, 6],
            [1, 10, 3],
            [5, 0, 5],
            [6, 0, 3],
            [2, 3, 1]
            ])

kmeans = KMeans(n_clusters=2)
X_feature = X_feature.reshape((-1,3))

df = pd.DataFrame({1:X_feature[:,0], 2:X_feature[:,1],3:X_feature[:,2]})
gh = df.drop([3], axis=1)
kmeans.fit(gh)
y_pred = kmeans.predict(gh)


plt.scatter(gh.values[:, 0], gh.values[:, 1], c=y_pred, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.xlabel('x axis labels')
plt.ylabel('y axis labels')
plt.title("by values column use")
plt.show()

plt.scatter(gh.iloc[:, 0], gh.iloc[:, 1], c=y_pred, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.xlabel('x axis labels')
plt.ylabel('y axis labels')
plt.title("by iloc column use")
plt.show()









