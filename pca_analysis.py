import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from skimage import io, transform, util
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn import mixture
import hdbscan
import time
import fastcluster
from scipy.cluster.hierarchy import fcluster
import umap

def read_image(path):
    try:
        img = io.imread(path)
        return (transform.rescale(util.img_as_float32(img), 0.02, multichannel=True).reshape(-1), path) ## (24, 38, 3)
    except:
        return (None, path)

listing = np.load(
    r'Z:\ftp\sprayers\IntelligentSprayTechnology\Connor_Field_Data\2019_ImageLibrary_FieldLogs\imgs_array_0p1.npy',
    allow_pickle=True)
none_inds = [i for i, x in enumerate(listing) if x[0] is None]
for n in none_inds:
    try:
        listing[n] = read_image(listing[n][1])
    except:
        pass
none_inds = [i for i, x in enumerate(listing) if x[0] is not None]
listing = listing[none_inds, :]

small_inds = [i for i,x in enumerate(listing) if x[0].shape[0] == 2736]

imgs = np.vstack(listing[small_inds,0])
imgs_paths = np.vstack(listing[small_inds,1]).squeeze()
## remove None values......

## eliminate tunnel/circle images, wallpaper images, and high/low extreme stdev images.  Then re-run PCA and
## find outliers and eliminate bad data from those (at least some tunnel images will be left that aren't centered)
pca = PCA(n_components=10)
# pca.fit(imgs[inds[:50000],:])
pca.fit(imgs)
X = pca.transform(imgs)

# stdevs = np.std(imgs, axis=1)
# temp = np.argwhere(np.logical_and(stdevs < .14, stdevs > 0.015)).squeeze() ## < 0.13 avoids tunnel images
#
#

################### dbscan #####################
# db = DBSCAN(eps=1, min_samples=10).fit(X[:,:2])
# core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
# core_samples_mask[db.core_sample_indices_] = True
# labels = db.labels_
#
# # Number of clusters in labels, ignoring noise if present.
# n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
# n_noise_ = list(labels).count(-1)
#
# # Black removed and is used for noise instead.
# unique_labels = set(labels)
# colors = [plt.cm.Spectral(each)
#           for each in np.linspace(0, 1, len(unique_labels))]
# for k, col in zip(unique_labels, colors):
#     if k == -1:
#         # Black used for noise.
#         col = [0, 0, 0, 1]
#
#     class_member_mask = (labels == k)
#
#     xy = X[class_member_mask & core_samples_mask]
#     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
#              markeredgecolor='k', markersize=14)
#
#     xy = X[class_member_mask & ~core_samples_mask]
#     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
#              markeredgecolor='k', markersize=6)
#
# plt.title('Estimated number of clusters: %d' % n_clusters_)
# plt.show()

plot_kwds = {'alpha' : 0.25, 's' : 80, 'linewidths':0}
def plot_clusters(data, algorithm):
    start_time = time.time()
    labels = algorithm.labels_
    end_time = time.time()
    palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    plt.scatter(data.T[0], data.T[1], c=colors, **plot_kwds)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    # plt.title('Clusters found by {}'.format(str(algorithm.__name__)), fontsize=24)
    # plt.text(-0.5, 0.7, 'Clustering took {:.2f} s'.format(end_time - start_time), fontsize=14)

# clusterer = hdbscan.HDBSCAN(min_cluster_size=1000, min_samples=100).fit(X)
# plot_clusters(X[:,:5], clusterer)

link_mat = fastcluster.linkage_vector(X, method='ward') ## fc_cluster
labels = fcluster(link_mat, 8, criterion='maxclust')
# plt.clf();plt.scatter(X.T[0], X.T[1], c=(labels==6).astype(np.int))

X = X[labels!=6, :]

## remove shots of sky for light normalization....
imgs = imgs[labels!=6]
imgs_paths = imgs_paths[labels!=6]

## generate additional features
stdevs = [np.std(t.reshape((24, 38, 3)).reshape(-1,3), axis=0) for t in imgs]
avg_vals = np.array([np.mean(t.reshape((24, 38, 3)).reshape(-1,3), axis=0) for t in imgs])

################ gmm ################
gmm = mixture.GaussianMixture(n_components=8, covariance_type='full', verbose=False, max_iter=500, n_init=3, init_params='random')
gmm.fit(np.hstack((X, stdevs, avg_vals)))
X_label = gmm.predict(np.hstack((X, stdevs, avg_vals)))
# plt.clf();plt.scatter(X.T[0], X.T[4], c=(X_label==0).astype(np.int))
# plt.clf();plt.scatter(X.T[0], X.T[7], c=(X_label==0).astype(np.int))

## remove wallpaper images....and shots of sky
# imgs = imgs[np.logical_and(X_label!=3, labels!=5)]
# imgs_paths = imgs_paths[np.logical_and(X_label!=3, labels!=5)]
imgs = imgs[X_label!=0]
imgs_paths = imgs_paths[X_label!=0]

#### rinse and repeat
pca = PCA(n_components=10)
# pca.fit(imgs[inds[:50000],:])
pca.fit(imgs)
X = pca.transform(imgs)
stdevs = np.array([np.std(t.reshape((24, 38, 3)).reshape(-1,3), axis=0) for t in imgs])
avg_vals = np.array([np.mean(t.reshape((24, 38, 3)).reshape(-1,3), axis=0) for t in imgs])

## remove low stdev
min_stdevs = np.min(stdevs, axis=1)
imgs = imgs[min_stdevs > 0.007]
imgs_paths = imgs_paths[min_stdevs > 0.007]
X = X[min_stdevs > 0.007]
stdevs = np.array([np.std(t.reshape((24, 38, 3)).reshape(-1,3), axis=0) for t in imgs])
avg_vals = np.array([np.mean(t.reshape((24, 38, 3)).reshape(-1,3), axis=0) for t in imgs])

## remove stitched images
gmm = mixture.GaussianMixture(n_components=8, covariance_type='full', verbose=False, max_iter=500, n_init=3, init_params='kmeans')
gmm.fit(stdevs)
X_label_stdvs = gmm.predict(stdevs)
# plt.clf();plt.scatter(stdevs.T[1], stdevs.T[2], c=(X_label_stdvs==0).astype(np.int))

imgs = imgs[gmm.score_samples(stdevs)>2]
imgs_paths = imgs_paths[gmm.score_samples(stdevs)>2]

# nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(np.hstack((X, avg_vals, stdevs)))
# _, nearest = nbrs.kneighbors(np.hstack((X, avg_vals, stdevs))[218095].reshape(1,-1), 10)

# cols = ['p' + str(i) for i in range(X.shape[1])]
# cols.append('xlabel')
# df_gmm = pd.DataFrame(np.hstack((X,X_label.reshape(-1,1))), columns=cols)
# sns.pairplot(vars=cols[:2], data=df_gmm, hue=cols[-1], size=5)
##
# scores = pca.score_samples(imgs)
# plt.scatter(X.T[0], X.T[4], c=(X_label==1).astype(np.int), s=1)

# t0 = time.clock()
# clusterable_embedding = umap.UMAP(
#     n_neighbors=30,
#     min_dist=0.01,
#     n_components=3,
#     random_state=42,
# ).fit_transform(X)
# t1 = time.clock()
#
# labels = hdbscan.HDBSCAN(
#     min_samples=10,
#     min_cluster_size=500,
# ).fit_predict(clusterable_embedding)
#
# plt.scatter(clusterable_embedding[:, 0], clusterable_embedding[:, 1],
#             c=labels, s=0.1, cmap='Spectral');

## camera black/white tunnel indices [i for i,x in enumerate(imgs_paths) if '-1_3_3149924166.7193_3149924166.7193_614_3__41842' in x]

    # np.savetxt(r'Z:\ftp\sprayers\IntelligentSprayTechnology\Connor_Field_Data\2019_ImageLibrary_FieldLogs\img_pca.csv', pca.transform(listing), delimiter=',')
#
#
#
# # '''
# # load images, resize/rescale, standardize and then do PCA and identify outlier images like r'Z:\ftp\sprayers\IntelligentSprayTechnology\Connor_Field_Data\2019_
# # ImageLibrary_FieldLogs\2019-06-04_AR_Soybeans\Extracted Images\PCPM2HA000168.04062019-18-15-00\-1_3_3149923268.2782_3149923268.2782_10_3__20937.bmp'
# # which are not images of crop and eliminate from consideration to focus on extracting crop features
# # '''