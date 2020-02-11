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

def read_image(path):
    try:
        img = io.imread(path)
        return (transform.rescale(util.img_as_float32(img), 0.02, multichannel=True).reshape(-1), path)
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





stdevs = np.std(imgs, axis=1)
temp = np.argwhere(np.logical_and(stdevs < .14, stdevs > 0.015)).squeeze() ## < 0.13 avoids tunnel images

pca = PCA(n_components=10)
# pca.fit(imgs[inds[:50000],:])
pca.fit(imgs)
X = pca.transform(imgs)
nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(X)
nearest = nbrs.kneighbors(X[temp[-10]].reshape(1,-1), 10)

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

################ gmm ################
gmm = mixture.GaussianMixture(n_components=2, covariance_type='tied', verbose=False, max_iter=100, n_init=3, init_params='kmeans')
gmm.fit(X[:,4:5])
X_label = gmm.predict(X[:,4:5])
cols = ['p' + str(i) for i in range(X.shape[1])]
cols.append('xlabel')
df_gmm = pd.DataFrame(np.hstack((X,X_label.reshape(-1,1))), columns=cols)
sns.pairplot(vars=cols[:2], data=df_gmm, hue=cols[-1], size=5)
##
scores = pca.score_samples(imgs)

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