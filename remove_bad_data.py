import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from skimage import io, transform, util
from multiprocessing import Pool
import tqdm

def read_image(path):
    try:
        img = io.imread(path)
        return transform.rescale(util.img_as_float32(img), 0.02, multichannel=True).reshape(-1)
    except:
        return None
    # return transform.rescale(util.img_as_float32(img), 0.02, multichannel=True)

if __name__ == '__main__':
    df = pd.read_pickle(r'Z:\ftp\sprayers\IntelligentSprayTechnology\Connor_Field_Data\2019_ImageLibrary_FieldLogs\pandas_path_time_crop_array.pickle')

    # listing = np.array([x for x in list(tqdm.tqdm(map(read_image, df.path[:10].values), total=len(df.path[:10].values))) if x is not None])
    with Pool(24) as p:
        listing = np.array([x for x in list(tqdm.tqdm(p.imap(read_image, df.path.values), total=len(df.path.values))) if x is not None])

    np.save(r'Z:\ftp\sprayers\IntelligentSprayTechnology\Connor_Field_Data\2019_ImageLibrary_FieldLogs\imgs_array_0p1.npy', listing)
    inds = np.random.permutation(listing.shape[0])
    pca = PCA(n_components=10)
    pca.fit(listing[inds[:50000]])
    np.savetxt(r'Z:\ftp\sprayers\IntelligentSprayTechnology\Connor_Field_Data\2019_ImageLibrary_FieldLogs\img_pca.csv', pca.transform(listing), delimiter=',')



# '''
# load images, resize/rescale, standardize and then do PCA and identify outlier images like r'Z:\ftp\sprayers\IntelligentSprayTechnology\Connor_Field_Data\2019_
# ImageLibrary_FieldLogs\2019-06-04_AR_Soybeans\Extracted Images\PCPM2HA000168.04062019-18-15-00\-1_3_3149923268.2782_3149923268.2782_10_3__20937.bmp'
# which are not images of crop and eliminate from consideration to focus on extracting crop features
# '''
