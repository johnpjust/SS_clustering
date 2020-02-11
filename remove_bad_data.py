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
        return (transform.rescale(util.img_as_float32(img), 0.02, multichannel=True).reshape(-1), path)
    except:
        return (None, path)
    # return transform.rescale(util.img_as_float32(img), 0.02, multichannel=True)

if __name__ == '__main__':
    df = pd.read_pickle(r'Z:\ftp\sprayers\IntelligentSprayTechnology\Connor_Field_Data\2019_ImageLibrary_FieldLogs\pandas_path_time_crop_array.pickle')

    # listing = np.array([x for x in list(tqdm.tqdm(map(read_image, df.path[:10].values), total=len(df.path[:10].values))) if x is not None])
    with Pool(24) as p:
        listing = np.array(list(tqdm.tqdm(p.imap(read_image, df.path.values), total=len(df.path.values))))

    np.save(r'Z:\ftp\sprayers\IntelligentSprayTechnology\Connor_Field_Data\2019_ImageLibrary_FieldLogs\imgs_array_0p1.npy', listing)


######### shadow imgs #############
######### Z:\ftp\sprayers\IntelligentSprayTechnology\Connor_Field_Data\2019_ImageLibrary_FieldLogs\2019-05-15_AR_Cotton\Extracted_Images\PCPM2HA000183.15052019-10-00-00