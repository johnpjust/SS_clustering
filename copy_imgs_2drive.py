import os
import numpy as np
import re
from pathlib import Path
from skimage import io, transform, util
from multiprocessing import Pool
import tqdm

datapath = r'Z:\ftp\sprayers\IntelligentSprayTechnology\Connor_Field_Data\2019_ImageLibrary_FieldLogs'
target_dir = r'J:\SaS'
def copy_image(path):
    Path(os.path.join(target_dir, os.path.split(path)[0])).mkdir(parents=True, exist_ok=True)
    img = io.imread(os.path.join(datapath, path))
    io.imsave(os.path.join(target_dir, os.path.splitext(path)[0] + '.png'), util.img_as_ubyte(transform.rescale(util.img_as_float32(img), 0.1, multichannel=True)))

if __name__ == '__main__':

    # crop_pat = re.compile('(?<=_)(?:.(?!_))+$')

    imgs_paths = np.load(os.path.join(datapath, 'filtered_img_paths.npy'))
    # class_names = np.array([crop_pat.search(x.split(os.sep)[-4])[0] for x in imgs_paths])
    # imgs_paths = np.array([os.path.join(datapath, filepath) for filepath in imgs_paths])

    # for crop in np.unique(class_names):
    #     Path(os.path.join(target_dir, crop)).mkdir(parents=True, exist_ok=True)

    with Pool(24) as p:
        list(tqdm.tqdm(p.imap_unordered(copy_image, imgs_paths), total=len(imgs_paths)))