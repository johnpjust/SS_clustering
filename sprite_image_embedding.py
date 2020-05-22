import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from skimage import io
from PIL import Image
from skimage import io

sprite_size = 8192
def create_sprite(img_data, n_h, n_w):
    """
    Tile images into sprite image.
    Add any necessary padding
    """

    n = int(np.ceil(np.sqrt(img_data.shape[0])))
    padding = ((0, n_h*n_w - img_data.shape[0]), (0, 0), (0, 0), (0, 0))
    data = np.pad(img_data, padding, mode='constant', constant_values=0)

    # Tile images into sprite
    data = data.reshape((n_h, n_w) + data.shape[1:]).transpose((0, 2, 1, 3, 4))
    # print(data.shape) => (n, image_height, n, image_width, 3)

    data = data.reshape((n_h * data.shape[1], n_w * data.shape[3]) + data.shape[4:])
    # print(data.shape) => (n * image_height, n * image_width, 3)
    return data


embeds = np.genfromtxt(r'D:\Just\pyprojects\SAS_disentangle\tensorboard\SaS_2020-05-16-13-42-18\full_image_embeds.csv', delimiter=',')

imgs_raw = np.load(
        r'T:\ftp\sprayers\IntelligentSprayTechnology\Connor_Field_Data\2020_ImageLibrary_FieldLogs_Jabil\Extracted_Logs\imgs_compressed.npy',
        allow_pickle=True)
fn_time_crop_list = pd.read_pickle(r'T:\ftp\sprayers\IntelligentSprayTechnology\Connor_Field_Data\2020_ImageLibrary_FieldLogs_Jabil\Extracted_Logs\pandas_path_time_crop_array.pickle')
# crops = np.array([ii[2] for ii in fn_time_crop_list])
crops = fn_time_crop_list.crop.values
crops[crops == 'FallowSoybeansStubble'] = 'FallowSoybeanStubble'
imgs_raw = imgs_raw[crops != 'Issues']
crops = crops[crops != 'Issues']

fn_time_crop_list = []
CLASS_NAMES, class_counts = np.unique(crops, return_counts=True)

img_num = len(imgs_raw)
## if len(imgs_raw) is large then will need to subsample by class
# inds = np.random.permutation(len(imgs_raw))[:img_num]
# imgs_raw = imgs_raw[inds]
# embeds = embeds[inds]
# crops = crops[inds]
h, w, d = tf.image.decode_png(imgs_raw[0]).shape
c = w/h

image_height = int(sprite_size / np.sqrt(img_num*c))
image_width = int(c*image_height)        # tensorboard supports sprite images up to 8192 x 8192
n_w = int(sprite_size/image_width)
n_h = int(sprite_size/image_height)


img_data = []
for i in range(img_num):
    # row = i // grid  # added integer divide
    # col = i % grid
    img = tf.image.decode_png(imgs_raw[i])
    img = tf.image.resize(img, (image_height, image_width), preserve_aspect_ratio=True, antialias=False)
    # row_loc = row * image_height
    # col_loc = col * image_width
    img_data.append(img.numpy())
    # big_image.paste(img, (col_loc, row_loc)) # NOTE: the order is reverse due to PIL saving
    # print(row_loc, col_loc)
img_data = np.array(img_data)

sprite = create_sprite(img_data, n_h, n_w)
# save image
np.savetxt(r'D:\Just\pyprojects\SAS_disentangle\tensorboard\SaS_2020-05-16-13-42-18\embedding_projector\metadata.tsv', crops, fmt='%s', delimiter='\t')
io.imsave(r'D:\Just\pyprojects\SAS_disentangle\tensorboard\SaS_2020-05-16-13-42-18\embedding_projector\sprite.jpeg', sprite, quality=100) ## quality = [1 100], with 100 being best and 1 being worst