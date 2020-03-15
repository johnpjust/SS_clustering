import os
import json
import datetime
import tensorflow as tf
from tensorflow.keras import layers, Input, Model

import tensorflow_probability as tfp
import numpy as np
from earlystopping import *
import random
import pathlib
import functools
import re
from pathlib import Path
import ntpath
import resnet_models
from multiprocessing import Pool
import tqdm
from skimage import transform
from sklearn.decomposition import PCA

from tensorflow.keras import backend as K
def swish_activation(x):
        return (K.sigmoid(x) * x)
tf.keras.utils.get_custom_objects().update({'swish': tf.keras.layers.Activation(swish_activation)})

crop_pat = re.compile('(?<=_)(?:.(?!_))+$')
gpstime_pat = re.compile('\d+\.\d+')

class parser_:
    pass

args = parser_()
args.device = '/cpu:0'  # '/gpu:0'
args.clip_norm = 0.1
args.epochs = 5000
args.patience = 10
args.load = r''
args.save = True
args.tensorboard = r'C:\Users\justjo\PycharmProjects\SaS_clustering\tensorboard'
args.early_stopping = 50
args.manualSeed = None
args.manualSeedw = None
args.prefetch_size = 10  # data pipeline prefetch buffer size
args.parallel = 8  # data pipeline parallel processes
args.preserve_aspect_ratio = True;  ##when resizing
args.p_val = 0.2
args.downscale = 10
args.take = 1000
args.batch_dim = 100
args.crop_size = [40, 40, 3]
args.spacing = 10
args.path = os.path.join(args.tensorboard, 'SaS_{}'.format(str(datetime.datetime.now())[:-7].replace(' ', '-').replace(':', '-')))

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


model = tf.keras.models.load_model(r'D:\pycharm_projects\SaS\tensorboard\SaS_2020-03-12-01-09-00\_model_simclr')
# embeds = tf.keras.Model(model.input, model.layers[-3].output, name='embeds') ## might just directly use model(input).layers[-3].output  ???
def pre_process_inference(img_crop):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_png(img_crop, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)

    rows = img.shape[0] - args.crop_size[0]
    cols = img.shape[1] - args.crop_size[1]
    heatmap = np.zeros((np.int(rows / args.spacing)+1, np.int(cols / args.spacing)+1, model.output_shape[-1]))
    im_breakup_array = np.zeros((np.int(cols / args.spacing)+1, *args.crop_size), dtype=np.float32)
    with tf.device(args.device):
        for i in range(0, rows+1, args.spacing):
            for j in range(0, cols+1, args.spacing):
                im_breakup_array[np.int(j / args.spacing), :] = tf.image.crop_to_bounding_box(img, i, j, args.crop_size[0], args.crop_size[1])
            heatmap[np.int(i / args.spacing), :] = model(im_breakup_array, training=False).numpy()

    return heatmap

def rescaler(img):
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return transform.rescale(img, 0.3, multichannel=True).reshape(-1)

if __name__ == "__main__":

    ## deep learning inference
    imgs_raw = np.load(r'D:\pycharm_projects\SaS\imgs_raw_coded_png_bytes.npy')
    # r = list(tqdm.tqdm(map(pre_process_inference, imgs_raw), total=imgs_raw.shape[0]))
    with Pool(8) as p:
        r = list(tqdm.tqdm(p.imap(pre_process_inference, imgs_raw), total=imgs_raw.shape[0]))

    np.save(r'D:\pycharm_projects\SaS\tensorboard\SaS_2020-03-12-01-09-00\embeds_32', r)

    # ## PCA
    # with Pool(8) as p:
    #     r = np.array(list(tqdm.tqdm(p.imap(rescaler, imgs_raw), total=imgs_raw.shape[0])))
    # pca = PCA(n_components=10)
    # pca.fit(r)
    # X = pca.transform(r)
    # np.savetxt(r'C:\Users\justjo\PycharmProjects\SaS_clustering\tensorboard\SaS_2020-03-04-22-10-42\PCA_embeds.csv', X, delimiter=',')


# from sklearn.neighbors import NearestNeighbors
# import numpy as np
# import matplotlib.pyplot as plt
# import tensorflow as tf
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#
# embeds = np.array(np.load(r'D:\pycharm_projects\SaS\tensorboard\SaS_2020-03-12-01-09-00\embeds_32.npy', allow_pickle=True))
# # embeds_std = np.array([x.reshape(-1,32).std(axis=0) for x in embeds])
# embeds = np.array([x.reshape(-1,32).mean(axis=0) for x in embeds])
# embeds = embeds[:, embeds.std(axis=0)>0]
# # embeds_std = embeds_std[:, embeds_std.std(axis=0)>0]
# imgs_raw = np.load(r'D:\pycharm_projects\SaS\imgs_raw_coded_png_bytes.npy')
# # embeds_stacked = np.hstack((embeds, embeds_std))
# nbrs = NearestNeighbors(n_neighbors=100, algorithm='ball_tree').fit(embeds)
# # nbrs_std = NearestNeighbors(n_neighbors=100, algorithm='ball_tree').fit(embeds_std)
# # nbrs_stacked = NearestNeighbors(n_neighbors=100, algorithm='ball_tree').fit(embeds_stacked)
# _, indices = nbrs.kneighbors(embeds[20000].reshape(1,-1), 1000)
# # _, indices_std = nbrs_std.kneighbors(embeds_std[194485].reshape(1,-1), 1000)
# # _, indices_stacked = nbrs_std.kneighbors(embeds_std[200000].reshape(1,-1), 1000)
#
# fn_time_crop_list = np.load(r'J:\SaS\fn_time_crop.npy')
# crops = np.array([ii[2] for ii in fn_time_crop_list])


