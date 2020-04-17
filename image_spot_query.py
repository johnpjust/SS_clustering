import matplotlib.pyplot as plt
from skimage import io, transform, util
import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import tensorflow as tf

fn_time_crop_list = np.load(r'J:\SaS\fn_time_crop.npy')
full_shape = io.imread(os.path.join(r'Z:\ftp\sprayers\IntelligentSprayTechnology\Connor_Field_Data\2019_ImageLibrary_FieldLogs_Motec', os.path.splitext(fn_time_crop_list[0][0])[0] + '.bmp')).shape
img_array = np.load(r'J:\models\Tensorboard\SaS_2020-04-08-12-55-54_sup+L2\embeds_32.npy')
IC_img_arr_mean = np.load(r'J:\models\Tensorboard\SaS_2020-04-05-23-59-15_3Loss\embeds_32.npy')
alg_img_shape = img_array[0].shape
red_factor = np.array(alg_img_shape[:2])/np.array(full_shape[:2])
model = tf.keras.models.load_model(r'J:\models\Tensorboard\SaS_2020-04-08-12-55-54_sup+L2\model_simclr')

nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(img_array.reshape(-1,32))

fig = plt.figure()
coords = []
def onclick(event):
    # global ix, iy
    global coords
    ix, iy = event.xdata, event.ydata
    coords = [int(iy), int(ix)]
    # c = (np.round(coords * red_factor, 0)).astype(np.int)
    # cc = [np.min((c[0], 8)), np.min((c[1], 15))]
    # coords = cc
    # print('y = %d, x = %d' % (cc[0], cc[1]))
    print('y = %d, x = %d' % (coords[0], coords[1]))
    return

cid = fig.canvas.mpl_connect('button_press_event', onclick)

def plt_fnc(indx=77308, resize=False):
    plt.clf()
    img = io.imread(os.path.join(r'Z:\ftp\sprayers\IntelligentSprayTechnology\Connor_Field_Data\2019_ImageLibrary_FieldLogs_Motec', os.path.splitext(fn_time_crop_list[indx][0])[0] + '.bmp'))
    if resize:
        img = util.img_as_ubyte(transform.rescale(util.img_as_float32(img), 0.1, multichannel=True))
    plt.imshow(img)
    return img


# fig.canvas.mpl_disconnect(cid)

69925
index = 77308
index = 71260
index = 71267
index = 78221
index = 77547
plt.clf()
plt_fnc(index)
_, nn_inds = nbrs.kneighbors(img_array[index, coords[0], coords[1], :].reshape(1, -1))
print(np.floor(nn_inds/np.prod(alg_img_shape[:2])))
temp=np.mod(nn_inds,np.prod(alg_img_shape[:2]))
print(np.unravel_index(temp, (9,16)))

img = io.imread(
    os.path.join(r'Z:\ftp\sprayers\IntelligentSprayTechnology\Connor_Field_Data\2019_ImageLibrary_FieldLogs_Motec',
                 os.path.splitext(fn_time_crop_list[69061][0])[0] + '.bmp'))
plt.imsave(r'C:\Users\justjo\Desktop\cotton.png', img)




# hm=tf.image.resize(IC_img_arr_mean[index], img.shape[:2])[:,:,IC_index];plt.imshow(hm, cmap='hot', alpha=0.3)
index = 76598
plt.clf()
plt_fnc(index)
hm=tf.image.resize(IC_img_arr_mean[index], img.shape[:2])[:,:,1];plt.imshow(hm, cmap='hot', alpha=0.3)

index = 77308
crop_size = [40,40,3]
plt.clf()
img = plt_fnc(index, resize=True)
img_crop = tf.image.crop_to_bounding_box(tf.image.convert_image_dtype(img, tf.float32), tf.clip_by_value(coords[0]-crop_size[0]//2,0,img.shape[0]-1-crop_size[0]//2),
                                         tf.clip_by_value(coords[1]-crop_size[0]//2,0,img.shape[1]-1-crop_size[0]//2), crop_size[0], crop_size[1])
_, nn_inds = nbrs.kneighbors(model(tf.expand_dims(img_crop, axis=0)).numpy().reshape(1,-1))
print(np.floor(nn_inds/np.prod(alg_img_shape[:2])))
temp=np.mod(nn_inds,np.prod(alg_img_shape[:2]))
print(np.unravel_index(temp, (9,16)))