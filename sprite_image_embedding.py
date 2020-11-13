import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from skimage import io
from PIL import Image
from skimage import io
import os

sprite_size = 8192
def create_sprite(img_data, n_h, n_w):
    """
    Tile images into sprite image.
    Add any necessary padding
    """
    # n_h = n_h - int((n_h * n_w - img_data.shape[0]) / n_w)
    # n = int(np.ceil(np.sqrt(img_data.shape[0])))
    padding = ((0, n_h*n_w - img_data.shape[0]), (0, 0), (0, 0), (0, 0))
    data = np.pad(img_data, padding, mode='constant', constant_values=0)

    # Tile images into sprite
    data = data.reshape((n_h-int((n_h*n_w - img_data.shape[0]) / n_w), n_w) + data.shape[1:]).transpose((0, 2, 1, 3, 4))
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

# n_h = n_h - int((n_h * n_w - img_data.shape[0]) / n_w) ## all images, fill in with blanks
n_h = int(n_h - np.ceil((n_h * n_w - img_data.shape[0]) / n_w)) ## leave out last incomplete row
inds = np.random.permutation(n_h*n_w)
img_data = img_data[inds,:,:,:]
sprite = create_sprite(img_data, n_h, n_w)


# sprite = create_sprite(img_data, n_h, n_w)
# save image
log_dir = r'D:\Just\pyprojects\SAS_disentangle\tensorboard\SaS_2020-05-16-13-42-18\embedding_projector'
np.savetxt(os.path.join(log_dir, 'metadata.tsv'), crops, fmt='%s', delimiter='\t')
io.imsave(os.path.join(log_dir, 'sprite.jpeg'), sprite, quality=100) ## quality = [1 100], with 100 being best and 1 being worst



# from tensorboard.plugins import projector
# # Create a checkpoint from embedding, the filename and key are
# # name of the tensor.
# embeddings = tf.Variable(embeds, name='test_embedding')
# checkpoint = tf.train.Checkpoint(embedding=embeddings)
# checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))
#
# # Set up config
# config = projector.ProjectorConfig()
# embedding = config.embeddings.add()
# # The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`
# embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
# embedding.metadata_path = 'metadata.tsv'
# embedding.sprite.image_path = os.path.join(log_dir, 'sprite.jpeg')
# embedding.sprite.single_image_dim.extend([image_width, image_height])
# projector.visualize_embeddings(log_dir, config)



from lapjv import lapjv
from scipy.spatial.distance import cdist
import umap
from tensorflow.python.keras.preprocessing import image

def generate_umap(activations):
    umap_ = umap.UMAP(n_neighbors=200, n_components=2, min_dist=0.5)
    X_2d = umap_.fit_transform(activations)
    X_2d -= X_2d.min(axis=0)
    X_2d /= X_2d.max(axis=0)
    return X_2d

def save_umap_grid(img_collection, X_2d, out_res=[image_height, image_width], out_dim=[n_h, n_w]):
    grid = np.dstack(np.meshgrid(np.linspace(0, 1, out_dim[0]), np.linspace(0, 1, out_dim[1]))).reshape(-1, 2)
    cost_matrix = cdist(grid, X_2d, "sqeuclidean").astype(np.float32)
    cost_matrix = cost_matrix * (100000 / cost_matrix.max())
    row_asses, col_asses, _ = lapjv(cost_matrix)
    grid_jv = grid[col_asses]
    out = np.ones((out_dim[0]*out_res[0], out_dim[1]*out_res[1], 3), dtype=np.uint8)

    for pos, img in zip(grid_jv, img_collection):
        h_range = int(np.floor(pos[0]* (out_dim[0] - 1) * out_res[0]))
        w_range = int(np.floor(pos[1]* (out_dim[1] - 1) * out_res[1]))
        out[h_range:h_range + out_res[0], w_range:w_range + out_res[1]] = image.img_to_array(img)

    im = image.array_to_img(out)
    im.save(os.path.join(log_dir, 'sprite_arranged.png'), quality=100)

X_2d = generate_umap(embeds[inds,:])
save_umap_grid(img_data, X_2d)