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
import more_itertools as mit
import efficientnet.tfkeras as efn


from tensorflow.keras import backend as K
def swish_activation(x):
        return (K.sigmoid(x) * x)
tf.keras.utils.get_custom_objects().update({'swish': tf.keras.layers.Activation(swish_activation)})

crop_pat = re.compile('(?<=_)(?:.(?!_))+$')
gpstime_pat = re.compile('\d+\.\d+')

class parser_:
    pass

args = parser_()
args.device = '/gpu:0'  # '/gpu:0'
args.clip_norm = 0.1
args.epochs = 5000
args.patience = 10
args.load = r''
args.save = True
args.tensorboard = r'D:\pycharm_projects\SaS\tensorboard'
args.early_stopping = 50
args.manualSeed = None
args.manualSeedw = None
args.prefetch_size = 10  # data pipeline prefetch buffer size
args.parallel = 8  # data pipeline parallel processes
args.preserve_aspect_ratio = True;  ##when resizing
args.p_val = 0.2
args.take = 256
args.batch_dim = 64
# args.crop_size = [40, 40, 3]
args.spacing = 10
args.wind_size = 11 ## must be an odd number
args.wind_time_max = 6
args.wind_size_half = args.wind_size//2

args.path = os.path.join(args.tensorboard, 'SaS_{}'.format(str(datetime.datetime.now())[:-7].replace(' ', '-').replace(':', '-')))


imgs_raw = np.load(r'D:\pycharm_projects\SaS\imgs_raw_coded_png_bytes.npy')
fn_time_crop_list = np.load(r'D:\pycharm_projects\SaS\fn_time_crop.npy')
crops = np.array([ii[2] for ii in fn_time_crop_list])
times = np.array([float(ii[1]) for ii in fn_time_crop_list])
folders = np.array([ii[0].split(os.sep)[0] for ii in fn_time_crop_list])
fn_time_crop_list = []
args.CLASS_NAMES, args.class_counts = np.unique(crops, return_counts=True)
# args.class_weights = np.max(args.class_counts)/args.class_counts
# args.class_weights = args.class_weights/np.sum(args.class_weights)


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

@tf.function
def zoom(x):
    # bbox = tf.stack([*args.crop_size[:2]*tf.random.uniform(shape=[], minval=0.8, maxval=1), 3])
    rval = tf.random.uniform(shape=[], minval=0.95, maxval=1.05)
    bbox = [args.crop_size[0]*rval, args.crop_size[1]*rval, 3]
    y = tf.image.random_crop(x, size=bbox)
    return tf.image.resize(y, size=args.crop_size[:2])

args.aug_prob = [0]
args.aug_prob.extend([0.85]*5)
augmentations = [lambda x: tf.image.random_brightness(x, 0.2),
                 tf.image.random_flip_left_right,
                 tf.image.random_flip_up_down,
                 lambda x: tf.image.random_hue(x, 0.1),
                 lambda x: tf.image.random_saturation(x, 0.5, 1.5),
                 lambda x: tf.image.random_contrast(x, 0.7, 1.3)]

# @tf.function
# def pre_process_aug(img_crop):
#     # convert the compressed string to a 3D uint8 tensor
#     img = tf.image.decode_png(img_crop[0], channels=3)
#     # Use `convert_image_dtype` to convert to floats in the [0,1] range.
#     img = tf.image.convert_image_dtype(img, tf.float32)
#     '''
#         note for self-supervised method will need a function to product several random crops (random zoom/size) and resize back to orig
#         consider using two loss functions (alternating) between regular multi-class cross entropy and the NT-Xent loss which allows
#         more fine-level (even within-class) feature extraction.  Can use the multi-view [simultaneous] data from the multiple
#         cameras as well (or more generally, images close in time), while also sampling from each class in a balanced way.
#     '''
#     # img1 = tf.image.random_crop(img, size=args.crop_size) ## size = [crop_height, crop_width, 3]
#     img1 = zoom(img)
#     for f, prob in zip(augmentations, args.aug_prob):
#         img1 = tf.cond(tf.random.uniform([], 0, 1) > prob, lambda: f(img1), lambda: img1)
#     img1 = tf.clip_by_value(img1, 0, 1)
#
#     # img2 = tf.image.random_crop(img, size=args.crop_size) ## size = [crop_height, crop_width, 3]
#     img2 = zoom(img)
#     for f, prob in zip(augmentations, args.aug_prob):
#         img2 = tf.cond(tf.random.uniform([], 0, 1) > prob, lambda: f(img2), lambda: img2)
#
#     img2 = tf.clip_by_value(img2, 0, 1)
#
#     # stdrgb = np.array([27.52713196, 28.30034033, 29.1236649])
#
#     return tf.stack((img1, img2), axis=0), img_crop[1] == args.CLASS_NAMES

@tf.function
def pre_process_aug(img_crop):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_png(img_crop[0], channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img1 = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.decode_png(img_crop[1], channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img2 = tf.image.convert_image_dtype(img, tf.float32)
    '''
        note for self-supervised method will need a function to product several random crops (random zoom/size) and resize back to orig
        consider using two loss functions (alternating) between regular multi-class cross entropy and the NT-Xent loss which allows
        more fine-level (even within-class) feature extraction.  Can use the multi-view [simultaneous] data from the multiple
        cameras as well (or more generally, images close in time), while also sampling from each class in a balanced way.
    '''
    for f, prob in zip(augmentations, args.aug_prob):
        img1 = tf.cond(tf.random.uniform([], 0, 1) > prob, lambda: f(img1), lambda: img1)
    img1 = tf.clip_by_value(img1, 0, 1)

    for f, prob in zip(augmentations, args.aug_prob):
        img2 = tf.cond(tf.random.uniform([], 0, 1) > prob, lambda: f(img2), lambda: img2)

    img2 = tf.clip_by_value(img2, 0, 1)

    # stdrgb = np.array([27.52713196, 28.30034033, 29.1236649])

    return tf.stack((img1, img2), axis=0)


# class ArtificialDataset(tf.data.Dataset):
#     def _generator(indices):
#         for sample_idx in np.random.permutation(len(indices)):
#             yield imgs_raw[indices[sample_idx]], crops[indices[sample_idx]]
#
#     def __new__(cls, indices):
#         return tf.data.Dataset.from_generator(
#             cls._generator,
#             output_types=imgs_raw[0].dtype,
#             output_shapes=None,
#             args=(indices, )
#             )

nearby_indices = np.lexsort((times,crops))
imgs_raw = imgs_raw[nearby_indices]
crops = crops[nearby_indices]
times = times[nearby_indices]
#### max/min indices for random time window sampler ########
temp = np.array([w for w in mit.windowed(times, args.wind_size)])
temp = temp - temp[:,args.wind_size//2].reshape(-1,1)
temp = np.transpose((np.abs(temp) < args.wind_time_max).nonzero())
_, idx = np.unique(temp[:,0], return_index=True)
minind = np.minimum.reduceat(temp[:,1], idx) - args.wind_size_half
maxind = -np.maximum.reduceat(temp[:,1], idx) + args.wind_size_half
contr_inds = [np.concatenate((np.arange(minind[sample_idx],0), -np.arange(maxind[sample_idx], 0))) for sample_idx in range(len(minind))]
############################################################
class ArtificialDataset(tf.data.Dataset):
    def _generator(indices):
        for sample_idx in np.random.permutation(len(indices)):
            contr_ind = np.random.choice(contr_inds[indices[sample_idx]]) ## draw nonzero index sample
            yield imgs_raw[indices[sample_idx] + args.wind_size_half], imgs_raw[indices[sample_idx] + args.wind_size_half + contr_ind]

    def __new__(cls, indices):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=imgs_raw[0].dtype,
            output_shapes=None,
            args=(indices, )
            )

############################################# data loader no separation of classes #######################################
data_split_ind = np.random.permutation(len(minind)).squeeze()
train_ind = data_split_ind[:int((1-2*args.p_val)*len(data_split_ind))]
val_ind = data_split_ind[int((1 - 2 * args.p_val) * len(data_split_ind)):int((1 - args.p_val) * len(data_split_ind))]
test_ind = data_split_ind[int((1 - args.p_val) * len(data_split_ind)):]

train_ds = ArtificialDataset(train_ind).map(pre_process_aug, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size=args.batch_dim).prefetch(tf.data.experimental.AUTOTUNE)
val_ds = ArtificialDataset(val_ind).map(pre_process_aug, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size=args.batch_dim).prefetch(tf.data.experimental.AUTOTUNE)
test_ds = ArtificialDataset(test_ind).map(pre_process_aug, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size=args.batch_dim).prefetch(tf.data.experimental.AUTOTUNE)

# ############################################# data loader with separate crop classes #######################################
# dataset_train_list = []
# dataset_valid_list = []
# dataset_test_list = []
# for crop in args.CLASS_NAMES:
#     data_split_ind = np.random.permutation(np.argwhere(crops==crop)).squeeze()
#     train_ind = data_split_ind[:int((1-2*args.p_val)*len(data_split_ind))]
#     val_ind = data_split_ind[int((1 - 2 * args.p_val) * len(data_split_ind)):int((1 - args.p_val) * len(data_split_ind))]
#     test_ind = data_split_ind[int((1 - args.p_val) * len(data_split_ind)):]
#
#     dataset_train = ArtificialDataset(train_ind).map(pre_process_aug, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size=args.batch_dim).prefetch(tf.data.experimental.AUTOTUNE)
#     dataset_valid = ArtificialDataset(val_ind).map(pre_process_aug, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size=args.batch_dim).prefetch(tf.data.experimental.AUTOTUNE)
#     dataset_test = ArtificialDataset(test_ind).map(pre_process_aug, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size=args.batch_dim).prefetch(tf.data.experimental.AUTOTUNE)
#
#     dataset_train_list.append(dataset_train)
#     dataset_valid_list.append(dataset_valid)
#     dataset_test_list.append(dataset_test)
# #################################################################
# train_ds = tf.data.Dataset.zip(tuple(dataset_train_list))
# val_ds = tf.data.Dataset.zip(tuple(dataset_valid_list))
# test_ds = tf.data.Dataset.zip(tuple(dataset_test_list))
# #################################################################

data = next(test_ds.as_numpy_iterator())
# data = tf.concat([tf.concat((el[0], el[1]), axis=0) for el in data], axis=0)
data = np.concatenate(data, axis=0)
################# create Model ################
# img = tf.stack([tf.image.convert_image_dtype(tf.image.decode_png(x), dtype=tf.float32) for x in imgs_raw[:10]]) # debug
actfun = 'swish'
with tf.device(args.device):
    # model_ = resnet_models.ResNet50V2(include_top=False, weights=None, actfun = 'relu', pooling='avg')
    # model_ = resnet_models.ResNet50V2(input_shape=args.crop_size, include_top=False, weights=None, actfun=actfun, pooling='avg')
    model_ = efn.EfficientNetB0(weights=None, include_top=False, pooling='max', input_shape=data[0].shape)  # or weights='noisy-student'
    feats = layers.Dense(128, activation=actfun)(model_.output)
    ## simclr
    output = layers.Dense(32)(feats)
    model_simclr = tf.keras.Model(model_.input, output, name='simclr_model')

bn_layer_inds = [ind for ind, x in enumerate(model_simclr.layers) if 'bn' in x.name]
for ind in bn_layer_inds:
    model_simclr.layers[ind].momentum = np.float32(0)

def pair_cosine_similarity(x):
    normalized = tf.nn.l2_normalize(x, axis=1)
    return tf.matmul(normalized, normalized, adjoint_b=True)

idx = np.arange(data.shape[0])
idx[::2] += 1
idx[1::2] -= 1

def nt_xent(x, t=0.5):
    x = pair_cosine_similarity(x)
    x = tf.exp(x / t)
    x = tf.gather(x, idx, axis=0)
    x = tf.linalg.diag_part(x) / (tf.reduce_sum(x, axis=0) - tf.exp(1 / t))
    return -tf.math.log(tf.reduce_mean(x))

def train(model, model_simclr, optimizer, optimizer_simclr, scheduler, train_ds, val_ds, test_ds, args):

    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):

        for element in train_ds.take(args.take):
            ############### arange data ####################
            # x = tf.concat([tf.concat((el[0], el[1]),axis=0) for el in element], axis=0)
            # y = tf.concat([tf.concat([el[2], el[2]],axis=0) for el in element], axis=0)
            x = tf.reshape(element, (-1, *data.shape[-3:]))
            ################### self-supervised update ############################
            with tf.GradientTape() as tape:
                loss = nt_xent(model_simclr(x, training=True)) #+ tf.reduce_mean(model_simclr.losses)
            grads = tape.gradient(loss, model_simclr.trainable_variables)
            grads = [None if grad is None else tf.clip_by_norm(grad, clip_norm=args.clip_norm) for grad in grads]
            globalstep = optimizer_simclr.apply_gradients(zip(grads, model_simclr.trainable_variables))
            tf.summary.scalar('loss/train_simclr', loss, globalstep)
            ############### supervised classification update ####################
            # with tf.GradientTape(watch_accessed_variables=False) as tape:
            #     tape.watch((model.layers[-2].variables, model.layers[-1].variables))
            #     loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, model(x, training=False)))
            # grads = tape.gradient(loss, model.trainable_variables)
            # grads = [None if grad is None else tf.clip_by_norm(grad, clip_norm=args.clip_norm) for grad in grads]
            # globalstep = optimizer.apply_gradients(zip(grads, model.trainable_variables))
            # tf.summary.scalar('loss/train', loss, globalstep)

        ## potentially update batch norm variables manually
        ## variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='batch_normalization')
        ## update batch norm "moving averages" prior to validation.  Follow "state" path
        # model(x, training=False) ## clear MA values
        # for element in train_ds:
        #     x = tf.concat([el[0] for el in element], axis=0)
        #     model(x, training=True) ## aggregate BN values with weights frozen
        # model(x, training=None) ## update MA values

        ## batch norm population averaging
        ma_var_list = []
        ma_mean_list = []
        for ind0, element in train_ds.take(args.take).enumerate(start=1):
            # x = tf.concat([el[0] for el in element], axis=0)
            x = tf.reshape(element, (-1, *data.shape[-3:]))
            model_simclr(x, training=True)
            if ind0 == 1:
                for ind in bn_layer_inds:
                    ma_mean_list.append(model_simclr.layers[ind].moving_mean.numpy())
                    ma_var_list.append(model_simclr.layers[ind].moving_variance.numpy())
            else:
                for ind1, ind in enumerate(bn_layer_inds):
                    ma_mean_list[ind1] += model_simclr.layers[ind].moving_mean.numpy()
                    ma_var_list[ind1] += model_simclr.layers[ind].moving_variance.numpy()

        for ind1, ind in enumerate(bn_layer_inds):
            model_simclr.layers[ind].moving_mean.assign(ma_mean_list[ind1]/np.float32(ind0))
            model_simclr.layers[ind].moving_variance.assign(ma_var_list[ind1]/np.float32(ind0))

        # model_simclr(x, training=False) ## clear MA values
        # for element in train_ds:
        #     # x = tf.concat([el[0] for el in element], axis=0)
        #     x = tf.reshape(element, (-1, *data.shape[-3:]))
        #     model_simclr(x, training=True) ## aggregate BN values with weights frozen
        # model_simclr(x, training=None) ## update MA values

        validation_loss = []
        validation_loss_simclr = []
        for element in val_ds.take(args.take):
            # x = tf.concat([tf.concat((el[0], el[1]), axis=0) for el in element], axis=0)
            # y = tf.concat([tf.concat([el[2], el[2]],axis=0) for el in element], axis=0)
            x = tf.reshape(element, (-1, *data.shape[-3:]))
            ################### self-supervised update ############################
            loss = nt_xent(model_simclr(x, training=False)).numpy()
            validation_loss_simclr.append(loss)
            ############### supervised classification update ####################
            # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, model(x, training=False))).numpy()
            # validation_loss.append(loss)
        validation_loss_simclr = tf.reduce_mean(validation_loss_simclr)
        tf.summary.scalar('loss/validation_simclr', validation_loss_simclr, globalstep)
        # validation_loss = tf.reduce_mean(validation_loss)
        # tf.summary.scalar('loss/validation', validation_loss, globalstep)

        test_loss=[]
        test_loss_simclr = []
        for element in test_ds:
            # x = tf.concat([tf.concat((el[0], el[1]), axis=0) for el in element], axis=0)
            # y = tf.concat([tf.concat([el[2], el[2]],axis=0) for el in element], axis=0)
            x = tf.reshape(element, (-1, *data.shape[-3:]))
            ################### self-supervised update #############################
            loss = nt_xent(model_simclr(x, training=False)).numpy()
            test_loss_simclr.append(loss)
            ################## supervised classification update ####################
            # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, model(x, training=False))).numpy()
            # test_loss.append(loss)
        test_loss_simclr = tf.reduce_mean(test_loss_simclr)
        # tf.summary.scalar('loss/test_simclr', test_loss_simclr, globalstep)
        # # test_loss = tf.reduce_mean(test_loss)
        # # tf.summary.scalar('loss/test', test_loss, globalstep)  ##tf.compat.v1.train.get_global_step()

        stop = scheduler.on_epoch_end(epoch=epoch, monitor=validation_loss_simclr)

        if stop:
            break

# def load_model(args, root):
#     print('Loading model..')
#     root.restore(tf.train.latest_checkpoint(args.load or args.path))


# print('Loading dataset..')
# train_ds, val_ds, test_ds = load_dataset(args)


if args.save and not args.load:
    print('Creating directory experiment..')
    pathlib.Path(args.path).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(args.path, 'args.json'), 'w') as f:
        json.dump(str(args.__dict__), f, indent=4, sort_keys=True)

# pathlib.Path(args.tensorboard).mkdir(parents=True, exist_ok=True)

## tensorboard and saving
writer = tf.summary.create_file_writer(os.path.join(args.tensorboard, args.load or args.path))
writer.set_as_default()

args.start_epoch = 0

print('Creating optimizer..')
with tf.device(args.device):
    optimizer = tf.optimizers.Adam()
with tf.device(args.device):
    optimizer_simclr = tf.optimizers.Adam()

model=None
root=None
# root = tf.train.Checkpoint(optimizer=optimizer,
#                            model=model)

# if args.load:
#     load_model(args, root)

print('Creating scheduler..')
# use baseline to avoid saving early on
scheduler = EarlyStopping(model=model_simclr, patience=args.early_stopping, args=args, root=root)

# with tf.device(args.device):
#     train(model, optimizer, scheduler, train_ds, val_ds, test_ds, args)

with tf.device(args.device):
    train(model, model_simclr, optimizer, optimizer_simclr, scheduler, train_ds, val_ds, test_ds, args)

############################################ inference ###############################################################
# model = tf.keras.models.load_model(r'C:\Users\justjo\PycharmProjects\SaS_clustering\tensorboard\SaS_2020-03-04-22-10-42\best_model')
# dataset_test_list = []
#
# for cls in args.CLASS_NAMES:
#     data_split_ind = np.random.permutation(imgs_raw[crops==cls].shape[0])
#     test_ind = data_split_ind[int((1 - args.p_val) * len(data_split_ind)):]
#
#     dataset_test = tf.data.Dataset.from_tensor_slices(np.vstack(zip(imgs_raw[crops==cls][test_ind], crops[crops==cls][test_ind])))  # .float().to(args.device)
#     dataset_test = dataset_test.shuffle(buffer_size=len(test_ind)).map(pre_process, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(
#         batch_size=args.batch_dim).take(args.take).prefetch(tf.data.experimental.AUTOTUNE)
#
#     dataset_test_list.append(dataset_test)
#
# test_ds = tf.data.Dataset.zip(tuple(dataset_test_list))
#
# test_loss = []
# for element in test_ds:
#     x = tf.concat([el[0] for el in element], axis=0)
#     y = tf.concat([el[1] for el in element], axis=0)
#     loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, model(x, training=False))).numpy()
#     test_loss.append(loss)
# test_loss = tf.reduce_mean(test_loss)
#################################################################




#################################################################




#     embeds = tf.keras.Model(model.input, model.layers[-3].output, name='embeds')
#
#     # train_data = glob.glob(r'D:\GQC_Images\GQ_Images\Corn_2017_2018/*.png')
#     # train_data = np.vstack([np.expand_dims(img_load(x, args), axis=0) for x in train_data])/128.0 - 1
#     # test_data = glob.glob(r'D:\GQC_Images\GQ_Images\test_images_broken/*.png')
#     # test_data = np.vstack([np.expand_dims(img_load(x, args), axis=0) for x in test_data])/128.0 - 1
#     # all_data = np.concatenate((train_data, test_data))
#
#     rand_crops_embeds = []
#     for x in batch(imgs, 2*args.batch_dim):
#         rand_crops_embeds.extend(embeds(x))
#
#     rand_crops_embeds = np.stack(rand_crops_embeds)
#
#     np.savetxt(r'C:\Users\justjo\PycharmProjects\furrowFeatureExtractor\tensorboard\furrowfeat_2020-01-30-18-58-47\embeds.csv', rand_crops_embeds, delimiter=',')
#
#     import matplotlib.pyplot as plt
#     from sklearn.neighbors import NearestNeighbors
#     nbrs = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(rand_crops_embeds)
#     # distances, indices = nbrs.kneighbors(np.array([-0.35703278, -0.33590597, -0.8081483 , -0.01309389]).reshape(-1,4))  ## rand_crops_embeds[30212,:] [-0.84653145, -0.14351833, -0.8278878 , -0.7618342 ]
#     distances, indices = nbrs.kneighbors(np.array([-0.24539942, -0.5202056 , -0.61923814, -0.25972468]).reshape(-1, 4))
#     plt.figure();plt.imshow(np.uint8(imgs[indices[0][5]]*255))

# if __name__ == '__main__':
#     main()

#### tensorboard --logdir=C:\Users\justjo\PycharmProjects\furrowFeatureExtractor\tensorboard
#### tensorboard --logdir=C:\Users\justjo\PycharmProjects\SaS_clustering\tensorboard
## http://localhost:6006/