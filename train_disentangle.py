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
args.patience = 300
args.load = r''
args.save = True
args.tensorboard = r'D:\pycharm_projects\SaS\tensorboard'
args.p_val = 0.2
args.take = 1000
args.batch_dim = 10
args.crop_size = [32, 32, 3]

args.path = os.path.join(args.tensorboard, 'SaS_{}'.format(str(datetime.datetime.now())[:-7].replace(' ', '-').replace(':', '-')))


imgs_raw = np.load(r'D:\pycharm_projects\SaS\imgs_raw_coded_png_bytes.npy')
fn_time_crop_list = np.load(r'D:\pycharm_projects\SaS\fn_time_crop.npy')
crops = np.array([ii[2] for ii in fn_time_crop_list])
times = np.array([float(ii[1]) for ii in fn_time_crop_list])
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
    rval = tf.random.uniform(shape=[], minval=0.8, maxval=1.2)
    bbox = [args.crop_size[0]*rval, args.crop_size[1]*rval, 3]
    x = tf.image.random_crop(x, size=bbox)
    return tf.image.resize(x, size=args.crop_size[:2])

args.aug_prob = [0]
args.aug_prob.extend([0.6]*5)
augmentations = [lambda x: tf.image.random_brightness(x, 0.2),
                tf.image.random_flip_left_right,
                 tf.image.random_flip_up_down,
                 lambda x: tf.image.random_hue(x, 0.5),
                 lambda x: tf.image.random_saturation(x, 0.1, 10),
                 lambda x: tf.image.random_contrast(x, 0.7, 1.3)]

@tf.function
def pre_process_aug(img_crop):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_png(img_crop[0], channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    '''
        note for self-supervised method will need a function to product several random crops (random zoom/size) and resize back to orig
        consider using two loss functions (alternating) between regular multi-class cross entropy and the NT-Xent loss which allows
        more fine-level (even within-class) feature extraction.  Can use the multi-view [simultaneous] data from the multiple
        cameras as well (or more generally, images close in time), while also sampling from each class in a balanced way.
    '''
    img1 = zoom(img)
    # img1 = tf.image.random_crop(img, size=args.crop_size) ## size = [crop_height, crop_width, 3]
    for f, prob in zip(augmentations, args.aug_prob):
        img1 = tf.cond(tf.random.uniform([], 0, 1) > prob, lambda: f(img1), lambda: img1)
    img1 = tf.clip_by_value(img1, 0, 1)

    img2 = img
    # img2 = tf.image.random_crop(img, size=args.crop_size) ## size = [crop_height, crop_width, 3]
    for f, prob in zip(augmentations, args.aug_prob):
        img2 = tf.cond(tf.random.uniform([], 0, 1) > prob, lambda: f(img2), lambda: img2)

    img2 = tf.clip_by_value(img2, 0, 1)

    # stdrgb = np.array([27.52713196, 28.30034033, 29.1236649])

    return img1, img2, img_crop[1] == args.CLASS_NAMES

@tf.function
def pre_process(img_crop):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_png(img_crop[0], channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)

    img1 = tf.image.random_crop(img, size=args.crop_size) ## size = [crop_height, crop_width, 3]

    # stdrgb = np.array([27.52713196, 28.30034033, 29.1236649])

    return img1, img, img_crop[1] == args.CLASS_NAMES

class ArtificialDataset(tf.data.Dataset):
    def _generator(indices):
        for sample_idx in np.random.permutation(len(indices)):
            yield imgs_raw[indices[sample_idx]], crops[indices[sample_idx]]

    def __new__(cls, indices):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=imgs_raw[0].dtype,
            output_shapes=None,
            args=(indices, )
            )

############################################# data loader #######################################
dataset_train_list = []
dataset_valid_list = []
dataset_test_list = []

for cls in args.CLASS_NAMES:
    data_split_ind = np.random.permutation(np.argwhere(crops==cls)).squeeze()
    train_ind = data_split_ind[:int((1-2*args.p_val)*len(data_split_ind))]
    val_ind = data_split_ind[int((1 - 2 * args.p_val) * len(data_split_ind)):int((1 - args.p_val) * len(data_split_ind))]
    test_ind = data_split_ind[int((1 - args.p_val) * len(data_split_ind)):]

    dataset_train = ArtificialDataset(train_ind).map(pre_process_aug, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size=args.batch_dim).prefetch(tf.data.experimental.AUTOTUNE)
    dataset_valid = ArtificialDataset(val_ind).map(pre_process, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size=args.batch_dim).prefetch(tf.data.experimental.AUTOTUNE)
    dataset_test = ArtificialDataset(test_ind).map(pre_process, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size=args.batch_dim).prefetch(tf.data.experimental.AUTOTUNE)

    dataset_train_list.append(dataset_train)
    dataset_valid_list.append(dataset_valid)
    dataset_test_list.append(dataset_test)

#################################################################

train_ds = tf.data.Dataset.zip(tuple(dataset_train_list))
val_ds = tf.data.Dataset.zip(tuple(dataset_valid_list))
test_ds = tf.data.Dataset.zip(tuple(dataset_test_list))

data = next(test_ds.as_numpy_iterator())

actfun = 'swish'
with tf.device(args.device):

    model_ = efn.EfficientNetB1(weights=None, include_top=False, pooling='max', input_shape=data[0][1][0].shape)  # or weights='noisy-student'
    feats = layers.Dense(128, activation=actfun)(model_.output)
    output = layers.Dense(32)(feats)
    model_full = tf.keras.Model(model_.input, output, name='model_full')

    x = layers.Dense(32, activation=actfun)(output)
    output = layers.Dense(args.CLASS_NAMES.shape[0])(x)
    model_weaksup_full = tf.keras.Model(model_.input, output, name='class_model_full')

    model_ = efn.EfficientNetB0(weights=None, include_top=False, pooling='max', input_shape=data[0][0][0].shape)  # or weights='noisy-student'
    feats = layers.Dense(128, activation=actfun)(model_.output)
    output = layers.Dense(32)(feats)
    model_crop = tf.keras.Model(model_.input, output, name='model_crop')

    x = layers.Dense(32, activation=actfun)(output)
    output = layers.Dense(args.CLASS_NAMES.shape[0])(x)
    model_weaksup_crop = tf.keras.Model(model_.input, output, name='class_model_crop')

    # inputs = tf.keras.Input(shape=(64,))
    inputs = tf.keras.Input(shape=(32,))
    x = layers.Dense(32, activation=actfun)(inputs)
    output = layers.Dense(32)(x) ## hard threshold
    output = tf.nn.sigmoid(output) ## soft threshold
    mask_ann = tf.keras.Model(inputs, output, name='mask_ann')

model_parms_grouped = [item for sublist in [model_full.trainable_variables, model_crop.trainable_variables, mask_ann.trainable_variables] for item in sublist]

bn_layer_inds_full = [ind for ind, x in enumerate(model_full.layers) if 'bn' in x.name]
for ind in bn_layer_inds_full:
    model_full.layers[ind].momentum = np.float32(0)

bn_layer_inds_crop = [ind for ind, x in enumerate(model_crop.layers) if 'bn' in x.name]
for ind in bn_layer_inds_crop:
    model_crop.layers[ind].momentum = np.float32(0)

def pair_cosine_similarity(x):
    # x = tf.nn.l2_normalize(x, axis=1)
    return tf.matmul(x, x, adjoint_b=True)

def squared_distance(x):
    # x = tf.nn.l2_normalize(x, axis=1)
    r = tf.reduce_sum(x * x, 1)
    # turn r into column vector
    r = tf.reshape(r, [-1, 1])
    D = r - 2 * tf.matmul(x, tf.transpose(x)) + tf.transpose(r)
    return tf.maximum(D, 0.0)

data = tf.concat([el[0] for el in data], axis=0)
idx = np.arange(data.shape[0]*2)
idx[::2] += 1
idx[1::2] -= 1

def nt_xent(x, t=0.5):
    x = pair_cosine_similarity(x)
    x = tf.exp(x / t)
    x = tf.gather(x, idx, axis=0)
    x = tf.linalg.diag_part(x) / (tf.reduce_sum(x, axis=0) - tf.exp(1 / t))
    return -tf.math.log(tf.reduce_mean(x))

def nt_xent_euclid(x, t=0.5):
    x = squared_distance(x)
    x = tf.gather(x, idx, axis=0)
    x = tf.linalg.diag_part(x) / (tf.reduce_sum(x, axis=0) - 1)
    return tf.math.log(tf.reduce_mean(x))

def train(model_full, model_crop, model_weaksup_full, model_weaksup_crop, mask_ann, optimizer, optimizer_weak_sup_full, optimizer_weak_sup_crop, scheduler_full, scheduler_crop, scheduler_weak_sup_full, scheduler_weak_sup_crop, scheduler_mask, train_ds, val_ds, test_ds, args):

    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):

        for element in train_ds:
            ############### arange data ####################
            x_full = tf.concat([el[1] for el in element], axis=0)
            x_crop = tf.concat([el[0] for el in element], axis=0)
            y = tf.concat([el[2] for el in element], axis=0)
            ################## self-supervised update log-bilinear ############################
            with tf.GradientTape(persistent=False) as tape:
                mdl_full_out = model_full(x_full, training=True)
                mdl_crop_out = model_crop(x_crop, training=True)
                # mask = tf.cast(mask_ann(tf.concat([mdl_full_out, mdl_crop_out], axis=-1)) > 0, tf.float32)
                mdl_full_out = tf.nn.l2_normalize(mdl_full_out, axis=1)
                mdl_crop_out = tf.nn.l2_normalize(mdl_crop_out, axis=1)
                mask = mask_ann(10*mdl_crop_out)
                # mask = tf.cast(mask > 0, tf.float32)
                loss = nt_xent(tf.reshape(tf.stack([mdl_full_out*mask, mdl_crop_out*mask], axis=1), [-1, tf.shape(mdl_full_out)[1]]))
            grads = tape.gradient(loss, model_parms_grouped)
            grads = tf.clip_by_global_norm(grads, args.clip_norm)
            globalstep = optimizer.apply_gradients(zip(grads[0], model_parms_grouped))
            tf.summary.scalar('loss/train_contrastive', loss, globalstep)
            ############## supervised classification updates ####################
            # with tf.GradientTape(persistent=False) as tape:
            #     loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, model_weaksup_full(x_full, training=True))) + \
            #            tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, model_weaksup_crop(x_crop, training=True)))
            # grads = tape.gradient(loss, [model_weaksup_full.trainable_variables, model_weaksup_crop.trainable_variables])
            # grads_full = tf.clip_by_global_norm(grads[0], args.clip_norm)
            # globalstep = optimizer_weak_sup_full.apply_gradients(zip(grads_full[0], model_weaksup_full.trainable_variables))
            # tf.summary.scalar('loss/weak_sup_full', loss, globalstep)
            # grads_crop = tf.clip_by_global_norm(grads[1], args.clip_norm)
            # optimizer_weak_sup_crop.apply_gradients(zip(grads_crop[0], model_weaksup_crop.trainable_variables))
            # tf.summary.scalar('loss/weak_sup_crop', loss, globalstep)
            tf.summary.histogram('mask_hist', mask, globalstep)


        ## batch norm population averaging
        ma_var_list_full = []
        ma_mean_list_full = []
        ma_var_list_crop = []
        ma_mean_list_crop = []
        for ind0, element in train_ds.enumerate(start=1):
            x_full = tf.concat([el[1] for el in element], axis=0)
            x_crop = tf.concat([el[0] for el in element], axis=0)
            model_full(x_full, training=True)
            model_crop(x_crop, training=True)
            if ind0 == 1:
                for ind in bn_layer_inds_full:
                    ma_mean_list_full.append(model_full.layers[ind].moving_mean.numpy())
                    ma_var_list_full.append(model_full.layers[ind].moving_variance.numpy())
                for ind in bn_layer_inds_crop:
                    ma_mean_list_crop.append(model_crop.layers[ind].moving_mean.numpy())
                    ma_var_list_crop.append(model_crop.layers[ind].moving_variance.numpy())
            else:
                for ind1, ind in enumerate(bn_layer_inds_full):
                    ma_mean_list_full[ind1] += model_full.layers[ind].moving_mean.numpy()
                    ma_var_list_full[ind1] += model_full.layers[ind].moving_variance.numpy()
                for ind1, ind in enumerate(bn_layer_inds_crop):
                    ma_mean_list_crop[ind1] += model_crop.layers[ind].moving_mean.numpy()
                    ma_var_list_crop[ind1] += model_crop.layers[ind].moving_variance.numpy()

        for ind1, ind in enumerate(bn_layer_inds_full):
            model_full.layers[ind].moving_mean.assign(ma_mean_list_full[ind1]/np.float32(ind0))
            model_full.layers[ind].moving_variance.assign(ma_var_list_full[ind1]/np.float32(ind0))
        for ind1, ind in enumerate(bn_layer_inds_crop):
            model_crop.layers[ind].moving_mean.assign(ma_mean_list_crop[ind1]/np.float32(ind0))
            model_crop.layers[ind].moving_variance.assign(ma_var_list_crop[ind1]/np.float32(ind0))

        validation_loss_cont = []
        validation_loss_weak_sup_full = []
        validation_loss_weak_sup_crop = []
        for element in val_ds:
            x_full = tf.concat([el[1] for el in element], axis=0)
            x_crop = tf.concat([el[0] for el in element], axis=0)
            y = tf.concat([el[2] for el in element], axis=0)
            ################### self-supervised update ############################
            mdl_full_out = model_full(x_full, training=False)
            mdl_crop_out = model_crop(x_crop, training=False)
            mdl_full_out = tf.nn.l2_normalize(mdl_full_out, axis=1)
            mdl_crop_out = tf.nn.l2_normalize(mdl_crop_out, axis=1)
            mask = mask_ann(10*mdl_crop_out)
            # mask = tf.cast(mask > 0, tf.float32)
            loss = nt_xent(tf.reshape(tf.stack([mdl_full_out * mask, mdl_crop_out * mask], axis=1), [-1, tf.shape(mdl_full_out)[1]]))
            validation_loss_cont.append(loss)
            # ############### supervised classification update ####################
            # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, model_weaksup_full(x_full, training=False)))
            # validation_loss_weak_sup_full.append(loss)
            # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, model_weaksup_crop(x_crop, training=False)))
            # validation_loss_weak_sup_crop.append(loss)
        validation_loss_cont = tf.reduce_mean(validation_loss_cont)
        tf.summary.scalar('loss/validation_loss_cont', validation_loss_cont, globalstep)
        # validation_loss_weak_sup_full = tf.reduce_mean(validation_loss_weak_sup_full)
        # tf.summary.scalar('loss/validation_loss_weak_sup_full', validation_loss_weak_sup_full, globalstep)
        # validation_loss_weak_sup_crop = tf.reduce_mean(validation_loss_weak_sup_crop)
        # tf.summary.scalar('loss/validation_loss_weak_sup_crop', validation_loss_weak_sup_crop, globalstep)

        # test_loss=[]
        # test_loss_simclr = []
        # test_loss_simclr_euclid = []
        # for element in test_ds:
        #     x = tf.concat([tf.concat((el[0], el[1]), axis=0) for el in element], axis=0)
        #     y = tf.concat([tf.concat([el[2], el[2]],axis=0) for el in element], axis=0)
        #     ################### self-supervised update #############################
        #     loss = nt_xent(model_simclr(x, training=False)).numpy()
        #     test_loss_simclr.append(loss)
        #     ################### self-supervised euclid update #############################
        #     loss = nt_xent_euclid(model_simclr(x, training=False)).numpy()
        #     test_loss_simclr_euclid.append(loss)
        #     ################## supervised classification update ####################
        #     loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, model(x, training=False))).numpy()
        #     test_loss.append(loss)
        # test_loss_simclr = tf.reduce_mean(test_loss_simclr)
        # tf.summary.scalar('loss/test_simclr', test_loss_simclr, globalstep)
        # test_loss_simclr_euclid = tf.reduce_mean(test_loss_simclr_euclid)
        # tf.summary.scalar('loss/test_simclr_euclid', test_loss_simclr_euclid, globalstep)
        # test_loss = tf.reduce_mean(test_loss)
        # tf.summary.scalar('loss/test', test_loss, globalstep)  ##tf.compat.v1.train.get_global_step()

        # stop = scheduler.on_epoch_end(epoch=epoch, monitor=validation_loss_simclr_euclid) or scheduler_simclr.on_epoch_end(epoch=epoch, monitor=validation_loss_simclr)
        stop = scheduler_full.on_epoch_end(epoch=epoch, monitor=validation_loss_cont) or scheduler_crop.on_epoch_end(epoch=epoch, monitor=validation_loss_cont) or scheduler_mask.on_epoch_end(epoch=epoch, monitor=validation_loss_cont)

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
    optimizer_weak_sup_full = tf.optimizers.Adam()
with tf.device(args.device):
    optimizer_weak_sup_crop = tf.optimizers.Adam()
with tf.device(args.device):
    optimizer = tf.optimizers.Adam()

# if args.load:
#     load_model(args, root)

print('Creating scheduler..')
# use baseline to avoid saving early on
scheduler_full = EarlyStopping(model=model_full, patience=args.patience, args=args, root=None)
scheduler_crop = EarlyStopping(model=model_crop, patience=args.patience, args=args, root=None)
scheduler_weak_sup_full = EarlyStopping(model=model_weaksup_full, patience=args.patience, args=args, root=None)
scheduler_weak_sup_crop = EarlyStopping(model=model_weaksup_crop, patience=args.patience, args=args, root=None)
scheduler_mask = EarlyStopping(model=mask_ann, patience=args.patience, args=args, root=None)

# with tf.device(args.device):
#     train(model, optimizer, scheduler, train_ds, val_ds, test_ds, args)

with tf.device(args.device):
    train(model_full, model_crop, model_weaksup_full, model_weaksup_crop, mask_ann, optimizer, optimizer_weak_sup_full, optimizer_weak_sup_crop, scheduler_full, scheduler_crop, scheduler_weak_sup_full, scheduler_weak_sup_crop, scheduler_mask, train_ds, val_ds, test_ds, args)


#### tensorboard --logdir=C:\Users\justjo\PycharmProjects\furrowFeatureExtractor\tensorboard
#### tensorboard --logdir=C:\Users\justjo\PycharmProjects\SaS_clustering\tensorboard
## http://localhost:6006/

# C:\Program Files\NVIDIA Corporation\NVSMI
# nvidia-smi  -l 2
