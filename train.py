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
from augmentations import *

crop_pat = re.compile('(?<=_)(?:.(?!_))+$')
gpstime_pat = re.compile('\d+\.\d+')

class parser_:
    pass

args = parser_()
args.device = '/gpu:0'  # '/gpu:0'
args.batch_dim = 1000
args.clip_norm = 0.1
args.epochs = 5000
args.patience = 10
args.load = r''
args.save = True
args.tensorboard = r'C:\Users\justjo\PycharmProjects\S&S_clustering\tensorboard'
args.early_stopping = 10
args.manualSeed = None
args.manualSeedw = None
args.prefetch_size = 10  # data pipeline prefetch buffer size
args.parallel = 8  # data pipeline parallel processes
args.preserve_aspect_ratio = True;  ##when resizing
args.p_val = 0.2
args.downscale = 10
args.take = 50
args.crop_size = [40, 40, 3]

args.path = os.path.join(args.tensorboard, 'furrowfeat_{}'.format(str(datetime.datetime.now())[:-7].replace(' ', '-').replace(':', '-')))

imgs_raw = np.load(r'J:\SaS\imgs_raw_coded_png_bytes.npy')
fn_time_crop_list = np.load(r'J:\SaS\fn_time_crop.npy')
crops = np.array([ii[2] for ii in fn_time_crop_list])
times = np.array([float(ii[1]) for ii in fn_time_crop_list])
fn_time_crop_list = []
args.CLASS_NAMES, args.class_counts = np.unique(crops, return_counts=True)
args.class_weights = np.max(args.class_counts)/args.class_counts
args.class_weights = args.class_weights/np.sum(args.class_weights)

def zoom(x):
    # Generate 20 crop settings, ranging from a 1% to 20% crop.
    bbox = [*args.crop_size[:2]*tf.random.uniform(shape=(1,), minval=0.8, maxval=1), 3]
    rcrop = tf.image.random_crop(x, size=bbox)

    def random_crop(img):
        # Create different crops for an image
        crops = tf.image.crop_and_resize([img], boxes=boxes, box_ind=np.zeros(len(scales)), crop_size=args.crop_size[:2])
        # Return a random crop
        return crops[tf.random_uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)]

    choice = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)

    # Only apply cropping 50% of the time
    return random_crop(x)

aug_prob = [0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75]

augmentations = [tf.image.random_flip_left_right,
                 tf.image.random_flip_up_down,
                 lambda x: tf.image.random_hue(x, 0.5),
                 lambda x: tf.image.random_saturation(x, 0.1, 10),
                 lambda x: tf.image.random_brightness(x, 0.2),
                 lambda x: tf.image.random_contrast(x, 0.7, 1.3)
                 ]

def pre_process_(img_crop):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_png(img_crop[0], channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.random_crop(img, size=args.crop_size) ## size = [crop_height, crop_width, 3]

    for f, prob in zip(augmentations, aug_prob):
        img = tf.cond(tf.random.uniform([], 0, 1) > prob, lambda: f(x), lambda: x)
    img = tf.clip_by_value(img, 0, 1)

    # stdrgb = np.array([27.52713196, 28.30034033, 29.1236649])

    # # resize the image to the desired size.
    return tf.image.resize(img, [120, 192]), img_crop[1] == args.CLASS_NAMES

def load_dataset(args):
    tf.random.set_seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    random.seed(args.manualSeed)
    args.imgs_paths = np.array([str(x) for x in args.imgs_paths])
    pre_process = functools.partial(pre_process_, args.CLASS_NAMES)

    data_split_ind = np.random.permutation(args.imgs_paths.shape[0])
    train_ind = data_split_ind[:int((1-2*args.p_val)*len(data_split_ind))]
    val_ind = data_split_ind[int((1 - 2 * args.p_val) * len(data_split_ind)):int((1 - args.p_val) * len(data_split_ind))]
    test_ind = data_split_ind[int((1 - args.p_val) * len(data_split_ind)):]

    dataset_train = tf.data.Dataset.from_tensor_slices(args.imgs_paths[train_ind])  # .float().to(args.device)
    dataset_train = dataset_train.shuffle(buffer_size=len(train_ind)).map(pre_process, num_parallel_calls=args.parallel).batch(
        batch_size=args.batch_dim).take(args.take).prefetch(tf.data.experimental.AUTOTUNE)
    # dataset_train = dataset_train.shuffle(buffer_size=len(train)).batch(batch_size=args.batch_dim).prefetch(buffer_size=args.prefetch_size)

    dataset_valid = tf.data.Dataset.from_tensor_slices(args.imgs_paths[val_ind])  # .float().to(args.device)
    dataset_valid = dataset_valid.shuffle(buffer_size=len(val_ind)).map(pre_process, num_parallel_calls=args.parallel).batch(
        batch_size=args.batch_dim).take(args.take).prefetch(tf.data.experimental.AUTOTUNE)
    # dataset_valid = dataset_valid.batch(batch_size=args.batch_dim*2).prefetch(buffer_size=args.prefetch_size)

    dataset_test = tf.data.Dataset.from_tensor_slices(args.imgs_paths[test_ind])  # .float().to(args.device)
    dataset_test = dataset_test.shuffle(buffer_size=len(test_ind)).map(pre_process, num_parallel_calls=args.parallel).batch(
        batch_size=args.batch_dim).take(args.take).prefetch(tf.data.experimental.AUTOTUNE)

    return dataset_train, dataset_valid, dataset_test

def create_model(args):

    tf.random.set_seed(args.manualSeedw)
    np.random.seed(args.manualSeedw)

    actfun = tf.nn.elu

    inputs = tf.keras.Input(shape=args.img_size, name='img')  ## (108, 192, 3)
    x = layers.Conv2D(32, 7, activation=actfun, strides=3)(inputs)
    block_output = layers.Conv2D(32,3,strides=2, activation=None)(x)
    # block_output = layers.MaxPooling2D(3, strides=2)(x)

    x = actfun(block_output)
    x = layers.Conv2D(32, 1, activation=actfun, padding='same')(x)
    x = layers.Conv2D(32, 3, activation=None, padding='same')(x)
    x = layers.add([x, block_output])
    x = layers.Conv2D(64, 1, activation=actfun)(x)
    block_output = layers.MaxPooling2D(pool_size=3, strides=2)(x)

    x = layers.Conv2D(64, 1, activation=actfun, padding='same')(block_output)
    x = layers.Conv2D(64, 3, activation=None, padding='same')(x)
    x = layers.add([x, block_output])
    x = layers.Conv2D(64, 1, activation=actfun)(x)
    # block_output = layers.MaxPooling2D(2, strides=2)(x)

    # x = layers.Conv2D(32, 1, activation=actfun)(x)
    # x = layers.Conv2D(32, 3, activation=None)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Flatten()(x)

    x = layers.Dense(32, activation=actfun)(x)
    output = layers.Dense(args.CLASS_NAMES.shape[0])(x)
    model = tf.keras.Model(inputs, output, name='resnet_model')
    model.summary()
    return model

def train(model, optimizer, scheduler, train_ds, val_ds, test_ds, args):

    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):

        for x, y in train_ds:
            with tf.GradientTape() as tape:
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, model(x, training=True)))
            grads = tape.gradient(loss, model.trainable_variables)
            grads = [None if grad is None else tf.clip_by_norm(grad, clip_norm=args.clip_norm) for grad in grads]
            globalstep = optimizer.apply_gradients(zip(grads, model.trainable_variables))
            tf.summary.scalar('loss/train', loss, globalstep)

        ## potentially update batch norm variables manually
        ## variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='batch_normalization')

        validation_loss = []
        for x, y in val_ds:
            with tf.GradientTape() as tape:
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, model(x, training=False))).numpy()
            validation_loss.append(loss)
        validation_loss = tf.reduce_mean(validation_loss)
        # print("validation loss:  " + str(validation_loss))

        # test_loss=[]
        # for ind in batch(test_ind, 2*args.batch_dim):
        #     loss = tf.reduce_mean(tf.math.squared_difference(quality_y[ind,:], model(imgs[ind,:,:,:], training=False))).numpy()
        #     test_loss.append(loss)
        # test_loss = tf.reduce_mean(test_loss)

        # print("test loss:  " + str(test_loss))

        stop = scheduler.on_epoch_end(epoch=epoch, monitor=validation_loss)

        #### tensorboard
        # tf.summary.scalar('loss/train', train_loss, tf.compat.v1.train.get_global_step())
        tf.summary.scalar('loss/validation', validation_loss, globalstep)
        # tf.summary.scalar('loss/test', test_loss, globalstep) ##tf.compat.v1.train.get_global_step()

        if stop:
            break

def load_model(args, root):
    print('Loading model..')
    root.restore(tf.train.latest_checkpoint(args.load or args.path))


# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# config.log_device_placement = True
# tf.compat.v1.enable_eager_execution(config=config)

# tf.config.experimental_run_functions_eagerly(True)
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





# ### organize data for ease of processing/training/reading
# args.imgs_paths = np.array(list(Path(args.datapath).rglob('*.png')))
# classes = np.array([crop_pat.search(x.parts[-4])[0] for x in args.imgs_paths])
#
# img_list = []
# for xx in args.imgs_paths:
#     img_list.append(tf.io.read_file(str(xx)).numpy())
#
# np.save('J:\SaS\imgs_raw_coded_png_bytes.npy', np.array(img_list))
#
# fn_time_list = []
# for crop, p in zip(classes, [s for s in args.imgs_paths]):
#     fn_time_list.append((str(p.relative_to(r'J:\SaS')), float(gpstime_pat.search(ntpath.split(p)[1])[0]), crop))
# np.save(r'J:\SaS\fn_time_crop.npy', np.array(fn_time_list))

print('Loading dataset..')
train_ds, val_ds, test_ds = load_dataset(args)


if args.save and not args.load:
    print('Creating directory experiment..')
    pathlib.Path(args.path).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(args.path, 'args.json'), 'w') as f:
        json.dump(str(args.__dict__), f, indent=4, sort_keys=True)

# pathlib.Path(args.tensorboard).mkdir(parents=True, exist_ok=True)

print('Creating model..')
with tf.device(args.device):
    model = create_model(args)

## tensorboard and saving
writer = tf.summary.create_file_writer(os.path.join(args.tensorboard, args.load or args.path))
writer.set_as_default()

root = None
args.start_epoch = 0

print('Creating optimizer..')
with tf.device(args.device):
    optimizer = tf.optimizers.Adam()
root = tf.train.Checkpoint(optimizer=optimizer,
                           model=model,
                           optimizer_step=tf.compat.v1.train.get_global_step())

if args.load:
    load_model(args, root)

print('Creating scheduler..')
# use baseline to avoid saving early on
scheduler = EarlyStopping(model=model, patience=args.early_stopping, args=args, root=root)

with tf.device(args.device):
    train(model, optimizer, scheduler, train_ds, val_ds, test_ds, args)

# ###################### inference #################################
    embeds = tf.keras.Model(model.input, model.layers[-3].output, name='embeds')

    # train_data = glob.glob(r'D:\GQC_Images\GQ_Images\Corn_2017_2018/*.png')
    # train_data = np.vstack([np.expand_dims(img_load(x, args), axis=0) for x in train_data])/128.0 - 1
    # test_data = glob.glob(r'D:\GQC_Images\GQ_Images\test_images_broken/*.png')
    # test_data = np.vstack([np.expand_dims(img_load(x, args), axis=0) for x in test_data])/128.0 - 1
    # all_data = np.concatenate((train_data, test_data))

    rand_crops_embeds = []
    for x in batch(imgs, 2*args.batch_dim):
        rand_crops_embeds.extend(embeds(x))

    rand_crops_embeds = np.stack(rand_crops_embeds)

    np.savetxt(r'C:\Users\justjo\PycharmProjects\furrowFeatureExtractor\tensorboard\furrowfeat_2020-01-30-18-58-47\embeds.csv', rand_crops_embeds, delimiter=',')

    import matplotlib.pyplot as plt
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(rand_crops_embeds)
    # distances, indices = nbrs.kneighbors(np.array([-0.35703278, -0.33590597, -0.8081483 , -0.01309389]).reshape(-1,4))  ## rand_crops_embeds[30212,:] [-0.84653145, -0.14351833, -0.8278878 , -0.7618342 ]
    distances, indices = nbrs.kneighbors(np.array([-0.24539942, -0.5202056 , -0.61923814, -0.25972468]).reshape(-1, 4))
    plt.figure();plt.imshow(np.uint8(imgs[indices[0][5]]*255))

# if __name__ == '__main__':
#     main()

#### tensorboard --logdir=C:\Users\justjo\PycharmProjects\furrowFeatureExtractor\tensorboard
## http://localhost:6006/