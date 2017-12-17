#!/usr/bin/env python
# coding: utf-8

# # Import packages

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import json
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation, BatchNormalization, GlobalMaxPooling2D
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical

from Enet import Enet
import tensorflow as tf


#!/usr/bin/env python
# coding: utf-8

# # Import packages

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import json
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation, BatchNormalization, GlobalMaxPooling2D
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical


PROJ_ROOT = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..')


# # Functions
def get_images(df, state, normalization='log'):
    images = []

    if state == 'train':
        for idx, row in df.iterrows():
            band_1 = np.array(row['band_1']).reshape(75, 75)
            band_2 = np.array(row['band_2']).reshape(75, 75)

            # Pre-Processing:
            #  - (Re)scaling
            #  - Standardization
            #  - Stretching
            #  - ...
            # band_1 /= 255
            # band_2 /= 255

            # TODO: Log normalization
            if normalization == 'normal':
                min_band_1 = np.min(band_1)
                min_band_2 = np.min(band_2)

                band_1 = (band_1-min_band_1)
                band_2 = (band_2-min_band_2)

                max_band_1 = np.max(band_1)
                max_band_2 = np.max(band_2)

                band_1 /= max_band_1
                band_2 /= max_band_2

            elif normalization == 'log':
                band_1 = np.power(10, band_1/10)
                band_2 = np.power(10, band_2/10)

                pos_1 = np.where(band_1 > 1.0)
                band_1[pos_1] = np.log10(band_1[pos_1]) + 1

                pos_2 = np.where(band_2 > 1.0)
                band_2[pos_2] = np.log10(band_2[pos_2]) + 1

            bands = np.dstack((band_1, band_2))
            images.append(bands)

    elif state == 'test':
        for row in df:
            band_1 = np.array(row['band_1']).reshape(75, 75)
            band_2 = np.array(row['band_2']).reshape(75, 75)

            # Pre-Processing:
            #  - (Re)scaling
            #  - Standardization
            #  - Stretching
            #  - ...
            # band_1 /= 255
            # band_2 /= 255

            # TODO: Log normalization
            if normalization == 'normal':
                min_band_1 = np.min(band_1)
                min_band_2 = np.min(band_2)

                band_1 = (band_1-min_band_1)
                band_2 = (band_2-min_band_2)

                max_band_1 = np.max(band_1)
                max_band_2 = np.max(band_2)

                band_1 /= max_band_1
                band_2 /= max_band_2

            elif normalization == 'log':
                band_1 = np.power(10, band_1/10)
                band_2 = np.power(10, band_2/10)

                pos_1 = np.where(band_1 > 1.0)
                band_1[pos_1] = np.log10(band_1[pos_1]) + 1

                pos_2 = np.where(band_2 > 1.0)
                band_2[pos_2] = np.log10(band_2[pos_2]) + 1

            bands = np.dstack((band_1, band_2))
            images.append(bands)

    return np.array(images)


def plot_acc(histobj):
    plt.figure(figsize=(10,10))
    plt.plot(histobj.history['acc'])
    plt.plot(histobj.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


def plot_loss(histobj):
    plt.figure(figsize=(10,10))
    plt.plot(histobj.history['loss'])
    plt.plot(histobj.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


def show_image(img):
    fig = plt.figure(figsize=(12, 5))
    ax = plt.subplot(1, 2, 1)
    ax.imshow(img[0, :, :, 0], cmap=cm.inferno)
    ax.set_title('Band 1')

    ax = plt.subplot(1, 2, 2)
    im = ax.imshow(img[0, :, :, 1], cmap=cm.inferno)
    ax.set_title('Band 2')

    cax = fig.add_axes([0.95, 0.1, 0.03, 0.8])
    fig.colorbar(im, cax=cax, label='[dB]')

    plt.show(block=True)

def get_augment_data(image, method='flip'):
    # image = np.zeros((100, 10, 10, 2))
    # image[:, 3:8, 5:6, :] = 1.0
    # show_image(image)
    if method == 'flip':
        image = image[:, :, ::-1, :]
    else:
        for i in range(image.shape[0]):
            image[i, :, :, 0] = [list(reversed(t)) for t in zip(*image[i, :, :, 0])]
            image[i, :, :, 1] = [list(reversed(t)) for t in zip(*image[i, :, :, 1])]
    # show_image(image)

    return image

def get_class(pred, label, img):
    classes = ['ship', 'iceberg']
    pred_i = np.argmax(pred)
    label_i = np.argmax(label)
    print('Prediction class = {}'.format(classes[pred_i]))
    print('Prediction value (%) = {}'.format(pred[pred_i]))
    print('Label class = {}'.format(classes[label_i]))
    #show_image(img)


def load_json(filename):
    with open(filename, 'r') as fp:
        return json.load(fp)


# # Import data

tech = 'log'

# train_df = pd.read_json(os.path.join(PROJ_ROOT, 'data', 'train.json'), dtype='float32')
# test_df = pd.read_json(os.path.join(PROJ_ROOT, 'data', 'test.json'),  dtype='float32')

train_df = pd.read_json('../../train.json', dtype='float32')

#test_set = load_json('../../test.json')

test_df = []
TEST_labels = []
i = 1
for file in os.listdir('../../data/test_data'):
    print(i)
    i += 1
    test_df.append(pd.read_json('../../data/test_data/' + file))
    name = file[:-5]
    TEST_labels.append(name)
    if i == 9:
        break

train_df.head(5)
norm = 'gg'
X = get_images(train_df, 'train', normalization=norm)
Y = to_categorical(train_df.is_iceberg.values, num_classes=2)  # [0. 1.]=iceberg, [1. 0.]=ship
Y = Y[:, 1]
TEST = get_images(test_df, 'test', normalization=norm)
train_X, val_X, train_Y, val_Y = train_test_split(X, Y, test_size=0.02,  shuffle=True, random_state=12)
if False:
    X_aug_flip = get_augment_data(train_X, method='flip')
    train_X_flip = np.concatenate((train_X, X_aug_flip), axis=0)
    X_aug_rot = get_augment_data(train_X, method='rotate')
    train_X = np.concatenate((train_X_flip, X_aug_rot), axis=0)

    train_Y_con = np.concatenate((train_Y, train_Y), axis=0)
    train_Y = np.concatenate((train_Y, train_Y_con), axis=0)
    train_X, val_X_, train_Y, val_Y_ = train_test_split(train_X, train_Y, test_size=0.0,  shuffle=True, random_state=9)
# # CNN Model Configuration
batch_size = 32
train_X = np.reshape(train_X, (-1, 75, 75, 2))
train_Y = np.reshape(train_Y, (-1, 1))

val_X = np.reshape(val_X, (-1, 75, 75, 2))
val_Y = np.reshape(val_Y, (-1, 1))

input_image = tf.placeholder(shape=[batch_size, 75, 75, 2], dtype=tf.float32)
ground_truth = tf.placeholder(shape=[batch_size, 1], dtype=tf.float32)
logits, prediction = Enet(input_image, is_training=True)

input_image_val = tf.placeholder(shape=[1, 75, 75, 2], dtype=tf.float32)
ground_truth_val = tf.placeholder(shape=[1, 1], dtype=tf.float32)
if True:
    logits_val, prediction_val = Enet(input_image_val, is_training=False)
    loss_val = tf.losses.absolute_difference(ground_truth_val, prediction_val)
#loss = tf.losses.mean_squared_error(ground_truth, prediction)
loss = tf.losses.mean_squared_error(
    labels=ground_truth,
    predictions=prediction)

#vars = tf.trainable_variables()
#lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars
#                   if 'bias' not in v.name]) * 0.001

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
loss = tf.reduce_mean(loss)

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter('train', sess.graph)
    train_writer.add_graph(sess.graph)
    sess.run(tf.global_variables_initializer())
    val_loss = 0.2
    for epochs in range(1001):
        test_preds = []
        loss_validation = []
        loss_train_all = []
        preds = []
        i = 0
        for step in range(0, len(train_X)-batch_size, batch_size):
            i += 1
            val_X_tmp = np.expand_dims(val_X[0], axis=0)
            val_Y_tmp = np.expand_dims(val_Y[0], axis=0)
            _, loss_train, pred = sess.run([train_op, loss, prediction], feed_dict={input_image: train_X[step:step+batch_size], ground_truth: train_Y[step:step+batch_size], input_image_val: val_X_tmp, ground_truth_val: val_Y_tmp})

            loss_train_all.append(loss_train)
            preds.append(abs(pred-train_Y[step:step+batch_size]))
        print('------------Training------------')
        print('epochs: ', epochs)
        print("loss train: ", np.mean(loss_train_all))
        print("mean error", np.mean(preds))
        preds = []
        if True:
            for step in range(0, len(val_X)):
                val_X_tmp = np.expand_dims(val_X[step], axis=0)
                val_Y_tmp = np.expand_dims(val_Y[step], axis=0)
                loss_validation_tmp, pred = sess.run([loss_val, prediction_val], feed_dict={input_image: train_X[step:step+batch_size], ground_truth: train_Y[step:step+batch_size], input_image_val: val_X_tmp, ground_truth_val: val_Y_tmp})
                loss_validation.append(loss_validation_tmp)
                preds.append(abs(pred-val_Y_tmp))
            print('-----Validation-----')
            print("loss validation: ", np.mean(loss_validation))
            loss_val_tmp = np.mean(loss_validation)
        preds = []
        gg = 0

        if loss_val_tmp < val_loss:
            val_loss = loss_val_tmp
            for step in range(0, len(TEST)):
                val_X_tmp = np.expand_dims(TEST[step], axis=0)
                val_Y_tmp = np.expand_dims(val_Y[0], axis=0)
                gg += 1
                pred = sess.run([prediction_val], feed_dict={input_image: train_X[0:0+batch_size], ground_truth: train_Y[0:0+batch_size], input_image_val: val_X_tmp, ground_truth_val: val_Y_tmp})
                preds.append(pred)

            with open('lowest' + "Enet.csv", 'w') as fp:
                fp.write('id,is_iceberg\n')
                for i in range(len(TEST_labels)):
                    fp.write('{0:},{1:.10f}\n'.format(TEST_labels[i], preds[i][0][0][0]))
            preds = []
            gg = 0

        elif epochs % 10 == 0:
            for step in range(0, len(TEST)):
                val_X_tmp = np.expand_dims(TEST[step], axis=0)
                val_Y_tmp = np.expand_dims(val_Y[0], axis=0)
                gg += 1
                pred = sess.run([prediction_val], feed_dict={input_image: train_X[0:0+batch_size], ground_truth: train_Y[0:0+batch_size], input_image_val: val_X_tmp, ground_truth_val: val_Y_tmp})
                preds.append(pred)

            with open(str(epochs) + "Enet.csv", 'w') as fp:
                fp.write('id,is_iceberg\n')
                for i in range(len(TEST_labels)):
                    fp.write('{0:},{1:.10f}\n'.format(TEST_labels[i], preds[i][0][0][0]))

with open("base_batch_aug.csv", 'w') as fp:
    fp.write('id,is_iceberg\n')
    for i in range(len(TEST_labels)):
        fp.write('{0:},{1:.10f}\n'.format(TEST_labels[i], preds[i][0][0][0]))

preds[preds > 0.8] = 1.0
preds[preds < 0.2] = 0.0

with open("base_batch_aug" + "_round" + ".csv", 'w') as fp:
    fp.write('id,is_iceberg\n')
    for i in range(len(TEST_labels)):
        fp.write('{0:},{1:.10f}\n'.format(TEST_labels[i], preds[i][0][0][1]))
