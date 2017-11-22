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
def get_images(df):
    images = []

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
    ax.imshow(img[:, :, 0], cmap=cm.inferno)
    ax.set_title('Band 1')

    ax = plt.subplot(1, 2, 2)
    im = ax.imshow(img[:, :, 1], cmap=cm.inferno)
    ax.set_title('Band 2')

    cax = fig.add_axes([0.95, 0.1, 0.03, 0.8])
    fig.colorbar(im, cax=cax, label='[dB]')

    plt.show()


def get_class(pred, label, img):
    classes = ['ship', 'iceberg']
    pred_i = np.argmax(pred)
    label_i = np.argmax(label)
    print('Prediction class = {}'.format(classes[pred_i]))
    print('Prediction value (%) = {}'.format(pred[pred_i]))
    print('Label class = {}'.format(classes[label_i]))
    show_image(img)


# # Import data

train_df = pd.read_json(os.path.join(PROJ_ROOT, 'data', 'train.json'), dtype='float32')
test_df  = pd.read_json(os.path.join(PROJ_ROOT, 'data', 'test.json'),  dtype='float32')

train_df.head(5)

X = get_images(train_df)
Y = to_categorical(train_df.is_iceberg.values, num_classes=2) # [0. 1.]=iceberg, [1. 0.]=ship
TEST = get_images(test_df)
TEST_labels = test_df['id']

train_X, val_X, train_Y, val_Y = train_test_split(X, Y, test_size=0.10,  shuffle=True, random_state=12)


# # CNN Model Configuration

model = Sequential()
model.add(BatchNormalization(input_shape = (75, 75, 2)))
model.add(Conv2D(32, kernel_size = (5,5)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64, kernel_size = (4,4)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(128, kernel_size = (3,3)))
model.add(MaxPooling2D((2,2)))
model.add(GlobalMaxPooling2D())
model.add(Dropout(0.5))
model.add(Dense(8))
model.add(Dense(2, activation = 'softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

model = Sequential()
model.add(BatchNormalization(input_shape = (75, 75, 2)))
for i in range(4):
    model.add(Conv2D(8*2**i, kernel_size = (3,3)))
    model.add(MaxPooling2D((2,2)))
model.add(GlobalMaxPooling2D())
model.add(Dropout(0.5))
model.add(Dense(8))
model.add(Dense(2, activation = 'softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model_hist = model.fit(train_X, train_Y, validation_data=(val_X, val_Y), batch_size=32, epochs=10)

plot_loss(model_hist)

plot_acc(model_hist)

predicts = model.predict(train_X, batch_size=32)

val_preds = model.predict(val_X, batch_size=32)

test_preds = model.predict(TEST, batch_size=32)

sample = 100
get_class(predicts[sample], train_Y[sample], train_X[sample])

sample = 30
get_class(val_preds[sample], val_Y[sample], val_X[sample])

is_ice = test_preds[:, 1]
ids = TEST_labels

ids.shape

test_pd = pd.DataFrame([ids, is_ice], columns=['id', 'is_iceberg'])

test_pd.head(5)

ids.values

with open('subv1.csv', 'w') as fp:
    fp.write('id,is_iceberg\n')
    for i in range(len(TEST_labels)):
        fp.write('{0:},{1:.10f}\n'.format(TEST_labels[i], test_preds[i,1]))
