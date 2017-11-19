#!/usr/bin/env python

# Python script version of correlation_analysis.ipynb
# If the pyhon script becomes out of sync with the notebook,
# 'jupyter nbconvert --to script CNN_sb_v1.ipynb'
# can be used to convert it again. Be prepared to do some manual fixes in that
# case

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    plt.plot(histobj.history['acc'])
    plt.plot(histobj.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


def plot_loss(histobj):
    plt.plot(histobj.history['loss'])
    plt.plot(histobj.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


# # Import data


train_df = pd.read_json(os.path.join(PROJ_ROOT, 'data', 'train.json'), dtype='float32')
# test_df = pd.read_json(os.path.join(PROJ_ROOT, 'data', 'test.json')


X = get_images(train_df)
Y = to_categorical(train_df.is_iceberg.values, num_classes=2) # [0. 1.]=iceberg, [1. 0.]=ship


train_X, val_X, train_Y, val_Y = train_test_split(X, Y, test_size=0.10,  shuffle=True, random_state=12)


# # CNN Model Configuration


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
