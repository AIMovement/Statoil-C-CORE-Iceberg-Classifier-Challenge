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
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation, concatenate, Input
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
        band_1 /= 255
        band_2 /= 255


        bands = np.dstack((band_1, band_2))
        images.append(bands)

    return np.array(images)


def get_angles(df):
    angles = []
    max_angle = df['inc_angle'].max()

    for idx, row in df.iterrows():
        angle = np.array(row['inc_angle'])

        # Pre-Processing:
        #  - ...

        #angle /= max_angle

        angles.append(angle)

    return np.array(angles)


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


# #### Drop 'na' values (133 of them) from the training set

train_df = train_df[train_df['inc_angle'] != 'na']


# #### Randomize pandas dataframe (all input features)

state = 100
train_ran_df = train_df.sample(frac=1, random_state=state)
train_ran_df = train_ran_df.reset_index(drop=True)

X_cnn = get_images(train_ran_df)
X_ang = get_angles(train_ran_df)
Y = to_categorical(train_ran_df.is_iceberg.values, num_classes=2) # [0. 1.]=iceberg, [1. 0.]=ship
X_cnn_test = get_images(test_df)
X_ang_test = get_angles(test_df)
X_ids = test_df['id']

train_samples = np.round(0.8*len(X_cnn))
train_samples = train_samples.astype('int')
X_cnn_train = X_cnn[0:train_samples]
X_ang_train = X_ang[0:train_samples]
Y_train = Y[0:train_samples]
X_cnn_val = X_cnn[train_samples+1:-1]
X_ang_val = X_ang[train_samples+1:-1]
Y_val = Y[train_samples+1:-1]


# # CNN/FNN Model Configuration

input1 = Input(shape=(75,75,2), name='CNN-Input')

x1 = Conv2D(filters=64, kernel_size=(5,5), activation='relu')(input1)
x1 = Dropout(0.2)(x1)
x1 = MaxPooling2D(pool_size=(2,2))(x1)
x1 = Conv2D(filters=32, kernel_size=(5,5), activation='relu')(x1)
x1 = MaxPooling2D(pool_size=(2,2))(x1)
x1 = Conv2D(filters=32, kernel_size=(3,3), activation='relu')(x1)
x1 = MaxPooling2D(pool_size=(2,2))(x1)
x1 = Flatten()(x1)
x1 = Dense(127, activation='relu')(x1)

input2 = Input(shape=(1,), name='Angle-Input')

x2 = concatenate([x1, input2], axis=1, name='Merge-Layer')
x2 = Dense(32, activation='relu')(x2)

predictions = Dense(2, activation='softmax', name='Model-Output')(x2)

model = Model(inputs=[input1, input2], outputs=predictions)
model.summary()

model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model_hist = model.fit([X_cnn_train, X_ang_train], Y_train, validation_data=([X_cnn_val, X_ang_val], Y_val), batch_size=32, epochs=50)

plot_loss(model_hist)

plot_acc(model_hist)

test_preds = model.predict([X_cnn_test, X_ang_test], batch_size=32)

#  sample = 30
#  get_class(val_preds[sample], val_Y[sample], val_X[sample])

#  is_ice = test_preds[:, 1]
#  ids = TEST_labels

with open('subv3.csv', 'w') as fp:
    fp.write('id,is_iceberg\n')
    for i in range(len(X_ids)):
        fp.write('{0:},{1:.10f}\n'.format(X_ids[i], test_preds[i,1]))
