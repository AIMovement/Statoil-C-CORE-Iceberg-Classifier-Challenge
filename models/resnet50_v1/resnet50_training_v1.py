from keras.applications.resnet50 import ResNet50
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation, concatenate, Input
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy.ndimage import zoom
import pickle

""" 
    DUE TO MEMORY ISSUES WHEN TRAINING AND LOADING TEST DATA 
    WHEN RUNNING ON GPU, THE RESNET50 CODE HAS BEEN SPLIT 
    INTO TWO SCRIPTS 
"""

""" Create ResNet50 + Fully connected layers """
# Import ResNet pretrained model
model = ResNet50(weights=None, include_top=False, input_shape=(225, 225, 3))

# Adding custom Layers
x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(2, activation="softmax")(x)

# creating the final model
model_final = Model(inputs=model.input, outputs=predictions)

# compile the model
model_final.compile(loss="categorical_crossentropy", optimizer=SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])
model_final.summary()


def get_images(df):
    images = []
    max1 = np.mean(np.max(df['band_1']))
    max2 = np.mean(np.max(df['band_2']))

    min1 = np.mean(np.min(df['band_1']))
    min2 = np.mean(np.min(df['band_2']))

    for idx, row in df.iterrows():
        band_1 = np.array(row['band_1']).reshape(75, 75)
        band_2 = np.array(row['band_2']).reshape(75, 75)

        # Bilinear interpolation
        band_1 = zoom(band_1, 3, order=1)
        band_1 = (band_1-min1)/(max1-min1)

        band_2 = zoom(band_2, 3, order=1)
        band_2 = (band_2-min2)/(max2-min2)

        band_3 = band_2 - band_1

        X = np.dstack((band_1, band_2, band_3))
        images.append(X)

    return np.array(images), {'Max band_1': max1, 'Max band_2': max2, 'Min band_1': min1, 'Min band_2': min2}


""" Import data """
train_df = pd.read_json('../../data/train.json', dtype='float32')

""" Drop N/A """
train_df = train_df[train_df['inc_angle'] != 'na']

" Train/val/test split "
state = 100
train_ran_df = train_df.sample(frac=1, random_state=state)
train_ran_df = train_ran_df.reset_index(drop=True)

X, max_min_dict = get_images(train_ran_df)
Y = to_categorical(train_ran_df.is_iceberg.values, num_classes=2)  # [0. 1.]=iceberg, [1. 0.]=ship

train_samples = np.round(0.8 * (X[:, 0, 0, 0].shape[0]))
train_samples = np.int(train_samples)

X_train = X[0:train_samples]
Y_train = Y[0:train_samples]

X_val = X[train_samples + 1:-1]
Y_val = Y[train_samples + 1:-1]

""" Train model """
model_hist = model_final.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=32, epochs=50)

" Save history "
with open('resnet50_custom', 'wb') as hist:
    pickle.dump(model_hist.history, hist)

hist.close()

" Save normalization factors "
with open('max_min_dict', 'wb') as max_min:
    pickle.dump(max_min_dict, max_min)


# Save model
model_final.save('resnet50_custom_model.h5')