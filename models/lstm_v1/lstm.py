from __future__ import print_function
from keras.models import Model
from keras.layers import Input, Dense, TimeDistributed, Dropout
from keras.layers import LSTM
from keras.utils.np_utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import os

def standard_scale(*args):
    """
    Standardize features by removing the mean and scaling to unit variance
    :param args: One or multidimensional numpy array.
    :return: The scaler-object(s)
    """
    from sklearn.preprocessing import StandardScaler

    for idx, arg in enumerate(args):
        scaler = StandardScaler().fit(arg)

    return scaler


def get_images(df):
    images = []

    for idx, row in df.iterrows():
        band_1 = np.array(row['band_1']).reshape(75, 75)
        band_2 = np.array(row['band_2']).reshape(75, 75)

        band_1_scaler = standard_scale(band_1)
        band_2_scaler = standard_scale(band_2)

        band_1 = band_1_scaler.transform(band_1)
        band_2 = band_2_scaler.transform(band_2)

        bands = np.dstack((band_1, band_2))
        images.append(bands)

    return np.array(images)

PROJ_ROOT = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..')

# Training parameters.
batch_size = 32
num_classes = 2
epochs = 5

# Embedding dimensions.
row_hidden = 128
col_hidden = 128

train_df = pd.read_json(os.path.join(PROJ_ROOT, 'data', 'train.json'), dtype='float32')
test_df = pd.read_json(os.path.join(PROJ_ROOT, 'data', 'test.json'),  dtype='float32')

train_X = get_images(train_df)
train_y = to_categorical(train_df.is_iceberg.values, num_classes=num_classes)

train_X, test_X, y_train, y_test = train_test_split(train_X, train_y, test_size=0.10,  shuffle=True, random_state=12)

# Reshapes data to 4D for Hierarchical RNN.
x_train = train_X.reshape(train_X.shape[0], 75, 75, 2)
x_test = test_X.reshape(test_X.shape[0], 75, 75, 2)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

row, col, pixel = x_train.shape[1:]

# 4D input.
x = Input(shape=(row, col, pixel))

# Encodes a row of pixels using TimeDistributed Wrapper.
encoded_rows = TimeDistributed(LSTM(row_hidden))(x)

# Encodes columns of encoded rows.
encoded_columns = LSTM(col_hidden)(encoded_rows)
dense_1 = Dense(64, activation='relu')(encoded_columns)
dropout = Dropout(0.2)(dense_1)
dense_2 = Dense(32, activation='relu')(dropout)

# Final predictions and model.
prediction = Dense(num_classes, activation='softmax')(dense_2)
model = Model(x, prediction)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()
# Training.
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

# Evaluation.
scores = model.evaluate(x_test, y_test, verbose=0)

TEST = get_images(test_df)
TEST_labels = test_df['id']
preds = model.predict(TEST)

with open('sub_lstm.csv', 'w') as fp:
    fp.write('id,is_iceberg\n')
    for i in range(len(TEST_labels)):
        fp.write('{0:},{1:.10f}\n'.format(TEST_labels[i], preds[i, 1]))

print('Test loss:', scores[0])
print('Test accuracy:', scores[1])