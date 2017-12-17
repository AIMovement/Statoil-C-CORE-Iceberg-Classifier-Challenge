#!/usr/bin/env python
# coding: utf-8
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import scripts.dattools as dt
import scripts.pptools as pp

# Define global variables
PROJ_ROOT = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..')
RANDOM_STATE = 12

## Import data
print('Loading data...')
train_df = pd.read_json(os.path.join(PROJ_ROOT, 'data', 'train.json'), dtype='float32')
test_df = pd.read_json(os.path.join(PROJ_ROOT, 'data', 'test.json'),  dtype='float32')

band1_X, band2_X = dt.getbands(train_df)
Y = to_categorical(train_df.is_iceberg.values, num_classes=2) # [0. 1.]=iceberg, [1. 0.]=ship

## Split data
print('Splitting data...')
band1_X_train, band1_X_val, band1_Y_train, band1_Y_val = \
    train_test_split(band1_X, Y, test_size=0.10,  shuffle=True, random_state=RANDOM_STATE)

band2_X_train, band2_X_val, band2_Y_train, band2_Y_val = \
    train_test_split(band2_X, Y, test_size=0.10,  shuffle=True, random_state=RANDOM_STATE)


print
