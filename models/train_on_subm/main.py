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

# ----------------------------------------------------------------------------------------------------------------------

old_train = pd.read_json(os.path.join(PROJ_ROOT, 'data', 'train.json'), dtype='float32')

subm = pd.read_csv('C:/Users/okarnbla/Downloads/submission.csv')
subm.is_iceberg[subm['is_iceberg'] <= 0.5] = 0
subm.is_iceberg[subm['is_iceberg'] >  0.5] = 1

frames = [old_train, subm]
train = pd.concat(frames)

print