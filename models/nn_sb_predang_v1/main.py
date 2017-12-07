import os
import scripts.dattools as dt
import pandas as pd
import numpy as np
from scripts.mltools import mltools
from sklearn.model_selection import train_test_split

# DATA PRE-PROCESSING
PROJ_ROOT = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..')

train_df = pd.read_json(os.path.join(PROJ_ROOT, 'data', 'train.json'))  #Place "train.json" in data
test_df = pd.read_json(os.path.join(PROJ_ROOT, 'data', 'train.json'))   #Place "test.json" in data

Na1 = train_df[train_df['inc_angle'] == 'na']
Na2 = test_df[test_df['inc_angle'] == 'na']
train_df = train_df[train_df['inc_angle'] != 'na']
test_df = test_df[test_df['inc_angle'] != 'na']

X_train = dt.getimages(train_df)
X_test = dt.getimages(test_df)
Y_train = train_df.inc_angle.values
Y_test = test_df.inc_angle.values

X = np.concatenate((X_train, X_test), axis=0)
Y = np.concatenate((Y_train, Y_test), axis=0)

train_X, val_X, train_Y, val_Y = train_test_split(X, Y, test_size=0.10,  shuffle=True, random_state=12)

# MODEL CONFIGURATION
MDLCONF = {
    'LR':           0.001,
    'BATCHSIZE':    32,
    'EPOCHS':       30,
    'MOMENTUM':     0.1,
    'DECAY':        0.0,
    'NESTEROV':     False,
    'OPTIMIZER':    'sgd',
    'LOSS':         'mean_squared_error',
    'METRICS':      'accuracy'
}

# MODEL INITIALIZATION
mlobj = mltools(MDLCONF)
mlobj.summaryflag = True
mlobj.x_train = train_X
mlobj.y_train = train_Y
mlobj.x_val = val_X
mlobj.y_val = val_Y
mdl1 = mlobj.predang_cnn_v1()
mdlhist = mlobj.train(mdl1)
