import scripts.dattools as dt
import pandas as pd
import numpy as np
from scripts.mltools import mltools
from sklearn.model_selection import train_test_split

# DATA PRE-PROCESSING
train_df = pd.read_json('C:\\Saudin\\Other\\Iceberg_data\\train.json', dtype='float32')
test_df = pd.read_json('C:\\Saudin\\Other\\Iceberg_data\\test.json',  dtype='float32')

T = train_df[train_df['inc_angle'] == 'na']
train_df = train_df[train_df['inc_angle'] != 'na']

X_train = dt.getimages(train_df)
X_test = dt.getimages(test_df)
Y_train = train_df.inc_angle.values
Y_test = test_df.inc_angle.values

X = np.concatenate((X_train, X_test), axis=0)
Y = np.concatenate((Y_train, Y_test), axis=0)

Y = dt.minmaxnorm(X[0:1])

train_X, val_X, train_Y, val_Y = train_test_split(X, Y, test_size=0.20,  shuffle=True, random_state=12)

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
mdl1 = mlobj.cnnmdl_predang_v1()
mdlhist = mlobj.train_mdl(mdl1)
