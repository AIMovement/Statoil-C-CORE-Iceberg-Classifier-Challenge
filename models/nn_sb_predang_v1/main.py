import os
import scripts.dattools as dt
import pandas as pd
import numpy as np
from scripts.mltools import mltools
from sklearn.model_selection import train_test_split

# DATA PRE-PROCESSING
PROJ_ROOT = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..')

train1_df = pd.read_json(os.path.join(PROJ_ROOT, 'data', 'train.json'))
train2_df = pd.read_json(os.path.join(PROJ_ROOT, 'data', 'test.json'))

na1_df = train1_df[train1_df['inc_angle'] == 'na']
train1_df = train1_df[train1_df['inc_angle'] != 'na']

X_train1 = dt.getimages(train1_df, normflag=True)
X_train2 = dt.getimages(train2_df, normflag=True)
Y_train1 = dt.getangles(train1_df, normflag=True)
Y_train2 = dt.getangles(train2_df, normflag=True)
Na_test1 = dt.getimages(na1_df, normflag=True)

X = np.concatenate((X_train1, X_train2), axis=0)
Y = np.concatenate((Y_train1, Y_train2), axis=0)

train_X, val_X, train_Y, val_Y = train_test_split(X, Y, test_size=0.20,  shuffle=True, random_state=12)

# MODEL CONFIGURATION
MDLCONF = {
    'LR':           0.001,
    'BATCHSIZE':    64,
    'EPOCHS':       10,
    'MOMENTUM':     0.2,
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
predicts = mlobj.predict(mdl1, Na_test1)
print(predicts*45)

