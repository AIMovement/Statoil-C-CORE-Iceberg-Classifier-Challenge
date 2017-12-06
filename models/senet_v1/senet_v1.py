import numpy as np
import os, sys
import keras.backend as K
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))
from keras_squeezenet import SqueezeNet
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical
from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

def get_images(df):
    images = []

    for idx, row in df.iterrows():
        band_1 = np.array(row['band_1']).reshape(75, 75)
        band_2 = np.array(row['band_2']).reshape(75, 75)
        band_3 = band_1 + band_2

        bands = np.dstack((band_1, band_2, band_3))
        images.append(bands)

    return np.array(images)


if __name__ == "__main__":
    train_df = pd.read_json('../../data/train.json', dtype='float32')
#    test_df = pd.read_json('../../data/test.json', dtype='float32')

    train_image = train_df['band_1'][0]

    # Bilinear interpolation
    train_image = zoom(train_image, 3, order=1)

    plt.imshow(train_image.reshape(225,225))
    plt.show()

    band_1 = np.array(train_image).reshape(227, 227)
    band_2 = np.array(train_image).reshape(227, 227)
    band_3 = band_1 + band_2

    X = np.dstack((band_1, band_2, band_3))

    X = np.expand_dims(X, axis=0)
    X = preprocess_input(X)

    # Get weights that are not trained (i.e., randomly initialized)
    # NOTE: This requires keras==2.0.0!
    model = SqueezeNet(weights=None)
    preds = model.predict(X)
    print('Predicted:', decode_predictions(preds))