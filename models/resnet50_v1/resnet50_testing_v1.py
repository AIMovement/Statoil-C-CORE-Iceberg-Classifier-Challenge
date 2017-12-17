from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy.ndimage import zoom
import pickle
from keras.models import load_model


""" 
    DUE TO MEMORY ISSUES WHEN TRAINING AND LOADING TEST DATA 
    WHEN RUNNING ON GPU, THE RESNET50 CODE HAS BEEN SPLIT 
    INTO TWO SCRIPTS 
"""


def plot_acc(histobj):
    plt.figure(figsize=(10, 10))
    plt.plot(histobj.history['acc'])
    plt.plot(histobj.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


def plot_loss(histobj):
    plt.figure(figsize=(10, 10))
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


def get_images(df, max_min):
    images = []

    max1 = max_min('Max band_1')
    max2 = max_min('Max band_2')

    min1 = max_min('Min band_1')
    min2 = max_min('Min band_2')

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

    return np.array(images)


def get_class(pred, label, img):
    classes = ['ship', 'iceberg']
    pred_i = np.argmax(pred)
    label_i = np.argmax(label)
    print('Prediction class = {}'.format(classes[pred_i]))
    print('Prediction value (%) = {}'.format(pred[pred_i]))
    print('Label class = {}'.format(classes[label_i]))
    show_image(img)


" Load test data "
test_df = pd.read_json('../../data/test.json', dtype='float32')

" Load pretrained model from h5 "
model_final = load_model('models/resnet50_v1/resnet50_custom_model.h5')

model_hist_obj = open('models/resnet50_v1/resnet50_custom', 'rb')
model_hist = pickle.load(model_hist_obj)
model_hist_obj.close()

max_min_obj = open('max_min_dict', 'rb')
max_min = pickle.load(max_min_obj)
max_min_obj.close()

X_test = get_images(test_df, max_min)
TEST_labels = test_df['id']

" Plot losses "
plot_loss(model_hist)

" Plot accuracy "
plot_acc(model_hist)

" Make predictions "
test_preds = model_final.predict(X_test, batch_size=32)

" "
sample = 30
is_ice = test_preds[:, 1]
ids = TEST_labels

with open('subv5.csv', 'w') as fp:
    fp.write('id,is_iceberg\n')
    for i in range(len(ids)):
        fp.write('{0:},{1:.10f}\n'.format(ids[i], test_preds[i,1]))