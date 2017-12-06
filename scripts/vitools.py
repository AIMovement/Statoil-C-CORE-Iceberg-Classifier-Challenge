def mdlacc(histobj):
    """
    Plot Keras model accuracy.
    :param histobj: Keras history object.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10,10))
    plt.plot(histobj.history['acc'])
    plt.plot(histobj.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.show()


def mdlloss(histobj):
    """
    Plot Keras model loss.
    :param histobj: Keras history object
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10,10))
    plt.plot(histobj.history['loss'])
    plt.plot(histobj.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.show()


def get_class(pred, label, img):
    """
    Plot image with predicted class and corresponding label.
    :param pred: 1-Dim Numpy array, containing probabilites for one prediction.
    :param label: 1-Dim Numpy array, containing categorical label i.e [0., 1.] for iceberg.
    :param img: Numpy tuple, containing an image, with size (width, height, channels)
    """
    import numpy as np

    classes = ['ship', 'iceberg']
    idxpred = np.argmax(pred)
    idxlabel = np.argmax(label)

    dispimg(img)
    print('Prediction class = {}'.format(classes[idxpred]))
    print('Prediction value (%) = {}'.format(pred[idxpred]))
    print('Label class = {}'.format(classes[idxlabel]))


def dispimg(img):
    """
    Plot both HV and HH channels for one image.
    :param img: Numpy tuple, containing an image, with size (width, height, channels)
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

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
