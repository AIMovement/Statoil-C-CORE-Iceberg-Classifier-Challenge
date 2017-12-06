def getimages(df):
    """
    Get images from Pandas dataframe.
    :param df: Pandas dataframe.
    :return: Numpy tuple, containing images, with shape (samples, width, height, channels).
    """
    import numpy as np

    images = []

    for idx, row in df.iterrows():
        band_1 = np.array(row['band_1']).reshape(75, 75)
        band_2 = np.array(row['band_2']).reshape(75, 75)

        bands = np.dstack((band_1, band_2))
        images.append(bands)

    return np.array(images)


def get_angles(df):
    """
    Get angles from Pandas dataframe.
    :param df: Pandas dataframe.
    :return: Numpy tuple, containing angles, with shape (len(df['inc_angle'], 1).
    """
    import numpy as np

    angles = []

    for idx, row in df.iterrows():
        angle = np.array(row['inc_angle'])

        angles.append(angle)

    return np.array(angles)


def writecsv(filename, predicts):
    """
    Write submission csv file.
    :param filename: Name of csv file.
    :param predicts: Numpy array with id in first column and is_iceberg probability in second column.
    """
    with open(filename, 'w') as fp:
        fp.write('id,is_iceberg\n')
        for i in range(len(predicts)):
            fp.write('{0:},{1:.10f}\n'.format(predicts[i, 0], predicts[i, 1]))


"""
def minmaxnorm(dat):

    Entire range of values of data from min to max are mapped to range 0 - 1.
    :param data: Numpy array.
    :return: Normalized Numpy array.

    import numpy as np

    if len(dat.shape) == 1:
        normdata = (dat - np.amin(dat)) / (np.amax(dat) - np.amin(dat))

    elif len(dat.shape) >= 3:
        

    return normdata
"""

