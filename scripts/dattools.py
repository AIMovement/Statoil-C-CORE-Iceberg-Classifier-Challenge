def getimages(df, normflag=False):
    """
    Get images from Pandas dataframe.
    :param df: Pandas dataframe.
    :param normflag: Flag for normalizing array.
    :param normtype: Type of normalization technique.
    :return: Numpy tuple, containing images, with shape (samples, width, height, channels).
    """
    import numpy as np
    import scripts.pptools as pp

    images = []

    for idx, row in df.iterrows():
        band_1 = np.array(row['band_1']).reshape(75, 75)
        band_2 = np.array(row['band_2']).reshape(75, 75)

        if normflag:
            band_1, b1_max = pp.maxnorm(band_1)
            band_2, b2_max = pp.maxnorm(band_2)

        images.append(np.dstack((band_1, band_2)))

    return np.array(images)


def getbands(df):
    """
    Get bands from Pandas dataframe.
    :param df: Pandas dataframe.
    :return: Numpy tuple, containing images, with shape (samples, width, height, channels).
    """
    import numpy as np
    import scripts.pptools as pp

    band1 = []
    band2 = []
    for idx, row in df.iterrows():
        band_1 = np.array(row['band_1']).reshape(75, 75)
        band_2 = np.array(row['band_2']).reshape(75, 75)

        band_1, b1_max = pp.maxnorm(band_1)
        band_2, b2_max = pp.maxnorm(band_2)

        band1.append(band_1)
        band2.append(band_2)

    return np.array(band1), np.array(band2)


def getangles(df, normflag=False):
    """
    Get angles from Pandas dataframe.
    :param df: Pandas dataframe.
    :return: Numpy tuple, containing angles, with shape (len(df['inc_angle'], 1).
    """
    import numpy as np
    import scripts.pptools as pp

    angles = []

    for idx, row in df.iterrows():
        angle = np.array(row['inc_angle'])

        angles.append(angle)

    angles = np.array(angles)

    if normflag:
        angles, angles_max = pp.maxnorm(angles)

    return angles


def writesubmissioncsv(filename, predicts):
    """
    Write submission csv file.
    :param filename: Name of csv file.
    :param predicts: Numpy array with id in first column and is_iceberg probability in second column.
    """
    with open(filename, 'w') as fp:
        fp.write('id,is_iceberg\n')
        for i in range(len(predicts)):
            fp.write('{0:},{1:.10f}\n'.format(predicts[i, 0], predicts[i, 1]))

